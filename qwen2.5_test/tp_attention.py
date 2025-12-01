import torch
import torch.nn as nn
from tp_linear import ColumnParallelLinear, RowParallelLinear
import torch.distributed as dist

class TPQwenAttention(nn.Module):
    def __init__(self, config, world_size, rank):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.world_size = world_size
        self.rank = rank
        
        # 计算每个rank的实际头数 (仅用于逻辑，不用于初始化Linear的总维度)
        self.num_heads_per_rank = self.num_heads // self.world_size
        self.num_key_value_heads_per_rank = self.num_key_value_heads // self.world_size
        
        # 计算总投影尺寸 (这是传入ColumnParallelLinear所需的)
        self.q_proj_total_size = self.num_heads * self.head_dim  # 14 * 64 = 896
        self.kv_proj_total_size = self.num_key_value_heads * self.head_dim  # 2 * 64 = 128
        
        self.q_proj_per_rank = self.num_heads_per_rank * self.head_dim
        self.kv_proj_per_rank = self.num_key_value_heads_per_rank * self.head_dim
        
        if rank == 0:
            print(f"[Rank {rank}] Attention config:")
            print(f"  num_heads: {self.num_heads}, num_key_value_heads: {self.num_key_value_heads}")
            print(f"  q_proj_total_size (passing to Linear): {self.q_proj_total_size}")
        
        # 修正：传入 total_size，让 ColumnParallelLinear 内部去处理除法
        self.q_proj = ColumnParallelLinear(
            self.hidden_size, 
            self.q_proj_total_size,  # <--- 修改这里：传入总维度 896
            world_size, rank, bias=True
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size, 
            self.kv_proj_total_size, # <--- 修改这里：传入总维度 128
            world_size, rank, bias=True
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size, 
            self.kv_proj_total_size, # <--- 修改这里：传入总维度 128
            world_size, rank, bias=True
        )
        self.o_proj = RowParallelLinear(
            self.q_proj_total_size,
            self.hidden_size, world_size, rank, bias=False
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        bsz, q_len, _ = hidden_states.size()
        
        num_heads_per_rank = self.num_heads // self.world_size
        num_key_value_heads_per_rank = self.num_key_value_heads // self.world_size
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, num_heads_per_rank, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, num_key_value_heads_per_rank, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_key_value_heads_per_rank, self.head_dim).transpose(1, 2)
        
        if num_key_value_heads_per_rank < num_heads_per_rank:
            repeat_times = num_heads_per_rank // num_key_value_heads_per_rank
            key_states = key_states.repeat_interleave(repeat_times, dim=1)
            value_states = value_states.repeat_interleave(repeat_times, dim=1)
        
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, num_heads_per_rank * self.head_dim)

        attn_output = self.o_proj(attn_output)

        if use_cache:
            present_kv = (key_states, value_states)
            return attn_output, attn_weights, present_kv
        else:
            return attn_output, attn_weights, None