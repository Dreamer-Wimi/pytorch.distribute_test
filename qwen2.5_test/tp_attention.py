# tp_attention.py
import torch
import torch.nn as nn
from tp_linear import ColumnParallelLinear, RowParallelLinear

class TPQwenAttention(nn.Module):
    def __init__(self, config, world_size, rank):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.q_proj = ColumnParallelLinear(self.hidden_size, self.num_heads * self.head_dim, world_size, rank, bias=True)
        self.k_proj = ColumnParallelLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, world_size, rank, bias=True)
        self.v_proj = ColumnParallelLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, world_size, rank, bias=True)
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim, self.hidden_size, world_size, rank, bias=False)

    def forward(
        self,
        hidden_states,
        past_key_value=None,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
    ):
        bsz, q_len, _ = hidden_states.size()
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None

        return attn_output, None, present_key_value
