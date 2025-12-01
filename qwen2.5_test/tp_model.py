# tp_model.py
import torch
import torch.nn as nn
from tp_linear import ColumnParallelLinear, RowParallelLinear
import math

# --- 组件 1: RMSNorm ---
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# --- 组件 2: RoPE ---
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # q, k: [batch, seq, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim] -> [batch, seq, 1, head_dim]
    
    # 简单的选取逻辑: 假设 position_ids 是 [batch, seq]
    # 我们需要根据 position_ids 从 cos/sin 缓存中拿出对应的 embedding
    cos = cos[position_ids].unsqueeze(2) # [batch, seq, 1, head_dim]
    sin = sin[position_ids].unsqueeze(2)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to("cpu") / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype=torch.float32), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype=torch.float32), persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len, ...].to(dtype=x.dtype), self.sin_cached[:seq_len, ...].to(dtype=x.dtype)

# --- Attention ---
class TPQwenAttention(nn.Module):
    def __init__(self, config, world_size, rank):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.world_size = world_size
        self.rank = rank
        
        self.num_heads_per_rank = self.num_heads // self.world_size
        self.num_key_value_heads_per_rank = self.num_key_value_heads // self.world_size
        
        self.q_proj_total_size = self.num_heads * self.head_dim
        self.kv_proj_total_size = self.num_key_value_heads * self.head_dim
        
        self.q_proj = ColumnParallelLinear(self.hidden_size, self.q_proj_total_size, world_size, rank, bias=True)
        self.k_proj = ColumnParallelLinear(self.hidden_size, self.kv_proj_total_size, world_size, rank, bias=True)
        self.v_proj = ColumnParallelLinear(self.hidden_size, self.kv_proj_total_size, world_size, rank, bias=True)
        self.o_proj = RowParallelLinear(self.q_proj_total_size, self.hidden_size, world_size, rank, bias=False)
        
        self.rotary_emb = Qwen2RotaryEmbedding(self.head_dim, max_position_embeddings=config.max_position_embeddings)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads_per_rank, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads_per_rank, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads_per_rank, self.head_dim)
        
        # RoPE
        kv_seq_len = q_len + (past_key_value[0].shape[1] if past_key_value is not None else 0)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        
        present_kv = (key_states, value_states) if use_cache else None

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.num_key_value_heads_per_rank < self.num_heads_per_rank:
            key_states = torch.repeat_interleave(key_states, dim=1, repeats=self.num_heads_per_rank // self.num_key_value_heads_per_rank)
            value_states = torch.repeat_interleave(value_states, dim=1, repeats=self.num_heads_per_rank // self.num_key_value_heads_per_rank)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads_per_rank * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, present_kv

# --- MLP ---
class TPSwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size, world_size, rank):
        super().__init__()
        # w1: Gate, w3: Up, w2: Down
        self.w1 = ColumnParallelLinear(hidden_size, intermediate_size, world_size, rank, bias=False)
        self.w3 = ColumnParallelLinear(hidden_size, intermediate_size, world_size, rank, bias=False)
        self.w2 = RowParallelLinear(intermediate_size, hidden_size, world_size, rank, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

# --- Layers & Model ---
class TPQwenLayer(nn.Module):
    def __init__(self, config, world_size, rank):
        super().__init__()
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = TPQwenAttention(config, world_size, rank)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TPSwiGLU(config.hidden_size, config.intermediate_size, world_size, rank)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs, _, present_kv = self.self_attn(
            hidden_states, attention_mask, position_ids, past_key_value, use_cache
        )
        hidden_states = residual + attn_outputs

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states, present_kv) if use_cache else (hidden_states,)

class TPQwenModel(nn.Module):
    def __init__(self, config, world_size, rank):
        super().__init__()
        self.config = config # 保存 config 供 loader 使用
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TPQwenLayer(config, world_size, rank) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        if position_ids is None:
            seq_length = input_ids.shape[1]
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[1]
                position_ids = position_ids + past_length
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        next_decoder_cache = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = layer(
                hidden_states, attention_mask, position_ids, past_kv, use_cache
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache.append(layer_outputs[1])

        hidden_states = self.norm(hidden_states)
        return (hidden_states, next_decoder_cache) if use_cache else (hidden_states,)

class TPQwenForCausalLM(nn.Module):
    def __init__(self, config, world_size, rank):
        super().__init__()
        self.config = config
        self.model = TPQwenModel(config, world_size, rank)
        self.lm_head = RowParallelLinear(config.hidden_size, config.vocab_size, world_size, rank, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False):
        # 1. 获取完整的 hidden_states [batch, seq, hidden_size]
        outputs = self.model(input_ids, attention_mask, position_ids, past_key_values, use_cache)
        hidden_states = outputs[0]
        
        # 2. 关键修复：手动切分 hidden_states
        # RowParallelLinear 的输入需要是 [batch, seq, hidden_size // world_size]
        # 因为它把权重按列切分了（对应输入的特征维度）
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        world_size = self.lm_head.world_size
        rank = self.lm_head.rank
        
        # 计算当前 rank 对应的特征维度范围
        hidden_size_per_rank = hidden_size // world_size
        start_idx = rank * hidden_size_per_rank
        end_idx = (rank + 1) * hidden_size_per_rank
        
        # 切片
        hidden_states_shard = hidden_states[..., start_idx:end_idx]
        
        # 3. 传入 lm_head
        # output_parallel = hidden_states_shard @ weight_shard
        # 然后 RowParallelLinear 内部会做 All-Reduce Sum
        logits = self.lm_head(hidden_states_shard)
        
        return (logits,) + (outputs[1:] if use_cache else ())