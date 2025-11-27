# tp_model.py
import torch
import torch.nn as nn
from tp_linear import ColumnParallelLinear, RowParallelLinear

class TPSwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size, world_size, rank):
        super().__init__()
        self.w1 = ColumnParallelLinear(hidden_size, intermediate_size, world_size, rank, bias=False)
        self.w2 = RowParallelLinear(intermediate_size, hidden_size, world_size, rank, bias=False)
        self.w3 = ColumnParallelLinear(hidden_size, intermediate_size, world_size, rank, bias=False)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w3(x)
        hidden = torch.nn.functional.silu(x1) * x2
        return self.w2(hidden)

class TPQwenMLP(nn.Module):
    def __init__(self, config, world_size, rank):
        super().__init__()
        self.mlp = TPSwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            world_size=world_size,
            rank=rank
        )

    def forward(self, x):
        return self.mlp(x)

class TPQwenLayer(nn.Module):
    def __init__(self, config, world_size, rank):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TPQwenMLP(config, world_size, rank)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

class TPQwenModel(nn.Module):
    def __init__(self, config, world_size, rank):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TPQwenLayer(config, world_size, rank) for _ in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)

class TPQwenForCausalLM(nn.Module):
    def __init__(self, config, world_size, rank):
        super().__init__()
        self.model = TPQwenModel(config, world_size, rank)
        self.lm_head = RowParallelLinear(config.hidden_size, config.vocab_size, world_size, rank, bias=False)

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)
