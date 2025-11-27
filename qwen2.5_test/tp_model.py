# tp_model.py
import torch
import torch.nn as nn
from tp_linear import ColumnParallelLinear, RowParallelLinear
from tp_attention import TPQwenAttention

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
        self.self_attn = TPQwenAttention(config, world_size, rank)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = TPQwenMLP(config, world_size, rank)

    def forward(
        self,
        hidden_states,
        past_key_value=None,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs, _, present_kv = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_outputs

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if use_cache:
            return (hidden_states, present_kv)
        else:
            return (hidden_states,)

class TPQwenModel(nn.Module):
    def __init__(self, config, world_size, rank):
        super().__init__()
        # Embedding 层（未切分）
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 所有 Transformer 层
        self.layers = nn.ModuleList([
            TPQwenLayer(config, world_size, rank) for _ in range(config.num_hidden_layers)
        ])
        # 最后一层归一化
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # --- 生成 position_ids ---
        if position_ids is None and input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
            device = input_ids.device
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        # ---------------------------

        hidden_states = inputs_embeds
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)  # present_kv

        # 最终 LayerNorm
        hidden_states = self.norm(hidden_states)

        return (hidden_states, next_decoder_cache) if use_cache else (hidden_states,)


class TPQwenForCausalLM(nn.Module):
    def __init__(self, config, world_size, rank):
        super().__init__()
        self.model = TPQwenModel(config, world_size, rank)
        self.lm_head = RowParallelLinear(
            config.hidden_size,
            config.vocab_size,
            world_size,
            rank,
            bias=False
        )

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        return (logits,) + outputs[1:] if len(outputs) > 1 else (logits,)
