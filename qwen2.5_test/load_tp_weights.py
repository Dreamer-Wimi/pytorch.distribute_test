# load_tp_weights.py 
import os
import torch

def load_and_shard_weights(model_path, tp_model, world_size, rank):
    state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
    new_state_dict = {}

    for name, param in state_dict.items():
        # MLP 权重
        if "mlp.gate_proj.weight" in name:
            total_size = param.shape[0]
            chunk_size = total_size // world_size
            shard = param[rank * chunk_size : (rank + 1) * chunk_size]
            new_state_dict[name.replace("gate_proj", "w1")] = shard
        elif "mlp.up_proj.weight" in name:
            total_size = param.shape[0]
            chunk_size = total_size // world_size
            shard = param[rank * chunk_size : (rank + 1) * chunk_size]
            new_state_dict[name.replace("up_proj", "w3")] = shard
        elif "mlp.down_proj.weight" in name:
            total_size = param.shape[1]
            chunk_size = total_size // world_size
            shard = param[:, rank * chunk_size : (rank + 1) * chunk_size]
            new_state_dict[name.replace("down_proj", "w2")] = shard
        
        # Attention 权重
        elif "self_attn.q_proj.weight" in name:
            total_size = param.shape[0]
            chunk_size = total_size // world_size
            shard = param[rank * chunk_size : (rank + 1) * chunk_size]
            new_state_dict[name] = shard
        elif "self_attn.k_proj.weight" in name:
            total_size = param.shape[0]
            chunk_size = total_size // world_size
            shard = param[rank * chunk_size : (rank + 1) * chunk_size]
            new_state_dict[name] = shard
        elif "self_attn.v_proj.weight" in name:
            total_size = param.shape[0]
            chunk_size = total_size // world_size
            shard = param[rank * chunk_size : (rank + 1) * chunk_size]
            new_state_dict[name] = shard
        elif "self_attn.o_proj.weight" in name:
            total_size = param.shape[1]
            chunk_size = total_size // world_size
            shard = param[:, rank * chunk_size : (rank + 1) * chunk_size]
            new_state_dict[name] = shard
        
        # 其他权重
        elif "lm_head.weight" in name:
            total_size = param.shape[1]
            chunk_size = total_size // world_size
            shard = param[:, rank * chunk_size : (rank + 1) * chunk_size]
            new_state_dict[name] = shard
        elif "embed_tokens.weight" in name or "norm.weight" in name:
            new_state_dict[name] = param

    tp_model.load_state_dict(new_state_dict, strict=False)
    return tp_model
