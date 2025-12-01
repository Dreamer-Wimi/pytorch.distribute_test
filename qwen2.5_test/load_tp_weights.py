# load_tp_weights.py
import os
import torch

def load_and_shard_weights(model_path, tp_model, world_size, rank):
    # 加载原始 HuggingFace 权重 (CPU)
    state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
    new_state_dict = {}
    
    print(f"[Rank {rank}] Loading weights, world_size: {world_size}")
    
    # === 1. 自动探测模型期望的切分形状 ===
    # 我们直接查看 tp_model (已经初始化好的模型) 第一层的权重形状
    # 这样不需要依赖 config 里的数字，直接以模型定义的结构为准
    
    model_state_dict = tp_model.state_dict()
    
    # 探测 Attention 相关的切分大小
    q_proj_key = "model.layers.0.self_attn.q_proj.weight"
    if q_proj_key in model_state_dict:
        q_proj_per_rank = model_state_dict[q_proj_key].shape[0]
        # k_proj 和 v_proj 通常形状一致
        kv_proj_per_rank = model_state_dict["model.layers.0.self_attn.k_proj.weight"].shape[0]
    else:
        raise ValueError(f"Cannot find key {q_proj_key} in tp_model state_dict")

    # 探测 MLP 相关的切分大小
    # 注意：在 tp_model.py 中，路径是 model.layers.0.mlp.w1.weight
    gate_proj_key = "model.layers.0.mlp.w1.weight"
    if gate_proj_key in model_state_dict:
        intermediate_per_rank = model_state_dict[gate_proj_key].shape[0]
    else:
        raise ValueError(f"Cannot find key {gate_proj_key} in tp_model state_dict")

    print(f"[Rank {rank}] Detected TP shapes from initialized model:")
    print(f"  q_proj_per_rank: {q_proj_per_rank}")
    print(f"  kv_proj_per_rank: {kv_proj_per_rank}")
    print(f"  intermediate_per_rank: {intermediate_per_rank}")

    # === 2. 遍历权重并进行切分与重命名 ===
    for name, param in state_dict.items():
        # --- Attention Weights (Column Parallel) ---
        if "self_attn.q_proj.weight" in name:
            # param shape: [hidden_size, hidden_size] (HuggingFace stored usually as [out, in])
            # Check shape first
            if param.shape[0] == tp_model.config.hidden_size: # Standard HF
                shard = param[rank * q_proj_per_rank : (rank + 1) * q_proj_per_rank]
            else: # If HF weights are already specialized (rare)
                shard = param
            new_state_dict[name] = shard
            
        elif "self_attn.k_proj.weight" in name:
            shard = param[rank * kv_proj_per_rank : (rank + 1) * kv_proj_per_rank]
            new_state_dict[name] = shard
            
        elif "self_attn.v_proj.weight" in name:
            shard = param[rank * kv_proj_per_rank : (rank + 1) * kv_proj_per_rank]
            new_state_dict[name] = shard
            
        # --- Attention Output (Row Parallel) ---
        elif "self_attn.o_proj.weight" in name:
            # Row Parallel: 切分输入维度 (dim 1)
            shard = param[:, rank * q_proj_per_rank : (rank + 1) * q_proj_per_rank]
            new_state_dict[name] = shard
        
        # --- Bias Handling (Attention) ---
        elif "self_attn.q_proj.bias" in name:
            shard = param[rank * q_proj_per_rank : (rank + 1) * q_proj_per_rank]
            new_state_dict[name] = shard
        elif "self_attn.k_proj.bias" in name:
            shard = param[rank * kv_proj_per_rank : (rank + 1) * kv_proj_per_rank]
            new_state_dict[name] = shard
        elif "self_attn.v_proj.bias" in name:
            shard = param[rank * kv_proj_per_rank : (rank + 1) * kv_proj_per_rank]
            new_state_dict[name] = shard

        # --- MLP Weights (SwiGLU) ---
        # HF Name: mlp.gate_proj -> TP Model Name: mlp.w1 (Column Parallel)
        elif "mlp.gate_proj.weight" in name:
            shard = param[rank * intermediate_per_rank : (rank + 1) * intermediate_per_rank]
            new_name = name.replace("mlp.gate_proj", "mlp.w1") # 修正：去掉多余的 .mlp
            new_state_dict[new_name] = shard
            
        # HF Name: mlp.up_proj -> TP Model Name: mlp.w3 (Column Parallel)
        elif "mlp.up_proj.weight" in name:
            shard = param[rank * intermediate_per_rank : (rank + 1) * intermediate_per_rank]
            new_name = name.replace("mlp.up_proj", "mlp.w3") # 修正：去掉多余的 .mlp
            new_state_dict[new_name] = shard
            
        # HF Name: mlp.down_proj -> TP Model Name: mlp.w2 (Row Parallel)
        elif "mlp.down_proj.weight" in name:
            # Row Parallel: 切分输入维度 (dim 1)
            shard = param[:, rank * intermediate_per_rank : (rank + 1) * intermediate_per_rank]
            new_name = name.replace("mlp.down_proj", "mlp.w2") # 修正：去掉多余的 .mlp
            new_state_dict[new_name] = shard

        # --- LM Head (Row Parallel) ---
        elif "lm_head.weight" in name:
            # Row Parallel: 切分输入维度 (dim 1)
            # lm_head output is vocab_size, input is hidden_size
            hidden_per_rank = tp_model.config.hidden_size // world_size
            shard = param[:, rank * hidden_per_rank : (rank + 1) * hidden_per_rank]
            new_state_dict[name] = shard
            
        # --- Embeddings & Norms (Replicate / No Sharding) ---
        elif "embed_tokens.weight" in name:
            new_state_dict[name] = param
        elif any(x in name for x in ["norm.weight", "layernorm.weight", "layernorm.bias"]):
            new_state_dict[name] = param
        else:
            # 捕获其他未处理的参数
            new_state_dict[name] = param

    print(f"[Rank {rank}] Finished processing {len(new_state_dict)} weights")

    # === 3. 加载到模型 ===
    # strict=False 因为 HF 的 rotary_emb.inv_freq 我们不需要加载(会在 init 时重新计算)
    missing_keys, unexpected_keys = tp_model.load_state_dict(new_state_dict, strict=False)
    
    # 过滤掉 inv_freq 相关的 missing key 警告，这是正常的
    missing_keys = [k for k in missing_keys if "inv_freq" not in k]
    
    if missing_keys:
        print(f"[Rank {rank}] Warning: Missing keys: {missing_keys[:5]} ...")
    if unexpected_keys:
        print(f"[Rank {rank}] Warning: Unexpected keys: {unexpected_keys[:5]} ...")
    
    return tp_model