# main_inference.py
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer
from tp_model import TPQwenForCausalLM  # ← 确保这是完整的模型类
from load_tp_weights import load_and_shard_weights

def setup_distributed():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='gloo', init_method='env://')
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    rank, world_size, local_rank = setup_distributed()
    device = f'cuda:{local_rank}'
    
    model_path = "./qwen2.5-0.5b"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    # ✅ 正确定义 model
    model = TPQwenForCausalLM(config, world_size=world_size, rank=rank).to(device)
    model = load_and_shard_weights(model_path, model, world_size, rank)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # === 广播 input_ids ===
    if rank == 0:
        prompt = "你好，请介绍一下你自己。"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids_cpu = inputs.input_ids
        seq_len = input_ids_cpu.size(1)
    else:
        seq_len = 0

    seq_len_tensor = torch.tensor([seq_len], dtype=torch.long)
    dist.broadcast(seq_len_tensor, src=0)
    actual_seq_len = seq_len_tensor.item()

    if rank == 0:
        to_broadcast = input_ids_cpu
    else:
        to_broadcast = torch.empty((1, actual_seq_len), dtype=torch.long)
    dist.broadcast(to_broadcast, src=0)
    input_ids = to_broadcast.to(device)

    # === Prefill + Multi-token Generation ===
    max_new_tokens = 10
    generated_ids = []

    with torch.no_grad():
        # --- Prefill ---
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            position_ids=None
        )
        logits = outputs[0]
        past_key_values = outputs[1] if len(outputs) > 1 else None
        next_token = torch.argmax(logits[:, -1, :], dim=-1)  # [1]

        for step in range(max_new_tokens):
            # --- Decode Step ---
            outputs = model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
                position_ids=None
            )
            logits = outputs[0]
            past_key_values = outputs[1] if len(outputs) > 1 else None
            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            # 同步 next_token 到所有 rank
            dist.broadcast(next_token, src=0)
            generated_ids.append(next_token.item())

    # Only rank 0 打印
    if rank == 0:
        full_output = tokenizer.decode(input_ids[0].tolist() + generated_ids, skip_special_tokens=True)
        print(f"[Rank 0] Generated: {full_output}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
