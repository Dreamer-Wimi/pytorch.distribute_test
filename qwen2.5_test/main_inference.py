import os
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer
from tp_model import TPQwenForCausalLM
from load_tp_weights import load_and_shard_weights

def setup_distributed():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl', init_method='env://') # 推荐使用 nccl 做 GPU 通信
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    rank, world_size, local_rank = setup_distributed()
    device = f'cuda:{local_rank}'
    # 请修改为你的实际路径
    model_path = "./qwen2.5-0.5b" 
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 初始化模型
    with torch.device("meta"): # 使用 meta device 加速初始化，避免显存占用
        pass 
        # 这里为了演示简单，还是直接实例化到 CPU 或 CUDA
        # 如果模型很大，建议先在 meta 上初始化结构，再 load_state_dict

    model = TPQwenForCausalLM(config, world_size=world_size, rank=rank).to(device)

    # 加载权重
    model = load_and_shard_weights(model_path, model, world_size, rank)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # === 数据准备 ===
    if rank == 0:
        prompt = "你好，请介绍一下你自己。"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
    else:
        input_ids = None

    # 广播 input_ids
    # 1. 广播长度
    if rank == 0:
        length_tensor = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=device)
    else:
        length_tensor = torch.tensor([0], dtype=torch.long, device=device)
    dist.broadcast(length_tensor, src=0)

    # 2. 广播内容
    seq_len = length_tensor.item()
    if rank != 0:
        input_ids = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    dist.broadcast(input_ids, src=0)

    # === 生成循环 ===
    max_new_tokens = 2
    generated_ids = []

    # Causal Mask (Prefill 阶段需要)
    # 这里的 Mask 需要处理成加法 Mask (0 for keep, -inf for mask)
    # Qwen2 的实现通常由 attention 内部处理，但手动传入更稳妥
    # 简单起见，我们这里传 None，让 torch.matmul 的 causality 依赖实现
    # 在标准的 SDPA 或手动 Attention 中，需要通过 mask 屏蔽上三角

    # 简单构造一个 causal mask (仅用于 prefill)
    attention_mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device)
    attention_mask = torch.triu(attention_mask, diagonal=1)

    # === 🔥 添加 Profiler：从这里开始 ===
    trace_dir = f"./trace_rank{rank}"
    os.makedirs(trace_dir, exist_ok=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        profile_memory=False,  # 可设为 True，但会增大文件
        with_stack=False,      # 设为 True 可看调用栈（增大文件）
        with_flops=False,
    ) as prof:
        with torch.no_grad():
            # --- 1. Prefill ---
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask, # 传入 Mask
                use_cache=True
            )
            logits = outputs[0]
            past_key_values = outputs[1]
            
            # 贪婪采样
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated_ids.append(next_token.item())
            
            # --- 2. Decoding ---
            input_ids = next_token.unsqueeze(0) # [1, 1]
            
            for _ in range(max_new_tokens - 1):
                # Decoding 阶段不需要 mask (因为只看 past_kv 和当前 token)
                outputs = model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = outputs[0]
                past_key_values = outputs[1]
                
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                
                # 必须从 Rank 0 广播结果，保证所有卡上的输入一致
                # 虽然理论上 TP 应该算出来一样，但浮点误差可能导致漂移
                dist.broadcast(next_token, src=0)
                
                generated_ids.append(next_token.item())
                input_ids = next_token.unsqueeze(0)
            
            # 确保所有 GPU 操作完成
            torch.cuda.synchronize()
        prof.step()  # 触发保存 tracing 文件
    # === 🔥 Profiler 结束 ===

    if rank == 0:
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Input: {prompt}")
        print(f"Output: {output_text}")
        print(f"✅ Tracing file saved to {trace_dir}/")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()