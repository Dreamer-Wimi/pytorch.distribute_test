# main_inference.py
import os
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer
from tp_model import TPQwenForCausalLM
from load_tp_weights import load_and_shard_weights

def setup_distributed():
    
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth5'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth5'

    
    os.environ['GLOO_HOSTNAME'] = '192.168.39.176'
    master_addr = '192.168.39.176'
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # 显式指定 init_method 为 TCP URL
    master_addr = os.environ.get('MASTER_ADDR', '192.168.39.176')
    master_port = os.environ.get('MASTER_PORT', '29520')
    init_method = f'tcp://{os.getenv("MASTER_ADDR")}:{os.getenv("MASTER_PORT")}'
    
    dist.init_process_group(
        backend='gloo',
        init_method=init_method,  # ← 关键：显式指定
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    rank, world_size, local_rank = setup_distributed()
    device = f'cuda:{local_rank}'
    
    model_path = "./qwen2.5-0.5b"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    model = TPQwenForCausalLM(config, world_size=world_size, rank=rank).to(device)
    model = load_and_shard_weights(model_path, model, world_size, rank)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # === 广播 input_ids ===
    if rank == 0:
        prompt = "你好，请介绍一下你自己。"
        inputs = tokenizer(prompt, return_tensors="pt")  # on CPU
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
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            # 确保所有 GPU 操作完成
            torch.cuda.synchronize()
        prof.step()  # 触发保存 tracing 文件
    # === 🔥 Profiler 结束 ===

    if rank == 0:
        print(f"[Rank 0] Generated token: {tokenizer.decode(next_token.item())}")
        print(f"✅ Tracing file saved to {trace_dir}/")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
