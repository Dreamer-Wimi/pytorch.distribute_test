# download_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B"
save_dir = "./qwen2.5-0.5b"

print("Downloading model...")
# 关键：设置 from_tf=False, from_flax=False，并确保以 PyTorch 加载
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="auto",  # 自动匹配
    # 注意：即使原始是 safetensors，from_pretrained 会转为 torch.Tensor
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print(f"Saving as PyTorch format to {save_dir}...")
# 关键：使用 save_pretrained 会生成 pytorch_model.bin（如果模型不大）
model.save_pretrained(save_dir, safe_serialization=False)  # ← 禁用 safetensors！
tokenizer.save_pretrained(save_dir)

print("✅ Done! Check for pytorch_model.bin")
