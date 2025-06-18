import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

model_dir = "/data/public/models/Qwen2.5-Omni-7B" 

print(f"🔍 로딩 경로 확인: {model_dir}")

try:
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
    print("✅ 모델 로딩 성공")
    print(f"📦 model dtype: {model.dtype}")
    print(f"📦 model device map: {model.hf_device_map}")
except Exception as e:
    print("❌ 모델 로딩 실패:", e)
