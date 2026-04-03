from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3.5-0.8B"

# 让 transformers 自动匹配正确的类
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="cuda:3",
    torch_dtype="auto"
)