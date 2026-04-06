export CUDA_VISIBLE_DEVICES=0

# 参数说明
# --vllm_max_lora_rank: 设置vLLM的最大LoRA秩，与训练时的LoRA秩一致，确保评估时使用相同的参数。
swift eval \
    --model Qwen/Qwen3.5-0.8B \
    --adapters output/v52-20260404-111420/checkpoint-1 \
    --eval_backend Native \
    --infer_backend vllm \
    --eval_limit 50 \
    --eval_dataset live_code_bench \
    --eval_dataset_args '{"live_code_bench": {"trust_remote_code": true, "extra_params": {"start_date": "2023-01-01", "end_date": "2025-12-31"}}}' \
    --eval_generation_config '{"max_tokens": 4096, "temperature": 0.2}' \
    --vllm_max_lora_rank 32 \
    --use_hf