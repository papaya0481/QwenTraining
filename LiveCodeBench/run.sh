export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python -m lcb_runner.runner.main --model Qwen/Qwen3.5-0.8B \
     --scenario codegeneration \
     --vllm_max_gpu_memory 0.9 \
     --evaluate --n 1 --start_date 2023-01-01

# 侧载LoRA权重示例（与 swift eval --adapters 等效）：
# python -m lcb_runner.runner.main --model Qwen/Qwen3.5-0.8B \
#      --scenario codegeneration \
#      --vllm_max_gpu_memory 0.9 \
#      --evaluate --n 1 --start_date 2023-01-01 \
#      --lora_path output/v52-20260404-111420/checkpoint-1 \
#      --vllm_max_lora_rank 32
