export CUDA_VISIBLE_DEVICES=6
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export HF_HOME=/root/shared-nvme/.cache/huggingface
# export HF_DATASETS_CACHE=/root/shared-nvme/.cache/huggingface/datasets
# export VLLM_LOGGING_LEVEL=DEBUG
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
python -m lcb_runner.runner.main --model Qwen/Qwen3.5-0.8B \
     --scenario codegeneration \
     --vllm_max_gpu_memory 0.9 \
     --evaluate --n 20 --max_tokens 16384

# 侧载LoRA权重示例（与 swift eval --adapters 等效）：
# python -m lcb_runner.runner.main --model Qwen/Qwen3.5-0.8B \
#      --scenario codegeneration \
#      --vllm_max_gpu_memory 0.9 \
#      --evaluate --n 1 --start_date 2023-01-01 \
#      --lora_path output/v52-20260404-111420/checkpoint-1 \
#      --vllm_max_lora_rank 32
