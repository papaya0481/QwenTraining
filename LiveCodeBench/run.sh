export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export HF_HOME=/root/shared-nvme/.cache/huggingface
# export HF_DATASETS_CACHE=/root/shared-nvme/.cache/huggingface/datasets
# export VLLM_LOGGING_LEVEL=DEBUG
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# python -m lcb_runner.runner.main --model Qwen/Qwen3.5-0.8B \
#      --scenario codegeneration \
#      --vllm_max_gpu_memory 0.9 \
#      --evaluate --n 20 --max_tokens 16384

# 侧载LoRA权重示例（与 swift eval --adapters 等效）：
python -m lcb_runner.runner.main --model Qwen/Qwen3.5-0.8B \
     --scenario codegeneration \
     --vllm_max_gpu_memory 0.9 \
     --evaluate --n 1 --start_date 2023-01-01 \
     --lora_path /data/wuli_error/WRX/QwenTraining/outputs/verl/DAPO/DAPO-Qwen3.5-0.8B_04-17_11:55/global_step_3/actor \
     --vllm_max_lora_rank 32
