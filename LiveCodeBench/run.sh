# export CUDA_VISIBLE_DEVICES=0
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
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
python -m lcb_runner.runner.main --model Qwen/Qwen3.5-9B \
     --scenario codegeneration \
     --vllm_max_gpu_memory 0.9 \
     --evaluate --n 1 --start_date 2024-10-01 \
     --lora_path /root/lg/QwenTraining/outputs/verl/DAPO_AutoDL/DAPO-Qwen3.5-9B_04-23_2050/global_step_48/actor/lora_adapter \
     --vllm_max_lora_rank 64 --max_tokens 8192
