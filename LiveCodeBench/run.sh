export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/data/wuli_error/miniconda3/envs/llmqw/lib:$LD_LIBRARY_PATH

python -m lcb_runner.runner.main --model Qwen/Qwen3.5-0.8B \
     --scenario codegeneration \
     --n 3 \
     --vllm_max_gpu_memory 0.4 \
     --evaluate --release_version release_v1