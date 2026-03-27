export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH=/data/wuli_error/miniconda3/envs/llmqw/lib:$LD_LIBRARY_PATH

python -m lcb_runner.runner.main --model Qwen/Qwen3.5-9B \
     --scenario codegeneration \
     --vllm_max_gpu_memory 0.9 \
     --evaluate