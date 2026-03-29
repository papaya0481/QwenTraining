# export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH=/data2/ruixin/miniconda3/envs/llmqw/lib:$LD_LIBRARY_PATH

python -m lcb_runner.runner.main --model Qwen/Qwen3-0.6B \
     --scenario codegeneration \
     --vllm_max_gpu_memory 0.9 \
     --evaluate