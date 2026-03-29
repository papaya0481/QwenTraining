export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/data/wuli_error/miniconda3/envs/llmqw/lib:$LD_LIBRARY_PATH

# TEST ONLY
python -m lcb_runner.runner.main --model Qwen/Qwen3.5-0.8B \
     --scenario codeexecution \
     --vllm_max_gpu_memory 0.9 \
     --evaluate --n 1 --start_date 2023-01-01

# python -m pdb lcb_runner/runner/main.py --model Qwen/Qwen3.5-0.8B \
#      --scenario codegeneration \
#      --vllm_max_gpu_memory 0.9 \
#      --evaluate --n 1