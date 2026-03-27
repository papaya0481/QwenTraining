export CUDA_VISIBLE_DEVICES=4
export LD_LIBRARY_PATH=/data/wuli_error/miniconda3/envs/llmqw/lib:$LD_LIBRARY_PATH

python -m lcb_runner.runner.main --model Qwen/Qwen3.5-2B \
     --scenario codegeneration\
    --evaluate --release_version release_v2