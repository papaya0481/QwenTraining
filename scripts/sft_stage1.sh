# 显存占用：22GB
export CUDA_VISIBLE_DEVICES=3
export CUDA_HOME=$CONDA_PREFIX
# export NPROC_PER_NODE=1
# export VLLM_HOST_IP=127.0.0.1
# export NCCL_SOCKET_IFNAME=lo
# export GLOO_SOCKET_IFNAME=lo
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export HF_HOME=/root/shared-nvme/.cache/huggingface
# export HF_DATASETS_CACHE=/root/shared-nvme/.cache/huggingface/datasets
# export MODELSCOPE_CACHE=/root/shared-nvme/.cache/modelscope

# 参数解释
# --external_plugins 使用外部插件进行数据预处理
# --use_hf 使用 Hugging Face 的 模型和数据集下载
# --eval_on_start 在训练开始前进行一次评估
# --torch_empty_cache_steps 每隔多少步清空一次 CUDA 缓存，减少显存占用. 调大可以增加吞吐量，但可能导致显存不足错误，调小可以减少显存占用，但可能降低吞吐量。根据实际情况调整。
# --freeze-vit 冻结visual层的权重，减少训练时的显存占用和计算量。根据实际情况调整。
swift sft \
    --model Qwen/Qwen3.5-0.8B \
    --tuner_type lora \
    --external_plugins scripts/data_preprocess.py \
    --dataset codegen1_train:sft \
    --val_dataset codegen1_sft_val#20 \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner true \
    --gradient_accumulation_steps 16 \
    --eval_steps 1 \
    --save_steps 1 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --truncation_strategy right \
    --torch_empty_cache_steps 1 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --use_hf \
    --deepspeed "zero2" \
    --eval_use_evalscope \
    --eval_dataset "live_code_bench" \
    --eval_dataset_args '{"live_code_bench": {"trust_remote_code": true, "extra_params": {"start_date": "2023-01-01", "end_date": "2025-12-31"}}}' \
    --extra_eval_args '{"infer_backend": "vllm"}' \
    --eval_generation_config '{"max_tokens": 1024}' \
    --eval_limit 10
