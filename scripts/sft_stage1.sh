# 显存占用：22GB
export CUDA_VISIBLE_DEVICES=3
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export HF_HOME=/root/shared-nvme/.cache/huggingface
# export HF_DATASETS_CACHE=/root/shared-nvme/.cache/huggingface/datasets
# export MODELSCOPE_CACHE=/root/shared-nvme/.cache/modelscope

# 参数解释
# --external_plugins 使用外部插件进行数据预处理
# --use_hf 使用 Hugging Face 的 模型和数据集下载
# --eval_on_start 在训练开始前进行一次评估
swift sft \
    --model Qwen/Qwen3.5-0.8B \
    --tuner_type lora \
    --external_plugins scripts/data_preprocess.py \
    --dataset codegen1_train:sft \
    --val_dataset codegen1_sft_val \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 1 \
    --save_steps 1 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --truncation_strategy right \
    --torch_empty_cache_steps 1 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --use_hf \
    --eval_use_evalscope \
    --eval_dataset "live_code_bench" \
    --eval_dataset_args '{"live_code_bench": {"trust_remote_code": true, "extra_params": {"start_date": "2023-01-01", "end_date": "2025-12-31"}}}' \
    --eval_limit 100
