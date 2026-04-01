# 显存占用：22GB
CUDA_VISIBLE_DEVICES=7

# 参数解释
# --external_plugins 使用外部插件进行数据预处理
# --use_hf 使用 Hugging Face 的 模型和数据集下载
swift sft \
    --model Qwen/Qwen3.5-0.8B \
    --tuner_type lora \
    --external_plugins scripts/data_preprocess.py \
    --dataset BigfufuOuO/codegen1_merged_clean \
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
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 20000 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --use_hf