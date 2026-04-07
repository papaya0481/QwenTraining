# 显存占用：22GB
# export CUDA_VISIBLE_DEVICES=3
# export CUDA_HOME=$CONDA_PREFIX
export NPROC_PER_NODE=2
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_SOCKET_IFNAME=lo
export VLLM_LOGGING_LEVEL=DEBUG
# 训练进程内拉起 vLLM 时，fork 容易继承 CUDA/NCCL 上下文导致卡住，使用 spawn 更稳。
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/root/shared-nvme/.cache/huggingface
export HF_DATASETS_CACHE=/root/shared-nvme/.cache/huggingface/datasets
# export MODELSCOPE_CACHE=/root/shared-nvme/.cache/modelscope
export WANDB_PROJECT=Qwen_thu
export WABDB_API_KEY=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

# 参数解释
# --external_plugins 使用外部插件进行数据预处理
# --use_hf 使用 Hugging Face 的 模型和数据集下载
# --eval_on_start 在训练开始前进行一次评估
# --torch_empty_cache_steps 每隔多少步清空一次 CUDA 缓存，减少显存占用. 调大可以增加吞吐量，但可能导致显存不足错误，调小可以减少显存占用，但可能降低吞吐量。根据实际情况调整。
# --freeze-vit 冻结visual层的权重，减少训练时的显存占用和计算量。根据实际情况调整。
# --extra_eval_args 参考 GRPO 的 vLLM 使用方式，优先限制并发和显存占比，避免训练中评测拉起 vLLM 时 OOM。
swift sft \
    --model Qwen/Qwen3.5-0.8B \
    --tuner_type lora \
    --external_plugins scripts/data_preprocess.py \
    --dataset codegen1_train:sft \
    --val_dataset codegen1_sft_val#20 \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner true \
    --attn_impl flash_attention_2 \
    --gradient_accumulation_steps 16 \
    --eval_steps 2 \
    --save_steps 2 \
    --metric_for_best_model eval_loss \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 5000 \
    --truncation_strategy right \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --use_hf \
    --deepspeed "zero2" \
    --gradient_checkpointing true \
    --report_to wandb \
    # --eval_use_evalscope \
    # --eval_dataset "live_code_bench" \
    # --eval_dataset_args '{"live_code_bench": {"trust_remote_code": true}' \
    # --eval_generation_config '{"max_tokens": 16384}' \
    # --extra_eval_args '{"infer_backend": "vllm", "vllm_tensor_parallel_size": 2}' \
    # --eval_limit 100
