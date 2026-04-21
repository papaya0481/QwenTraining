# export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/root/shared-nvme/.cache/huggingface
export HF_DATASETS_CACHE=/root/shared-nvme/.cache/huggingface/datasets
export MODELSCOPE_CACHE=/root/shared-nvme/.cache/modelscope

# 奖励执行器的资源限制，可按机器情况调整。
export CODE_REWARD_TIMEOUT_SEC=3
export CODE_REWARD_MEMORY_MB=512

# 可选：如果你已经有SFT LoRA权重，取消注释并填写路径。
# export SFT_ADAPTER=output/your_sft_checkpoint

# 说明：
# 1) 使用GRPO框架并将loss_type设置为dapo，即为DAPO训练。
# 2) reward_funcs仅使用external_codegen1_pass_rate，即测试用例通过率奖励。
# 3) 本脚本是12G显存的smoke test配置，只用于验证流程可跑通。

swift rlhf \
    --rlhf_type grpo \
    --loss_type dapo \
    --model Qwen/Qwen3.5-0.8B \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --external_plugins scripts/rl_dapo_plugin.py \
    --dataset codegen1_train_rl:rl#128 \
    --load_from_cache_file true \
    --use_hf \
    --reward_funcs external_codegen1_pass_rate \
    --use_vllm false \
    --torch_dtype float16 \
    --max_length 1536 \
    --max_completion_length 768 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-7 \
    --gradient_accumulation_steps 1 \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output/dapo_codegen1_12g_smoke \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 1 \
    --generation_batch_size 2 \
    --num_generations 2 \
    --num_generations_eval 1 \
    --temperature 0.9 \
    --beta 0.001 \
    --num_iterations 1 \
    --log_completions true \
    --top_k 0

# 如果你有SFT LoRA初始化，可在上方命令中补充：
# --adapters ${SFT_ADAPTER} --ref_adapters ${SFT_ADAPTER}
