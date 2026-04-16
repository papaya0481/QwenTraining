#!/usr/bin/env bash
set -euo pipefail

# Full DAPO runner (single-GPU friendly) for Qwen3.5-0.8B + LoRA adapter.
# This script uses recipe/dapo/main_dapo.py so that dynamic sampling + group
# filtering is actually enabled (main_ppo does not include that trainer logic).
#
# DAPO 4 improvements over vanilla GRPO enabled in this script:
# 1) Decoupled clip (clip-higher): clip_ratio_low / clip_ratio_high
# 2) Dynamic sampling with group filtering: algorithm.filter_groups.* + data.gen_batch_size
# 3) Token-level policy gradient loss: actor.loss_agg_mode=token-mean
# 4) Overlong reward shaping: reward.reward_kwargs.overlong_buffer_cfg.*
#
# Example:
#   SFT_ADAPTER_PATH=/root/QwenTraining/output/sft_codegen1_12g_smoke/.../checkpoint-32 \
#   DATA_DIR=/root/shared-nvme/output/taco \
#   OUTPUT_DIR=/root/shared-nvme/output/verl_dapo_taco_verified_full \
#   bash scripts/verl_dapo_taco_verified_full.sh

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export HF_HOME=${HF_HOME:-/root/shared-nvme/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/root/shared-nvme/.cache/huggingface/datasets}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python || command -v python3)}

# vLLM's CuMemAllocator is incompatible with expandable_segments.
if [[ "${PYTORCH_CUDA_ALLOC_CONF}" == *"expandable_segments:True"* ]]; then
  echo "[WARN] Removing incompatible setting from PYTORCH_CUDA_ALLOC_CONF for vLLM: ${PYTORCH_CUDA_ALLOC_CONF}"
  unset PYTORCH_CUDA_ALLOC_CONF
fi

if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
VERL_DIR="${PROJECT_DIR}/verl"

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3.5-0.8B}
DATA_DIR=${DATA_DIR:-/root/shared-nvme/output/taco}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/val.parquet}

OUTPUT_DIR=${OUTPUT_DIR:-/root/shared-nvme/output/verl_dapo_taco_verified_full}
PROJECT_NAME=${PROJECT_NAME:-qwen_dapo_taco_verified}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_08b_dapo_taco_full}
LOGGER_BACKENDS=${LOGGER_BACKENDS:-["console"]}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard/${EXPERIMENT_NAME}}

# Single-GPU conservative defaults.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-5000}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2000}
# ROLLOUT_N=${ROLLOUT_N:-2}
# TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}

# DAPO knobs.
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
# FILTER_METRIC=${FILTER_METRIC:-acc}
# FILTER_MAX_NUM_GEN_BATCHES=${FILTER_MAX_NUM_GEN_BATCHES:-8}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-256}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}

patch_vllm_qwen35_rope_bug() {
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

try:
  import vllm.transformers_utils.configs.qwen3_5 as qwen35_cfg
except Exception:
  print("[WARN] vLLM not importable when applying qwen3.5 rope hotfix; skip patch.")
  raise SystemExit(0)

cfg_path = Path(qwen35_cfg.__file__)
text = cfg_path.read_text(encoding="utf-8")
old = 'kwargs["ignore_keys_at_rope_validation"] = [\n            "mrope_section",\n            "mrope_interleaved",\n        ]'
new = 'kwargs["ignore_keys_at_rope_validation"] = {\n            "mrope_section",\n            "mrope_interleaved",\n        }'

if new in text:
  print(f"[INFO] vLLM qwen3.5 rope hotfix already applied: {cfg_path}")
  raise SystemExit(0)

if old not in text:
  print(f"[WARN] Hotfix pattern not found in {cfg_path}; skip patch.")
  raise SystemExit(0)

cfg_path.write_text(text.replace(old, new, 1), encoding="utf-8")
print(f"[INFO] Applied vLLM qwen3.5 rope hotfix: {cfg_path}")
PY
}

resolve_model_path() {
  local model_ref="$1"
  local cache_root="$2"
  local model_dir snap

  if [[ -d "${model_ref}" ]]; then
    echo "${model_ref}"
    return 0
  fi

  model_dir="${cache_root}/hub/models--${model_ref//\//--}/snapshots"
  if [[ -d "${model_dir}" ]]; then
    snap=$(find "${model_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
      | sort -n \
      | tail -n 1 \
      | cut -d' ' -f2- || true)
    if [[ -n "${snap}" && -f "${snap}/tokenizer_config.json" ]]; then
      echo "${snap}"
      return 0
    fi
  fi

  return 1
}

MODEL_PATH="${MODEL_NAME}"
if resolved=$(resolve_model_path "${MODEL_NAME}" "${HF_HOME}"); then
  MODEL_PATH="${resolved}"
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  echo "[INFO] Using local cached model snapshot: ${MODEL_PATH}"
else
  echo "[WARN] Local snapshot not found for ${MODEL_NAME}, will try online loading."
fi

SFT_ADAPTER_PATH=${SFT_ADAPTER_PATH:-}
if [[ -z "${SFT_ADAPTER_PATH}" ]]; then
  echo "[ERROR] SFT_ADAPTER_PATH is required."
  exit 1
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[ERROR] train parquet not found: ${TRAIN_FILE}"
  exit 1
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  echo "[ERROR] val parquet not found: ${VAL_FILE}"
  exit 1
fi

# mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_DIR}"
# export TENSORBOARD_DIR

patch_vllm_qwen35_rope_bug

# Custom reward module imports `data.code_excutor` from project root.
# Add project root to PYTHONPATH because we launch from ${VERL_DIR}.
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

pushd "${VERL_DIR}" >/dev/null
"${PYTHON_BIN}" -m recipe.dapo.main_dapo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  algorithm.filter_groups.enable=True \
  algorithm.filter_groups.metric="acc" \
  algorithm.filter_groups.max_num_gen_batches=8 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.gen_batch_size=4 \
  data.train_batch_size=2 \
  data.dataloader_num_workers=0 \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.shuffle=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  +actor_rollout_ref.ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.lora_adapter_path="${SFT_ADAPTER_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  trainer.critic_warmup=0 \
  reward.reward_manager.name=dapo \
  reward.num_workers=1 \
  reward.reward_kwargs.overlong_buffer_cfg.enable=True \
  reward.reward_kwargs.overlong_buffer_cfg.len=1024 \
  reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
  reward.reward_kwargs.overlong_buffer_cfg.log=False \
  reward.reward_kwargs.max_resp_len=2000 \
  reward.custom_reward_function.path="${PROJECT_DIR}/scripts/plugins/verl_codegen1_reward.py" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.timeout_sec=4 \
  +reward.custom_reward_function.reward_kwargs.memory_limit_mb=1024 \
  trainer.logger="${LOGGER_BACKENDS}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.save_freq=1 \
  trainer.test_freq=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=20
popd >/dev/null
