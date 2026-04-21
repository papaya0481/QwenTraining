#!/usr/bin/env bash
set -euo pipefail

# Single-GPU smoke config for 12GB VRAM (Qwen3.5-0.8B + LoRA adapter).
# Expected usage:
#   SFT_ADAPTER_PATH=output/.../checkpoint-xxx bash scripts/verl_dapo_codegen1_smoke.sh

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

# Optional mirror, e.g. https://hf-mirror.com
if [[ -n "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT
fi

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3.5-0.8B}
PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)

DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/output/verl_codegen1_data}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/val.parquet}

OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/output/verl_dapo_codegen1_smoke}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_08b_codegen1_dapo_smoke}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}
ROLLOUT_N=${ROLLOUT_N:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
SAVE_FREQ=${SAVE_FREQ:-1000}
TEST_FREQ=${TEST_FREQ:-1000}
RL_LORA_RANK=${RL_LORA_RANK:-16}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-false}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-false}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-false}

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

  # If user passes a local path directly, use it.
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
  if [[ -n "${HF_ENDPOINT:-}" ]]; then
    echo "[INFO] HF_ENDPOINT=${HF_ENDPOINT}"
  fi
fi

SFT_ADAPTER_PATH=${SFT_ADAPTER_PATH:-}
if [[ -z "${SFT_ADAPTER_PATH}" ]]; then
  echo "[ERROR] SFT_ADAPTER_PATH is required."
  echo "Example: SFT_ADAPTER_PATH=output/sft_codegen1_12g_smoke/checkpoint-20 bash scripts/verl_dapo_codegen1_smoke.sh"
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

mkdir -p "${OUTPUT_DIR}"

# Work around vLLM qwen3.5 config bug that causes:
# StrictDataclassClassValidationError -> TypeError: unsupported operand type(s) for -=: 'set' and 'list'
patch_vllm_qwen35_rope_bug

if ! "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("verl.trainer.main_ppo") is None:
  print("[ERROR] verl.trainer.main_ppo not found in current Python environment.")
  print(f"[ERROR] PYTHON_BIN={sys.executable}")
  raise SystemExit(1)

print(f"[INFO] Using Python: {sys.executable}")
PY
then
  exit 1
fi

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  trainer.val_before_train=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=1 \
  data.dataloader_num_workers=0 \
  data.max_prompt_length=5000 \
  data.max_response_length=2000 \
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
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  reward.reward_manager.name=dapo \
  reward.num_workers=1 \
  +reward.reward_kwargs.overlong_buffer_cfg.enable=False \
  +reward.reward_kwargs.overlong_buffer_cfg.len=64 \
  +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
  +reward.reward_kwargs.max_resp_len=2560 \
  reward.custom_reward_function.path="${PROJECT_DIR}/scripts/verl_codegen1_reward.py" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.timeout_sec=4 \
  +reward.custom_reward_function.reward_kwargs.memory_limit_mb=1024 \
  trainer.logger='["console"]' \
  trainer.project_name='qwen_codegen1_smoke' \
  trainer.experiment_name="qwen35_08b_codegen1_dapo_smoke" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=20