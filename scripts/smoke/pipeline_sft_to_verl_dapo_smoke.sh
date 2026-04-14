#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke pipeline (12GB vGPU):
# 1) Build verl parquet from BigfufuOuO/codegen1_merged_clean
# 2) Run Swift SFT for a few steps and save LoRA adapter
# 3) Run verl GRPO+DAPO reward manager initialized from that adapter

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "${PROJECT_DIR}"

export HF_HOME=${HF_HOME:-/root/shared-nvme/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/root/shared-nvme/.cache/huggingface/datasets}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3.5-0.8B}
SFT_OUT_DIR=${SFT_OUT_DIR:-${PROJECT_DIR}/output/sft_codegen1_12g_smoke}
VERL_DATA_DIR=${VERL_DATA_DIR:-${PROJECT_DIR}/output/verl_codegen1_data}
START_STEP=${START_STEP:-1}

# RL data source selector: codegen1 | taco
RL_DATASET_KIND=${RL_DATASET_KIND:-taco}

# If set to 1, reuse existing SFT checkpoint/adapter and skip step 2.
SKIP_SFT_IF_CHECKPOINT_EXISTS=${SKIP_SFT_IF_CHECKPOINT_EXISTS:-1}
# If set to 1, always run SFT even when an existing checkpoint is found.
FORCE_RUN_SFT=${FORCE_RUN_SFT:-0}

# If set to 1, skip step 1 when train/val parquet already exists.
SKIP_PREPROCESS_IF_DATA_EXISTS=${SKIP_PREPROCESS_IF_DATA_EXISTS:-1}

resolve_adapter_path() {
  local base_dir="$1"
  local latest_ckpt
  local latest_adapter_dir

  # Common Swift layout:
  # output_dir/v0-YYYYMMDD-HHMMSS/checkpoint-*/
  # We search recursively and pick the latest by mtime.
  latest_ckpt=$(find "${base_dir}" -maxdepth 3 -type d -name 'checkpoint-*' -printf '%T@ %p\n' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2- || true)
  if [[ -n "${latest_ckpt}" ]]; then
    echo "${latest_ckpt}"
    return 0
  fi

  # Some runs may store adapter files directly under output dir.
  if [[ -f "${base_dir}/adapter_config.json" ]]; then
    echo "${base_dir}"
    return 0
  fi

  # Also check one level deeper timestamp run dirs.
  latest_adapter_dir=$(find "${base_dir}" -maxdepth 2 -type f -name 'adapter_config.json' -printf '%T@ %p\n' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2- || true)
  if [[ -n "${latest_adapter_dir}" ]]; then
    dirname "${latest_adapter_dir}"
    return 0
  fi

  return 1
}

if [[ "${START_STEP}" -le 1 ]]; then
  if [[ "${SKIP_PREPROCESS_IF_DATA_EXISTS}" == "1" && -f "${VERL_DATA_DIR}/train.parquet" && -f "${VERL_DATA_DIR}/val.parquet" ]]; then
    echo "[1/4] Skip preprocess (existing parquet found)."
  else
    if [[ "${RL_DATASET_KIND}" == "taco" ]]; then
      echo "[1/4] Build verl parquet dataset from BigfufuOuO/taco_verified..."
      python3 scripts/verl/taco_data.py \
        --local-dir "${VERL_DATA_DIR}" \
        --val-size ${VERL_VAL_SIZE:-64} \
        --seed ${VERL_DATA_SEED:-42}
    else
      echo "[1/4] Build verl parquet dataset from BigfufuOuO/codegen1_merged_clean..."
      python3 scripts/verl_codegen1_preprocess.py \
        --dataset_id BigfufuOuO/codegen1_merged_clean \
        --subset rl \
        --split train \
        --output_dir "${VERL_DATA_DIR}" \
        --max_samples ${VERL_MAX_SAMPLES:-256} \
        --val_size ${VERL_VAL_SIZE:-32}
    fi
  fi
else
  echo "[1/4] Skip by START_STEP=${START_STEP}"
fi

if [[ "${START_STEP}" -le 2 ]]; then
  if [[ "${FORCE_RUN_SFT}" != "1" && "${SKIP_SFT_IF_CHECKPOINT_EXISTS}" == "1" ]] && SFT_ADAPTER_PATH=$(resolve_adapter_path "${SFT_OUT_DIR}"); then
    echo "[2/4] Skip SFT (reuse existing adapter): ${SFT_ADAPTER_PATH}"
  else
    echo "[2/4] Run Swift SFT smoke training..."
    swift sft \
      --model "${MODEL_NAME}" \
      --tuner_type lora \
      --external_plugins scripts/data_preprocess.py \
      --dataset codegen1_train:sft#${SFT_TRAIN_SAMPLES:-256} \
      --val_dataset codegen1_sft_val#${SFT_VAL_SAMPLES:-32} \
      --load_from_cache_file true \
      --use_hf \
      --torch_dtype bfloat16 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --learning_rate 1e-4 \
      --lora_rank 16 \
      --lora_alpha 32 \
      --target_modules all-linear \
      --gradient_accumulation_steps 8 \
      --gradient_checkpointing true \
      --eval_steps 10 \
      --save_steps 10 \
      --save_total_limit 2 \
      --logging_steps 5 \
      --max_length 4096 \
      --truncation_strategy right \
      --output_dir "${SFT_OUT_DIR}" \
      --warmup_ratio 0.03 \
      --dataloader_num_workers 2
  fi
else
  echo "[2/4] Skip by START_STEP=${START_STEP}"
fi

echo "[3/4] Resolve SFT adapter path..."
if ! SFT_ADAPTER_PATH=$(resolve_adapter_path "${SFT_OUT_DIR}"); then
  echo "[ERROR] Cannot find SFT adapter under: ${SFT_OUT_DIR}"
  exit 1
fi
echo "Using adapter: ${SFT_ADAPTER_PATH}"

echo "[4/4] Run verl DAPO smoke from SFT adapter..."
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH}" \
DATA_DIR="${VERL_DATA_DIR}" \
MODEL_NAME="${MODEL_NAME}" \
HF_ENDPOINT="${HF_ENDPOINT:-}" \
bash scripts/verl_dapo_codegen1_smoke.sh

echo "Pipeline finished."
