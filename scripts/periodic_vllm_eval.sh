#!/usr/bin/env bash
set -euo pipefail

# Periodically evaluate newly generated checkpoints with external vLLM-based swift eval.
#
# Usage:
#   bash scripts/periodic_vllm_eval.sh <run_dir> <base_model> [train_pid]
#
# Example:
#   bash scripts/periodic_vllm_eval.sh output/v19-20260403-171649 Qwen/Qwen3.5-0.8B 12345
#
# Optional env vars:
#   EVAL_GPU=1
#   POLL_SECONDS=30
#   EVAL_DATASET=live_code_bench
#   EVAL_LIMIT=20
#   EVAL_DATASET_ARGS='{"live_code_bench": {"trust_remote_code": true, "extra_params": {"start_date": "2024-08-01", "end_date": "2025-12-31", "scenario": "codegeneration"}}}'
#   EVAL_OUTPUT_DIR=<run_dir>/external_eval_vllm

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/periodic_vllm_eval.sh <run_dir> <base_model> [train_pid]"
  exit 1
fi

RUN_DIR="$1"
BASE_MODEL="$2"
TRAIN_PID="${3:-}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[ERROR] run_dir not found: $RUN_DIR"
  exit 1
fi

POLL_SECONDS="${POLL_SECONDS:-30}"
EVAL_GPU="${EVAL_GPU:-0}"
EVAL_DATASET="${EVAL_DATASET:-live_code_bench}"
EVAL_LIMIT="${EVAL_LIMIT:-20}"
EVAL_DATASET_ARGS="${EVAL_DATASET_ARGS:-{\"live_code_bench\": {\"trust_remote_code\": true, \"extra_params\": {\"start_date\": \"2024-08-01\", \"end_date\": \"2025-12-31\", \"scenario\": \"codegeneration\"}}}}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$RUN_DIR/external_eval_vllm}"

DONE_FILE="$RUN_DIR/.external_vllm_eval_done"
mkdir -p "$EVAL_OUTPUT_DIR"
touch "$DONE_FILE"

echo "[INFO] run_dir=$RUN_DIR"
echo "[INFO] base_model=$BASE_MODEL"
echo "[INFO] eval_gpu=$EVAL_GPU"
echo "[INFO] poll_seconds=$POLL_SECONDS"
echo "[INFO] eval_dataset=$EVAL_DATASET"
echo "[INFO] eval_limit=$EVAL_LIMIT"
echo "[INFO] eval_output_dir=$EVAL_OUTPUT_DIR"
if [[ -n "$TRAIN_PID" ]]; then
  echo "[INFO] train_pid=$TRAIN_PID"
fi

has_pending_checkpoints() {
  local ckpt
  while IFS= read -r ckpt; do
    if ! grep -Fxq "$ckpt" "$DONE_FILE"; then
      return 0
    fi
  done < <(find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
  return 1
}

eval_one_checkpoint() {
  local ckpt="$1"
  local stamp
  stamp="$(date +%Y%m%d-%H%M%S)"
  local ckpt_name
  ckpt_name="$(basename "$ckpt")"
  local log_file="$EVAL_OUTPUT_DIR/eval_${ckpt_name}_${stamp}.log"

  echo "[INFO] Start vLLM eval for $ckpt_name"
  set +e
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" swift eval \
    --model "$BASE_MODEL" \
    --adapters "$ckpt" \
    --infer_backend vllm \
    --eval_backend Native \
    --eval_dataset "$EVAL_DATASET" \
    --eval_dataset_args "$EVAL_DATASET_ARGS" \
    --eval_limit "$EVAL_LIMIT" \
    --eval_num_proc 1 \
    --eval_output_dir "$EVAL_OUTPUT_DIR" \
    >"$log_file" 2>&1
  local status=$?
  set -e

  if [[ $status -eq 0 ]]; then
    echo "$ckpt" >> "$DONE_FILE"
    echo "[INFO] Eval success: $ckpt_name"
    echo "[INFO] Eval log: $log_file"
  else
    echo "[WARN] Eval failed: $ckpt_name"
    echo "[WARN] Eval log: $log_file"
    echo "[WARN] Keep checkpoint pending for retry in next polling round."
  fi
}

while true; do
  while IFS= read -r ckpt; do
    if grep -Fxq "$ckpt" "$DONE_FILE"; then
      continue
    fi
    eval_one_checkpoint "$ckpt"
  done < <(find "$RUN_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

  if [[ -n "$TRAIN_PID" ]]; then
    if ! kill -0 "$TRAIN_PID" >/dev/null 2>&1; then
      if ! has_pending_checkpoints; then
        echo "[INFO] Training process ended and no pending checkpoint. Exit."
        break
      fi
      echo "[INFO] Training process ended, finishing pending checkpoints..."
    fi
  fi

  sleep "$POLL_SECONDS"
done
