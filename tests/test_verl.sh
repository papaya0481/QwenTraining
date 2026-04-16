export CUDA_VISIBLE_DEVICES=5

export DATA_DIR=/data/wuli_error/WRX/QwenTraining/rl_data
export VAL_FILE=${DATA_DIR}/test.parquet
export SFT_ADAPTER_PATH=/data/wuli_error/WRX/QwenTraining/output/v52-20260404-111420/checkpoint-2
export OUTPUT_DIR=/data/wuli_error/WRX/QwenTraining/outputs/verl_dapo_taco_test

bash scripts/verl_dapo_taco_verified_full.sh