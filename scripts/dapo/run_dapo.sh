set -euo pipefail
NGPUS=2
TP=1

# export LD_LIBRARY_PATH=/data/wuli_error/miniconda3/envs/llmqw/lib
# export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_USE_V1=1
# export CUDA_VISIBLE_DEVICES=2
export PJ_ROOT=$(pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# verl_root=${script_dir}/verl

DATA_DIR=${DATA_DIR:-${script_dir}/rl_data}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3.5-0.8B}
export EXP_TIME_SUFFIX=$(date +%m-%d_%H%M)

# Ray
ray_address=http://localhost:8900
working_dir=${script_dir}
runtime_env=${script_dir}/verl/verl/trainer/runtime_env.yaml

# The config uses oc.env for these values, so export them for Hydra resolution.

python3 -m scripts.dapo.main_dapo \
    --config-path="${script_dir}/scripts/dapo/config" \
    --config-name=dapo_qwen3_5_0_8b \
    trainer.n_gpus_per_node=$NGPUS \
    trainer.experiment_name=DAPO-Qwen3.5-0.8B_${EXP_TIME_SUFFIX} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    "$@"

# ray job submit \
#     --address="${ray_address}" \
#     --runtime-env="${runtime_env}" \
#     --working-dir "${working_dir}" \
#     -- python3 -m scripts.dapo.main_dapo \
#     --config-name=dapo_qwen3_5_0_8b \
#     trainer.n_gpus_per_node=$NGPUS \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
#     "$@"