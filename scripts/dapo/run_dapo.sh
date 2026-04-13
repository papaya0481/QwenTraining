set -euo pipefail
NGPUS=1
TP=1

export HYDRA_FULL_ERROR=1
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
export LD_LIBRARY_PATH=/data/wuli_error/miniconda3/envs/llmqw/lib

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# verl_root=${script_dir}/verl

# Ray
ray_address=http://localhost:8900
working_dir=${script_dir}
runtime_env=${script_dir}/verl/verl/trainer/runtime_env.yaml

# The config uses oc.env for these values, so export them for Hydra resolution.

ray job submit \
    --address="${ray_address}" \
    --runtime-env="${runtime_env}" \
    --working-dir "${working_dir}" \
    -- python3 -m scripts.dapo.main_dapo \
    --config-name=dapo_qwen3_5_0_8b \
    trainer.n_gpus_per_node=$NGPUS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    "$@"