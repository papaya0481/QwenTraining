set -euo pipefail

script_dir=$(pwd)
verl_root=${script_dir}/verl

# Ray
ray_address=http://localhost:8265
working_dir=${verl_root}
runtime_env=${working_dir}/verl/trainer/runtime_env.yaml

# The config uses oc.env for these values, so export them for Hydra resolution.

ray job submit --no-wait \
    --address="${ray_address}" \
    --runtime-env="${runtime_env}" \
    --working-dir "${working_dir}" \
    -- python3 -m scripts.dapo.main_dapo \
    --config-name=dapo_qwen3_5_9b \
    "$@"