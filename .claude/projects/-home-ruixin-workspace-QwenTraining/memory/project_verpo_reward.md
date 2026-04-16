---
name: VeRPO reward implementation
description: Group-level rho_j reward fix aligning code with paper Section 3/4
type: project
---

Implemented paper-aligned VeRPO reward (April 2026).

**Why:** Original compute_score used global EMA for rho_j (per-testcase pass rate), but paper eq.1 requires rho_j computed from the current rollout group G_x. Also R^turn was a normalised average, not a weighted sum (eq.4). Efficiency decay used response tokens, not turn count (eq.5).

**Changes made:**
- `scripts/plugins/verl_codegen1_reward.py`: Rewrote. Added `execute_single()` (raw execution, returns passed_flags). Added `compute_group_dense_reward()` (group-level rho_j → w_j → w_j' → R^turn = sum_j w_j'*p_j, weighted SUM not avg). `compute_score()` kept as EMA fallback for validation.
- `scripts/plugins/verpo_reward_manager.py`: New file. `VeRPORewardManager` registered as "verpo" in workers registry. Collects all N samples, groups by uid, computes group-level rho_j, then scores each sample.
- `verl/verl/experimental/reward_loop/reward_manager/verpo.py`: New file. Same manager but inheriting `RewardManagerBase` (async interface). Implements `run_batch()` for group-level scoring; `run_single()` falls back to EMA path.
- `verl/verl/experimental/reward_loop/reward_manager/__init__.py`: Imports VeRPORewardManager.
- `verl/verl/workers/reward_manager/__init__.py`: Imports VeRPORewardManager.
- `verl/verl/experimental/reward_loop/reward_loop.py`: `compute_score_batch()` now calls `run_batch()` if the manager has it.
- `scripts/dapo/config/dapo_qwen3_5_0_8b.yaml`: `reward_manager.name: verpo`, `efficiency_mode: turn_count`.

**How to apply:** When touching reward logic, remember rho_j must come from the group, not EMA. R^turn is a weighted SUM (not normalised avg). Decay is by turn count.
