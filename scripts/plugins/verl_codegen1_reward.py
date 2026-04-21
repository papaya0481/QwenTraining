"""Custom reward function for VeRPO on codegen1.

Key fix vs. original: rho_j (per-testcase pass rate) is computed from the
current rollout GROUP, not from a global EMA.  The reward manager
(VeRPORewardManager in verpo_reward_manager.py) collects all N trajectories
for a prompt, calls compute_score_single to get raw execution results, then
calls compute_group_dense_reward to compute the group-level rho_j / w_j / w_j'
and the final R^turn = sum_j w_j' * p_j  (weighted sum, not normalised avg).

compute_score is kept as the single-sample entry-point used by DAPORewardManager
for backward compatibility (e.g. validation), but it now falls back to a
per-sample EMA only when called outside a group context.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from functools import lru_cache
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.code_excutor import ModelResponseCodeExecutor


_TEST_STAT_LOCK = threading.Lock()
# Fallback global EMA stats (used only when called outside a group context).
# key -> {"ema_pass_rate": float, "seen": int}
_TESTCASE_STATS: dict[str, dict[str, float | int]] = {}


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _get_executor(timeout_sec: int, memory_limit_mb: int) -> ModelResponseCodeExecutor:
    return ModelResponseCodeExecutor(timeout=timeout_sec, memory_limit_mb=memory_limit_mb)


# ---------------------------------------------------------------------------
# Test-case key helpers
# ---------------------------------------------------------------------------

def _normalize_io_keys_to_legacy(test_cases: Any) -> Any:
    def _convert(item: dict) -> dict:
        c = dict(item)
        if "input" not in c and "inputs" in c:
            c["input"] = c.pop("inputs")
        else:
            c.pop("inputs", None)
        if "output" not in c and "outputs" in c:
            c["output"] = c.pop("outputs")
        else:
            c.pop("outputs", None)
        return c

    if isinstance(test_cases, dict):
        return _convert(test_cases)
    if isinstance(test_cases, list):
        return [_convert(i) if isinstance(i, dict) else i for i in test_cases]
    return test_cases


def _parse_test_cases(ground_truth: Any) -> Any:
    if ground_truth is None:
        return None
    if isinstance(ground_truth, str):
        text = ground_truth.strip()
        if not text:
            return None
        try:
            return _normalize_io_keys_to_legacy(json.loads(text))
        except Exception:
            return None
    return _normalize_io_keys_to_legacy(ground_truth)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _get_testcase_keys(extra_info: dict[str, Any] | None, total: int) -> list[str]:
    if not extra_info:
        return [f"tc_{i}" for i in range(total)]
    test_hash = extra_info.get("test_hash")
    if isinstance(test_hash, list) and len(test_hash) == total:
        return [str(x) for x in test_hash]
    index = extra_info.get("index", "na")
    return [f"{index}_tc_{i}" for i in range(total)]


# ---------------------------------------------------------------------------
# Fallback EMA stats (single-sample path only)
# ---------------------------------------------------------------------------

def _ema_estimate_pass_rates(keys: list[str], default_pass_rate: float) -> list[float]:
    rates: list[float] = []
    with _TEST_STAT_LOCK:
        for key in keys:
            stat = _TESTCASE_STATS.get(key)
            rates.append(_safe_float(stat.get("ema_pass_rate"), default_pass_rate) if stat else default_pass_rate)
    return rates


def _ema_update_stats(keys: list[str], passed_flags: list[float], momentum: float) -> None:
    with _TEST_STAT_LOCK:
        for key, passed in zip(keys, passed_flags):
            stat = _TESTCASE_STATS.get(key)
            if stat is None:
                _TESTCASE_STATS[key] = {"ema_pass_rate": passed, "seen": 1}
            else:
                old = _safe_float(stat.get("ema_pass_rate"), passed)
                _TESTCASE_STATS[key] = {
                    "ema_pass_rate": momentum * old + (1.0 - momentum) * passed,
                    "seen": int(stat.get("seen", 0)) + 1,
                }


# ---------------------------------------------------------------------------
# Core dense-reward computation (paper Section 3)
# ---------------------------------------------------------------------------

def compute_group_dense_reward(
    group_passed_flags: list[list[float]],  # shape [N_samples][N_tests]
    query_passed_flags: list[float],        # shape [N_tests] — the sample being scored
    difficulty_alpha: float,
    density_eps: float,
    sigma_floor: float,
) -> tuple[float, float, float]:
    """Compute R^turn for one sample using group-level rho_j.

    Paper eq. (1):  rho_j = (sum over all turns/trajectories of p_{t,i}^{(j)})
                             / (total number of turns across group)
    For single-turn: each trajectory has 1 turn, so denominator = N.

    Paper eq. (4):  R^turn = sum_j  w_j' * p_j   (weighted SUM, not avg)
    """
    if not group_passed_flags or not query_passed_flags:
        return 0.0, 0.0, sigma_floor

    # rho_j: group-level pass rate per test case (eq. 1)  shape: [N_tests]
    group_arr = np.array(group_passed_flags, dtype=np.float64)  # [N_samples, N_tests]
    rho = group_arr.mean(axis=0)                                 # [N_tests]

    # w_j = exp(-alpha * rho_j)  (eq. 2)
    base_weights = np.exp(-difficulty_alpha * rho)               # [N_tests]

    # KDE bandwidth sigma
    sigma = float(max(rho.std() / 2.0, sigma_floor))

    # w_j' = w_j / (density_j + eps)  (eq. 3)
    # density_j = sum_k exp(-(rho_j - rho_k)^2 / (2*sigma^2 + eps))
    diff = rho[:, None] - rho[None, :]                           # [N_tests, N_tests]
    density = np.exp(-(diff ** 2) / (2.0 * sigma * sigma + density_eps)).sum(axis=1)
    normalized_weights = base_weights / (density + density_eps)  # [N_tests]

    # R^turn = sum_j w_j' * p_j  (eq. 4 — weighted SUM, not normalised avg)
    query_arr = np.asarray(query_passed_flags, dtype=np.float64)
    r_turn = float(np.dot(normalized_weights, query_arr))
    avg_weight = float(base_weights.mean())
    return r_turn, avg_weight, sigma


# ---------------------------------------------------------------------------
# Single-sample execution (returns raw results for group aggregation)
# ---------------------------------------------------------------------------

def execute_single(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
    timeout_sec: int,
    memory_limit_mb: int,
) -> dict[str, Any] | None:
    """Run the code executor and return raw per-testcase results.

    Returns None on parse/execution failure.
    Returns dict with keys:
        passed_flags: list[float]  (1.0 / 0.0 per test)
        passed: int
        total: int
        pass_rate: float
        testcase_keys: list[str]
        outcome_reward: float  (1.0 if all pass, else 0.0)
        turn_count: int
    """
    test_cases = _parse_test_cases(ground_truth)
    if test_cases is None:
        return None

    fn_mode = "auto"
    if extra_info:
        fn_mode = extra_info.get("fn_mode", "auto") or "auto"

    executor = _get_executor(timeout_sec=int(timeout_sec), memory_limit_mb=int(memory_limit_mb))
    try:
        results = executor.evaluate(model_response=solution_str, test_samples=test_cases, mode=fn_mode)
    except Exception:
        return None

    if not results:
        return None

    passed_flags = [1.0 if item.get("passed", False) else 0.0 for item in results]
    passed = int(sum(passed_flags))
    total = len(results)
    testcase_keys = _get_testcase_keys(extra_info, total)

    turn_count = 1
    if extra_info is not None:
        turn_count = int(extra_info.get("turn_count", 1))

    return {
        "passed_flags": passed_flags,
        "passed": passed,
        "total": total,
        "pass_rate": float(passed) / float(total),
        "testcase_keys": testcase_keys,
        "outcome_reward": 1.0 if passed == total else 0.0,
        "turn_count": turn_count,
    }


def _empty_result() -> dict[str, float | int | bool]:
    return {
        "score": 0.0,
        "acc": 0.0,
        "passed": 0,
        "total": 0,
        "pass_rate": 0.0,
        "dense_reward": 0.0,
        "traj_reward": 0.0,
        "efficiency_decay": 1.0,
    }


# ---------------------------------------------------------------------------
# compute_score: single-sample entry-point (fallback / validation path)
# Uses EMA-based rho_j when no group context is available.
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    timeout_sec: int = 4,
    memory_limit_mb: int = 1024,
    difficulty_alpha: float = 2.0,
    stats_momentum: float = 0.9,
    default_pass_rate: float = 0.5,
    density_eps: float = 1e-6,
    density_sigma_floor: float = 1e-3,
    dense_reward_weight: float = 1.0,
    traj_reward_weight: float = 1.0,
    efficiency_gamma: float = 0.995,
    efficiency_mode: str = "turn_count",
    **kwargs,
) -> dict[str, float | int | bool]:
    del data_source, kwargs

    raw = execute_single(solution_str, ground_truth, extra_info, timeout_sec, memory_limit_mb)
    if raw is None:
        return _empty_result()

    passed_flags = raw["passed_flags"]
    testcase_keys = raw["testcase_keys"]

    # EMA-based rho_j (fallback — not group-level)
    pass_rates = _ema_estimate_pass_rates(testcase_keys, default_pass_rate)

    # Use single-sample as its own "group" of size 1 for the dense reward formula
    dense_reward, avg_difficulty_weight, density_sigma = compute_group_dense_reward(
        group_passed_flags=[passed_flags],
        query_passed_flags=passed_flags,
        difficulty_alpha=difficulty_alpha,
        density_eps=density_eps,
        sigma_floor=density_sigma_floor,
    )

    _ema_update_stats(testcase_keys, passed_flags, momentum=stats_momentum)

    # Efficiency decay
    efficiency_decay = 1.0
    if efficiency_mode == "turn_count":
        turn_count = raw["turn_count"]
        efficiency_decay = efficiency_gamma ** max(turn_count, 1)
    elif efficiency_mode == "response_tokens":
        response_tokens = max(len(solution_str.split()), 1)
        efficiency_decay = efficiency_gamma ** max(response_tokens - 1, 0)

    traj_reward = raw["outcome_reward"] * efficiency_decay

    mix_denom = max(dense_reward_weight + traj_reward_weight, 1e-8)
    mixed_reward_preview = (dense_reward_weight * dense_reward + traj_reward_weight * traj_reward) / mix_denom

    return {
        "score": traj_reward,  # Final reward used by training in this path.
        "acc": raw["pass_rate"],  # Fraction of test cases passed by this completion, in [0, 1].
        "passed": raw["passed"],  # Count of passed test cases.
        "total": raw["total"],  # Count of all evaluated test cases.
        "pass_rate": raw["pass_rate"],  # Same value as acc: passed / total.
        "dense_reward": dense_reward,  # Dense shaping reward from per-test-case outcomes and difficulty weights.
        "traj_reward": traj_reward,  # Outcome reward after applying the efficiency penalty.
        "mixed_reward_preview": mixed_reward_preview,  # Debug-only preview of the weighted mix of dense_reward and traj_reward.
        "outcome_reward": raw["outcome_reward"],  # Raw task outcome before efficiency penalty: 1.0 if all tests pass, else 0.0.
        "efficiency_decay": efficiency_decay,  # Multiplier that penalizes inefficient trajectories, e.g. too many turns or tokens.
        "avg_difficulty_weight": avg_difficulty_weight,  # Average testcase weight used when computing dense_reward.
        "density_sigma": density_sigma,  # Spread term from the dense reward formula, kept for debugging/analysis.
    }
