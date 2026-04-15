#!/usr/bin/env python3
"""Custom reward function for verl on codegen1.

The reward is pass-rate over provided test cases.
"""

from __future__ import annotations

import json
import math
import sys
import threading
from pathlib import Path
from typing import Any

# Ensure project-root imports work when this file is loaded by Ray workers.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.code_excutor import ModelResponseCodeExecutor


_EXECUTOR = None
_TEST_STAT_LOCK = threading.Lock()
# Global online statistics for testcase pass-rate estimation.
# key -> {"ema_pass_rate": float, "seen": int}
_TESTCASE_STATS: dict[str, dict[str, float | int]] = {}


def _get_executor(timeout_sec: int, memory_limit_mb: int) -> ModelResponseCodeExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ModelResponseCodeExecutor(timeout=timeout_sec, memory_limit_mb=memory_limit_mb)
    return _EXECUTOR


def _parse_test_cases(ground_truth: Any) -> Any:
    if ground_truth is None:
        return None
    if isinstance(ground_truth, str):
        text = ground_truth.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None
    return ground_truth


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


def _estimate_pass_rates_before_update(keys: list[str], default_pass_rate: float) -> list[float]:
    rates: list[float] = []
    with _TEST_STAT_LOCK:
        for key in keys:
            stat = _TESTCASE_STATS.get(key)
            if stat is None:
                rates.append(default_pass_rate)
            else:
                rates.append(_safe_float(stat.get("ema_pass_rate"), default_pass_rate))
    return rates


def _update_testcase_stats(keys: list[str], passed_flags: list[float], momentum: float) -> None:
    with _TEST_STAT_LOCK:
        for key, passed in zip(keys, passed_flags):
            stat = _TESTCASE_STATS.get(key)
            if stat is None:
                _TESTCASE_STATS[key] = {"ema_pass_rate": passed, "seen": 1}
                continue

            old_rate = _safe_float(stat.get("ema_pass_rate"), passed)
            seen = int(stat.get("seen", 0))
            new_rate = momentum * old_rate + (1.0 - momentum) * passed
            _TESTCASE_STATS[key] = {"ema_pass_rate": new_rate, "seen": seen + 1}


def _compute_dynamic_dense_reward(
    pass_rates: list[float],
    passed_flags: list[float],
    difficulty_alpha: float,
    density_eps: float,
    sigma_floor: float,
) -> tuple[float, float, float]:
    if not pass_rates:
        return 0.0, 0.0, sigma_floor

    base_weights = [math.exp(-difficulty_alpha * rho) for rho in pass_rates]

    mean_rho = sum(pass_rates) / len(pass_rates)
    var_rho = sum((rho - mean_rho) ** 2 for rho in pass_rates) / len(pass_rates)
    sigma = max(math.sqrt(var_rho) / 2.0, sigma_floor)

    normalized_weights: list[float] = []
    for rho_j, w_j in zip(pass_rates, base_weights):
        density = 0.0
        for rho_k in pass_rates:
            density += math.exp(-((rho_j - rho_k) ** 2) / (2.0 * sigma * sigma + density_eps))
        normalized_weights.append(w_j / (density + density_eps))

    weight_sum = sum(normalized_weights)
    if weight_sum <= 0:
        return 0.0, 0.0, sigma

    weighted_pass = sum(w * p for w, p in zip(normalized_weights, passed_flags)) / weight_sum
    avg_weight = sum(base_weights) / len(base_weights)
    return weighted_pass, avg_weight, sigma


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
    efficiency_mode: str = "response_tokens",
    **kwargs,
) -> dict[str, float | int | bool]:
    # data_source is kept for interface compatibility with verl.
    del data_source
    del kwargs

    test_cases = _parse_test_cases(ground_truth)
    if test_cases is None:
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

    fn_mode = "auto"
    if extra_info:
        fn_mode = extra_info.get("fn_mode", "auto") or "auto"

    executor = _get_executor(timeout_sec=timeout_sec, memory_limit_mb=memory_limit_mb)
    try:
        results = executor.evaluate(model_response=solution_str, test_samples=test_cases, mode=fn_mode)
    except Exception:
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

    if not results:
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

    passed = sum(1 for item in results if item.get("passed", False))
    total = len(results)
    pass_rate = float(passed) / float(total)

    passed_flags = [1.0 if item.get("passed", False) else 0.0 for item in results]
    testcase_keys = _get_testcase_keys(extra_info=extra_info, total=total)
    pass_rates = _estimate_pass_rates_before_update(testcase_keys, default_pass_rate=default_pass_rate)

    dense_reward, avg_difficulty_weight, density_sigma = _compute_dynamic_dense_reward(
        pass_rates=pass_rates,
        passed_flags=passed_flags,
        difficulty_alpha=difficulty_alpha,
        density_eps=density_eps,
        sigma_floor=density_sigma_floor,
    )

    # Update online statistics after current sample is scored.
    _update_testcase_stats(testcase_keys, passed_flags, momentum=stats_momentum)

    # Trajectory-level outcome anchor (single-turn setup).
    outcome_reward = 1.0 if passed == total else 0.0

    # Efficiency decay: in single-turn training, use response length as trajectory proxy.
    efficiency_decay = 1.0
    if efficiency_mode == "response_tokens":
        response_tokens = max(len(solution_str.split()), 1)
        efficiency_decay = efficiency_gamma ** max(response_tokens - 1, 0)
    elif efficiency_mode == "turn_count":
        turn_count = 1
        if extra_info is not None:
            turn_count = int(extra_info.get("turn_count", 1))
        efficiency_decay = efficiency_gamma ** max(turn_count, 1)

    traj_reward = outcome_reward * efficiency_decay

    # Keep reward channels separated for VeRPO:
    # - score: trajectory anchor used by default reward tensor path
    # - dense_reward / traj_reward: consumed by verpo advantage estimator separately
    # We still expose a preview mixed score for diagnostics only.
    mix_denom = max(dense_reward_weight + traj_reward_weight, 1e-8)
    mixed_reward_preview = (dense_reward_weight * dense_reward + traj_reward_weight * traj_reward) / mix_denom

    return {
        "score": traj_reward,
        "acc": pass_rate,
        "passed": passed,
        "total": total,
        "pass_rate": pass_rate,
        "dense_reward": dense_reward,
        "traj_reward": traj_reward,
        "mixed_reward_preview": mixed_reward_preview,
        "outcome_reward": outcome_reward,
        "efficiency_decay": efficiency_decay,
        "avg_difficulty_weight": avg_difficulty_weight,
        "density_sigma": density_sigma,
    }