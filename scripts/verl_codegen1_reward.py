#!/usr/bin/env python3
"""Custom reward function for verl on codegen1.

The reward is pass-rate over provided test cases.
"""

from __future__ import annotations

import json
from typing import Any

from data.code_excutor import ModelResponseCodeExecutor


_EXECUTOR = None


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


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    timeout_sec: int = 4,
    memory_limit_mb: int = 1024,
    **kwargs,
) -> dict[str, float | int | bool]:
    # data_source is kept for interface compatibility with verl.
    del data_source
    del kwargs

    test_cases = _parse_test_cases(ground_truth)
    if test_cases is None:
        return {"score": 0.0, "acc": 0.0, "passed": 0, "total": 0}

    fn_mode = "auto"
    if extra_info:
        fn_mode = extra_info.get("fn_mode", "auto") or "auto"

    executor = _get_executor(timeout_sec=timeout_sec, memory_limit_mb=memory_limit_mb)
    try:
        results = executor.evaluate(model_response=solution_str, test_samples=test_cases, mode=fn_mode)
    except Exception:
        return {"score": 0.0, "acc": 0.0, "passed": 0, "total": 0}

    if not results:
        return {"score": 0.0, "acc": 0.0, "passed": 0, "total": 0}

    passed = sum(1 for item in results if item.get("passed", False))
    total = len(results)
    score = float(passed) / float(total)
    return {"score": score, "acc": score, "passed": passed, "total": total}