#!/usr/bin/env python3
"""Custom reward function for verl on codegen1.

The reward is pass-rate over provided test cases.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from data.code_excutor import ModelResponseCodeExecutor


@lru_cache(maxsize=16)
def _get_executor(timeout_sec: int, memory_limit_mb: int) -> ModelResponseCodeExecutor:
    # One cached executor per (timeout, memory_limit_mb) tuple in each worker process.
    return ModelResponseCodeExecutor(timeout=timeout_sec, memory_limit_mb=memory_limit_mb)


def _normalize_io_keys_to_legacy(test_cases: Any) -> Any:
    """Normalize top-level testcase keys to legacy input/output.

    Supported external formats:
    1) {"input": ..., "output": ...} (kept)
    2) {"inputs": ..., "outputs": ...} (mapped)
    3) [{"input"|"inputs": ..., "output"|"outputs": ...}, ...] (per-item mapped)
    """

    def _convert_case_dict(item: dict[str, Any]) -> dict[str, Any]:
        converted = dict(item)

        # Prefer explicit legacy keys when both exist.
        if "input" not in converted and "inputs" in converted:
            converted["input"] = converted["inputs"]
        if "output" not in converted and "outputs" in converted:
            converted["output"] = converted["outputs"]

        converted.pop("inputs", None)
        converted.pop("outputs", None)
        return converted

    if isinstance(test_cases, dict):
        return _convert_case_dict(test_cases)

    if isinstance(test_cases, list):
        normalized = []
        for item in test_cases:
            if isinstance(item, dict):
                normalized.append(_convert_case_dict(item))
            else:
                normalized.append(item)
        return normalized

    return test_cases


def _parse_test_cases(ground_truth: Any) -> Any:
    if ground_truth is None:
        return None
    if isinstance(ground_truth, str):
        text = ground_truth.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return _normalize_io_keys_to_legacy(parsed)
        except Exception:
            return None
    return _normalize_io_keys_to_legacy(ground_truth)


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

    timeout_sec = int(timeout_sec)
    memory_limit_mb = int(memory_limit_mb)
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