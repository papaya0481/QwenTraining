"""Raw CodeGen1 reward for DAPO/GRPO training.

The reward is intentionally sparse: pass every provided test case and receive
1.0, otherwise receive 0.0. The DAPO reward manager consumes the returned
``score`` field and logs the remaining fields as reward extras.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.code_excutor import ModelResponseCodeExecutor


@lru_cache(maxsize=16)
def _get_executor(timeout_sec: int, memory_limit_mb: int) -> ModelResponseCodeExecutor:
    return ModelResponseCodeExecutor(timeout=timeout_sec, memory_limit_mb=memory_limit_mb)


def _normalize_io_keys_to_legacy(test_cases: Any) -> Any:
    def _convert(item: dict) -> dict:
        converted = dict(item)
        if "input" not in converted and "inputs" in converted:
            converted["input"] = converted.pop("inputs")
        else:
            converted.pop("inputs", None)
        if "output" not in converted and "outputs" in converted:
            converted["output"] = converted.pop("outputs")
        else:
            converted.pop("outputs", None)
        return converted

    if isinstance(test_cases, dict):
        return _convert(test_cases)
    if isinstance(test_cases, list):
        return [_convert(item) if isinstance(item, dict) else item for item in test_cases]
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


def _empty_result() -> dict[str, float | int | bool]:
    return {
        "score": 0.0,
        "acc": 0.0,
        "passed": 0,
        "total": 0,
        "pass_rate": 0.0,
        "all_passed": False,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    timeout_sec: int = 4,
    memory_limit_mb: int = 1024,
    **kwargs,
) -> dict[str, float | int | bool]:
    del data_source, kwargs

    test_cases = _parse_test_cases(ground_truth)
    if test_cases is None:
        return _empty_result()

    fn_mode = "auto"
    if extra_info:
        fn_mode = extra_info.get("fn_mode", "auto") or "auto"

    executor = _get_executor(timeout_sec=int(timeout_sec), memory_limit_mb=int(memory_limit_mb))
    try:
        results = executor.evaluate(model_response=solution_str, test_samples=test_cases, mode=fn_mode)
    except Exception:
        return _empty_result()

    if not results:
        return _empty_result()

    passed = sum(1 for item in results if item.get("passed", False))
    total = len(results)
    all_passed = passed == total
    score = 1.0 if all_passed else 0.0
    pass_rate = float(passed) / float(total)

    return {
        "score": score,
        "acc": score,
        "passed": passed,
        "total": total,
        "pass_rate": pass_rate,
        "all_passed": all_passed,
    }
