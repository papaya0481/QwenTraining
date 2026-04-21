#!/usr/bin/env python3
"""Prepare BigfufuOuO/codegen1_merged_clean (rl subset) for verl RL training.

Output schema follows verl RL dataset expectations:
- prompt: list[dict(role, content)]
- data_source: str
- reward_model: dict(ground_truth=...)
- extra_info: dict(index=..., fn_mode=..., ...)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from datasets import Dataset, load_dataset


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _build_record(row: dict[str, Any], idx: int) -> dict[str, Any]:
    system = row.get("system", "") or ""
    user = row.get("user", "") or ""
    test_cases = _safe_json_loads(row.get("test_cases"))
    fn_mode = row.get("fn_mode", "auto") or "auto"

    prompt = []
    if system:
        prompt.append({"role": "system", "content": system})
    prompt.append({"role": "user", "content": user})

    return {
        "prompt": prompt,
        "data_source": "codegen1_custom",
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": test_cases,
        },
        "extra_info": {
            "index": idx,
            "fn_mode": fn_mode,
            "task": row.get("task", "code_generation"),
            "source_dataset": row.get("source_dataset", "BigfufuOuO/codegen1_merged_clean"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default="BigfufuOuO/codegen1_merged_clean")
    parser.add_argument("--subset", default="rl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="output/verl_codegen1_data")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--val_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_id, args.subset, split=args.split)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    records = [_build_record(dataset[i], i) for i in range(len(dataset))]
    out = Dataset.from_list(records)

    val_size = max(1, min(args.val_size, len(out) // 5 if len(out) > 5 else 1))
    splits = out.train_test_split(test_size=val_size, seed=args.seed, shuffle=True)

    os.makedirs(args.output_dir, exist_ok=True)
    train_file = os.path.join(args.output_dir, "train.parquet")
    val_file = os.path.join(args.output_dir, "val.parquet")
    splits["train"].to_parquet(train_file)
    splits["test"].to_parquet(val_file)

    print(f"Saved train: {train_file} ({len(splits['train'])} samples)")
    print(f"Saved val:   {val_file} ({len(splits['test'])} samples)")


if __name__ == "__main__":
    main()
