import os
import json
import hashlib
import argparse

import datasets

from verl.utils.hdfs_io import copy, makedirs


def _identity_selection(dataset):
    return dataset


def _easy_selection(dataset):
    if "difficulty" not in dataset.column_names:
        raise ValueError("Dataset does not contain a 'difficulty' column, so training_option='easy' is unavailable.")
    return dataset.filter(lambda example: example.get("difficulty") == "EASY")


TRAINING_OPTION_REGISTRY = {
    "default": {
        "select_fn": _identity_selection,
        "output_prefix": "",
        "description": "Use the current split behavior on the full dataset.",
    },
    "easy": {
        "select_fn": _easy_selection,
        "output_prefix": "easy_",
        "description": "Filter to samples with difficulty == EASY before splitting.",
    },
}


def apply_training_option(dataset, training_option):
    option_cfg = TRAINING_OPTION_REGISTRY.get(training_option)
    if option_cfg is None:
        available_options = ", ".join(sorted(TRAINING_OPTION_REGISTRY))
        raise ValueError(f"Unknown training_option: {training_option}. Available options: {available_options}")
    selected_dataset = option_cfg["select_fn"](dataset)
    return selected_dataset, option_cfg


def validate_test_size(test_size):
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"--test-size must be in the open interval (0, 1), but got {test_size}")


def maybe_limit_samples(dataset, max_samples):
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError(f"--max-samples must be a positive integer, but got {max_samples}")
    if max_samples >= len(dataset):
        return dataset
    return dataset.select(range(max_samples))


def make_map_fn(split):

    def process_fn(example, idx):
        # Build prompt from system + user fields
        prompt = [
            {"role": "system", "content": example["system"]},
            {"role": "user",   "content": example["user"]},
        ]

        # Build a stable per-test-case hash: sha256 of each individual test case,
        # truncated to 4 hex digits, prefixed with the sample index and test-case index.
        # Format: "{example_idx}_{test_case_idx}_{hash:4}"
        raw_tc = example["test_cases"] or ""
        normalized_tc = raw_tc
        try:
            tc_payload = json.loads(raw_tc)
            inputs  = tc_payload.get("input",  tc_payload.get("inputs",  []))
            outputs = tc_payload.get("output", tc_payload.get("outputs", []))
            # Normalize to {"input": [...], "output": [...]}
            normalized_payload = {"input": inputs, "output": outputs}
            normalized_tc = json.dumps(normalized_payload, ensure_ascii=False)
            n = max(len(inputs), len(outputs))
            test_hash = [
                f"{idx}_{tc_i}_{hashlib.sha256(json.dumps([inputs[tc_i] if tc_i < len(inputs) else None, outputs[tc_i] if tc_i < len(outputs) else None], ensure_ascii=False).encode()).hexdigest()[:4]}"
                for tc_i in range(n)
            ]
        except Exception:
            # Fallback: hash the whole string as a single entry
            h = hashlib.sha256(raw_tc.encode()).hexdigest()[:4]
            test_hash = [f"{idx}_0_{h}"]

        data = {
            "data_source": "taco_rl",
            "prompt": prompt,
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": normalized_tc
            },
            "extra_info": {
                "index":      idx,
                "fn_mode":    example["fn_mode"],
                "fn_name":    example["fn_name"],
                "difficulty": example["difficulty"],
                # "answer": example["answer"],
                # "test_hash":  test_hash,
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--hdfs-dir",  default=None)
    parser.add_argument(
        "--training-option",
        default="default",
        help="Training data selection option. Supported values are registered in TRAINING_OPTION_REGISTRY.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.05,
        help="Fraction of the selected dataset to use as the test split.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the total number of selected samples before train/test splitting.",
    )
    args = parser.parse_args()
    validate_test_size(args.test_size)

    data_source = "BigfufuOuO/taco_verified"
    raw_data = datasets.load_dataset(data_source, "passed")
    selected_dataset, option_cfg = apply_training_option(raw_data["train"], args.training_option)
    selected_dataset = maybe_limit_samples(selected_dataset, args.max_samples)

    print(f"Training option: {args.training_option}")
    print(f"Option description: {option_cfg['description']}")
    print(f"Selected dataset size before split: {len(selected_dataset)}")
    print(f"Max samples: {args.max_samples if args.max_samples is not None else 'all'}")
    print(f"Train/test ratio: {1.0 - args.test_size:.2%}/{args.test_size:.2%}")

    # The dataset only has a train split; split it according to the requested ratio.
    split_ds = selected_dataset.train_test_split(test_size=args.test_size, seed=42)
    train_dataset = split_ds["train"]
    test_dataset  = split_ds["test"]

    original_columns = train_dataset.column_names
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, remove_columns=original_columns)
    test_dataset  = test_dataset.map(function=make_map_fn("test"),  with_indices=True, remove_columns=original_columns)

    # Show 1 example after preprocessing
    print("=" * 60)
    print("Example after preprocessing (train[0]):")
    print("=" * 60)
    import pprint
    pprint.pprint(train_dataset[0])
    print("=" * 60)

    os.makedirs(args.local_dir, exist_ok=True)
    train_filename = f"{option_cfg['output_prefix']}train.parquet"
    test_filename = f"{option_cfg['output_prefix']}test.parquet"
    train_dataset.to_parquet(os.path.join(args.local_dir, train_filename))
    test_dataset.to_parquet(os.path.join(args.local_dir, test_filename))

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
