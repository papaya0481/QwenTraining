import os
import json
import hashlib
import argparse
import datasets

from verl.utils.hdfs_io import copy, makedirs


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
        try:
            tc_payload = json.loads(raw_tc)
            inputs  = tc_payload.get("input",  tc_payload.get("inputs",  []))
            outputs = tc_payload.get("output", tc_payload.get("outputs", []))
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
            "data_source": "taco",
            "prompt": prompt,
            "ability": "code",
            "ground_truth": example["answer"],
            "extra_info": {
                "index":      idx,
                "fn_mode":    example["fn_mode"],
                "fn_name":    example["fn_name"],
                "difficulty": example["difficulty"],
                "test_cases": example["test_cases"],
                "test_hash":  test_hash,
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--hdfs-dir",  default=None)
    args = parser.parse_args()

    data_source = "BigfufuOuO/taco_verified"
    raw_data = datasets.load_dataset(data_source, "passed")

    # The dataset only has a train split; use a 95/5 train-test split.
    # split_ds = raw_data["train"].train_test_split(test_size=0.05, seed=42)
    train_dataset = raw_data["train"]

    original_columns = train_dataset.column_names
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, remove_columns=original_columns)
    # test_dataset  = test_dataset.map(function=make_map_fn("test"),  with_indices=True)

    # Show 1 example after preprocessing
    print("=" * 60)
    print("Example after preprocessing (train[0]):")
    print("=" * 60)
    import pprint
    pprint.pprint(train_dataset[0])
    print("=" * 60)

    os.makedirs(args.local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    # test_dataset.to_parquet(os.path.join(args.local_dir,  "test.parquet"))

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
