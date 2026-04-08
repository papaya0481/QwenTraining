from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
import importlib.util
import json
from pathlib import Path
from datasketch import MinHash, MinHashLSH
import re
from transformers import AutoTokenizer

FENCED_BLOCK_PATTERN = re.compile(r"```([^\n`]*)\n([\s\S]*?)```")


def normalize_python_fence_after_think(text):
    """在 </think> 之后，把非 python 标记的代码块统一改为 ```python。"""
    if not isinstance(text, str) or not text:
        return text

    think_end = text.find("</think>")
    if think_end == -1:
        return text

    split_pos = think_end + len("</think>")
    prefix = text[:split_pos]
    suffix = text[split_pos:]

    def _replace_block(match):
        lang = (match.group(1) or "").strip().lower()
        code = match.group(2)
        if lang.startswith("python"):
            return match.group(0)
        return f"```python\n{code}```"

    return prefix + FENCED_BLOCK_PATTERN.sub(_replace_block, suffix)


def normalize_assistant_python_fence(sample):
    assistant = sample.get("assistant")
    if isinstance(assistant, str):
        sample["assistant"] = normalize_python_fence_after_think(assistant)
    return sample


def parse_test_cases(raw_test_cases):
    if raw_test_cases is None:
        return None

    data = raw_test_cases
    if isinstance(raw_test_cases, str):
        try:
            data = json.loads(raw_test_cases)
        except Exception:
            return None

    if isinstance(data, dict):
        if "inputs" in data and "outputs" in data:
            return {"input": data["inputs"], "output": data["outputs"]}
        if "input" in data and "output" in data:
            return data
        return None

    if isinstance(data, list):
        return data

    return None


def _sample_result(solution, raw_test_cases, mode="auto"):
    """Returns (passed: bool, error_reason: str | None)"""
    solution = solution or ""
    test_samples = parse_test_cases(raw_test_cases)
    if not solution.strip():
        return False, "empty solution"
    if not test_samples:
        return False, "no test cases"

    try:
        results = executor.evaluate(
            model_response=solution,
            test_samples=test_samples,
            mode=mode,
        )
    except Exception as e:
        return False, f"executor exception: {e}"

    if not results:
        return False, "empty results"

    failed = [item for item in results if not item.get("passed", False)]
    if failed:
        reasons = {
            "pass": [item.get("passed", False) for item in results],
            "error_code": [item.get("error_code", "UNKNOWN") if not item.get("passed", False) else "" for item in results],
            "error_message": [item.get("error_message", "Unknown Error") if not item.get("passed", False) else "" for item in results],
        }
        return False, reasons

    return True, None


def mark_sample_passed(sample):
    passed, error_reason = _sample_result(
        sample.get("assistant", ""),
        sample.get("test_cases"),
        mode=sample.get("fn_mode", "auto")
    )
    return {
        "_sample_passed": passed,
        "_error": "" if error_reason is None else (error_reason if isinstance(error_reason, str) else json.dumps(error_reason)),
    }


def get_ngrams(text, n=3):
    """Extract n-grams from text"""
    text = re.sub(r'\s+', '', text)
    return [text[i:i+n] for i in range(len(text)-n+1)]


def deduplicate_with_lsh(dataset, column_name='user', threshold=0.8, num_perm=128):
    """Deduplicate dataset using LSH on specified column"""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_indices = []

    print(f"Starting LSH deduplication on column '{column_name}' with threshold={threshold}")

    for idx in range(len(dataset)):
        text = dataset[idx][column_name]
        if not text or not isinstance(text, str):
            continue

        ngrams = set(get_ngrams(text))
        if not ngrams:
            continue

        m = MinHash(num_perm=num_perm)
        for ngram in ngrams:
            m.update(ngram.encode('utf8'))

        result = lsh.query(m)

        if not result:
            lsh.insert(str(idx), m)
            unique_indices.append(idx)

        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1} samples, kept {len(unique_indices)} unique")

    print(f"Deduplication complete: {len(unique_indices)} / {len(dataset)} samples kept")
    return unique_indices


if __name__ == "__main__":
    data_openthought = load_dataset("BigfufuOuO/code_gen", split="train")
    data_codeforce = load_dataset("ZyWOuO/clean-codeforces", split="train")

    # 1. 记录原始数据量
    original_size = len(data_codeforce)

    # 2. 定义验证函数（只做判断，不修改数据）
    def is_valid_test_cases(example):
        tc = example.get("test_cases", None)
        
        # 拒绝 None 或空数据
        if not tc:
            return False
            
        # 如果是字典或列表，测试是否能被正常 JSON 序列化
        if isinstance(tc, (dict, list)):
            try:
                json.dumps(tc)
                return True
            except Exception:
                return False
                
        # 如果已经是字符串了，予以放行（或者你也可以加个 json.loads 测试它是不是合法JSON串）
        if isinstance(tc, str):
            return True
            
        # 其他奇奇怪怪的类型全部拒绝
        return False

    # 3. 执行过滤，丢弃不合格的样本
    data_codeforce = data_codeforce.filter(is_valid_test_cases, load_from_cache_file=False)

    # 4. 统计并打印丢弃结果
    valid_size = len(data_codeforce)
    discarded_size = original_size - valid_size
    print(f"🚨 拦截清理完毕！共丢弃了 {discarded_size} 个异常样本，保留了 {valid_size} 个有效样本。")

    # 5. 定义转换函数（此时进来的都是绝对安全的数据了）
    def normalize_test_cases(example, idx):
        tc = example["test_cases"]
        if isinstance(tc, (dict, list)):
            example["test_cases"] = json.dumps(tc)
        return example

    # 6. 执行映射，强制对齐 Schema
    data_codeforce = data_codeforce.map(
        normalize_test_cases, 
        with_indices=True, 
        load_from_cache_file=False,
        features=data_openthought.features # 注入目标特征
    )
    # --------------------------------------------------------

    merged_data = concatenate_datasets([data_openthought, data_codeforce])
    merged_data = merged_data.add_column("fn_mode", ["stdio"] * len(merged_data))

    print(merged_data)
    for col in merged_data.column_names:
        print(f"Column: {col}, Type: {merged_data.features[col]}")

    # Deduplicate using LSH on 'user' column
    unique_indices = deduplicate_with_lsh(merged_data, column_name='user', threshold=0.8, num_perm=128)
    merged_data = merged_data.select(unique_indices)
    print(f"After deduplication: {len(merged_data)} samples")

    # 规范化 assistant：仅处理 </think> 之后的代码块，把 ``` 统一为 ```python。
    merged_data = merged_data.map(normalize_assistant_python_fence, num_proc=16)
    print("Normalized assistant code fences after </think> to use python language tag.")
    
    # 去除user, assistant, sytem中最外层的引号
    def remove_outer_quotes(sample):
        for key in ["test_cases", "user", "assistant", "system"]:
            if key in sample and isinstance(sample[key], str):
                sample[key] = sample[key].strip('"')
        return sample
    
    def has_outer_quotes(sample):
        for key in ["test_cases", "user", "assistant", "system"]:
            if key in sample and isinstance(sample[key], str):
                text = sample[key].strip()
                if (
                    (text.startswith('"') and text.endswith('"')) or
                    (text.startswith("'") and text.endswith("'"))
                ):
                    return True
        return False

    # merged_data = merged_data.filter(lambda x: not has_outer_quotes(x))
    # # 过滤掉最外层有引号的样本，避免它们干扰后续的代码执行验证。
    # print(f"After filtering outer quotes: {len(merged_data)} samples")

    # Filter to samples that already passed
    # merged_passed = merged_data.filter(lambda x: x["pass"] == True)
    # print(f"Samples with pass=True: {len(merged_passed)} / {len(merged_data)}")

    # Filter to samples that already passed
    # merged_passed = merged_data.filter(lambda x: x["pass"] == True)
    # print(f"Samples with pass=True: {len(merged_passed)} / {len(merged_data)}")

    # Load code executor
    executor_path = Path(__file__).resolve().parents[1] / "code_excutor.py"
    spec = importlib.util.spec_from_file_location("code_excutor", executor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load code executor from: {executor_path}")
    code_executor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code_executor_module)
    ModelResponseCodeExecutor = code_executor_module.ModelResponseCodeExecutor

    executor = ModelResponseCodeExecutor(timeout=5, memory_limit_mb=2048)

    # Re-verify with code executor
    # marked_data = merged_data.map(mark_sample_passed, num_proc=32)
    # passed_count = sum(1 for s in marked_data if s["_sample_passed"])
    # print(f"Re-verified passed samples: {passed_count} / {len(marked_data)}")

    # # Keep only re-verified passing samples
    # final_data = marked_data.filter(lambda x: x["_sample_passed"])
    # final_data = final_data.remove_columns(["_sample_passed", "_error"])

    # final_data.save_to_disk("/data2/ruixin/qwen/merged_codegen_verified")
    # print(f"Saved. Total samples: {len(final_data)}")
    merged_data.push_to_hub("BigfufuOuO/codegen1_merged")
    merged_data.save_to_disk("/data2/ruixin/qwen/merged_codegen1")
    
    # 把pass列=True的样本单独保存到一个新的数据集中，供后续分析和对比使用。
    passed_data = merged_data.filter(lambda x: x["pass"] == True)
    # 1. 强烈建议先打乱数据，保证切分出的样本分布均匀
    passed_data = passed_data.shuffle(seed=42)

    total_len = len(passed_data)
    print(f"Total passed samples: {total_len}")

    # 2. 使用 tokenizer 计算 token 数量
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

    def compute_token_lengths(batch):
        def tok_len(texts):
            encoded = tokenizer(texts, add_special_tokens=False, truncation=False)
            return [len(ids) for ids in encoded["input_ids"]]

        assistant_lens = tok_len([s or "" for s in batch["assistant"]])
        user_lens = tok_len([s or "" for s in batch["user"]])
        system_lens = tok_len([s or "" for s in batch["system"]])

        total_lens = [a + u + s for a, u, s in zip(assistant_lens, user_lens, system_lens)]
        return {
            "_assistant_tokens": assistant_lens,
            "_total_tokens": total_lens,
        }

    print("Computing token lengths...")
    passed_data = passed_data.map(
        compute_token_lengths,
        batched=True,
        batch_size=512,
        num_proc=1,
        load_from_cache_file=False,
    )
    # 记录全局索引，用于后续从 passed_data 中排除 SFT 样本
    passed_data = passed_data.map(
        lambda _, idx: {"_idx": idx},
        with_indices=True,
        load_from_cache_file=False,
    )

    # 3. 按 token 数量过滤，得到 sft_select_data
    sft_select_data = passed_data.filter(
        lambda x: x["_assistant_tokens"] <= 8192 and x["_total_tokens"] <= 10000,
        load_from_cache_file=False,
    )
    print(f"SFT eligible samples (assistant<=8192, total<=10000): {len(sft_select_data)}")

    # 4. 从 sft_select_data 中选 5500 条作为 SFT
    # sft_size = min(5200, len(sft_select_data))
    sft_data = sft_select_data
    print(f"SFT data size: {len(sft_data)}")

    # 5. 为 SFT 数据划分 train 和 test (10% 作为 test)
    sft_dataset_dict = sft_data.train_test_split(test_size=0.06, seed=42)

    # 6. 从 passed_data 中去掉 SFT 已用的样本，再按 assistant<=10000 过滤得到 rl_select_data
    sft_idx_set = set(sft_data["_idx"])
    rl_select_data = passed_data.filter(
        lambda x: x["_idx"] not in sft_idx_set and x["_assistant_tokens"] <= 20000,
        load_from_cache_file=False,
    )
    print(f"RL eligible samples (excluding SFT, assistant<=20000): {len(rl_select_data)}")

    # 7. 从 rl_select_data 中选 10000 条；不足时从 sft_data 中补充
    rl_target_size = 10000
    rl_available = len(rl_select_data)

    if rl_available >= rl_target_size:
        rl_data = rl_select_data.select(range(rl_target_size))
    else:
        rl_shortage = rl_target_size - rl_available
        print(f"RL 数据不足 {rl_target_size} 条（仅 {rl_available} 条），从 SFT 中借用 {rl_shortage} 条。")
        borrow_size = min(rl_shortage, len(sft_data))
        overlap_data_for_rl = sft_data.select(range(borrow_size))
        rl_data = concatenate_datasets([rl_select_data, overlap_data_for_rl])
        print(f"最终 RL 数据集大小：{len(rl_data)} 条（含 {borrow_size} 条来自 SFT 的重合数据）")

    # 7. 为 RL 数据构造 DatasetDict
    rl_dataset_dict = DatasetDict({
        "train": rl_data
    })

    # 8. 推送到 Hugging Face，使用 config_name 来区分 Subset
    repo_id = "BigfufuOuO/codegen1_merged_clean"

    print(f"Pushing SFT subset to {repo_id}...")
    sft_dataset_dict.push_to_hub(repo_id, config_name="sft")

    # print(f"Pushing RL subset to {repo_id}...")
    # rl_dataset_dict.push_to_hub(repo_id, config_name="rl")

    print("Done! Data successfully pushed as two subsets: 'sft' and 'rl'.")