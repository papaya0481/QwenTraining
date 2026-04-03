from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
import importlib.util
import json
from pathlib import Path
from datasketch import MinHash, MinHashLSH
import re


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


def _sample_result(solution, raw_test_cases, mode="stdio"):
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
        mode=sample.get("fn_mode", "stdio")
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
    data_openthought = load_from_disk("/data2/ruixin/qwen/cleaned_openthoughts")
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

    merged_data = merged_data.filter(lambda x: not has_outer_quotes(x))
    # 过滤掉最外层有引号的样本，避免它们干扰后续的代码执行验证。
    print(f"After filtering outer quotes: {len(merged_data)} samples")

    # Filter to samples that already passed
    # merged_passed = merged_data.filter(lambda x: x["pass"] == True)
    # print(f"Samples with pass=True: {len(merged_passed)} / {len(merged_data)}")

    # Filter to samples that already passed
    # merged_passed = merged_data.filter(lambda x: x["pass"] == True)
    # print(f"Samples with pass=True: {len(merged_passed)} / {len(merged_data)}")

    # # Load code executor
    # executor_path = Path(__file__).resolve().parents[1] / "code_excutor.py"
    # spec = importlib.util.spec_from_file_location("code_excutor", executor_path)
    # if spec is None or spec.loader is None:
    #     raise RuntimeError(f"Failed to load code executor from: {executor_path}")
    # code_executor_module = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(code_executor_module)
    # ModelResponseCodeExecutor = code_executor_module.ModelResponseCodeExecutor

    # executor = ModelResponseCodeExecutor(timeout=5, memory_limit_mb=2048)

    # # Re-verify with code executor
    # marked_data = merged_passed.map(mark_sample_passed, num_proc=32)
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
    # passed_data.push_to_hub("BigfufuOuO/codegen1_merged_clean")
    # 分割为sft，rl两部分，分为两个subset。全部都为pass=True的样本，sft总共5000，rl为剩下的。
    # 1. 强烈建议先打乱数据，保证切分出的样本分布均匀
    passed_data = passed_data.shuffle(seed=42)
    
    total_len = len(passed_data)
    print(f"Total passed samples: {total_len}")

    # 2. 划分为 SFT (前 5000 条) 和 RL (剩余的样本)
    sft_size = min(5500, total_len) # 防止总数据量不足 5000 报错
    
    sft_data = passed_data.select(range(sft_size))
    # sft data添加: 过滤 assistant > 62000 字符的样本，避免它们干扰 SFT 训练。
    sft_data = sft_data.filter(lambda x: len(x["assistant"]) <= 62000)
    
    rl_data = passed_data.select(range(sft_size, total_len))
    unused_count = len(rl_data)
    
    # 3. 为 SFT 数据划分 train 和 test (这里默认划出 10% 作为 test，即 500 条)
    # train_test_split 会自动返回一个包含 "train" 和 "test" 的 DatasetDict
    sft_dataset_dict = sft_data.train_test_split(test_size=0.1, seed=42)
    
    # 5. 构建 10000 条的 RL 数据集
    rl_target_size = 10000
    failed_data = passed_data.filter(lambda x: x["pass"] == False)
    failed_data_count = 0

    if unused_count < rl_target_size:
        # 如果剩下没用过的数据不够，计算需要从 SFT 借多少条
        rl_shortage = rl_target_size - unused_count
        print(f"RL 数据不足 {rl_target_size} 条，需要从 SFT 中借用 {rl_shortage} 条重合数据。")
        
        # 从 SFT 数据中抽取不足的部分
        overlap_data_for_rl = sft_data.select(range(rl_shortage))
        
        # 拼接：未使用的 Pass 数据 + SFT 重合数据
        rl_data = concatenate_datasets([rl_data, overlap_data_for_rl])
        # 添加一部分failed数据来补充到12000条
        failed_to_add = failed_data.select(range(max(failed_data_count, rl_target_size - len(rl_data))))
        rl_data = concatenate_datasets([rl_data, failed_to_add])
        print(f"最终 RL 数据集大小：{len(rl_data)} 条（包含 {len(overlap_data_for_rl)} 条 SFT 重合数据和 {len(failed_to_add)} 条 failed 数据）")
    
    # 4. 为 RL 数据构造 DatasetDict (通常 RL 只需要 train split)
    rl_dataset_dict = DatasetDict({
        "train": rl_data
    })
    
    # 5. 推送到 Hugging Face，使用 config_name 来区分 Subset
    repo_id = "BigfufuOuO/codegen1_merged_clean"
    
    print(f"Pushing SFT subset to {repo_id}...")
    sft_dataset_dict.push_to_hub(repo_id, config_name="sft")
    
    print(f"Pushing RL subset to {repo_id}...")
    rl_dataset_dict.push_to_hub(repo_id, config_name="rl")
    
    print("Done! Data successfully pushed as two subsets: 'sft' and 'rl'.")