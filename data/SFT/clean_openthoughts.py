from datasets import load_dataset
import importlib.util
import json
from pathlib import Path

SYSTEM_PROMPT = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. "
PREFIX_USER_PROMPT = """
Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output."""

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
        # OpenThoughts code 样本常见格式: {"inputs": [...], "outputs": [...]}。
        if "inputs" in data and "outputs" in data:
            return {"input": data["inputs"], "output": data["outputs"]}
        if "input" in data and "output" in data:
            return data
        return None

    if isinstance(data, list):
        return data

    return None

# 对 deepseek_solution中的代码，提取出代码块，并使用test_cases中的输入输出进行测试，保留测试通过的样本。
def _sample_result(solution, raw_test_cases):
    """返回 (passed: bool, error_reason: str | None)"""
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
            mode="auto",
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
        sample.get("deepseek_solution", ""),
        sample.get("test_cases"),
    )
    return {
        "_sample_passed": passed,
        "_error": "" if error_reason is None else (error_reason if isinstance(error_reason, str) else json.dumps(error_reason)),
    }


if __name__ == "__main__":
    dataset = load_dataset("open-thoughts/OpenThoughts-114k", 
                           name="metadata", 
                           split="train")
    print(dataset)
    # 打印所有的键
    print(dataset.column_names)
    
    # 选择 domain 为 "code" 的数据
    code_domain_data = dataset.filter(lambda x: x["domain"] == "code")
    # 选择 deepseek_reasoning 与 deepseek_solution 长度之和 < 64000 个字符的数据
    code_domain_data = code_domain_data.filter(lambda x: len(x["deepseek_reasoning"] + x["deepseek_solution"]) < 64000)     # 剩余 17946
    print(code_domain_data)
    
    # 添加：对test_cases裁剪到只有10个
    def trim_test_cases(sample):
        test_cases = parse_test_cases(sample.get("test_cases"))
        if not test_cases:
            return sample
        if isinstance(test_cases, dict) and "input" in test_cases and "output" in test_cases:
            trimmed_test_cases = {
                "input": test_cases["input"][:10],
                "output": test_cases["output"][:10]
            }
            sample["test_cases"] = json.dumps(trimmed_test_cases)
        elif isinstance(test_cases, list):
            sample["test_cases"] = json.dumps(test_cases[:10])
        return sample

    # 动态加载本地代码执行器，避免依赖路径差异导致的 import 问题。
    executor_path = Path(__file__).resolve().parents[1] / "code_excutor.py"
    spec = importlib.util.spec_from_file_location("code_excutor", executor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load code executor from: {executor_path}")
    code_executor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code_executor_module)
    ModelResponseCodeExecutor = code_executor_module.ModelResponseCodeExecutor

    executor = ModelResponseCodeExecutor(timeout=10, memory_limit_mb=2048)

    trimmed_code_domain_data = code_domain_data.map(trim_test_cases)
    marked_code_domain_data = trimmed_code_domain_data.map(mark_sample_passed, num_proc=32)
    passed_count = sum(1 for s in marked_code_domain_data if s["_sample_passed"])
    print(f"Passed samples: {passed_count} / {len(marked_code_domain_data)}")

    # 开始排布样本，构建最终的训练数据格式。
    def format_sample(sample):
        if not sample["deepseek_reasoning"].strip().startswith("<think>") and not sample["deepseek_reasoning"].strip().endswith("</think>"):
            wrapped_reasoning = f"<think>\n{sample['deepseek_reasoning']}\n</think>\n"
        else:
            wrapped_reasoning = sample["deepseek_reasoning"]
        passed = sample["_sample_passed"]
        return {
            "system": SYSTEM_PROMPT,
            "user": PREFIX_USER_PROMPT + "\n" + sample["problem"],
            "assistant": wrapped_reasoning + "\n" + sample["deepseek_solution"],
            "test_cases": sample["test_cases"],
            "source_dataset": "OpenThoughts-114k",
            "task": "code_generation",
            "pass": passed,
            "errors": "" if passed else sample["_error"],
        }

    # 保存最终的训练数据（包含所有样本，pass列标记是否通过）
    final_data = marked_code_domain_data.map(format_sample, num_proc=16)
    final_data = final_data.remove_columns(["_sample_passed", "_error"])
    final_data = final_data.remove_columns(["problem", "deepseek_reasoning", "deepseek_solution", "ground_truth_solution", "domain", "source", "starter_code"])
    # final_data.save_to_disk("/data2/ruixin/cleaned_openthoughts")
    import datasets
    final_data.push_to_hub("BigfufuOuO/code_gen")
    print("Final cleaned data saved. Total samples:", len(final_data))