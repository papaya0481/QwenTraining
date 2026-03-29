from datasets import load_dataset
import importlib.util
import json
from pathlib import Path

SYSTEM_PROMPT = ""
PREFIX_USER_PROMPT = """
Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output."""


def evaluate_code_domain_data(code_domain_data, executor, sample_limit=100, max_workers=16):
    # 对 deepseek_solution中的代码，提取出代码块，并使用test_cases中的输入输出进行测试。
    # 返回是否通过以及每个测试样例的错误信息（通过则为空字符串）。
    def evaluate_for_map(sample):
        solution = (sample.get("deepseek_solution", "") or "")
        raw_test_cases = sample.get("test_cases", "")

        test_samples = None
        if raw_test_cases is None:
            test_samples = None
        else:
            data = raw_test_cases
            if isinstance(raw_test_cases, str):
                try:
                    data = json.loads(raw_test_cases)
                except Exception:
                    data = None

            if isinstance(data, dict):
                # OpenThoughts code 样本常见格式: {"inputs": [...], "outputs": [...]}。
                if "inputs" in data and "outputs" in data:
                    test_samples = {"input": data["inputs"], "output": data["outputs"]}
                elif "input" in data and "output" in data:
                    test_samples = data
            elif isinstance(data, list):
                test_samples = data

        if test_samples is None:
            return {
                "_is_passed": False,
                "test_case_errors": ["Invalid or missing test_cases"],
            }

        if isinstance(test_samples, dict):
            inputs = test_samples.get("input", [])
            if not isinstance(inputs, list):
                inputs = [inputs]
            case_count = len(inputs)
        elif isinstance(test_samples, list):
            case_count = len(test_samples)
        else:
            case_count = 0

        if case_count <= 0:
            return {
                "_is_passed": False,
                "test_case_errors": ["No executable test cases"],
            }

        if not solution.strip():
            return {
                "_is_passed": False,
                "test_case_errors": ["Empty solution"] * case_count,
            }

        try:
            results = executor.evaluate(
                model_response=solution,
                test_samples=test_samples,
                mode="stdio",
                use_multiprocessing=True,
                max_workers=max_workers,
            )
        except Exception as exc:
            return {
                "_is_passed": False,
                "test_case_errors": [repr(exc)] * case_count,
            }

        if not results:
            return {
                "_is_passed": False,
                "test_case_errors": ["Executor returned no test results"],
            }

        test_case_errors = []
        for item in results:
            if item.get("passed", False):
                test_case_errors.append("")
                continue

            error_message = (item.get("error_message") or "").strip()
            if not error_message:
                error_code = item.get("error_code")
                if error_code:
                    error_message = f"Execution failed ({error_code})"
                else:
                    error_message = "Execution failed without detailed error"
            test_case_errors.append(error_message)

        is_passed = all(item.get("passed", False) for item in results)
        if not is_passed and all(not err for err in test_case_errors):
            test_case_errors = ["Failed but no detailed error message"] * len(results)

        return {
            "_is_passed": is_passed,
            "test_case_errors": test_case_errors,
        }

    # 先选择前 sample_limit 个样本进行测试，避免一次性处理过多数据导致资源耗尽。
    test_code_domain_data = code_domain_data.select(range(min(sample_limit, len(code_domain_data))))
    evaluated_code_domain_data = test_code_domain_data.map(evaluate_for_map)

    passed_code_domain_data = evaluated_code_domain_data.filter(lambda sample: sample["_is_passed"])
    failed_code_domain_data = evaluated_code_domain_data.filter(lambda sample: not sample["_is_passed"])
    return passed_code_domain_data, failed_code_domain_data


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

    # 动态加载本地代码执行器，避免依赖路径差异导致的 import 问题。
    executor_path = Path(__file__).resolve().parents[1] / "code_excutor.py"
    spec = importlib.util.spec_from_file_location("code_excutor", executor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load code executor from: {executor_path}")
    code_executor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code_executor_module)
    ModelResponseCodeExecutor = code_executor_module.ModelResponseCodeExecutor

    executor = ModelResponseCodeExecutor(timeout=4, memory_limit_mb=512)

    passed_code_domain_data, failed_code_domain_data = evaluate_code_domain_data(
        code_domain_data=code_domain_data,
        executor=executor,
        sample_limit=100,
        max_workers=16,
    )

    print("Passed samples:", len(passed_code_domain_data))
    print("Failed samples:", len(failed_code_domain_data))

    failed_output_path = Path(__file__).resolve().with_name("failed_code_domain_data.json")
    failed_code_domain_data = failed_code_domain_data.remove_columns(["_is_passed"])
    failed_code_domain_data.to_json(
        str(failed_output_path),
        orient="records",
        lines=False,
        force_ascii=False,
    )
    print("Failed samples saved to:", failed_output_path)
    
    # 临时保存通过测试的样本，避免后续处理过程中出现问题导致数据丢失。
    # passed_code_domain_data.save_to_disk("passed_code_domain_data")