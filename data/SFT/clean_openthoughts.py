from datasets import load_dataset
import importlib.util
import json
from pathlib import Path

SYSTEM_PROMPT = ""
PREFIX_USER_PROMPT = """
Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output."""


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
    def _sample_passed(solution, raw_test_cases):
        solution = solution or ""
        test_samples = parse_test_cases(raw_test_cases)
        if not solution.strip() or not test_samples:
            return False

        try:
            results = executor.evaluate(
                model_response=solution,
                test_samples=test_samples,
                mode="stdio",
                use_multiprocessing=True,
                max_workers=16,
            )
        except Exception:
            return False

        return bool(results) and all(item.get("passed", False) for item in results)

    def extract_code_and_test(batch):
        solutions = batch.get("deepseek_solution", [])
        test_cases = batch.get("test_cases", [])
        return [_sample_passed(solution, raw_test_cases) for solution, raw_test_cases in zip(solutions, test_cases)]

    passed_code_domain_data = code_domain_data.filter(
        extract_code_and_test,
        batched=False,
        batch_size=16,
    )
    print("Passed samples:", len(passed_code_domain_data))
    
    # 临时保存通过测试的样本，避免后续处理过程中出现问题导致数据丢失。
    passed_code_domain_data.save_to_disk("passed_code_domain_data")