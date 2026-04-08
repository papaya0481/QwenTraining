from datasets import load_dataset, load_from_disk
import importlib.util
import json
import re
from pathlib import Path


SYSTEM_PROMPT = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. "
PREFIX_USER_PROMPT = """You will be given a competitive programming problem. Generate an executable Python function generated from the given prompt. Please reason step by step about your solution approach, then provide a complete implementation in Python 3 that is thoroughly optimized for both speed and memory usage.

Put your final solution within a single code block:
```python
<your code here>
```

### Question:
"""
FORMAT_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

HTTP_LINK_RE = re.compile(r"(?i)https?://|www\\.")
SENSITIVE_WORDS = [
    "porn",
    "nude",
    "sex",
    "gambling",
    "drugs",
    "terror",
    "terrorist",
    "bomb",
    "suicide",
    "色情",
    "赌博",
    "毒品",
    "恐怖",
    "自杀",
]

DROP_DIFFICULTIES = {"VERY_HARD", "UNKNOWN_DIFFICULTY"}
DROP_COLUMNS = [
    "name",
    "url",
    "Expected Auxiliary Space",
    "time_limit",
    "picture_num",
    "memory_limit",
    "Expected Time Complexity",
]

MAX_TEST_CASES = 10
VERIFY_TIMEOUT = 8
VERIFY_NUM_PROC = 16
VERIFY_CACHE_PATH = Path("/data2/ruixin/qwen/taco_v1_marked_cache")
FINAL_SAVE_PATH = Path("/data2/ruixin/qwen/cleaned_taco_v1")


def extract_fenced_code_blocks(text):
    if not text:
        return ""
    blocks = re.findall(r"```[\\s\\S]*?```", text)
    return "\n\n".join(blocks).strip()


def parse_json_payload(value):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return value


def parse_input_output(raw_input_output):
    payload = parse_json_payload(raw_input_output)
    if not isinstance(payload, dict):
        return None

    inputs = payload.get("inputs", payload.get("input"))
    outputs = payload.get("outputs", payload.get("output"))
    fn_name = payload.get("fn_name", payload.get("fn_nam"))

    if inputs is None or outputs is None:
        return None

    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(outputs, list):
        outputs = [outputs]

    if len(inputs) == 0 or len(outputs) == 0 or len(inputs) != len(outputs):
        return None

    return {
        "inputs": inputs,
        "outputs": outputs,
        "fn_name": fn_name if isinstance(fn_name, str) and fn_name.strip() else None,
    }


def parse_solutions(raw_solutions):
    payload = parse_json_payload(raw_solutions)

    if isinstance(payload, str):
        return [payload]

    if isinstance(payload, list):
        solutions = []
        for item in payload:
            if isinstance(item, str):
                solutions.append(item)
                continue
            if isinstance(item, dict):
                for key in ("solution", "code", "content", "text"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        solutions.append(value)
                        break
        return solutions

    if isinstance(payload, dict):
        for key in ("solution", "code", "content", "text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return [value]

    if isinstance(raw_solutions, str) and raw_solutions.strip():
        return [raw_solutions]

    return []


def contains_http_link(text):
    if not isinstance(text, str):
        return False
    return bool(HTTP_LINK_RE.search(text))


def contains_sensitive_words(text):
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(word.lower() in lowered for word in SENSITIVE_WORDS)


def trim_to_max_test_cases(parsed_input_output):
    return {
        "input": parsed_input_output["inputs"][:MAX_TEST_CASES],
        "output": parsed_input_output["outputs"][:MAX_TEST_CASES],
    }


def normalize_answer(solution):
    solution = solution or ""
    fenced = extract_fenced_code_blocks(solution)
    if fenced:
        return fenced

    stripped = solution.strip()
    if not stripped:
        return ""
    return f"```python\n{stripped}\n```"


def build_question(sample):
    question = (sample.get("question") or "").strip()
    starter_code = (sample.get("starter_code") or "").strip()

    prompt = PREFIX_USER_PROMPT + question + "\n"
    if starter_code:
        prompt += (
            f"\n### Format: {FORMAT_WITH_STARTER_CODE}\n"
            f"```python\n{starter_code}\n```\n"
        )
    return prompt.strip()


def prepare_sample(sample):
    parsed_input_output = parse_input_output(sample.get("input_output"))
    solutions = parse_solutions(sample.get("solutions"))

    fn_name = parsed_input_output["fn_name"] if parsed_input_output else None
    fn_mode = "call_based" if fn_name else "auto"

    test_cases = trim_to_max_test_cases(parsed_input_output) if parsed_input_output else None

    return {
        "fn_mode": fn_mode,
        "fn_name": fn_name or "",
        "test_cases": json.dumps(test_cases, ensure_ascii=False) if test_cases else "",
        "_solutions": solutions,
    }


def parse_test_cases(raw_test_cases):
    payload = parse_json_payload(raw_test_cases)
    if not isinstance(payload, dict):
        return None

    inputs = payload.get("inputs", payload.get("input"))
    outputs = payload.get("outputs", payload.get("output"))

    if inputs is None or outputs is None:
        return None

    return {
        "input": inputs,
        "output": outputs,
    }


def _sample_result(solution, raw_test_cases, fn_mode, fn_name):
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
            mode=fn_mode,
            fn_name=fn_name if fn_name else None,
        )
    except Exception as e:
        return False, f"executor exception: {e}"

    if not results:
        return False, "empty results"

    failed = [item for item in results if not item.get("passed", False)]
    if failed:
        reasons = {
            "pass": [item.get("passed", False) for item in results],
            "error_code": [
                item.get("error_code", "UNKNOWN") if not item.get("passed", False) else ""
                for item in results
            ],
            "error_message": [
                item.get("error_message", "Unknown Error") if not item.get("passed", False) else ""
                for item in results
            ],
        }
        return False, reasons

    return True, None


def mark_sample_passed(sample):
    solutions = sample.get("_solutions") or []
    if not isinstance(solutions, list):
        solutions = []

    all_errors = []
    for idx, solution in enumerate(solutions):
        passed, error_reason = _sample_result(
            solution=solution,
            raw_test_cases=sample.get("test_cases", ""),
            fn_mode=sample.get("fn_mode", "auto"),
            fn_name=sample.get("fn_name", ""),
        )
        if passed:
            return {
                "_sample_passed": True,
                "_error": "",
                "_selected_solution": solution,
                "_selected_solution_idx": idx,
            }
        all_errors.append(
            {
                "solution_idx": idx,
                "error": error_reason if isinstance(error_reason, str) else error_reason,
            }
        )

    fallback_solution = solutions[0] if solutions else ""
    return {
        "_sample_passed": False,
        "_error": json.dumps(all_errors, ensure_ascii=False),
        "_selected_solution": fallback_solution,
        "_selected_solution_idx": -1,
    }


def format_sample(sample):
    return {
        "system": SYSTEM_PROMPT,
        "user": build_question(sample),
        "answer": normalize_answer(sample.get("_selected_solution", "")),
        "test_cases": sample.get("test_cases", ""),
        "fn_mode": sample.get("fn_mode", "auto"),
        "fn_name": sample.get("fn_name", ""),
        "source_dataset": "TACO-verified",
        "task": "code_generation",
        "difficulty": sample.get("difficulty", "UNKNOWN_DIFFICULTY"),
        "tags": sample.get("tags", []),
        "skill_types": sample.get("skill_types", []),
    }


if __name__ == "__main__":
    data = load_dataset("likaixin/TACO-verified", split="train")
    print(data)
    print("Columns:", data.column_names)

    # 1) 过滤没有 input_output 的样本
    data = data.filter(lambda x: parse_input_output(x.get("input_output")) is not None)
    print("After input_output filter:", len(data))

    # 2) 过滤 question 中含 http(s) 链接
    data = data.filter(lambda x: not contains_http_link(x.get("question", "")))
    print("After http-link filter:", len(data))

    # 3) 过滤 question 中含敏感词
    # data = data.filter(lambda x: not contains_sensitive_words(x.get("question", "")))
    # print("After sensitive-word filter:", len(data))

    # 5) difficulty 过滤
    if "difficulty" in data.column_names:
        data = data.filter(lambda x: x.get("difficulty") not in DROP_DIFFICULTIES)
        print("After difficulty filter:", len(data))

    # 6) 去掉不需要列（如果存在）
    removable_columns = [column for column in DROP_COLUMNS if column in data.column_names]
    if removable_columns:
        data = data.remove_columns(removable_columns)
        print("Dropped columns:", removable_columns)

    # 4) 裁剪测试样本到 10，新增 fn_mode/fn_name，并保留全部 solutions
    data = data.map(prepare_sample)
    data = data.filter(lambda x: len(x.get("_solutions") or []) > 0 and bool(x.get("test_cases")))
    print("After solution/test_cases filter:", len(data))

    # 动态加载本地代码执行器
    executor_path = Path(__file__).resolve().parents[1] / "code_excutor.py"
    spec = importlib.util.spec_from_file_location("code_excutor", executor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load code executor from: {executor_path}")
    code_executor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code_executor_module)
    ModelResponseCodeExecutor = code_executor_module.ModelResponseCodeExecutor

    executor = ModelResponseCodeExecutor(timeout=VERIFY_TIMEOUT, memory_limit_mb=2048)

    # 按缓存优先的方式进行验证，避免重复跑
    if VERIFY_CACHE_PATH.exists():
        print(f"Loading verification cache from: {VERIFY_CACHE_PATH}")
        marked_data = load_from_disk(str(VERIFY_CACHE_PATH))
        required_columns = {"_sample_passed", "_selected_solution", "_selected_solution_idx", "_error"}
        cache_ok = required_columns.issubset(set(marked_data.column_names)) and len(marked_data) == len(data)
        if not cache_ok:
            print("Verification cache is stale for all-solutions mode, rebuilding cache...")
            marked_data = data.map(mark_sample_passed, num_proc=VERIFY_NUM_PROC)
            marked_data.save_to_disk(str(VERIFY_CACHE_PATH))
            print(f"Refreshed verification cache at: {VERIFY_CACHE_PATH}")
    else:
        marked_data = data.map(mark_sample_passed, num_proc=VERIFY_NUM_PROC)
        marked_data.save_to_disk(str(VERIFY_CACHE_PATH))
        print(f"Saved verification cache to: {VERIFY_CACHE_PATH}")

    passed_data = marked_data.filter(lambda x: x.get("_sample_passed", False))
    print(f"Passed samples: {len(passed_data)} / {len(marked_data)}")

    # 7) 最终重排字段
    final_data = marked_data.map(
        format_sample,
        remove_columns=marked_data.column_names,
    )
    final_data.save_to_disk(str(FINAL_SAVE_PATH))
    print("Final cleaned data saved to:", FINAL_SAVE_PATH)
    print("Total samples:", len(final_data))
    print(final_data[0])
    final_data.push_to_hub("BigfufuOuO/taco_verified")