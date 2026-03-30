import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class NormalizedTestCase:
    index: int
    input_data: Any
    expected_output: Any


class CodeBlockExtractor:
    """Extract executable Python code from model responses."""

    _FENCED_BLOCK_RE = re.compile(
        r"```(?:python|py|Python)?\s*\n(.*?)```", re.DOTALL
    )

    def extract(self, model_response: str) -> str:
        matches = self._FENCED_BLOCK_RE.findall(model_response)
        if matches:
            return matches[-1].strip()
        return model_response.strip()


class TestCaseNormalizer:
    """Normalize testcase payloads to a unified list format.

    Supported inputs:
    1) [{"input": ..., "output": ...}, ...]
    2) {"input": [...], "output": [...]} (also supports inputs/outputs)
    """

    def normalize(self, test_samples: Any) -> List[NormalizedTestCase]:
        if isinstance(test_samples, dict):
            return self._normalize_dict_style(test_samples)
        if isinstance(test_samples, list):
            return self._normalize_list_style(test_samples)
        raise TypeError("test_samples must be list[dict] or dict with input/output fields")

    def _normalize_dict_style(self, payload: Dict[str, Any]) -> List[NormalizedTestCase]:
        inputs = payload.get("input", payload.get("inputs"))
        outputs = payload.get("output", payload.get("outputs"))
        if inputs is None or outputs is None:
            raise ValueError("dict-style test samples must contain input(s) and output(s)")

        if not isinstance(inputs, list):
            inputs = [inputs]
        if not isinstance(outputs, list):
            outputs = [outputs]

        if len(inputs) != len(outputs):
            raise ValueError("input and output lengths do not match")

        return [
            NormalizedTestCase(
                index=i,
                input_data=self._maybe_json_decode(inp),
                expected_output=self._maybe_json_decode(out),
            )
            for i, (inp, out) in enumerate(zip(inputs, outputs))
        ]

    def _normalize_list_style(self, payload: List[Any]) -> List[NormalizedTestCase]:
        cases: List[NormalizedTestCase] = []
        for i, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError("list-style test samples must contain dict items")
            if "input" not in item and "inputs" not in item:
                raise ValueError(f"test case at index {i} missing input/inputs")
            if "output" not in item and "outputs" not in item:
                raise ValueError(f"test case at index {i} missing output/outputs")

            raw_input = item.get("input", item.get("inputs"))
            raw_output = item.get("output", item.get("outputs"))
            cases.append(
                NormalizedTestCase(
                    index=i,
                    input_data=self._maybe_json_decode(raw_input),
                    expected_output=self._maybe_json_decode(raw_output),
                )
            )
        return cases

    @staticmethod
    def _maybe_json_decode(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            return value
        likely_json_prefixes = ("{", "[", '"', "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "true", "false", "null")
        if not stripped.startswith(likely_json_prefixes):
            return value
        try:
            return json.loads(stripped)
        except Exception:
            return value


class SafePythonValidator:
    """Validate generated Python code in a restricted subprocess."""

    _WORKER_SCRIPT = textwrap.dedent(
        """
        import io
        import json
        import os
        import platform
        import signal
        import sys
        import types


        def _safe_serialize(obj):
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            if isinstance(obj, tuple):
                return [_safe_serialize(x) for x in obj]
            if isinstance(obj, list):
                return [_safe_serialize(x) for x in obj]
            if isinstance(obj, set):
                return sorted([_safe_serialize(x) for x in obj], key=lambda x: repr(x))
            if isinstance(obj, dict):
                return {str(k): _safe_serialize(v) for k, v in obj.items()}
            return repr(obj)


        def _reliability_guard(maximum_memory_bytes):
            if maximum_memory_bytes is not None:
                try:
                    import resource
                    resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
                    resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
                    # Do NOT limit RLIMIT_STACK to the full memory size — that
                    # causes the worker process to crash immediately on Linux.
                    # A 64 MB stack is more than enough for generated code.
                except Exception:
                    pass

            import builtins
            builtins.quit = None

            os.environ["OMP_NUM_THREADS"] = "1"
            os.kill = None
            os.system = None
            os.putenv = None
            os.remove = None
            os.removedirs = None
            os.rmdir = None
            os.fchdir = None
            os.setuid = None
            os.fork = None
            os.forkpty = None
            os.killpg = None
            os.rename = None
            os.renames = None
            os.truncate = None
            os.replace = None
            os.unlink = None
            os.fchmod = None
            os.fchown = None
            os.chmod = None
            os.chown = None
            os.chroot = None
            os.lchflags = None
            os.lchmod = None
            os.lchown = None
            os.getcwd = None
            os.chdir = None

            import shutil
            shutil.rmtree = None
            shutil.move = None
            shutil.chown = None

            import subprocess
            subprocess.Popen = None

            try:
                import builtins as _builtins
                _builtins.help = None
            except Exception:
                pass
            sys.modules["ipdb"] = None
            sys.modules["joblib"] = None
            sys.modules["resource"] = None
            sys.modules["psutil"] = None
            sys.modules["tkinter"] = None


        def _resolve_callable(namespace, fn_name):
            if fn_name and fn_name in namespace and callable(namespace[fn_name]):
                return namespace[fn_name]

            if fn_name and "Solution" in namespace:
                try:
                    solver = namespace["Solution"]()
                    if hasattr(solver, fn_name):
                        return getattr(solver, fn_name)
                except Exception:
                    pass

            if "solve" in namespace and callable(namespace["solve"]):
                return namespace["solve"]

            candidates = []
            for name, obj in namespace.items():
                if callable(obj) and getattr(obj, "__module__", None) == "__main__" and not name.startswith("_"):
                    lineno = getattr(getattr(obj, "__code__", None), "co_firstlineno", 10 ** 9)
                    candidates.append((lineno, name, obj))
            if not candidates:
                return None
            candidates.sort(key=lambda x: x[0])
            return candidates[0][2]


        def _run_call_based(namespace, fn_name, case_input):
            func = _resolve_callable(namespace, fn_name)
            if func is None:
                return {"ok": False, "error_code": "NO_FUNCTION", "error_message": "No callable function found"}

            try:
                if isinstance(case_input, dict):
                    result = func(**case_input)
                elif isinstance(case_input, (list, tuple)):
                    result = func(*case_input)
                else:
                    result = func(case_input)
                return {"ok": True, "actual": _safe_serialize(result)}
            except Exception as exc:
                return {"ok": False, "error_code": "RUNTIME_ERROR", "error_message": repr(exc)}


        def _run_stdio(namespace, code, case_input):
            input_text = case_input if isinstance(case_input, str) else json.dumps(case_input, ensure_ascii=False)
            fake_stdin = io.StringIO(input_text)
            fake_stdout = io.StringIO()

            old_stdin = sys.stdin
            old_stdout = sys.stdout
            try:
                sys.stdin = fake_stdin
                sys.stdout = fake_stdout
                exec(code, namespace)
                return {"ok": True, "actual": fake_stdout.getvalue()}
            except SystemExit:
                # 🎯 核心魔法在这里：
                # 如果代码调用了 exit()，会抛出 SystemExit 异常跳到这里。
                # 在竞赛题中，这通常意味着程序想提前结束并输出结果。
                # 我们认为这是正常的，直接返回之前收集到的 stdout 即可！
                return {"ok": True, "actual": fake_stdout.getvalue()}
            except Exception as exc:
                return {"ok": False, "error_code": "RUNTIME_ERROR", "error_message": repr(exc)}
            finally:
                sys.stdin = old_stdin
                sys.stdout = old_stdout


        def main():
            payload = json.loads(sys.stdin.read())
            code = payload["code"]
            mode = payload["mode"]
            fn_name = payload.get("fn_name")
            case_input = payload["case_input"]
            timeout = int(payload.get("timeout", 4))
            memory_limit_mb = int(payload.get("memory_limit_mb", 1024))

            _reliability_guard(memory_limit_mb * 1024 * 1024)

            def _timeout_handler(signum, frame):
                raise TimeoutError(f"Time limit exceeded for {timeout} seconds")

            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

            namespace = {
                "__name__": "__main__",
                "__builtins__": __builtins__,
            }

            try:
                if mode == "call_based":
                    exec(code, namespace)
                    result = _run_call_based(namespace, fn_name, case_input)
                else:
                    # Parse first so syntax issues are reported as compile errors,
                    # then execute once with redirected stdio for this testcase.
                    compile(code, "<generated>", "exec")
                    result = _run_stdio(namespace, code, case_input)
            except TimeoutError as exc:
                result = {"ok": False, "error_code": "TLE", "error_message": repr(exc)}
            except SyntaxError as exc:
                result = {"ok": False, "error_code": "COMPILE_ERROR", "error_message": repr(exc)}
            except Exception as exc:
                result = {"ok": False, "error_code": "RUNTIME_ERROR", "error_message": repr(exc)}
            except BaseException as exc: # 拦截 SystemExit, KeyboardInterrupt 等
                if isinstance(exc, SystemExit):
                    result = {"ok": False, "error_code": "RUNTIME_ERROR", "error_message": "Code attempted to exit() prematurely."}
                else:
                    result = {"ok": False, "error_code": "RUNTIME_ERROR", "error_message": f"Critical Error: {repr(exc)}"}
            finally:
                signal.alarm(0)

            sys.stdout.write(json.dumps(result, ensure_ascii=False))


        if __name__ == "__main__":
            main()
        """
    )

    def __init__(self, timeout: int = 4, memory_limit_mb: int = 1024) -> None:
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb

    def run_case(
        self,
        code: str,
        case_input: Any,
        mode: str,
        fn_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "code": code,
            "case_input": case_input,
            "mode": mode,
            "fn_name": fn_name,
            "timeout": self.timeout,
            "memory_limit_mb": self.memory_limit_mb,
        }

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", self._WORKER_SCRIPT],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=self.timeout + 1,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error_code": "TLE",
                "error_message": "Subprocess timed out",
                "exec_ms": round((time.perf_counter() - start) * 1000, 3),
            }

        exec_ms = round((time.perf_counter() - start) * 1000, 3)

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            return {
                "ok": False,
                "error_code": "RUNTIME_ERROR",
                "error_message": stderr or f"Subprocess exited with code {proc.returncode}",
                "exec_ms": exec_ms,
            }

        try:
            data = json.loads(proc.stdout.strip())
        except Exception as e:
            return {
                "ok": False,
                "error_code": "BAD_WORKER_OUTPUT",
                "error_message": f"Worker output is not valid JSON: {repr(e)}; raw output: {proc.stdout.strip()}",
                "exec_ms": exec_ms,
            }

        data["exec_ms"] = exec_ms
        return data


class ModelResponseCodeExecutor:
    """Top-level class: extract code, run tests, return per-case results."""

    def __init__(
        self,
        timeout: int = 4,
        memory_limit_mb: int = 1024,
    ) -> None:
        self.extractor = CodeBlockExtractor()
        self.normalizer = TestCaseNormalizer()
        self.validator = SafePythonValidator(timeout=timeout, memory_limit_mb=memory_limit_mb)

    def evaluate(
        self,
        model_response: str,
        test_samples: Any,
        fn_name: Optional[str] = None,
        mode: str = "auto",
    ) -> List[Dict[str, Any]]:
        code = self.extractor.extract(model_response)
        cases = self.normalizer.normalize(test_samples)
        run_mode = self._resolve_mode(code=code, fn_name=fn_name, cases=cases, mode=mode)

        worker_results = self._run_cases(
            code=code,
            cases=cases,
            run_mode=run_mode,
            fn_name=fn_name,
        )

        results: List[Dict[str, Any]] = []
        for case in cases:
            worker_result = worker_results[case.index]

            row = {
                "index": case.index,
                "input": case.input_data,
                "expected": case.expected_output,
                "mode": run_mode,
                "exec_ms": worker_result.get("exec_ms"),
            }

            if not worker_result.get("ok"):
                row.update(
                    {
                        "passed": False,
                        "actual": None,
                        "error_code": worker_result.get("error_code"),
                        "error_message": worker_result.get("error_message"),
                    }
                )
                results.append(row)
                continue

            actual = worker_result.get("actual")
            passed = self._compare(actual, case.expected_output, run_mode)
            row.update(
                {
                    "passed": passed,
                    "actual": actual,
                    "error_code": None if passed else "WRONG_ANSWER",
                    "error_message": None if passed else self._build_mismatch_reason(actual, case.expected_output, run_mode),
                }
            )
            results.append(row)

        return results

    def _run_cases(
        self,
        code: str,
        cases: Sequence[NormalizedTestCase],
        run_mode: str,
        fn_name: Optional[str],
    ) -> Dict[int, Dict[str, Any]]:
        return {
            case.index: self.validator.run_case(
                code=code,
                case_input=case.input_data,
                mode=run_mode,
                fn_name=fn_name,
            )
            for case in cases
        }

    @staticmethod
    def _resolve_mode(
        code: str,
        fn_name: Optional[str],
        cases: Sequence[NormalizedTestCase],
        mode: str,
    ) -> str:
        if mode not in {"auto", "call_based", "stdio"}:
            raise ValueError("mode must be one of: auto, call_based, stdio")
        if mode != "auto":
            return mode
        if fn_name:
            return "call_based"
        if "input(" in code or "sys.stdin" in code:
            return "stdio"
        if cases and all(isinstance(case.input_data, str) for case in cases):
            return "stdio"
        return "call_based"

    def _compare(self, actual: Any, expected: Any, mode: str) -> bool:
        if mode == "stdio":
            return self._compare_text(str(actual), str(expected))
        return self._normalize_value(actual) == self._normalize_value(expected)

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, tuple):
            return [ModelResponseCodeExecutor._normalize_value(x) for x in value]
        if isinstance(value, list):
            return [ModelResponseCodeExecutor._normalize_value(x) for x in value]
        if isinstance(value, dict):
            return {
                k: ModelResponseCodeExecutor._normalize_value(v)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            }
        return value

    @staticmethod
    def _compare_text(actual: str, expected: str) -> bool:
        actual_lines = [line.strip() for line in actual.strip().splitlines()]
        expected_lines = [line.strip() for line in expected.strip().splitlines()]

        if len(actual_lines) != len(expected_lines):
            return False

        for act_line, exp_line in zip(actual_lines, expected_lines):
            if act_line == exp_line:
                continue
            act_nums = ModelResponseCodeExecutor._parse_decimal_line(act_line)
            exp_nums = ModelResponseCodeExecutor._parse_decimal_line(exp_line)
            if act_nums is None or exp_nums is None:
                return False
            if act_nums != exp_nums:
                return False
        return True

    def _build_mismatch_reason(self, actual: Any, expected: Any, mode: str) -> str:
        if mode == "stdio":
            return self._build_text_mismatch_reason(str(actual), str(expected))

        normalized_actual = self._normalize_value(actual)
        normalized_expected = self._normalize_value(expected)
        return (
            "Output mismatch: expected "
            f"{self._truncate_repr(normalized_expected)}, "
            f"got {self._truncate_repr(normalized_actual)}"
        )

    @classmethod
    def _build_text_mismatch_reason(cls, actual: str, expected: str) -> str:
        actual_lines = [line.strip() for line in actual.strip().splitlines()]
        expected_lines = [line.strip() for line in expected.strip().splitlines()]

        if len(actual_lines) != len(expected_lines):
            return (
                "Output line count mismatch: "
                f"expected {len(expected_lines)}, got {len(actual_lines)}"
            )

        for line_idx, (act_line, exp_line) in enumerate(zip(actual_lines, expected_lines), start=1):
            if act_line == exp_line:
                continue

            act_nums = cls._parse_decimal_line(act_line)
            exp_nums = cls._parse_decimal_line(exp_line)
            if act_nums is not None and exp_nums is not None and act_nums != exp_nums:
                return (
                    f"Numeric mismatch at line {line_idx}: "
                    f"expected {cls._truncate_repr(exp_line, 80)}, "
                    f"got {cls._truncate_repr(act_line, 80)}"
                )
            return (
                f"Text mismatch at line {line_idx}: "
                f"expected {cls._truncate_repr(exp_line, 80)}, "
                f"got {cls._truncate_repr(act_line, 80)}"
            )

        return "Output does not match expected value"

    @staticmethod
    def _truncate_repr(value: Any, max_len: int = 200) -> str:
        text = repr(value)
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    @staticmethod
    def _parse_decimal_line(line: str) -> Optional[Tuple[Decimal, ...]]:
        if not line:
            return tuple()
        parts = line.split()
        decimals: List[Decimal] = []
        for part in parts:
            try:
                decimals.append(Decimal(part))
            except InvalidOperation:
                return None
        return tuple(decimals)


__all__ = [
    "CodeBlockExtractor",
    "TestCaseNormalizer",
    "SafePythonValidator",
    "ModelResponseCodeExecutor",
]


def minimal_runnable_example() -> Dict[str, List[Dict[str, Any]]]:
    """A minimal example for both testcase formats and optional multiprocessing."""
    executor = ModelResponseCodeExecutor(timeout=2, memory_limit_mb=512)

    model_response1 = """
这里是模型回答：
```python
def add(a, b):
    return a + b
```
"""

    list_style_samples = [
        {"input": [1, 2], "output": 3},
        {"input": [10, -3], "output": 7},
    ]

    dict_style_samples = {
        "input": [[4, 6], [5, 8]],
        "output": [20, 26],
    }

    list_style_results = executor.evaluate(
        model_response=model_response1,
        test_samples=list_style_samples,
        fn_name="add",
        use_multiprocessing=True,
    )

    model_response2 = """
请实现一个函数add，接受两个参数a和b，返回它们的和。
```
def add(a, b):
    return a + b
```
"""

    dict_style_results = executor.evaluate(
        model_response=model_response2,
        test_samples=dict_style_samples,
        fn_name="add",
        use_multiprocessing=True,
    )

    return {
        "list_style": list_style_results,
        "dict_style": dict_style_results,
    }

if __name__ == "__main__":
    import pprint

    results = minimal_runnable_example()
    pprint.pprint(results)