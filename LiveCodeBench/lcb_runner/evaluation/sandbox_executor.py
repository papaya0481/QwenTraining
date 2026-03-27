import asyncio
import json
import sys
sys.path.insert(0, '/data/wuli_error/WRX/QwenTraining/ms-enclave')

from ms_enclave.sandbox.manager import LocalSandboxManager
from ms_enclave.sandbox.model import DockerSandboxConfig, SandboxType


_manager = None
_pool_initialized = False
_use_dummy = False


async def get_manager():
    """Get or create sandbox manager pool. If Docker fails, fallback to local dummy mode."""
    global _manager, _pool_initialized, _use_dummy

    if _manager is None:
        _manager = LocalSandboxManager()
        await _manager.__aenter__()

    if not _pool_initialized:
        try:
            config = DockerSandboxConfig(
                image='python:3.11-slim',
                tools_config={'python_executor': {}}
            )
            await _manager.initialize_pool(pool_size=4, sandbox_type=SandboxType.DOCKER, config=config)
            _use_dummy = False
        except Exception as e:
            # ms-enclave in this workspace does not provide a registered DUMMY sandbox.
            # Mark fallback mode and execute via local testing_util path instead.
            print(f"Docker sandbox failed, falling back to local DUMMY mode: {e}")
            _use_dummy = True
        _pool_initialized = True

    return _manager


async def execute_in_sandbox(code: str, timeout: int = 6):
    """Execute code in sandbox pool"""
    manager = await get_manager()
    result = await manager.execute_tool_in_pool('python_executor', {'code': code}, timeout=timeout)
    return result.output, result.exit_code


def run_test_sandbox(sample, test, timeout=6):
    """Execute test in sandbox"""
    global _use_dummy

    # Initialize sandbox backend once so we can decide docker vs local dummy path.
    if not _pool_initialized:
        asyncio.run(get_manager())

    # Local dummy fallback: delegate to non-sandbox path, which applies reliability_guard.
    if _use_dummy:
        from lcb_runner.evaluation.testing_util import run_test
        return run_test(sample, test=test, debug=False, timeout=timeout, use_sandbox=False)

    try:
        in_outs = json.loads(sample["input_output"])

        # Determine execution type
        if in_outs.get("fn_name"):
            # Call-based execution
            return _run_call_based_sandbox(test, in_outs, timeout)
        else:
            # Standard input execution
            return _run_stdio_sandbox(test, in_outs, timeout)
    except Exception as e:
        return [-4], {"error": str(e), "error_code": -4, "error_message": "Execution Error"}


def _run_call_based_sandbox(code: str, in_outs: dict, timeout: int):
    """Run call-based tests in sandbox"""
    fn_name = in_outs["fn_name"]
    all_results = []

    for gt_inp, gt_out in zip(in_outs["inputs"], in_outs["outputs"]):
        inputs = [json.loads(line) for line in gt_inp.split("\n")]
        expected = json.loads(gt_out)

        exec_code = f"""
{code}

# Execute function
result = {fn_name}(*{inputs})
print(repr(result))
"""
        try:
            output, exit_code = asyncio.run(execute_in_sandbox(exec_code, timeout))
            if exit_code != 0:
                return all_results + [-4], {"error_code": -4, "error_message": "Runtime Error"}

            actual = eval(output.strip())
            if actual == expected:
                all_results.append(True)
            else:
                return all_results + [-2], {
                    "output": str(actual), "expected": str(expected),
                    "error_code": -2, "error_message": "Wrong Answer"
                }
        except Exception as e:
            return all_results + [-4], {"error": str(e), "error_code": -4, "error_message": "Runtime Error"}

    return all_results, {"execution time": 0}


def _run_stdio_sandbox(code: str, in_outs: dict, timeout: int):
    """Run stdio tests in sandbox"""
    all_results = []

    for gt_inp, gt_out in zip(in_outs["inputs"], in_outs["outputs"]):
        exec_code = f"""
import sys
from io import StringIO
sys.stdin = StringIO({repr(gt_inp)})

{code}
"""
        try:
            output, exit_code = asyncio.run(execute_in_sandbox(exec_code, timeout))
            if exit_code != 0:
                return all_results + [-4], {"error_code": -4, "error_message": "Runtime Error"}

            if output.strip() == gt_out.strip():
                all_results.append(True)
            else:
                return all_results + [-2], {
                    "output": output.strip(), "expected": gt_out.strip(),
                    "error_code": -2, "error_message": "Wrong Answer"
                }
        except Exception as e:
            return all_results + [-4], {"error": str(e), "error_code": -4, "error_message": "Runtime Error"}

    return all_results, {"execution time": 0}
