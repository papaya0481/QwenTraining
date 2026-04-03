import os
import sys
import traceback
from pathlib import Path
from typing import Optional


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _setup_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    ms_swift_root = project_root / "ms-swift"
    if ms_swift_root.exists():
        sys.path.insert(0, str(ms_swift_root))


def _run_single_stage(*, model_id: str, lora_path: Optional[str], stage_name: str) -> int:
    """Run one vLLM stage and return 0 on success, 1 on runtime error, 2 on bad input."""
    _setup_import_path()

    gpu_memory_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.3"))
    max_num_seqs = int(os.environ.get("VLLM_MAX_NUM_SEQS", "1"))
    tensor_parallel_size = int(os.environ.get("VLLM_TP_SIZE", "1"))
    pipeline_parallel_size = int(os.environ.get("VLLM_PP_SIZE", "1"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "32"))
    enforce_eager = _to_bool(os.environ.get("VLLM_ENFORCE_EAGER"), default=False)

    max_model_len_env = os.environ.get("VLLM_MAX_MODEL_LEN", "4096")
    max_model_len = int(max_model_len_env) if max_model_len_env else None

    if lora_path is not None and not Path(lora_path).exists():
        print(f"[ERROR][{stage_name}] LoRA path not found: {lora_path}")
        return 2

    try:
        import torch
        from swift.infer_engine import InferRequest, RequestConfig
        from swift.infer_engine.vllm_engine import VllmEngine

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[INFO][{stage_name}] Building VllmEngine...")
        print(f"[INFO][{stage_name}] model_id={model_id}")
        print(f"[INFO][{stage_name}] lora_path={lora_path}")
        print(
            f"[INFO][{stage_name}] vllm_args="
            f"gpu_memory_utilization={gpu_memory_util}, max_num_seqs={max_num_seqs}, "
            f"tp={tensor_parallel_size}, pp={pipeline_parallel_size}, "
            f"max_model_len={max_model_len}, enforce_eager={enforce_eager}"
        )

        engine_kwargs = dict(
            gpu_memory_utilization=gpu_memory_util,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            use_hf=True,
        )
        if lora_path is not None:
            engine_kwargs["adapters"] = [lora_path]

        engine = VllmEngine(model_id, **engine_kwargs)

        req = InferRequest(
            messages=[
                {
                    "role": "user",
                    "content": "Say hello in one short sentence.",
                }
            ]
        )
        cfg = RequestConfig(max_tokens=max_tokens, temperature=0)

        print(f"[INFO][{stage_name}] Running one inference request...")
        resp = engine.infer([req], cfg)
        text = resp[0].choices[0].message.content
        print(f"[INFO][{stage_name}] Inference succeeded.")
        print(f"[INFO][{stage_name}] output_preview={text[:200]!r}")
        return 0
    except Exception:
        print(f"[ERROR][{stage_name}] vLLM stage failed with exception:")
        traceback.print_exc()
        return 1


def run_vllm_two_stage_smoke_test() -> int:
    _setup_import_path()

    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")
    lora_path = os.environ.get(
        "LORA_PATH", "/data/wuli_error/WRX/QwenTraining/output/v37-20260403-202951/checkpoint-2"
    )

    print("[INFO] Stage-1: vLLM load base model only")
    stage1 = _run_single_stage(model_id=model_id, lora_path=None, stage_name="base_only")
    print(f"[INFO] Stage-1 result={stage1}")

    print("[INFO] Stage-2: vLLM load base model + LoRA")
    stage2 = _run_single_stage(model_id=model_id, lora_path=lora_path, stage_name="with_lora")
    print(f"[INFO] Stage-2 result={stage2}")

    if stage1 != 0:
        return 10 + stage1
    if stage2 != 0:
        return 20 + stage2
    return 0


def test_vllm_qwen35_lora_load_smoke() -> None:
    code = run_vllm_two_stage_smoke_test()
    assert code == 0, f"Two-stage VLLM smoke test failed with exit code {code}"


if __name__ == "__main__":
    raise SystemExit(run_vllm_two_stage_smoke_test())
