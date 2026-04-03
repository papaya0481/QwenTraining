import os
import sys
import traceback
from pathlib import Path


def _to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _setup_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    ms_swift_root = project_root / "ms-swift"
    if ms_swift_root.exists():
        sys.path.insert(0, str(ms_swift_root))


def run_vllm_lora_smoke_test() -> int:
    _setup_import_path()

    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")
    lora_path = os.environ.get(
        "LORA_PATH", "/data/wuli_error/WRX/QwenTraining/output/v37-20260403-202951/checkpoint-2"
    )

    gpu_memory_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.3"))
    max_num_seqs = int(os.environ.get("VLLM_MAX_NUM_SEQS", "1"))
    tensor_parallel_size = int(os.environ.get("VLLM_TP_SIZE", "1"))
    pipeline_parallel_size = int(os.environ.get("VLLM_PP_SIZE", "1"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "32"))
    enforce_eager = _to_bool(os.environ.get("VLLM_ENFORCE_EAGER"), default=False)

    max_model_len_env = os.environ.get("VLLM_MAX_MODEL_LEN")
    max_model_len = int(max_model_len_env) if max_model_len_env else None

    if not Path(lora_path).exists():
        print(f"[ERROR] LoRA path not found: {lora_path}")
        return 2

    try:
        import torch
        from swift.infer_engine import InferRequest, RequestConfig
        from swift.infer_engine.vllm_engine import VllmEngine

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[INFO] Building VllmEngine...")
        print(f"[INFO] model_id={model_id}")
        print(f"[INFO] lora_path={lora_path}")
        print(
            "[INFO] vllm_args="
            f"gpu_memory_utilization={gpu_memory_util}, max_num_seqs={max_num_seqs}, "
            f"tp={tensor_parallel_size}, pp={pipeline_parallel_size}, "
            f"max_model_len={max_model_len}, enforce_eager={enforce_eager}"
        )

        engine = VllmEngine(
            model_id,
            adapters=[lora_path],
            gpu_memory_utilization=gpu_memory_util,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
        )

        req = InferRequest(
            messages=[
                {
                    "role": "user",
                    "content": "Say hello in one short sentence.",
                }
            ]
        )
        cfg = RequestConfig(max_tokens=max_tokens, temperature=0)

        print("[INFO] Running one inference request...")
        resp = engine.infer([req], cfg)
        text = resp[0].choices[0].message.content
        print("[INFO] Inference succeeded.")
        print(f"[INFO] output_preview={text[:200]!r}")
        return 0
    except Exception:
        print("[ERROR] VLLM+LoRA smoke test failed with exception:")
        traceback.print_exc()
        return 1


def test_vllm_qwen35_lora_load_smoke() -> None:
    code = run_vllm_lora_smoke_test()
    assert code == 0, f"VLLM+LoRA smoke test failed with exit code {code}"


if __name__ == "__main__":
    raise SystemExit(run_vllm_lora_smoke_test())
