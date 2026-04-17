#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two safetensors files by keys and tensor values."
    )
    parser.add_argument("file_a", type=Path, help="Path to the first safetensors file")
    parser.add_argument("file_b", type=Path, help="Path to the second safetensors file")
    parser.add_argument(
        "--max-report",
        type=int,
        default=20,
        help="Maximum number of mismatched tensor details to print (default: 20)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Optional relative tolerance for float tensors (uses exact equality when omitted)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance used with --rtol (default: 0.0)",
    )
    return parser


def _validate_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")


def _tensors_equal(a: torch.Tensor, b: torch.Tensor, rtol: float | None, atol: float) -> bool:
    if a.dtype != b.dtype:
        return False
    if a.shape != b.shape:
        return False
    if rtol is None:
        return torch.equal(a, b)
    if a.dtype.is_floating_point or a.dtype.is_complex:
        return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    return torch.equal(a, b)


def _value_diff_summary(a: torch.Tensor, b: torch.Tensor) -> str:
    if a.shape != b.shape:
        return f"shape differs: {tuple(a.shape)} vs {tuple(b.shape)}"
    if a.dtype != b.dtype:
        return f"dtype differs: {a.dtype} vs {b.dtype}"
    if a.numel() == 0:
        return "empty tensor differs"

    if a.dtype.is_floating_point or a.dtype.is_complex:
        aa = a.to(torch.float64)
        bb = b.to(torch.float64)
        max_abs = (aa - bb).abs().max().item()
        return f"max_abs_diff={max_abs:.6e}"

    neq = (a != b).sum().item()
    return f"different_elements={int(neq)}"


def compare_safetensors(
    file_a: Path,
    file_b: Path,
    max_report: int,
    rtol: float | None,
    atol: float,
) -> Tuple[bool, List[str]]:
    data_a: Dict[str, torch.Tensor] = load_file(str(file_a))
    data_b: Dict[str, torch.Tensor] = load_file(str(file_b))

    keys_a = set(data_a.keys())
    keys_b = set(data_b.keys())

    lines: List[str] = []
    all_equal = True

    if keys_a != keys_b:
        all_equal = False
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        lines.append("[KEY MISMATCH] Keys are different.")
        lines.append(f"  only_in_a: {len(only_a)}")
        lines.append(f"  only_in_b: {len(only_b)}")
        if only_a:
            lines.append(f"  sample_only_in_a: {only_a[:min(10, len(only_a))]}")
        if only_b:
            lines.append(f"  sample_only_in_b: {only_b[:min(10, len(only_b))]}")

    common_keys = sorted(keys_a & keys_b)
    mismatch_count = 0

    for key in common_keys:
        ta = data_a[key]
        tb = data_b[key]
        if not _tensors_equal(ta, tb, rtol=rtol, atol=atol):
            all_equal = False
            mismatch_count += 1
            if mismatch_count <= max_report:
                lines.append(
                    f"[VALUE MISMATCH] key={key} | {_value_diff_summary(ta, tb)}"
                )

    if mismatch_count > max_report:
        lines.append(
            f"[VALUE MISMATCH] ... and {mismatch_count - max_report} more mismatched tensors"
        )

    lines.append(
        f"[SUMMARY] keys_a={len(keys_a)}, keys_b={len(keys_b)}, common={len(common_keys)}, value_mismatches={mismatch_count}"
    )

    return all_equal, lines


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        _validate_input(args.file_a)
        _validate_input(args.file_b)

        all_equal, report_lines = compare_safetensors(
            file_a=args.file_a,
            file_b=args.file_b,
            max_report=args.max_report,
            rtol=args.rtol,
            atol=args.atol,
        )
        for line in report_lines:
            print(line)

        if all_equal:
            print("[RESULT] EQUAL: keys and values are all equal.")
            return 0

        print("[RESULT] NOT EQUAL: key set or tensor values differ.")
        return 1
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
