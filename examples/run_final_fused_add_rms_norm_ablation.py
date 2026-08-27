#!/usr/bin/env python3
"""Run the four final Fused Add + RMSNorm ablation treatments."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Sequence

import run_final_matmul_ablation as ablation_driver


DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/ccnas2/bdp/js1824/project/"
    "Compiler-Centric-Performance-Modeling-and-Optimization-for-AI-on-Modern-GPUs/"
    "results/final_eval/fused_ablation"
)


def build_forwarded_arguments(arguments: Sequence[str]) -> list[str]:
    """Inject fused-workload defaults before user-provided overrides."""

    repository_root = Path(__file__).resolve().parents[1]
    defaults = [
        "--source",
        str(
            repository_root
            / "examples"
            / "tilelang_fused_add_rms_norm_kernel.py"
        ),
        "--task",
        str(
            repository_root
            / "examples"
            / "tilelang_fused_add_rms_norm_task.json"
        ),
        "--output-root",
        str(DEFAULT_OUTPUT_ROOT),
        "--report-title",
        "Final Fused Add + RMSNorm Ablation",
    ]
    return defaults + list(arguments)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = sys.argv[1:] if argv is None else list(argv)
    return ablation_driver.main(build_forwarded_arguments(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
