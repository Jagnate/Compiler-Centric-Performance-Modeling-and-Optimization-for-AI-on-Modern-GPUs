#!/usr/bin/env python3
"""Run native plus five related-system styles on shape family 1."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Sequence

import run_final_related_system_baselines as baseline_driver


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "final_eval"
    / "related_system_baselines_shape_1"
)
DEFAULT_WORKLOAD_SUITE = REPOSITORY_ROOT / "examples" / "related_system_shape_1.json"


def build_forwarded_arguments(arguments: Sequence[str]) -> list[str]:
    defaults = [
        "--include-native",
        "--workload-suite",
        str(DEFAULT_WORKLOAD_SUITE),
        "--output-root",
        str(DEFAULT_OUTPUT_ROOT),
    ]
    return defaults + list(arguments)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = sys.argv[1:] if argv is None else list(argv)
    return baseline_driver.main(build_forwarded_arguments(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
