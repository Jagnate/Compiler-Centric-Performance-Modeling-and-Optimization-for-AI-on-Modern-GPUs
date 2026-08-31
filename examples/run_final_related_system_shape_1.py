#!/usr/bin/env python3
"""Compatibility wrapper for the unified shape-family runner."""

from __future__ import annotations

import sys
from typing import Optional, Sequence

import run_final_related_system_baselines as baseline_driver


DEFAULT_SHAPE_CONFIG = "special"


def build_forwarded_arguments(arguments: Sequence[str]) -> list[str]:
    has_shape_selection = any(
        argument in ("--shape-config", "--workload-suite")
        or argument.startswith("--shape-config=")
        or argument.startswith("--workload-suite=")
        for argument in arguments
    )
    defaults = [] if has_shape_selection else ["--shape-config", DEFAULT_SHAPE_CONFIG]
    return defaults + list(arguments)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = sys.argv[1:] if argv is None else list(argv)
    return baseline_driver.main(build_forwarded_arguments(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
