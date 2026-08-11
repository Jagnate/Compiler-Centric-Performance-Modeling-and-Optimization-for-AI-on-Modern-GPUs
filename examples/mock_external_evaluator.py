#!/usr/bin/env python3
"""Reference implementation of the subprocess evaluator JSON contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from kernel_optimization.backends.mock import MockPerformanceBackend  # noqa: E402
from kernel_optimization.schema import Candidate, TaskSpec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("model", "measure", "profile"), required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    task = TaskSpec.from_dict(request["task"])
    candidate = Candidate(**request["candidate"])
    configuration = dict(task.metadata.get("external_mock_configuration") or {})
    backend = MockPerformanceBackend(configuration)
    if args.stage == "model":
        result = backend.model(task, candidate)
    elif args.stage == "measure":
        result = backend.measure(task, candidate)
    else:
        result = backend.profile(task, candidate)
    args.response.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

