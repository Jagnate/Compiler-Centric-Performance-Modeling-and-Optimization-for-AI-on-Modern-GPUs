#!/usr/bin/env python3
"""Validate one seed source through the task's generic evaluator without an API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict

from kernel_optimization.factory import create_backend
from kernel_optimization.schema import Candidate, TaskSpec
from kernel_optimization.source_validation import SourceValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run model, measurement, and fresh final validation for one seed kernel."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Also collect one NCU profile after correctness passes.",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Skip held-out cases and robust final timing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.source).expanduser().resolve()
    task_path = Path(args.task).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit("Kernel source file does not exist: %s" % source_path)
    if not task_path.is_file():
        raise SystemExit("Task file does not exist: %s" % task_path)
    output.mkdir(parents=True, exist_ok=True)

    task = TaskSpec.from_json_file(task_path)
    source = source_path.read_text(encoding="utf-8")
    candidate = Candidate.seed(task, source, source_path.name)
    SourceValidator().validate(task, source, source_path.name)
    backend = create_backend(
        task,
        task_path.parent,
        artifact_directory=output / "evaluator_attempts",
    )

    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "task_id": task.task_id,
        "source": candidate.to_dict(include_source=False),
    }
    print("[1/3] TileSight model", flush=True)
    model = backend.model(task, candidate)
    report["model"] = model.to_dict()
    if not model.valid:
        _write_report(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    print("[2/3] Public multi-case correctness and CUDA Event timing", flush=True)
    measurement = backend.measure(task, candidate)
    report["measurement"] = measurement.to_dict()
    if not measurement.correct:
        _write_report(output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 3

    if args.profile:
        print("[optional] NCU profile", flush=True)
        report["profile"] = backend.profile(task, candidate).to_dict()

    if not args.skip_final:
        print("[3/3] Fresh held-out correctness and robust timing", flush=True)
        finalizer = getattr(backend, "finalize", None)
        if not callable(finalizer):
            raise RuntimeError("configured backend does not support final validation")
        final = finalizer(task, candidate)
        report["final"] = final.to_dict()
        if not final.correct:
            _write_report(output, report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 4

    report_path = _write_report(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("Validation report: %s" % report_path)
    return 0


def _write_report(output: Path, report: Dict[str, Any]) -> Path:
    path = output / "seed_validation.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    raise SystemExit(main())
