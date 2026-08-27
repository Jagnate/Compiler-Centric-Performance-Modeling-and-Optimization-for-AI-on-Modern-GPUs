#!/usr/bin/env python3
"""Run four final ablation treatments, using Matmul defaults."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence


DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/ccnas2/bdp/js1824/project/"
    "Compiler-Centric-Performance-Modeling-and-Optimization-for-AI-on-Modern-GPUs/"
    "results/final_eval"
)


@dataclass(frozen=True)
class Treatment:
    directory: str
    title: str
    description: str
    evaluation_policy: str
    tir_evidence_policy: str
    strategy_allocation_policy: str


TREATMENTS = (
    Treatment(
        directory="01_full_system",
        title="Full system",
        description="TileSight and visible TIR evidence with AI-planned search slots.",
        evaluation_policy="tilesight",
        tir_evidence_policy="visible",
        strategy_allocation_policy="ai-planned",
    ),
    Treatment(
        directory="02_no_tir",
        title="No TIR evidence",
        description=(
            "TileSight still ranks candidates, but TIR and TileSight evidence are "
            "hidden from the AI."
        ),
        evaluation_policy="tilesight",
        tir_evidence_policy="hidden",
        strategy_allocation_policy="ai-planned",
    ),
    Treatment(
        directory="03_ncu_only",
        title="NCU without TileSight",
        description=(
            "TileSight is disabled; every correct candidate is timed with CUDA Event "
            "and profiled with NCU."
        ),
        evaluation_policy="ncu",
        tir_evidence_policy="auto",
        strategy_allocation_policy="ai-planned",
    ),
    Treatment(
        directory="04_ai_unconstrained",
        title="AI-directed search",
        description=(
            "The AI chooses all search directions without controller-assigned strategy "
            "slots; common correctness and AST validity gates remain enabled."
        ),
        evaluation_policy="tilesight",
        tir_evidence_policy="visible",
        strategy_allocation_policy="unconstrained",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the four final Matmul ablation experiments."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=repository_root / "examples" / "tilelang_matmul_kernel.py",
    )
    parser.add_argument(
        "--task",
        type=Path,
        default=repository_root / "examples" / "tilelang_matmul_task.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that receives one subdirectory per treatment.",
    )
    parser.add_argument(
        "--report-title",
        default="Final Matmul Ablation",
        help="Heading used by the aggregate Markdown report.",
    )
    parser.add_argument(
        "--agent-workers",
        type=int,
        default=int(os.environ.get("KERNEL_OPT_AGENT_WORKERS", "1")),
    )
    parser.add_argument(
        "--incumbent-snapshot-interval-seconds",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("KERNEL_OPT_API_URL"),
    )
    parser.add_argument(
        "--api-model",
        default=os.environ.get("KERNEL_OPT_API_MODEL"),
    )
    parser.add_argument(
        "--api-key-env",
        default="KERNEL_OPT_API_KEY",
    )
    parser.add_argument(
        "--only",
        choices=tuple(item.directory for item in TREATMENTS),
        action="append",
        help="Run only the named treatment; repeat this flag to select several.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without creating output or calling the API/GPU.",
    )
    return parser


def build_command(
    *,
    treatment: Treatment,
    source: Path,
    task: Path,
    output: Path,
    agent_workers: int,
    snapshot_interval_seconds: float,
    api_url: str,
    api_model: str,
    api_key_env: str,
    resume: bool = False,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "kernel_optimization.cli",
        "--source",
        str(source),
        "--task",
        str(task),
        "--output",
        str(output),
        "--api-url",
        api_url,
        "--api-model",
        api_model,
        "--api-key-env",
        api_key_env,
        "--agent-workers",
        str(agent_workers),
        "--incumbent-snapshot-interval-seconds",
        str(snapshot_interval_seconds),
        "--selection-policy",
        "adaptive",
        "--profile-policy",
        "milestone",
        "--structural-search-policy",
        "enforce",
        "--evaluation-policy",
        treatment.evaluation_policy,
        "--tir-evidence-policy",
        treatment.tir_evidence_policy,
        "--strategy-allocation-policy",
        treatment.strategy_allocation_policy,
    ]
    if resume:
        command.append("--resume")
    return command


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(__file__).resolve().parents[1]
    source = args.source.expanduser().resolve()
    task = args.task.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not source.is_file():
        raise SystemExit("Kernel source does not exist: %s" % source)
    if not task.is_file():
        raise SystemExit("Kernel task does not exist: %s" % task)
    if args.agent_workers <= 0:
        raise SystemExit("--agent-workers must be positive")
    if args.incumbent_snapshot_interval_seconds < 0:
        raise SystemExit(
            "--incumbent-snapshot-interval-seconds must be non-negative"
        )
    if not args.api_url or not args.api_model:
        raise SystemExit(
            "Set KERNEL_OPT_API_URL and KERNEL_OPT_API_MODEL, or pass "
            "--api-url and --api-model."
        )
    if not args.dry_run and not os.environ.get(args.api_key_env):
        raise SystemExit(
            "API key environment variable %s is not set" % args.api_key_env
        )

    selected = [
        item
        for item in TREATMENTS
        if args.only is None or item.directory in set(args.only)
    ]
    environment = dict(os.environ)
    source_root = repository_root / "src"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if not existing_pythonpath
        else str(source_root) + os.pathsep + existing_pythonpath
    )

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for index, treatment in enumerate(selected, start=1):
        output = output_root / treatment.directory
        summary_path = output / "summary.json"
        if summary_path.is_file():
            print(
                "\n[%d/%d] %s is already complete; reusing %s"
                % (index, len(selected), treatment.title, summary_path),
                flush=True,
            )
            rows.append(_summary_row(treatment, _read_json(summary_path), "reused"))
            continue

        resume = output.is_dir() and any(output.iterdir())
        command = build_command(
            treatment=treatment,
            source=source,
            task=task,
            output=output,
            agent_workers=args.agent_workers,
            snapshot_interval_seconds=args.incumbent_snapshot_interval_seconds,
            api_url=args.api_url,
            api_model=args.api_model,
            api_key_env=args.api_key_env,
            resume=resume,
        )
        print(
            "\n[%d/%d] %s%s\n%s"
            % (
                index,
                len(selected),
                treatment.title,
                " (resume)" if resume else "",
                shlex.join(command),
            ),
            flush=True,
        )
        if args.dry_run:
            rows.append(_summary_row(treatment, {}, "dry-run"))
            continue

        completed = subprocess.run(command, cwd=repository_root, env=environment)
        if completed.returncode != 0:
            rows.append(
                _summary_row(
                    treatment,
                    {"error": "exit code %d" % completed.returncode},
                    "failed",
                )
            )
            _write_reports(
                output_root, source, task, rows, args.report_title
            )
            raise SystemExit(
                "%s failed with exit code %d; rerun this script to resume it."
                % (treatment.title, completed.returncode)
            )
        rows.append(_summary_row(treatment, _read_json(summary_path), "completed"))
        _write_reports(output_root, source, task, rows, args.report_title)

    if args.dry_run:
        return 0
    _write_reports(output_root, source, task, rows, args.report_title)
    print("\nAll selected treatments completed.", flush=True)
    print("Summary: %s" % (output_root / "ablation_summary.md"), flush=True)
    return 0


def _summary_row(
    treatment: Treatment,
    summary: Mapping[str, Any],
    status: str,
) -> Dict[str, Any]:
    return {
        "directory": treatment.directory,
        "title": treatment.title,
        "description": treatment.description,
        "status": status,
        "evaluation_policy": treatment.evaluation_policy,
        "tir_evidence_policy": treatment.tir_evidence_policy,
        "strategy_allocation_policy": treatment.strategy_allocation_policy,
        "seed_latency_ms": summary.get("seed_latency_ms"),
        "best_latency_ms": summary.get("best_latency_ms"),
        "speedup_over_seed": summary.get("speedup_over_seed"),
        "generated_candidates": summary.get("generated_candidates"),
        "measured_candidates": summary.get("measured_candidates"),
        "profile_calls": summary.get("profile_calls"),
        "error": summary.get("error"),
    }


def _write_reports(
    output_root: Path,
    source: Path,
    task: Path,
    rows: Sequence[Mapping[str, Any]],
    report_title: str = "Final Matmul Ablation",
) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": str(source),
        "task": str(task),
        "output_root": str(output_root),
        "report_title": report_title,
        "treatments": list(rows),
    }
    (output_root / "ablation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# %s" % report_title,
        "",
        "Source: `%s`" % source,
        "",
        "Task: `%s`" % task,
        "",
        "| Treatment | Status | Evaluator | TIR evidence | Search allocation | Seed (ms) | Best (ms) | Speedup | Measured | NCU |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {title} | {status} | {evaluation_policy} | {tir_evidence_policy} | "
            "{strategy_allocation_policy} | {seed_latency_ms} | {best_latency_ms} | "
            "{speedup_over_seed} | {measured_candidates} | {profile_calls} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "The NCU treatment ranks candidates by clean CUDA Event latency; NCU is "
            "used for feedback on every correctness-passing candidate.",
            "",
            "The AI-directed treatment removes controller-assigned strategy slots but "
            "keeps the common correctness, source, and AST validity gates.",
            "",
        ]
    )
    (output_root / "ablation_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
