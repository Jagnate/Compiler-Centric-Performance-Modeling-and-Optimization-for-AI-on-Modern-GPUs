#!/usr/bin/env python3
"""Run equal-budget search-policy experiments and aggregate their summaries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_POLICIES = ("adaptive", "model-top", "random")
VALID_POLICIES = DEFAULT_POLICIES + ("measure-all",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run comparable kernel optimization policy experiments."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help="Comma-separated promotion policies.",
    )
    parser.add_argument(
        "--promotions-per-round",
        type=int,
        help="Equal CUDA measurement budget; defaults to task max promotions.",
    )
    parser.add_argument(
        "--profile-policy",
        choices=("milestone", "every-round", "none"),
        default="every-round",
    )
    parser.add_argument(
        "--strategy-allocation-policy",
        choices=("ai-planned", "fixed", "unconstrained"),
        default="ai-planned",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--api-url")
    parser.add_argument("--api-model")
    parser.add_argument("--api-key-env", default="KERNEL_OPT_API_KEY")
    parser.add_argument("--api-temperature", type=float)
    parser.add_argument("--agent-workers", type=int, default=1)
    parser.add_argument(
        "--incumbent-snapshot-interval-seconds",
        type=float,
        default=300.0,
    )
    parser.add_argument("--api-input-price-per-million", type=float)
    parser.add_argument("--api-output-price-per-million", type=float)
    return parser


def build_run_command(
    *,
    source: Path,
    task: Path,
    output: Path,
    policy: str,
    promotions_per_round: int,
    profile_policy: str,
    strategy_allocation_policy: str = "ai-planned",
    api_url: Optional[str] = None,
    api_model: Optional[str] = None,
    api_key_env: str = "KERNEL_OPT_API_KEY",
    api_temperature: Optional[float] = None,
    agent_workers: int = 1,
    incumbent_snapshot_interval_seconds: float = 300.0,
    input_price: Optional[float] = None,
    output_price: Optional[float] = None,
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
        "--selection-policy",
        policy,
        "--promotions-per-round",
        str(promotions_per_round),
        "--profile-policy",
        profile_policy,
        "--strategy-allocation-policy",
        strategy_allocation_policy,
        "--api-key-env",
        api_key_env,
        "--agent-workers",
        str(agent_workers),
        "--incumbent-snapshot-interval-seconds",
        str(incumbent_snapshot_interval_seconds),
    ]
    for flag, value in (
        ("--api-url", api_url),
        ("--api-model", api_model),
        ("--api-temperature", api_temperature),
        ("--api-input-price-per-million", input_price),
        ("--api-output-price-per-million", output_price),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    if resume:
        command.append("--resume")
    return command


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source).expanduser().resolve()
    task_path = Path(args.task).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not source.is_file() or not task_path.is_file():
        raise SystemExit("--source and --task must name existing files")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    promotions = args.promotions_per_round
    if promotions is None:
        promotions = int(
            dict(task.get("budget") or {}).get("max_promotions_per_round", 4)
        )
    if promotions <= 0:
        raise SystemExit("--promotions-per-round must be positive")
    if args.agent_workers <= 0:
        raise SystemExit("--agent-workers must be positive")
    if (
        not math.isfinite(args.incumbent_snapshot_interval_seconds)
        or args.incumbent_snapshot_interval_seconds < 0
    ):
        raise SystemExit(
            "--incumbent-snapshot-interval-seconds must be finite and non-negative"
        )
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    if not policies or any(item not in VALID_POLICIES for item in policies):
        raise SystemExit("--policies must contain: %s" % ", ".join(VALID_POLICIES))
    if len(set(policies)) != len(policies):
        raise SystemExit("--policies cannot contain duplicates")

    commands = []
    for policy in policies:
        output = output_root / policy
        commands.append(
            build_run_command(
                source=source,
                task=task_path,
                output=output,
                policy=policy,
                promotions_per_round=promotions,
                profile_policy=args.profile_policy,
                strategy_allocation_policy=args.strategy_allocation_policy,
                api_url=args.api_url,
                api_model=args.api_model,
                api_key_env=args.api_key_env,
                api_temperature=args.api_temperature,
                agent_workers=args.agent_workers,
                incumbent_snapshot_interval_seconds=(
                    args.incumbent_snapshot_interval_seconds
                ),
                input_price=args.api_input_price_per_million,
                output_price=args.api_output_price_per_million,
                resume=args.resume and output.exists(),
            )
        )
    if args.dry_run:
        for command in commands:
            print(" ".join(_shell_quote(item) for item in command))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    repository_root = Path(__file__).resolve().parents[1]
    source_root = repository_root / "src"
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if not existing_pythonpath
        else str(source_root) + os.pathsep + existing_pythonpath
    )
    rows = []
    for policy, command in zip(policies, commands):
        print("\n=== Running policy: %s ===" % policy, flush=True)
        completed = subprocess.run(command, cwd=repository_root, env=environment)
        if completed.returncode != 0:
            raise SystemExit(
                "policy %s failed with exit code %d" % (policy, completed.returncode)
            )
        summary_path = output_root / policy / "summary.json"
        rows.append(_suite_row(policy, json.loads(summary_path.read_text())))

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "task_id": task.get("task_id"),
        "source": str(source),
        "task": str(task_path),
        "promotions_per_round": promotions,
        "profile_policy": args.profile_policy,
        "strategy_allocation_policy": args.strategy_allocation_policy,
        "agent_workers": args.agent_workers,
        "incumbent_snapshot_interval_seconds": (
            args.incumbent_snapshot_interval_seconds
        ),
        "runs": rows,
        "comparison_note": (
            "adaptive, model-top, and random use the same per-round promotion budget; "
            "measure-all intentionally uses more hardware and is not an equal-budget run."
        ),
    }
    (output_root / "suite_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "suite_summary.md").write_text(
        _suite_markdown(payload), encoding="utf-8"
    )
    print("\nSuite report: %s" % (output_root / "suite_summary.md"))
    return 0


def _suite_row(policy: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    cost = dict(summary.get("cost_ledger") or {})
    evaluator = dict(cost.get("evaluator") or {})
    api = dict(cost.get("api") or {})
    return {
        "policy": policy,
        "best_latency_ms": summary.get("best_latency_ms"),
        "speedup_over_seed": summary.get("speedup_over_seed"),
        "measured_candidates": summary.get("measured_candidates"),
        "profile_calls": summary.get("profile_calls"),
        "generator_calls": summary.get("generator_calls"),
        "input_tokens": api.get("input_tokens"),
        "output_tokens": api.get("output_tokens"),
        "estimated_api_cost_usd": api.get("estimated_cost_usd"),
        "api_wall_seconds": api.get("wall_seconds"),
        "evaluator_wall_seconds": evaluator.get("total_wall_seconds"),
        "output_directory": summary.get("output_directory"),
    }


def _suite_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Search Policy Experiment Suite",
        "",
        "Task: `%s`" % payload.get("task_id"),
        "",
        "Fixed promotions per round: %s" % payload["promotions_per_round"],
        "Strategy allocation: `%s`" % payload["strategy_allocation_policy"],
        "",
        "| Policy | Final latency (ms) | Speedup | Measured | NCU | API calls | API time (s) | Evaluator time (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["runs"]:
        lines.append(
            "| {policy} | {best_latency_ms} | {speedup_over_seed} | "
            "{measured_candidates} | {profile_calls} | {generator_calls} | "
            "{api_wall_seconds} | {evaluator_wall_seconds} |".format(**row)
        )
    lines.extend(["", payload["comparison_note"], ""])
    return "\n".join(lines)


def _shell_quote(value: str) -> str:
    if value and all(character.isalnum() or character in "-._/:" for character in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
