#!/usr/bin/env python3
"""Run and summarize compressed versus uncompressed prompt-context treatments."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence


TREATMENTS = (
    ("compressed", "key-metrics-v1"),
    ("uncompressed", "none"),
)
EXPECTED_LIMIT_ERROR = "local_prompt_budget_exceeded"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same kernel optimization with and without deterministic "
            "metadata compression, then export token-growth tables."
        )
    )
    parser.add_argument(
        "--source",
        default="examples/tilelang_matmul_kernel.py",
        help="Kernel source; defaults to the stable Matmul compression workload.",
    )
    parser.add_argument(
        "--task",
        default="examples/tilelang_matmul_task.json",
        help="Task contract; defaults to the Matmul compression workload.",
    )
    parser.add_argument(
        "--output",
        default="results/final_eval/metadata_compression_matmul",
        help="Parent directory for compressed, uncompressed, and comparison artifacts.",
    )
    parser.add_argument(
        "--snapshot-interval-seconds",
        type=float,
        default=300.0,
        help="Wall-clock token and incumbent sampling interval.",
    )
    parser.add_argument(
        "--api-max-input-tokens",
        type=int,
        default=60000,
        help="Shared local safety limit. An uncompressed limit hit is a valid outcome.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete treatment directories and reuse completed treatments.",
    )
    parser.add_argument(
        "forwarded",
        nargs=argparse.REMAINDER,
        help="Additional kernel_optimization.cli arguments after --.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.snapshot_interval_seconds <= 0:
        raise SystemExit("--snapshot-interval-seconds must be positive")
    if args.api_max_input_tokens <= 0:
        raise SystemExit("--api-max-input-tokens must be positive")

    repository_root = Path(__file__).resolve().parents[1]
    source = Path(args.source).expanduser().resolve()
    task = Path(args.task).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_file():
        raise SystemExit("Kernel source does not exist: %s" % source)
    if not task.is_file():
        raise SystemExit("Task contract does not exist: %s" % task)
    forwarded = list(args.forwarded)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    _validate_forwarded_arguments(forwarded)
    output.mkdir(parents=True, exist_ok=True)

    return_codes: Dict[str, int] = {}
    for label, policy in TREATMENTS:
        treatment = output / label
        if (treatment / "summary.json").is_file():
            print("[%s] Completed artifacts already exist; reusing them." % label)
            return_codes[label] = 0
            continue
        if treatment.exists() and any(treatment.iterdir()) and not args.resume:
            raise SystemExit(
                "Treatment directory is not empty; add --resume or choose a fresh "
                "--output: %s" % treatment
            )
        command = build_treatment_command(
            source=source,
            task=task,
            output=treatment,
            policy=policy,
            snapshot_interval_seconds=args.snapshot_interval_seconds,
            api_max_input_tokens=args.api_max_input_tokens,
            resume=args.resume and treatment.exists() and any(treatment.iterdir()),
            forwarded=forwarded,
        )
        return_codes[label] = _run_treatment(
            command, repository_root, output / (label + ".log"), label
        )
        _write_comparison_artifacts(output)

    summaries = _write_comparison_artifacts(output)
    report = output / "compression_comparison.md"
    print("Compression comparison: %s" % report)
    unexpected = [
        item
        for item in summaries
        if item["status"] not in {"completed", "prompt-budget-exceeded"}
    ]
    if unexpected:
        return 1
    if return_codes.get("compressed", 0) != 0:
        return 1
    return 0


def build_treatment_command(
    *,
    source: Path,
    task: Path,
    output: Path,
    policy: str,
    snapshot_interval_seconds: float,
    api_max_input_tokens: int,
    resume: bool,
    forwarded: Sequence[str],
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
        "--metadata-compression-policy",
        policy,
        "--incumbent-snapshot-interval-seconds",
        str(snapshot_interval_seconds),
        "--api-max-input-tokens",
        str(api_max_input_tokens),
    ]
    if resume:
        command.append("--resume")
    command.extend(forwarded)
    return command


def _run_treatment(
    command: Sequence[str], repository_root: Path, log_path: Path, label: str
) -> int:
    environment = dict(os.environ)
    source_root = str(repository_root / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing else source_root + os.pathsep + existing
    )
    print("[%s] $ %s" % (label, " ".join(command)))
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=str(repository_root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print("[%s] %s" % (label, line), end="", flush=True)
        return process.wait()


def _write_comparison_artifacts(output: Path) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    time_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for label, policy in TREATMENTS:
        treatment = output / label
        treatment_calls = _load_api_calls(treatment, label, policy)
        calls.extend(treatment_calls)
        time_rows.extend(_load_time_snapshots(treatment, label, policy))
        summaries.append(
            _summarize_treatment(treatment, label, policy, treatment_calls)
        )

    _write_csv(output / "token_usage_by_call.csv", calls)
    _write_csv(output / "token_usage_by_round.csv", _round_rows(calls))
    _write_csv(output / "token_usage_by_time.csv", time_rows)
    (output / "compression_comparison.json").write_text(
        json.dumps({"treatments": summaries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "compression_comparison.md").write_text(
        _markdown_report(summaries, _round_rows(calls)), encoding="utf-8"
    )
    return summaries


def _load_api_calls(
    treatment: Path, label: str, policy: str
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cumulative_input = 0.0
    cumulative_output = 0.0
    for sequence, path in enumerate(sorted((treatment / "api_calls").glob("*.json")), 1):
        payload = _read_json(path)
        metadata = _mapping(payload.get("metadata"))
        provider = _mapping(metadata.get("provider")) or metadata
        exchange = _mapping(payload.get("exchange"))
        usage = _mapping(provider.get("usage"))
        prompt_size = _mapping(provider.get("prompt_size"))
        if not prompt_size:
            prompt_size = _mapping(exchange.get("prompt_size"))
        input_tokens = _token_value(usage, "prompt_tokens", "input_tokens")
        output_tokens = _token_value(
            usage, "completion_tokens", "output_tokens"
        )
        total_tokens = _token_value(usage, "total_tokens")
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens
        cumulative_input += input_tokens
        cumulative_output += output_tokens
        kind = str(payload.get("kind") or path.stem)
        api_phase = _api_phase(kind)
        round_number = metadata.get("round")
        if round_number is None:
            match = re.search(r"round-(\d+)", kind)
            round_number = int(match.group(1)) if match else 0
        error = _mapping(provider.get("error")) or _mapping(exchange.get("error"))
        rows.append(
            {
                "treatment": label,
                "metadata_compression_policy": policy,
                "sequence": sequence,
                "timestamp": payload.get("timestamp"),
                "kind": kind,
                "api_phase": api_phase,
                "round": round_number,
                "status": "failed" if error else "completed",
                "error_code": error.get("error_code"),
                "estimated_input_tokens": prompt_size.get(
                    "estimated_input_tokens"
                ),
                "max_output_tokens": prompt_size.get("max_output_tokens"),
                "estimated_reserved_tokens": prompt_size.get(
                    "estimated_reserved_tokens"
                ),
                "actual_input_tokens": input_tokens,
                "actual_output_tokens": output_tokens,
                "actual_total_tokens": total_tokens,
                "cumulative_input_tokens": cumulative_input,
                "cumulative_output_tokens": cumulative_output,
                "cumulative_total_tokens": cumulative_input + cumulative_output,
                "artifact": str(path),
            }
        )
    return rows


def _load_time_snapshots(
    treatment: Path, label: str, policy: str
) -> List[Dict[str, Any]]:
    path = treatment / "incumbent_history.csv"
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for item in csv.DictReader(handle):
            if item.get("reason") not in {
                "interval",
                "round-completed",
                "run-completed",
                "run-failed",
                "time-budget-stopped",
            }:
                continue
            rows.append(
                {
                    "treatment": label,
                    "metadata_compression_policy": policy,
                    "reason": item.get("reason"),
                    "scheduled_elapsed_seconds": item.get(
                        "scheduled_elapsed_seconds"
                    ),
                    "elapsed_seconds": item.get("elapsed_seconds"),
                    "completed_round": item.get("completed_round"),
                    "active_round": item.get("active_round"),
                    "api_input_tokens": item.get("api_input_tokens"),
                    "api_output_tokens": item.get("api_output_tokens"),
                    "api_total_tokens": item.get("api_total_tokens"),
                    "incumbent_latency_ms": item.get("incumbent_latency_ms"),
                    "speedup_over_seed": item.get("speedup_over_seed"),
                }
            )
    return rows


def _round_rows(calls: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, int], List[Mapping[str, Any]]] = {}
    for item in calls:
        key = (str(item["treatment"]), int(item.get("round") or 0))
        grouped.setdefault(key, []).append(item)
    rows: List[Dict[str, Any]] = []
    cumulative: Dict[str, float] = {}
    for (treatment, round_number), items in sorted(grouped.items()):
        input_tokens = sum(float(item.get("actual_input_tokens") or 0) for item in items)
        output_tokens = sum(
            float(item.get("actual_output_tokens") or 0) for item in items
        )
        cumulative[treatment] = cumulative.get(treatment, 0.0) + input_tokens + output_tokens
        estimates = [
            float(item["estimated_input_tokens"])
            for item in items
            if item.get("estimated_input_tokens") is not None
        ]
        rows.append(
            {
                "treatment": treatment,
                "metadata_compression_policy": items[0][
                    "metadata_compression_policy"
                ],
                "round": round_number,
                "api_calls": len(items),
                "actual_input_tokens": input_tokens,
                "actual_output_tokens": output_tokens,
                "actual_total_tokens": input_tokens + output_tokens,
                "generation_repair_input_tokens": sum(
                    float(item.get("actual_input_tokens") or 0)
                    for item in items
                    if item.get("api_phase") in {"generate", "repair"}
                ),
                "planner_input_tokens": sum(
                    float(item.get("actual_input_tokens") or 0)
                    for item in items
                    if item.get("api_phase") == "plan"
                ),
                "cumulative_total_tokens": cumulative[treatment],
                "max_estimated_input_tokens_per_call": max(estimates)
                if estimates
                else None,
                "prompt_budget_failure": any(
                    item.get("error_code") == EXPECTED_LIMIT_ERROR for item in items
                ),
            }
        )
    return rows


def _summarize_treatment(
    treatment: Path,
    label: str,
    policy: str,
    calls: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    summary = _read_json(treatment / "summary.json")
    state = _read_json(treatment / "state.json")
    failure = _read_json(treatment / "failure.json")
    prompt_limit = any(
        item.get("error_code") == EXPECTED_LIMIT_ERROR for item in calls
    )
    status = (
        "completed"
        if summary
        else "prompt-budget-exceeded"
        if prompt_limit
        else "failed"
        if failure
        else "not-started"
    )
    estimates = [
        float(item["estimated_input_tokens"])
        for item in calls
        if item.get("estimated_input_tokens") is not None
    ]
    actual_input = sum(float(item.get("actual_input_tokens") or 0) for item in calls)
    actual_output = sum(float(item.get("actual_output_tokens") or 0) for item in calls)
    generation_repair_input = sum(
        float(item.get("actual_input_tokens") or 0)
        for item in calls
        if item.get("api_phase") in {"generate", "repair"}
    )
    planner_input = sum(
        float(item.get("actual_input_tokens") or 0)
        for item in calls
        if item.get("api_phase") == "plan"
    )
    return {
        "treatment": label,
        "metadata_compression_policy": policy,
        "status": status,
        "completed_rounds": summary.get(
            "completed_rounds", state.get("completed_round", 0)
        ),
        "api_calls_archived": len(calls),
        "actual_input_tokens": actual_input,
        "actual_output_tokens": actual_output,
        "actual_total_tokens": actual_input + actual_output,
        "generation_repair_input_tokens": generation_repair_input,
        "planner_input_tokens": planner_input,
        "maximum_estimated_input_tokens_per_call": max(estimates)
        if estimates
        else None,
        "prompt_budget_reached": prompt_limit,
        "best_latency_ms": summary.get("best_latency_ms", state.get("best_latency_ms")),
        "speedup_over_seed": summary.get("speedup_over_seed"),
        "failure": {
            "error_type": failure.get("error_type"),
            "message": failure.get("message"),
        }
        if failure
        else None,
    }


def _markdown_report(
    summaries: Sequence[Mapping[str, Any]],
    round_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Metadata Compression Ablation",
        "",
        "Generated at `%s`." % datetime.now().astimezone().isoformat(),
        "",
        "The compressed treatment uses deterministic `key-metrics-v1` context. "
        "The uncompressed treatment includes complete accumulated generation and "
        "repair metadata, while retaining the same local input-token safety limit.",
        "",
        "Cumulative tokens should grow in both treatments because every API call "
        "has a cost. The primary compression signal is whether per-call input "
        "tokens remain bounded; cumulative uncompressed usage may grow "
        "superlinearly as history accumulates.",
        "",
        "## Summary",
        "",
        "| Treatment | Policy | Status | Rounds | Generate/repair input | Planner input | All output | Total tokens | Max estimated input/call | Hit limit |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summaries:
        lines.append(
            "| {treatment} | `{metadata_compression_policy}` | {status} | "
            "{completed_rounds} | {generation_repair_input_tokens:.0f} | "
            "{planner_input_tokens:.0f} | {actual_output_tokens:.0f} | "
            "{actual_total_tokens:.0f} | "
            "{maximum_estimated_input_tokens_per_call} | {limit} |".format(
                **item,
                limit="yes" if item["prompt_budget_reached"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Per-Round Growth",
            "",
            "| Treatment | Round | API calls | Generate/repair input | Planner input | Output tokens | Cumulative total | Max estimated input/call | Limit failure |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in round_rows:
        lines.append(
            "| {treatment} | {round} | {api_calls} | "
            "{generation_repair_input_tokens:.0f} | {planner_input_tokens:.0f} | "
            "{actual_output_tokens:.0f} | {cumulative_total_tokens:.0f} | "
            "{max_estimated_input_tokens_per_call} | {limit} |".format(
                **item,
                limit="yes" if item["prompt_budget_failure"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `token_usage_by_call.csv`: one row per archived API call.",
            "- `token_usage_by_round.csv`: actual provider usage grouped by round.",
            "- `token_usage_by_time.csv`: fixed-interval and round-completion snapshots.",
            "- `compression_comparison.json`: machine-readable treatment summary.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _validate_forwarded_arguments(values: Sequence[str]) -> None:
    controlled = {
        "--source",
        "--task",
        "--output",
        "--metadata-compression-policy",
        "--incumbent-snapshot-interval-seconds",
        "--api-max-input-tokens",
        "--resume",
    }
    conflict = next((item for item in values if item in controlled), None)
    if conflict:
        raise SystemExit("The ablation driver controls %s" % conflict)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _token_value(value: Mapping[str, Any], *names: str) -> float:
    for name in names:
        item = value.get(name)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return float(item)
    return 0.0


def _api_phase(kind: str) -> str:
    lowered = kind.lower()
    if "repair" in lowered:
        return "repair"
    if "generate" in lowered:
        return "generate"
    if "plan" in lowered:
        return "plan"
    if "preflight" in lowered:
        return "preflight"
    return "other"


if __name__ == "__main__":
    raise SystemExit(main())
