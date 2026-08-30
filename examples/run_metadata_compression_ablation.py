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
PROVIDER_LIMIT_ERROR = "provider_token_limit_exceeded"
DEFAULT_COMPRESSED_INPUT_LIMIT = 60000
DEFAULT_UNCOMPRESSED_LOCAL_GUARD = 1000000
DEFAULT_ABLATION_ROUNDS = 8


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
        help=(
            "Legacy shared local limit override. Prefer the separate compressed "
            "and uncompressed controls below."
        ),
    )
    parser.add_argument(
        "--compressed-api-max-input-tokens",
        type=int,
        default=DEFAULT_COMPRESSED_INPUT_LIMIT,
        help="Hard local guard for the compressed treatment.",
    )
    parser.add_argument(
        "--uncompressed-api-max-input-tokens",
        type=int,
        default=DEFAULT_UNCOMPRESSED_LOCAL_GUARD,
        help=(
            "High local guard for the uncompressed provider-limit probe. The "
            "provider should reject first when accumulated context becomes too large."
        ),
    )
    parser.add_argument(
        "--compressed-context-target-tokens",
        type=int,
        default=DEFAULT_COMPRESSED_INPUT_LIMIT,
        help=(
            "Budget that deterministically trims compact history and lessons. "
            "Per-request input should approach this value and then remain bounded."
        ),
    )
    parser.add_argument(
        "--compressed-history-limit",
        type=int,
        default=64,
        help="Compact history reservoir used only by the compressed treatment.",
    )
    parser.add_argument(
        "--compressed-lesson-limit",
        type=int,
        default=64,
        help="Compact lesson reservoir used only by the compressed treatment.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ABLATION_ROUNDS,
        help=(
            "Search rounds in the derived ablation task. Eight rounds normally "
            "make the bounded and accumulating trends visible."
        ),
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
    for name in (
        "compressed_api_max_input_tokens",
        "uncompressed_api_max_input_tokens",
        "compressed_context_target_tokens",
        "compressed_history_limit",
        "compressed_lesson_limit",
        "rounds",
    ):
        if int(getattr(args, name)) <= 0:
            raise SystemExit("--%s must be positive" % name.replace("_", "-"))
    if args.api_max_input_tokens is not None:
        if args.api_max_input_tokens <= 0:
            raise SystemExit("--api-max-input-tokens must be positive")
        args.compressed_api_max_input_tokens = args.api_max_input_tokens
        args.uncompressed_api_max_input_tokens = args.api_max_input_tokens
        args.compressed_context_target_tokens = min(
            args.compressed_context_target_tokens,
            args.api_max_input_tokens,
        )
    if (
        args.compressed_context_target_tokens
        > args.compressed_api_max_input_tokens
    ):
        raise SystemExit(
            "--compressed-context-target-tokens cannot exceed the compressed "
            "local input limit"
        )

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
    ablation_task = _materialize_ablation_task(task, output, args.rounds)

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
            task=ablation_task,
            output=treatment,
            policy=policy,
            snapshot_interval_seconds=args.snapshot_interval_seconds,
            api_max_input_tokens=(
                args.compressed_api_max_input_tokens
                if label == "compressed"
                else args.uncompressed_api_max_input_tokens
            ),
            compressed_context_target_tokens=(
                args.compressed_context_target_tokens
                if label == "compressed"
                else None
            ),
            metadata_history_limit=(
                args.compressed_history_limit if label == "compressed" else 8
            ),
            metadata_lesson_limit=(
                args.compressed_lesson_limit if label == "compressed" else 8
            ),
            resume=args.resume and treatment.exists() and any(treatment.iterdir()),
            forwarded=forwarded,
        )
        return_codes[label] = _run_treatment(
            command, repository_root, output / (label + ".log"), label
        )
        _write_comparison_artifacts(
            output, snapshot_interval_seconds=args.snapshot_interval_seconds
        )

    summaries = _write_comparison_artifacts(
        output, snapshot_interval_seconds=args.snapshot_interval_seconds
    )
    report = output / "compression_comparison.md"
    print("Compression comparison: %s" % report)
    unexpected = [
        item
        for item in summaries
        if item["status"]
        not in {"completed", "local-limit-exceeded", "provider-limit-exceeded"}
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
    compressed_context_target_tokens: Optional[int],
    metadata_history_limit: int,
    metadata_lesson_limit: int,
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
        "--metadata-history-limit",
        str(metadata_history_limit),
        "--metadata-lesson-limit",
        str(metadata_lesson_limit),
        "--agent-workers",
        "1",
    ]
    if compressed_context_target_tokens is not None:
        command.extend(
            [
                "--compressed-context-target-tokens",
                str(compressed_context_target_tokens),
            ]
        )
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


def _write_comparison_artifacts(
    output: Path, *, snapshot_interval_seconds: float = 300.0
) -> List[Dict[str, Any]]:
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

    round_rows = _round_rows(calls)
    five_minute_rows = _five_minute_rows(
        time_rows, snapshot_interval_seconds=snapshot_interval_seconds
    )
    _write_csv(output / "token_usage_by_call.csv", calls)
    _write_csv(output / "token_usage_by_round.csv", round_rows)
    _write_csv(output / "token_usage_by_time.csv", time_rows)
    _write_csv(output / "token_usage_by_5min.csv", five_minute_rows)
    (output / "compression_comparison.json").write_text(
        json.dumps(
            {
                "snapshot_interval_seconds": snapshot_interval_seconds,
                "treatments": summaries,
                "five_minute_tokens": five_minute_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "compression_comparison.md").write_text(
        _markdown_report(summaries, round_rows, five_minute_rows),
        encoding="utf-8",
    )
    return summaries


def _load_api_calls(
    treatment: Path, label: str, policy: str
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cumulative_input = 0.0
    cumulative_output = 0.0
    generator_config = _run_generator_config(treatment)
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
        limit = _classify_limit_error(error)
        context_budget = _mapping(prompt_size.get("compressed_context_budget"))
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
                "error_status_code": error.get("status_code"),
                "limit_kind": limit.get("kind"),
                "provider_limit_tokens": limit.get("limit_tokens"),
                "provider_requested_tokens": limit.get("requested_tokens"),
                "configured_local_input_limit": generator_config.get(
                    "max_input_tokens"
                ),
                "compressed_context_target_tokens": generator_config.get(
                    "compressed_context_target_tokens"
                ),
                "estimated_input_tokens": prompt_size.get(
                    "estimated_input_tokens"
                ),
                "max_output_tokens": prompt_size.get("max_output_tokens"),
                "estimated_reserved_tokens": prompt_size.get(
                    "estimated_reserved_tokens"
                ),
                "context_budget_trimmed": context_budget.get("trimmed"),
                "history_records_dropped_for_budget": context_budget.get(
                    "dropped_history_records"
                ),
                "lessons_dropped_for_budget": context_budget.get(
                    "dropped_shared_lessons"
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


def _five_minute_rows(
    snapshots: Sequence[Mapping[str, Any]], *, snapshot_interval_seconds: float
) -> List[Dict[str, Any]]:
    """Convert cumulative recorder snapshots into readable interval deltas."""

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for item in snapshots:
        grouped.setdefault(str(item.get("treatment")), []).append(item)
    rows: List[Dict[str, Any]] = []
    terminal_reasons = {"run-completed", "run-failed", "time-budget-stopped"}
    for treatment, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda item: _float(item.get("elapsed_seconds")))
        selected = [item for item in ordered if item.get("reason") == "interval"]
        terminals = [
            item for item in ordered if item.get("reason") in terminal_reasons
        ]
        if terminals:
            terminal = terminals[-1]
            if not selected or (
                _float(terminal.get("elapsed_seconds"))
                > _float(selected[-1].get("elapsed_seconds")) + 1e-6
            ):
                selected.append(terminal)

        previous_end = 0.0
        previous_total = 0.0
        for item in selected:
            scheduled = _float(item.get("scheduled_elapsed_seconds"))
            elapsed = _float(item.get("elapsed_seconds"))
            end_seconds = (
                scheduled
                if item.get("reason") == "interval" and scheduled > 0
                else elapsed
            )
            cumulative = _float(item.get("api_total_tokens"))
            rows.append(
                {
                    "treatment": treatment,
                    "metadata_compression_policy": item.get(
                        "metadata_compression_policy"
                    ),
                    "window_start_minutes": previous_end / 60.0,
                    "window_end_minutes": end_seconds / 60.0,
                    "window_label": "%s-%s min"
                    % (_minutes(previous_end), _minutes(end_seconds)),
                    "snapshot_reason": item.get("reason"),
                    "tokens_in_window": max(0.0, cumulative - previous_total),
                    "cumulative_total_tokens": cumulative,
                    "cumulative_input_tokens": _float(
                        item.get("api_input_tokens")
                    ),
                    "cumulative_output_tokens": _float(
                        item.get("api_output_tokens")
                    ),
                    "completed_round": item.get("completed_round"),
                    "active_round": item.get("active_round"),
                    "incumbent_latency_ms": item.get("incumbent_latency_ms"),
                    "speedup_over_seed": item.get("speedup_over_seed"),
                    "nominal_interval_seconds": snapshot_interval_seconds,
                }
            )
            previous_end = end_seconds
            previous_total = cumulative
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
        local_limits = [
            float(item["configured_local_input_limit"])
            for item in items
            if item.get("configured_local_input_limit") is not None
        ]
        context_targets = [
            float(item["compressed_context_target_tokens"])
            for item in items
            if item.get("compressed_context_target_tokens") is not None
        ]
        provider_limits = [
            float(item["provider_limit_tokens"])
            for item in items
            if item.get("provider_limit_tokens") is not None
        ]
        limit_kinds = [
            str(item["limit_kind"])
            for item in items
            if item.get("limit_kind")
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
                "configured_local_input_limit": max(local_limits)
                if local_limits
                else None,
                "compressed_context_target_tokens": max(context_targets)
                if context_targets
                else None,
                "provider_limit_tokens": max(provider_limits)
                if provider_limits
                else None,
                "limit_kind": limit_kinds[-1] if limit_kinds else None,
                "context_budget_trimmed": any(
                    bool(item.get("context_budget_trimmed")) for item in items
                ),
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
    provider_limit = any(
        item.get("limit_kind") == PROVIDER_LIMIT_ERROR for item in calls
    )
    status = (
        "completed"
        if summary
        else "local-limit-exceeded"
        if prompt_limit
        else "provider-limit-exceeded"
        if provider_limit
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
    generator_config = _run_generator_config(treatment)
    provider_limits = [
        float(item["provider_limit_tokens"])
        for item in calls
        if item.get("provider_limit_tokens") is not None
    ]
    provider_requests = [
        float(item["provider_requested_tokens"])
        for item in calls
        if item.get("provider_requested_tokens") is not None
    ]
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
        "provider_limit_reached": provider_limit,
        "configured_local_input_limit": generator_config.get("max_input_tokens"),
        "compressed_context_target_tokens": generator_config.get(
            "compressed_context_target_tokens"
        ),
        "provider_limit_tokens": provider_limits[-1]
        if provider_limits
        else None,
        "provider_requested_tokens": provider_requests[-1]
        if provider_requests
        else None,
        "context_budget_trimmed_calls": sum(
            1 for item in calls if item.get("context_budget_trimmed")
        ),
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
    five_minute_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Metadata Compression Ablation",
        "",
        "Generated at `%s`." % datetime.now().astimezone().isoformat(),
        "",
        "The compressed treatment uses deterministic `key-metrics-v1` packing "
        "toward a configured local context target. The uncompressed treatment "
        "keeps accumulating canonical history and uses a deliberately high local "
        "guard so the provider token limit is the expected terminal boundary.",
        "",
        "Actual token columns contain provider-billed usage from successful calls. "
        "A rejected limit-crossing request appears only in the estimated per-call "
        "column because it produced no successful-call usage.",
        "",
        "## Summary",
        "",
        "| Treatment | Status | Rounds | Actual total tokens | Max estimated "
        "input/call | Local guard | Compressed target | Provider limit | "
        "Terminal boundary |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summaries:
        lines.append(
            "| {treatment} | {status} | {completed_rounds} | "
            "{actual_total_tokens:.0f} | {maximum_estimated_input_tokens_per_call} | "
            "{local} | {target} | {provider} | {boundary} |".format(
                **item,
                local=_display_number(item.get("configured_local_input_limit")),
                target=_display_number(
                    item.get("compressed_context_target_tokens")
                ),
                provider=_display_number(item.get("provider_limit_tokens")),
                boundary=(
                    "provider"
                    if item.get("provider_limit_reached")
                    else "local"
                    if item.get("prompt_budget_reached")
                    else "completed"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Tokens Every Five Minutes",
            "",
            "`Tokens in window` is the newly consumed amount; `Cumulative` is the "
            "total provider usage up to that snapshot.",
            "",
            "| Treatment | Window | Tokens in window | Cumulative | Round state | Snapshot |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for item in five_minute_rows:
        round_state = "completed {completed}; active {active}".format(
            completed=item.get("completed_round") or 0,
            active=item.get("active_round") or "-",
        )
        lines.append(
            "| {treatment} | {window_label} | {window:.0f} | {cumulative:.0f} | "
            "{round_state} | {reason} |".format(
                treatment=item.get("treatment"),
                window_label=item.get("window_label"),
                window=float(item.get("tokens_in_window") or 0),
                cumulative=float(item.get("cumulative_total_tokens") or 0),
                round_state=round_state,
                reason=item.get("snapshot_reason"),
            )
        )
    summary_by_treatment = {
        str(item.get("treatment")): item for item in summaries
    }
    lines.extend(
        [
            "",
            "## Per-Round Growth",
            "",
            "| Treatment | Round | Actual tokens this round | Cumulative actual | "
            "Max estimated input/call | Applicable boundary | Boundary fill | Event |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for item in round_rows:
        treatment_summary = summary_by_treatment.get(str(item["treatment"]), {})
        boundary = (
            treatment_summary.get("compressed_context_target_tokens")
            or treatment_summary.get("provider_limit_tokens")
            or treatment_summary.get("configured_local_input_limit")
        )
        event = item.get("limit_kind") or (
            "compressed-budget-trim"
            if item.get("context_budget_trimmed")
            else ""
        )
        lines.append(
            "| {treatment} | {round} | {actual_total_tokens:.0f} | "
            "{cumulative_total_tokens:.0f} | {max_estimated_input_tokens_per_call} | "
            "{boundary} | {fill} | {event} |".format(
                **item,
                boundary=_display_number(boundary),
                fill=_limit_bar(item.get("max_estimated_input_tokens_per_call"), boundary),
                event=event or "-",
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
            "- `token_usage_by_5min.csv`: newly consumed and cumulative tokens "
            "for each five-minute window.",
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


def _materialize_ablation_task(
    source_task: Path, output: Path, rounds: int
) -> Path:
    """Create one immutable task contract shared by both treatments."""

    source_task = source_task.resolve()
    value = _read_json(source_task)
    if not value:
        raise SystemExit("Task contract is not a JSON object: %s" % source_task)
    _anchor_evaluator_working_directory(value, source_task.parent)
    budget = _mapping(value.get("budget"))
    budget["rounds"] = rounds
    value["budget"] = budget
    metadata = _mapping(value.get("metadata"))
    metadata["metadata_compression_ablation"] = {
        "source_task": str(source_task),
        "rounds": rounds,
        "treatments": [label for label, _policy in TREATMENTS],
    }
    value["metadata"] = metadata
    path = output / "ablation_task.json"
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") != rendered:
        existing = _read_json(path)
        _anchor_evaluator_working_directory(existing, source_task.parent)
        existing_rendered = json.dumps(existing, indent=2, sort_keys=True) + "\n"
        if (
            existing_rendered != rendered
            and any((output / label).exists() for label, _policy in TREATMENTS)
        ):
            raise SystemExit(
                "Existing ablation task differs from the requested configuration; "
                "choose a fresh --output directory"
            )
    path.write_text(rendered, encoding="utf-8")
    return path


def _anchor_evaluator_working_directory(
    task: Dict[str, Any], source_task_directory: Path
) -> None:
    """Preserve evaluator path semantics after copying a task into results/."""

    evaluator_value = task.get("evaluator")
    if not isinstance(evaluator_value, Mapping):
        return
    evaluator = dict(evaluator_value)
    working_directory_value = evaluator.get("working_directory")
    if not working_directory_value:
        return
    working_directory = Path(str(working_directory_value)).expanduser()
    if not working_directory.is_absolute():
        working_directory = source_task_directory / working_directory
    evaluator["working_directory"] = str(working_directory.resolve())
    task["evaluator"] = evaluator


def _run_generator_config(treatment: Path) -> Dict[str, Any]:
    task = _read_json(treatment / "task.json")
    run = _mapping(task.get("run"))
    return _mapping(run.get("generator"))


def _classify_limit_error(error: Mapping[str, Any]) -> Dict[str, Any]:
    if not error:
        return {}
    if error.get("error_code") == EXPECTED_LIMIT_ERROR:
        return {"kind": "local_prompt_budget_exceeded"}
    text = " ".join(
        str(error.get(key) or "")
        for key in ("message", "response_body", "error_code")
    )
    lowered = text.lower()
    status_code = error.get("status_code")
    provider_boundary = (
        status_code == 429
        and (
            "request too large" in lowered
            or "tokens per min" in lowered
            or ("requested" in lowered and "limit" in lowered)
        )
    ) or "context_length_exceeded" in lowered or "maximum context length" in lowered
    if not provider_boundary:
        return {}
    limit_tokens, requested_tokens = _parse_provider_limit_tokens(text)
    return {
        "kind": PROVIDER_LIMIT_ERROR,
        "limit_tokens": limit_tokens,
        "requested_tokens": requested_tokens,
    }


def _parse_provider_limit_tokens(text: str) -> tuple[Optional[int], Optional[int]]:
    match = re.search(
        r"Limit\s+([0-9,]+).*?Requested\s+([0-9,]+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None, None
    return (
        int(match.group(1).replace(",", "")),
        int(match.group(2).replace(",", "")),
    )


def _float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _minutes(seconds: float) -> str:
    minutes = seconds / 60.0
    return str(int(minutes)) if minutes.is_integer() else "%.1f" % minutes


def _display_number(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return "{:,}".format(int(float(value)))


def _limit_bar(value: Any, limit: Any, width: int = 10) -> str:
    if value is None or limit in {None, 0, ""}:
        return "-"
    ratio = max(0.0, float(value) / float(limit))
    filled = min(width, int(round(min(ratio, 1.0) * width)))
    return "`%s%s` %.1f%%" % (
        "#" * filled,
        "." * (width - filled),
        ratio * 100.0,
    )


def _validate_forwarded_arguments(values: Sequence[str]) -> None:
    controlled = {
        "--source",
        "--task",
        "--output",
        "--metadata-compression-policy",
        "--incumbent-snapshot-interval-seconds",
        "--api-max-input-tokens",
        "--metadata-history-limit",
        "--metadata-lesson-limit",
        "--compressed-context-target-tokens",
        "--agent-workers",
        "--resume",
    }
    conflict = next(
        (item for item in values if item.split("=", 1)[0] in controlled), None
    )
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
