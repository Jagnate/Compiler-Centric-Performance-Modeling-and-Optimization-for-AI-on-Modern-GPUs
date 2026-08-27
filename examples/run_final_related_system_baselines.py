#!/usr/bin/env python3
"""Run five related-system style baselines across five example kernels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "results" / "final_eval" / "related_system_baselines"
)
DEFAULT_MAX_SEARCH_SECONDS = 1800.0


@dataclass(frozen=True)
class KernelWorkload:
    key: str
    title: str
    source_filename: str
    task_filename: str

    def source_path(self, root: Path = REPOSITORY_ROOT) -> Path:
        return root / "examples" / self.source_filename

    def task_path(self, root: Path = REPOSITORY_ROOT) -> Path:
        return root / "examples" / self.task_filename


@dataclass(frozen=True)
class RelatedStyle:
    key: str
    title: str


@dataclass(frozen=True)
class Treatment:
    kernel: KernelWorkload
    style: RelatedStyle

    @property
    def relative_output(self) -> Path:
        return Path(self.kernel.key) / self.style.key


KERNELS: Tuple[KernelWorkload, ...] = (
    KernelWorkload(
        "matmul", "GEMM", "tilelang_matmul_kernel.py", "tilelang_matmul_task.json"
    ),
    KernelWorkload(
        "rms_norm",
        "RMSNorm",
        "tilelang_rms_norm_kernel.py",
        "tilelang_rms_norm_task.json",
    ),
    KernelWorkload(
        "conv2d", "Conv2D", "tilelang_conv2d_kernel.py", "tilelang_conv2d_task.json"
    ),
    KernelWorkload(
        "flash_attention",
        "Flash Attention",
        "tilelang_flash_attention_kernel.py",
        "tilelang_flash_attention_task.json",
    ),
    KernelWorkload(
        "fused_add_rms_norm",
        "Fused Add + RMSNorm",
        "tilelang_fused_add_rms_norm_kernel.py",
        "tilelang_fused_add_rms_norm_task.json",
    ),
)

RELATED_STYLES: Tuple[RelatedStyle, ...] = (
    RelatedStyle("kernelagent", "KernelAgent style"),
    RelatedStyle("kernelevolve", "KernelEvolve style"),
    RelatedStyle("kernelbench", "KernelBench style"),
    RelatedStyle("avo", "AVO style"),
    RelatedStyle("tilefoundry", "TileFoundry style"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run five related-system styles on all five existing kernel tasks "
            "(25 treatments total)."
        )
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument(
        "--only-kernel",
        choices=tuple(kernel.key for kernel in KERNELS),
        action="append",
        help="Run only this kernel; repeat to select several.",
    )
    parser.add_argument(
        "--only-style",
        choices=tuple(style.key for style in RELATED_STYLES),
        action="append",
        help="Run only this style; repeat to select several.",
    )
    parser.add_argument(
        "--max-search-seconds",
        type=float,
        default=DEFAULT_MAX_SEARCH_SECONDS,
        help="Soft controller limit per treatment; default 1800, or 0 to disable.",
    )
    parser.add_argument(
        "--incumbent-snapshot-interval-seconds", type=float, default=300.0
    )
    parser.add_argument(
        "--api-url", default=os.environ.get("KERNEL_OPT_API_URL")
    )
    parser.add_argument(
        "--api-model", default=os.environ.get("KERNEL_OPT_API_MODEL")
    )
    parser.add_argument("--api-key-env", default="KERNEL_OPT_API_KEY")
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--api-temperature", type=float)
    parser.add_argument("--api-max-output-tokens", type=int, default=12000)
    parser.add_argument("--api-planner-max-output-tokens", type=int, default=2000)
    parser.add_argument("--api-max-input-tokens", type=int, default=60000)
    parser.add_argument(
        "--api-max-tokens-field",
        choices=("max_tokens", "max_completion_tokens"),
        default="max_completion_tokens",
    )
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--api-retry-backoff", type=float, default=2.0)
    parser.add_argument("--api-input-price-per-million", type=float)
    parser.add_argument("--api-output-price-per-million", type=float)
    parser.add_argument("--skip-api-preflight", action="store_true")
    parser.add_argument("--no-json-response-format", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue later treatments after a child process fails.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without API/GPU work."
    )
    return parser


def selected_treatments(
    only_kernels: Optional[Sequence[str]] = None,
    only_styles: Optional[Sequence[str]] = None,
) -> List[Treatment]:
    """Return the deterministic kernel-major comparison matrix."""

    kernel_filter = set(only_kernels or ())
    style_filter = set(only_styles or ())
    return [
        Treatment(kernel, style)
        for kernel in KERNELS
        if not kernel_filter or kernel.key in kernel_filter
        for style in RELATED_STYLES
        if not style_filter or style.key in style_filter
    ]


def build_command(
    *,
    treatment: Treatment,
    output: Path,
    max_search_seconds: float,
    snapshot_interval_seconds: float,
    api_url: str,
    api_model: str,
    api_key_env: str,
    api_timeout: float = 120.0,
    api_temperature: Optional[float] = None,
    api_max_output_tokens: int = 12000,
    api_planner_max_output_tokens: int = 2000,
    api_max_input_tokens: int = 60000,
    api_max_tokens_field: str = "max_completion_tokens",
    api_retries: int = 3,
    api_retry_backoff: float = 2.0,
    api_input_price_per_million: Optional[float] = None,
    api_output_price_per_million: Optional[float] = None,
    skip_api_preflight: bool = False,
    no_json_response_format: bool = False,
    resume: bool = False,
    repository_root: Path = REPOSITORY_ROOT,
) -> List[str]:
    """Build one canonical single-agent style-emulation command."""

    command = [
        sys.executable,
        "-u",
        "-m",
        "kernel_optimization.cli",
        "--source",
        str(treatment.kernel.source_path(repository_root)),
        "--task",
        str(treatment.kernel.task_path(repository_root)),
        "--output",
        str(output),
        "--baseline-style",
        treatment.style.key,
        "--max-search-seconds",
        str(max_search_seconds),
        "--incumbent-snapshot-interval-seconds",
        str(snapshot_interval_seconds),
        "--api-url",
        api_url,
        "--api-model",
        api_model,
        "--api-key-env",
        api_key_env,
        "--api-timeout",
        str(api_timeout),
        "--api-max-output-tokens",
        str(api_max_output_tokens),
        "--api-planner-max-output-tokens",
        str(api_planner_max_output_tokens),
        "--api-max-input-tokens",
        str(api_max_input_tokens),
        "--api-max-tokens-field",
        api_max_tokens_field,
        "--api-retries",
        str(api_retries),
        "--api-retry-backoff",
        str(api_retry_backoff),
    ]
    optional_values = (
        ("--api-temperature", api_temperature),
        ("--api-input-price-per-million", api_input_price_per_million),
        ("--api-output-price-per-million", api_output_price_per_million),
    )
    for flag, value in optional_values:
        if value is not None:
            command.extend([flag, str(value)])
    if skip_api_preflight:
        command.append("--skip-api-preflight")
    if no_json_response_format:
        command.append("--no-json-response-format")
    if resume:
        command.append("--resume")
    return command


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_arguments(args)
    output_root = args.output_root.expanduser().resolve()
    treatments = selected_treatments(args.only_kernel, args.only_style)
    workloads = _load_workload_metadata(REPOSITORY_ROOT, treatments)
    graph_bound = sum(
        workloads[item.kernel.key]["candidate_graph_upper_bound"]
        for item in treatments
    )

    print(
        "Selected %d treatments: %d kernels x %d styles"
        % (
            len(treatments),
            len({item.kernel.key for item in treatments}),
            len({item.style.key for item in treatments}),
        ),
        flush=True,
    )
    print("Agent workers: 1 (canonical preset default)", flush=True)
    print("Candidate graph upper bound: %d nodes" % graph_bound, flush=True)
    if args.max_search_seconds > 0:
        print(
            "Search cap: %.0f seconds per treatment, %.2f hours total maximum"
            % (
                args.max_search_seconds,
                args.max_search_seconds * len(treatments) / 3600.0,
            ),
            flush=True,
        )
    else:
        print("Search cap: disabled", flush=True)

    environment = dict(os.environ)
    source_root = REPOSITORY_ROOT / "src"
    pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root)
        if not pythonpath
        else str(source_root) + os.pathsep + pythonpath
    )
    environment["PYTHONUNBUFFERED"] = "1"
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    failure_code = 0
    for index, treatment in enumerate(treatments, start=1):
        output = output_root / treatment.relative_output
        summary_path = output / "summary.json"
        if summary_path.is_file():
            print(
                "\n[%d/%d] %s / %s already complete; reusing %s"
                % (
                    index,
                    len(treatments),
                    treatment.kernel.title,
                    treatment.style.title,
                    summary_path,
                ),
                flush=True,
            )
            rows.append(_summary_row(treatment, _read_json(summary_path), "reused"))
            continue

        resume = output.is_dir() and any(output.iterdir())
        command = build_command(
            treatment=treatment,
            output=output,
            max_search_seconds=args.max_search_seconds,
            snapshot_interval_seconds=args.incumbent_snapshot_interval_seconds,
            api_url=args.api_url,
            api_model=args.api_model,
            api_key_env=args.api_key_env,
            api_timeout=args.api_timeout,
            api_temperature=args.api_temperature,
            api_max_output_tokens=args.api_max_output_tokens,
            api_planner_max_output_tokens=args.api_planner_max_output_tokens,
            api_max_input_tokens=args.api_max_input_tokens,
            api_max_tokens_field=args.api_max_tokens_field,
            api_retries=args.api_retries,
            api_retry_backoff=args.api_retry_backoff,
            api_input_price_per_million=args.api_input_price_per_million,
            api_output_price_per_million=args.api_output_price_per_million,
            skip_api_preflight=args.skip_api_preflight,
            no_json_response_format=args.no_json_response_format,
            resume=resume,
        )
        print(
            "\n[%d/%d] %s / %s%s\n%s"
            % (
                index,
                len(treatments),
                treatment.kernel.title,
                treatment.style.title,
                " (resume)" if resume else "",
                shlex.join(command),
            ),
            flush=True,
        )
        if args.dry_run:
            continue

        completed = subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment)
        if completed.returncode != 0:
            failure_code = failure_code or completed.returncode
            rows.append(
                _summary_row(
                    treatment,
                    {"error": "exit code %d" % completed.returncode},
                    "failed",
                )
            )
            _write_reports(output_root, treatments, workloads, rows, args.max_search_seconds)
            if not args.continue_on_error:
                print(
                    "Treatment failed. Fix the issue and rerun; completed runs "
                    "will be reused and this run will resume.",
                    flush=True,
                )
                return completed.returncode
            continue

        if summary_path.is_file():
            rows.append(_summary_row(treatment, _read_json(summary_path), "completed"))
        else:
            failure_code = failure_code or 1
            rows.append(
                _summary_row(
                    treatment,
                    {"error": "child exited successfully without summary.json"},
                    "failed",
                )
            )
        _write_reports(output_root, treatments, workloads, rows, args.max_search_seconds)

    if args.dry_run:
        print("\nDry run complete; no output was created.", flush=True)
        return 0

    _write_reports(output_root, treatments, workloads, rows, args.max_search_seconds)
    print("\nSuite summary: %s" % (output_root / "suite_summary.md"), flush=True)
    if failure_code:
        print("Some treatments failed; rerun the same command to resume them.")
        return failure_code
    print("All %d treatments completed." % len(treatments), flush=True)
    return 0


def _validate_arguments(args: argparse.Namespace) -> None:
    for name in (
        "max_search_seconds",
        "incumbent_snapshot_interval_seconds",
        "api_timeout",
        "api_retry_backoff",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise SystemExit(
                "--%s must be finite and non-negative" % name.replace("_", "-")
            )
    for name in (
        "api_max_output_tokens",
        "api_planner_max_output_tokens",
        "api_max_input_tokens",
        "api_retries",
    ):
        if int(getattr(args, name)) <= 0:
            raise SystemExit("--%s must be positive" % name.replace("_", "-"))
    for name in ("api_input_price_per_million", "api_output_price_per_million"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise SystemExit("--%s must be non-negative" % name.replace("_", "-"))
    if (args.api_input_price_per_million is None) != (
        args.api_output_price_per_million is None
    ):
        raise SystemExit("API cost reporting requires both token prices")
    if not args.api_url or not args.api_model:
        raise SystemExit(
            "Set KERNEL_OPT_API_URL and KERNEL_OPT_API_MODEL, or pass "
            "--api-url and --api-model."
        )
    if not args.dry_run and not os.environ.get(args.api_key_env):
        raise SystemExit("API key environment variable %s is not set" % args.api_key_env)
    for kernel in KERNELS:
        for path in (kernel.source_path(), kernel.task_path()):
            if not path.is_file():
                raise SystemExit("Required example file does not exist: %s" % path)


def _load_workload_metadata(
    repository_root: Path, treatments: Sequence[Treatment]
) -> Dict[str, Dict[str, Any]]:
    selected = {item.kernel.key for item in treatments}
    metadata: Dict[str, Dict[str, Any]] = {}
    for kernel in KERNELS:
        if kernel.key not in selected:
            continue
        task_path = kernel.task_path(repository_root)
        task = _read_json(task_path)
        budget = dict(task.get("budget") or {})
        rounds = int(budget.get("rounds", 0))
        proposals = int(budget.get("proposals_per_round", 0))
        repairs = int(budget.get("max_repairs_per_round", 0))
        metadata[kernel.key] = {
            "kernel": kernel.key,
            "title": kernel.title,
            "source": str(kernel.source_path(repository_root)),
            "task": str(task_path),
            "task_id": task.get("task_id"),
            "factory_arguments": dict(
                dict(task.get("workload") or {}).get("factory_arguments") or {}
            ),
            "rounds": rounds,
            "proposals_per_round": proposals,
            "max_repairs_per_round": repairs,
            "candidate_graph_upper_bound": 1 + rounds * (proposals + repairs),
        }
    return metadata


def _summary_row(
    treatment: Treatment, summary: Mapping[str, Any], status: str
) -> Dict[str, Any]:
    ledger = dict(summary.get("cost_ledger") or {})
    api = dict(ledger.get("api") or {})
    evaluator = dict(ledger.get("evaluator") or {})
    return {
        "kernel": treatment.kernel.key,
        "kernel_title": treatment.kernel.title,
        "style": treatment.style.key,
        "style_title": treatment.style.title,
        "status": status,
        "seed_latency_ms": summary.get("seed_latency_ms"),
        "search_best_latency_ms": summary.get("search_best_latency_ms"),
        "best_latency_ms": summary.get("best_latency_ms"),
        "speedup_over_seed": summary.get("speedup_over_seed"),
        "completed_rounds": summary.get("completed_rounds"),
        "generated_candidates": summary.get("generated_candidates"),
        "measured_candidates": summary.get("measured_candidates"),
        "profile_calls": summary.get("profile_calls"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "termination_reason": summary.get("termination_reason"),
        "time_budget_exhausted": summary.get("time_budget_exhausted"),
        "candidate_graph_upper_bound": summary.get("candidate_graph_upper_bound"),
        "api_input_tokens": api.get("input_tokens"),
        "api_output_tokens": api.get("output_tokens"),
        "estimated_api_cost_usd": api.get("estimated_cost_usd"),
        "evaluator_wall_seconds": evaluator.get("total_wall_seconds"),
        "error": summary.get("error"),
        "output": str(treatment.relative_output),
    }


def _write_reports(
    output_root: Path,
    treatments: Sequence[Treatment],
    workloads: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    max_search_seconds: float,
) -> None:
    status_counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    graph_bound = sum(
        int(workloads[item.kernel.key]["candidate_graph_upper_bound"])
        for item in treatments
    )
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "claim": (
            "Common-harness control-flow style emulations; these are not exact "
            "reimplementations and do not reproduce published results."
        ),
        "output_root": str(output_root),
        "planned_treatments": len(treatments),
        "agent_workers": 1,
        "max_search_seconds_per_treatment": max_search_seconds,
        "maximum_configured_search_seconds": (
            max_search_seconds * len(treatments) if max_search_seconds > 0 else None
        ),
        "candidate_graph_upper_bound": graph_bound,
        "status_counts": status_counts,
        "styles": [style.key for style in RELATED_STYLES],
        "workloads": [dict(workloads[key]) for key in workloads],
        "runs": list(rows),
    }
    (output_root / "suite_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "suite_summary.md").write_text(
        _suite_markdown(payload), encoding="utf-8"
    )


def _suite_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Related-System Baseline Suite",
        "",
        str(payload["claim"]),
        "",
        "## Suite Configuration",
        "",
        "| Setting | Value |",
        "| --- | ---: |",
        "| Planned treatments | %s |" % payload["planned_treatments"],
        "| Agent workers | 1 |",
        "| Search cap per treatment | %s seconds |"
        % _format_value(payload["max_search_seconds_per_treatment"]),
        "| Candidate graph upper bound | %s nodes |"
        % payload["candidate_graph_upper_bound"],
        "",
        "## Unchanged Workloads",
        "",
        "| Kernel | Task | Primary factory arguments | Graph bound per style |",
        "| --- | --- | --- | ---: |",
    ]
    for workload in payload["workloads"]:
        shape = json.dumps(
            workload["factory_arguments"], sort_keys=True, separators=(",", ":")
        )
        lines.append(
            "| {title} | `{task_id}` | `{shape}` | {bound} |".format(
                title=workload["title"],
                task_id=workload["task_id"],
                shape=shape,
                bound=workload["candidate_graph_upper_bound"],
            )
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Kernel | Style | Status | Seed (ms) | Search best (ms) | Final best (ms) | Speedup | Rounds | Generated | Measured | NCU | Elapsed (s) | Termination |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["runs"]:
        lines.append(
            "| {kernel_title} | {style_title} | {status} | {seed} | "
            "{search_best} | {best} | {speedup} | {rounds} | {generated} | "
            "{measured} | {ncu} | {elapsed} | {termination} |".format(
                kernel_title=row["kernel_title"],
                style_title=row["style_title"],
                status=row["status"],
                seed=_format_value(row.get("seed_latency_ms")),
                search_best=_format_value(row.get("search_best_latency_ms")),
                best=_format_value(row.get("best_latency_ms")),
                speedup=_format_speedup(row.get("speedup_over_seed")),
                rounds=_format_value(row.get("completed_rounds")),
                generated=_format_value(row.get("generated_candidates")),
                measured=_format_value(row.get("measured_candidates")),
                ncu=_format_value(row.get("profile_calls")),
                elapsed=_format_value(row.get("elapsed_seconds")),
                termination=row.get("termination_reason") or row.get("error") or "-",
            )
        )
    lines.extend(
        [
            "",
            "Each row uses the original source and task JSON. The runner changes "
            "only `--baseline-style` and API/time controls; it does not rewrite shapes.",
            "",
            "A `time-budget` termination is a valid fixed-time result. Rerunning "
            "the suite reuses completed rows and resumes an interrupted row.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    return "%.4f" % value if isinstance(value, float) else str(value)


def _format_speedup(value: Any) -> str:
    return "-" if value is None else "%.4fx" % float(value)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
