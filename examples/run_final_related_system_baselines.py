#!/usr/bin/env python3
"""Run the native system and five related-system styles across five kernels."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPE_CONFIG = "basic"
DEFAULT_MAX_SEARCH_SECONDS = 900.0
DEFAULT_MEASUREMENT_REPEATS = 3
DEFAULT_TIME_BUDGET_ROUND_CEILING = 128


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

# Canonical workload families live beside the runner so one command controls the
# complete experiment matrix. Custom JSON suites remain available through
# --workload-suite, but published runs should use one of these named snapshots.
SHAPE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "basic": {
        "suite_id": "related_system_basic_2048_shapes",
        "title": "Related-System Baselines: Basic 2048-Class Shapes",
        "description": (
            "One stable RTX 3090 shape per kernel, anchored by 2048 square "
            "GEMM. Search and final validation use the same shape."
        ),
        "output_directory": "related_system_baselines_{budget}_basic_2048",
        "workloads": {
            "matmul": {
                "task_id": "tilelang_matmul_m2048_n2048_k2048_rtx3090_basic",
                "rationale": "Square 2048 GEMM with stable sub-millisecond timing.",
                "factory_arguments": {"m": 2048, "n": 2048, "k": 2048},
                "search_cases": [
                    {"case_id": "primary-m2048-n2048-k2048"}
                ],
                "final_cases": [
                    {"case_id": "final-m2048-n2048-k2048"}
                ],
            },
            "rms_norm": {
                "task_id": "tilelang_rms_norm_r8192_h4096_rtx3090_basic",
                "rationale": "Observed optimizable 4096-wide normalization workload.",
                "factory_arguments": {
                    "rows": 8192,
                    "hidden_size": 4096,
                    "epsilon": 1e-6,
                },
                "search_cases": [{"case_id": "primary-r8192-h4096"}],
                "final_cases": [{"case_id": "final-r8192-h4096"}],
            },
            "conv2d": {
                "task_id": (
                    "tilelang_conv2d_n32_h56_w56_c64_f128_k3_rtx3090_basic"
                ),
                "rationale": "Observed optimizable ResNet-like 3x3 convolution.",
                "factory_arguments": {
                    "batch": 32,
                    "in_height": 56,
                    "in_width": 56,
                    "in_channels": 64,
                    "out_channels": 128,
                    "kernel_size": 3,
                    "stride": 1,
                    "dilation": 1,
                    "padding": 1,
                },
                "search_cases": [
                    {"case_id": "primary-n32-h56-w56-c64-f128-k3-s1"}
                ],
                "final_cases": [
                    {"case_id": "final-n32-h56-w56-c64-f128-k3-s1"}
                ],
            },
            "flash_attention": {
                "task_id": (
                    "tilelang_flash_attention_b1_h32_s1024_d64_rtx3090_basic"
                ),
                "rationale": "Standard H32 S1024 non-causal attention.",
                "factory_arguments": {
                    "batch": 1,
                    "heads": 32,
                    "seq_len": 1024,
                    "dim": 64,
                    "is_causal": False,
                },
                "search_cases": [
                    {"case_id": "primary-b1-h32-s1024-d64"}
                ],
                "final_cases": [
                    {"case_id": "final-b1-h32-s1024-d64"}
                ],
            },
            "fused_add_rms_norm": {
                "task_id": (
                    "tilelang_fused_add_rms_norm_r8192_h4096_rtx3090_basic"
                ),
                "rationale": "Observed optimizable fused residual-add normalization.",
                "factory_arguments": {
                    "rows": 8192,
                    "hidden_size": 4096,
                    "epsilon": 1e-6,
                },
                "search_cases": [{"case_id": "primary-r8192-h4096"}],
                "final_cases": [{"case_id": "final-r8192-h4096"}],
            },
        },
    },
    "large": {
        "suite_id": "related_system_large_shapes",
        "title": "Related-System Baselines: Large Stable Shapes",
        "description": (
            "Larger single-shape workloads for stable RTX 3090 timing. Search "
            "and final validation use the same shape."
        ),
        "output_directory": "related_system_baselines_{budget}_large",
        "workloads": {
            "matmul": {
                "task_id": "tilelang_matmul_4096_rtx3090_large",
                "rationale": "Square 4096 GEMM with eight times the 2048 FLOPs.",
                "factory_arguments": {"m": 4096, "n": 4096, "k": 4096},
                "search_cases": [
                    {"case_id": "primary-m4096-n4096-k4096"}
                ],
                "final_cases": [
                    {"case_id": "final-m4096-n4096-k4096"}
                ],
            },
            "rms_norm": {
                "task_id": "tilelang_rms_norm_r16384_h4096_rtx3090_large",
                "rationale": "Double-row 4096-wide normalization for stable timing.",
                "factory_arguments": {
                    "rows": 16384,
                    "hidden_size": 4096,
                    "epsilon": 1e-6,
                },
                "search_cases": [
                    {"case_id": "primary-r16384-h4096"}
                ],
                "final_cases": [
                    {"case_id": "final-r16384-h4096"}
                ],
            },
            "conv2d": {
                "task_id": (
                    "tilelang_conv2d_n64_h56_w56_c64_f128_k3_rtx3090_large"
                ),
                "rationale": "Double-batch spatial 3x3 Conv2D workload.",
                "factory_arguments": {
                    "batch": 64,
                    "in_height": 56,
                    "in_width": 56,
                    "in_channels": 64,
                    "out_channels": 128,
                    "kernel_size": 3,
                    "stride": 1,
                    "dilation": 1,
                    "padding": 1,
                },
                "search_cases": [
                    {"case_id": "primary-n64-h56-w56-c64-f128-k3-s1"}
                ],
                "final_cases": [
                    {"case_id": "final-n64-h56-w56-c64-f128-k3-s1"}
                ],
            },
            "flash_attention": {
                "task_id": (
                    "tilelang_flash_attention_b1_h32_s2048_d64_rtx3090_large"
                ),
                "rationale": "Longer non-causal attention with four times the pairs.",
                "factory_arguments": {
                    "batch": 1,
                    "heads": 32,
                    "seq_len": 2048,
                    "dim": 64,
                    "is_causal": False,
                },
                "search_cases": [
                    {"case_id": "primary-b1-h32-s2048-d64"}
                ],
                "final_cases": [
                    {"case_id": "final-b1-h32-s2048-d64"}
                ],
            },
            "fused_add_rms_norm": {
                "task_id": (
                    "tilelang_fused_add_rms_norm_r16384_h4096_rtx3090_large"
                ),
                "rationale": "Double-row fused residual-add RMSNorm workload.",
                "factory_arguments": {
                    "rows": 16384,
                    "hidden_size": 4096,
                    "epsilon": 1e-6,
                },
                "search_cases": [
                    {"case_id": "primary-r16384-h4096"}
                ],
                "final_cases": [
                    {"case_id": "final-r16384-h4096"}
                ],
            },
        },
    },
    "special": {
        "suite_id": "related_system_special_shapes",
        "title": "Related-System Baselines: Special Schedule-Sensitive Shapes",
        "description": (
            "Rectangular GEMM, wider normalization, channel-heavy Conv2D, and "
            "longer causal attention. Search and final validation use the same "
            "shape."
        ),
        "output_directory": "related_system_baselines_{budget}_special",
        "workloads": {
            "matmul": {
                "task_id": "tilelang_matmul_m4096_n1024_k4096_rtx3090_special",
                "rationale": "Tall rectangular GEMM with a different aspect ratio.",
                "factory_arguments": {"m": 4096, "n": 1024, "k": 4096},
                "search_cases": [
                    {"case_id": "primary-m4096-n1024-k4096"}
                ],
                "final_cases": [
                    {"case_id": "final-m4096-n1024-k4096"}
                ],
            },
            "rms_norm": {
                "task_id": "tilelang_rms_norm_r4096_h8192_rtx3090_special",
                "rationale": "Wider reduction makes block and thread choices visible.",
                "factory_arguments": {
                    "rows": 4096,
                    "hidden_size": 8192,
                    "epsilon": 1e-6,
                },
                "search_cases": [
                    {"case_id": "primary-r4096-h8192"}
                ],
                "final_cases": [
                    {"case_id": "final-r4096-h8192"}
                ],
            },
            "conv2d": {
                "task_id": (
                    "tilelang_conv2d_n32_h28_w28_c128_f256_k3_rtx3090_special"
                ),
                "rationale": "Later-stage channel-heavy 3x3 convolution.",
                "factory_arguments": {
                    "batch": 32,
                    "in_height": 28,
                    "in_width": 28,
                    "in_channels": 128,
                    "out_channels": 256,
                    "kernel_size": 3,
                    "stride": 1,
                    "dilation": 1,
                    "padding": 1,
                },
                "search_cases": [
                    {"case_id": "primary-n32-h28-w28-c128-f256-k3-s1"}
                ],
                "final_cases": [
                    {"case_id": "final-n32-h28-w28-c128-f256-k3-s1"}
                ],
            },
            "flash_attention": {
                "task_id": (
                    "tilelang_flash_attention_b1_h16_s3072_d64_causal_rtx3090_special"
                ),
                "rationale": "Long causal sequence stresses tiling and pipeline depth.",
                "factory_arguments": {
                    "batch": 1,
                    "heads": 16,
                    "seq_len": 3072,
                    "dim": 64,
                    "is_causal": True,
                },
                "search_cases": [
                    {"case_id": "primary-b1-h16-s3072-d64-causal"}
                ],
                "final_cases": [
                    {"case_id": "final-b1-h16-s3072-d64-causal"}
                ],
            },
            "fused_add_rms_norm": {
                "task_id": (
                    "tilelang_fused_add_rms_norm_r4096_h8192_rtx3090_special"
                ),
                "rationale": "Wider fused reduction preserves retention headroom.",
                "factory_arguments": {
                    "rows": 4096,
                    "hidden_size": 8192,
                    "epsilon": 1e-6,
                },
                "search_cases": [
                    {"case_id": "primary-r4096-h8192"}
                ],
                "final_cases": [
                    {"case_id": "final-r4096-h8192"}
                ],
            },
        },
    },
}

RELATED_STYLES: Tuple[RelatedStyle, ...] = (
    RelatedStyle("kernelagent", "KernelAgent style"),
    RelatedStyle("kernelevolve", "KernelEvolve style"),
    RelatedStyle("kernelbench", "KernelBench style"),
    RelatedStyle("avo", "AVO style"),
    RelatedStyle("tilefoundry", "TileFoundry style"),
)
NATIVE_STYLE = RelatedStyle("native", "Full system")
ALL_STYLES: Tuple[RelatedStyle, ...] = (NATIVE_STYLE,) + RELATED_STYLES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run related-system style comparisons on the five kernel tasks. "
            "The default is 30 treatments: the native full system plus five "
            "proxy styles across five kernels."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Result directory. By default each named shape configuration uses "
            "its own directory under results/final_eval."
        ),
    )
    parser.add_argument(
        "--only-kernel",
        choices=tuple(kernel.key for kernel in KERNELS),
        action="append",
        help="Run only this kernel; repeat to select several.",
    )
    parser.add_argument(
        "--only-style",
        choices=tuple(style.key for style in ALL_STYLES),
        action="append",
        help="Run only this style; repeat to select several.",
    )
    native_group = parser.add_mutually_exclusive_group()
    native_group.add_argument(
        "--include-native",
        dest="include_native",
        action="store_true",
        help=(
            "Include this project's native full system before the five proxies "
            "(the default; retained for command compatibility)."
        ),
    )
    native_group.add_argument(
        "--exclude-native",
        dest="include_native",
        action="store_false",
        help="Run only the five related-system proxy styles (25 treatments).",
    )
    parser.set_defaults(include_native=True)
    shape_group = parser.add_mutually_exclusive_group()
    shape_group.add_argument(
        "--shape-config",
        choices=tuple(SHAPE_CONFIGS),
        default=DEFAULT_SHAPE_CONFIG,
        help=(
            "Built-in workload family: basic (2048-class defaults), large "
            "(longer stable timings), or special (schedule-sensitive shapes)."
        ),
    )
    shape_group.add_argument(
        "--workload-suite",
        type=Path,
        help=(
            "Custom JSON workload-family overrides. This is an escape hatch for "
            "new shapes; named experiments should use --shape-config."
        ),
    )
    parser.add_argument(
        "--max-search-seconds",
        type=float,
        default=DEFAULT_MAX_SEARCH_SECONDS,
        help=(
            "Controller search budget per treatment; default 900 seconds "
            "(15 minutes). Final validation runs after this deadline."
        ),
    )
    parser.add_argument(
        "--budget-mode",
        choices=("fixed-time", "rounds"),
        default="fixed-time",
        help=(
            "Use wall-clock time as the primary stopping condition, or preserve "
            "the task's normal round budget."
        ),
    )
    parser.add_argument(
        "--time-budget-round-ceiling",
        type=int,
        default=DEFAULT_TIME_BUDGET_ROUND_CEILING,
        help=(
            "Safety round ceiling materialized for fixed-time runs; default 128 "
            "is intentionally above what a 30-minute hosted-model run can finish."
        ),
    )
    parser.add_argument(
        "--measurement-repeats",
        type=int,
        default=DEFAULT_MEASUREMENT_REPEATS,
        help="CUDA Event measurement repeats per search candidate; default 3.",
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
    include_native: bool = True,
) -> List[Treatment]:
    """Return the deterministic kernel-major comparison matrix."""

    kernel_filter = set(only_kernels or ())
    style_filter = set(only_styles or ())
    styles = ALL_STYLES if include_native or "native" in style_filter else RELATED_STYLES
    return [
        Treatment(kernel, style)
        for kernel in KERNELS
        if not kernel_filter or kernel.key in kernel_filter
        for style in styles
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
    search_until_time_budget: bool = False,
    resume: bool = False,
    repository_root: Path = REPOSITORY_ROOT,
    task_path: Optional[Path] = None,
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
        str(task_path or treatment.kernel.task_path(repository_root)),
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
    if search_until_time_budget:
        command.append("--search-until-time-budget")
    if resume:
        command.append("--resume")
    return command


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_arguments(args)
    workload_suite = (
        _load_workload_suite(args.workload_suite)
        if args.workload_suite is not None
        else _builtin_workload_suite(args.shape_config)
    )
    shape_config_name = (
        "custom" if args.workload_suite is not None else args.shape_config
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else _default_output_root(
            workload_suite,
            args.max_search_seconds if args.budget_mode == "fixed-time" else 0.0,
        )
    )
    treatments = selected_treatments(
        args.only_kernel,
        args.only_style,
        include_native=args.include_native,
    )
    if not treatments:
        raise SystemExit("No treatments matched the requested filters")
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    task_paths, task_payloads = _prepare_treatment_tasks(
        REPOSITORY_ROOT,
        output_root,
        treatments,
        workload_suite,
        dry_run=args.dry_run,
        measurement_repeats=args.measurement_repeats,
        round_ceiling=(
            args.time_budget_round_ceiling
            if args.budget_mode == "fixed-time"
            else None
        ),
    )
    workloads = _load_workload_metadata(
        REPOSITORY_ROOT,
        treatments,
        task_payloads=task_payloads,
        task_paths=task_paths,
    )
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
    print(
        "Budget mode: %s; measurement repeats: %d"
        % (args.budget_mode, args.measurement_repeats),
        flush=True,
    )
    print(
        "Shape config: %s (%s)"
        % (shape_config_name, workload_suite["suite_id"]),
        flush=True,
    )
    print("Output root: %s" % output_root, flush=True)
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
    rows: List[Dict[str, Any]] = []
    failure_code = 0
    for index, treatment in enumerate(treatments, start=1):
        output = output_root / treatment.relative_output
        summary_path = output / "summary.json"
        if summary_path.is_file():
            summary = _read_json(summary_path)
            expected_final_validations = int(
                task_payloads[treatment.kernel.key]
                .get("budget", {})
                .get("final_validation_candidates", 0)
                or 0
            )
            if _summary_uses_current_timing_protocol(
                summary,
                budget_mode=args.budget_mode,
                expected_final_validations=expected_final_validations,
                expected_max_search_seconds=args.max_search_seconds,
            ):
                status = _completed_treatment_status(
                    summary, budget_mode=args.budget_mode, reused=True
                )
                print(
                    "\n[%d/%d] %s / %s already complete (%s); reusing %s"
                    % (
                        index,
                        len(treatments),
                        treatment.kernel.title,
                        treatment.style.title,
                        status,
                        summary_path,
                    ),
                    flush=True,
                )
                rows.append(_summary_row(treatment, summary, status))
                if status == "ended-early":
                    failure_code = failure_code or 1
                continue
            print(
                "\n[%d/%d] %s / %s has a legacy summary without "
                "post-budget final validation; resuming it once to finalize."
                % (
                    index,
                    len(treatments),
                    treatment.kernel.title,
                    treatment.style.title,
                ),
                flush=True,
            )

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
            search_until_time_budget=args.budget_mode == "fixed-time",
            resume=resume,
            task_path=task_paths[(treatment.kernel.key, treatment.style.key)],
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
            _write_reports(
                output_root,
                treatments,
                workloads,
                rows,
                args.max_search_seconds,
                workload_suite,
                budget_mode=args.budget_mode,
                measurement_repeats=args.measurement_repeats,
            )
            if not args.continue_on_error:
                print(
                    "Treatment failed. Fix the issue and rerun; completed runs "
                    "will be reused and this run will resume.",
                    flush=True,
                )
                return completed.returncode
            continue

        if summary_path.is_file():
            summary = _read_json(summary_path)
            status = _completed_treatment_status(
                summary, budget_mode=args.budget_mode, reused=False
            )
            rows.append(_summary_row(treatment, summary, status))
            if status == "ended-early":
                failure_code = failure_code or 1
        else:
            failure_code = failure_code or 1
            rows.append(
                _summary_row(
                    treatment,
                    {"error": "child exited successfully without summary.json"},
                    "failed",
                )
            )
        _write_reports(
            output_root,
            treatments,
            workloads,
            rows,
            args.max_search_seconds,
            workload_suite,
            budget_mode=args.budget_mode,
            measurement_repeats=args.measurement_repeats,
        )

    if args.dry_run:
        print("\nDry run complete; no output was created.", flush=True)
        return 0

    _write_reports(
        output_root,
        treatments,
        workloads,
        rows,
        args.max_search_seconds,
        workload_suite,
        budget_mode=args.budget_mode,
        measurement_repeats=args.measurement_repeats,
    )
    print("\nSuite summary: %s" % (output_root / "suite_summary.md"), flush=True)
    if failure_code:
        print(
            "Some treatments failed or ended before the fixed-time budget; "
            "inspect suite_summary.md."
        )
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
        "measurement_repeats",
        "time_budget_round_ceiling",
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
    if args.budget_mode == "fixed-time" and args.max_search_seconds <= 0:
        raise SystemExit(
            "fixed-time budget mode requires a positive --max-search-seconds"
        )
    if not args.api_url or not args.api_model:
        raise SystemExit(
            "Set KERNEL_OPT_API_URL and KERNEL_OPT_API_MODEL, or pass "
            "--api-url and --api-model."
        )
    if not args.dry_run and not os.environ.get(args.api_key_env):
        raise SystemExit("API key environment variable %s is not set" % args.api_key_env)
    if args.workload_suite is not None and not args.workload_suite.expanduser().is_file():
        raise SystemExit("Workload suite does not exist: %s" % args.workload_suite)
    for kernel in KERNELS:
        for path in (kernel.source_path(), kernel.task_path()):
            if not path.is_file():
                raise SystemExit("Required example file does not exist: %s" % path)


def _builtin_workload_suite(name: str) -> Dict[str, Any]:
    if name not in SHAPE_CONFIGS:
        raise SystemExit("Unknown shape configuration: %s" % name)
    value = copy.deepcopy(SHAPE_CONFIGS[name])
    value["config_name"] = name
    return _validate_workload_suite(value, config_path="builtin:%s" % name)


def _load_workload_suite(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    value = _read_json(resolved)
    value["config_name"] = "custom"
    return _validate_workload_suite(value, config_path=str(resolved))


def _validate_workload_suite(
    value: Mapping[str, Any], *, config_path: str
) -> Dict[str, Any]:
    value = copy.deepcopy(dict(value))
    suite_id = value.get("suite_id")
    workloads = value.get("workloads")
    if not isinstance(suite_id, str) or not suite_id.strip():
        raise SystemExit("Workload suite requires a non-empty suite_id")
    if not isinstance(workloads, dict):
        raise SystemExit("Workload suite requires a workloads object")
    expected = {kernel.key for kernel in KERNELS}
    missing = expected - set(workloads)
    unknown = set(workloads) - expected
    if missing or unknown:
        raise SystemExit(
            "Workload suite kernel mismatch; missing=%s unknown=%s"
            % (sorted(missing), sorted(unknown))
        )
    for kernel_key, raw_override in workloads.items():
        if not isinstance(raw_override, dict):
            raise SystemExit("Workload override %s must be an object" % kernel_key)
        task_id = raw_override.get("task_id")
        arguments = raw_override.get("factory_arguments")
        search_cases = raw_override.get("search_cases")
        final_cases = raw_override.get("final_cases")
        if not isinstance(task_id, str) or not task_id.strip():
            raise SystemExit("Workload override %s requires task_id" % kernel_key)
        if not isinstance(arguments, dict) or not arguments:
            raise SystemExit(
                "Workload override %s requires factory_arguments" % kernel_key
            )
        for name, cases in (("search_cases", search_cases), ("final_cases", final_cases)):
            if not isinstance(cases, list) or not cases:
                raise SystemExit(
                    "Workload override %s requires non-empty %s"
                    % (kernel_key, name)
                )
            for case in cases:
                if not isinstance(case, dict) or not str(case.get("case_id", "")).strip():
                    raise SystemExit(
                        "Every %s.%s entry requires case_id" % (kernel_key, name)
                    )
                case_arguments = case.get("factory_arguments")
                if case_arguments is not None and not isinstance(case_arguments, dict):
                    raise SystemExit(
                        "%s.%s factory_arguments must be an object"
                        % (kernel_key, case["case_id"])
                    )
        case_ids = [
            str(case["case_id"]) for case in list(search_cases) + list(final_cases)
        ]
        if len(case_ids) != len(set(case_ids)):
            raise SystemExit("Workload override %s has duplicate case IDs" % kernel_key)
    output_directory = value.get("output_directory")
    if output_directory is not None:
        output_path = Path(str(output_directory))
        if (
            output_path.is_absolute()
            or len(output_path.parts) != 1
            or output_path.name in ("", ".", "..")
        ):
            raise SystemExit(
                "Workload suite output_directory must be one directory name"
            )
    value["config_path"] = config_path
    return value


def _search_budget_label(max_search_seconds: float) -> str:
    if max_search_seconds <= 0:
        return "rounds"
    minutes = max_search_seconds / 60.0
    if minutes.is_integer():
        return "%dm" % int(minutes)
    return "%ds" % int(round(max_search_seconds))


def _default_output_root(
    workload_suite: Mapping[str, Any],
    max_search_seconds: float = DEFAULT_MAX_SEARCH_SECONDS,
) -> Path:
    directory = workload_suite.get("output_directory") or workload_suite["suite_id"]
    directory = str(directory).replace(
        "{budget}", _search_budget_label(max_search_seconds)
    )
    return (
        REPOSITORY_ROOT / "results" / "final_eval" / directory
    ).resolve()


def _prepare_treatment_tasks(
    repository_root: Path,
    output_root: Path,
    treatments: Sequence[Treatment],
    workload_suite: Optional[Mapping[str, Any]],
    *,
    dry_run: bool,
    measurement_repeats: Optional[int] = None,
    round_ceiling: Optional[int] = None,
) -> tuple[Dict[tuple[str, str], Path], Dict[str, Dict[str, Any]]]:
    task_paths: Dict[tuple[str, str], Path] = {}
    task_payloads: Dict[str, Dict[str, Any]] = {}
    for treatment in treatments:
        kernel = treatment.kernel
        treatment_key = (kernel.key, treatment.style.key)
        base_task_path = kernel.task_path(repository_root)
        if workload_suite is None:
            task_paths[treatment_key] = base_task_path
            task_payloads.setdefault(kernel.key, _read_json(base_task_path))
            continue

        override = dict(workload_suite["workloads"][kernel.key])
        treatment_output = output_root / treatment.relative_output
        task = _derive_task(
            _read_json(base_task_path),
            kernel,
            treatment,
            override,
            workload_suite,
            repository_root,
            treatment_output,
            measurement_repeats=measurement_repeats,
            round_ceiling=round_ceiling,
        )
        task_path = (
            output_root
            / "_tasks"
            / kernel.key
            / (treatment.style.key + ".json")
        )
        task_paths[treatment_key] = task_path
        task_payloads.setdefault(kernel.key, task)
        if not dry_run:
            _write_immutable_json(task_path, task)
    return task_paths, task_payloads


def _derive_task(
    base_task: Mapping[str, Any],
    kernel: KernelWorkload,
    treatment: Treatment,
    override: Mapping[str, Any],
    workload_suite: Mapping[str, Any],
    repository_root: Path,
    treatment_output: Path,
    *,
    measurement_repeats: Optional[int] = None,
    round_ceiling: Optional[int] = None,
) -> Dict[str, Any]:
    task = copy.deepcopy(dict(base_task))
    base_task_id = str(task.get("task_id"))
    task["task_id"] = str(override["task_id"])
    if override.get("description"):
        task["description"] = str(override["description"])
    workload = dict(task.get("workload") or {})
    workload["factory_arguments"] = copy.deepcopy(
        dict(override["factory_arguments"])
    )
    task["workload"] = workload

    if round_ceiling is not None:
        budget = dict(task.get("budget") or {})
        budget["rounds"] = int(round_ceiling)
        task["budget"] = budget

    evaluator = dict(task.get("evaluator") or {})
    runtime = dict(evaluator.get("runtime") or {})
    runtime["search_cases"] = copy.deepcopy(list(override["search_cases"]))
    runtime["final_cases"] = copy.deepcopy(list(override["final_cases"]))
    runtime["profile_directory"] = str(
        (treatment_output / "ncu_profiles").resolve()
    )
    if measurement_repeats is not None:
        runtime["measurement_repeats"] = int(measurement_repeats)
    evaluator["runtime"] = runtime
    evaluator["working_directory"] = str(repository_root.resolve())
    task["evaluator"] = evaluator

    metadata = dict(task.get("metadata") or {})
    metadata["shape_suite"] = {
        "suite_id": workload_suite["suite_id"],
        "suite_title": workload_suite.get("title"),
        "config_path": workload_suite.get("config_path"),
        "base_task_id": base_task_id,
        "kernel": kernel.key,
        "style": treatment.style.key,
        "rationale": override.get("rationale"),
    }
    metadata["comparison_budget"] = {
        "mode": "fixed-time" if round_ceiling is not None else "rounds",
        "round_ceiling": round_ceiling,
        "measurement_repeats": measurement_repeats,
    }
    task["metadata"] = metadata
    return task


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        archived = _read_json(path)
        if archived != value and not _tasks_differ_only_in_suite_provenance(
            archived, value
        ):
            raise SystemExit(
                "Existing effective task differs from the requested workload "
                "suite; choose a fresh output root: %s" % path
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tasks_differ_only_in_suite_provenance(
    archived: Mapping[str, Any], requested: Mapping[str, Any]
) -> bool:
    def without_provenance(value: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = copy.deepcopy(dict(value))
        metadata = dict(normalized.get("metadata") or {})
        shape_suite = dict(metadata.get("shape_suite") or {})
        for name in ("config_path", "rationale", "suite_title"):
            shape_suite.pop(name, None)
        metadata["shape_suite"] = shape_suite
        normalized["metadata"] = metadata
        return normalized

    return without_provenance(archived) == without_provenance(requested)


def _load_workload_metadata(
    repository_root: Path,
    treatments: Sequence[Treatment],
    *,
    task_payloads: Optional[Mapping[str, Mapping[str, Any]]] = None,
    task_paths: Optional[Mapping[tuple[str, str], Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    selected = {item.kernel.key for item in treatments}
    metadata: Dict[str, Dict[str, Any]] = {}
    for kernel in KERNELS:
        if kernel.key not in selected:
            continue
        representative = next(
            item for item in treatments if item.kernel.key == kernel.key
        )
        task_path = (
            task_paths[(kernel.key, representative.style.key)]
            if task_paths is not None
            else kernel.task_path(repository_root)
        )
        task = (
            dict(task_payloads[kernel.key])
            if task_payloads is not None
            else _read_json(task_path)
        )
        budget = dict(task.get("budget") or {})
        runtime = dict(dict(task.get("evaluator") or {}).get("runtime") or {})
        task_metadata = dict(task.get("metadata") or {})
        shape_suite = dict(task_metadata.get("shape_suite") or {})
        comparison_budget = dict(task_metadata.get("comparison_budget") or {})
        rounds = int(budget.get("rounds", 0))
        proposals = int(budget.get("proposals_per_round", 0))
        repairs = int(budget.get("max_repairs_per_round", 0))
        metadata[kernel.key] = {
            "kernel": kernel.key,
            "title": kernel.title,
            "source": str(kernel.source_path(repository_root)),
            "task": str(task_path),
            "task_id": task.get("task_id"),
            "base_task_id": shape_suite.get("base_task_id"),
            "shape_suite_id": shape_suite.get("suite_id"),
            "shape_rationale": shape_suite.get("rationale"),
            "factory_arguments": dict(
                dict(task.get("workload") or {}).get("factory_arguments") or {}
            ),
            "search_cases": copy.deepcopy(list(runtime.get("search_cases") or [])),
            "final_cases": copy.deepcopy(list(runtime.get("final_cases") or [])),
            "rounds": rounds,
            "proposals_per_round": proposals,
            "max_repairs_per_round": repairs,
            "measurement_repeats": runtime.get("measurement_repeats"),
            "comparison_budget_mode": comparison_budget.get("mode"),
            "round_ceiling": comparison_budget.get("round_ceiling"),
            "candidate_graph_upper_bound": 1 + rounds * (proposals + repairs),
        }
    return metadata


def _summary_row(
    treatment: Treatment, summary: Mapping[str, Any], status: str
) -> Dict[str, Any]:
    ledger = dict(summary.get("cost_ledger") or {})
    api = dict(ledger.get("api") or {})
    evaluator = dict(ledger.get("evaluator") or {})
    legacy_elapsed = summary.get("elapsed_seconds")
    search_elapsed = summary.get("search_elapsed_seconds")
    total_elapsed = summary.get("total_elapsed_seconds")
    return {
        "kernel": treatment.kernel.key,
        "kernel_title": treatment.kernel.title,
        "style": treatment.style.key,
        "style_title": treatment.style.title,
        "status": status,
        "seed_latency_ms": summary.get("seed_latency_ms"),
        "final_seed_latency_ms": summary.get("final_seed_latency_ms"),
        "search_best_latency_ms": summary.get("search_best_latency_ms"),
        "best_latency_ms": summary.get("best_latency_ms"),
        "speedup_over_seed": summary.get("speedup_over_seed"),
        "completed_rounds": summary.get("completed_rounds"),
        "generated_candidates": summary.get("generated_candidates"),
        "measured_candidates": summary.get("measured_candidates"),
        "profile_calls": summary.get("profile_calls"),
        "final_validation_calls": summary.get("final_validation_calls"),
        "search_elapsed_seconds": (
            legacy_elapsed if search_elapsed is None else search_elapsed
        ),
        "final_validation_seconds": summary.get("final_validation_seconds", 0.0),
        "total_elapsed_seconds": (
            legacy_elapsed if total_elapsed is None else total_elapsed
        ),
        "elapsed_seconds": legacy_elapsed,
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


def _completed_treatment_status(
    summary: Mapping[str, Any], *, budget_mode: str, reused: bool
) -> str:
    if budget_mode == "fixed-time" and not (
        summary.get("termination_reason") == "time-budget"
        and bool(summary.get("time_budget_exhausted"))
    ):
        return "ended-early"
    return "reused" if reused else "completed"


def _summary_uses_current_timing_protocol(
    summary: Mapping[str, Any],
    *,
    budget_mode: str,
    expected_final_validations: int,
    expected_max_search_seconds: Optional[float] = None,
) -> bool:
    if budget_mode != "fixed-time":
        return True
    required_timing_fields = (
        "search_elapsed_seconds",
        "final_validation_seconds",
        "total_elapsed_seconds",
    )
    if any(summary.get(name) is None for name in required_timing_fields):
        return False
    if expected_final_validations > 0 and int(
        summary.get("final_validation_calls", 0) or 0
    ) <= 0:
        return False
    if expected_max_search_seconds is not None:
        archived_budget = summary.get("max_search_seconds")
        if not isinstance(archived_budget, (int, float)) or not math.isclose(
            float(archived_budget),
            float(expected_max_search_seconds),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            return False
    return True


def _apply_shared_seed_references(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Use one robust final-seed latency for every style of each kernel."""

    grouped: Dict[str, List[Tuple[str, float]]] = {}
    for row in rows:
        if row.get("status") == "failed":
            continue
        value = row.get("final_seed_latency_ms")
        if value is None:
            value = row.get("seed_latency_ms")
        if (
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0
        ):
            grouped.setdefault(str(row["kernel"]), []).append(
                (str(row["style"]), float(value))
            )

    references: Dict[str, Dict[str, Any]] = {}
    for kernel, samples in grouped.items():
        latency = float(statistics.median(value for _style, value in samples))
        references[kernel] = {
            "policy": "median-of-final-seed-latencies",
            "latency_ms": latency,
            "sample_count": len(samples),
            "styles": [style for style, _value in samples],
            "treatment_seed_latencies_ms": {
                style: value for style, value in samples
            },
        }

    normalized: List[Dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        row["treatment_seed_latency_ms"] = row.get("seed_latency_ms")
        row["treatment_speedup_over_seed"] = row.get("speedup_over_seed")
        reference = references.get(str(row.get("kernel")))
        best = row.get("best_latency_ms")
        if reference is not None:
            shared = float(reference["latency_ms"])
            row["shared_seed_latency_ms"] = shared
            row["shared_seed_sample_count"] = int(reference["sample_count"])
            row["seed_latency_ms"] = shared
            row["speedup_over_seed"] = (
                shared / float(best)
                if isinstance(best, (int, float)) and float(best) > 0
                else None
            )
        else:
            row["shared_seed_latency_ms"] = None
            row["shared_seed_sample_count"] = 0
            row["speedup_over_seed"] = None
        normalized.append(row)
    return normalized, references


def _write_reports(
    output_root: Path,
    treatments: Sequence[Treatment],
    workloads: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    max_search_seconds: float,
    workload_suite: Optional[Mapping[str, Any]] = None,
    *,
    budget_mode: str = "rounds",
    measurement_repeats: Optional[int] = None,
) -> None:
    normalized_rows, shared_seed_references = _apply_shared_seed_references(rows)
    status_counts: Dict[str, int] = {}
    for row in normalized_rows:
        status = str(row.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    graph_bound = sum(
        int(workloads[item.kernel.key]["candidate_graph_upper_bound"])
        for item in treatments
    )
    style_names = list(dict.fromkeys(item.style.key for item in treatments))
    includes_native = "native" in style_names
    suite_metadata = None
    if workload_suite is not None:
        suite_metadata = {
            "suite_id": workload_suite.get("suite_id"),
            "config_name": workload_suite.get("config_name"),
            "title": workload_suite.get("title"),
            "description": workload_suite.get("description"),
            "config_path": workload_suite.get("config_path"),
        }
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "claim": (
            (
                "This project's native full system plus five common-harness "
                "control-flow style emulations."
                if includes_native
                else "Common-harness control-flow style emulations."
            )
            + " The proxies are not exact reimplementations and do not reproduce "
            "published results."
        ),
        "output_root": str(output_root),
        "planned_treatments": len(treatments),
        "agent_workers": 1,
        "budget_mode": budget_mode,
        "measurement_repeats": measurement_repeats,
        "shared_seed_policy": "median-of-final-seed-latencies",
        "shared_seed_references": shared_seed_references,
        "time_budget_round_ceiling": (
            max(
                int(item.get("round_ceiling") or 0)
                for item in workloads.values()
            )
            if budget_mode == "fixed-time" and workloads
            else None
        ),
        "max_search_seconds_per_treatment": max_search_seconds,
        "maximum_configured_search_seconds": (
            max_search_seconds * len(treatments) if max_search_seconds > 0 else None
        ),
        "candidate_graph_upper_bound": graph_bound,
        "status_counts": status_counts,
        "styles": style_names,
        "workload_suite": suite_metadata,
        "workloads": [dict(workloads[key]) for key in workloads],
        "runs": normalized_rows,
    }
    (output_root / "suite_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "suite_summary.md").write_text(
        _suite_markdown(payload), encoding="utf-8"
    )


def _suite_markdown(payload: Mapping[str, Any]) -> str:
    workload_suite = payload.get("workload_suite") or {}
    title = workload_suite.get("title") or "Related-System Baseline Suite"
    workload_heading = (
        "Configured Shape Family" if workload_suite else "Unchanged Workloads"
    )
    lines = [
        "# %s" % title,
        "",
        str(payload["claim"]),
        "",
        "## Suite Configuration",
        "",
        "| Setting | Value |",
        "| --- | ---: |",
        "| Planned treatments | %s |" % payload["planned_treatments"],
        "| Agent workers | 1 |",
        "| Budget mode | `%s` |" % payload.get("budget_mode"),
        "| Measurement repeats | %s |"
        % _format_value(payload.get("measurement_repeats")),
        "| Shared seed policy | `%s` |" % payload.get("shared_seed_policy"),
        "| Time-budget round ceiling | %s |"
        % _format_value(payload.get("time_budget_round_ceiling")),
        "| Search cap per treatment | %s seconds |"
        % _format_value(payload["max_search_seconds_per_treatment"]),
        "| Candidate graph upper bound | %s nodes |"
        % payload["candidate_graph_upper_bound"],
    ]
    if workload_suite:
        lines.extend(
            [
                "| Shape config | `%s` |"
                % workload_suite.get("config_name"),
                "| Workload suite | `%s` |" % workload_suite.get("suite_id"),
                "| Workload config | `%s` |" % workload_suite.get("config_path"),
            ]
        )
    lines.extend(
        [
            "",
            "## %s" % workload_heading,
            "",
            "| Kernel | Task | Primary factory arguments | Graph bound per style |",
            "| --- | --- | --- | ---: |",
        ]
    )
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
    if workload_suite:
        lines.extend(["", "### Search and Final Cases", ""])
        for workload in payload["workloads"]:
            lines.extend(
                [
                    "**%s**" % workload["title"],
                    "",
                    "- Rationale: %s"
                    % (workload.get("shape_rationale") or "Not specified."),
                    "- Search: %s"
                    % _format_cases(
                        workload.get("search_cases") or [],
                        workload["factory_arguments"],
                    ),
                    "- Final: %s"
                    % _format_cases(
                        workload.get("final_cases") or [],
                        workload["factory_arguments"],
                    ),
                    "",
                ]
            )
    lines.extend(
        [
            "",
            "## Shared Seed References",
            "",
            "| Kernel | Shared seed (ms) | Final-seed measurements |",
            "| --- | ---: | ---: |",
        ]
    )
    references = dict(payload.get("shared_seed_references") or {})
    for workload in payload["workloads"]:
        reference = dict(references.get(workload["kernel"]) or {})
        lines.append(
            "| {title} | {latency} | {samples} |".format(
                title=workload["title"],
                latency=_format_value(reference.get("latency_ms")),
                samples=_format_value(reference.get("sample_count")),
            )
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Kernel | Style | Status | Shared seed (ms) | Search best (ms) | Exported best (ms) | Speedup | Rounds | Generated | Measured | NCU | Final checks | Search (s) | Final (s) | Total (s) | Termination |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["runs"]:
        lines.append(
            "| {kernel_title} | {style_title} | {status} | {seed} | "
            "{search_best} | {best} | {speedup} | {rounds} | {generated} | "
            "{measured} | {ncu} | {final_checks} | {search_elapsed} | "
            "{final_elapsed} | {total_elapsed} | {termination} |".format(
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
                final_checks=_format_value(row.get("final_validation_calls")),
                search_elapsed=_format_value(row.get("search_elapsed_seconds")),
                final_elapsed=_format_value(row.get("final_validation_seconds")),
                total_elapsed=_format_value(row.get("total_elapsed_seconds")),
                termination=row.get("termination_reason") or row.get("error") or "-",
            )
        )
    source_note = (
        "Each row uses the original kernel source and a materialized task from "
        "the declared workload suite. Base task files are not modified."
        if workload_suite
        else "Each row uses the original source and task JSON. The runner changes "
        "only `--baseline-style` and API/time controls; it does not rewrite shapes."
    )
    lines.extend(
        [
            "",
            source_note,
            "",
            "Every style for a kernel uses the same shared seed latency: the "
            "median of all available fresh final-seed measurements for that "
            "kernel. Per-treatment seed latency and speedup remain available "
            "in `suite_summary.json` as diagnostic fields, but the displayed "
            "speedup always uses the shared reference.",
            "",
            "For `fixed-time` runs, only a `time-budget` termination is a valid "
            "equal-budget result. The controller exports the best verified "
            "incumbent at its safe stopping checkpoint, freezes `Search (s)`, then "
            "runs separate final validation outside the search budget. `Final (s)` "
            "and `Total (s)` expose that additional verification cost. Rerunning the "
            "suite reuses current completed rows, upgrades legacy fixed-time rows, "
            "and resumes interrupted rows.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_cases(
    cases: Sequence[Mapping[str, Any]], primary: Mapping[str, Any]
) -> str:
    rendered = []
    for case in cases:
        arguments = dict(primary)
        arguments.update(dict(case.get("factory_arguments") or {}))
        rendered.append(
            "`%s=%s`"
            % (
                case.get("case_id"),
                json.dumps(arguments, sort_keys=True, separators=(",", ":")),
            )
        )
    return "; ".join(rendered) if rendered else "-"


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
