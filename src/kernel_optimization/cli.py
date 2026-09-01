"""Command-line entry point for hosted-API kernel source optimization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

from .archive import ArtifactStore
from .baseline_styles import BASELINE_STYLE_NAMES, resolve_baseline_style
from .controller import OptimizationController, resolve_evaluation_policies
from .factory import create_backend
from .generators import (
    ApiGeneratorConfig,
    OpenAICompatibleGenerator,
    ParallelCandidateGenerator,
)
from .manifest import collect_environment_manifest
from .progress import ProgressReporter
from .prompt_compression import (
    COMPRESSION_VERSION,
    DEFAULT_HISTORY_LIMIT,
    DEFAULT_LESSON_LIMIT,
    METADATA_COMPRESSION_POLICIES,
)
from .schema import Candidate, TaskSpec
from .source_validation import SourceValidator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize a GPU kernel source with hosted API proposals."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Input TileLang Python kernel source. This file is never modified.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="JSON task contract describing semantics, workload, target, and evaluator.",
    )
    parser.add_argument(
        "--output",
        help="Fresh artifact directory. Defaults to results/<task>_<UTC timestamp>.",
    )
    parser.add_argument(
        "--baseline-style",
        choices=BASELINE_STYLE_NAMES,
        default="native",
        help=(
            "Run the native controller or an auditable style emulation of a "
            "related system. This does not reproduce that system's implementation."
        ),
    )
    parser.add_argument(
        "--api-url",
        help="OpenAI-compatible chat-completions URL, or KERNEL_OPT_API_URL.",
    )
    parser.add_argument(
        "--api-model", help="Hosted model identifier, or KERNEL_OPT_API_MODEL."
    )
    parser.add_argument(
        "--api-key-env",
        default="KERNEL_OPT_API_KEY",
        help="Environment variable containing the hosted API key.",
    )
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument(
        "--api-temperature",
        type=float,
        help="Optional sampling temperature. Omitted by default for model compatibility.",
    )
    parser.add_argument("--api-max-output-tokens", type=int, default=12000)
    parser.add_argument(
        "--api-planner-max-output-tokens",
        type=int,
        default=2000,
        help="Small output budget for the strategy-allocation request.",
    )
    parser.add_argument(
        "--api-max-input-tokens",
        type=int,
        default=60000,
        help=(
            "Conservative local input-token budget checked before network access. "
            "The configured prompt context is prepared before this check."
        ),
    )
    parser.add_argument(
        "--metadata-compression-policy",
        choices=METADATA_COMPRESSION_POLICIES,
        default=COMPRESSION_VERSION,
        help=(
            "Use the bounded key-metric prompt context, or disable compression "
            "for a controlled token-growth ablation."
        ),
    )
    parser.add_argument(
        "--metadata-history-limit",
        type=int,
        default=DEFAULT_HISTORY_LIMIT,
        help=(
            "Maximum compact history records offered to a compressed generation "
            "or repair prompt. The default preserves existing behavior."
        ),
    )
    parser.add_argument(
        "--metadata-lesson-limit",
        type=int,
        default=DEFAULT_LESSON_LIMIT,
        help=(
            "Maximum compact evidence lessons offered to a compressed generation "
            "or repair prompt. The default preserves existing behavior."
        ),
    )
    parser.add_argument(
        "--compressed-context-target-tokens",
        type=int,
        help=(
            "Optional input-token target for compressed generation and repair "
            "prompts. Oldest compact history and lowest-priority lessons are "
            "trimmed deterministically; omitted by default."
        ),
    )
    parser.add_argument(
        "--api-max-tokens-field",
        choices=("max_tokens", "max_completion_tokens"),
        default="max_completion_tokens",
    )
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--api-retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--agent-workers",
        type=int,
        default=1,
        help=(
            "Maximum concurrent hosted-API generation agents. GPU evaluation "
            "remains serialized; default 1 preserves the existing behavior."
        ),
    )
    parser.add_argument(
        "--no-json-response-format",
        action="store_true",
        help="Do not request the OpenAI-compatible JSON object response format.",
    )
    parser.add_argument(
        "--skip-api-preflight",
        action="store_true",
        help="Skip the tiny API request that normally runs before GPU evaluation.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run from --output checkpoints.",
    )
    parser.add_argument(
        "--incumbent-snapshot-interval-seconds",
        type=float,
        default=300.0,
        help=(
            "Wall-clock interval for best-verified-kernel snapshots; default "
            "300 seconds, or 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-search-seconds",
        type=float,
        default=0.0,
        help=(
            "Soft wall-clock limit for controller search, including seed "
            "evaluation; 0 disables the limit. In-flight API/compiler/GPU calls "
            "finish before the controller stops."
        ),
    )
    parser.add_argument(
        "--search-until-time-budget",
        action="store_true",
        help=(
            "Treat --max-search-seconds as the primary search budget. Empty "
            "candidate rounds continue until the deadline; the task must provide "
            "a sufficiently high round ceiling."
        ),
    )
    parser.add_argument(
        "--selection-policy",
        choices=("adaptive", "model-top", "random", "measure-all"),
        default="adaptive",
        help="Candidate promotion strategy. Use a fixed budget for equal-cost ablations.",
    )
    parser.add_argument(
        "--evaluation-policy",
        choices=("tilesight", "cuda-event", "ncu"),
        default="tilesight",
        help=(
            "Use TileSight promotion, measure every candidate with CUDA Event, "
            "or measure and NCU-profile every correct candidate."
        ),
    )
    parser.add_argument(
        "--tir-evidence-policy",
        choices=("auto", "visible", "hidden"),
        default="auto",
        help=(
            "Control whether successful TIR/TileSight performance evidence is "
            "included in AI prompts; auto hides it when TileSight is disabled."
        ),
    )
    parser.add_argument(
        "--promotions-per-round",
        type=int,
        help="Fix CUDA measurement promotions per round across search strategies.",
    )
    parser.add_argument(
        "--profile-policy",
        choices=("milestone", "every-round", "none"),
        default="milestone",
        help="NCU allocation policy used by this experiment.",
    )
    parser.add_argument(
        "--no-compiled-dedup",
        action="store_true",
        help="Measure candidates even when their compiled execution identity matches.",
    )
    parser.add_argument(
        "--structural-search-policy",
        choices=("enforce", "observe", "off"),
        default="enforce",
        help=(
            "Enforce, observe, or disable parent-relative AST novelty checks."
        ),
    )
    parser.add_argument(
        "--strategy-allocation-policy",
        choices=("ai-planned", "fixed", "hardware-adaptive", "unconstrained"),
        default="ai-planned",
        help=(
            "Use an AI plan, the legacy fixed portfolio, measured-reward adaptive "
            "allocation, or no explicit strategy slots."
        ),
    )
    parser.add_argument(
        "--strategy-plan-interval-rounds",
        type=int,
        help=(
            "Reuse an evidence-stable hosted strategy plan for this many rounds. "
            "Defaults to 3 for the native system and 1 for related-system styles."
        ),
    )
    parser.add_argument(
        "--api-input-price-per-million",
        type=float,
        help="Optional hosted-model input-token price used only for cost reporting.",
    )
    parser.add_argument(
        "--api-output-price-per-million",
        type=float,
        help="Optional hosted-model output-token price used only for cost reporting.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress lines.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    if args.agent_workers <= 0:
        raise SystemExit("--agent-workers must be positive")
    if args.promotions_per_round is not None and args.promotions_per_round <= 0:
        raise SystemExit("--promotions-per-round must be positive")
    if (
        args.strategy_plan_interval_rounds is not None
        and args.strategy_plan_interval_rounds <= 0
    ):
        raise SystemExit("--strategy-plan-interval-rounds must be positive")
    if (
        not math.isfinite(args.incumbent_snapshot_interval_seconds)
        or args.incumbent_snapshot_interval_seconds < 0
    ):
        raise SystemExit(
            "--incumbent-snapshot-interval-seconds must be finite and non-negative"
        )
    if not math.isfinite(args.max_search_seconds) or args.max_search_seconds < 0:
        raise SystemExit("--max-search-seconds must be finite and non-negative")
    if args.search_until_time_budget and args.max_search_seconds <= 0:
        raise SystemExit(
            "--search-until-time-budget requires a positive --max-search-seconds"
        )
    for name in ("api_input_price_per_million", "api_output_price_per_million"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise SystemExit("--%s must be non-negative" % name.replace("_", "-"))
    if (args.api_input_price_per_million is None) != (
        args.api_output_price_per_million is None
    ):
        raise SystemExit(
            "API cost reporting requires both input and output token prices"
        )
    source_path = Path(args.source).expanduser().resolve()
    task_path = Path(args.task).expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit("Kernel source file does not exist: %s" % source_path)
    if not task_path.is_file():
        raise SystemExit("Task file does not exist: %s" % task_path)
    task = TaskSpec.from_json_file(task_path)
    style = resolve_baseline_style(
        name=args.baseline_style,
        task=task,
        agent_workers=args.agent_workers,
        evaluation_policy=args.evaluation_policy,
        tir_evidence_policy=args.tir_evidence_policy,
        selection_policy=args.selection_policy,
        profile_policy=args.profile_policy,
        compiled_deduplication=not args.no_compiled_dedup,
        structural_search_policy=args.structural_search_policy,
        strategy_allocation_policy=args.strategy_allocation_policy,
        explicit_fields=_explicit_baseline_fields(raw_argv),
    )
    task = style.task
    args.agent_workers = style.agent_workers
    args.evaluation_policy = style.evaluation_policy
    args.tir_evidence_policy = style.tir_evidence_policy
    args.selection_policy = style.selection_policy
    args.profile_policy = style.profile_policy
    args.no_compiled_dedup = not style.compiled_deduplication
    args.structural_search_policy = style.structural_search_policy
    args.strategy_allocation_policy = style.strategy_allocation_policy
    if args.strategy_plan_interval_rounds is None:
        args.strategy_plan_interval_rounds = (
            3 if style.preset.name == "native" else 1
        )
    try:
        policies = resolve_evaluation_policies(
            evaluation_policy=args.evaluation_policy,
            tir_evidence_policy=args.tir_evidence_policy,
            selection_policy=args.selection_policy,
            profile_policy=args.profile_policy,
            compiled_deduplication=not args.no_compiled_dedup,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    source_code = source_path.read_text(encoding="utf-8")
    seed = Candidate.seed(task, source_code, source_path.name)
    SourceValidator().validate(task, source_code, source_path.name)
    api_url = args.api_url or os.environ.get("KERNEL_OPT_API_URL")
    api_model = args.api_model or os.environ.get("KERNEL_OPT_API_MODEL")
    if not api_url or not api_model:
        raise SystemExit(
            "API source optimization requires --api-url/--api-model or "
            "KERNEL_OPT_API_URL/KERNEL_OPT_API_MODEL"
        )
    if not os.environ.get(args.api_key_env):
        raise SystemExit("API key environment variable %s is not set" % args.api_key_env)

    progress = ProgressReporter(enabled=not args.quiet)

    def report_api_request(size):
        progress.emit(
            "api_request_ready",
            "API request prepared with the configured prompt context.",
            kind=size.get("kind"),
            estimated_input_tokens=size.get("estimated_input_tokens"),
            max_output_tokens=size.get("max_output_tokens"),
            estimated_reserved_tokens=size.get("estimated_reserved_tokens"),
            input_budget=size.get("max_input_tokens"),
            user_characters=size.get("user_characters"),
            compressed_context_budget=size.get("compressed_context_budget"),
        )

    api_config = ApiGeneratorConfig(
        api_url=api_url,
        model=api_model,
        api_key_environment_variable=args.api_key_env,
        timeout_seconds=args.api_timeout,
        temperature=args.api_temperature,
        max_output_tokens=args.api_max_output_tokens,
        planner_max_output_tokens=args.api_planner_max_output_tokens,
        max_input_tokens=args.api_max_input_tokens,
        max_tokens_field=args.api_max_tokens_field,
        max_retries=args.api_retries,
        retry_backoff_seconds=args.api_retry_backoff,
        use_json_object=not args.no_json_response_format,
        metadata_compression_policy=args.metadata_compression_policy,
        metadata_history_limit=args.metadata_history_limit,
        metadata_lesson_limit=args.metadata_lesson_limit,
        compressed_context_target_tokens=args.compressed_context_target_tokens,
    )

    def create_generator():
        return OpenAICompatibleGenerator(
            api_config,
            request_observer=report_api_request,
        )

    if args.agent_workers == 1:
        generator = create_generator()
    else:
        generator = ParallelCandidateGenerator(
            create_generator,
            max_workers=args.agent_workers,
        )
    output = (
        Path(args.output).expanduser()
        if args.output
        else _default_output(task, args.baseline_style)
    )
    output = output.resolve()
    if args.resume and not output.exists():
        raise SystemExit("Cannot resume because the output directory does not exist: %s" % output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(
            "Output directory is not empty; choose a fresh path or add --resume: %s"
            % output
        )

    store = ArtifactStore(output)
    style_metadata = style.to_dict()
    style_metadata["controller_policies"] = dict(policies)
    run_metadata = {
        "framework_version": "0.13.0",
        "baseline_style": style_metadata,
        "generator": {
            "type": "hosted-api",
            "api_url": api_url,
            "model": api_model,
            "temperature": args.api_temperature,
            "timeout_seconds": args.api_timeout,
            "max_output_tokens": args.api_max_output_tokens,
            "planner_max_output_tokens": args.api_planner_max_output_tokens,
            "max_input_tokens": args.api_max_input_tokens,
            "max_tokens_field": args.api_max_tokens_field,
            "max_retries": args.api_retries,
            "api_key_environment_variable": args.api_key_env,
            "agent_workers": args.agent_workers,
            "metadata_compression_policy": args.metadata_compression_policy,
            "metadata_history_limit": args.metadata_history_limit,
            "metadata_lesson_limit": args.metadata_lesson_limit,
            "compressed_context_target_tokens": (
                args.compressed_context_target_tokens
            ),
        },
        "experiment": {
            "baseline_style": style.preset.name,
            "baseline_style_mode": (
                "style-emulation" if style.emulated else "native"
            ),
            "baseline_style_canonical": style.canonical,
            "evaluation_policy": policies["evaluation_policy"],
            "requested_tir_evidence_policy": policies[
                "requested_tir_evidence_policy"
            ],
            "tir_evidence_policy": policies["tir_evidence_policy"],
            "requested_selection_policy": policies[
                "requested_selection_policy"
            ],
            "selection_policy": policies["selection_policy"],
            "fixed_promotions_per_round": args.promotions_per_round,
            "requested_profile_policy": policies["requested_profile_policy"],
            "profile_policy": policies["profile_policy"],
            "requested_compiled_deduplication": policies[
                "requested_compiled_deduplication"
            ],
            "compiled_deduplication": policies["compiled_deduplication"],
            "structural_search_policy": args.structural_search_policy,
            "strategy_allocation_policy": args.strategy_allocation_policy,
            "strategy_plan_interval_rounds": (
                args.strategy_plan_interval_rounds
            ),
            "metadata_compression_policy": args.metadata_compression_policy,
            "metadata_history_limit": args.metadata_history_limit,
            "metadata_lesson_limit": args.metadata_lesson_limit,
            "compressed_context_target_tokens": (
                args.compressed_context_target_tokens
            ),
            "agent_workers": args.agent_workers,
            "api_input_price_per_million": args.api_input_price_per_million,
            "api_output_price_per_million": args.api_output_price_per_million,
            "incumbent_snapshot_interval_seconds": (
                args.incumbent_snapshot_interval_seconds
            ),
            "max_search_seconds": args.max_search_seconds,
            "search_until_time_budget": args.search_until_time_budget,
            "candidate_graph_upper_bound": (
                1
                + task.budget.rounds
                * (
                    task.budget.proposals_per_round
                    + task.budget.max_repairs_per_round
                )
            ),
        },
        "resume_requested": args.resume,
    }
    store.initialize(task, seed, run_metadata)
    backend = create_backend(
        task,
        task_path.parent,
        artifact_directory=store.evaluator_directory,
    )
    store.save_environment_manifest(
        collect_environment_manifest(task, backend, source_path, task_path)
    )

    preflight_calls = 0
    preflight_usage = {}
    preflight_elapsed_seconds = 0.0
    if not args.skip_api_preflight:
        progress.emit(
            "api_preflight",
            "Checking API authentication, model access, and quota before GPU work.",
            model=api_model,
        )
        try:
            preflight = generator.preflight()
        except Exception as error:
            metadata = dict(generator.last_call_metadata)
            store.save_api_call(
                "preflight-failed", metadata, generator.last_exchange
            )
            store.save_failure(
                {
                    "stage": "api-preflight",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "resumable": True,
                }
            )
            store.append_event(
                "api_preflight_failed",
                {
                    "error": "%s: %s" % (type(error).__name__, error),
                    "provider": metadata,
                },
            )
            raise SystemExit(
                "API preflight failed before any compiler/GPU work: %s" % error
            ) from error
        preflight_calls = int(preflight.get("attempts", 1))
        preflight_usage = dict(preflight.get("usage") or {})
        preflight_elapsed_seconds = float(preflight.get("elapsed_seconds", 0.0))
        store.save_api_call("preflight", preflight, generator.last_exchange)
        store.append_event("api_preflight_completed", preflight)
        progress.emit(
            "api_ready",
            "Hosted API preflight passed.",
            attempts=preflight_calls,
            elapsed_seconds=round(float(preflight.get("elapsed_seconds", 0.0)), 3),
        )

    controller = OptimizationController(
        task=task,
        source_code=source_code,
        source_name=source_path.name,
        generator=generator,
        backend=backend,
        store=store,
        run_metadata=run_metadata,
        progress=progress,
        resume=args.resume,
        preflight_calls=preflight_calls,
        preflight_usage=preflight_usage,
        preflight_elapsed_seconds=preflight_elapsed_seconds,
        selection_policy=args.selection_policy,
        profile_policy=args.profile_policy,
        fixed_promotions_per_round=args.promotions_per_round,
        compiled_deduplication=not args.no_compiled_dedup,
        structural_search_policy=args.structural_search_policy,
        strategy_allocation_policy=args.strategy_allocation_policy,
        strategy_plan_interval_rounds=args.strategy_plan_interval_rounds,
        api_input_price_per_million=args.api_input_price_per_million,
        api_output_price_per_million=args.api_output_price_per_million,
        incumbent_snapshot_interval_seconds=(
            args.incumbent_snapshot_interval_seconds
        ),
        max_search_seconds=args.max_search_seconds,
        search_until_time_budget=args.search_until_time_budget,
        evaluation_policy=args.evaluation_policy,
        tir_evidence_policy=args.tir_evidence_policy,
        metadata_compression_policy=args.metadata_compression_policy,
    )
    summary = controller.run()
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print("Best kernel: %s" % summary.best_source_path)
    print("Summary: %s" % (output / "summary.json"))
    if args.incumbent_snapshot_interval_seconds > 0:
        print("Incumbent history: %s" % (output / "incumbent_history.csv"))
    return 0


def _default_output(task: TaskSpec, baseline_style: str = "native") -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    style_suffix = "" if baseline_style == "native" else "_" + baseline_style
    return Path("results") / (task.task_id + style_suffix + "_" + timestamp)


_BASELINE_OPTION_FIELDS = {
    "--agent-workers": "agent_workers",
    "--selection-policy": "selection_policy",
    "--evaluation-policy": "evaluation_policy",
    "--tir-evidence-policy": "tir_evidence_policy",
    "--profile-policy": "profile_policy",
    "--no-compiled-dedup": "compiled_deduplication",
    "--structural-search-policy": "structural_search_policy",
    "--strategy-allocation-policy": "strategy_allocation_policy",
}


def _explicit_baseline_fields(argv: Sequence[str]) -> frozenset[str]:
    """Return style-controlled fields explicitly present on the command line."""

    fields = set()
    for token in argv:
        option = token.split("=", 1)[0]
        field = _BASELINE_OPTION_FIELDS.get(option)
        if field is not None:
            fields.add(field)
    return frozenset(fields)


if __name__ == "__main__":
    raise SystemExit(main())
