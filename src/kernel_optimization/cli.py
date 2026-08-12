"""Command-line entry point for hosted-API kernel source optimization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Optional, Sequence

from .archive import ArtifactStore
from .controller import OptimizationController
from .factory import create_backend
from .generators import ApiGeneratorConfig, OpenAICompatibleGenerator
from .manifest import collect_environment_manifest
from .progress import ProgressReporter
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
        "--api-max-tokens-field",
        choices=("max_tokens", "max_completion_tokens"),
        default="max_completion_tokens",
    )
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--api-retry-backoff", type=float, default=2.0)
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
        "--selection-policy",
        choices=("adaptive", "model-top", "random", "measure-all"),
        default="adaptive",
        help="Candidate promotion strategy. Use a fixed budget for equal-cost ablations.",
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
    args = build_parser().parse_args(argv)
    if args.promotions_per_round is not None and args.promotions_per_round <= 0:
        raise SystemExit("--promotions-per-round must be positive")
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

    generator = OpenAICompatibleGenerator(
        ApiGeneratorConfig(
            api_url=api_url,
            model=api_model,
            api_key_environment_variable=args.api_key_env,
            timeout_seconds=args.api_timeout,
            temperature=args.api_temperature,
            max_output_tokens=args.api_max_output_tokens,
            max_tokens_field=args.api_max_tokens_field,
            max_retries=args.api_retries,
            retry_backoff_seconds=args.api_retry_backoff,
            use_json_object=not args.no_json_response_format,
        )
    )
    output = Path(args.output).expanduser() if args.output else _default_output(task)
    output = output.resolve()
    if args.resume and not output.exists():
        raise SystemExit("Cannot resume because the output directory does not exist: %s" % output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(
            "Output directory is not empty; choose a fresh path or add --resume: %s"
            % output
        )

    progress = ProgressReporter(enabled=not args.quiet)
    store = ArtifactStore(output)
    run_metadata = {
        "framework_version": "0.6.0",
        "generator": {
            "type": "hosted-api",
            "api_url": api_url,
            "model": api_model,
            "temperature": args.api_temperature,
            "timeout_seconds": args.api_timeout,
            "max_output_tokens": args.api_max_output_tokens,
            "max_tokens_field": args.api_max_tokens_field,
            "max_retries": args.api_retries,
            "api_key_environment_variable": args.api_key_env,
        },
        "experiment": {
            "selection_policy": args.selection_policy,
            "fixed_promotions_per_round": args.promotions_per_round,
            "profile_policy": args.profile_policy,
            "compiled_deduplication": not args.no_compiled_dedup,
            "api_input_price_per_million": args.api_input_price_per_million,
            "api_output_price_per_million": args.api_output_price_per_million,
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
        api_input_price_per_million=args.api_input_price_per_million,
        api_output_price_per_million=args.api_output_price_per_million,
    )
    summary = controller.run()
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print("Best kernel: %s" % summary.best_source_path)
    print("Summary: %s" % (output / "summary.json"))
    return 0


def _default_output(task: TaskSpec) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / (task.task_id + "_" + timestamp)


if __name__ == "__main__":
    raise SystemExit(main())
