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
from .schema import TaskSpec


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
    parser.add_argument("--api-temperature", type=float, default=0.4)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    source_path = Path(args.source).expanduser().resolve()
    task_path = Path(args.task).expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit("Kernel source file does not exist: %s" % source_path)
    if not task_path.is_file():
        raise SystemExit("Task file does not exist: %s" % task_path)

    task = TaskSpec.from_json_file(task_path)
    source_code = source_path.read_text(encoding="utf-8")
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
        )
    )
    backend = create_backend(task, task_path.parent)
    output = Path(args.output).expanduser() if args.output else _default_output(task)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            "Output directory is not empty; choose a fresh path: %s" % output
        )

    controller = OptimizationController(
        task=task,
        source_code=source_code,
        source_name=source_path.name,
        generator=generator,
        backend=backend,
        store=ArtifactStore(output),
        run_metadata={
            "framework_version": "0.2.0",
            "generator": {
                "type": "hosted-api",
                "api_url": api_url,
                "model": api_model,
                "temperature": args.api_temperature,
                "timeout_seconds": args.api_timeout,
                "api_key_environment_variable": args.api_key_env,
            },
        },
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
