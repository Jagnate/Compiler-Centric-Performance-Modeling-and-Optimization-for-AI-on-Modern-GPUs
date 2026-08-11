"""Command-line entry point for adaptive kernel optimization."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .archive import ArtifactStore
from .controller import OptimizationController
from .factory import create_backend
from .generators import (
    ApiGeneratorConfig,
    DeterministicMockGenerator,
    OpenAICompatibleGenerator,
)
from .schema import TaskSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run adaptive-fidelity GPU kernel optimization."
    )
    parser.add_argument("--task", required=True, help="Path to a JSON TaskSpec.")
    parser.add_argument(
        "--generator",
        choices=("mock", "api"),
        default="mock",
        help="Candidate generator. Mock mode is deterministic and requires no network.",
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
    task_path = Path(args.task).expanduser().resolve()
    if not task_path.is_file():
        raise SystemExit("Task file does not exist: %s" % task_path)
    task = TaskSpec.from_json_file(task_path)

    if args.generator == "mock":
        generator = DeterministicMockGenerator()
    else:
        api_url = args.api_url or os.environ.get("KERNEL_OPT_API_URL")
        api_model = args.api_model or os.environ.get("KERNEL_OPT_API_MODEL")
        if not api_url or not api_model:
            raise SystemExit(
                "API mode requires --api-url/--api-model or "
                "KERNEL_OPT_API_URL/KERNEL_OPT_API_MODEL"
            )
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
        generator=generator,
        backend=backend,
        store=ArtifactStore(output),
    )
    summary = controller.run()
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    print("Summary: %s" % (output / "summary.json"))
    return 0


def _default_output(task: TaskSpec) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results") / (task.task_id + "_" + timestamp)


if __name__ == "__main__":
    raise SystemExit(main())

