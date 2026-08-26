"""Factories for task-configured evaluator backends."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .backends import CommandBackend
from .protocols import PerformanceBackend
from .schema import TaskSpec


def create_backend(
    task: TaskSpec,
    base_directory: Path,
    artifact_directory: Optional[Path] = None,
) -> PerformanceBackend:
    """Create the evaluator declared by TaskSpec.evaluator."""

    configuration = dict(task.evaluator)
    backend_type = str(configuration.pop("type", "command"))
    if backend_type == "command":
        command = configuration.pop("command", None)
        if not isinstance(command, list) or not command:
            raise ValueError("a command evaluator needs a non-empty command list")
        working_directory_value = configuration.pop("working_directory", None)
        working_directory = None
        if working_directory_value:
            working_directory = Path(str(working_directory_value))
            if not working_directory.is_absolute():
                working_directory = base_directory / working_directory
        timeout = float(configuration.pop("timeout_seconds", 300.0))
        environment = {
            str(name): str(value)
            for name, value in dict(
                configuration.pop("environment", None) or {}
            ).items()
        }
        if artifact_directory is not None:
            default_cache = artifact_directory.parent / ".tilelang_cache"
            cache_directory = Path(
                environment.get("TILELANG_CACHE_DIR")
                or os.environ.get("TILELANG_CACHE_DIR")
                or default_cache
            ).expanduser()
            temporary_directory = Path(
                environment.get("TILELANG_TMP_DIR")
                or os.environ.get("TILELANG_TMP_DIR")
                or cache_directory / "tmp"
            ).expanduser()
            cache_directory.mkdir(parents=True, exist_ok=True)
            temporary_directory.mkdir(parents=True, exist_ok=True)
            environment.setdefault("TILELANG_CACHE_DIR", str(cache_directory))
            environment.setdefault("TILELANG_TMP_DIR", str(temporary_directory))
        runtime = configuration.pop("runtime", None)
        if runtime is not None and not isinstance(runtime, dict):
            raise ValueError("command evaluator runtime must be a JSON object")
        if configuration:
            raise ValueError(
                "unsupported command evaluator fields: %s"
                % ", ".join(sorted(configuration))
            )
        return CommandBackend(
            command=command,
            timeout_seconds=timeout,
            working_directory=working_directory,
            environment=environment,
            artifact_directory=artifact_directory,
        )
    raise ValueError(
        "unsupported evaluator type %r; source optimization requires a command evaluator"
        % backend_type
    )
