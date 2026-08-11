"""Factories for task-configured evaluator backends."""

from __future__ import annotations

from pathlib import Path

from .backends import CommandBackend
from .protocols import PerformanceBackend
from .schema import TaskSpec


def create_backend(task: TaskSpec, base_directory: Path) -> PerformanceBackend:
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
        environment = configuration.pop("environment", None)
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
        )
    raise ValueError(
        "unsupported evaluator type %r; source optimization requires a command evaluator"
        % backend_type
    )
