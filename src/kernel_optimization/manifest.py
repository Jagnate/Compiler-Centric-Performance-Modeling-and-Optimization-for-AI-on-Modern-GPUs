"""Reproducibility metadata that can be collected without compiler imports."""

from __future__ import annotations

import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional

from .protocols import PerformanceBackend
from .schema import TaskSpec


def collect_environment_manifest(
    task: TaskSpec,
    backend: PerformanceBackend,
    source_path: Path,
    task_path: Path,
) -> Dict[str, Any]:
    """Describe the controller and evaluator boundary without exposing values."""

    backend_manifest = getattr(backend, "environment_manifest", None)
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "process": {
            "working_directory": str(Path.cwd()),
            "environment_names": sorted(os.environ),
        },
        "repository": {
            "root": str(Path(__file__).resolve().parents[2]),
            "revision": _git_revision(Path(__file__).resolve().parents[2]),
        },
        "inputs": {
            "source_path": str(source_path),
            "task_path": str(task_path),
            "task_id": task.task_id,
            "target": task.target,
        },
        "evaluator": backend_manifest() if backend_manifest else {"type": type(backend).__name__},
    }


def _git_revision(repository: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repository),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None

