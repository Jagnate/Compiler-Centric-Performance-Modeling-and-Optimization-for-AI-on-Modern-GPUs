"""Subprocess JSON contract for TileSight and other external evaluators."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ..schema import (
    Candidate,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    TaskSpec,
)


class CommandBackend:
    """Invoke an external evaluator without importing its compiler runtime."""

    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: float = 300.0,
        working_directory: Optional[Path] = None,
        environment: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not command:
            raise ValueError("command cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.command = [str(item) for item in command]
        self.timeout_seconds = timeout_seconds
        self.working_directory = (
            working_directory.resolve() if working_directory else None
        )
        self.environment = dict(environment or {})

    def model(self, task: TaskSpec, candidate: Candidate) -> ModelEvaluation:
        return ModelEvaluation.from_dict(self._run("model", task, candidate))

    def measure(self, task: TaskSpec, candidate: Candidate) -> Measurement:
        return Measurement.from_dict(self._run("measure", task, candidate))

    def profile(self, task: TaskSpec, candidate: Candidate) -> ProfileEvaluation:
        return ProfileEvaluation.from_dict(self._run("profile", task, candidate))

    def _run(
        self, stage: str, task: TaskSpec, candidate: Candidate
    ) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="kernel-opt-command-") as directory:
            root = Path(directory)
            request_path = root / "request.json"
            response_path = root / "response.json"
            source_path = root / candidate.source_name
            source_path.write_text(candidate.source_code, encoding="utf-8")
            request_path.write_text(
                json.dumps(
                    {
                        "stage": stage,
                        "task": task.to_dict(),
                        "candidate": candidate.to_dict(include_source=False),
                        "source": {
                            "path": str(source_path),
                            "filename": candidate.source_name,
                            "sha256": candidate.source_sha256,
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            command = self.command + [
                "--stage",
                stage,
                "--request",
                str(request_path),
                "--response",
                str(response_path),
            ]
            environment = os.environ.copy()
            for name, value in self.environment.items():
                if name == "PYTHONPATH" and environment.get(name):
                    environment[name] = value + os.pathsep + environment[name]
                else:
                    environment[name] = value
            completed = subprocess.run(
                command,
                cwd=str(self.working_directory) if self.working_directory else None,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "external evaluator stage %s failed with exit code %d: %s"
                    % (stage, completed.returncode, completed.stdout[-4000:])
                )
            if not response_path.is_file():
                raise RuntimeError(
                    "external evaluator stage %s did not write %s"
                    % (stage, response_path)
                )
            response = json.loads(response_path.read_text(encoding="utf-8"))
            if not isinstance(response, dict):
                raise ValueError("external evaluator response must be a JSON object")
            return response
