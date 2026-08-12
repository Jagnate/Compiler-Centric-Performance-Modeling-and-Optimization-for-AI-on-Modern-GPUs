"""Subprocess JSON contract for TileSight and other external evaluators."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

from ..schema import (
    Candidate,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    TaskSpec,
)


_SENSITIVE_ENVIRONMENT_NAME = re.compile(
    r"(^|_)(API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)(_|$)",
    re.IGNORECASE,
)


class CommandBackend:
    """Invoke an external evaluator without importing its compiler runtime."""

    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: float = 300.0,
        working_directory: Optional[Path] = None,
        environment: Optional[Mapping[str, str]] = None,
        artifact_directory: Optional[Path] = None,
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
        self.artifact_directory = (
            artifact_directory.resolve() if artifact_directory else None
        )
        if self.artifact_directory:
            self.artifact_directory.mkdir(parents=True, exist_ok=True)
        self.last_stage_metadata: Dict[str, Any] = {}

    def model(self, task: TaskSpec, candidate: Candidate) -> ModelEvaluation:
        return ModelEvaluation.from_dict(self._run("model", task, candidate))

    def measure(self, task: TaskSpec, candidate: Candidate) -> Measurement:
        return Measurement.from_dict(self._run("measure", task, candidate))

    def profile(self, task: TaskSpec, candidate: Candidate) -> ProfileEvaluation:
        return ProfileEvaluation.from_dict(self._run("profile", task, candidate))

    def environment_manifest(self) -> Dict[str, Any]:
        environment, removed = self._build_environment()
        return {
            "backend": "command",
            "command": list(self.command),
            "working_directory": (
                str(self.working_directory) if self.working_directory else None
            ),
            "timeout_seconds": self.timeout_seconds,
            "configured_environment_names": sorted(self.environment),
            "forwarded_environment_names": sorted(environment),
            "removed_sensitive_environment_names": sorted(removed),
        }

    def _run(
        self, stage: str, task: TaskSpec, candidate: Candidate
    ) -> Dict[str, Any]:
        with self._attempt_directory(candidate, stage) as root:
            request_path = root / "request.json"
            response_path = root / "response.json"
            source_path = root / candidate.source_name
            stdout_path = root / "stdout.log"
            result_path = root / "attempt.json"
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
            environment, removed = self._build_environment()
            started_at = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    cwd=(
                        str(self.working_directory)
                        if self.working_directory
                        else None
                    ),
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                elapsed = time.perf_counter() - started_at
                output = _timeout_output(error)
                stdout_path.write_text(output, encoding="utf-8")
                self.last_stage_metadata = {
                    "stage": stage,
                    "candidate_id": candidate.candidate_id,
                    "elapsed_seconds": elapsed,
                    "status": "timeout",
                    "artifact_directory": str(root),
                    "removed_sensitive_environment_names": sorted(removed),
                }
                result_path.write_text(
                    json.dumps(self.last_stage_metadata, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                raise RuntimeError(
                    "external evaluator stage %s timed out after %.1f seconds: %s"
                    % (stage, self.timeout_seconds, output[-4000:])
                ) from error

            elapsed = time.perf_counter() - started_at
            stdout_path.write_text(completed.stdout or "", encoding="utf-8")
            self.last_stage_metadata = {
                "stage": stage,
                "candidate_id": candidate.candidate_id,
                "elapsed_seconds": elapsed,
                "status": "completed" if completed.returncode == 0 else "failed",
                "exit_code": completed.returncode,
                "artifact_directory": str(root),
                "removed_sensitive_environment_names": sorted(removed),
            }
            result_path.write_text(
                json.dumps(self.last_stage_metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "external evaluator stage %s failed with exit code %d: %s"
                    % (stage, completed.returncode, (completed.stdout or "")[-4000:])
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

    def _build_environment(self) -> tuple[Dict[str, str], Sequence[str]]:
        environment = dict(os.environ)
        removed = []
        for name in list(environment):
            if self._is_sensitive_environment_name(name):
                removed.append(name)
                environment.pop(name, None)
        for name, value in self.environment.items():
            if self._is_sensitive_environment_name(name):
                if name not in removed:
                    removed.append(name)
                continue
            if name == "PYTHONPATH" and environment.get(name):
                environment[name] = value + os.pathsep + environment[name]
            else:
                environment[name] = value
        environment["KERNEL_OPT_UNTRUSTED_EVALUATOR"] = "1"
        return environment, removed

    @staticmethod
    def _is_sensitive_environment_name(name: str) -> bool:
        return bool(
            name.upper().startswith("KERNEL_OPT_API_")
            or _SENSITIVE_ENVIRONMENT_NAME.search(name)
        )

    @contextmanager
    def _attempt_directory(
        self, candidate: Candidate, stage: str
    ) -> Iterator[Path]:
        if self.artifact_directory is None:
            with tempfile.TemporaryDirectory(prefix="kernel-opt-command-") as directory:
                yield Path(directory)
            return
        parent = self.artifact_directory / candidate.candidate_id
        parent.mkdir(parents=True, exist_ok=True)
        sequence = len(list(parent.glob(stage + "_*"))) + 1
        root = parent / ("%s_%03d" % (stage, sequence))
        root.mkdir(parents=True, exist_ok=False)
        yield root


def _timeout_output(error: subprocess.TimeoutExpired) -> str:
    value = error.stdout or error.output or ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
