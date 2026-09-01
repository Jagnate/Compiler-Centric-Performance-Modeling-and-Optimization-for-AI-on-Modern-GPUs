"""Subprocess JSON contract for TileSight and other external evaluators."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
import re
import selectors
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

from ..schema import (
    Candidate,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    TIRAnalysis,
    TaskSpec,
)


_SENSITIVE_ENVIRONMENT_NAME = re.compile(
    r"(^|_)(API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)(_|$)",
    re.IGNORECASE,
)

_WORKER_RESPONSE_PREFIX = "KERNEL_OPT_WORKER_RESPONSE "


class CommandBackend:
    """Invoke an external evaluator without importing its compiler runtime."""

    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: float = 300.0,
        working_directory: Optional[Path] = None,
        environment: Optional[Mapping[str, str]] = None,
        artifact_directory: Optional[Path] = None,
        persistent_process: bool = False,
        worker_max_requests: int = 32,
    ) -> None:
        if not command:
            raise ValueError("command cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if worker_max_requests <= 0:
            raise ValueError("worker_max_requests must be positive")
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
        self.persistent_process = bool(persistent_process)
        self.worker_max_requests = int(worker_max_requests)
        self.last_stage_metadata: Dict[str, Any] = {}
        self._worker: Optional[subprocess.Popen[str]] = None
        self._worker_requests = 0
        self._worker_starts = 0
        self._worker_lock = threading.Lock()

    def model(self, task: TaskSpec, candidate: Candidate) -> ModelEvaluation:
        return ModelEvaluation.from_dict(self._run("model", task, candidate))

    def analyze_tir(self, task: TaskSpec, candidate: Candidate) -> TIRAnalysis:
        return TIRAnalysis.from_dict(self._run("tir", task, candidate))

    def measure(self, task: TaskSpec, candidate: Candidate) -> Measurement:
        return Measurement.from_dict(self._run("measure", task, candidate))

    def profile(self, task: TaskSpec, candidate: Candidate) -> ProfileEvaluation:
        return ProfileEvaluation.from_dict(self._run("profile", task, candidate))

    def finalize(self, task: TaskSpec, candidate: Candidate) -> Measurement:
        return Measurement.from_dict(self._run("final", task, candidate))

    def environment_manifest(self) -> Dict[str, Any]:
        environment, removed = self._build_environment()
        return {
            "backend": "command",
            "command": list(self.command),
            "working_directory": (
                str(self.working_directory) if self.working_directory else None
            ),
            "timeout_seconds": self.timeout_seconds,
            "persistent_process": self.persistent_process,
            "worker_max_requests": self.worker_max_requests,
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
            if self.persistent_process:
                if stage != "final":
                    return self._run_persistent(
                        stage=stage,
                        candidate=candidate,
                        request_path=request_path,
                        response_path=response_path,
                        stdout_path=stdout_path,
                        result_path=result_path,
                        root=root,
                        environment=environment,
                        removed=removed,
                    )
                # Final validation is deliberately process-isolated. Stop the
                # warmed search worker before launching the regular one-shot
                # command so compiler and CUDA state cannot leak into the gate.
                with self._worker_lock:
                    self._stop_worker()
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
                    "execution_mode": "one-shot",
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
                "execution_mode": "one-shot",
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

    def _run_persistent(
        self,
        *,
        stage: str,
        candidate: Candidate,
        request_path: Path,
        response_path: Path,
        stdout_path: Path,
        result_path: Path,
        root: Path,
        environment: Mapping[str, str],
        removed: Sequence[str],
    ) -> Dict[str, Any]:
        """Send one isolated artifact request through a warmed evaluator process."""

        request_id = uuid.uuid4().hex
        stdout_path.write_text("", encoding="utf-8")
        started_at = time.perf_counter()
        with self._worker_lock:
            worker = self._ensure_worker(environment)
            worker_request_index = self._worker_requests + 1
            message = {
                "request_id": request_id,
                "stage": stage,
                "request": str(request_path),
                "response": str(response_path),
                "stdout": str(stdout_path),
            }
            try:
                assert worker.stdin is not None
                worker.stdin.write(json.dumps(message, sort_keys=True) + "\n")
                worker.stdin.flush()
                self._worker_requests = worker_request_index
                acknowledgement, protocol_output = self._read_worker_response(
                    worker, request_id, self.timeout_seconds
                )
            except Exception as error:
                elapsed = time.perf_counter() - started_at
                self._append_protocol_output(stdout_path, locals().get("protocol_output", ""))
                self._terminate_worker()
                status = "timeout" if isinstance(error, TimeoutError) else "failed"
                self.last_stage_metadata = {
                    "stage": stage,
                    "candidate_id": candidate.candidate_id,
                    "elapsed_seconds": elapsed,
                    "status": status,
                    "artifact_directory": str(root),
                    "removed_sensitive_environment_names": sorted(removed),
                    "execution_mode": "persistent-worker",
                    "worker_request_index": worker_request_index,
                    "worker_start_count": self._worker_starts,
                    "error": "%s: %s" % (type(error).__name__, error),
                }
                result_path.write_text(
                    json.dumps(self.last_stage_metadata, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                raise RuntimeError(
                    "persistent evaluator stage %s %s: %s"
                    % (stage, status, error)
                ) from error

            elapsed = time.perf_counter() - started_at
            self._append_protocol_output(stdout_path, protocol_output)
            status = str(acknowledgement.get("status", "failed"))
            self.last_stage_metadata = {
                "stage": stage,
                "candidate_id": candidate.candidate_id,
                "elapsed_seconds": elapsed,
                "status": "completed" if status == "ok" else "failed",
                "exit_code": 0 if status == "ok" else 1,
                "artifact_directory": str(root),
                "removed_sensitive_environment_names": sorted(removed),
                "execution_mode": "persistent-worker",
                "worker_pid": worker.pid,
                "worker_request_index": worker_request_index,
                "worker_start_count": self._worker_starts,
            }
            if status != "ok":
                self.last_stage_metadata["worker_error"] = dict(
                    acknowledgement.get("error") or {}
                )
            result_path.write_text(
                json.dumps(self.last_stage_metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if status != "ok":
                error = dict(acknowledgement.get("error") or {})
                # A protocol-level failure means the evaluator did not produce a
                # normal stage response. Rewarm it before serving another request;
                # ordinary invalid candidates are returned as valid JSON responses
                # and do not take this path.
                self._terminate_worker()
                raise RuntimeError(
                    "persistent evaluator stage %s failed: %s: %s"
                    % (
                        stage,
                        error.get("type", "RuntimeError"),
                        error.get("message", "unknown worker failure"),
                    )
                )
            if not response_path.is_file():
                raise RuntimeError(
                    "persistent evaluator stage %s did not write %s"
                    % (stage, response_path)
                )
            response = json.loads(response_path.read_text(encoding="utf-8"))
            if not isinstance(response, dict):
                raise ValueError("external evaluator response must be a JSON object")
            return response

    def _ensure_worker(self, environment: Mapping[str, str]) -> subprocess.Popen[str]:
        if self._worker_requests >= self.worker_max_requests:
            self._stop_worker()
        if self._worker is not None and self._worker.poll() is None:
            return self._worker
        self._terminate_worker()
        command = self.command + ["--persistent-worker"]
        self._worker = subprocess.Popen(
            command,
            cwd=(str(self.working_directory) if self.working_directory else None),
            env=dict(environment),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._worker_requests = 0
        self._worker_starts += 1
        return self._worker

    @staticmethod
    def _read_worker_response(
        worker: subprocess.Popen[str], request_id: str, timeout_seconds: float
    ) -> tuple[Dict[str, Any], str]:
        assert worker.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(worker.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout_seconds
        output = []
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "request %s exceeded %.1f seconds"
                        % (request_id, timeout_seconds)
                    )
                if not selector.select(remaining):
                    raise TimeoutError(
                        "request %s exceeded %.1f seconds"
                        % (request_id, timeout_seconds)
                    )
                line = worker.stdout.readline()
                if not line:
                    return_code = worker.poll()
                    raise RuntimeError(
                        "worker exited before acknowledging request %s (exit code %s)"
                        % (request_id, return_code)
                    )
                if not line.startswith(_WORKER_RESPONSE_PREFIX):
                    output.append(line)
                    continue
                value = json.loads(line[len(_WORKER_RESPONSE_PREFIX) :])
                if str(value.get("request_id")) != request_id:
                    output.append(line)
                    continue
                return value, "".join(output)
        finally:
            selector.close()

    @staticmethod
    def _append_protocol_output(path: Path, output: str) -> None:
        if not output:
            return
        with path.open("a", encoding="utf-8") as stream:
            stream.write(output)

    def close(self) -> None:
        """Stop a persistent evaluator without raising during run teardown."""

        with self._worker_lock:
            self._stop_worker()

    def _stop_worker(self) -> None:
        worker = self._worker
        if worker is None:
            return
        if worker.poll() is None:
            try:
                assert worker.stdin is not None
                worker.stdin.write(json.dumps({"action": "shutdown"}) + "\n")
                worker.stdin.flush()
                worker.wait(timeout=5.0)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._terminate_worker()
            else:
                self._close_worker_pipes(worker)
                self._worker = None
                self._worker_requests = 0
        else:
            self._close_worker_pipes(worker)
            self._worker = None
            self._worker_requests = 0

    def _terminate_worker(self) -> None:
        worker = self._worker
        self._worker = None
        self._worker_requests = 0
        if worker is None:
            return
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=2.0)
        self._close_worker_pipes(worker)

    @staticmethod
    def _close_worker_pipes(worker: subprocess.Popen[str]) -> None:
        for stream in (worker.stdin, worker.stdout, worker.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def __del__(self) -> None:
        try:
            self._terminate_worker()
        except Exception:
            pass

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
