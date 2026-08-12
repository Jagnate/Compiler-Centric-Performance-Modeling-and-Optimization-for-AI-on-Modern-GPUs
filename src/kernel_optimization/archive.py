"""Append-only artifacts and atomic snapshots for a source optimization run."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional

from .schema import Candidate, CandidateRecord, SearchSummary, TaskSpec


class ArtifactStore:
    """Persist source candidates, evidence, events, state, and summaries."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.candidate_directory = self.root / "candidates"
        self.checkpoint_directory = self.root / "checkpoints"
        self.api_directory = self.root / "api_calls"
        self.evaluator_directory = self.root / "evaluator_attempts"
        self.root.mkdir(parents=True, exist_ok=True)
        self.candidate_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.api_directory.mkdir(parents=True, exist_ok=True)
        self.evaluator_directory.mkdir(parents=True, exist_ok=True)
        self._event_sequence = self._line_count(self.event_path)
        self._checkpoint_sequence = len(list(self.checkpoint_directory.glob("*.json")))
        self._api_sequence = len(list(self.api_directory.glob("*.json")))

    @property
    def event_path(self) -> Path:
        return self.root / "events.jsonl"

    def initialize(
        self,
        task: TaskSpec,
        seed: Candidate,
        run_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        marker = self.root / "task.json"
        payload = task.to_dict()
        payload["input_source"] = {
            "filename": seed.source_name,
            "sha256": seed.source_sha256,
        }
        payload["run"] = dict(run_metadata or {})
        payload["created_at"] = self._timestamp()
        if marker.exists():
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if existing.get("task_id") != task.task_id:
                raise ValueError(
                    "artifact directory already belongs to task %r"
                    % existing.get("task_id")
                )
            if existing.get("input_source") != payload["input_source"]:
                raise ValueError("artifact directory belongs to a different input source")
        else:
            self._write_json(marker, payload)

    def candidate_root(self, candidate: Candidate) -> Path:
        return self.candidate_directory / candidate.candidate_id

    def candidate_source_path(self, candidate: Candidate) -> Path:
        return self.candidate_root(candidate) / candidate.source_name

    def save_candidate(self, record: CandidateRecord) -> Path:
        root = self.candidate_root(record.candidate)
        root.mkdir(parents=True, exist_ok=True)
        source_path = self.candidate_source_path(record.candidate)
        if source_path.exists():
            existing = source_path.read_text(encoding="utf-8")
            if existing != record.candidate.source_code:
                raise ValueError(
                    "candidate %s source is immutable" % record.candidate.candidate_id
                )
        else:
            self._write_text(source_path, record.candidate.source_code)
        path = root / "record.json"
        payload = record.to_dict(include_source=False)
        self._write_json(path, payload)
        history = root / "history"
        history.mkdir(parents=True, exist_ok=True)
        sequence = len(list(history.glob("*.json"))) + 1
        history_path = history / ("%04d_%s.json" % (sequence, record.state))
        self._write_json(history_path, payload)
        return path

    def save_best(self, record: CandidateRecord) -> Path:
        suffix = Path(record.candidate.source_name).suffix or ".py"
        path = self.root / ("best_kernel" + suffix)
        self._write_text(path, record.candidate.source_code)
        self._write_json(
            self.root / "best_candidate.json",
            record.to_dict(include_source=False),
        )
        return path

    def append_event(self, event: str, payload: Dict[str, Any]) -> None:
        self._event_sequence += 1
        record = {
            "sequence": self._event_sequence,
            "timestamp": self._timestamp(),
            "event": event,
            "payload": payload,
        }
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def save_state(self, state: Dict[str, Any]) -> None:
        payload = dict(state)
        payload["saved_at"] = self._timestamp()
        self._write_json(self.root / "state.json", payload)
        self._checkpoint_sequence += 1
        active_round = payload.get("active_round")
        round_label = "none" if active_round is None else "%03d" % int(active_round)
        phase = str(payload.get("phase", "unknown")).replace("/", "-")
        snapshot = self.checkpoint_directory / (
            "%05d_round_%s_%s.json"
            % (self._checkpoint_sequence, round_label, phase)
        )
        self._write_json(snapshot, payload)

    def load_state(self) -> Optional[Dict[str, Any]]:
        path = self.root / "state.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("state.json must contain a JSON object")
        return value

    def load_candidates(self) -> Dict[str, CandidateRecord]:
        records: Dict[str, CandidateRecord] = {}
        for identifier in self.candidate_ids():
            root = self.candidate_directory / identifier
            value = json.loads((root / "record.json").read_text(encoding="utf-8"))
            candidate = dict(value.get("candidate") or {})
            source_name = str(candidate.get("source_name", ""))
            source_path = root / source_name
            if not source_path.is_file():
                raise ValueError("candidate %s is missing its source file" % identifier)
            candidate["source_code"] = source_path.read_text(encoding="utf-8")
            value["candidate"] = candidate
            record = CandidateRecord.from_dict(value)
            if record.candidate.candidate_id != identifier:
                raise ValueError("candidate directory identity mismatch for %s" % identifier)
            records[identifier] = record
        return records

    def save_api_call(
        self,
        kind: str,
        metadata: Mapping[str, Any],
        exchange: Optional[Mapping[str, Any]],
    ) -> Path:
        self._api_sequence += 1
        path = self.api_directory / (
            "%04d_%s.json" % (self._api_sequence, kind.replace("/", "-"))
        )
        self._write_json(
            path,
            {
                "timestamp": self._timestamp(),
                "kind": kind,
                "metadata": self._redact(dict(metadata)),
                "exchange": self._redact(dict(exchange or {})),
            },
        )
        return path

    def save_environment_manifest(self, value: Mapping[str, Any]) -> Path:
        path = self.root / "environment.json"
        self._write_json(path, dict(value))
        return path

    def save_failure(self, value: Mapping[str, Any]) -> Path:
        path = self.root / "failure.json"
        payload = dict(value)
        payload.setdefault("timestamp", self._timestamp())
        self._write_json(path, payload)
        return path

    def save_summary(self, summary: SearchSummary) -> Path:
        path = self.root / "summary.json"
        self._write_json(path, summary.to_dict())
        return path

    def candidate_ids(self) -> Iterable[str]:
        for path in self.candidate_directory.iterdir():
            if path.is_dir() and (path / "record.json").is_file():
                yield path.name

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _line_count(path: Path) -> int:
        if not path.is_file():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    @classmethod
    def _redact(cls, value: Any, key: str = "") -> Any:
        lowered = key.lower()
        if re.search(
            r"(^|_)(authorization|api_?key|access_token|refresh_token|secret|password|passwd|credentials?)(_|$)",
            lowered,
        ):
            return "<redacted>"
        if isinstance(value, dict):
            return {str(name): cls._redact(item, str(name)) for name, item in value.items()}
        if isinstance(value, list):
            return [cls._redact(item) for item in value]
        return value

    @staticmethod
    def _write_json(path: Path, value: Dict[str, Any]) -> None:
        ArtifactStore._write_text(
            path,
            json.dumps(value, indent=2, sort_keys=True) + "\n",
        )

    @staticmethod
    def _write_text(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(path)
        except Exception:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise
