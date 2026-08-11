"""Append-only artifacts and atomic snapshots for a source optimization run."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional

from .schema import Candidate, CandidateRecord, SearchSummary, TaskSpec


class ArtifactStore:
    """Persist source candidates, evidence, events, state, and summaries."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.candidate_directory = self.root / "candidates"
        self.root.mkdir(parents=True, exist_ok=True)
        self.candidate_directory.mkdir(parents=True, exist_ok=True)

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
        self._write_text(
            self.candidate_source_path(record.candidate),
            record.candidate.source_code,
        )
        path = root / "record.json"
        self._write_json(path, record.to_dict(include_source=False))
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
        record = {"event": event, "payload": payload}
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def save_state(self, state: Dict[str, Any]) -> None:
        self._write_json(self.root / "state.json", state)

    def save_summary(self, summary: SearchSummary) -> Path:
        path = self.root / "summary.json"
        self._write_json(path, summary.to_dict())
        return path

    def candidate_ids(self) -> Iterable[str]:
        for path in self.candidate_directory.iterdir():
            if path.is_dir() and (path / "record.json").is_file():
                yield path.name

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
