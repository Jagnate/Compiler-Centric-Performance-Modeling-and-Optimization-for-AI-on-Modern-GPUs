"""Append-only artifacts and atomic snapshots for an optimization run."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable

from .schema import CandidateRecord, SearchSummary, TaskSpec


class ArtifactStore:
    """Persist task inputs, candidate evidence, events, state, and summaries."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.candidate_directory = self.root / "candidates"
        self.root.mkdir(parents=True, exist_ok=True)
        self.candidate_directory.mkdir(parents=True, exist_ok=True)

    @property
    def event_path(self) -> Path:
        return self.root / "events.jsonl"

    def initialize(self, task: TaskSpec) -> None:
        marker = self.root / "task.json"
        if marker.exists():
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if existing.get("task_id") != task.task_id:
                raise ValueError(
                    "artifact directory already belongs to task %r"
                    % existing.get("task_id")
                )
        else:
            self._write_json(marker, task.to_dict())

    def save_candidate(self, record: CandidateRecord) -> Path:
        path = self.candidate_directory / (record.candidate.candidate_id + ".json")
        self._write_json(path, record.to_dict())
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
        for path in self.candidate_directory.glob("*.json"):
            yield path.stem

    @staticmethod
    def _write_json(path: Path, value: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(path)
        except Exception:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise

