"""Wall-clock snapshots of the best correctness-verified kernel found so far."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .archive import ArtifactStore


INCUMBENT_JSONL = "incumbent_history.jsonl"
INCUMBENT_CSV = "incumbent_history.csv"

_FINAL_PHASES = {"finalized", "run_completed"}
_CSV_FIELDS = (
    "sequence",
    "timestamp",
    "reason",
    "scheduled_elapsed_seconds",
    "elapsed_seconds",
    "phase",
    "completed_round",
    "active_round",
    "evaluation_policy",
    "tir_evidence_policy",
    "strategy_allocation_policy",
    "candidate_id",
    "generation",
    "state",
    "selection_basis",
    "incumbent_latency_ms",
    "measured_latency_ms",
    "final_latency_ms",
    "seed_comparable_latency_ms",
    "speedup_over_seed",
    "predicted_latency_ms",
    "calibrated_predicted_latency_ms",
    "model_bottleneck",
    "profile_bottleneck",
    "diagnosis_category",
    "model_ddr_util",
    "model_l2_hit_rate",
    "model_l2_util",
    "model_smem_util",
    "model_tensor_util",
    "model_cuda_util",
    "model_sfu_util",
    "registers_per_thread",
    "shared_memory_per_block",
    "tiles_per_sm",
    "generated_candidates",
    "modeled_candidates",
    "measured_candidates",
    "correct_candidates",
    "profiled_candidates",
    "profile_calls",
    "generator_calls",
    "planner_calls",
    "api_request_attempts",
)


class PeriodicIncumbentRecorder:
    """Persist fixed-interval and event-driven snapshots without blocking search."""

    def __init__(
        self,
        store: ArtifactStore,
        interval_seconds: float = 300.0,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        interval = float(interval_seconds)
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("incumbent snapshot interval must be finite and non-negative")
        self.store = store
        self.interval_seconds = interval
        self._clock = clock
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at: Optional[float] = None
        self._elapsed_offset = self._load_elapsed_offset()
        self._sequence = self._line_count(self.jsonl_path)
        self._last_signature = self._load_last_signature()
        self._last_error: Optional[str] = None

    @property
    def jsonl_path(self) -> Path:
        return self.store.root / INCUMBENT_JSONL

    @property
    def csv_path(self) -> Path:
        return self.store.root / INCUMBENT_CSV

    @property
    def enabled(self) -> bool:
        return self.interval_seconds > 0

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def start(self, *, started_at: Optional[float] = None, resumed: bool = False) -> None:
        if not self.enabled:
            return
        if self._thread is not None:
            raise RuntimeError("incumbent recorder has already been started")
        self._started_at = self._clock() if started_at is None else float(started_at)
        self.record_now("run-resumed" if resumed else "run-start")
        self._thread = threading.Thread(
            target=self._periodic_loop,
            name="incumbent-snapshot-recorder",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def record_now(self, reason: str) -> Optional[Dict[str, Any]]:
        """Record an event snapshot at the current active wall-clock time."""

        return self._record(reason=reason, scheduled_elapsed_seconds=None, changed_only=False)

    def record_if_changed(self, reason: str) -> Optional[Dict[str, Any]]:
        """Record an event point only when the verified incumbent changed."""

        return self._record(reason=reason, scheduled_elapsed_seconds=None, changed_only=True)

    def _periodic_loop(self) -> None:
        current = self._elapsed_seconds()
        next_scheduled = (
            math.floor(current / self.interval_seconds) + 1
        ) * self.interval_seconds
        while not self._stop_event.is_set():
            delay = max(0.0, next_scheduled - self._elapsed_seconds())
            if self._stop_event.wait(delay):
                return
            self._record(
                reason="interval",
                scheduled_elapsed_seconds=next_scheduled,
                changed_only=False,
            )
            next_scheduled += self.interval_seconds

    def _record(
        self,
        *,
        reason: str,
        scheduled_elapsed_seconds: Optional[float],
        changed_only: bool,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled or self._started_at is None:
            return None
        try:
            with self._lock:
                snapshot = self._build_snapshot(
                    reason=reason,
                    scheduled_elapsed_seconds=scheduled_elapsed_seconds,
                )
                signature = self._incumbent_signature(snapshot)
                if changed_only and signature == self._last_signature:
                    return None
                self._sequence += 1
                snapshot["sequence"] = self._sequence
                self._append_jsonl(snapshot)
                self._append_csv(snapshot)
                self._last_signature = signature
                self._last_error = None
                return snapshot
        except Exception as error:
            self._last_error = "%s: %s" % (type(error).__name__, error)
            return None

    def _build_snapshot(
        self,
        *,
        reason: str,
        scheduled_elapsed_seconds: Optional[float],
    ) -> Dict[str, Any]:
        records, scan_errors = self._load_record_payloads()
        state = self._load_state()
        phase = str(state.get("phase") or "")
        incumbent = self._select_incumbent(records, state, phase)
        seed = self._select_seed(records)
        selection_basis = (
            "final-measurement"
            if phase in _FINAL_PHASES
            and incumbent is not None
            and _valid_measurement(incumbent.get("final_measurement"))
            else "search-measurement"
        )
        incumbent_latency = _comparable_latency(incumbent, selection_basis)
        seed_latency = _comparable_latency(seed, selection_basis)
        speedup = (
            seed_latency / incumbent_latency
            if seed_latency is not None and incumbent_latency is not None
            else None
        )
        return {
            "schema_version": 1,
            "timestamp": _timestamp(),
            "reason": str(reason),
            "scheduled_elapsed_seconds": (
                float(scheduled_elapsed_seconds)
                if scheduled_elapsed_seconds is not None
                else None
            ),
            "elapsed_seconds": self._elapsed_seconds(),
            "phase": phase or None,
            "completed_round": state.get("completed_round"),
            "active_round": state.get("active_round"),
            "policies": dict(state.get("experiment") or {}),
            "selection_basis": selection_basis,
            "incumbent_latency_ms": incumbent_latency,
            "seed_comparable_latency_ms": seed_latency,
            "speedup_over_seed": speedup,
            "incumbent": self._incumbent_payload(incumbent),
            "counts": {
                "generated_candidates": len(records),
                "modeled_candidates": sum(
                    item.get("model") is not None for item in records
                ),
                "measured_candidates": sum(
                    item.get("measurement") is not None for item in records
                ),
                "correct_candidates": sum(
                    _valid_measurement(item.get("measurement")) for item in records
                ),
                "profiled_candidates": sum(
                    item.get("profile") is not None for item in records
                ),
                "profile_calls": state.get("profile_calls", 0),
                "generator_calls": state.get("generator_calls", 0),
                "planner_calls": state.get("planner_calls", 0),
                "api_request_attempts": state.get("api_request_attempts", 0),
            },
            "beam_candidate_ids": list(state.get("beam") or []),
            "scan_errors": scan_errors,
        }

    def _select_incumbent(
        self,
        records: List[Dict[str, Any]],
        state: Mapping[str, Any],
        phase: str,
    ) -> Optional[Dict[str, Any]]:
        measured = [
            item for item in records if _valid_measurement(item.get("measurement"))
        ]
        if not measured:
            return None
        if phase in _FINAL_PHASES:
            identifier = state.get("best_candidate_id")
            for item in measured:
                candidate = dict(item.get("candidate") or {})
                if candidate.get("candidate_id") == identifier:
                    return item
        return min(
            measured,
            key=lambda item: (
                float(dict(item["measurement"])["latency_ms"]),
                str(dict(item.get("candidate") or {}).get("candidate_id") or ""),
            ),
        )

    @staticmethod
    def _select_seed(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        seeds = [
            item
            for item in records
            if int(dict(item.get("candidate") or {}).get("generation", -1)) == 0
            and _valid_measurement(item.get("measurement"))
        ]
        return seeds[0] if seeds else None

    def _incumbent_payload(
        self, record: Optional[Mapping[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if record is None:
            return None
        candidate = dict(record.get("candidate") or {})
        identifier = str(candidate.get("candidate_id") or "")
        source_name = str(candidate.get("source_name") or "")
        source_path = self.store.candidate_directory / identifier / source_name
        return {
            "candidate_id": identifier,
            "parent_id": candidate.get("parent_id"),
            "generation": candidate.get("generation"),
            "lineage_kind": candidate.get("lineage_kind"),
            "source_name": source_name,
            "source_sha256": candidate.get("source_sha256"),
            "source_path": str(source_path),
            "hypothesis": candidate.get("hypothesis"),
            "proposal_metadata": dict(candidate.get("proposal_metadata") or {}),
            "state": record.get("state"),
            "model": record.get("model"),
            "measurement": record.get("measurement"),
            "profile": record.get("profile"),
            "final_measurement": record.get("final_measurement"),
            "diagnosis": record.get("diagnosis"),
            "selection_reasons": list(record.get("selection_reasons") or []),
            "decision_reason": record.get("decision_reason"),
        }

    def _load_record_payloads(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        records: List[Dict[str, Any]] = []
        errors: List[str] = []
        for path in sorted(self.store.candidate_directory.glob("*/record.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    records.append(value)
                else:
                    errors.append("%s: record is not an object" % path)
            except (OSError, ValueError) as error:
                errors.append("%s: %s" % (path, error))
        return records, errors

    def _load_state(self) -> Dict[str, Any]:
        path = self.store.root / "state.json"
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def _append_jsonl(self, snapshot: Mapping[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(snapshot), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _append_csv(self, snapshot: Mapping[str, Any]) -> None:
        write_header = not self.csv_path.is_file() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_CSV_FIELDS))
            if write_header:
                writer.writeheader()
            writer.writerow(_csv_row(snapshot))
            handle.flush()
            os.fsync(handle.fileno())

    def _elapsed_seconds(self) -> float:
        if self._started_at is None:
            return self._elapsed_offset
        return max(
            self._elapsed_offset,
            self._elapsed_offset + self._clock() - self._started_at,
        )

    def _load_elapsed_offset(self) -> float:
        last = self._last_jsonl_record()
        if last is None:
            return 0.0
        try:
            return max(0.0, float(last.get("elapsed_seconds", 0.0)))
        except (TypeError, ValueError):
            return 0.0

    def _load_last_signature(self) -> Optional[Tuple[Any, Any]]:
        last = self._last_jsonl_record()
        return self._incumbent_signature(last) if last is not None else None

    def _last_jsonl_record(self) -> Optional[Dict[str, Any]]:
        if not self.jsonl_path.is_file():
            return None
        last: Optional[Dict[str, Any]] = None
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    last = value
        return last

    @staticmethod
    def _incumbent_signature(snapshot: Mapping[str, Any]) -> Tuple[Any, Any]:
        incumbent = snapshot.get("incumbent")
        if not isinstance(incumbent, Mapping):
            return (None, None)
        measurement = incumbent.get("measurement")
        latency = (
            measurement.get("latency_ms")
            if isinstance(measurement, Mapping)
            else None
        )
        return (incumbent.get("candidate_id"), latency)

    @staticmethod
    def _line_count(path: Path) -> int:
        if not path.is_file():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)


def _valid_measurement(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value.get("correct"):
        return False
    latency = value.get("latency_ms")
    return (
        isinstance(latency, (int, float))
        and not isinstance(latency, bool)
        and math.isfinite(float(latency))
        and float(latency) > 0
    )


def _comparable_latency(
    record: Optional[Mapping[str, Any]], selection_basis: str
) -> Optional[float]:
    if record is None:
        return None
    if selection_basis == "final-measurement" and _valid_measurement(
        record.get("final_measurement")
    ):
        return float(dict(record["final_measurement"])["latency_ms"])
    if _valid_measurement(record.get("measurement")):
        return float(dict(record["measurement"])["latency_ms"])
    return None


def _csv_row(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    incumbent = snapshot.get("incumbent")
    incumbent = dict(incumbent) if isinstance(incumbent, Mapping) else {}
    model = incumbent.get("model")
    model = dict(model) if isinstance(model, Mapping) else {}
    model_metrics = model.get("metrics")
    model_metrics = dict(model_metrics) if isinstance(model_metrics, Mapping) else {}
    measurement = incumbent.get("measurement")
    measurement = dict(measurement) if isinstance(measurement, Mapping) else {}
    final = incumbent.get("final_measurement")
    final = dict(final) if isinstance(final, Mapping) else {}
    profile = incumbent.get("profile")
    profile = dict(profile) if isinstance(profile, Mapping) else {}
    diagnosis = incumbent.get("diagnosis")
    diagnosis = dict(diagnosis) if isinstance(diagnosis, Mapping) else {}
    counts = snapshot.get("counts")
    counts = dict(counts) if isinstance(counts, Mapping) else {}
    policies = snapshot.get("policies")
    policies = dict(policies) if isinstance(policies, Mapping) else {}
    return {
        "sequence": snapshot.get("sequence"),
        "timestamp": snapshot.get("timestamp"),
        "reason": snapshot.get("reason"),
        "scheduled_elapsed_seconds": snapshot.get("scheduled_elapsed_seconds"),
        "elapsed_seconds": snapshot.get("elapsed_seconds"),
        "phase": snapshot.get("phase"),
        "completed_round": snapshot.get("completed_round"),
        "active_round": snapshot.get("active_round"),
        "evaluation_policy": policies.get("evaluation_policy"),
        "tir_evidence_policy": policies.get("tir_evidence_policy"),
        "strategy_allocation_policy": policies.get(
            "strategy_allocation_policy"
        ),
        "candidate_id": incumbent.get("candidate_id"),
        "generation": incumbent.get("generation"),
        "state": incumbent.get("state"),
        "selection_basis": snapshot.get("selection_basis"),
        "incumbent_latency_ms": snapshot.get("incumbent_latency_ms"),
        "measured_latency_ms": measurement.get("latency_ms"),
        "final_latency_ms": final.get("latency_ms"),
        "seed_comparable_latency_ms": snapshot.get("seed_comparable_latency_ms"),
        "speedup_over_seed": snapshot.get("speedup_over_seed"),
        "predicted_latency_ms": model.get("predicted_latency_ms"),
        "calibrated_predicted_latency_ms": model.get("calibrated_latency_ms"),
        "model_bottleneck": model.get("bottleneck"),
        "profile_bottleneck": profile.get("bottleneck"),
        "diagnosis_category": diagnosis.get("category"),
        "model_ddr_util": model_metrics.get("ddr_util"),
        "model_l2_hit_rate": model_metrics.get("l2_hit_rate"),
        "model_l2_util": model_metrics.get("l2_util"),
        "model_smem_util": model_metrics.get("smem_util"),
        "model_tensor_util": model_metrics.get("tensor_util"),
        "model_cuda_util": model_metrics.get("cuda_util"),
        "model_sfu_util": model_metrics.get("sfu_util"),
        "registers_per_thread": model_metrics.get("registers_per_thread"),
        "shared_memory_per_block": model_metrics.get("shared_memory_per_block"),
        "tiles_per_sm": model_metrics.get("tiles_per_sm"),
        "generated_candidates": counts.get("generated_candidates"),
        "modeled_candidates": counts.get("modeled_candidates"),
        "measured_candidates": counts.get("measured_candidates"),
        "correct_candidates": counts.get("correct_candidates"),
        "profiled_candidates": counts.get("profiled_candidates"),
        "profile_calls": counts.get("profile_calls"),
        "generator_calls": counts.get("generator_calls"),
        "planner_calls": counts.get("planner_calls"),
        "api_request_attempts": counts.get("api_request_attempts"),
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
