"""Run-level shared evidence memory for all source-generation parents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from .schema import CandidateRecord


@dataclass(frozen=True)
class EvidenceLesson:
    lesson_id: str
    round_number: int
    candidate_id: str
    parent_id: Optional[str]
    kind: str
    bottleneck: str
    observed_fact: str
    hypothesis_tested: str
    result: str
    confidence: str
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    source_regions: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceLesson":
        data = dict(value)
        data["supporting_evidence"] = dict(data.get("supporting_evidence") or {})
        data["source_regions"] = list(data.get("source_regions") or [])
        data["recommended_actions"] = list(data.get("recommended_actions") or [])
        return cls(**data)


class GlobalEvidenceMemory:
    """Store durable lessons and retrieve a relevant shared view per parent."""

    def __init__(self, lessons: Optional[Sequence[EvidenceLesson]] = None) -> None:
        self.lessons = list(lessons or [])
        self._lesson_ids: Set[str] = {item.lesson_id for item in self.lessons}

    def ingest(
        self,
        record: CandidateRecord,
        parent: Optional[CandidateRecord],
        round_number: int,
    ) -> Optional[EvidenceLesson]:
        lesson = self._build_lesson(record, parent, round_number)
        if lesson is None or lesson.lesson_id in self._lesson_ids:
            return None
        self.lessons.append(lesson)
        self._lesson_ids.add(lesson.lesson_id)
        return lesson

    def for_prompt(
        self, parent: CandidateRecord, limit: Optional[int] = 12
    ) -> List[Dict[str, Any]]:
        parent_bottleneck = (
            parent.diagnosis.category if parent.diagnosis is not None else "unknown"
        )

        def score(item: EvidenceLesson) -> tuple[float, int, str]:
            value = float(item.round_number)
            if item.candidate_id == parent.candidate.candidate_id:
                value += 8.0
            if item.parent_id == parent.candidate.candidate_id:
                value += 4.0
            if item.bottleneck == parent_bottleneck:
                value += 5.0
            if item.kind == "hardware-profile":
                value += 3.0
            if item.kind == "failure":
                value += 1.0
            value += {"high": 2.0, "medium": 1.0}.get(item.confidence, 0.0)
            return value, item.round_number, item.lesson_id

        selected = sorted(self.lessons, key=score, reverse=True)
        if limit is not None:
            selected = selected[:limit]
        return [item.to_dict() for item in selected]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "lessons": [item.to_dict() for item in self.lessons],
        }

    @classmethod
    def from_snapshot(
        cls, value: Optional[Mapping[str, Any]]
    ) -> "GlobalEvidenceMemory":
        data = dict(value or {})
        return cls(
            [EvidenceLesson.from_dict(item) for item in data.get("lessons", [])]
        )

    def _build_lesson(
        self,
        record: CandidateRecord,
        parent: Optional[CandidateRecord],
        round_number: int,
    ) -> Optional[EvidenceLesson]:
        diagnosis = record.diagnosis
        bottleneck = diagnosis.category if diagnosis else "unknown"
        recommendations = diagnosis.recommendations if diagnosis else []
        source_regions = record.candidate.proposal_metadata.get("changed_regions", [])
        if not isinstance(source_regions, list):
            source_regions = [str(source_regions)]

        if record.failure is not None:
            kind = "failure"
            observed = "%s failure during %s" % (
                record.failure.category,
                record.failure.stage,
            )
            result = record.failure.message
            confidence = "high"
            evidence = {
                "failure": record.failure.to_dict(),
                "diagnosis": diagnosis.to_dict() if diagnosis else None,
            }
        elif record.profile is not None and record.profile.valid:
            kind = "hardware-profile"
            observed = "NCU profile identified %s" % bottleneck
            result = diagnosis.summary if diagnosis else record.profile.bottleneck
            confidence = "high"
            evidence = {
                "profile_bottleneck": record.profile.bottleneck,
                "profile_metrics": dict(record.profile.metrics),
                "measured_latency_ms": (
                    record.measurement.latency_ms if record.measurement else None
                ),
            }
        elif record.is_measured_correct:
            kind = "measured-outcome"
            current = float(record.measurement.latency_ms)
            if parent is not None and parent.is_measured_correct:
                previous = float(parent.measurement.latency_ms)
                change = (current - previous) / previous
                direction = "improved" if change < 0 else "regressed"
                observed = "CUDA Event latency %s by %.2f%% versus parent" % (
                    direction,
                    abs(change) * 100.0,
                )
            else:
                change = None
                observed = "CUDA Event latency was measured"
            result = "latency_ms=%.9g" % current
            confidence = "high"
            evidence = {
                "measured_latency_ms": current,
                "relative_change_vs_parent": change,
                "raw_predicted_latency_ms": (
                    record.model.predicted_latency_ms if record.model else None
                ),
                "calibrated_predicted_latency_ms": (
                    record.model.calibrated_latency_ms if record.model else None
                ),
                "diagnosis": diagnosis.to_dict() if diagnosis else None,
            }
        else:
            return None

        payload = {
            "candidate_id": record.candidate.candidate_id,
            "kind": kind,
            "observed": observed,
            "result": result,
            "bottleneck": bottleneck,
        }
        identifier = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        return EvidenceLesson(
            lesson_id=identifier,
            round_number=round_number,
            candidate_id=record.candidate.candidate_id,
            parent_id=record.candidate.parent_id,
            kind=kind,
            bottleneck=bottleneck,
            observed_fact=observed,
            hypothesis_tested=record.candidate.hypothesis,
            result=result,
            confidence=confidence,
            supporting_evidence=evidence,
            source_regions=[str(item) for item in source_regions],
            recommended_actions=list(recommendations),
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
