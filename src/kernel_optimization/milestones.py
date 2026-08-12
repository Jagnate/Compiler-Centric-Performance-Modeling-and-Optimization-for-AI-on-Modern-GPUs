"""Event-triggered policy for expensive hardware profiling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from .schema import BudgetConfig, CandidateRecord


@dataclass(frozen=True)
class ProfileDecision:
    candidate_id: str
    reason: str


class NcuMilestonePolicy:
    """Choose at most one new NCU profile per optimization round."""

    def __init__(self, budget: BudgetConfig) -> None:
        self.budget = budget
        self.last_profile_round = 0
        self.plateau_rounds = 0
        self.profiled_ids: Set[str] = set()

    def mark_profiled(self, record: CandidateRecord, round_number: int) -> None:
        self.profiled_ids.add(record.candidate.candidate_id)
        self.last_profile_round = round_number

    def snapshot(self) -> Dict[str, Any]:
        return {
            "last_profile_round": self.last_profile_round,
            "plateau_rounds": self.plateau_rounds,
            "profiled_ids": sorted(self.profiled_ids),
        }

    def restore(self, value: Optional[Mapping[str, Any]]) -> None:
        data = dict(value or {})
        self.last_profile_round = int(data.get("last_profile_round", 0))
        self.plateau_rounds = int(data.get("plateau_rounds", 0))
        self.profiled_ids = set(str(item) for item in data.get("profiled_ids", []))

    def decide(
        self,
        round_number: int,
        best_before: CandidateRecord,
        best_after: CandidateRecord,
        measured_this_round: Sequence[CandidateRecord],
        all_records: Sequence[CandidateRecord],
    ) -> Optional[ProfileDecision]:
        before_latency = best_before.measurement.latency_ms
        after_latency = best_after.measurement.latency_ms
        improved = (
            before_latency is not None
            and after_latency is not None
            and after_latency < before_latency
        )
        if improved:
            self.plateau_rounds = 0
            improvement = (before_latency - after_latency) / before_latency
            if (
                improvement >= self.budget.ncu_improvement_threshold
                and best_after.candidate.candidate_id not in self.profiled_ids
            ):
                return ProfileDecision(
                    best_after.candidate.candidate_id,
                    "meaningful-measured-improvement",
                )
        else:
            self.plateau_rounds += 1

        disagreement = self._largest_disagreement(measured_this_round, all_records)
        if disagreement is not None:
            return ProfileDecision(
                disagreement.candidate.candidate_id,
                "model-measurement-disagreement",
            )

        low_confidence = [
            item
            for item in measured_this_round
            if item.candidate.candidate_id not in self.profiled_ids
            and item.model is not None
            and item.model.confidence.lower()
            in {"low", "conservative", "fallback", "unknown"}
        ]
        if low_confidence:
            candidate = min(low_confidence, key=lambda item: item.measurement.latency_ms)
            return ProfileDecision(candidate.candidate.candidate_id, "low-model-confidence")

        stale = (
            round_number - self.last_profile_round
            >= self.budget.ncu_max_staleness_rounds
        )
        plateau = self.plateau_rounds >= self.budget.ncu_plateau_rounds
        if stale or plateau:
            unprofiled = [
                item
                for item in all_records
                if item.is_measured_correct
                and item.candidate.candidate_id not in self.profiled_ids
            ]
            if unprofiled:
                candidate = min(unprofiled, key=lambda item: item.measurement.latency_ms)
                return ProfileDecision(
                    candidate.candidate.candidate_id,
                    "profile-staleness" if stale else "search-plateau",
                )
        return None

    def _largest_disagreement(
        self,
        measured: Sequence[CandidateRecord],
        all_records: Sequence[CandidateRecord],
    ) -> Optional[CandidateRecord]:
        by_id = {item.candidate.candidate_id: item for item in all_records}
        candidates = []
        for record in measured:
            if record.candidate.candidate_id in self.profiled_ids:
                continue
            parent = by_id.get(record.candidate.parent_id or "")
            if (
                parent is None
                or record.model is None
                or parent.model is None
                or record.measurement is None
                or parent.measurement is None
                or record.model.predicted_latency_ms is None
                or parent.model.predicted_latency_ms is None
                or record.measurement.latency_ms is None
                or parent.measurement.latency_ms is None
            ):
                continue
            predicted_improves = (
                record.model.predicted_latency_ms < parent.model.predicted_latency_ms
            )
            measured_improves = record.measurement.latency_ms < parent.measurement.latency_ms
            if predicted_improves != measured_improves:
                magnitude = abs(
                    (record.model.predicted_latency_ms - parent.model.predicted_latency_ms)
                    / parent.model.predicted_latency_ms
                )
                candidates.append((magnitude, record.candidate.candidate_id, record))
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1]))[2]
