"""Confidence-aware selection from source candidates to hardware evaluation."""

from __future__ import annotations

from difflib import SequenceMatcher
import random
from typing import Any, List, Sequence

from .schema import BudgetConfig, CandidateRecord, TaskSpec
from .trust import TrustTracker


_CONFIDENCE = {
    "high": 1.0,
    "medium": 0.65,
    "external": 0.65,
    "low": 0.3,
    "conservative": 0.2,
    "fallback": 0.0,
    "unknown": 0.0,
}


class AdaptiveSelectionPolicy:
    """Mix model exploitation with uncertainty, source diversity, and audit."""

    def __init__(self, budget: BudgetConfig) -> None:
        self.budget = budget
        self.random = random.Random(budget.random_seed)

    def promotion_count(self, available: int, trust: TrustTracker) -> int:
        if available <= 0:
            return 0
        low = self.budget.min_promotions_per_round
        high = self.budget.max_promotions_per_round
        adaptive = int(round(high - trust.score * (high - low)))
        return min(available, max(low, min(high, adaptive)))

    def snapshot(self) -> Any:
        """Return JSON-serializable random state for deterministic resume."""

        return self.random.getstate()

    def restore(self, value: Any) -> None:
        if value is not None:
            self.random.setstate(_nested_tuple(value))

    def select(
        self,
        task: TaskSpec,
        modeled: Sequence[CandidateRecord],
        measured_beam: Sequence[CandidateRecord],
        trust: TrustTracker,
    ) -> List[CandidateRecord]:
        del task
        valid = [
            record
            for record in modeled
            if record.model is not None
            and record.model.valid
            and record.model.predicted_latency_ms is not None
        ]
        target = self.promotion_count(len(valid), trust)
        if target >= len(valid):
            for record in valid:
                record.selection_reasons.append("all-available")
            return valid

        selected: List[CandidateRecord] = []
        selected_ids = set()

        def add(record: CandidateRecord, reason: str) -> None:
            identifier = record.candidate.candidate_id
            if identifier in selected_ids or len(selected) >= target:
                return
            record.selection_reasons.append(reason)
            selected.append(record)
            selected_ids.add(identifier)

        ranked = sorted(
            valid,
            key=lambda item: (
                float(item.model.predicted_latency_ms),
                item.candidate.candidate_id,
            ),
        )
        exploit_count = max(1, target // 2)
        for record in ranked[:exploit_count]:
            add(record, "model-top")

        remaining = [item for item in valid if item.candidate.candidate_id not in selected_ids]
        if remaining:
            uncertain = min(
                remaining,
                key=lambda item: (
                    _CONFIDENCE.get(item.model.confidence.lower(), 0.0),
                    item.candidate.candidate_id,
                ),
            )
            add(uncertain, "low-confidence-audit")

        remaining = [item for item in valid if item.candidate.candidate_id not in selected_ids]
        if remaining:
            references = [item.candidate.source_code for item in measured_beam]
            references.extend(item.candidate.source_code for item in selected)
            diverse = max(
                remaining,
                key=lambda item: (
                    self._distance_from_set(item.candidate.source_code, references),
                    item.candidate.candidate_id,
                ),
            )
            add(diverse, "source-diversity")

        remaining = sorted(
            [item for item in valid if item.candidate.candidate_id not in selected_ids],
            key=lambda item: item.candidate.candidate_id,
        )
        while remaining and len(selected) < target:
            index = self.random.randrange(len(remaining))
            add(remaining.pop(index), "random-audit")
        return selected

    @staticmethod
    def _distance_from_set(source_code: str, references: Sequence[str]) -> float:
        if not references:
            return 1.0
        source_lines = source_code.splitlines()
        return min(
            1.0
            - SequenceMatcher(
                None,
                source_lines,
                reference.splitlines(),
                autojunk=False,
            ).ratio()
            for reference in references
        )


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value
