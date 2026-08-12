"""Budget-controlled policies for source-to-hardware candidate promotion."""

from __future__ import annotations

from difflib import SequenceMatcher
import random
from typing import Any, List, Optional, Sequence

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


class SelectionPolicy:
    """Common promotion-budget behavior shared by experiment strategies."""

    name = "base"

    def __init__(
        self,
        budget: BudgetConfig,
        fixed_promotions_per_round: Optional[int] = None,
    ) -> None:
        if fixed_promotions_per_round is not None and fixed_promotions_per_round <= 0:
            raise ValueError("fixed_promotions_per_round must be positive")
        self.budget = budget
        self.fixed_promotions_per_round = fixed_promotions_per_round
        self.random = random.Random(budget.random_seed)

    def promotion_count(self, available: int, trust: TrustTracker) -> int:
        if available <= 0:
            return 0
        if self.fixed_promotions_per_round is not None:
            return min(available, self.fixed_promotions_per_round)
        low = self.budget.min_promotions_per_round
        high = self.budget.max_promotions_per_round
        adaptive = int(round(high - trust.score * (high - low)))
        return min(available, max(low, min(high, adaptive)))

    def snapshot(self) -> Any:
        """Return JSON-serializable policy state for deterministic resume."""

        return {"random_state": self.random.getstate()}

    def restore(self, value: Any) -> None:
        if isinstance(value, dict):
            value = value.get("random_state")
        if value is not None:
            self.random.setstate(_nested_tuple(value))

    @staticmethod
    def _valid(modeled: Sequence[CandidateRecord]) -> List[CandidateRecord]:
        return [
            record
            for record in modeled
            if record.model is not None
            and record.model.valid
            and record.model.ranking_latency_ms is not None
            and record.compiled_equivalent_to is None
        ]


class AdaptiveSelectionPolicy(SelectionPolicy):
    """Mix model exploitation with uncertainty, source diversity, and audit."""

    name = "adaptive"

    def select(
        self,
        task: TaskSpec,
        modeled: Sequence[CandidateRecord],
        measured_beam: Sequence[CandidateRecord],
        trust: TrustTracker,
    ) -> List[CandidateRecord]:
        del task
        valid = self._valid(modeled)
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
                float(item.model.ranking_latency_ms),
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


class ModelTopSelectionPolicy(SelectionPolicy):
    """Promote only the lowest model-ranked candidates under the same budget."""

    name = "model-top"

    def select(self, task, modeled, measured_beam, trust):
        del task, measured_beam
        valid = sorted(
            self._valid(modeled),
            key=lambda item: (
                float(item.model.ranking_latency_ms),
                item.candidate.candidate_id,
            ),
        )
        selected = valid[: self.promotion_count(len(valid), trust)]
        for record in selected:
            record.selection_reasons.append("model-top-only")
        return selected


class RandomSelectionPolicy(SelectionPolicy):
    """Deterministic random hardware allocation for an equal-budget baseline."""

    name = "random"

    def select(self, task, modeled, measured_beam, trust):
        del task, measured_beam
        remaining = sorted(
            self._valid(modeled), key=lambda item: item.candidate.candidate_id
        )
        target = self.promotion_count(len(remaining), trust)
        selected: List[CandidateRecord] = []
        while remaining and len(selected) < target:
            selected.append(remaining.pop(self.random.randrange(len(remaining))))
        for record in selected:
            record.selection_reasons.append("random-baseline")
        return selected


class MeasureAllSelectionPolicy(SelectionPolicy):
    """Measure every model-valid candidate as a non-equal-budget upper baseline."""

    name = "measure-all"

    def promotion_count(self, available: int, trust: TrustTracker) -> int:
        del trust
        return available

    def select(self, task, modeled, measured_beam, trust):
        del task, measured_beam, trust
        selected = self._valid(modeled)
        for record in selected:
            record.selection_reasons.append("measure-all-baseline")
        return selected


def create_selection_policy(
    name: str,
    budget: BudgetConfig,
    fixed_promotions_per_round: Optional[int] = None,
) -> SelectionPolicy:
    policies = {
        "adaptive": AdaptiveSelectionPolicy,
        "model-top": ModelTopSelectionPolicy,
        "random": RandomSelectionPolicy,
        "measure-all": MeasureAllSelectionPolicy,
    }
    try:
        policy_type = policies[name]
    except KeyError as error:
        raise ValueError("unsupported selection policy %r" % name) from error
    return policy_type(budget, fixed_promotions_per_round)


def _nested_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value
