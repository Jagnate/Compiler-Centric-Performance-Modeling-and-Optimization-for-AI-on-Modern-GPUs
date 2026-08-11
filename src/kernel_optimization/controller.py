"""Deterministic adaptive-fidelity optimization controller."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .archive import ArtifactStore
from .milestones import NcuMilestonePolicy
from .protocols import CandidateGenerator, PerformanceBackend
from .schema import (
    Candidate,
    CandidateRecord,
    SearchSummary,
    TaskSpec,
)
from .selection import AdaptiveSelectionPolicy
from .trust import TrustTracker


class OptimizationController:
    """Generate, model, promote, measure, profile, and archive candidates."""

    def __init__(
        self,
        task: TaskSpec,
        generator: CandidateGenerator,
        backend: PerformanceBackend,
        store: ArtifactStore,
    ) -> None:
        self.task = task
        self.generator = generator
        self.backend = backend
        self.store = store
        self.selection = AdaptiveSelectionPolicy(task.budget)
        self.milestones = NcuMilestonePolicy(task.budget)
        self.trust = TrustTracker()
        self.records: Dict[str, CandidateRecord] = {}
        self.beam: List[CandidateRecord] = []
        self.profile_calls = 0

    def run(self) -> SearchSummary:
        self.store.initialize(self.task)
        seed = self._initialize_seed()
        self.beam = [seed]
        completed_rounds = 0

        for round_number in range(1, self.task.budget.rounds + 1):
            generated = self._generate_round(round_number)
            if not generated:
                self.store.append_event(
                    "round_stopped",
                    {"round": round_number, "reason": "no-new-candidates"},
                )
                break

            modeled = self._model_candidates(generated)
            promotable = [item for item in modeled if item.model and item.model.valid]
            selected = self.selection.select(
                self.task, promotable, self.beam, self.trust
            )
            selected_ids = {item.candidate.candidate_id for item in selected}
            for record in promotable:
                if record.candidate.candidate_id not in selected_ids:
                    record.state = "modeled-not-promoted"
                    record.decision_reason = "Not selected by adaptive promotion policy."
                    self.store.save_candidate(record)

            best_before = self.beam[0]
            measured = self._measure_candidates(selected)
            self._update_beam(measured)
            best_after = self.beam[0]
            self._maybe_profile_round(
                round_number,
                best_before,
                best_after,
                measured,
            )
            completed_rounds = round_number
            self._save_round_state(round_number)

        summary = self._summary(completed_rounds, seed)
        self.store.save_summary(summary)
        self.store.append_event("run_completed", summary.to_dict())
        return summary

    def _initialize_seed(self) -> CandidateRecord:
        candidate = Candidate.seed(self.task)
        record = CandidateRecord(candidate=candidate)
        self.records[candidate.candidate_id] = record
        record.model = self.backend.model(self.task, candidate)
        if not record.model.valid:
            record.state = "seed-model-invalid"
            self.store.save_candidate(record)
            raise RuntimeError("the seed candidate is invalid according to the model backend")
        record.measurement = self.backend.measure(self.task, candidate)
        if not record.measurement.correct:
            record.state = "seed-correctness-failed"
            self.store.save_candidate(record)
            raise RuntimeError("the seed candidate failed correctness")
        record.profile = self.backend.profile(self.task, candidate)
        self.profile_calls += 1
        self.milestones.mark_profiled(record, round_number=0)
        record.state = "measured-beam"
        record.selection_reasons.append("initial-seed")
        record.decision_reason = "Verified and measured initial search seed."
        self.store.save_candidate(record)
        self.store.append_event(
            "seed_initialized",
            {
                "candidate_id": candidate.candidate_id,
                "latency_ms": record.measurement.latency_ms,
            },
        )
        return record

    def _generate_round(self, round_number: int) -> List[CandidateRecord]:
        proposals_left = self.task.budget.proposals_per_round
        parents = list(self.beam)
        generated: List[CandidateRecord] = []
        for index, parent_record in enumerate(parents):
            parents_left = len(parents) - index
            request_count = int(math.ceil(proposals_left / parents_left))
            if request_count <= 0:
                break
            proposals = self.generator.generate(
                task=self.task,
                parent=parent_record.candidate,
                evidence=self._evidence(parent_record),
                history=self._history(),
                count=request_count,
            )
            proposals_left -= len(proposals)
            for proposal in proposals:
                try:
                    candidate = Candidate.from_proposal(
                        self.task,
                        parent_record.candidate,
                        proposal,
                        generation=round_number,
                    )
                except ValueError as error:
                    self.store.append_event(
                        "proposal_rejected",
                        {
                            "round": round_number,
                            "parent_id": parent_record.candidate.candidate_id,
                            "error": str(error),
                            "proposal": proposal.to_dict(),
                        },
                    )
                    continue
                if candidate.candidate_id in self.records:
                    self.store.append_event(
                        "candidate_deduplicated",
                        {
                            "round": round_number,
                            "candidate_id": candidate.candidate_id,
                        },
                    )
                    continue
                record = CandidateRecord(candidate=candidate)
                self.records[candidate.candidate_id] = record
                self.store.save_candidate(record)
                generated.append(record)
        self.store.append_event(
            "round_generated",
            {
                "round": round_number,
                "new_candidates": len(generated),
                "requested_proposals": self.task.budget.proposals_per_round,
            },
        )
        return generated

    def _model_candidates(
        self, records: Sequence[CandidateRecord]
    ) -> List[CandidateRecord]:
        for record in records:
            record.model = self.backend.model(self.task, record.candidate)
            if record.model.valid:
                record.state = "modeled"
            else:
                record.state = "model-invalid"
                record.decision_reason = "Rejected by the low-cost model backend."
            self.store.save_candidate(record)
        return list(records)

    def _measure_candidates(
        self, records: Sequence[CandidateRecord]
    ) -> List[CandidateRecord]:
        measured: List[CandidateRecord] = []
        for record in records:
            record.measurement = self.backend.measure(self.task, record.candidate)
            parent = self.records.get(record.candidate.parent_id or "")
            if record.measurement.correct:
                record.state = "measured"
                record.decision_reason = "Correctness passed and latency was measured."
                measured.append(record)
                self.trust.update(record, parent)
            else:
                record.state = "correctness-failed"
                record.decision_reason = record.measurement.error or "Correctness failed."
            self.store.save_candidate(record)
        return measured

    def _update_beam(self, measured: Sequence[CandidateRecord]) -> None:
        pool = {item.candidate.candidate_id: item for item in self.beam}
        for record in measured:
            if record.is_measured_correct:
                pool[record.candidate.candidate_id] = record
        ranked = sorted(
            pool.values(),
            key=lambda item: (
                float(item.measurement.latency_ms),
                item.candidate.candidate_id,
            ),
        )
        self.beam = ranked[: self.task.budget.beam_width]
        beam_ids = {item.candidate.candidate_id for item in self.beam}
        for record in self.records.values():
            if not record.is_measured_correct:
                continue
            if record.candidate.candidate_id in beam_ids:
                record.state = "measured-beam"
            elif record.state == "measured-beam" or record.state == "measured":
                record.state = "measured-not-in-beam"
            self.store.save_candidate(record)

    def _maybe_profile_round(
        self,
        round_number: int,
        best_before: CandidateRecord,
        best_after: CandidateRecord,
        measured: Sequence[CandidateRecord],
    ) -> None:
        decision = self.milestones.decide(
            round_number=round_number,
            best_before=best_before,
            best_after=best_after,
            measured_this_round=measured,
            all_records=list(self.records.values()),
        )
        if decision is None:
            return
        record = self.records[decision.candidate_id]
        record.profile = self.backend.profile(self.task, record.candidate)
        record.selection_reasons.append("ncu:" + decision.reason)
        self.profile_calls += 1
        self.milestones.mark_profiled(record, round_number)
        self.store.save_candidate(record)
        self.store.append_event(
            "candidate_profiled",
            {
                "round": round_number,
                "candidate_id": decision.candidate_id,
                "reason": decision.reason,
            },
        )

    def _evidence(self, record: CandidateRecord) -> Dict[str, Any]:
        return {
            "observed": {
                "measurement": record.measurement.to_dict()
                if record.measurement
                else None,
                "profile": record.profile.to_dict() if record.profile else None,
            },
            "predicted": record.model.to_dict() if record.model else None,
            "model_trust": self.trust.to_dict(),
        }

    def _history(self) -> List[Dict[str, Any]]:
        records = sorted(
            self.records.values(),
            key=lambda item: (
                item.candidate.generation,
                item.candidate.candidate_id,
            ),
        )
        return [
            {
                "candidate_id": item.candidate.candidate_id,
                "parent_id": item.candidate.parent_id,
                "generation": item.candidate.generation,
                "parameters": item.candidate.parameters,
                "hypothesis": item.candidate.hypothesis,
                "state": item.state,
                "predicted_latency_ms": (
                    item.model.predicted_latency_ms if item.model else None
                ),
                "measured_latency_ms": (
                    item.measurement.latency_ms if item.measurement else None
                ),
            }
            for item in records[-20:]
        ]

    def _save_round_state(self, round_number: int) -> None:
        self.store.save_state(
            {
                "round": round_number,
                "beam": [item.candidate.candidate_id for item in self.beam],
                "best_candidate_id": self.beam[0].candidate.candidate_id,
                "best_latency_ms": self.beam[0].measurement.latency_ms,
                "trust": self.trust.to_dict(),
                "profile_calls": self.profile_calls,
            }
        )

    def _summary(
        self, completed_rounds: int, seed: CandidateRecord
    ) -> SearchSummary:
        best = self.beam[0]
        seed_latency = float(seed.measurement.latency_ms)
        best_latency = float(best.measurement.latency_ms)
        measured = [item for item in self.records.values() if item.measurement]
        correct = [item for item in measured if item.measurement.correct]
        return SearchSummary(
            task_id=self.task.task_id,
            best_candidate_id=best.candidate.candidate_id,
            best_latency_ms=best_latency,
            seed_latency_ms=seed_latency,
            speedup_over_seed=seed_latency / best_latency,
            completed_rounds=completed_rounds,
            generated_candidates=len(self.records),
            modeled_candidates=sum(
                item.model is not None for item in self.records.values()
            ),
            measured_candidates=len(measured),
            correct_candidates=len(correct),
            profile_calls=self.profile_calls,
            trust=self.trust.to_dict(),
            output_directory=str(self.store.root),
        )

