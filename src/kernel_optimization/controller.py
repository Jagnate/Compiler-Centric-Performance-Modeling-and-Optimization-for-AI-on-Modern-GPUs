"""Adaptive-fidelity controller for API-generated kernel source candidates."""

from __future__ import annotations

import math
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TypeVar

from .archive import ArtifactStore
from .calibration import LatencyCalibrator
from .diagnosis import BottleneckAnalyzer, FailureClassifier
from .evidence import GlobalEvidenceMemory
from .milestones import NcuMilestonePolicy
from .progress import NullProgressReporter, ProgressReporter
from .protocols import CandidateGenerator, PerformanceBackend
from .schema import (
    Candidate,
    CandidateProposal,
    CandidateRecord,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    SearchSummary,
    TaskSpec,
)
from .selection import AdaptiveSelectionPolicy
from .source_validation import SourceValidationError, SourceValidator
from .trust import TrustTracker


T = TypeVar("T")

_PHASE_ORDER = {
    "": -1,
    "seed_completed": 0,
    "generated": 1,
    "modeled": 2,
    "selected": 3,
    "measured": 4,
    "beam_updated": 5,
    "round_completed": 6,
    "search_stopped": 7,
    "run_completed": 8,
}


class OptimizationController:
    """Generate source, model, promote, measure, profile, and archive candidates."""

    def __init__(
        self,
        task: TaskSpec,
        source_code: str,
        source_name: str,
        generator: CandidateGenerator,
        backend: PerformanceBackend,
        store: ArtifactStore,
        source_validator: Optional[SourceValidator] = None,
        run_metadata: Optional[Mapping[str, Any]] = None,
        progress: Optional[ProgressReporter] = None,
        resume: bool = False,
        preflight_calls: int = 0,
    ) -> None:
        self.task = task
        self.source_code = source_code
        self.source_name = source_name
        self.generator = generator
        self.backend = backend
        self.store = store
        self.source_validator = source_validator or SourceValidator()
        self.run_metadata = dict(run_metadata or {})
        self.progress = progress or NullProgressReporter()
        self.resume = resume
        self.preflight_calls = preflight_calls
        self.selection = AdaptiveSelectionPolicy(task.budget)
        self.milestones = NcuMilestonePolicy(task.budget)
        self.trust = TrustTracker()
        self.calibrator = LatencyCalibrator(task.budget.calibration_min_samples)
        self.analyzer = BottleneckAnalyzer()
        self.evidence_memory = GlobalEvidenceMemory()
        self.records: Dict[str, CandidateRecord] = {}
        self.beam: List[CandidateRecord] = []
        self.profile_calls = 0
        self.generator_calls = 0
        self.generator_usage: Dict[str, float] = {}
        self.stage_timings: Dict[str, Dict[str, float]] = {}
        self.repair_calls = 0
        self.repair_candidates = 0
        self._repairs_by_round: Dict[int, int] = {}
        self.was_resumed = False

    def run(self) -> SearchSummary:
        started_at = time.perf_counter()
        try:
            seed_candidate = Candidate.seed(
                self.task,
                source_code=self.source_code,
                source_name=self.source_name,
            )
            self.source_validator.validate(
                self.task,
                seed_candidate.source_code,
                seed_candidate.source_name,
            )
            self.store.initialize(self.task, seed_candidate, self.run_metadata)

            state = self.store.load_state() if self.resume else None
            if state is not None:
                seed = self._restore(seed_candidate, state)
                completed_rounds = int(state.get("completed_round", 0))
                self.progress.emit(
                    "run_resumed",
                    "Restored optimization state.",
                    completed_rounds=completed_rounds,
                    phase=state.get("phase"),
                    candidates=len(self.records),
                )
            else:
                self.progress.emit("seed_started", "Evaluating the input kernel.")
                seed = self._initialize_seed(seed_candidate)
                self.beam = [seed]
                completed_rounds = 0
                state = self._checkpoint(
                    phase="seed_completed",
                    completed_round=0,
                    active_round=None,
                )

            active_round = state.get("active_round") if state else None
            start_round = (
                int(active_round)
                if active_round is not None
                else completed_rounds + 1
            )
            for round_number in range(start_round, self.task.budget.rounds + 1):
                round_state = (
                    state
                    if state and state.get("active_round") == round_number
                    else {}
                )
                completed = self._run_round(round_number, round_state)
                state = self.store.load_state() or {}
                if not completed:
                    break
                completed_rounds = round_number

            best_source_path = self.store.save_best(self.beam[0])
            summary = self._summary(
                completed_rounds,
                seed,
                best_source_path,
                elapsed_seconds=time.perf_counter() - started_at,
            )
            self.store.save_summary(summary)
            self._checkpoint(
                phase="run_completed",
                completed_round=completed_rounds,
                active_round=None,
            )
            self.store.append_event("run_completed", summary.to_dict())
            self.progress.emit(
                "run_completed",
                "Optimization finished.",
                best_candidate_id=summary.best_candidate_id,
                best_latency_ms=summary.best_latency_ms,
                speedup=summary.speedup_over_seed,
            )
            return summary
        except BaseException as error:
            failure = {
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "resumable": self.store.load_state() is not None,
                "last_checkpoint": self.store.load_state(),
            }
            self.store.save_failure(failure)
            self.store.append_event("run_failed", failure)
            self.progress.emit(
                "run_failed",
                "Optimization stopped with an error.",
                error_type=type(error).__name__,
                error_message=str(error),
            )
            raise

    def _run_round(
        self, round_number: int, resume_state: Mapping[str, Any]
    ) -> bool:
        phase = str(resume_state.get("phase", ""))
        candidate_ids = [str(item) for item in resume_state.get("round_candidate_ids", [])]
        selected_ids = [str(item) for item in resume_state.get("selected_ids", [])]
        best_before_id = resume_state.get("best_before_id")
        self.progress.emit(
            "round_started",
            "Starting search round %d." % round_number,
            round=round_number,
            resumed=bool(resume_state),
        )

        if not self._phase_at_least(phase, "generated"):
            generated = self._generate_round(round_number)
            candidate_ids = [item.candidate.candidate_id for item in generated]
            if not generated:
                self.store.append_event(
                    "round_stopped",
                    {"round": round_number, "reason": "no-new-valid-source-candidates"},
                )
                self._checkpoint(
                    phase="search_stopped",
                    completed_round=round_number - 1,
                    active_round=None,
                )
                self.progress.emit(
                    "search_stopped",
                    "No new valid source candidates were produced.",
                    round=round_number,
                )
                return False
            resume_state = self._checkpoint(
                phase="generated",
                completed_round=round_number - 1,
                active_round=round_number,
                round_candidate_ids=candidate_ids,
            )
            phase = "generated"
        generated = self._records_for_ids(candidate_ids)

        if not self._phase_at_least(phase, "modeled"):
            modeled = self._model_candidates(generated, round_number)
            repaired = self._repair_model_failures(modeled, round_number)
            modeled.extend(repaired)
            candidate_ids = _unique(
                candidate_ids
                + [item.candidate.candidate_id for item in repaired]
            )
            resume_state = self._checkpoint(
                phase="modeled",
                completed_round=round_number - 1,
                active_round=round_number,
                round_candidate_ids=candidate_ids,
            )
            phase = "modeled"
        else:
            modeled = generated

        promotable = [item for item in modeled if item.model and item.model.valid]
        if not self._phase_at_least(phase, "selected"):
            selected = self.selection.select(
                self.task, promotable, self.beam, self.trust
            )
            selected_ids = [item.candidate.candidate_id for item in selected]
            selected_set = set(selected_ids)
            for record in promotable:
                if record.candidate.candidate_id not in selected_set:
                    record.state = "modeled-not-promoted"
                    record.decision_reason = (
                        "Not selected by adaptive promotion policy."
                    )
                    self.store.save_candidate(record)
            best_before_id = self.beam[0].candidate.candidate_id
            resume_state = self._checkpoint(
                phase="selected",
                completed_round=round_number - 1,
                active_round=round_number,
                round_candidate_ids=candidate_ids,
                selected_ids=selected_ids,
                best_before_id=best_before_id,
            )
            phase = "selected"
            self.progress.emit(
                "candidates_selected",
                "Promoted candidates to hardware evaluation.",
                round=round_number,
                selected=len(selected_ids),
                modeled=len(promotable),
            )
        selected = self._records_for_ids(selected_ids)

        if not self._phase_at_least(phase, "measured"):
            measured = self._measure_candidates(selected, round_number)
            repair_records, repair_measured = self._repair_measurement_failures(
                selected, round_number
            )
            measured.extend(repair_measured)
            candidate_ids = _unique(
                candidate_ids
                + [item.candidate.candidate_id for item in repair_records]
            )
            selected_ids = _unique(
                selected_ids
                + [item.candidate.candidate_id for item in repair_records]
            )
            resume_state = self._checkpoint(
                phase="measured",
                completed_round=round_number - 1,
                active_round=round_number,
                round_candidate_ids=candidate_ids,
                selected_ids=selected_ids,
                best_before_id=best_before_id,
            )
            phase = "measured"
        else:
            measured = [item for item in selected if item.is_measured_correct]
            self._rebuild_trust()

        if not self._phase_at_least(phase, "beam_updated"):
            best_before = self.records[str(best_before_id)]
            self._update_beam(measured)
            resume_state = self._checkpoint(
                phase="beam_updated",
                completed_round=round_number - 1,
                active_round=round_number,
                round_candidate_ids=candidate_ids,
                selected_ids=selected_ids,
                best_before_id=best_before_id,
            )
            phase = "beam_updated"
        else:
            best_before = self.records[str(best_before_id)]

        if not self._phase_at_least(phase, "round_completed"):
            self._maybe_profile_round(
                round_number,
                best_before,
                self.beam[0],
                measured,
            )
            self._checkpoint(
                phase="round_completed",
                completed_round=round_number,
                active_round=None,
                round_candidate_ids=candidate_ids,
                selected_ids=selected_ids,
                best_before_id=best_before_id,
            )
        self.progress.emit(
            "round_completed",
            "Finished search round %d." % round_number,
            round=round_number,
            best_latency_ms=self.beam[0].measurement.latency_ms,
            beam=[item.candidate.candidate_id for item in self.beam],
        )
        return True

    def _initialize_seed(self, candidate: Candidate) -> CandidateRecord:
        record = CandidateRecord(candidate=candidate)
        self.records[candidate.candidate_id] = record
        self.store.save_candidate(record)
        try:
            record.model = self._evaluate(
                "model", candidate, lambda: self.backend.model(self.task, candidate)
            )
        except Exception as error:
            record.model = ModelEvaluation(
                valid=False,
                bottleneck="compile-or-model-failure",
                diagnostics=["%s: %s" % (type(error).__name__, error)],
            )
        if not record.model.valid:
            record.failure = FailureClassifier.model(record.model)
            self._diagnose_and_remember(record, None, round_number=0)
            record.state = "seed-model-invalid"
            self.store.save_candidate(record)
            raise RuntimeError("the input kernel is invalid according to the model backend")
        record.model = self.calibrator.apply(self.task, record.model)
        try:
            record.measurement = self._evaluate(
                "measure", candidate, lambda: self.backend.measure(self.task, candidate)
            )
        except Exception as error:
            record.measurement = Measurement(
                correct=False,
                error="%s: %s" % (type(error).__name__, error),
            )
        if not record.measurement.correct:
            record.failure = FailureClassifier.measurement(record.measurement)
            self._diagnose_and_remember(record, None, round_number=0)
            record.state = "seed-correctness-failed"
            self.store.save_candidate(record)
            raise RuntimeError("the input kernel failed correctness")
        self.calibrator.observe(self.task, record)
        record.profile = self._profile_candidate(candidate)
        if record.profile.valid:
            self.milestones.mark_profiled(record, round_number=0)
        record.state = "measured-beam"
        record.selection_reasons.append("initial-source")
        record.decision_reason = "Verified and measured input kernel."
        self._diagnose_and_remember(record, None, round_number=0)
        self.store.save_candidate(record)
        self.store.append_event(
            "seed_initialized",
            {
                "candidate_id": candidate.candidate_id,
                "source_sha256": candidate.source_sha256,
                "latency_ms": record.measurement.latency_ms,
                "profile_valid": record.profile.valid,
            },
        )
        self.progress.emit(
            "seed_completed",
            "Input kernel passed correctness and timing.",
            candidate_id=candidate.candidate_id,
            latency_ms=record.measurement.latency_ms,
            profile_valid=record.profile.valid,
        )
        return record

    def _restore(
        self, expected_seed: Candidate, state: Mapping[str, Any]
    ) -> CandidateRecord:
        self.records = self.store.load_candidates()
        seed_records = [
            item for item in self.records.values() if item.candidate.generation == 0
        ]
        if len(seed_records) != 1:
            raise ValueError("resume requires exactly one archived seed candidate")
        seed = seed_records[0]
        if seed.candidate.candidate_id != expected_seed.candidate_id:
            raise ValueError("resume input source does not match the archived seed")
        beam_ids = [str(item) for item in state.get("beam", [])]
        if not beam_ids:
            beam_ids = [
                item.candidate.candidate_id
                for item in sorted(
                    (record for record in self.records.values() if record.is_measured_correct),
                    key=lambda record: (
                        float(record.measurement.latency_ms),
                        record.candidate.candidate_id,
                    ),
                )[: self.task.budget.beam_width]
            ]
        self.beam = self._records_for_ids(beam_ids)
        if not self.beam:
            raise ValueError("resume checkpoint has no measured beam")
        self.profile_calls = int(state.get("profile_calls", 0))
        self.generator_calls = int(state.get("generator_calls", 0))
        self.generator_usage = {
            str(name): float(value)
            for name, value in dict(state.get("generator_usage") or {}).items()
        }
        self.stage_timings = {
            str(name): {
                str(metric): float(value)
                for metric, value in dict(metrics).items()
            }
            for name, metrics in dict(state.get("stage_timings") or {}).items()
        }
        self.repair_calls = int(state.get("repair_calls", 0))
        self.repair_candidates = int(state.get("repair_candidates", 0))
        self._repairs_by_round = {
            int(name): int(value)
            for name, value in dict(state.get("repairs_by_round") or {}).items()
        }
        self.calibrator = LatencyCalibrator.from_snapshot(
            state.get("calibration"), self.task.budget.calibration_min_samples
        )
        if not self.calibrator.observations:
            for record in sorted(
                self.records.values(),
                key=lambda item: (
                    item.candidate.generation,
                    item.candidate.candidate_id,
                ),
            ):
                self.calibrator.observe(self.task, record)
        self.evidence_memory = GlobalEvidenceMemory.from_snapshot(
            state.get("evidence_memory")
        )
        if not self.evidence_memory.lessons:
            for record in sorted(
                self.records.values(),
                key=lambda item: (
                    item.candidate.generation,
                    item.candidate.candidate_id,
                ),
            ):
                if record.diagnosis is not None:
                    parent = self.records.get(record.candidate.parent_id or "")
                    self.evidence_memory.ingest(
                        record, parent, record.candidate.generation
                    )
        self.trust = TrustTracker.from_snapshot(state.get("trust_snapshot"))
        self._rebuild_trust()
        self.milestones.restore(state.get("milestones"))
        self.selection.restore(state.get("selection_random_state"))
        self.preflight_calls += int(state.get("preflight_calls", 0))
        self.was_resumed = True
        return seed

    def _generate_round(self, round_number: int) -> List[CandidateRecord]:
        existing = [
            item
            for item in self.records.values()
            if item.candidate.generation == round_number
            and item.candidate.lineage_kind == "proposal"
        ]
        proposals_left = max(
            0, self.task.budget.proposals_per_round - len(existing)
        )
        parents = list(self.beam)
        static_failures = [
            item
            for item in self.records.values()
            if item.candidate.generation == round_number
            and item.state == "static-invalid"
        ]
        for index, parent_record in enumerate(parents):
            parents_left = len(parents) - index
            request_count = int(math.ceil(proposals_left / parents_left))
            if request_count <= 0:
                break
            proposals = self._generate_from_parent(
                round_number, parent_record, request_count
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
                            "hypothesis": proposal.hypothesis,
                            "error": str(error),
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
                try:
                    self.source_validator.validate(
                        self.task,
                        candidate.source_code,
                        candidate.source_name,
                    )
                except SourceValidationError as error:
                    record.state = "static-invalid"
                    record.failure = FailureClassifier.static(error)
                    record.decision_reason = str(error)
                    self._diagnose_and_remember(
                        record, parent_record, round_number
                    )
                    static_failures.append(record)
                    self.store.append_event(
                        "proposal_rejected",
                        {
                            "round": round_number,
                            "candidate_id": candidate.candidate_id,
                            "parent_id": parent_record.candidate.candidate_id,
                            "hypothesis": proposal.hypothesis,
                            "failure": record.failure.to_dict(),
                        },
                    )
                self.store.save_candidate(record)

        repair_queue = list(static_failures)
        while repair_queue and self._repair_budget_available(round_number):
            failed = repair_queue.pop(0)
            repaired = self._repair_candidate(failed, round_number)
            if repaired is not None and repaired.state == "static-invalid":
                repair_queue.append(repaired)

        generated = sorted(
            (
                item
                for item in self.records.values()
                if item.candidate.generation == round_number
                and item.failure is None
            ),
            key=lambda item: item.candidate.candidate_id,
        )
        self.store.append_event(
            "round_generated",
            {
                "round": round_number,
                "new_candidates": len(generated),
                "static_failures": len(static_failures),
                "repair_candidates": sum(
                    item.candidate.lineage_kind == "repair"
                    for item in self.records.values()
                    if item.candidate.generation == round_number
                ),
                "requested_proposals": self.task.budget.proposals_per_round,
            },
        )
        self.progress.emit(
            "generation_done",
            "Generated source candidates.",
            round=round_number,
            candidates=len(generated),
        )
        return generated

    def _generate_from_parent(
        self,
        round_number: int,
        parent_record: CandidateRecord,
        request_count: int,
    ):
        self.generator_calls += 1
        started_at = time.perf_counter()
        self.progress.emit(
            "api_started",
            "Requesting candidate source files.",
            round=round_number,
            parent_id=parent_record.candidate.candidate_id,
            requested=request_count,
        )
        try:
            proposals = self.generator.generate(
                task=self.task,
                parent=parent_record.candidate,
                evidence=self._evidence(parent_record),
                history=self._history(),
                count=request_count,
            )
        except Exception as error:
            elapsed = time.perf_counter() - started_at
            self._record_stage_timing("api_generate", elapsed, failed=True)
            metadata = dict(
                getattr(self.generator, "last_call_metadata", {}) or {}
            )
            self.store.save_api_call(
                "round-%03d-generate-failed" % round_number,
                {
                    "round": round_number,
                    "parent_id": parent_record.candidate.candidate_id,
                    "requested_candidates": request_count,
                    "provider": metadata,
                    "error": "%s: %s" % (type(error).__name__, error),
                },
                getattr(self.generator, "last_exchange", None),
            )
            self.store.append_event(
                "generator_failed",
                {
                    "round": round_number,
                    "parent_id": parent_record.candidate.candidate_id,
                    "error": "%s: %s" % (type(error).__name__, error),
                    "provider": metadata,
                },
            )
            self._persist_runtime_counters()
            raise
        elapsed = time.perf_counter() - started_at
        self._record_stage_timing("api_generate", elapsed, failed=False)
        metadata = dict(getattr(self.generator, "last_call_metadata", {}) or {})
        self._accumulate_generator_usage(metadata.get("usage"))
        self.store.save_api_call(
            "round-%03d-generate" % round_number,
            {
                "round": round_number,
                "parent_id": parent_record.candidate.candidate_id,
                "requested_candidates": request_count,
                "returned_candidates": len(proposals),
                "provider": metadata,
            },
            getattr(self.generator, "last_exchange", None),
        )
        self.store.append_event(
            "generator_called",
            {
                "round": round_number,
                "parent_id": parent_record.candidate.candidate_id,
                "requested_candidates": request_count,
                "returned_candidates": len(proposals),
                "provider": metadata,
            },
        )
        self.progress.emit(
            "api_completed",
            "Hosted API returned candidate sources.",
            round=round_number,
            returned=len(proposals),
            elapsed_seconds=round(elapsed, 3),
        )
        return proposals

    def _repair_candidate(
        self, failed: CandidateRecord, round_number: int
    ) -> Optional[CandidateRecord]:
        repair_method = getattr(self.generator, "repair", None)
        failure = failed.failure
        if (
            not callable(repair_method)
            or failure is None
            or not failure.retryable
            or failed.candidate.repair_depth >= self.task.budget.max_repair_depth
            or not self._repair_budget_available(round_number)
        ):
            return None

        self.repair_calls += 1
        self.generator_calls += 1
        self._repairs_by_round[round_number] = (
            self._repairs_by_round.get(round_number, 0) + 1
        )
        started_at = time.perf_counter()
        self.progress.emit(
            "repair_started",
            "Requesting a bounded source repair.",
            round=round_number,
            failed_candidate_id=failed.candidate.candidate_id,
            failure_category=failure.category,
            repair_depth=failed.candidate.repair_depth + 1,
        )
        failure_payload = {
            "failure": failure.to_dict(),
            "diagnosis": failed.diagnosis.to_dict() if failed.diagnosis else None,
        }
        try:
            proposal: CandidateProposal = repair_method(
                task=self.task,
                failed=failed.candidate,
                failure=failure_payload,
                evidence=self._evidence(failed),
                history=self._history(),
            )
        except Exception as error:
            elapsed = time.perf_counter() - started_at
            self._record_stage_timing("api_repair", elapsed, failed=True)
            metadata = dict(
                getattr(self.generator, "last_call_metadata", {}) or {}
            )
            self.store.save_api_call(
                "round-%03d-repair-failed" % round_number,
                {
                    "round": round_number,
                    "failed_candidate_id": failed.candidate.candidate_id,
                    "failure": failure_payload,
                    "provider": metadata,
                    "error": "%s: %s" % (type(error).__name__, error),
                },
                getattr(self.generator, "last_exchange", None),
            )
            self.store.append_event(
                "repair_api_failed",
                {
                    "round": round_number,
                    "failed_candidate_id": failed.candidate.candidate_id,
                    "error": "%s: %s" % (type(error).__name__, error),
                    "provider": metadata,
                },
            )
            self._persist_runtime_counters()
            return None

        elapsed = time.perf_counter() - started_at
        self._record_stage_timing("api_repair", elapsed, failed=False)
        metadata = dict(getattr(self.generator, "last_call_metadata", {}) or {})
        self._accumulate_generator_usage(metadata.get("usage"))
        self.store.save_api_call(
            "round-%03d-repair" % round_number,
            {
                "round": round_number,
                "failed_candidate_id": failed.candidate.candidate_id,
                "failure": failure_payload,
                "provider": metadata,
            },
            getattr(self.generator, "last_exchange", None),
        )

        try:
            candidate = Candidate.from_repair(
                self.task,
                failed.candidate,
                proposal,
                generation=round_number,
            )
        except ValueError as error:
            self.store.append_event(
                "repair_rejected",
                {
                    "round": round_number,
                    "failed_candidate_id": failed.candidate.candidate_id,
                    "error": str(error),
                },
            )
            return None
        if candidate.candidate_id in self.records:
            self.store.append_event(
                "repair_deduplicated",
                {
                    "round": round_number,
                    "failed_candidate_id": failed.candidate.candidate_id,
                    "candidate_id": candidate.candidate_id,
                },
            )
            return None

        record = CandidateRecord(candidate=candidate)
        self.records[candidate.candidate_id] = record
        self.repair_candidates += 1
        try:
            self.source_validator.validate(
                self.task, candidate.source_code, candidate.source_name
            )
        except SourceValidationError as error:
            record.state = "static-invalid"
            record.failure = FailureClassifier.static(error)
            record.decision_reason = str(error)
            self._diagnose_and_remember(record, failed, round_number)
            event = "repair_static_failed"
        else:
            record.state = "generated"
            record.decision_reason = "Generated by bounded repair."
            self.store.save_candidate(record)
            event = "repair_generated"
        self.store.append_event(
            event,
            {
                "round": round_number,
                "failed_candidate_id": failed.candidate.candidate_id,
                "candidate_id": candidate.candidate_id,
                "repair_depth": candidate.repair_depth,
                "elapsed_seconds": elapsed,
            },
        )
        self.progress.emit(
            event,
            "Repair candidate materialized.",
            round=round_number,
            candidate_id=candidate.candidate_id,
            static_valid=record.failure is None,
        )
        return record

    def _repair_budget_available(self, round_number: int) -> bool:
        return (
            self.task.budget.max_repairs_per_round > 0
            and self.task.budget.max_repair_depth > 0
            and self._repairs_by_round.get(round_number, 0)
            < self.task.budget.max_repairs_per_round
        )

    def _model_candidates(
        self, records: Sequence[CandidateRecord], round_number: int
    ) -> List[CandidateRecord]:
        for index, record in enumerate(records, start=1):
            if record.model is not None:
                continue
            self.progress.emit(
                "model_progress",
                "Running analytical model.",
                round=round_number,
                candidate=index,
                total=len(records),
                candidate_id=record.candidate.candidate_id,
            )
            try:
                record.model = self._evaluate(
                    "model",
                    record.candidate,
                    lambda record=record: self.backend.model(
                        self.task, record.candidate
                    ),
                )
            except Exception as error:
                record.model = ModelEvaluation(
                    valid=False,
                    bottleneck="compile-or-model-failure",
                    diagnostics=["%s: %s" % (type(error).__name__, error)],
                )
            if record.model.valid:
                record.model = self.calibrator.apply(self.task, record.model)
                record.state = "modeled"
                record.failure = None
                parent = self.records.get(record.candidate.parent_id or "")
                record.diagnosis = self.analyzer.diagnose(record, parent)
            else:
                record.state = "model-invalid"
                record.decision_reason = "Rejected by compiler or performance model."
                record.failure = FailureClassifier.model(record.model)
                parent = self.records.get(record.candidate.parent_id or "")
                self._diagnose_and_remember(record, parent, round_number)
            self.store.save_candidate(record)
        return list(records)

    def _repair_model_failures(
        self, records: Sequence[CandidateRecord], round_number: int
    ) -> List[CandidateRecord]:
        repaired_records: List[CandidateRecord] = []
        queue = [item for item in records if item.failure is not None]
        while queue and self._repair_budget_available(round_number):
            failed = queue.pop(0)
            repaired = self._repair_candidate(failed, round_number)
            if repaired is None:
                continue
            if repaired.failure is not None:
                queue.append(repaired)
                continue
            self._model_candidates([repaired], round_number)
            repaired_records.append(repaired)
            if repaired.failure is not None:
                queue.append(repaired)
        return repaired_records

    def _measure_candidates(
        self, records: Sequence[CandidateRecord], round_number: int
    ) -> List[CandidateRecord]:
        measured: List[CandidateRecord] = []
        for index, record in enumerate(records, start=1):
            if record.measurement is None:
                self.progress.emit(
                    "measure_progress",
                    "Checking correctness and timing on the GPU.",
                    round=round_number,
                    candidate=index,
                    total=len(records),
                    candidate_id=record.candidate.candidate_id,
                )
                try:
                    record.measurement = self._evaluate(
                        "measure",
                        record.candidate,
                        lambda record=record: self.backend.measure(
                            self.task, record.candidate
                        ),
                    )
                except Exception as error:
                    record.measurement = Measurement(
                        correct=False,
                        error="%s: %s" % (type(error).__name__, error),
                    )
            if record.measurement.correct:
                record.state = "measured"
                record.decision_reason = "Correctness passed and latency was measured."
                record.failure = None
                self.calibrator.observe(self.task, record)
                measured.append(record)
            else:
                record.state = "correctness-failed"
                record.decision_reason = (
                    record.measurement.error or "Correctness failed."
                )
                record.failure = FailureClassifier.measurement(record.measurement)
            parent = self.records.get(record.candidate.parent_id or "")
            self._diagnose_and_remember(record, parent, round_number)
        self._rebuild_trust()
        return measured

    def _repair_measurement_failures(
        self, records: Sequence[CandidateRecord], round_number: int
    ) -> tuple[List[CandidateRecord], List[CandidateRecord]]:
        repair_records: List[CandidateRecord] = []
        measured_repairs: List[CandidateRecord] = []
        queue = [item for item in records if item.failure is not None]
        while queue and self._repair_budget_available(round_number):
            failed = queue.pop(0)
            repaired = self._repair_candidate(failed, round_number)
            if repaired is None:
                continue
            if repaired.failure is not None:
                queue.append(repaired)
                continue
            self._model_candidates([repaired], round_number)
            repair_records.append(repaired)
            if repaired.failure is not None:
                queue.append(repaired)
                continue
            measured = self._measure_candidates([repaired], round_number)
            if measured:
                measured_repairs.extend(measured)
            elif repaired.failure is not None:
                queue.append(repaired)
        return repair_records, measured_repairs

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
            elif record.state in {"measured-beam", "measured"}:
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
            self.store.append_event(
                "profile_skipped", {"round": round_number, "reason": "no-milestone"}
            )
            return
        record = self.records[decision.candidate_id]
        if record.profile is None:
            self.progress.emit(
                "profile_started",
                "Collecting an NCU milestone profile.",
                round=round_number,
                candidate_id=decision.candidate_id,
                reason=decision.reason,
            )
            record.profile = self._profile_candidate(record.candidate)
        if record.profile.valid:
            reason = "ncu:" + decision.reason
            if reason not in record.selection_reasons:
                record.selection_reasons.append(reason)
            self.milestones.mark_profiled(record, round_number)
            event = "candidate_profiled"
        else:
            reason = "ncu-failed:" + decision.reason
            if reason not in record.selection_reasons:
                record.selection_reasons.append(reason)
            event = "candidate_profile_failed"
        parent = self.records.get(record.candidate.parent_id or "")
        self._diagnose_and_remember(record, parent, round_number)
        self.store.save_candidate(record)
        self.store.append_event(
            event,
            {
                "round": round_number,
                "candidate_id": decision.candidate_id,
                "reason": decision.reason,
                "error": record.profile.error,
            },
        )

    def _profile_candidate(self, candidate: Candidate) -> ProfileEvaluation:
        try:
            profile = self._evaluate(
                "profile",
                candidate,
                lambda: self.backend.profile(self.task, candidate),
            )
        except Exception as error:
            profile = ProfileEvaluation(
                bottleneck="profile-unavailable",
                valid=False,
                error="%s: %s" % (type(error).__name__, error),
                metrics={"exception_type": type(error).__name__},
            )
        self.profile_calls += 1
        return profile

    def _evaluate(
        self, stage: str, candidate: Candidate, operation: Callable[[], T]
    ) -> T:
        self.store.append_event(
            "evaluator_stage_started",
            {"stage": stage, "candidate_id": candidate.candidate_id},
        )
        started_at = time.perf_counter()
        try:
            result = operation()
        except Exception as error:
            elapsed = time.perf_counter() - started_at
            self._record_stage_timing(stage, elapsed, failed=True)
            self.store.append_event(
                "evaluator_stage_failed",
                {
                    "stage": stage,
                    "candidate_id": candidate.candidate_id,
                    "elapsed_seconds": elapsed,
                    "error": "%s: %s" % (type(error).__name__, error),
                    "backend": dict(
                        getattr(self.backend, "last_stage_metadata", {}) or {}
                    ),
                },
            )
            raise
        elapsed = time.perf_counter() - started_at
        self._record_stage_timing(stage, elapsed, failed=False)
        self.store.append_event(
            "evaluator_stage_completed",
            {
                "stage": stage,
                "candidate_id": candidate.candidate_id,
                "elapsed_seconds": elapsed,
                "backend": dict(
                    getattr(self.backend, "last_stage_metadata", {}) or {}
                ),
            },
        )
        return result

    def _diagnose_and_remember(
        self,
        record: CandidateRecord,
        parent: Optional[CandidateRecord],
        round_number: int,
    ) -> None:
        record.diagnosis = self.analyzer.diagnose(record, parent)
        self.store.save_candidate(record)
        lesson = self.evidence_memory.ingest(record, parent, round_number)
        self.store.save_evidence_memory(self.evidence_memory.snapshot())
        if lesson is not None:
            self.store.append_event(
                "evidence_lesson_added",
                {
                    "round": round_number,
                    "lesson_id": lesson.lesson_id,
                    "candidate_id": record.candidate.candidate_id,
                    "kind": lesson.kind,
                    "bottleneck": lesson.bottleneck,
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
            "calibration": self.calibrator.to_dict(),
            "shared_memory": self.evidence_memory.for_prompt(record),
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
                "source_sha256": item.candidate.source_sha256,
                "hypothesis": item.candidate.hypothesis,
                "state": item.state,
                "lineage_kind": item.candidate.lineage_kind,
                "repair_depth": item.candidate.repair_depth,
                "predicted_latency_ms": (
                    item.model.predicted_latency_ms if item.model else None
                ),
                "calibrated_predicted_latency_ms": (
                    item.model.calibrated_latency_ms if item.model else None
                ),
                "measured_latency_ms": (
                    item.measurement.latency_ms if item.measurement else None
                ),
                "failure": item.failure.to_dict() if item.failure else None,
                "diagnosis": (
                    {
                        "category": item.diagnosis.category,
                        "confidence": item.diagnosis.confidence,
                        "summary": item.diagnosis.summary,
                        "recommendations": item.diagnosis.recommendations,
                    }
                    if item.diagnosis
                    else None
                ),
                "profile_bottleneck": (
                    item.profile.bottleneck if item.profile else None
                ),
            }
            for item in records[-20:]
        ]

    def _checkpoint(
        self,
        *,
        phase: str,
        completed_round: int,
        active_round: Optional[int],
        round_candidate_ids: Optional[Sequence[str]] = None,
        selected_ids: Optional[Sequence[str]] = None,
        best_before_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = {
            "schema_version": 1,
            "phase": phase,
            "completed_round": completed_round,
            "active_round": active_round,
            "round_candidate_ids": list(round_candidate_ids or []),
            "selected_ids": list(selected_ids or []),
            "best_before_id": best_before_id,
            "beam": [item.candidate.candidate_id for item in self.beam],
            "best_candidate_id": (
                self.beam[0].candidate.candidate_id if self.beam else None
            ),
            "best_source_sha256": (
                self.beam[0].candidate.source_sha256 if self.beam else None
            ),
            "best_latency_ms": (
                self.beam[0].measurement.latency_ms if self.beam else None
            ),
            "trust": self.trust.to_dict(),
            "trust_snapshot": self.trust.snapshot(),
            "calibration": self.calibrator.snapshot(),
            "evidence_memory": self.evidence_memory.snapshot(),
            "milestones": self.milestones.snapshot(),
            "selection_random_state": self.selection.snapshot(),
            "profile_calls": self.profile_calls,
            "preflight_calls": self.preflight_calls,
            "generator_calls": self.generator_calls,
            "generator_usage": dict(self.generator_usage),
            "repair_calls": self.repair_calls,
            "repair_candidates": self.repair_candidates,
            "repairs_by_round": {
                str(name): value for name, value in self._repairs_by_round.items()
            },
            "stage_timings": self.stage_timings,
        }
        self.store.save_state(state)
        return state

    def _persist_runtime_counters(self) -> None:
        state = self.store.load_state()
        if state is None:
            return
        state.update(
            {
                "profile_calls": self.profile_calls,
                "preflight_calls": self.preflight_calls,
                "generator_calls": self.generator_calls,
                "generator_usage": dict(self.generator_usage),
                "repair_calls": self.repair_calls,
                "repair_candidates": self.repair_candidates,
                "repairs_by_round": {
                    str(name): value
                    for name, value in self._repairs_by_round.items()
                },
                "calibration": self.calibrator.snapshot(),
                "evidence_memory": self.evidence_memory.snapshot(),
                "stage_timings": self.stage_timings,
            }
        )
        self.store.save_state(state)

    def _record_stage_timing(
        self, stage: str, elapsed_seconds: float, *, failed: bool
    ) -> None:
        metrics = self.stage_timings.setdefault(
            stage, {"calls": 0.0, "failures": 0.0, "seconds": 0.0}
        )
        metrics["calls"] += 1.0
        metrics["seconds"] += float(elapsed_seconds)
        if failed:
            metrics["failures"] += 1.0

    def _rebuild_trust(self) -> None:
        tracker = TrustTracker()
        for record in sorted(
            self.records.values(),
            key=lambda item: (
                item.candidate.generation,
                item.candidate.candidate_id,
            ),
        ):
            if record.candidate.generation == 0:
                continue
            parent = self.records.get(record.candidate.parent_id or "")
            tracker.update(record, parent)
        self.trust = tracker

    def _records_for_ids(self, identifiers: Sequence[str]) -> List[CandidateRecord]:
        missing = [identifier for identifier in identifiers if identifier not in self.records]
        if missing:
            raise ValueError("checkpoint references missing candidates: %s" % missing)
        return [self.records[identifier] for identifier in identifiers]

    @staticmethod
    def _phase_at_least(current: str, expected: str) -> bool:
        return _PHASE_ORDER.get(current, -1) >= _PHASE_ORDER[expected]

    def _accumulate_generator_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        for name, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            self.generator_usage[name] = self.generator_usage.get(name, 0.0) + float(value)

    def _summary(
        self,
        completed_rounds: int,
        seed: CandidateRecord,
        best_source_path: Path,
        elapsed_seconds: float,
    ) -> SearchSummary:
        best = self.beam[0]
        seed_latency = float(seed.measurement.latency_ms)
        best_latency = float(best.measurement.latency_ms)
        measured = [item for item in self.records.values() if item.measurement]
        correct = [item for item in measured if item.measurement.correct]
        return SearchSummary(
            task_id=self.task.task_id,
            best_candidate_id=best.candidate.candidate_id,
            best_source_path=str(best_source_path),
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
            generator_calls=self.generator_calls,
            generator_usage=dict(self.generator_usage),
            elapsed_seconds=elapsed_seconds,
            trust=self.trust.to_dict(),
            output_directory=str(self.store.root),
            preflight_calls=self.preflight_calls,
            resumed=self.was_resumed,
            stage_timings=self.stage_timings,
            repair_calls=self.repair_calls,
            repair_candidates=self.repair_candidates,
            evidence_lessons=len(self.evidence_memory.lessons),
            calibration=self.calibrator.to_dict(),
        )


def _unique(values: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
