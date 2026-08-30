"""Adaptive-fidelity controller for API-generated kernel source candidates."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TypeVar

from .archive import ArtifactStore
from .calibration import LatencyCalibrator
from .costs import build_cost_ledger
from .diagnosis import (
    BottleneckAnalyzer,
    FailureClassifier,
    is_infrastructure_failure,
)
from .evidence import GlobalEvidenceMemory
from .incumbent_tracking import PeriodicIncumbentRecorder
from .milestones import create_profile_policy
from .progress import NullProgressReporter, ProgressReporter
from .prompt_compression import (
    COMPRESSION_VERSION,
    METADATA_COMPRESSION_POLICIES,
    NO_COMPRESSION_POLICY,
)
from .protocols import CandidateGenerator, PerformanceBackend
from .reporting import expected_report_paths, write_research_artifacts
from .schema import (
    Candidate,
    CandidateProposal,
    CandidateRecord,
    FailureEvidence,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    SearchSummary,
    TaskSpec,
)
from .selection import create_selection_policy
from .source_validation import SourceValidationError, SourceValidator
from .structural_search import (
    SourceNoveltyAnalyzer,
    StrategyAssignment,
    StrategyPlan,
    StructuralStrategyPortfolio,
    bind_strategy_assignments,
)
from .trust import TrustTracker


T = TypeVar("T")

_INFRASTRUCTURE_FAILURE_LIMIT = 2

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
    "finalized": 8,
    "run_completed": 9,
}


def resolve_evaluation_policies(
    *,
    evaluation_policy: str,
    tir_evidence_policy: str,
    selection_policy: str,
    profile_policy: str,
    compiled_deduplication: bool,
) -> Dict[str, Any]:
    """Resolve requested ablation controls into executable controller policies."""

    if evaluation_policy not in {"tilesight", "cuda-event", "ncu"}:
        raise ValueError(
            "evaluation_policy must be tilesight, cuda-event, or ncu"
        )
    if tir_evidence_policy not in {"auto", "visible", "hidden"}:
        raise ValueError(
            "tir_evidence_policy must be auto, visible, or hidden"
        )
    effective_tir_evidence = (
        "visible"
        if tir_evidence_policy == "auto" and evaluation_policy == "tilesight"
        else "hidden"
        if tir_evidence_policy == "auto"
        else tir_evidence_policy
    )
    if evaluation_policy != "tilesight" and effective_tir_evidence == "visible":
        raise ValueError(
            "tir_evidence_policy=visible requires evaluation_policy=tilesight"
        )

    model_enabled = evaluation_policy == "tilesight"
    return {
        "evaluation_policy": evaluation_policy,
        "requested_tir_evidence_policy": tir_evidence_policy,
        "tir_evidence_policy": effective_tir_evidence,
        "requested_selection_policy": selection_policy,
        "selection_policy": selection_policy if model_enabled else "measure-all",
        "requested_profile_policy": profile_policy,
        "profile_policy": (
            profile_policy
            if model_enabled
            else "every-candidate"
            if evaluation_policy == "ncu"
            else "none"
        ),
        "requested_compiled_deduplication": bool(compiled_deduplication),
        "compiled_deduplication": bool(compiled_deduplication) and model_enabled,
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
        preflight_usage: Optional[Mapping[str, float]] = None,
        preflight_elapsed_seconds: float = 0.0,
        selection_policy: str = "adaptive",
        profile_policy: str = "milestone",
        fixed_promotions_per_round: Optional[int] = None,
        compiled_deduplication: bool = True,
        api_input_price_per_million: Optional[float] = None,
        api_output_price_per_million: Optional[float] = None,
        structural_search_policy: str = "off",
        strategy_allocation_policy: str = "fixed",
        incumbent_snapshot_interval_seconds: float = 300.0,
        max_search_seconds: float = 0.0,
        evaluation_policy: str = "tilesight",
        tir_evidence_policy: str = "auto",
        metadata_compression_policy: str = COMPRESSION_VERSION,
        time_budget_clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.task = task
        self.source_code = source_code
        self.source_name = source_name
        self.generator = generator
        self.backend = backend
        self.store = store
        self.source_validator = source_validator or SourceValidator()
        self.run_metadata = dict(run_metadata or {})
        baseline_style = dict(self.run_metadata.get("baseline_style") or {})
        self.baseline_style = str(baseline_style.get("name") or "native")
        self.baseline_style_mode = str(
            baseline_style.get("mode") or "native"
        )
        self.baseline_style_canonical = bool(
            baseline_style.get("canonical", True)
        )
        self.progress = progress or NullProgressReporter()
        self.resume = resume
        self.preflight_calls = preflight_calls
        self.api_request_attempts = preflight_calls
        self.preflight_elapsed_seconds = float(preflight_elapsed_seconds)
        policies = resolve_evaluation_policies(
            evaluation_policy=evaluation_policy,
            tir_evidence_policy=tir_evidence_policy,
            selection_policy=selection_policy,
            profile_policy=profile_policy,
            compiled_deduplication=compiled_deduplication,
        )
        self.evaluation_policy = str(policies["evaluation_policy"])
        self.requested_tir_evidence_policy = str(
            policies["requested_tir_evidence_policy"]
        )
        self.tir_evidence_policy = str(policies["tir_evidence_policy"])
        if metadata_compression_policy not in METADATA_COMPRESSION_POLICIES:
            raise ValueError(
                "metadata_compression_policy must be one of: %s"
                % ", ".join(METADATA_COMPRESSION_POLICIES)
            )
        self.metadata_compression_policy = metadata_compression_policy
        self.requested_selection_policy_name = str(
            policies["requested_selection_policy"]
        )
        self.selection_policy_name = str(policies["selection_policy"])
        self.requested_profile_policy_name = str(
            policies["requested_profile_policy"]
        )
        self.profile_policy_name = str(policies["profile_policy"])
        self.fixed_promotions_per_round = fixed_promotions_per_round
        self.requested_compiled_deduplication = bool(
            policies["requested_compiled_deduplication"]
        )
        self.compiled_deduplication = bool(policies["compiled_deduplication"])
        self.api_input_price_per_million = api_input_price_per_million
        self.api_output_price_per_million = api_output_price_per_million
        if structural_search_policy not in {"enforce", "observe", "off"}:
            raise ValueError(
                "structural_search_policy must be enforce, observe, or off"
            )
        self.structural_search_policy = structural_search_policy
        if strategy_allocation_policy not in {
            "ai-planned",
            "fixed",
            "unconstrained",
        }:
            raise ValueError(
                "strategy_allocation_policy must be ai-planned, fixed, or unconstrained"
            )
        self.strategy_allocation_policy = strategy_allocation_policy
        interval = float(incumbent_snapshot_interval_seconds)
        if not math.isfinite(interval) or interval < 0:
            raise ValueError(
                "incumbent_snapshot_interval_seconds must be finite and non-negative"
            )
        self.incumbent_snapshot_interval_seconds = interval
        self.incumbent_recorder = PeriodicIncumbentRecorder(
            store, interval_seconds=interval
        )
        max_seconds = float(max_search_seconds)
        if not math.isfinite(max_seconds) or max_seconds < 0:
            raise ValueError("max_search_seconds must be finite and non-negative")
        self.max_search_seconds = max_seconds
        self.candidate_graph_upper_bound = (
            1
            + task.budget.rounds
            * (
                task.budget.proposals_per_round
                + task.budget.max_repairs_per_round
            )
        )
        self._time_budget_clock = time_budget_clock
        self._time_budget_started_at: Optional[float] = None
        self._time_budget_context: Optional[Dict[str, Any]] = None
        self.termination_reason: Optional[str] = None
        self.strategy_portfolio = StructuralStrategyPortfolio()
        self.novelty_analyzer = SourceNoveltyAnalyzer()
        for name, value in (
            ("api_input_price_per_million", api_input_price_per_million),
            ("api_output_price_per_million", api_output_price_per_million),
        ):
            if value is not None and value < 0:
                raise ValueError("%s must be non-negative" % name)
        self.selection = create_selection_policy(
            self.selection_policy_name, task.budget, fixed_promotions_per_round
        )
        milestone_policy = (
            self.profile_policy_name
            if self.evaluation_policy == "tilesight"
            else "none"
        )
        self.milestones = create_profile_policy(milestone_policy, task.budget)
        self.trust = TrustTracker()
        self.calibrator = LatencyCalibrator(task.budget.calibration_min_samples)
        self.analyzer = BottleneckAnalyzer()
        self.evidence_memory = GlobalEvidenceMemory()
        self.records: Dict[str, CandidateRecord] = {}
        self.beam: List[CandidateRecord] = []
        self.profile_calls = 0
        self.generator_calls = 0
        self.planner_calls = 0
        self.planner_fallbacks = 0
        self.strategy_plans: List[Dict[str, Any]] = []
        self.generator_usage: Dict[str, float] = {
            str(name): float(value)
            for name, value in dict(preflight_usage or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        self.stage_timings: Dict[str, Dict[str, float]] = {}
        if preflight_calls > 0:
            self.stage_timings["api_preflight"] = {
                "calls": 1.0,
                "failures": 0.0,
                "seconds": self.preflight_elapsed_seconds,
            }
        self.repair_calls = 0
        self.repair_candidates = 0
        self._repairs_by_round: Dict[int, int] = {}
        self.final_validation_calls = 0
        self.final_validation_passes = 0
        self._search_best_id: Optional[str] = None
        self.was_resumed = False
        self.compiled_hash_owners: Dict[str, str] = {}
        self._infrastructure_failure_category: Optional[str] = None
        self._consecutive_infrastructure_failures = 0

    def run(self) -> SearchSummary:
        started_at = time.perf_counter()
        self._time_budget_started_at = self._time_budget_clock()
        self._time_budget_context = None
        self.termination_reason = None
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
            self.incumbent_recorder.start(
                started_at=started_at,
                resumed=self.resume,
            )

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
                self.incumbent_recorder.record_if_changed("seed-completed")

            active_round = state.get("active_round") if state else None
            start_round = (
                int(active_round)
                if active_round is not None
                else completed_rounds + 1
            )
            for round_number in range(start_round, self.task.budget.rounds + 1):
                if self._time_budget_exhausted(round_number, "round-start"):
                    self._stop_for_time_budget(
                        round_number=round_number,
                        completed_round=completed_rounds,
                        resume_phase="",
                    )
                    break
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

            if self.termination_reason is None:
                self.termination_reason = (
                    "round-budget-completed"
                    if completed_rounds >= self.task.budget.rounds
                    else "search-stopped"
                )

            if self._search_best_id is None:
                self._search_best_id = self.beam[0].candidate.candidate_id
            search_best = self.records[self._search_best_id]
            final_seed = seed
            should_finalize = (
                self.task.budget.final_validation_candidates > 0
                and self.termination_reason != "time-budget"
            )
            if should_finalize and self._time_budget_exhausted(
                completed_rounds + 1, "before-final-validation"
            ):
                should_finalize = False
            if should_finalize:
                final_best, final_seed = self._finalize_candidates(
                    seed, completed_rounds
                )
                remaining = sorted(
                    [
                    item
                    for item in self.beam
                    if item.candidate.candidate_id
                    != final_best.candidate.candidate_id
                    and item.final_measurement is not None
                    and item.final_measurement.correct
                    ],
                    key=lambda item: (
                        float(item.final_measurement.latency_ms),
                        item.candidate.candidate_id,
                    ),
                )
                self.beam = ([final_best] + remaining)[: self.task.budget.beam_width]
                self._checkpoint(
                    phase="finalized",
                    completed_round=completed_rounds,
                    active_round=None,
                    termination_reason=(
                        "time-budget"
                        if self.termination_reason == "time-budget"
                        else None
                    ),
                )
                self.incumbent_recorder.record_if_changed(
                    "final-incumbent-updated"
                )
            best_source_path = self.store.save_best(self.beam[0])
            summary = self._summary(
                completed_rounds,
                seed,
                search_best,
                final_seed,
                best_source_path,
                elapsed_seconds=time.perf_counter() - started_at,
            )
            self.store.save_summary(summary)
            write_research_artifacts(
                self.store,
                self.task,
                list(self.records.values()),
                summary,
                self.run_metadata,
                self.strategy_plans,
            )
            if self.termination_reason == "time-budget":
                self._persist_runtime_counters()
                self.incumbent_recorder.record_now("time-budget-stopped")
                self.store.append_event("run_stopped", summary.to_dict())
                self.progress.emit(
                    "time_budget_stopped",
                    "Search time budget reached; exported the current incumbent.",
                    elapsed_seconds=round(self._search_elapsed_seconds(), 3),
                    max_search_seconds=self.max_search_seconds,
                    best_candidate_id=summary.best_candidate_id,
                    best_latency_ms=summary.best_latency_ms,
                )
            else:
                self._checkpoint(
                    phase="run_completed",
                    completed_round=completed_rounds,
                    active_round=None,
                )
                self.incumbent_recorder.record_now("run-completed")
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
            self.incumbent_recorder.record_now("run-failed")
            raise
        finally:
            self.incumbent_recorder.stop()

    def _search_elapsed_seconds(self) -> float:
        if self._time_budget_started_at is None:
            return 0.0
        return max(
            0.0,
            float(self._time_budget_clock() - self._time_budget_started_at),
        )

    def _time_budget_exhausted(self, round_number: int, stage: str) -> bool:
        if self.max_search_seconds <= 0:
            return False
        elapsed = self._search_elapsed_seconds()
        if elapsed < self.max_search_seconds:
            return False
        if self._time_budget_context is None:
            self.termination_reason = "time-budget"
            self._time_budget_context = {
                "round": round_number,
                "stage": stage,
                "elapsed_seconds": elapsed,
                "max_search_seconds": self.max_search_seconds,
                "overshoot_seconds": max(0.0, elapsed - self.max_search_seconds),
            }
            self.store.append_event(
                "time_budget_exhausted", dict(self._time_budget_context)
            )
            self.progress.emit(
                "time_budget_exhausted",
                "Search time budget reached; stopping at a safe checkpoint.",
                **self._time_budget_context,
            )
        return True

    def _stop_for_time_budget(
        self,
        *,
        round_number: int,
        completed_round: int,
        resume_phase: str,
        round_candidate_ids: Sequence[str] = (),
        selected_ids: Sequence[str] = (),
        best_before_id: Optional[str] = None,
    ) -> bool:
        self._time_budget_exhausted(round_number, resume_phase or "round-start")
        self._checkpoint(
            phase=resume_phase,
            completed_round=completed_round,
            active_round=round_number,
            round_candidate_ids=round_candidate_ids,
            selected_ids=selected_ids,
            best_before_id=best_before_id,
            termination_reason="time-budget",
        )
        self.incumbent_recorder.record_now("time-budget-checkpoint")
        return False

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
            if self._time_budget_exhausted(round_number, "generation"):
                return self._stop_for_time_budget(
                    round_number=round_number,
                    completed_round=round_number - 1,
                    resume_phase=phase,
                    round_candidate_ids=candidate_ids,
                )
            if not generated:
                self.termination_reason = "no-new-valid-candidates"
                self._record_round_strategy_outcomes(round_number)
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
            if self.evaluation_policy == "tilesight":
                modeled = self._model_candidates(generated, round_number)
                repaired = self._repair_model_failures(modeled, round_number)
            else:
                modeled = self._skip_model_candidates(generated, round_number)
                repaired = []
            modeled.extend(repaired)
            candidate_ids = _unique(
                candidate_ids
                + [item.candidate.candidate_id for item in repaired]
            )
            if self._time_budget_exhausted(round_number, "model-or-repair"):
                return self._stop_for_time_budget(
                    round_number=round_number,
                    completed_round=round_number - 1,
                    resume_phase=phase,
                    round_candidate_ids=candidate_ids,
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

        if self.evaluation_policy == "tilesight":
            promotable = [
                item
                for item in modeled
                if item.model
                and item.model.valid
                and item.compiled_equivalent_to is None
            ]
        else:
            promotable = [
                item
                for item in modeled
                if item.failure is None and item.measurement is None
            ]
        if not self._phase_at_least(phase, "selected"):
            if self.evaluation_policy == "tilesight":
                selected = self.selection.select(
                    self.task, promotable, self.beam, self.trust
                )
            else:
                selected = list(promotable)
                reason = "evaluation-policy:%s:measure-all" % self.evaluation_policy
                for record in selected:
                    if reason not in record.selection_reasons:
                        record.selection_reasons.append(reason)
                    record.decision_reason = (
                        "TileSight is disabled; every static-valid candidate is "
                        "scheduled for hardware evaluation."
                    )
                    self.store.save_candidate(record)
            selected_ids = [item.candidate.candidate_id for item in selected]
            selected_set = set(selected_ids)
            for record in promotable:
                if record.candidate.candidate_id not in selected_set:
                    record.state = "modeled-not-promoted"
                    record.decision_reason = (
                        "Not selected by the configured promotion policy."
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
                evaluation_policy=self.evaluation_policy,
            )
        if self._time_budget_exhausted(round_number, "before-measurement"):
            return self._stop_for_time_budget(
                round_number=round_number,
                completed_round=round_number - 1,
                resume_phase=phase,
                round_candidate_ids=candidate_ids,
                selected_ids=selected_ids,
                best_before_id=best_before_id,
            )
        selected = self._records_for_ids(selected_ids)

        if not self._phase_at_least(phase, "measured"):
            measured = self._measure_candidates(selected, round_number)
            repair_records, repair_measured = self._repair_measurement_failures(
                selected, round_number
            )
            measured.extend(repair_measured)
            if self.evaluation_policy == "ncu":
                self._profile_all_measured(measured, round_number)
            candidate_ids = _unique(
                candidate_ids
                + [item.candidate.candidate_id for item in repair_records]
            )
            selected_ids = _unique(
                selected_ids
                + [item.candidate.candidate_id for item in repair_records]
            )
            if self._time_budget_exhausted(
                round_number, "measurement-repair-or-profile"
            ):
                self._update_beam(measured)
                self.incumbent_recorder.record_if_changed(
                    "time-budget-incumbent-updated"
                )
                return self._stop_for_time_budget(
                    round_number=round_number,
                    completed_round=round_number - 1,
                    resume_phase=phase,
                    round_candidate_ids=candidate_ids,
                    selected_ids=selected_ids,
                    best_before_id=best_before_id,
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
            self.incumbent_recorder.record_if_changed("incumbent-updated")
            phase = "beam_updated"
        else:
            best_before = self.records[str(best_before_id)]

        if self._time_budget_exhausted(round_number, "before-round-profile"):
            return self._stop_for_time_budget(
                round_number=round_number,
                completed_round=round_number - 1,
                resume_phase=phase,
                round_candidate_ids=candidate_ids,
                selected_ids=selected_ids,
                best_before_id=best_before_id,
            )

        if not self._phase_at_least(phase, "round_completed"):
            if self.evaluation_policy == "tilesight":
                self._maybe_profile_round(
                    round_number,
                    best_before,
                    self.beam[0],
                    measured,
                )
            if self._time_budget_exhausted(round_number, "round-profile"):
                return self._stop_for_time_budget(
                    round_number=round_number,
                    completed_round=round_number - 1,
                    resume_phase=phase,
                    round_candidate_ids=candidate_ids,
                    selected_ids=selected_ids,
                    best_before_id=best_before_id,
                )
            self._record_round_strategy_outcomes(round_number)
            self._checkpoint(
                phase="round_completed",
                completed_round=round_number,
                active_round=None,
                round_candidate_ids=candidate_ids,
                selected_ids=selected_ids,
                best_before_id=best_before_id,
            )
            self.incumbent_recorder.record_now("round-completed")
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
        if self.evaluation_policy == "tilesight":
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
                record.state = (
                    "seed-infrastructure-failed"
                    if is_infrastructure_failure(record.failure)
                    else "seed-model-invalid"
                )
                self.store.save_candidate(record)
                if is_infrastructure_failure(record.failure):
                    raise RuntimeError(
                        "the input kernel could not be modeled because the evaluator "
                        "environment failed: %s" % record.failure.message
                    )
                raise RuntimeError(
                    "the input kernel is invalid according to the model backend "
                    "[%s]: %s"
                    % (record.failure.category, record.failure.message)
                )
            record.model = self.calibrator.apply(self.task, record.model)
            self._register_compiled_identity(record)
        else:
            record.state = "model-skipped"
            record.decision_reason = (
                "TileSight model evaluation is disabled by the evaluation policy."
            )
            self.store.save_candidate(record)
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
            record.state = (
                "seed-infrastructure-failed"
                if is_infrastructure_failure(record.failure)
                else "seed-correctness-failed"
            )
            self.store.save_candidate(record)
            if is_infrastructure_failure(record.failure):
                raise RuntimeError(
                    "the input kernel could not be measured because the evaluator "
                    "environment failed: %s" % record.failure.message
                )
            raise RuntimeError("the input kernel failed correctness")
        self._clear_infrastructure_failures()
        self.calibrator.observe(self.task, record)
        if self.evaluation_policy == "ncu":
            record.profile = self._profile_candidate(candidate)
            self._handle_profile_infrastructure(record, round_number=0)
        elif self.evaluation_policy == "tilesight" and self.milestones.profile_seed:
            record.profile = self._profile_candidate(candidate)
            if record.profile.valid:
                self.milestones.mark_profiled(record, round_number=0)
            self._handle_profile_infrastructure(record, round_number=0)
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
                "profile_valid": record.profile.valid if record.profile else None,
                "evaluation_policy": self.evaluation_policy,
            },
        )
        self.progress.emit(
            "seed_completed",
            "Input kernel passed correctness and timing.",
            candidate_id=candidate.candidate_id,
            latency_ms=record.measurement.latency_ms,
            profile_valid=record.profile.valid if record.profile else None,
        )
        return record

    def _restore(
        self, expected_seed: Candidate, state: Mapping[str, Any]
    ) -> CandidateRecord:
        archived_experiment = state.get("experiment")
        if archived_experiment and self._normalized_experiment_config(
            archived_experiment
        ) != self._normalized_experiment_config(self._experiment_config()):
            raise ValueError(
                "resume experiment policies do not match the archived checkpoint"
            )
        current_preflight_usage = dict(self.generator_usage)
        current_stage_timings = dict(self.stage_timings)
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
        self.api_request_attempts += int(
            state.get("api_request_attempts", state.get("preflight_calls", 0))
        )
        self.generator_calls = int(state.get("generator_calls", 0))
        self.planner_calls = int(state.get("planner_calls", 0))
        self.planner_fallbacks = int(state.get("planner_fallbacks", 0))
        self.strategy_plans = [
            dict(item)
            for item in state.get("strategy_plans", [])
            if isinstance(item, Mapping)
        ]
        self.generator_usage = {
            str(name): float(value)
            for name, value in dict(state.get("generator_usage") or {}).items()
        }
        for name, value in current_preflight_usage.items():
            self.generator_usage[name] = self.generator_usage.get(name, 0.0) + value
        self.stage_timings = {
            str(name): {
                str(metric): float(value)
                for metric, value in dict(metrics).items()
            }
            for name, metrics in dict(state.get("stage_timings") or {}).items()
        }
        for stage, metrics in current_stage_timings.items():
            for name, value in metrics.items():
                stored = self.stage_timings.setdefault(stage, {})
                stored[name] = stored.get(name, 0.0) + float(value)
        self.repair_calls = int(state.get("repair_calls", 0))
        self.repair_candidates = int(state.get("repair_candidates", 0))
        self._repairs_by_round = {
            int(name): int(value)
            for name, value in dict(state.get("repairs_by_round") or {}).items()
        }
        self.final_validation_calls = int(state.get("final_validation_calls", 0))
        self.final_validation_passes = int(state.get("final_validation_passes", 0))
        search_best_id = state.get("search_best_id")
        self._search_best_id = str(search_best_id) if search_best_id else None
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
                if (
                    record.diagnosis is not None
                    and not is_infrastructure_failure(record.failure)
                ):
                    parent = self.records.get(record.candidate.parent_id or "")
                    self.evidence_memory.ingest(
                        record, parent, record.candidate.generation
                    )
        self.trust = TrustTracker.from_snapshot(state.get("trust_snapshot"))
        self._rebuild_trust()
        self.milestones.restore(state.get("milestones"))
        self.selection.restore(
            state.get("selection_state", state.get("selection_random_state"))
        )
        self._rebuild_compiled_hash_owners()
        self.preflight_calls += int(state.get("preflight_calls", 0))
        self.was_resumed = True
        return seed

    def _plan_round_strategies(self, round_number: int) -> StrategyPlan:
        """Choose, normalize, archive, and checkpoint this round's slots."""

        for archived in self.strategy_plans:
            if int(archived.get("round", -1)) == round_number:
                return StrategyPlan.from_dict(archived)

        count = self.task.budget.proposals_per_round
        evidence = self._strategy_planning_evidence(
            self.beam[0], round_number
        )
        if self.strategy_allocation_policy == "unconstrained":
            plan = StrategyPlan(
                round_number=round_number,
                policy="unconstrained",
                source="unconstrained",
                requested_count=count,
                assignments=[],
                round_rationale=(
                    "Candidate intent is left entirely to the source generator."
                ),
            )
            return self._archive_strategy_plan(plan, evidence)
        if self.strategy_allocation_policy == "fixed":
            plan = self.strategy_portfolio.fixed_plan(
                self.task,
                round_number,
                count,
                evidence=evidence,
            )
            return self._archive_strategy_plan(plan, evidence)

        planner = getattr(self.generator, "plan_strategies", None)
        if not callable(planner):
            self.planner_fallbacks += 1
            plan = self.strategy_portfolio.fallback_plan(
                self.task,
                round_number,
                count,
                evidence,
                "candidate generator does not implement strategy planning",
            )
            return self._archive_strategy_plan(plan, evidence)

        self.planner_calls += 1
        self.generator_calls += 1
        started_at = time.perf_counter()
        self.progress.emit(
            "strategy_planning_started",
            "Requesting an evidence-guided strategy allocation.",
            round=round_number,
            requested_slots=count,
        )
        try:
            raw_plan = planner(
                task=self.task,
                parent=self.beam[0].candidate,
                planning_context=evidence,
                strategies=self.strategy_portfolio.strategy_catalog(self.task),
                count=count,
                round_number=round_number,
            )
        except Exception as error:
            elapsed = time.perf_counter() - started_at
            self._record_stage_timing("api_plan", elapsed, failed=True)
            metadata = dict(
                getattr(self.generator, "last_call_metadata", {}) or {}
            )
            self._record_api_attempts(metadata)
            self._accumulate_generator_usage(metadata.get("usage"))
            self.store.save_api_call(
                "round-%03d-plan-failed" % round_number,
                {
                    "round": round_number,
                    "requested_slots": count,
                    "provider": metadata,
                    "error": "%s: %s" % (type(error).__name__, error),
                },
                getattr(self.generator, "last_exchange", None),
            )
            self.store.append_event(
                "strategy_planner_failed",
                {
                    "round": round_number,
                    "error": "%s: %s" % (type(error).__name__, error),
                    "provider": metadata,
                    "action": "deterministic-fallback",
                },
            )
            self.planner_fallbacks += 1
            plan = self.strategy_portfolio.fallback_plan(
                self.task,
                round_number,
                count,
                evidence,
                "%s: %s" % (type(error).__name__, error),
            )
            return self._archive_strategy_plan(plan, evidence)

        elapsed = time.perf_counter() - started_at
        self._record_stage_timing("api_plan", elapsed, failed=False)
        metadata = dict(getattr(self.generator, "last_call_metadata", {}) or {})
        self._record_api_attempts(metadata)
        self._accumulate_generator_usage(metadata.get("usage"))
        self.store.save_api_call(
            "round-%03d-plan" % round_number,
            {
                "round": round_number,
                "requested_slots": count,
                "provider": metadata,
            },
            getattr(self.generator, "last_exchange", None),
        )
        plan = self.strategy_portfolio.normalize_ai_plan(
            self.task,
            round_number,
            count,
            raw_plan,
            evidence=evidence,
        )
        if plan.source == "deterministic-fallback":
            self.planner_fallbacks += 1
        self.progress.emit(
            "strategy_planning_completed",
            "Strategy allocation is ready for candidate generation.",
            round=round_number,
            source=plan.source,
            allocation=plan.to_dict()["allocation"],
            overrides=len(plan.overrides),
            elapsed_seconds=round(elapsed, 3),
        )
        return self._archive_strategy_plan(plan, evidence)

    def _archive_strategy_plan(
        self,
        plan: StrategyPlan,
        planning_evidence: Mapping[str, Any],
    ) -> StrategyPlan:
        payload = plan.to_dict()
        payload["planning_evidence"] = dict(planning_evidence)
        self.strategy_plans = [
            item
            for item in self.strategy_plans
            if int(item.get("round", -1)) != plan.round_number
        ]
        self.strategy_plans.append(payload)
        self.strategy_plans.sort(key=lambda item: int(item.get("round", 0)))
        self.store.save_json_artifact(
            "strategy_plans.json", {"rounds": self.strategy_plans}
        )
        self.store.append_event(
            "strategy_plan_normalized",
            {
                "round": plan.round_number,
                "policy": plan.policy,
                "source": plan.source,
                "allocation": payload["allocation"],
                "overrides": plan.overrides,
                "fallback_reason": plan.fallback_reason,
            },
        )
        self._persist_runtime_counters()
        return plan

    def _generate_round(self, round_number: int) -> List[CandidateRecord]:
        existing = [
            item
            for item in self.records.values()
            if item.candidate.generation == round_number
            and item.candidate.lineage_kind == "proposal"
        ]
        plan = self._plan_round_strategies(round_number)
        assignments = plan.assignments
        used_slots = {
            str(item.candidate.proposal_metadata.get("strategy_slot"))
            for item in existing
            if item.candidate.proposal_metadata.get("strategy_slot")
        }
        remaining_assignments = [
            item for item in assignments if item.slot not in used_slots
        ]
        proposals_left = (
            len(remaining_assignments)
            if assignments
            else max(0, self.task.budget.proposals_per_round - len(existing))
        )
        if assignments:
            self.store.append_event(
                "strategy_portfolio_planned",
                {
                    "round": round_number,
                    "allocation_policy": self.strategy_allocation_policy,
                    "novelty_policy": self.structural_search_policy,
                    "plan_source": plan.source,
                    "planning_candidate_id": self.beam[0].candidate.candidate_id,
                    "assignments": [item.to_dict() for item in assignments],
                    "remaining_slots": [item.slot for item in remaining_assignments],
                },
            )
        parents = list(self.beam)
        static_failures = [
            item
            for item in self.records.values()
            if item.candidate.generation == round_number
            and item.state == "static-invalid"
        ]
        for index, parent_record in enumerate(parents):
            if self._time_budget_exhausted(round_number, "candidate-generation"):
                break
            parents_left = len(parents) - index
            request_count = int(math.ceil(proposals_left / parents_left))
            if request_count <= 0:
                break
            requested_assignments = remaining_assignments[:request_count]
            proposals = self._generate_from_parent(
                round_number,
                parent_record,
                request_count,
                requested_assignments,
            )
            if assignments:
                returned_slots = {
                    str(proposal.metadata.get("strategy_slot"))
                    for proposal in proposals
                    if proposal.metadata.get("strategy_slot")
                }
                remaining_assignments = [
                    item
                    for item in remaining_assignments
                    if item.slot not in returned_slots
                ]
                proposals_left = len(remaining_assignments)
            else:
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
                else:
                    self._validate_candidate_novelty(
                        record,
                        parent_record,
                        round_number,
                    )
                self.store.save_candidate(record)

        repair_queue = list(static_failures)
        while (
            repair_queue
            and self._repair_budget_available(round_number)
            and not self._time_budget_exhausted(round_number, "static-repair")
        ):
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
                "strategy_failures": sum(
                    item.state == "strategy-invalid"
                    for item in self.records.values()
                    if item.candidate.generation == round_number
                ),
                "repair_candidates": sum(
                    item.candidate.lineage_kind == "repair"
                    for item in self.records.values()
                    if item.candidate.generation == round_number
                ),
                "requested_proposals": self.task.budget.proposals_per_round,
                "structural_search_policy": self.structural_search_policy,
                "strategy_allocation_policy": self.strategy_allocation_policy,
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
        strategy_assignments: Sequence[StrategyAssignment] = (),
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
        evidence = self._evidence(parent_record)
        if strategy_assignments:
            evidence = dict(evidence)
            evidence["generation_request"] = {
                "round": round_number,
                "strategy_assignments": [
                    item.to_dict() for item in strategy_assignments
                ],
                "discovered_strategy_memory": self._discovered_strategy_memory(
                    round_number
                ),
            }
        try:
            proposals = self.generator.generate(
                task=self.task,
                parent=parent_record.candidate,
                evidence=evidence,
                history=self._history(),
                count=request_count,
            )
        except Exception as error:
            elapsed = time.perf_counter() - started_at
            self._record_stage_timing("api_generate", elapsed, failed=True)
            metadata = dict(
                getattr(self.generator, "last_call_metadata", {}) or {}
            )
            self._record_api_attempts(metadata)
            self._accumulate_generator_usage(metadata.get("usage"))
            self.store.save_api_call(
                "round-%03d-generate-failed" % round_number,
                {
                    "round": round_number,
                    "parent_id": parent_record.candidate.candidate_id,
                    "requested_candidates": request_count,
                    "strategy_assignments": [
                        item.to_dict() for item in strategy_assignments
                    ],
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
        self._record_api_attempts(metadata)
        self._accumulate_generator_usage(metadata.get("usage"))
        parsing = dict(metadata.get("candidate_parsing") or {})
        rejected_candidates = int(parsing.get("rejected_candidates", 0) or 0)
        if rejected_candidates:
            self.progress.emit(
                "api_candidates_rejected",
                "Rejected malformed API candidates and kept usable complete sources.",
                round=round_number,
                accepted_candidates=int(
                    parsing.get("accepted_candidates", 0) or 0
                ),
                rejected_candidates=rejected_candidates,
                response_validation_attempts=int(
                    metadata.get("response_validation_attempts", 1) or 1
                ),
            )
        failed_workers = int(metadata.get("failed_workers", 0) or 0)
        if failed_workers:
            self.progress.emit(
                "api_partial_failure",
                "Some generation agents failed; continuing with completed results.",
                round=round_number,
                failed_workers=failed_workers,
                successful_workers=int(metadata.get("successful_workers", 0) or 0),
            )
        proposals = bind_strategy_assignments(proposals, strategy_assignments)
        self.store.save_api_call(
            "round-%03d-generate" % round_number,
            {
                "round": round_number,
                "parent_id": parent_record.candidate.candidate_id,
                "requested_candidates": request_count,
                "returned_candidates": len(proposals),
                "strategy_assignments": [
                    item.to_dict() for item in strategy_assignments
                ],
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
                "strategy_slots": [
                    proposal.metadata.get("strategy_slot")
                    for proposal in proposals
                ],
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
        self._persist_runtime_counters()
        return proposals

    def _validate_candidate_novelty(
        self,
        record: CandidateRecord,
        baseline: CandidateRecord,
        round_number: int,
    ) -> bool:
        """Record AST provenance and enforce the assigned portfolio lane."""

        if self.structural_search_policy == "off":
            return True
        report = self.novelty_analyzer.analyze(
            baseline.candidate.source_code,
            record.candidate.source_code,
            self.task.entrypoint,
        )
        metadata = dict(record.candidate.proposal_metadata)
        structural_required = bool(metadata.get("structural_required"))
        open_strategy = (
            metadata.get("strategy_id") == "open-structural-exploration"
        )
        discovered_strategy = str(
            metadata.get("discovered_strategy") or ""
        ).strip()
        missing_open_strategy_name = open_strategy and not discovered_strategy
        no_meaningful_change = not report.meaningful_change
        structural_mismatch = structural_required and not report.structural_change
        mismatch = (
            no_meaningful_change
            or structural_mismatch
            or missing_open_strategy_name
        )
        status = (
            "rejected"
            if mismatch and self.structural_search_policy == "enforce"
            else "observed-mismatch"
            if mismatch
            else "accepted"
        )
        metadata.update(
            {
                "novelty_parent_id": baseline.candidate.candidate_id,
                "ast_novelty": report.to_dict(),
                "strategy_validation": status,
            }
        )
        record.candidate = replace(
            record.candidate,
            proposal_metadata=metadata,
        )
        event = {
            "round": round_number,
            "candidate_id": record.candidate.candidate_id,
            "parent_id": baseline.candidate.candidate_id,
            "strategy_slot": metadata.get("strategy_slot"),
            "strategy_id": metadata.get("strategy_id"),
            "structural_required": structural_required,
            "discovered_strategy": discovered_strategy or None,
            "policy": self.structural_search_policy,
            "status": status,
            "novelty": report.to_dict(),
        }
        self.store.append_event("proposal_novelty_checked", event)
        if status != "rejected":
            if mismatch:
                record.decision_reason = (
                    "AST novelty mismatch retained by observe policy."
                )
            return True

        if missing_open_strategy_name:
            category = "open-strategy-metadata-missing"
            message = (
                "Open structural exploration must name the discovered strategy "
                "in metadata.discovered_strategy."
            )
        elif structural_mismatch:
            category = "structural-strategy-mismatch"
            message = (
                "Strategy %s requires a structural AST change, but the candidate "
                "was classified as %s."
                % (metadata.get("strategy_id"), report.classification)
            )
        else:
            category = "no-meaningful-source-change"
            message = (
                "Candidate changes only formatting, docstrings, or local names."
            )
        record.state = "strategy-invalid"
        record.failure = FailureEvidence(
            stage="strategy-validation",
            category=category,
            message=message,
            diagnostics=[
                "Changing only existing tile/thread/stage values is accepted only "
                "in the parameter-tuning strategy lane."
            ],
            details=event,
            retryable=False,
        )
        record.decision_reason = message
        self.store.append_event(
            "proposal_strategy_rejected",
            {**event, "failure": record.failure.to_dict()},
        )
        return False

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
            or self._time_budget_exhausted(round_number, "source-repair")
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
            "diagnosis": self._diagnosis_for_prompt(failed),
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
            self._record_api_attempts(metadata)
            self._accumulate_generator_usage(metadata.get("usage"))
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
        self._record_api_attempts(metadata)
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
        self._persist_runtime_counters()

        inherited_metadata = dict(proposal.metadata)
        for name in (
            "strategy_slot",
            "strategy_id",
            "requested_strategy_title",
            "structural_required",
            "strategy_binding",
            "reported_strategy_slot",
            "reported_strategy_id",
            "novelty_parent_id",
            "strategy_selection_reason",
            "strategy_evidence_terms",
            "discovered_strategy",
            "related_existing_strategies",
        ):
            if name in failed.candidate.proposal_metadata:
                inherited_metadata[name] = failed.candidate.proposal_metadata[name]
        proposal = CandidateProposal(
            hypothesis=proposal.hypothesis,
            source_code=proposal.source_code,
            expected_effect=proposal.expected_effect,
            metadata=inherited_metadata,
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
            baseline_id = (
                record.candidate.proposal_metadata.get("novelty_parent_id")
                or failed.candidate.parent_id
            )
            baseline = self.records.get(str(baseline_id)) if baseline_id else None
            if baseline is not None and not self._validate_candidate_novelty(
                record, baseline, round_number
            ):
                event = "repair_strategy_failed"
            else:
                record.state = "generated"
                record.decision_reason = "Generated by bounded repair."
                event = "repair_generated"
        self.store.save_candidate(record)
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
            if self._time_budget_exhausted(round_number, "model-candidate"):
                break
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
                self._clear_infrastructure_failures()
                record.model = self.calibrator.apply(self.task, record.model)
                record.failure = None
                parent = self.records.get(record.candidate.parent_id or "")
                record.diagnosis = self.analyzer.diagnose(record, parent)
                equivalent_to = self._register_compiled_identity(record)
                if equivalent_to is not None:
                    record.state = "compiled-equivalent"
                    record.decision_reason = (
                        "Skipped hardware evaluation because the compiled execution "
                        "identity matches candidate %s." % equivalent_to
                    )
                    self.store.append_event(
                        "compiled_candidate_deduplicated",
                        {
                            "round": round_number,
                            "candidate_id": record.candidate.candidate_id,
                            "equivalent_to": equivalent_to,
                            "compiled_identity_sha256": self._compiled_identity_hash(record),
                        },
                    )
                else:
                    record.state = "modeled"
            else:
                record.failure = FailureClassifier.model(record.model)
                infrastructure = is_infrastructure_failure(record.failure)
                record.state = (
                    "infrastructure-failed" if infrastructure else "model-invalid"
                )
                record.decision_reason = (
                    "Evaluator infrastructure failed while modeling the candidate."
                    if infrastructure
                    else "Rejected by compiler or performance model."
                )
                parent = self.records.get(record.candidate.parent_id or "")
                self._diagnose_and_remember(record, parent, round_number)
                if infrastructure:
                    self._record_infrastructure_failure(record, round_number)
            self.store.save_candidate(record)
        return list(records)

    def _repair_model_failures(
        self, records: Sequence[CandidateRecord], round_number: int
    ) -> List[CandidateRecord]:
        repaired_records: List[CandidateRecord] = []
        queue = [item for item in records if item.failure is not None]
        while (
            queue
            and self._repair_budget_available(round_number)
            and not self._time_budget_exhausted(round_number, "model-repair")
        ):
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

    def _skip_model_candidates(
        self, records: Sequence[CandidateRecord], round_number: int
    ) -> List[CandidateRecord]:
        """Mark candidates for direct hardware evaluation without TileSight."""

        for record in records:
            if record.failure is not None or record.measurement is not None:
                continue
            record.state = "model-skipped"
            record.decision_reason = (
                "TileSight model evaluation is disabled by evaluation policy %s."
                % self.evaluation_policy
            )
            self.store.save_candidate(record)
        self.store.append_event(
            "round_model_skipped",
            {
                "round": round_number,
                "evaluation_policy": self.evaluation_policy,
                "candidates": len(records),
            },
        )
        return list(records)

    def _measure_candidates(
        self, records: Sequence[CandidateRecord], round_number: int
    ) -> List[CandidateRecord]:
        measured: List[CandidateRecord] = []
        for index, record in enumerate(records, start=1):
            if self._time_budget_exhausted(round_number, "measure-candidate"):
                break
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
                self._clear_infrastructure_failures()
                record.state = "measured"
                record.decision_reason = "Correctness passed and latency was measured."
                record.failure = None
                self.calibrator.observe(self.task, record)
                measured.append(record)
            else:
                record.failure = FailureClassifier.measurement(record.measurement)
                infrastructure = is_infrastructure_failure(record.failure)
                record.state = (
                    "infrastructure-failed"
                    if infrastructure
                    else "correctness-failed"
                )
                record.decision_reason = (
                    "Evaluator infrastructure failed during measurement: %s"
                    % record.failure.message
                    if infrastructure
                    else record.measurement.error or "Correctness failed."
                )
            parent = self.records.get(record.candidate.parent_id or "")
            self._diagnose_and_remember(record, parent, round_number)
            if is_infrastructure_failure(record.failure):
                self._record_infrastructure_failure(record, round_number)
        self._rebuild_trust()
        return measured

    def _profile_all_measured(
        self, records: Sequence[CandidateRecord], round_number: int
    ) -> None:
        """Collect NCU feedback for every correctness-passing candidate."""

        unique = {
            record.candidate.candidate_id: record
            for record in records
            if record.is_measured_correct
        }
        ordered = [unique[name] for name in sorted(unique)]
        for index, record in enumerate(ordered, start=1):
            if self._time_budget_exhausted(round_number, "profile-candidate"):
                break
            if record.profile is None:
                self.progress.emit(
                    "profile_progress",
                    "Collecting an NCU profile for every correct candidate.",
                    round=round_number,
                    candidate=index,
                    total=len(ordered),
                    candidate_id=record.candidate.candidate_id,
                )
                record.profile = self._profile_candidate(record.candidate)
            reason = "ncu:evaluation-policy:every-candidate"
            if reason not in record.selection_reasons:
                record.selection_reasons.append(reason)
            parent = self.records.get(record.candidate.parent_id or "")
            self._diagnose_and_remember(record, parent, round_number)
            self.store.save_candidate(record)
            self.store.append_event(
                (
                    "candidate_profiled"
                    if record.profile.valid
                    else "candidate_profile_failed"
                ),
                {
                    "round": round_number,
                    "candidate_id": record.candidate.candidate_id,
                    "reason": "evaluation-policy:every-candidate",
                    "error": record.profile.error,
                },
            )
            self._handle_profile_infrastructure(record, round_number)

    def _repair_measurement_failures(
        self, records: Sequence[CandidateRecord], round_number: int
    ) -> tuple[List[CandidateRecord], List[CandidateRecord]]:
        repair_records: List[CandidateRecord] = []
        measured_repairs: List[CandidateRecord] = []
        queue = [item for item in records if item.failure is not None]
        while (
            queue
            and self._repair_budget_available(round_number)
            and not self._time_budget_exhausted(
                round_number, "measurement-repair"
            )
        ):
            failed = queue.pop(0)
            repaired = self._repair_candidate(failed, round_number)
            if repaired is None:
                continue
            if repaired.failure is not None:
                queue.append(repaired)
                continue
            if self.evaluation_policy == "tilesight":
                self._model_candidates([repaired], round_number)
            else:
                self._skip_model_candidates([repaired], round_number)
            repair_records.append(repaired)
            if repaired.failure is not None:
                queue.append(repaired)
                continue
            if repaired.compiled_equivalent_to is not None:
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
        if self._time_budget_exhausted(round_number, "milestone-profile"):
            return
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
        self._handle_profile_infrastructure(record, round_number)

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

    def _handle_profile_infrastructure(
        self,
        record: CandidateRecord,
        round_number: int,
    ) -> None:
        profile = record.profile
        if profile is None:
            return
        if profile.valid:
            self._clear_infrastructure_failures()
            return
        failure = FailureClassifier.profile(profile.error or "")
        if not is_infrastructure_failure(failure):
            return
        self.store.append_event(
            "infrastructure_failure_archived",
            {
                "round": round_number,
                "candidate_id": record.candidate.candidate_id,
                "category": failure.category,
                "stage": failure.stage,
                "message": failure.message,
            },
        )
        self._record_infrastructure_evidence(
            failure,
            record.candidate.candidate_id,
            round_number,
        )

    def _finalize_candidates(
        self, seed: CandidateRecord, completed_rounds: int
    ) -> tuple[CandidateRecord, CandidateRecord]:
        limit = self.task.budget.final_validation_candidates
        search_candidates = list(self.beam[:limit])
        candidates = [seed] + search_candidates
        unique_candidates: List[CandidateRecord] = []
        seen = set()
        for record in candidates:
            identifier = record.candidate.candidate_id
            if identifier not in seen:
                seen.add(identifier)
                unique_candidates.append(record)

        self.progress.emit(
            "final_started",
            "Running fresh held-out correctness and robust timing.",
            candidates=len(unique_candidates),
        )
        for index, record in enumerate(unique_candidates, start=1):
            if index > 1 and self._time_budget_exhausted(
                completed_rounds + 1, "final-validation"
            ):
                break
            if record.final_measurement is None:
                self.progress.emit(
                    "final_progress",
                    "Validating a finalist in a fresh evaluator process.",
                    candidate=index,
                    total=len(unique_candidates),
                    candidate_id=record.candidate.candidate_id,
                )
                finalizer = getattr(self.backend, "finalize", None)
                try:
                    if callable(finalizer):
                        record.final_measurement = self._evaluate(
                            "final",
                            record.candidate,
                            lambda record=record: finalizer(
                                self.task, record.candidate
                            ),
                        )
                    else:
                        record.final_measurement = self._evaluate(
                            "final",
                            record.candidate,
                            lambda record=record: self.backend.measure(
                                self.task, record.candidate
                            ),
                        )
                except Exception as error:
                    record.final_measurement = Measurement(
                        correct=False,
                        error="%s: %s" % (type(error).__name__, error),
                        metrics={"stage": "final", "infrastructure_failure": True},
                    )
                self.final_validation_calls += 1
                if record.final_measurement.correct:
                    self._clear_infrastructure_failures()
                    self.final_validation_passes += 1
                    record.selection_reasons.append("fresh-final-validation")
                    record.state = "final-validated"
                else:
                    final_failure = FailureClassifier.measurement(
                        record.final_measurement
                    )
                    infrastructure = is_infrastructure_failure(final_failure)
                    record.state = (
                        "final-infrastructure-failed"
                        if infrastructure
                        else "final-validation-failed"
                    )
                    record.decision_reason = (
                        record.final_measurement.error
                        or "Held-out final validation failed."
                    )
                    if infrastructure:
                        record.failure = final_failure
                self.store.save_candidate(record)
                self.store.append_event(
                    "candidate_finalized",
                    {
                        "completed_rounds": completed_rounds,
                        "candidate_id": record.candidate.candidate_id,
                        "correct": record.final_measurement.correct,
                        "latency_ms": record.final_measurement.latency_ms,
                        "error": record.final_measurement.error,
                    },
                )
                if is_infrastructure_failure(record.failure):
                    if record.candidate.candidate_id == seed.candidate.candidate_id:
                        raise RuntimeError(
                            "the seed kernel could not complete final validation "
                            "because the evaluator environment failed: %s"
                            % record.failure.message
                        )
                    self._record_infrastructure_failure(record, completed_rounds)

        finalized_records = [
            item for item in self.records.values() if item.final_measurement is not None
        ]
        self.final_validation_calls = len(finalized_records)
        self.final_validation_passes = sum(
            item.final_measurement.correct for item in finalized_records
        )

        if seed.final_measurement is None or not seed.final_measurement.correct:
            raise RuntimeError(
                "the seed kernel failed fresh final validation; the task or evaluator is not stable"
            )
        eligible = [
            item
            for item in search_candidates
            if item.final_measurement is not None
            and item.final_measurement.correct
            and item.final_measurement.latency_ms is not None
        ]
        if all(
            item.candidate.candidate_id != seed.candidate.candidate_id
            for item in eligible
        ):
            eligible.append(seed)
        best = min(
            eligible,
            key=lambda item: (
                float(item.final_measurement.latency_ms),
                item.candidate.candidate_id,
            ),
        )
        best.state = "final-validated-best"
        best.decision_reason = (
            "Passed fresh held-out correctness and had the best robust final latency."
        )
        self.store.save_candidate(best)
        self.progress.emit(
            "final_completed",
            "Fresh final validation selected the exported kernel.",
            candidate_id=best.candidate.candidate_id,
            latency_ms=best.final_measurement.latency_ms,
            passes=self.final_validation_passes,
        )
        return best, seed

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
        if is_infrastructure_failure(record.failure):
            self.store.append_event(
                "infrastructure_failure_archived",
                {
                    "round": round_number,
                    "candidate_id": record.candidate.candidate_id,
                    "category": record.failure.category,
                    "stage": record.failure.stage,
                    "message": record.failure.message,
                },
            )
            return
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

    def _record_infrastructure_failure(
        self,
        record: CandidateRecord,
        round_number: int,
    ) -> None:
        failure = record.failure
        if not is_infrastructure_failure(failure):
            return
        self._record_infrastructure_evidence(
            failure,
            record.candidate.candidate_id,
            round_number,
        )

    def _record_infrastructure_evidence(
        self,
        failure: FailureEvidence,
        candidate_id: str,
        round_number: int,
    ) -> None:
        if failure.category == self._infrastructure_failure_category:
            self._consecutive_infrastructure_failures += 1
        else:
            self._infrastructure_failure_category = failure.category
            self._consecutive_infrastructure_failures = 1
        self.store.append_event(
            "infrastructure_failure_observed",
            {
                "round": round_number,
                "candidate_id": candidate_id,
                "category": failure.category,
                "consecutive_count": self._consecutive_infrastructure_failures,
                "limit": _INFRASTRUCTURE_FAILURE_LIMIT,
            },
        )
        if self._consecutive_infrastructure_failures < _INFRASTRUCTURE_FAILURE_LIMIT:
            return
        self.store.append_event(
            "infrastructure_circuit_breaker_opened",
            {
                "round": round_number,
                "category": failure.category,
                "consecutive_count": self._consecutive_infrastructure_failures,
                "message": failure.message,
            },
        )
        raise RuntimeError(
            "stopping after %d consecutive %s failures; fix the evaluator "
            "environment and start a new output directory: %s"
            % (
                self._consecutive_infrastructure_failures,
                failure.category,
                failure.message,
            )
        )

    def _clear_infrastructure_failures(self) -> None:
        self._infrastructure_failure_category = None
        self._consecutive_infrastructure_failures = 0

    def _evidence(self, record: CandidateRecord) -> Dict[str, Any]:
        return {
            "observed": {
                "measurement": record.measurement.to_dict()
                if record.measurement
                else None,
                "profile": record.profile.to_dict() if record.profile else None,
                "diagnosis": self._diagnosis_for_prompt(record),
            },
            "predicted": (
                record.model.to_dict()
                if self.tir_evidence_policy == "visible" and record.model
                else None
            ),
            "model_trust": (
                self.trust.to_dict()
                if self.tir_evidence_policy == "visible"
                else None
            ),
            "calibration": (
                self.calibrator.to_dict()
                if self.tir_evidence_policy == "visible"
                else None
            ),
            "shared_memory": self._shared_evidence_for_prompt(record),
        }

    def _diagnosis_for_prompt(
        self, record: CandidateRecord
    ) -> Optional[Dict[str, Any]]:
        diagnosis = record.diagnosis
        if diagnosis is None:
            return None
        if (
            self.tir_evidence_policy == "hidden"
            and "tilesight" in diagnosis.source.lower()
        ):
            return None
        return diagnosis.to_dict()

    def _shared_evidence_for_prompt(
        self, record: CandidateRecord
    ) -> List[Dict[str, Any]]:
        lesson_limit = (
            None
            if self.metadata_compression_policy == NO_COMPRESSION_POLICY
            else 64
        )
        lessons = self.evidence_memory.for_prompt(record, limit=lesson_limit)
        if self.tir_evidence_policy == "visible":
            return lessons

        sanitized = []
        for original in lessons:
            lesson = dict(original)
            supporting = dict(lesson.get("supporting_evidence") or {})
            diagnosis = supporting.get("diagnosis")
            if (
                isinstance(diagnosis, Mapping)
                and "tilesight" in str(diagnosis.get("source", "")).lower()
            ):
                supporting.pop("diagnosis", None)
            if lesson.get("kind") == "measured-outcome":
                supporting.pop("raw_predicted_latency_ms", None)
                supporting.pop("calibrated_predicted_latency_ms", None)
                supporting.pop("diagnosis", None)
                lesson["bottleneck"] = "unknown"
                lesson["recommended_actions"] = []
            lesson["supporting_evidence"] = supporting
            sanitized.append(lesson)
        return sanitized

    def _strategy_outcomes_for_prompt(
        self, outcomes: Mapping[str, Any]
    ) -> Dict[str, Any]:
        if self.tir_evidence_policy == "visible":
            return {str(name): dict(value) for name, value in outcomes.items()}
        sanitized = {}
        for name, value in outcomes.items():
            item = dict(value) if isinstance(value, Mapping) else {}
            item.pop("model_valid", None)
            item.pop("compiled_unique", None)
            sanitized[str(name)] = item
        return sanitized

    def _state_for_prompt(self, record: CandidateRecord) -> str:
        if self.tir_evidence_policy == "visible":
            return record.state
        if record.state in {
            "modeled",
            "modeled-not-promoted",
            "compiled-equivalent",
            "model-skipped",
        }:
            return "not-measured"
        if record.state in {"model-invalid", "seed-model-invalid"}:
            return "compiler-invalid"
        return record.state

    def _strategy_planning_evidence(
        self, record: CandidateRecord, round_number: int
    ) -> Dict[str, Any]:
        """Return bounded observed, predicted, and strategy-outcome evidence."""

        prior_plans = []
        for plan in self.strategy_plans[-3:]:
            prior_plans.append(
                {
                    "round": plan.get("round"),
                    "source": plan.get("source"),
                    "allocation": plan.get("allocation", []),
                    "overrides": plan.get("overrides", []),
                    "fallback_reason": plan.get("fallback_reason"),
                    "realized_outcomes": self._strategy_outcomes_for_prompt(
                        dict(plan.get("realized_outcomes") or {})
                    ),
                }
            )
        diagnosis = self._diagnosis_for_prompt(record)
        return {
            "current_best": {
                "candidate_id": record.candidate.candidate_id,
                "generation": record.candidate.generation,
                "measured_latency_ms": (
                    record.measurement.latency_ms
                    if record.measurement and record.measurement.correct
                    else None
                ),
            },
            "model": (
                {
                    "predicted_latency_ms": record.model.predicted_latency_ms,
                    "calibrated_latency_ms": record.model.calibrated_latency_ms,
                    "bottleneck": record.model.bottleneck,
                    "confidence": record.model.confidence,
                    "diagnostics": record.model.diagnostics,
                }
                if self.tir_evidence_policy == "visible" and record.model
                else None
            ),
            "profile": (
                {
                    "bottleneck": record.profile.bottleneck,
                    "metrics": {
                        name: value
                        for name, value in record.profile.metrics.items()
                        if name
                        in {
                            "achieved_occupancy",
                            "achieved_occupancy_percent",
                            "ddr_util",
                            "dram_utilization",
                            "l2_hit_rate",
                            "l2_util",
                            "registers_per_thread",
                            "shared_memory_per_block",
                            "smem_util",
                            "tensor_util",
                            "cuda_util",
                        }
                    },
                }
                if record.profile and record.profile.valid
                else None
            ),
            "diagnosis": (
                {
                    "category": diagnosis.get("category"),
                    "confidence": diagnosis.get("confidence"),
                    "summary": diagnosis.get("summary"),
                    "limiting_factors": diagnosis.get("limiting_factors"),
                    "recommendations": diagnosis.get("recommendations"),
                    "source": diagnosis.get("source"),
                }
                if diagnosis
                else None
            ),
            "model_trust": (
                self.trust.to_dict()
                if self.tir_evidence_policy == "visible"
                else None
            ),
            "calibration": (
                self.calibrator.to_dict()
                if self.tir_evidence_policy == "visible"
                else None
            ),
            "analytical_model_coverage": (
                {
                    "directly_represented": [
                        "global, L2, and shared-memory traffic volume",
                        "tensor, CUDA-core, and SFU operation volume",
                        "register/shared-memory occupancy limits",
                        "CTA wave and launch underfill",
                    ],
                    "not_directly_represented": [
                        "shared-memory bank conflicts",
                        "instruction scheduling and dependency stalls",
                        "compiler code-generation quality",
                    ],
                    "routing_rule": (
                        "Do not devote most slots to an unmodeled effect. Use a "
                        "small diverse probe and rely on CUDA-event or NCU evidence "
                        "before exploiting that direction."
                    ),
                }
                if self.tir_evidence_policy == "visible"
                else None
            ),
            "strategy_outcomes": self._strategy_outcomes_for_prompt(
                self._strategy_outcome_summary(before_round=round_number)
            ),
            "discovered_strategy_memory": self._discovered_strategy_memory(
                round_number
            ),
            "prior_plans": prior_plans,
            "search_state": {
                "round": round_number,
                "total_rounds": self.task.budget.rounds,
                "candidate_slots": self.task.budget.proposals_per_round,
                "structural_search_policy": self.structural_search_policy,
                "evaluation_policy": self.evaluation_policy,
                "tir_evidence_policy": self.tir_evidence_policy,
                "measured_candidates": sum(
                    item.is_measured_correct for item in self.records.values()
                ),
            },
        }

    def _strategy_outcome_summary(
        self,
        *,
        before_round: Optional[int] = None,
        generation: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Attribute compiler and hardware outcomes to requested strategies."""

        grouped: Dict[str, Dict[str, Any]] = {}
        improvements: Dict[str, List[float]] = {}
        for record in self.records.values():
            candidate = record.candidate
            if candidate.generation == 0:
                continue
            if before_round is not None and candidate.generation >= before_round:
                continue
            if generation is not None and candidate.generation != generation:
                continue
            strategy_id = str(
                candidate.proposal_metadata.get("strategy_id") or "unassigned"
            )
            outcome = grouped.setdefault(
                strategy_id,
                {
                    "generated": 0,
                    "ast_checked": 0,
                    "ast_accepted": 0,
                    "model_valid": 0,
                    "compiled_unique": 0,
                    "measured_correct": 0,
                    "measured_improvements": 0,
                    "profiled": 0,
                    "profile_bottlenecks": {},
                    "best_candidate_id": None,
                    "best_measured_latency_ms": None,
                    "failure_categories": {},
                },
            )
            outcome["generated"] += 1
            strategy_validation = candidate.proposal_metadata.get(
                "strategy_validation"
            )
            if strategy_validation is not None:
                outcome["ast_checked"] += 1
            if strategy_validation in {
                "accepted",
                "observed-mismatch",
            }:
                outcome["ast_accepted"] += 1
            if record.model is not None and record.model.valid:
                outcome["model_valid"] += 1
                if record.compiled_equivalent_to is None:
                    outcome["compiled_unique"] += 1
            if (
                record.failure is not None
                and not is_infrastructure_failure(record.failure)
            ):
                failures = outcome["failure_categories"]
                failures[record.failure.category] = (
                    failures.get(record.failure.category, 0) + 1
                )
            if record.profile is not None and record.profile.valid:
                outcome["profiled"] += 1
                bottleneck = str(record.profile.bottleneck or "unknown")
                bottlenecks = outcome["profile_bottlenecks"]
                bottlenecks[bottleneck] = bottlenecks.get(bottleneck, 0) + 1
            if not record.is_measured_correct:
                continue
            outcome["measured_correct"] += 1
            latency = float(record.measurement.latency_ms)
            if (
                outcome["best_measured_latency_ms"] is None
                or latency < outcome["best_measured_latency_ms"]
            ):
                outcome["best_measured_latency_ms"] = latency
                outcome["best_candidate_id"] = candidate.candidate_id
            parent = self.records.get(candidate.parent_id or "")
            if parent is None or not parent.is_measured_correct:
                continue
            parent_latency = float(parent.measurement.latency_ms)
            if parent_latency <= 0:
                continue
            relative = (parent_latency - latency) / parent_latency
            improvements.setdefault(strategy_id, []).append(relative)
            if relative > 0:
                outcome["measured_improvements"] += 1

        for strategy_id, outcome in grouped.items():
            values = improvements.get(strategy_id, [])
            outcome["best_relative_improvement_vs_parent"] = (
                max(values) if values else None
            )
            outcome["mean_relative_improvement_vs_parent"] = (
                sum(values) / len(values) if values else None
            )
        return grouped

    def _record_round_strategy_outcomes(self, round_number: int) -> None:
        outcomes = self._strategy_outcome_summary(generation=round_number)
        changed = False
        for plan in self.strategy_plans:
            if int(plan.get("round", -1)) == round_number:
                plan["realized_outcomes"] = outcomes
                changed = True
                break
        if changed:
            self.store.save_json_artifact(
                "strategy_plans.json", {"rounds": self.strategy_plans}
            )
            self.store.append_event(
                "strategy_outcomes_recorded",
                {"round": round_number, "outcomes": outcomes},
            )

    def _discovered_strategy_memory(
        self, round_number: int
    ) -> List[Dict[str, Any]]:
        """Summarize earlier AI-discovered strategies and measured outcomes."""

        memory = []
        for record in self.records.values():
            if record.candidate.generation >= round_number:
                continue
            proposal = record.candidate.proposal_metadata
            if proposal.get("strategy_id") != "open-structural-exploration":
                continue
            discovered = str(
                proposal.get("discovered_strategy") or ""
            ).strip()[:160]
            if not discovered:
                continue
            related_value = proposal.get("related_existing_strategies") or []
            if isinstance(related_value, str):
                related = [related_value]
            elif isinstance(related_value, Sequence):
                related = [str(item) for item in related_value]
            else:
                related = []
            parent = self.records.get(record.candidate.parent_id or "")
            relative_improvement = None
            status = "not-measured"
            if record.is_measured_correct:
                status = "measured"
                if parent is not None and parent.is_measured_correct:
                    parent_latency = float(parent.measurement.latency_ms)
                    relative_improvement = (
                        parent_latency - float(record.measurement.latency_ms)
                    ) / parent_latency
                    status = (
                        "measured-improvement"
                        if relative_improvement > 0
                        else "measured-regression-or-tie"
                    )
            elif record.failure is not None:
                status = "rejected:%s" % record.failure.category
            elif (
                self.tir_evidence_policy == "visible"
                and record.model is not None
                and record.model.valid
            ):
                status = "model-only"
            memory.append(
                {
                    "candidate_id": record.candidate.candidate_id,
                    "generation": record.candidate.generation,
                    "discovered_strategy": discovered,
                    "related_existing_strategies": related[:6],
                    "hypothesis": record.candidate.hypothesis[:500],
                    "status": status,
                    "relative_improvement_vs_parent": relative_improvement,
                    "measured_latency_ms": (
                        record.measurement.latency_ms
                        if record.measurement and record.measurement.correct
                        else None
                    ),
                }
            )
        memory.sort(
            key=lambda item: (
                item["status"] != "measured-improvement",
                -float(item["relative_improvement_vs_parent"] or 0.0),
                -int(item["generation"]),
                str(item["candidate_id"]),
            )
        )
        return memory[:8]

    def _history(self) -> List[Dict[str, Any]]:
        records = sorted(
            self.records.values(),
            key=lambda item: (
                item.candidate.generation,
                item.candidate.candidate_id,
            ),
        )
        if self.metadata_compression_policy == NO_COMPRESSION_POLICY:
            return [item.to_dict(include_source=False) for item in records]
        # The prompt compressor applies the model-facing limit. Keep a larger
        # compact-record reservoir so controlled budget-growth experiments can
        # pack additional history without changing normal eight-record prompts.
        prompt_records = records[-64:]
        return [
            {
                "candidate_id": item.candidate.candidate_id,
                "parent_id": item.candidate.parent_id,
                "generation": item.candidate.generation,
                "source_sha256": item.candidate.source_sha256,
                "hypothesis": item.candidate.hypothesis,
                "strategy_slot": item.candidate.proposal_metadata.get(
                    "strategy_slot"
                ),
                "strategy_id": item.candidate.proposal_metadata.get("strategy_id"),
                "strategy_validation": item.candidate.proposal_metadata.get(
                    "strategy_validation"
                ),
                "strategy_selection_reason": (
                    item.candidate.proposal_metadata.get(
                        "strategy_selection_reason"
                    )
                ),
                "discovered_strategy": item.candidate.proposal_metadata.get(
                    "discovered_strategy"
                ),
                "related_existing_strategies": (
                    item.candidate.proposal_metadata.get(
                        "related_existing_strategies"
                    )
                ),
                "novelty_classification": dict(
                    item.candidate.proposal_metadata.get("ast_novelty") or {}
                ).get("classification"),
                "structural_change": dict(
                    item.candidate.proposal_metadata.get("ast_novelty") or {}
                ).get("structural_change"),
                "state": self._state_for_prompt(item),
                "lineage_kind": item.candidate.lineage_kind,
                "repair_depth": item.candidate.repair_depth,
                "predicted_latency_ms": (
                    item.model.predicted_latency_ms
                    if self.tir_evidence_policy == "visible" and item.model
                    else None
                ),
                "calibrated_predicted_latency_ms": (
                    item.model.calibrated_latency_ms
                    if self.tir_evidence_policy == "visible" and item.model
                    else None
                ),
                "measured_latency_ms": (
                    item.measurement.latency_ms if item.measurement else None
                ),
                "failure": item.failure.to_dict() if item.failure else None,
                "diagnosis": (
                    {
                        "category": diagnosis.get("category"),
                        "confidence": diagnosis.get("confidence"),
                        "summary": diagnosis.get("summary"),
                        "recommendations": diagnosis.get("recommendations"),
                    }
                    if (diagnosis := self._diagnosis_for_prompt(item))
                    else None
                ),
                "profile_bottleneck": (
                    item.profile.bottleneck if item.profile else None
                ),
            }
            for item in prompt_records
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
        termination_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = {
            "schema_version": 1,
            "phase": phase,
            "completed_round": completed_round,
            "active_round": active_round,
            "round_candidate_ids": list(round_candidate_ids or []),
            "selected_ids": list(selected_ids or []),
            "best_before_id": best_before_id,
            "termination_reason": termination_reason,
            "max_search_seconds": self.max_search_seconds,
            "search_elapsed_seconds": self._search_elapsed_seconds(),
            "candidate_graph_upper_bound": self.candidate_graph_upper_bound,
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
            "selection_state": self.selection.snapshot(),
            "experiment": self._experiment_config(),
            "profile_calls": self.profile_calls,
            "preflight_calls": self.preflight_calls,
            "api_request_attempts": self.api_request_attempts,
            "generator_calls": self.generator_calls,
            "planner_calls": self.planner_calls,
            "planner_fallbacks": self.planner_fallbacks,
            "strategy_plans": self.strategy_plans,
            "generator_usage": dict(self.generator_usage),
            "repair_calls": self.repair_calls,
            "repair_candidates": self.repair_candidates,
            "final_validation_calls": self.final_validation_calls,
            "final_validation_passes": self.final_validation_passes,
            "search_best_id": self._search_best_id,
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
                "api_request_attempts": self.api_request_attempts,
                "generator_calls": self.generator_calls,
                "planner_calls": self.planner_calls,
                "planner_fallbacks": self.planner_fallbacks,
                "strategy_plans": self.strategy_plans,
                "generator_usage": dict(self.generator_usage),
                "repair_calls": self.repair_calls,
                "repair_candidates": self.repair_candidates,
                "final_validation_calls": self.final_validation_calls,
                "final_validation_passes": self.final_validation_passes,
                "search_best_id": self._search_best_id,
                "repairs_by_round": {
                    str(name): value
                    for name, value in self._repairs_by_round.items()
                },
                "calibration": self.calibrator.snapshot(),
                "evidence_memory": self.evidence_memory.snapshot(),
                "stage_timings": self.stage_timings,
                "termination_reason": self.termination_reason,
                "max_search_seconds": self.max_search_seconds,
                "search_elapsed_seconds": self._search_elapsed_seconds(),
                "candidate_graph_upper_bound": self.candidate_graph_upper_bound,
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

    def _record_api_attempts(self, metadata: Mapping[str, Any]) -> None:
        attempts = metadata.get("attempts", 1)
        if isinstance(attempts, (int, float)) and not isinstance(attempts, bool):
            self.api_request_attempts += max(0, int(attempts))

    @staticmethod
    def _compiled_identity_hash(record: CandidateRecord) -> Optional[str]:
        if record.model is None:
            return None
        value = record.model.metrics.get("compiled_identity_sha256")
        if not value:
            value = record.model.metrics.get("compiled_source_sha256")
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip().lower()

    def _register_compiled_identity(
        self, record: CandidateRecord
    ) -> Optional[str]:
        if not self.compiled_deduplication:
            record.compiled_equivalent_to = None
            return None
        compiled_hash = self._compiled_identity_hash(record)
        if compiled_hash is None:
            return None
        identifier = record.candidate.candidate_id
        owner = self.compiled_hash_owners.get(compiled_hash)
        if owner is not None and owner != identifier:
            record.compiled_equivalent_to = owner
            return owner
        self.compiled_hash_owners[compiled_hash] = identifier
        record.compiled_equivalent_to = None
        return None

    def _rebuild_compiled_hash_owners(self) -> None:
        self.compiled_hash_owners = {}
        if not self.compiled_deduplication:
            return
        for record in sorted(
            self.records.values(),
            key=lambda item: (
                item.candidate.generation,
                item.candidate.candidate_id,
            ),
        ):
            if record.model is None or not record.model.valid:
                continue
            compiled_hash = self._compiled_identity_hash(record)
            if compiled_hash is None:
                continue
            owner = record.compiled_equivalent_to or record.candidate.candidate_id
            self.compiled_hash_owners.setdefault(compiled_hash, owner)

    def _experiment_config(self) -> Dict[str, Any]:
        return {
            "baseline_style": self.baseline_style,
            "baseline_style_mode": self.baseline_style_mode,
            "baseline_style_canonical": self.baseline_style_canonical,
            "evaluation_policy": self.evaluation_policy,
            "requested_tir_evidence_policy": self.requested_tir_evidence_policy,
            "tir_evidence_policy": self.tir_evidence_policy,
            "requested_selection_policy": self.requested_selection_policy_name,
            "selection_policy": self.selection_policy_name,
            "requested_profile_policy": self.requested_profile_policy_name,
            "profile_policy": self.profile_policy_name,
            "fixed_promotions_per_round": self.fixed_promotions_per_round,
            "requested_compiled_deduplication": (
                self.requested_compiled_deduplication
            ),
            "compiled_deduplication": self.compiled_deduplication,
            "structural_search_policy": self.structural_search_policy,
            "strategy_allocation_policy": self.strategy_allocation_policy,
            "metadata_compression_policy": self.metadata_compression_policy,
            "api_input_price_per_million": self.api_input_price_per_million,
            "api_output_price_per_million": self.api_output_price_per_million,
        }

    @staticmethod
    def _normalized_experiment_config(value: Mapping[str, Any]) -> Dict[str, Any]:
        """Treat checkpoints created before ablation controls as default runs."""

        normalized = dict(value)
        normalized.setdefault("baseline_style", "native")
        normalized.setdefault("baseline_style_mode", "native")
        normalized.setdefault("baseline_style_canonical", True)
        normalized.setdefault("evaluation_policy", "tilesight")
        normalized.setdefault("metadata_compression_policy", COMPRESSION_VERSION)
        normalized.setdefault("requested_tir_evidence_policy", "auto")
        normalized.setdefault("tir_evidence_policy", "visible")
        normalized.setdefault(
            "requested_selection_policy", normalized.get("selection_policy")
        )
        normalized.setdefault(
            "requested_profile_policy", normalized.get("profile_policy")
        )
        normalized.setdefault(
            "requested_compiled_deduplication",
            normalized.get("compiled_deduplication", True),
        )
        return normalized

    def _summary(
        self,
        completed_rounds: int,
        seed: CandidateRecord,
        search_best: CandidateRecord,
        final_seed: CandidateRecord,
        best_source_path: Path,
        elapsed_seconds: float,
    ) -> SearchSummary:
        best = self.beam[0]
        final_enabled = self.task.budget.final_validation_candidates > 0
        seed_latency = float(
            final_seed.final_measurement.latency_ms
            if final_enabled and final_seed.final_measurement is not None
            else seed.measurement.latency_ms
        )
        best_latency = float(
            best.final_measurement.latency_ms
            if final_enabled and best.final_measurement is not None
            else best.measurement.latency_ms
        )
        measured = [item for item in self.records.values() if item.measurement]
        correct = [item for item in measured if item.measurement.correct]
        ledger = build_cost_ledger(
            self.records.values(),
            stage_timings=self.stage_timings,
            generator_usage=self.generator_usage,
            preflight_calls=self.preflight_calls,
            provider_api_requests=self.api_request_attempts,
            generator_calls=self.generator_calls,
            planner_calls=self.planner_calls,
            repair_calls=self.repair_calls,
            profile_calls=self.profile_calls,
            final_validation_calls=self.final_validation_calls,
            api_input_price_per_million=self.api_input_price_per_million,
            api_output_price_per_million=self.api_output_price_per_million,
        )
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
            search_best_candidate_id=search_best.candidate.candidate_id,
            search_best_latency_ms=float(search_best.measurement.latency_ms),
            final_validation_calls=self.final_validation_calls,
            final_validation_passes=self.final_validation_passes,
            final_seed_latency_ms=(seed_latency if final_enabled else None),
            evaluation_policy=self.evaluation_policy,
            requested_tir_evidence_policy=self.requested_tir_evidence_policy,
            tir_evidence_policy=self.tir_evidence_policy,
            requested_selection_policy=self.requested_selection_policy_name,
            selection_policy=self.selection_policy_name,
            requested_profile_policy=self.requested_profile_policy_name,
            profile_policy=self.profile_policy_name,
            fixed_promotions_per_round=self.fixed_promotions_per_round,
            requested_compiled_deduplication=(
                self.requested_compiled_deduplication
            ),
            compiled_deduplication=self.compiled_deduplication,
            compiled_equivalent_candidates=sum(
                item.compiled_equivalent_to is not None
                for item in self.records.values()
            ),
            structural_search_policy=self.structural_search_policy,
            strategy_allocation_policy=self.strategy_allocation_policy,
            metadata_compression_policy=self.metadata_compression_policy,
            planner_calls=self.planner_calls,
            planner_fallbacks=self.planner_fallbacks,
            structural_candidates=sum(
                bool(
                    dict(item.candidate.proposal_metadata.get("ast_novelty") or {}).get(
                        "structural_change"
                    )
                )
                for item in self.records.values()
            ),
            parameter_only_candidates=sum(
                dict(item.candidate.proposal_metadata.get("ast_novelty") or {}).get(
                    "classification"
                )
                == "parameter-only"
                for item in self.records.values()
            ),
            strategy_rejected_candidates=sum(
                item.state == "strategy-invalid" for item in self.records.values()
            ),
            open_exploration_candidates=sum(
                item.candidate.proposal_metadata.get("strategy_id")
                == "open-structural-exploration"
                for item in self.records.values()
            ),
            discovered_strategy_count=len(
                {
                    str(
                        item.candidate.proposal_metadata.get(
                            "discovered_strategy"
                        )
                    ).strip()
                    for item in self.records.values()
                    if item.candidate.proposal_metadata.get("strategy_id")
                    == "open-structural-exploration"
                    and str(
                        item.candidate.proposal_metadata.get(
                            "discovered_strategy"
                        )
                        or ""
                    ).strip()
                }
            ),
            baseline_style=self.baseline_style,
            baseline_style_mode=self.baseline_style_mode,
            baseline_style_canonical=self.baseline_style_canonical,
            incumbent_snapshot_interval_seconds=(
                self.incumbent_snapshot_interval_seconds
            ),
            max_search_seconds=self.max_search_seconds,
            time_budget_exhausted=(self.termination_reason == "time-budget"),
            termination_reason=(
                self.termination_reason or "round-budget-completed"
            ),
            candidate_graph_upper_bound=self.candidate_graph_upper_bound,
            cost_ledger=ledger,
            report_paths=expected_report_paths(self.store.root),
        )


def _unique(values: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
