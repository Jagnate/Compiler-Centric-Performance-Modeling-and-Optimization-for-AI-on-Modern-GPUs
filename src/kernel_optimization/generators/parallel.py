"""Bounded parallelism for independent hosted-API generation agents."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..protocols import CandidateGenerator
from ..schema import Candidate, CandidateProposal, TaskSpec


GeneratorFactory = Callable[[], CandidateGenerator]


@dataclass
class _GenerationOutcome:
    worker_id: int
    requested_candidates: int
    strategy_slots: List[str]
    proposals: List[CandidateProposal]
    metadata: Dict[str, Any]
    exchange: Dict[str, Any]
    error: Optional[Exception] = None


class ParallelCandidateGenerator:
    """Split one logical generation request across isolated API clients.

    Planning and repair remain serial. Only independent candidate-generation
    requests are executed concurrently; compiler and GPU evaluation continue to
    be owned by the controller.
    """

    def __init__(self, factory: GeneratorFactory, max_workers: int = 1) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.factory = factory
        self.max_workers = int(max_workers)
        self._primary = factory()
        self.last_call_metadata: Dict[str, Any] = {}
        self.last_exchange: Dict[str, Any] = {}
        self.total_api_requests = 0

    def preflight(self) -> Dict[str, Any]:
        try:
            result = self._primary.preflight()  # type: ignore[attr-defined]
        except Exception:
            self._capture_serial_call(self._primary)
            raise
        self._capture_serial_call(self._primary)
        return dict(result)

    def plan_strategies(
        self,
        task: TaskSpec,
        parent: Candidate,
        planning_context: Dict[str, Any],
        strategies: Sequence[Dict[str, Any]],
        count: int,
        round_number: int,
    ) -> Dict[str, Any]:
        try:
            result = self._primary.plan_strategies(
                task,
                parent,
                planning_context,
                strategies,
                count,
                round_number,
            )
        except Exception:
            self._capture_serial_call(self._primary)
            raise
        self._capture_serial_call(self._primary)
        return result

    def repair(
        self,
        task: TaskSpec,
        failed: Candidate,
        failure: Dict[str, Any],
        evidence: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
    ) -> CandidateProposal:
        try:
            result = self._primary.repair(task, failed, failure, evidence, history)
        except Exception:
            self._capture_serial_call(self._primary)
            raise
        self._capture_serial_call(self._primary)
        return result

    def generate(
        self,
        task: TaskSpec,
        parent: Candidate,
        evidence: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
        count: int,
    ) -> List[CandidateProposal]:
        if count <= 0:
            return []
        active_workers = min(self.max_workers, count)
        if active_workers == 1:
            try:
                result = self._primary.generate(
                    task, parent, evidence, history, count
                )
            except Exception:
                self._capture_serial_call(self._primary)
                raise
            self._capture_serial_call(self._primary)
            return result

        partitions = _balanced_partitions(count, active_workers)
        started_at = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=active_workers,
            thread_name_prefix="kernel-opt-agent",
        ) as executor:
            futures = [
                executor.submit(
                    self._generate_partition,
                    worker_id,
                    task,
                    parent,
                    _partition_evidence(evidence, start, partition_count),
                    history,
                    partition_count,
                )
                for worker_id, (start, partition_count) in enumerate(partitions)
            ]
            # Reading futures in submission order keeps candidate ordering stable
            # even when workers finish in a different order.
            outcomes = [future.result() for future in futures]

        elapsed = time.perf_counter() - started_at
        self._capture_parallel_call(outcomes, elapsed, count)
        successful = [item for item in outcomes if item.error is None]
        if not successful:
            first_error = next(
                item.error for item in outcomes if item.error is not None
            )
            raise first_error
        proposals: List[CandidateProposal] = []
        for outcome in successful:
            proposals.extend(outcome.proposals)
        return proposals[:count]

    def _generate_partition(
        self,
        worker_id: int,
        task: TaskSpec,
        parent: Candidate,
        evidence: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
        count: int,
    ) -> _GenerationOutcome:
        generator: Optional[CandidateGenerator] = None
        slots = _strategy_slots(evidence)
        try:
            generator = self.factory()
            proposals = generator.generate(task, parent, evidence, history, count)
            return _GenerationOutcome(
                worker_id=worker_id,
                requested_candidates=count,
                strategy_slots=slots,
                proposals=list(proposals),
                metadata=_metadata(generator),
                exchange=_exchange(generator),
            )
        except Exception as error:
            return _GenerationOutcome(
                worker_id=worker_id,
                requested_candidates=count,
                strategy_slots=slots,
                proposals=[],
                metadata=_metadata(generator),
                exchange=_exchange(generator),
                error=error,
            )

    def _capture_serial_call(self, generator: CandidateGenerator) -> None:
        self.last_call_metadata = _metadata(generator)
        self.last_exchange = _exchange(generator)
        self.total_api_requests += _attempts(self.last_call_metadata)

    def _capture_parallel_call(
        self,
        outcomes: Sequence[_GenerationOutcome],
        elapsed_seconds: float,
        requested_candidates: int,
    ) -> None:
        usage: Dict[str, float] = {}
        attempts = 0
        worker_calls = []
        exchanges = []
        for outcome in outcomes:
            attempts += _attempts(outcome.metadata)
            _merge_numeric_mapping(usage, outcome.metadata.get("usage"))
            error = _serialize_error(outcome.error)
            worker_calls.append(
                {
                    "worker_id": outcome.worker_id,
                    "requested_candidates": outcome.requested_candidates,
                    "returned_candidates": len(outcome.proposals),
                    "strategy_slots": outcome.strategy_slots,
                    "provider": outcome.metadata,
                    "error": error,
                }
            )
            exchanges.append(
                {
                    "worker_id": outcome.worker_id,
                    "requested_candidates": outcome.requested_candidates,
                    "strategy_slots": outcome.strategy_slots,
                    "exchange": outcome.exchange,
                    "error": error,
                }
            )
        failed_workers = sum(item.error is not None for item in outcomes)
        self.last_call_metadata = {
            "kind": "parallel-generate",
            "parallel": True,
            "configured_workers": self.max_workers,
            "active_workers": len(outcomes),
            "successful_workers": len(outcomes) - failed_workers,
            "failed_workers": failed_workers,
            "requested_candidates": requested_candidates,
            "returned_candidates": sum(len(item.proposals) for item in outcomes),
            "attempts": attempts,
            "elapsed_seconds": elapsed_seconds,
            "usage": usage,
            "worker_calls": worker_calls,
        }
        self.last_exchange = {
            "kind": "parallel-generate",
            "workers": exchanges,
        }
        self.total_api_requests += attempts


def _balanced_partitions(total: int, parts: int) -> List[Tuple[int, int]]:
    base, remainder = divmod(total, parts)
    partitions = []
    start = 0
    for index in range(parts):
        count = base + (1 if index < remainder else 0)
        partitions.append((start, count))
        start += count
    return partitions


def _partition_evidence(
    evidence: Mapping[str, Any], start: int, count: int
) -> Dict[str, Any]:
    value = deepcopy(dict(evidence))
    generation_request = value.get("generation_request")
    if not isinstance(generation_request, Mapping):
        return value
    request = dict(generation_request)
    assignments = request.get("strategy_assignments")
    if isinstance(assignments, list):
        request["strategy_assignments"] = assignments[start : start + count]
        value["generation_request"] = request
    return value


def _strategy_slots(evidence: Mapping[str, Any]) -> List[str]:
    generation_request = evidence.get("generation_request")
    if not isinstance(generation_request, Mapping):
        return []
    assignments = generation_request.get("strategy_assignments")
    if not isinstance(assignments, list):
        return []
    return [
        str(item.get("slot"))
        for item in assignments
        if isinstance(item, Mapping) and item.get("slot") is not None
    ]


def _metadata(generator: Optional[CandidateGenerator]) -> Dict[str, Any]:
    if generator is None:
        return {}
    return dict(getattr(generator, "last_call_metadata", {}) or {})


def _exchange(generator: Optional[CandidateGenerator]) -> Dict[str, Any]:
    if generator is None:
        return {}
    return dict(getattr(generator, "last_exchange", {}) or {})


def _attempts(metadata: Mapping[str, Any]) -> int:
    value = metadata.get("attempts", 0)
    return max(0, int(value)) if isinstance(value, (int, float)) else 0


def _merge_numeric_mapping(
    destination: Dict[str, float], value: Any
) -> None:
    if not isinstance(value, Mapping):
        return
    for name, amount in value.items():
        if isinstance(amount, (int, float)) and not isinstance(amount, bool):
            destination[str(name)] = destination.get(str(name), 0.0) + float(amount)


def _serialize_error(error: Optional[Exception]) -> Optional[Dict[str, Any]]:
    if error is None:
        return None
    to_dict = getattr(error, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, dict):
            return value
    return {"type": type(error).__name__, "message": str(error)}
