"""Extension protocols for candidate generation and performance evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Sequence

from .schema import (
    Candidate,
    CandidateProposal,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    TaskSpec,
)


class CandidateGenerator(Protocol):
    """Produce complete source candidates for one measured parent."""

    def generate(
        self,
        task: TaskSpec,
        parent: Candidate,
        evidence: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
        count: int,
    ) -> List[CandidateProposal]:
        ...

    def repair(
        self,
        task: TaskSpec,
        failed: Candidate,
        failure: Dict[str, Any],
        evidence: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
    ) -> CandidateProposal:
        ...


class PerformanceBackend(Protocol):
    """Evaluate candidates at progressively more expensive fidelity levels."""

    def model(self, task: TaskSpec, candidate: Candidate) -> ModelEvaluation:
        ...

    def measure(self, task: TaskSpec, candidate: Candidate) -> Measurement:
        ...

    def profile(self, task: TaskSpec, candidate: Candidate) -> ProfileEvaluation:
        ...

    def finalize(self, task: TaskSpec, candidate: Candidate) -> Measurement:
        ...
