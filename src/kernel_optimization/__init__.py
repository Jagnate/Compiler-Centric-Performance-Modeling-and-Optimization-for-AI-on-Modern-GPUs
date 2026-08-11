"""Adaptive-fidelity GPU kernel optimization framework."""

from .controller import OptimizationController
from .schema import (
    BudgetConfig,
    Candidate,
    CandidateProposal,
    CandidateRecord,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    SearchSummary,
    TaskSpec,
)

__all__ = [
    "BudgetConfig",
    "Candidate",
    "CandidateProposal",
    "CandidateRecord",
    "Measurement",
    "ModelEvaluation",
    "OptimizationController",
    "ProfileEvaluation",
    "SearchSummary",
    "TaskSpec",
]

__version__ = "0.1.0"

