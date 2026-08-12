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
from .source_validation import SourceValidationError, SourceValidator

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
    "SourceValidationError",
    "SourceValidator",
    "TaskSpec",
]

__version__ = "0.3.0"
