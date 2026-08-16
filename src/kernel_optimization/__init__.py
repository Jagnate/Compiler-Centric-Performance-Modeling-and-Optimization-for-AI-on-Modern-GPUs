"""Adaptive-fidelity GPU kernel optimization framework."""

from .controller import OptimizationController
from .schema import (
    BudgetConfig,
    BottleneckDiagnosis,
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
from .source_validation import SourceValidationError, SourceValidator
from .structural_search import (
    SourceNoveltyAnalyzer,
    SourceNoveltyReport,
    StrategyAssignment,
    StructuralStrategyPortfolio,
)

__all__ = [
    "BudgetConfig",
    "BottleneckDiagnosis",
    "Candidate",
    "CandidateProposal",
    "CandidateRecord",
    "FailureEvidence",
    "Measurement",
    "ModelEvaluation",
    "OptimizationController",
    "ProfileEvaluation",
    "SearchSummary",
    "SourceValidationError",
    "SourceValidator",
    "SourceNoveltyAnalyzer",
    "SourceNoveltyReport",
    "StrategyAssignment",
    "StructuralStrategyPortfolio",
    "TaskSpec",
]

__version__ = "0.7.0"
