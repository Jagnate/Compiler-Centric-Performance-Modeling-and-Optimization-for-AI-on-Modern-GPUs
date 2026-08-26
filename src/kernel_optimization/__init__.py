"""Adaptive-fidelity GPU kernel optimization framework."""

from .controller import OptimizationController
from .incumbent_tracking import PeriodicIncumbentRecorder
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
    StrategyPlan,
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
    "PeriodicIncumbentRecorder",
    "ProfileEvaluation",
    "SearchSummary",
    "SourceValidationError",
    "SourceValidator",
    "SourceNoveltyAnalyzer",
    "SourceNoveltyReport",
    "StrategyAssignment",
    "StrategyPlan",
    "StructuralStrategyPortfolio",
    "TaskSpec",
]

__version__ = "0.10.0"
