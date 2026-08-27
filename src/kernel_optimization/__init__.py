"""Adaptive-fidelity GPU kernel optimization framework."""

from .baseline_styles import (
    BASELINE_STYLE_NAMES,
    BaselineStylePreset,
    BaselineStyleResolution,
    get_baseline_style,
    resolve_baseline_style,
)
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
    "BASELINE_STYLE_NAMES",
    "BaselineStylePreset",
    "BaselineStyleResolution",
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
    "get_baseline_style",
    "resolve_baseline_style",
]

__version__ = "0.12.1"
