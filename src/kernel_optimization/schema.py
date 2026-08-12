"""Serializable contracts shared by source generation, evaluation, and search."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


JsonDict = Dict[str, Any]


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _require_nonnegative_float(name: str, value: Any) -> float:
    result = float(value)
    if result < 0:
        raise ValueError("%s must be non-negative" % name)
    return result


def source_digest(source_code: str) -> str:
    """Return the exact UTF-8 source digest used for candidate identity."""

    return hashlib.sha256(source_code.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BudgetConfig:
    """Search and hardware-evaluation budgets for one task."""

    rounds: int = 3
    proposals_per_round: int = 6
    beam_width: int = 3
    min_promotions_per_round: int = 2
    max_promotions_per_round: int = 4
    ncu_improvement_threshold: float = 0.05
    ncu_max_staleness_rounds: int = 3
    ncu_plateau_rounds: int = 2
    random_seed: int = 0

    def __post_init__(self) -> None:
        _require_positive_int("rounds", self.rounds)
        _require_positive_int("proposals_per_round", self.proposals_per_round)
        _require_positive_int("beam_width", self.beam_width)
        _require_positive_int(
            "min_promotions_per_round", self.min_promotions_per_round
        )
        _require_positive_int(
            "max_promotions_per_round", self.max_promotions_per_round
        )
        if self.min_promotions_per_round > self.max_promotions_per_round:
            raise ValueError(
                "min_promotions_per_round cannot exceed max_promotions_per_round"
            )
        _require_nonnegative_float(
            "ncu_improvement_threshold", self.ncu_improvement_threshold
        )
        _require_positive_int(
            "ncu_max_staleness_rounds", self.ncu_max_staleness_rounds
        )
        _require_positive_int("ncu_plateau_rounds", self.ncu_plateau_rounds)
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")

    @classmethod
    def from_dict(cls, value: Optional[Mapping[str, Any]]) -> "BudgetConfig":
        return cls(**dict(value or {}))


@dataclass(frozen=True)
class TaskSpec:
    """Optimization contract surrounding a user-provided kernel source file."""

    task_id: str
    description: str
    reference: str
    entrypoint: str
    language: str = "python"
    target: JsonDict = field(default_factory=dict)
    workload: JsonDict = field(default_factory=dict)
    constraints: JsonDict = field(default_factory=dict)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    evaluator: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("task_id", "description", "reference", "entrypoint"):
            if not str(getattr(self, name)).strip():
                raise ValueError("%s cannot be empty" % name)
        if self.language != "python":
            raise ValueError("only Python kernel sources are currently supported")
        if not isinstance(self.target, dict):
            raise ValueError("target must be a JSON object")
        if not isinstance(self.workload, dict):
            raise ValueError("workload must be a JSON object")
        if not isinstance(self.constraints, dict):
            raise ValueError("constraints must be a JSON object")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        data = dict(value)
        data["budget"] = BudgetConfig.from_dict(data.get("budget"))
        for name in ("target", "workload", "constraints", "evaluator", "metadata"):
            data[name] = dict(data.get(name) or {})
        return cls(**data)

    @classmethod
    def from_json_file(cls, path: Path) -> "TaskSpec":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class CandidateProposal:
    """One API-proposed complete replacement for the current kernel source."""

    hypothesis: str
    source_code: str
    expected_effect: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hypothesis.strip():
            raise ValueError("candidate hypothesis cannot be empty")
        if not self.source_code.strip():
            raise ValueError("candidate source_code cannot be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateProposal":
        data = dict(value)
        data["expected_effect"] = dict(data.get("expected_effect") or {})
        data["metadata"] = dict(data.get("metadata") or {})
        return cls(**data)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    """A complete kernel source candidate with a content-derived identity."""

    candidate_id: str
    task_id: str
    parent_id: Optional[str]
    generation: int
    source_name: str
    source_sha256: str
    source_code: str
    hypothesis: str
    expected_effect: JsonDict = field(default_factory=dict)
    proposal_metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def seed(cls, task: TaskSpec, source_code: str, source_name: str) -> "Candidate":
        return cls._create(
            task=task,
            parent_id=None,
            generation=0,
            source_name=source_name,
            source_code=source_code,
            hypothesis="Initial user-provided kernel source.",
            expected_effect={},
            proposal_metadata={"generator": "seed"},
        )

    @classmethod
    def from_proposal(
        cls,
        task: TaskSpec,
        parent: "Candidate",
        proposal: CandidateProposal,
        generation: int,
    ) -> "Candidate":
        return cls._create(
            task=task,
            parent_id=parent.candidate_id,
            generation=generation,
            source_name=parent.source_name,
            source_code=proposal.source_code,
            hypothesis=proposal.hypothesis,
            expected_effect=proposal.expected_effect,
            proposal_metadata=proposal.metadata,
        )

    @classmethod
    def _create(
        cls,
        task: TaskSpec,
        parent_id: Optional[str],
        generation: int,
        source_name: str,
        source_code: str,
        hypothesis: str,
        expected_effect: Mapping[str, Any],
        proposal_metadata: Mapping[str, Any],
    ) -> "Candidate":
        safe_name = Path(source_name).name
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("source_name must identify a file")
        digest = source_digest(source_code)
        identity = "%s\0%s" % (task.task_id, digest)
        candidate_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        return cls(
            candidate_id=candidate_id,
            task_id=task.task_id,
            parent_id=parent_id,
            generation=generation,
            source_name=safe_name,
            source_sha256=digest,
            source_code=source_code,
            hypothesis=hypothesis,
            expected_effect=dict(expected_effect),
            proposal_metadata=dict(proposal_metadata),
        )

    def to_dict(self, include_source: bool = True) -> JsonDict:
        result = asdict(self)
        if not include_source:
            result.pop("source_code")
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Candidate":
        data = dict(value)
        data["expected_effect"] = dict(data.get("expected_effect") or {})
        data["proposal_metadata"] = dict(data.get("proposal_metadata") or {})
        return cls(**data)


@dataclass(frozen=True)
class ModelEvaluation:
    """Low-cost analytical prediction for a candidate."""

    valid: bool
    predicted_latency_ms: Optional[float] = None
    bottleneck: str = "unknown"
    confidence: str = "unknown"
    metrics: JsonDict = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.valid:
            if self.predicted_latency_ms is None or self.predicted_latency_ms <= 0:
                raise ValueError(
                    "a valid model evaluation needs positive predicted_latency_ms"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelEvaluation":
        data = dict(value)
        data["metrics"] = dict(data.get("metrics") or {})
        data["diagnostics"] = list(data.get("diagnostics") or [])
        return cls(**data)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class Measurement:
    """Correctness result and optional measured CUDA-event latency."""

    correct: bool
    latency_ms: Optional[float] = None
    samples_ms: List[float] = field(default_factory=list)
    metrics: JsonDict = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.correct and (self.latency_ms is None or self.latency_ms <= 0):
            raise ValueError("a correct measurement needs positive latency_ms")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Measurement":
        data = dict(value)
        data["samples_ms"] = [float(item) for item in data.get("samples_ms") or []]
        data["metrics"] = dict(data.get("metrics") or {})
        return cls(**data)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class ProfileEvaluation:
    """Expensive hardware profile collected at a milestone."""

    bottleneck: str
    metrics: JsonDict = field(default_factory=dict)
    report_path: Optional[str] = None
    valid: bool = True
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProfileEvaluation":
        data = dict(value)
        data["metrics"] = dict(data.get("metrics") or {})
        return cls(**data)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class CandidateRecord:
    """Accumulated evidence and decision state for one source candidate."""

    candidate: Candidate
    state: str = "generated"
    model: Optional[ModelEvaluation] = None
    measurement: Optional[Measurement] = None
    profile: Optional[ProfileEvaluation] = None
    selection_reasons: List[str] = field(default_factory=list)
    decision_reason: Optional[str] = None

    @property
    def is_measured_correct(self) -> bool:
        return bool(
            self.measurement is not None
            and self.measurement.correct
            and self.measurement.latency_ms is not None
        )

    def to_dict(self, include_source: bool = False) -> JsonDict:
        return {
            "candidate": self.candidate.to_dict(include_source=include_source),
            "state": self.state,
            "model": self.model.to_dict() if self.model else None,
            "measurement": self.measurement.to_dict() if self.measurement else None,
            "profile": self.profile.to_dict() if self.profile else None,
            "selection_reasons": list(self.selection_reasons),
            "decision_reason": self.decision_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateRecord":
        data = dict(value)
        candidate = Candidate.from_dict(data["candidate"])
        model = data.get("model")
        measurement = data.get("measurement")
        profile = data.get("profile")
        return cls(
            candidate=candidate,
            state=str(data.get("state", "generated")),
            model=ModelEvaluation.from_dict(model) if model else None,
            measurement=Measurement.from_dict(measurement) if measurement else None,
            profile=ProfileEvaluation.from_dict(profile) if profile else None,
            selection_reasons=list(data.get("selection_reasons") or []),
            decision_reason=data.get("decision_reason"),
        )


@dataclass(frozen=True)
class SearchSummary:
    """Final optimization outcome and cost counters."""

    task_id: str
    best_candidate_id: str
    best_source_path: str
    best_latency_ms: float
    seed_latency_ms: float
    speedup_over_seed: float
    completed_rounds: int
    generated_candidates: int
    modeled_candidates: int
    measured_candidates: int
    correct_candidates: int
    profile_calls: int
    generator_calls: int
    generator_usage: JsonDict
    elapsed_seconds: float
    trust: JsonDict
    output_directory: str
    preflight_calls: int = 0
    resumed: bool = False
    stage_timings: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)
