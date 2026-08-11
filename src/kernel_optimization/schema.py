"""Serializable contracts shared by generators, evaluators, and search."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


JsonDict = Dict[str, Any]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return value


def _require_nonnegative_float(name: str, value: Any) -> float:
    result = float(value)
    if result < 0:
        raise ValueError("%s must be non-negative" % name)
    return result


@dataclass(frozen=True)
class BudgetConfig:
    """Search and hardware-evaluation budgets for one task."""

    rounds: int = 3
    proposals_per_round: int = 8
    beam_width: int = 3
    min_promotions_per_round: int = 3
    max_promotions_per_round: int = 6
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
    """Immutable optimization task and evaluator configuration."""

    task_id: str
    description: str
    reference: str
    base_parameters: JsonDict
    search_space: Dict[str, Sequence[Any]]
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    evaluator: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id cannot be empty")
        if not self.description.strip():
            raise ValueError("description cannot be empty")
        if not self.reference.strip():
            raise ValueError("reference cannot be empty")
        if not self.base_parameters:
            raise ValueError("base_parameters cannot be empty")
        unknown = set(self.base_parameters).difference(self.search_space)
        if unknown:
            raise ValueError(
                "base parameters are missing from search_space: %s"
                % ", ".join(sorted(unknown))
            )
        for name, values in self.search_space.items():
            if not values:
                raise ValueError("search_space[%r] cannot be empty" % name)
            if name in self.base_parameters and self.base_parameters[name] not in values:
                raise ValueError(
                    "base parameter %r=%r is not present in its search space"
                    % (name, self.base_parameters[name])
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        data = dict(value)
        data["budget"] = BudgetConfig.from_dict(data.get("budget"))
        data["base_parameters"] = dict(data.get("base_parameters") or {})
        data["search_space"] = {
            str(name): tuple(values)
            for name, values in dict(data.get("search_space") or {}).items()
        }
        data["evaluator"] = dict(data.get("evaluator") or {})
        data["metadata"] = dict(data.get("metadata") or {})
        return cls(**data)

    @classmethod
    def from_json_file(cls, path: Path) -> "TaskSpec":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> JsonDict:
        result = asdict(self)
        result["search_space"] = {
            name: list(values) for name, values in self.search_space.items()
        }
        return result

    def merge_parameters(self, updates: Mapping[str, Any]) -> JsonDict:
        unknown = set(updates).difference(self.search_space)
        if unknown:
            raise ValueError(
                "proposal contains unsupported parameters: %s"
                % ", ".join(sorted(unknown))
            )
        result = dict(self.base_parameters)
        result.update(updates)
        self.validate_parameters(result)
        return result

    def validate_parameters(self, parameters: Mapping[str, Any]) -> None:
        missing = set(self.base_parameters).difference(parameters)
        if missing:
            raise ValueError(
                "candidate is missing parameters: %s" % ", ".join(sorted(missing))
            )
        unknown = set(parameters).difference(self.search_space)
        if unknown:
            raise ValueError(
                "candidate has unsupported parameters: %s"
                % ", ".join(sorted(unknown))
            )
        for name, value in parameters.items():
            if value not in self.search_space[name]:
                raise ValueError(
                    "candidate parameter %r=%r is outside the declared search space"
                    % (name, value)
                )


@dataclass(frozen=True)
class CandidateProposal:
    """A generator hypothesis and the concrete edit it proposes."""

    hypothesis: str
    parameter_updates: JsonDict = field(default_factory=dict)
    expected_effect: JsonDict = field(default_factory=dict)
    source_patch: Optional[str] = None
    metadata: JsonDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hypothesis.strip():
            raise ValueError("candidate hypothesis cannot be empty")
        if not self.parameter_updates and not self.source_patch:
            raise ValueError(
                "a candidate proposal needs parameter_updates or source_patch"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateProposal":
        data = dict(value)
        data["parameter_updates"] = dict(data.get("parameter_updates") or {})
        data["expected_effect"] = dict(data.get("expected_effect") or {})
        data["metadata"] = dict(data.get("metadata") or {})
        return cls(**data)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    """A concrete candidate with a stable content-derived identity."""

    candidate_id: str
    task_id: str
    parent_id: Optional[str]
    generation: int
    parameters: JsonDict
    hypothesis: str
    expected_effect: JsonDict = field(default_factory=dict)
    source_patch: Optional[str] = None
    proposal_metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def seed(cls, task: TaskSpec) -> "Candidate":
        return cls._create(
            task=task,
            parent_id=None,
            generation=0,
            parameters=task.base_parameters,
            hypothesis="Initial verified seed candidate.",
            expected_effect={},
            source_patch=None,
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
        parameters = dict(parent.parameters)
        parameters.update(proposal.parameter_updates)
        task.validate_parameters(parameters)
        return cls._create(
            task=task,
            parent_id=parent.candidate_id,
            generation=generation,
            parameters=parameters,
            hypothesis=proposal.hypothesis,
            expected_effect=proposal.expected_effect,
            source_patch=proposal.source_patch,
            proposal_metadata=proposal.metadata,
        )

    @classmethod
    def _create(
        cls,
        task: TaskSpec,
        parent_id: Optional[str],
        generation: int,
        parameters: Mapping[str, Any],
        hypothesis: str,
        expected_effect: Mapping[str, Any],
        source_patch: Optional[str],
        proposal_metadata: Mapping[str, Any],
    ) -> "Candidate":
        identity = {
            "task_id": task.task_id,
            "parameters": dict(parameters),
            "source_patch": source_patch,
            "patch_parent_id": parent_id if source_patch else None,
        }
        candidate_id = hashlib.sha256(_stable_json(identity).encode("utf-8")).hexdigest()[:16]
        return cls(
            candidate_id=candidate_id,
            task_id=task.task_id,
            parent_id=parent_id,
            generation=generation,
            parameters=dict(parameters),
            hypothesis=hypothesis,
            expected_effect=dict(expected_effect),
            source_patch=source_patch,
            proposal_metadata=dict(proposal_metadata),
        )

    def to_dict(self) -> JsonDict:
        return asdict(self)


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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProfileEvaluation":
        data = dict(value)
        data["metrics"] = dict(data.get("metrics") or {})
        return cls(**data)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass
class CandidateRecord:
    """Accumulated evidence and decision state for one candidate."""

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

    def to_dict(self) -> JsonDict:
        return {
            "candidate": self.candidate.to_dict(),
            "state": self.state,
            "model": self.model.to_dict() if self.model else None,
            "measurement": self.measurement.to_dict() if self.measurement else None,
            "profile": self.profile.to_dict() if self.profile else None,
            "selection_reasons": list(self.selection_reasons),
            "decision_reason": self.decision_reason,
        }


@dataclass(frozen=True)
class SearchSummary:
    """Final optimization outcome and cost counters."""

    task_id: str
    best_candidate_id: str
    best_latency_ms: float
    seed_latency_ms: float
    speedup_over_seed: float
    completed_rounds: int
    generated_candidates: int
    modeled_candidates: int
    measured_candidates: int
    correct_candidates: int
    profile_calls: int
    trust: JsonDict
    output_directory: str

    def to_dict(self) -> JsonDict:
        return asdict(self)
