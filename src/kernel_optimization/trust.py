"""Online trust estimates for model-guided candidate promotion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import CandidateRecord


@dataclass
class TrustTracker:
    """Track absolute error and local optimization-direction agreement."""

    relative_errors: List[float] = field(default_factory=list)
    raw_relative_errors: List[float] = field(default_factory=list)
    direction_correct: int = 0
    direction_total: int = 0

    def update(
        self,
        record: CandidateRecord,
        parent: Optional[CandidateRecord],
    ) -> None:
        if not record.is_measured_correct or record.model is None:
            return
        predicted = record.model.ranking_latency_ms
        measured = record.measurement.latency_ms if record.measurement else None
        if predicted is None or measured is None or measured <= 0:
            return
        self.relative_errors.append(abs(predicted - measured) / measured)
        raw_predicted = record.model.predicted_latency_ms
        if raw_predicted is not None:
            self.raw_relative_errors.append(
                abs(raw_predicted - measured) / measured
            )

        if (
            parent is None
            or not parent.is_measured_correct
            or parent.model is None
            or parent.model.ranking_latency_ms is None
            or parent.measurement is None
            or parent.measurement.latency_ms is None
        ):
            return
        predicted_delta = predicted - parent.model.ranking_latency_ms
        measured_delta = measured - parent.measurement.latency_ms
        if abs(predicted_delta) <= 1e-12 or abs(measured_delta) <= 1e-12:
            return
        self.direction_total += 1
        if (predicted_delta < 0) == (measured_delta < 0):
            self.direction_correct += 1

    @property
    def mean_absolute_relative_error(self) -> Optional[float]:
        if not self.relative_errors:
            return None
        return sum(self.relative_errors) / len(self.relative_errors)

    @property
    def direction_accuracy(self) -> Optional[float]:
        if not self.direction_total:
            return None
        return self.direction_correct / self.direction_total

    @property
    def raw_mean_absolute_relative_error(self) -> Optional[float]:
        if not self.raw_relative_errors:
            return None
        return sum(self.raw_relative_errors) / len(self.raw_relative_errors)

    @property
    def score(self) -> float:
        """Return a conservative zero-to-one model trust score."""

        error = self.mean_absolute_relative_error
        if error is None:
            return 0.0
        absolute_score = max(0.0, 1.0 - min(error, 1.0))
        direction = self.direction_accuracy
        if direction is None:
            return 0.4 * absolute_score
        return max(0.0, min(1.0, 0.7 * direction + 0.3 * absolute_score))

    @property
    def screening_failure_reasons(self) -> List[str]:
        """Explain when model-only pruning is not supported by observations.

        A large raw scale error is useful as an early warning, but it does not
        permanently condemn a model whose ordering later agrees with hardware.
        Conversely, repeatedly choosing the wrong optimization direction is a
        direct reason to stop pruning candidates with the model.
        """

        reasons: List[str] = []
        samples = len(self.relative_errors)
        direction = self.direction_accuracy
        calibrated_error = self.mean_absolute_relative_error
        raw_error = self.raw_mean_absolute_relative_error
        direction_unverified = self.direction_total < 4

        if samples >= 3 and direction_unverified and raw_error is not None and raw_error > 4.0:
            reasons.append("extreme-raw-scale-error-before-direction-validation")
        if (
            samples >= 3
            and calibrated_error is not None
            and calibrated_error > 1.0
            and (direction is None or direction < 0.65)
        ):
            reasons.append("high-ranking-latency-error")
        if self.direction_total >= 4 and direction is not None and direction < 0.55:
            reasons.append("poor-hardware-direction-agreement")
        return reasons

    @property
    def screening_reliable(self) -> bool:
        """Whether the current evidence supports dropping model-ranked candidates."""

        return not self.screening_failure_reasons

    @property
    def screening_mode(self) -> str:
        if self.screening_failure_reasons:
            return "measure-all-fallback"
        if len(self.relative_errors) < 3 or self.direction_total < 2:
            return "evidence-warmup"
        return "adaptive-screening"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measured_samples": len(self.relative_errors),
            "mean_absolute_relative_error": self.mean_absolute_relative_error,
            "ranking_mean_absolute_relative_error": self.mean_absolute_relative_error,
            "raw_mean_absolute_relative_error": self.raw_mean_absolute_relative_error,
            "direction_correct": self.direction_correct,
            "direction_total": self.direction_total,
            "direction_accuracy": self.direction_accuracy,
            "score": self.score,
            "screening_mode": self.screening_mode,
            "screening_reliable": self.screening_reliable,
            "screening_failure_reasons": self.screening_failure_reasons,
        }

    def snapshot(self) -> Dict[str, Any]:
        """Return the lossless state needed to resume online calibration."""

        return {
            "relative_errors": list(self.relative_errors),
            "raw_relative_errors": list(self.raw_relative_errors),
            "direction_correct": self.direction_correct,
            "direction_total": self.direction_total,
        }

    @classmethod
    def from_snapshot(cls, value: Optional[Dict[str, Any]]) -> "TrustTracker":
        data = dict(value or {})
        return cls(
            relative_errors=[float(item) for item in data.get("relative_errors", [])],
            raw_relative_errors=[
                float(item) for item in data.get("raw_relative_errors", [])
            ],
            direction_correct=int(data.get("direction_correct", 0)),
            direction_total=int(data.get("direction_total", 0)),
        )
