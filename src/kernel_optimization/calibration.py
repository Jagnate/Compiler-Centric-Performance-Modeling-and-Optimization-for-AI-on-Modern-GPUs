"""Search-local latency calibration without modifying the analytical model."""

from __future__ import annotations

from dataclasses import dataclass, replace
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .schema import CandidateRecord, ModelEvaluation, TaskSpec


@dataclass(frozen=True)
class CalibrationObservation:
    candidate_id: str
    key: str
    coarse_key: str
    raw_latency_ms: float
    measured_latency_ms: float
    ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "key": self.key,
            "coarse_key": self.coarse_key,
            "raw_latency_ms": self.raw_latency_ms,
            "measured_latency_ms": self.measured_latency_ms,
            "ratio": self.ratio,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationObservation":
        return cls(
            candidate_id=str(value["candidate_id"]),
            key=str(value["key"]),
            coarse_key=str(value["coarse_key"]),
            raw_latency_ms=float(value["raw_latency_ms"]),
            measured_latency_ms=float(value["measured_latency_ms"]),
            ratio=float(value["ratio"]),
        )


class LatencyCalibrator:
    """Fit robust scale factors within architecture and resource regimes."""

    def __init__(self, min_samples: int = 2) -> None:
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")
        self.min_samples = min_samples
        self.observations: List[CalibrationObservation] = []
        self._candidate_ids = set()

    def apply(self, task: TaskSpec, model: ModelEvaluation) -> ModelEvaluation:
        if not model.valid or model.predicted_latency_ms is None:
            return model
        key, coarse_key = self._keys(task, model)
        exact = [item.ratio for item in self.observations if item.key == key]
        coarse = [
            item.ratio for item in self.observations if item.coarse_key == coarse_key
        ]
        if len(exact) >= self.min_samples:
            ratios = exact
            source = "exact-regime"
        elif len(coarse) >= self.min_samples:
            ratios = coarse
            source = "coarse-regime"
        else:
            return replace(
                model,
                calibrated_latency_ms=None,
                calibration={
                    "applied": False,
                    "reason": "insufficient-regime-samples",
                    "required_samples": self.min_samples,
                    "exact_samples": len(exact),
                    "coarse_samples": len(coarse),
                    "key": key,
                    "coarse_key": coarse_key,
                },
            )

        median_scale = statistics.median(ratios)
        # Regime calibration exists specifically to repair analytical-model
        # scale mismatch. Clipping that correction to [0.25, 4] left models
        # that were off by thousands of times effectively uncalibrated.
        scale = max(1e-9, min(1e9, median_scale))
        deviations = [abs(item - scale) for item in ratios]
        mad = statistics.median(deviations) if deviations else 0.0
        relative_mad = mad / scale if scale > 0 else 1.0
        calibrated = float(model.predicted_latency_ms) * scale
        confidence = model.confidence
        extreme_scale = median_scale < 0.25 or median_scale > 4.0
        if extreme_scale or relative_mad > 0.25:
            confidence = "low"
        elif len(ratios) < max(4, self.min_samples + 1) and confidence == "high":
            confidence = "medium"
        return replace(
            model,
            calibrated_latency_ms=calibrated,
            confidence=confidence,
            calibration={
                "applied": True,
                "source": source,
                "scale": scale,
                "median_scale": median_scale,
                "extreme_scale": extreme_scale,
                "median_absolute_deviation": mad,
                "relative_median_absolute_deviation": relative_mad,
                "samples": len(ratios),
                "key": key,
                "coarse_key": coarse_key,
                "raw_predicted_latency_ms": model.predicted_latency_ms,
                "calibrated_predicted_latency_ms": calibrated,
            },
        )

    def observe(self, task: TaskSpec, record: CandidateRecord) -> None:
        if (
            record.candidate.candidate_id in self._candidate_ids
            or not record.is_measured_correct
            or record.model is None
            or record.model.predicted_latency_ms is None
            or record.measurement is None
            or record.measurement.latency_ms is None
        ):
            return
        raw = float(record.model.predicted_latency_ms)
        measured = float(record.measurement.latency_ms)
        if raw <= 0 or measured <= 0:
            return
        key, coarse_key = self._keys(task, record.model)
        self.observations.append(
            CalibrationObservation(
                candidate_id=record.candidate.candidate_id,
                key=key,
                coarse_key=coarse_key,
                raw_latency_ms=raw,
                measured_latency_ms=measured,
                ratio=measured / raw,
            )
        )
        self._candidate_ids.add(record.candidate.candidate_id)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "min_samples": self.min_samples,
            "observations": [item.to_dict() for item in self.observations],
        }

    @classmethod
    def from_snapshot(
        cls, value: Optional[Mapping[str, Any]], default_min_samples: int
    ) -> "LatencyCalibrator":
        data = dict(value or {})
        tracker = cls(int(data.get("min_samples", default_min_samples)))
        for raw in data.get("observations", []):
            observation = CalibrationObservation.from_dict(raw)
            tracker.observations.append(observation)
            tracker._candidate_ids.add(observation.candidate_id)
        return tracker

    def to_dict(self) -> Dict[str, Any]:
        groups: Dict[str, List[float]] = {}
        for item in self.observations:
            groups.setdefault(item.coarse_key, []).append(item.ratio)
        return {
            "policy": "search-local-regime-median-scale",
            "min_samples": self.min_samples,
            "observation_count": len(self.observations),
            "groups": {
                key: {
                    "samples": len(ratios),
                    "median_scale": statistics.median(ratios),
                    "min_scale": min(ratios),
                    "max_scale": max(ratios),
                }
                for key, ratios in sorted(groups.items())
            },
        }

    @classmethod
    def _keys(cls, task: TaskSpec, model: ModelEvaluation) -> Tuple[str, str]:
        architecture = str(task.target.get("architecture", "unknown"))
        family = str(
            task.metadata.get("kernel_family")
            or task.metadata.get("workload_family")
            or task.entrypoint
        )
        bottleneck = str(model.bottleneck or "unknown")
        registers = cls._metric(
            model.metrics,
            "registers_per_thread",
            "reg_footprint",
            "register_footprint",
        )
        shared = cls._metric(
            model.metrics,
            "shared_memory_per_block",
            "smem_footprint",
            "shared_bytes",
        )
        register_regime = cls._bucket(registers, (64.0, 128.0, 192.0), "regs")
        shared_regime = cls._bucket(shared, (32768.0, 65536.0, 98304.0), "smem")
        coarse = "|".join((architecture, family, bottleneck))
        exact = "|".join((coarse, register_regime, shared_regime))
        return exact, coarse

    @staticmethod
    def _metric(metrics: Mapping[str, Any], *names: str) -> Optional[float]:
        for name in names:
            value = metrics.get(name)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @staticmethod
    def _bucket(
        value: Optional[float], boundaries: Sequence[float], prefix: str
    ) -> str:
        if value is None:
            return prefix + "-unknown"
        for boundary in boundaries:
            if value <= boundary:
                return "%s-le-%d" % (prefix, int(boundary))
        return "%s-gt-%d" % (prefix, int(boundaries[-1]))
