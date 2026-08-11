"""Deterministic imperfect performance backend for local tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Sequence

from ..schema import (
    Candidate,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    TaskSpec,
)


class MockPerformanceBackend:
    """Expose a hidden objective and a deliberately biased analytical model."""

    def __init__(self, configuration: Mapping[str, Any]) -> None:
        self.configuration = dict(configuration)
        self.model_calls = 0
        self.measure_calls = 0
        self.profile_calls = 0

    def model(self, task: TaskSpec, candidate: Candidate) -> ModelEvaluation:
        self.model_calls += 1
        if self._matches_any(
            candidate.parameters,
            self.configuration.get("model_invalid_combinations", []),
        ):
            return ModelEvaluation(
                valid=False,
                diagnostics=["Mock model rejected an explicitly invalid combination."],
            )
        measured = self._objective(task, candidate.parameters)
        distortion = float(self.configuration.get("model_bias", -0.04))
        errors = dict(self.configuration.get("model_error_by_parameter") or {})
        for name, by_value in errors.items():
            distortion += float(dict(by_value).get(str(candidate.parameters[name]), 0.0))
        predicted = max(measured * (1.0 + distortion), 1e-6)
        absolute_distortion = abs(distortion)
        confidence = (
            "high"
            if absolute_distortion <= 0.08
            else "medium"
            if absolute_distortion <= 0.20
            else "low"
        )
        return ModelEvaluation(
            valid=True,
            predicted_latency_ms=predicted,
            bottleneck=self._bottleneck(task, candidate.parameters),
            confidence=confidence,
            metrics={
                "mock_distortion": distortion,
                "estimated_occupancy": self._occupancy(task, candidate.parameters),
            },
        )

    def measure(self, task: TaskSpec, candidate: Candidate) -> Measurement:
        self.measure_calls += 1
        if self._matches_any(
            candidate.parameters,
            self.configuration.get("incorrect_combinations", []),
        ):
            return Measurement(
                correct=False,
                error="Mock correctness failure for an explicitly invalid combination.",
            )
        latency = self._objective(task, candidate.parameters)
        return Measurement(
            correct=True,
            latency_ms=latency,
            samples_ms=[latency * 1.01, latency * 0.995, latency, latency * 1.005],
            metrics={"measurement_source": "deterministic-mock"},
        )

    def profile(self, task: TaskSpec, candidate: Candidate) -> ProfileEvaluation:
        self.profile_calls += 1
        bottleneck = self._bottleneck(task, candidate.parameters)
        occupancy = self._occupancy(task, candidate.parameters)
        return ProfileEvaluation(
            bottleneck=bottleneck,
            metrics={
                "mock_ncu": True,
                "achieved_occupancy": occupancy,
                "compute_sol_pct": 70.0 if bottleneck == "tensor-core" else 45.0,
                "memory_sol_pct": 72.0 if bottleneck == "memory" else 40.0,
                "register_pressure_pct": 90.0
                if bottleneck == "register-pressure"
                else 55.0,
            },
        )

    def _objective(self, task: TaskSpec, parameters: Mapping[str, Any]) -> float:
        minimum = float(self.configuration.get("minimum_latency_ms", 0.20))
        target = dict(self.configuration.get("target_parameters") or task.base_parameters)
        weights = dict(self.configuration.get("parameter_weights") or {})
        penalty = 0.0
        for name, values in task.search_space.items():
            left = values.index(parameters[name])
            right = values.index(target[name])
            scale = max(len(values) - 1, 1)
            distance = abs(left - right) / scale
            penalty += float(weights.get(name, 0.12)) * distance * distance

        interactions = self.configuration.get("interaction_penalties") or []
        for interaction in interactions:
            when = dict(interaction.get("when") or {})
            if all(parameters.get(name) == value for name, value in when.items()):
                penalty += float(interaction.get("penalty_ms", 0.0))

        identity = json.dumps(dict(parameters), sort_keys=True).encode("utf-8")
        jitter = int(hashlib.sha256(identity).hexdigest()[:4], 16) / 0xFFFF
        jitter *= float(self.configuration.get("deterministic_jitter_ms", 0.0005))
        return minimum + penalty + jitter

    def _bottleneck(self, task: TaskSpec, parameters: Mapping[str, Any]) -> str:
        target = dict(self.configuration.get("target_parameters") or task.base_parameters)
        stages = parameters.get("num_stages")
        target_stages = target.get("num_stages")
        if isinstance(stages, (int, float)) and isinstance(target_stages, (int, float)):
            if stages > target_stages:
                return "register-pressure"
        block_m = parameters.get("block_m")
        target_m = target.get("block_m")
        if isinstance(block_m, (int, float)) and isinstance(target_m, (int, float)):
            if block_m < target_m:
                return "memory"
        return "tensor-core"

    def _occupancy(self, task: TaskSpec, parameters: Mapping[str, Any]) -> float:
        target = dict(self.configuration.get("target_parameters") or task.base_parameters)
        penalty = 0.0
        for name in ("num_stages", "threads"):
            if name not in parameters or name not in target:
                continue
            values = task.search_space[name]
            penalty += abs(values.index(parameters[name]) - values.index(target[name])) * 0.12
        return max(0.25, min(1.0, 0.82 - penalty))

    @staticmethod
    def _matches_any(
        parameters: Mapping[str, Any], combinations: Sequence[Mapping[str, Any]]
    ) -> bool:
        return any(
            all(parameters.get(name) == value for name, value in combination.items())
            for combination in combinations
        )

