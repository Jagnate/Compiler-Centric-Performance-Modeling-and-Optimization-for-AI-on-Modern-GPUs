"""Deterministic failure and GPU bottleneck diagnosis for search feedback."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import (
    BottleneckDiagnosis,
    CandidateRecord,
    FailureEvidence,
    Measurement,
    ModelEvaluation,
)


_UTILIZATION_ALIASES = {
    "dram": ("ddr_util", "dram_util", "dram_throughput", "dram__throughput"),
    "l2": ("l2_util", "lts_util", "l2_throughput", "lts__throughput"),
    "shared": (
        "smem_util",
        "shared_memory_util",
        "shared_throughput",
        "l1tex__throughput",
    ),
    "tensor": (
        "tensor_util",
        "tensor_core_util",
        "tensor_active",
        "sm__pipe_tensor",
    ),
    "cuda": (
        "cuda_util",
        "cuda_core_util",
        "alu_util",
        "sm__pipe_fma",
        "sm__pipe_fp",
    ),
    "sfu": ("sfu_util", "sm__pipe_sfu"),
}


class FailureClassifier:
    """Map raw static/compiler/runtime failures to stable repair categories."""

    @staticmethod
    def static(error: Exception) -> FailureEvidence:
        message = str(error)
        lowered = message.lower()
        if "syntax" in lowered or "valid python" in lowered:
            category = "syntax"
        elif "forbidden" in lowered or "null byte" in lowered:
            category = "policy-violation"
        elif "entrypoint" in lowered or "required fragment" in lowered:
            category = "interface-contract"
        elif "byte limit" in lowered or "bytes" in lowered:
            category = "source-size"
        else:
            category = "static-validation"
        return FailureEvidence(
            stage="static",
            category=category,
            message=message,
            diagnostics=[message],
            retryable=category != "policy-violation",
        )

    @staticmethod
    def model(model: ModelEvaluation) -> FailureEvidence:
        diagnostics = list(model.diagnostics)
        message = diagnostics[0] if diagnostics else "Compiler or model rejected candidate."
        lowered = " ".join(diagnostics).lower()
        if "layout infer" in lowered or "layout" in lowered and "conflict" in lowered:
            category = "layout-inference"
        elif "out of memory" in lowered or "resource" in lowered:
            category = "compile-resource"
        elif "timeout" in lowered:
            category = "compile-timeout"
        elif "internalerror" in lowered or "check failed" in lowered:
            category = "compiler-internal"
        else:
            category = "compile-or-lowering"
        return FailureEvidence(
            stage="model",
            category=category,
            message=message,
            diagnostics=diagnostics,
            details={"bottleneck": model.bottleneck},
        )

    @staticmethod
    def measurement(measurement: Measurement) -> FailureEvidence:
        message = measurement.error or "Correctness or runtime validation failed."
        lowered = message.lower()
        if any(word in lowered for word in ("mismatch", "incorrect", "correctness", "tolerance")):
            category = "correctness"
        elif "timeout" in lowered:
            category = "runtime-timeout"
        elif any(word in lowered for word in ("compile", "nvcc", "ptx", "lower")):
            category = "runtime-compile"
        elif any(word in lowered for word in ("illegal", "misaligned", "launch", "cuda")):
            category = "runtime-cuda"
        else:
            category = "runtime-or-correctness"
        return FailureEvidence(
            stage="measure",
            category=category,
            message=message,
            diagnostics=[message],
            details=dict(measurement.metrics),
        )


class BottleneckAnalyzer:
    """Interpret common TileSight and NCU metrics without kernel-specific rules."""

    def diagnose(
        self,
        record: CandidateRecord,
        parent: Optional[CandidateRecord] = None,
    ) -> BottleneckDiagnosis:
        if record.failure is not None:
            return self._failure_diagnosis(record.failure)

        profile_metrics = (
            record.profile.metrics
            if record.profile is not None and record.profile.valid
            else {}
        )
        model_metrics = record.model.metrics if record.model is not None else {}
        primary = profile_metrics or model_metrics
        source = "ncu" if profile_metrics else "tilesight"
        flat = _flatten_metrics(primary)
        utilizations = {
            name: _percentage(_first_metric(flat, aliases))
            for name, aliases in _UTILIZATION_ALIASES.items()
        }
        utilizations = {
            name: value for name, value in utilizations.items() if value is not None
        }

        occupancy = _percentage(
            _first_metric(
                flat,
                (
                    "achieved_occupancy",
                    "occupancy",
                    "sm__warps_active",
                    "theoretical_occupancy",
                ),
            )
        )
        registers = _first_metric(
            flat,
            ("registers_per_thread", "reg_footprint", "registers", "launch__registers"),
        )
        shared_bytes = _first_metric(
            flat,
            (
                "shared_memory_per_block",
                "smem_footprint",
                "shared_bytes",
                "launch__shared_mem_per_block",
            ),
        )
        spill_bytes = _sum_metrics(
            flat,
            (
                "spill_load_bytes",
                "spill_store_bytes",
                "local_load_bytes",
                "local_store_bytes",
            ),
        )
        bank_conflict = _first_metric(
            flat,
            (
                "shared_bank_conflict",
                "bank_conflict",
                "shared_transactions_per_request",
                "l1tex__data_bank_conflicts",
            ),
        )
        waves_per_sm = _first_metric(
            flat, ("waves_per_sm", "wave_count", "launch_waves_per_sm")
        )

        normalized = {
            "utilization_percent": utilizations,
            "achieved_occupancy_percent": occupancy,
            "registers_per_thread": registers,
            "shared_memory_per_block_bytes": shared_bytes,
            "spill_bytes": spill_bytes,
            "bank_conflict_indicator": bank_conflict,
            "waves_per_sm": waves_per_sm,
        }
        relative_error = self._relative_error(record)
        if relative_error is not None:
            normalized["model_measurement_relative_error"] = relative_error

        category, factors = self._category(
            utilizations,
            occupancy,
            registers,
            shared_bytes,
            spill_bytes,
            bank_conflict,
            waves_per_sm,
            record,
        )
        evidence = self._evidence(source, normalized)
        if parent is not None:
            delta = self._measured_delta(record, parent)
            if delta is not None:
                evidence.append(
                    {
                        "source": "cuda-events",
                        "metric": "latency_change_vs_parent",
                        "value": delta,
                        "unit": "fraction",
                        "interpretation": (
                            "negative is faster; positive is slower"
                        ),
                    }
                )
        confidence = "high" if source == "ncu" else (
            "medium" if record.measurement is not None else "low"
        )
        recommendations = _RECOMMENDATIONS.get(
            category, _RECOMMENDATIONS["unknown"]
        )
        return BottleneckDiagnosis(
            category=category,
            confidence=confidence,
            summary=self._summary(category, factors, source),
            limiting_factors=factors,
            evidence=evidence,
            recommendations=list(recommendations),
            source="deterministic-" + source,
            metrics=normalized,
        )

    @staticmethod
    def _failure_diagnosis(failure: FailureEvidence) -> BottleneckDiagnosis:
        recommendation = {
            "syntax": "Repair Python syntax while preserving the complete entrypoint contract.",
            "interface-contract": "Restore the required entrypoint and external interface exactly.",
            "layout-inference": "Make fragment layouts and parallel mappings mutually compatible.",
            "correctness": "Preserve indexing, masking, accumulation dtype, and output semantics before tuning.",
            "runtime-cuda": "Reduce invalid launch or memory behavior and re-run correctness first.",
        }.get(
            failure.category,
            "Use the archived diagnostic to make the smallest semantics-preserving repair.",
        )
        return BottleneckDiagnosis(
            category="failure:" + failure.category,
            confidence="high",
            summary="Candidate failed during %s: %s" % (failure.stage, failure.message),
            limiting_factors=[failure.category],
            evidence=[
                {
                    "source": failure.stage,
                    "metric": "failure",
                    "value": failure.message,
                    "interpretation": "candidate is not eligible for promotion",
                }
            ],
            recommendations=[recommendation],
            source="deterministic-failure-classifier",
            metrics=dict(failure.details),
        )

    @staticmethod
    def _category(
        utilizations: Mapping[str, float],
        occupancy: Optional[float],
        registers: Optional[float],
        shared_bytes: Optional[float],
        spill_bytes: float,
        bank_conflict: Optional[float],
        waves_per_sm: Optional[float],
        record: CandidateRecord,
    ) -> Tuple[str, List[str]]:
        factors: List[str] = []
        if spill_bytes > 0:
            factors.append("nonzero local-memory spill traffic")
            return "register-spill", factors
        if bank_conflict is not None and bank_conflict > 1.25:
            factors.append("shared-memory transaction amplification or bank conflicts")
            return "shared-memory-bank-conflict", factors
        if waves_per_sm is not None and waves_per_sm < 1.0:
            factors.append("less than one launch wave per SM")
            return "launch-underfill", factors
        resource_pressure = (
            (registers is not None and registers >= 128)
            or (shared_bytes is not None and shared_bytes >= 65536)
        )
        if occupancy is not None and occupancy < 35.0 and resource_pressure:
            if registers is not None and registers >= 128:
                factors.append("high registers per thread")
            if shared_bytes is not None and shared_bytes >= 65536:
                factors.append("high shared memory per block")
            factors.append("low achieved occupancy")
            return "occupancy-limited", factors

        if utilizations:
            resource, peak = max(utilizations.items(), key=lambda item: item[1])
            factors.append("%s has the highest observed utilization" % resource)
            if peak < 30.0:
                factors.append("all reported execution resources are weakly utilized")
                return "underutilized", factors
            mapping = {
                "dram": "memory-dram",
                "l2": "memory-l2",
                "shared": "memory-shared",
                "tensor": "compute-tensor",
                "cuda": "compute-cuda",
                "sfu": "compute-sfu",
            }
            return mapping.get(resource, "unknown"), factors
        if record.model is not None and record.model.bottleneck != "unknown":
            factors.append("only the analytical bottleneck label is available")
            return "model:" + record.model.bottleneck, factors
        return "unknown", ["insufficient normalized performance counters"]

    @staticmethod
    def _evidence(source: str, normalized: Mapping[str, Any]) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for name, value in normalized.items():
            if value is None:
                continue
            if name == "utilization_percent":
                for resource, percent in value.items():
                    evidence.append(
                        {
                            "source": source,
                            "metric": resource + "_utilization",
                            "value": percent,
                            "unit": "percent",
                            "interpretation": "normalized utilization or percent-of-peak",
                        }
                    )
            else:
                evidence.append(
                    {
                        "source": source,
                        "metric": name,
                        "value": value,
                        "interpretation": "normalized diagnostic input",
                    }
                )
        return evidence

    @staticmethod
    def _relative_error(record: CandidateRecord) -> Optional[float]:
        if (
            record.model is None
            or record.model.predicted_latency_ms is None
            or record.measurement is None
            or record.measurement.latency_ms is None
            or record.measurement.latency_ms <= 0
        ):
            return None
        return abs(
            record.model.predicted_latency_ms - record.measurement.latency_ms
        ) / record.measurement.latency_ms

    @staticmethod
    def _measured_delta(
        record: CandidateRecord, parent: CandidateRecord
    ) -> Optional[float]:
        if not record.is_measured_correct or not parent.is_measured_correct:
            return None
        current = float(record.measurement.latency_ms)
        previous = float(parent.measurement.latency_ms)
        return (current - previous) / previous if previous > 0 else None

    @staticmethod
    def _summary(category: str, factors: Sequence[str], source: str) -> str:
        detail = "; ".join(factors) if factors else "no dominant factor"
        return "%s diagnosis from %s evidence: %s." % (category, source, detail)


_RECOMMENDATIONS = {
    "register-spill": [
        "Reduce live accumulator or temporary state and verify that spill traffic disappears.",
        "Trade pipeline depth or tile size for lower register pressure before increasing parallelism.",
    ],
    "shared-memory-bank-conflict": [
        "Change shared-memory layout, padding, swizzle, or vector mapping to reduce transaction amplification.",
    ],
    "launch-underfill": [
        "Increase independent CTA count or reduce CTA tile size while retaining useful work per block.",
    ],
    "occupancy-limited": [
        "Reduce the limiting register or shared-memory footprint and re-check active CTAs per SM.",
        "Do not increase stages unless the extra overlap outweighs the occupancy loss.",
    ],
    "memory-dram": [
        "Reduce DRAM bytes through reuse, fusion, coalescing, vectorization, or fewer redundant loads.",
    ],
    "memory-l2": [
        "Improve tile reuse and access locality or reduce repeated L2 transactions per output tile.",
    ],
    "memory-shared": [
        "Reduce shared-memory traffic or improve vectorized and conflict-free shared accesses.",
    ],
    "compute-tensor": [
        "Increase useful Tensor Core work per scheduling overhead without raising resource pressure excessively.",
    ],
    "compute-cuda": [
        "Reduce scalar instruction count or expose more instruction-level and thread-level parallelism.",
    ],
    "compute-sfu": [
        "Reduce or overlap special-function operations while preserving numerical behavior.",
    ],
    "underutilized": [
        "Inspect dependency chains, launch underfill, synchronization, and latency hiding before tuning peak bandwidth.",
    ],
    "unknown": [
        "Collect a correctness-passing timing and milestone profile before making a narrow bottleneck claim.",
    ],
}


def _flatten_metrics(value: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    result: Dict[str, float] = {}
    for raw_name, item in value.items():
        name = _normalize_name(str(raw_name))
        path = prefix + "_" + name if prefix else name
        if isinstance(item, Mapping):
            result.update(_flatten_metrics(item, path))
        elif not isinstance(item, bool) and isinstance(item, (int, float)):
            result[path] = float(item)
            result.setdefault(name, float(item))
    return result


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _first_metric(
    metrics: Mapping[str, float], aliases: Iterable[str]
) -> Optional[float]:
    normalized = [_normalize_name(alias) for alias in aliases]
    for alias in normalized:
        if alias in metrics:
            return metrics[alias]
    for alias in normalized:
        for name, value in metrics.items():
            if name.endswith(alias) or alias in name:
                return value
    return None


def _sum_metrics(metrics: Mapping[str, float], aliases: Iterable[str]) -> float:
    values = []
    for alias in aliases:
        value = _first_metric(metrics, (alias,))
        if value is not None:
            values.append(value)
    return sum(values)


def _percentage(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return value * 100.0
    return value

