"""Deterministic, bounded views of archived evidence for hosted-model prompts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


COMPRESSION_VERSION = "key-metrics-v1"
NO_COMPRESSION_POLICY = "none"
METADATA_COMPRESSION_POLICIES = (COMPRESSION_VERSION, NO_COMPRESSION_POLICY)
DEFAULT_HISTORY_LIMIT = 8
DEFAULT_LESSON_LIMIT = 8

_DROPPED_METRIC_KEYS = {
    "captured_passes",
    "compiled_identity",
    "environment",
    "ncu_csv_path",
    "raw_metrics",
    "raw_units",
    "resource_provenance",
    "target",
}
_IMPORTANT_METRIC_NAMES = {
    "achieved_occupancy",
    "achieved_occupancy_percent",
    "bank_conflict",
    "bank_conflict_indicator",
    "cuda_util",
    "cuda_utilization",
    "ddr_util",
    "dram_read_bytes",
    "dram_utilization",
    "dram_write_bytes",
    "dram_write_bytes_per_cta",
    "grid_shape",
    "l1_hit_rate",
    "l2_hit_rate",
    "l2_read_bytes",
    "l2_util",
    "l2_utilization",
    "l2_write_bytes",
    "l2_write_bytes_per_cta",
    "latency_ms",
    "launch_waves_per_sm",
    "logical_global_write_bytes_per_cta",
    "model_measurement_relative_error",
    "primary_case_id",
    "registers_per_thread",
    "sfu_util",
    "sfu_utilization",
    "shared_bank_conflict",
    "shared_memory_per_block",
    "shared_memory_per_block_bytes",
    "shared_transactions_per_request",
    "smem_footprint",
    "smem_util",
    "spill_bytes",
    "spill_load_bytes",
    "spill_load_bytes_per_cta",
    "spill_store_bytes",
    "spill_store_bytes_per_cta",
    "tensor_util",
    "tensor_utilization",
    "threads_per_block",
    "tiles_per_sm",
    "waves_per_sm",
}
_METRIC_HINTS = (
    "bank",
    "bytes",
    "cuda",
    "ddr",
    "dram",
    "duration",
    "hit_rate",
    "latency",
    "l2",
    "occup",
    "register",
    "sfu",
    "shared",
    "smem",
    "spill",
    "tensor",
    "throughput",
    "util",
    "wave",
)


@dataclass(frozen=True)
class CompressedPromptContext:
    """Prompt-ready context plus lossless provenance of what was summarized."""

    evidence: Dict[str, Any]
    history: list[Dict[str, Any]]
    provenance: Dict[str, Any]


class PromptBudgetError(ValueError):
    """Raised before an API call when a compressed prompt remains too large."""


def compress_prompt_context(
    evidence: Optional[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    *,
    policy: str = COMPRESSION_VERSION,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
    lesson_limit: int = DEFAULT_LESSON_LIMIT,
) -> CompressedPromptContext:
    """Build a bounded model-facing view without mutating archived evidence."""

    if policy not in METADATA_COMPRESSION_POLICIES:
        raise ValueError(
            "metadata compression policy must be one of: %s"
            % ", ".join(METADATA_COMPRESSION_POLICIES)
        )
    raw_evidence = dict(evidence or {})
    raw_history = [dict(item) for item in history]
    if policy == NO_COMPRESSION_POLICY:
        compressed_evidence = raw_evidence
        compressed_history = raw_history
    else:
        compressed_evidence = {
            "observed": _compact_observed(raw_evidence.get("observed")),
            "predicted": _compact_model(raw_evidence.get("predicted")),
            "tir": _compact_tir(raw_evidence.get("tir")),
            "model_trust": _compact_trust(raw_evidence.get("model_trust")),
            "calibration": _compact_calibration(raw_evidence.get("calibration")),
            "shared_memory": _compact_lessons(
                raw_evidence.get("shared_memory"), lesson_limit
            ),
        }
        compressed_history = [
            _compact_history_record(item) for item in raw_history[-history_limit:]
        ]
    compressed_lessons = compressed_evidence.get("shared_memory") or []
    provenance = {
        "compression_version": policy,
        "raw_evidence_sha256": _digest(raw_evidence),
        "raw_history_sha256": _digest(raw_history),
        "history_records_available": len(raw_history),
        "history_records_included": len(compressed_history),
        "shared_lessons_available": len(raw_evidence.get("shared_memory") or []),
        "shared_lessons_included": len(compressed_lessons),
        "raw_context_characters": _json_size(raw_evidence) + _json_size(raw_history),
        "compressed_context_characters": (
            _json_size(compressed_evidence) + _json_size(compressed_history)
        ),
        "note": _compression_note(policy),
    }
    return CompressedPromptContext(
        evidence=compressed_evidence,
        history=compressed_history,
        provenance=provenance,
    )


def _compression_note(policy: str) -> str:
    if policy == NO_COMPRESSION_POLICY:
        return (
            "Compression is disabled for this ablation request. Full accumulated "
            "prior-candidate metadata is included; duplicate copies of the current "
            "candidate and its supporting evidence are represented by references."
        )
    return (
        "Full-fidelity evidence remains in run artifacts; this request contains "
        "a deterministic key-metric summary."
    )


def compact_failure_context(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Preserve actionable failure fields while bounding compiler diagnostics."""

    data = dict(value or {})
    if set(data) == {"failure"} and isinstance(data.get("failure"), Mapping):
        return {"failure": compact_failure_context(data["failure"])}
    compact: Dict[str, Any] = {}
    for key in ("stage", "category", "retryable", "error_type"):
        if key in data:
            compact[key] = _bounded_value(data[key], depth=1)
    if data.get("message") is not None:
        compact["message"] = _bounded_text(data["message"], 2400)
    diagnostics = data.get("diagnostics") or []
    if isinstance(diagnostics, Sequence) and not isinstance(
        diagnostics, (str, bytes)
    ):
        compact["diagnostics"] = [
            _bounded_text(item, 1200) for item in list(diagnostics)[:5]
        ]
    details = data.get("details")
    if isinstance(details, Mapping):
        compact["details"] = _compact_metrics(details, limit=24)
    for key, item in data.items():
        if key in compact or key in {"message", "diagnostics", "details"}:
            continue
        if key == "failure" and isinstance(item, Mapping):
            compact[key] = compact_failure_context(item)
        elif key == "diagnosis" and isinstance(item, Mapping):
            compact[key] = _compact_diagnosis(item)
        elif _is_small_scalar(item):
            compact[key] = item
    return compact


def prompt_size_metadata(
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: Optional[int],
) -> Dict[str, Any]:
    """Estimate request size without adding a tokenizer dependency.

    Three UTF-8 bytes per token is deliberately conservative for English JSON
    containing Python and compiler identifiers. Provider tokenization remains the
    authoritative count and is archived after a successful request.
    """

    system_bytes = len(system_prompt.encode("utf-8"))
    user_bytes = len(user_prompt.encode("utf-8"))
    input_bytes = system_bytes + user_bytes
    estimated_input_tokens = int(math.ceil(input_bytes / 3.0))
    output_tokens = int(max_output_tokens or 0)
    return {
        "estimator": "ceil(utf8_bytes/3)",
        "system_characters": len(system_prompt),
        "user_characters": len(user_prompt),
        "input_bytes": input_bytes,
        "estimated_input_tokens": estimated_input_tokens,
        "max_output_tokens": output_tokens,
        "estimated_reserved_tokens": estimated_input_tokens + output_tokens,
    }


def fit_compressed_prompt_to_budget(
    prompt: str,
    *,
    system_prompt: str,
    max_output_tokens: Optional[int],
    target_input_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """Trim bounded context arrays until a JSON prompt fits its input target.

    The current parent source, task contract, current evidence, and response schema
    are never removed. Only the oldest compact history records and the lowest
    priority compact lessons are eligible for trimming.
    """

    if target_input_tokens <= 0:
        raise ValueError("target_input_tokens must be positive")
    try:
        request = json.loads(prompt)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("compressed prompt must be a JSON object") from error
    if not isinstance(request, dict):
        raise ValueError("compressed prompt must be a JSON object")

    history = request.get("recent_history")
    lessons = request.get("shared_evidence_memory")
    if not isinstance(history, list):
        history = []
        request["recent_history"] = history
    if not isinstance(lessons, list):
        lessons = []
        request["shared_evidence_memory"] = lessons
    provenance = request.get("context_provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        request["context_provenance"] = provenance

    initial_history = len(history)
    initial_lessons = len(lessons)
    dropped_history = 0
    dropped_lessons = 0
    provenance.update(
        {
            "input_token_target": target_input_tokens,
            "budget_policy": (
                "preserve-current-context;drop-oldest-history-and-"
                "lowest-priority-lessons"
            ),
            "budget_trimmed": False,
            "history_records_before_budget": initial_history,
            "shared_lessons_before_budget": initial_lessons,
        }
    )

    size: Dict[str, Any] = {}
    while True:
        provenance["history_records_included"] = len(history)
        provenance["shared_lessons_included"] = len(lessons)
        provenance["history_records_dropped_for_budget"] = dropped_history
        provenance["shared_lessons_dropped_for_budget"] = dropped_lessons
        provenance["budget_trimmed"] = bool(dropped_history or dropped_lessons)
        rendered = json.dumps(request, indent=2, sort_keys=True)
        size = prompt_size_metadata(system_prompt, rendered, max_output_tokens)
        provenance["final_estimated_input_tokens"] = int(
            size["estimated_input_tokens"]
        )
        provenance["input_target_satisfied"] = (
            int(size["estimated_input_tokens"]) <= target_input_tokens
        )
        rendered = json.dumps(request, indent=2, sort_keys=True)
        size = prompt_size_metadata(system_prompt, rendered, max_output_tokens)
        if int(size["estimated_input_tokens"]) <= target_input_tokens:
            break

        history_size = _json_size(history[0]) if len(history) > 1 else -1
        lesson_size = _json_size(lessons[-1]) if len(lessons) > 1 else -1
        if history_size < 0 and lesson_size < 0:
            break
        if history_size >= lesson_size and len(history) > 1:
            history.pop(0)
            dropped_history += 1
        elif len(lessons) > 1:
            lessons.pop()
            dropped_lessons += 1

    provenance["final_estimated_input_tokens"] = int(
        size["estimated_input_tokens"]
    )
    provenance["input_target_satisfied"] = (
        int(size["estimated_input_tokens"]) <= target_input_tokens
    )
    rendered = json.dumps(request, indent=2, sort_keys=True)
    final_size = prompt_size_metadata(system_prompt, rendered, max_output_tokens)
    metadata = {
        "target_input_tokens": target_input_tokens,
        "initial_history_records": initial_history,
        "included_history_records": len(history),
        "dropped_history_records": dropped_history,
        "initial_shared_lessons": initial_lessons,
        "included_shared_lessons": len(lessons),
        "dropped_shared_lessons": dropped_lessons,
        "trimmed": bool(dropped_history or dropped_lessons),
        "target_satisfied": int(final_size["estimated_input_tokens"])
        <= target_input_tokens,
        "final_estimated_input_tokens": int(final_size["estimated_input_tokens"]),
    }
    return rendered, metadata


def enforce_prompt_budget(metadata: Mapping[str, Any], max_input_tokens: int) -> None:
    """Fail before network access when local estimation exceeds the configured cap."""

    estimated = int(metadata.get("estimated_input_tokens", 0))
    if estimated <= max_input_tokens:
        return
    raise PromptBudgetError(
        "compressed API prompt is estimated at %d input tokens, above the local "
        "--api-max-input-tokens budget of %d; inspect prompt_size in the archived "
        "API call instead of sending a provider request" % (estimated, max_input_tokens)
    )


def _compact_observed(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    data = dict(value)
    return {
        "measurement": _compact_measurement(data.get("measurement")),
        "profile": _compact_profile(data.get("profile")),
        "diagnosis": _compact_diagnosis(data.get("diagnosis")),
    }


def _compact_measurement(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    data = dict(value)
    result: Dict[str, Any] = {
        "correct": data.get("correct"),
        "latency_ms": data.get("latency_ms"),
        "sample_count": len(data.get("samples_ms") or []),
    }
    if data.get("error"):
        result["error"] = _bounded_text(data["error"], 1600)
    metrics = data.get("metrics")
    if isinstance(metrics, Mapping):
        metrics = dict(metrics)
        result["measurement_source"] = metrics.get("measurement_source")
        result["primary_case_id"] = metrics.get("primary_case_id")
        result["case_count"] = metrics.get("case_count")
        result["held_out_case_count"] = metrics.get("held_out_case_count")
        if isinstance(metrics.get("statistics"), Mapping):
            result["statistics"] = _bounded_mapping(
                metrics["statistics"], max_items=12, depth=2
            )
        cases = metrics.get("cases")
        if isinstance(cases, Sequence) and not isinstance(cases, (str, bytes)):
            result["cases"] = [_compact_case(item) for item in list(cases)[:6]]
    return _drop_none(result)


def _compact_case(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"value": _bounded_text(value, 300)}
    data = dict(value)
    result: Dict[str, Any] = {}
    for key in ("case_id", "held_out", "correctness"):
        if key in data:
            result[key] = _bounded_value(data[key], depth=1)
    arguments = data.get("factory_arguments")
    if isinstance(arguments, Mapping):
        result["factory_arguments"] = _bounded_mapping(
            arguments, max_items=16, depth=2
        )
    runtime = data.get("runtime")
    if isinstance(runtime, Mapping):
        result["runtime"] = _select_mapping(
            runtime,
            (
                "kernel_name",
                "benchmark_backend",
                "benchmark_latency_ms",
                "compile_seconds",
                "correctness_seconds",
            ),
        )
    return result


def _compact_profile(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    data = dict(value)
    result = {
        "valid": data.get("valid"),
        "bottleneck": data.get("bottleneck"),
        "metrics": _compact_metrics(data.get("metrics"), limit=48),
    }
    if data.get("error"):
        result["error"] = _bounded_text(data["error"], 1600)
    return _drop_none(result)


def _compact_model(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    data = dict(value)
    result: Dict[str, Any] = {
        "valid": data.get("valid"),
        "predicted_latency_ms": data.get("predicted_latency_ms"),
        "calibrated_latency_ms": data.get("calibrated_latency_ms"),
        "bottleneck": data.get("bottleneck"),
        "confidence": data.get("confidence"),
        "metrics": _compact_metrics(data.get("metrics"), limit=48),
        "calibration": _bounded_mapping(
            data.get("calibration") or {}, max_items=16, depth=3
        ),
    }
    diagnostics = data.get("diagnostics") or []
    if diagnostics:
        result["diagnostics"] = [
            _bounded_text(item, 1000) for item in list(diagnostics)[:5]
        ]
    return _drop_none(result)


def _compact_tir(value: Any) -> Optional[Dict[str, Any]]:
    """Bound static compiler evidence while preserving optimization signals."""

    if not isinstance(value, Mapping):
        return None
    data = dict(value)
    features = data.get("features")
    compact_features: Dict[str, Any] = {}
    if isinstance(features, Mapping):
        features = dict(features)
        for key in (
            "analysis_source",
            "symbol",
            "grid_shape",
            "threads_per_block",
            "warps_per_block",
            "estimated_shared_memory_bytes",
            "estimated_registers_per_thread",
            "operation_kind_counts",
            "resource_totals_per_source_iteration",
            "structural_fingerprint",
            "diagnostic_count",
            "primary_case_id",
            "target",
        ):
            if key in features:
                compact_features[key] = _bounded_value(features[key], depth=3)
        for key, limit in (("buffers", 16), ("loops", 16), ("operations", 24)):
            values = features.get(key)
            if isinstance(values, Sequence) and not isinstance(
                values, (str, bytes)
            ):
                compact_features[key] = [
                    _bounded_value(item, depth=3) for item in list(values)[:limit]
                ]
    result: Dict[str, Any] = {
        "valid": data.get("valid"),
        "features": compact_features,
    }
    diagnostics = data.get("diagnostics") or []
    if diagnostics:
        result["diagnostics"] = [
            _bounded_text(item, 800) for item in list(diagnostics)[:6]
        ]
    if data.get("error"):
        result["error"] = _bounded_text(data["error"], 1200)
    return _drop_none(result)


def _compact_diagnosis(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    data = dict(value)
    return _drop_none(
        {
            "category": data.get("category"),
            "confidence": data.get("confidence"),
            "summary": (
                _bounded_text(data["summary"], 900)
                if data.get("summary") is not None
                else None
            ),
            "limiting_factors": [
                _bounded_text(item, 400)
                for item in list(data.get("limiting_factors") or [])[:6]
            ],
            "recommendations": [
                _bounded_text(item, 500)
                for item in list(data.get("recommendations") or [])[:6]
            ],
            "metrics": _compact_metrics(data.get("metrics"), limit=32),
            "source": data.get("source"),
        }
    )


def _compact_lessons(value: Any, limit: int) -> list[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [_compact_lesson(item) for item in list(value)[:limit]]


def _compact_lesson(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"value": _bounded_text(value, 300)}
    data = dict(value)
    result: Dict[str, Any] = {}
    for key in (
        "lesson_id",
        "round_number",
        "candidate_id",
        "parent_id",
        "kind",
        "bottleneck",
        "confidence",
    ):
        if key in data:
            result[key] = _bounded_value(data[key], depth=1)
    for key in ("observed_fact", "hypothesis_tested", "result"):
        if data.get(key) is not None:
            result[key] = _bounded_text(data[key], 700)
    result["source_regions"] = [
        _bounded_text(item, 300) for item in list(data.get("source_regions") or [])[:8]
    ]
    result["recommended_actions"] = [
        _bounded_text(item, 500)
        for item in list(data.get("recommended_actions") or [])[:6]
    ]
    evidence = data.get("supporting_evidence")
    if isinstance(evidence, Mapping):
        compact_evidence: Dict[str, Any] = {}
        if isinstance(evidence.get("failure"), Mapping):
            compact_evidence["failure"] = compact_failure_context(evidence["failure"])
        if isinstance(evidence.get("diagnosis"), Mapping):
            compact_evidence["diagnosis"] = _compact_diagnosis(evidence["diagnosis"])
        if isinstance(evidence.get("profile_metrics"), Mapping):
            compact_evidence["profile_metrics"] = _compact_metrics(
                evidence["profile_metrics"], limit=40
            )
        for key in (
            "measured_latency_ms",
            "relative_change_vs_parent",
            "raw_predicted_latency_ms",
            "calibrated_predicted_latency_ms",
        ):
            if key in evidence:
                compact_evidence[key] = _bounded_value(evidence[key], depth=1)
        result["supporting_evidence"] = compact_evidence
    return result


def _compact_history_record(value: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(value)
    result: Dict[str, Any] = {}
    for key in (
        "candidate_id",
        "parent_id",
        "generation",
        "state",
        "lineage_kind",
        "repair_depth",
        "predicted_latency_ms",
        "calibrated_predicted_latency_ms",
        "measured_latency_ms",
        "profile_bottleneck",
        "strategy_slot",
        "strategy_id",
        "strategy_validation",
        "strategy_selection_reason",
        "discovered_strategy",
        "related_existing_strategies",
        "novelty_classification",
        "structural_change",
        "tir_structural_fingerprint",
    ):
        if key in data:
            result[key] = _bounded_value(data[key], depth=1)
    if data.get("hypothesis") is not None:
        result["hypothesis"] = _bounded_text(data["hypothesis"], 700)
    if isinstance(data.get("failure"), Mapping):
        result["failure"] = _compact_history_failure(data["failure"])
    if isinstance(data.get("diagnosis"), Mapping):
        result["diagnosis"] = _compact_diagnosis(data["diagnosis"])
    return result


def _compact_history_failure(value: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(value)
    result = _select_mapping(
        data, ("stage", "category", "retryable", "error_type")
    )
    if data.get("message") is not None:
        result["message"] = _bounded_text(data["message"], 700)
    diagnostics = data.get("diagnostics") or []
    if isinstance(diagnostics, Sequence) and not isinstance(
        diagnostics, (str, bytes)
    ):
        result["diagnostic_excerpt"] = [
            _bounded_text(item, 400) for item in list(diagnostics)[:1]
        ]
    return result


def _compact_trust(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    return _bounded_mapping(value, max_items=20, depth=3)


def _compact_calibration(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    data = dict(value)
    result = _select_mapping(
        data, ("policy", "min_samples", "observation_count")
    )
    groups = data.get("groups")
    if isinstance(groups, Mapping):
        selected = sorted(
            groups.items(),
            key=lambda item: float(
                item[1].get("samples", 0) if isinstance(item[1], Mapping) else 0
            ),
            reverse=True,
        )[:8]
        result["groups"] = {
            str(key): _bounded_mapping(value, max_items=8, depth=2)
            if isinstance(value, Mapping)
            else _bounded_value(value, depth=1)
            for key, value in selected
        }
    return result


def _compact_metrics(value: Any, limit: int) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    data = dict(value)
    selected: Dict[str, Any] = {}
    for key in sorted(data):
        lowered = str(key).lower()
        if lowered in _DROPPED_METRIC_KEYS:
            continue
        if lowered == "ptxas" and isinstance(data[key], Mapping):
            selected[str(key)] = _compact_ptxas(data[key])
            continue
        if lowered in _IMPORTANT_METRIC_NAMES or any(
            hint in lowered for hint in _METRIC_HINTS
        ):
            selected[str(key)] = _bounded_value(data[key], depth=3)
        if len(selected) >= limit:
            break
    return selected


def _compact_ptxas(value: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(value)
    result: Dict[str, Any] = {}
    for key in (
        "name",
        "registers_per_thread",
        "shared_bytes",
        "spill_store_bytes",
        "spill_load_bytes",
    ):
        if key in data:
            result[key] = _bounded_value(data[key], depth=1)
    function = data.get("function")
    if isinstance(function, Mapping):
        result["function"] = _select_mapping(
            function,
            (
                "name",
                "registers_per_thread",
                "shared_bytes",
                "spill_store_bytes",
                "spill_load_bytes",
            ),
        )
    return result


def _select_mapping(value: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    return {
        key: _bounded_value(value[key], depth=2)
        for key in keys
        if key in value and value[key] is not None
    }


def _bounded_mapping(
    value: Mapping[str, Any], *, max_items: int, depth: int
) -> Dict[str, Any]:
    return {
        str(key): _bounded_value(item, depth=depth - 1)
        for key, item in list(
            sorted(value.items(), key=lambda pair: str(pair[0]))
        )[:max_items]
    }


def _bounded_value(value: Any, *, depth: int) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _bounded_text(value, 700)
    if depth <= 0:
        return _bounded_text(value, 300)
    if isinstance(value, Mapping):
        return _bounded_mapping(value, max_items=16, depth=depth)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_bounded_value(item, depth=depth - 1) for item in list(value)[:12]]
    return _bounded_text(value, 300)


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + "... [truncated %d characters]" % omitted


def _drop_none(value: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if item is not None and item != [] and item != {}
    }


def _is_small_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float)) or (
        isinstance(value, str) and len(value) <= 300
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _json_size(value: Any) -> int:
    return len(_json(value))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
