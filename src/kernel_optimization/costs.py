"""Experiment cost accounting separated from optimization quality metrics."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

from .schema import CandidateRecord


def build_cost_ledger(
    records: Iterable[CandidateRecord],
    *,
    stage_timings: Mapping[str, Mapping[str, float]],
    generator_usage: Mapping[str, float],
    preflight_calls: int,
    provider_api_requests: int,
    generator_calls: int,
    repair_calls: int,
    profile_calls: int,
    final_validation_calls: int,
    api_input_price_per_million: Optional[float] = None,
    api_output_price_per_million: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a transparent ledger without equating latency samples with GPU time."""

    archived = list(records)
    input_tokens = _token_count(generator_usage, "prompt_tokens", "input_tokens")
    output_tokens = _token_count(
        generator_usage, "completion_tokens", "output_tokens"
    )
    estimated_cost = None
    if (
        api_input_price_per_million is not None
        and api_output_price_per_million is not None
    ):
        estimated_cost = (
            input_tokens * float(api_input_price_per_million)
            + output_tokens * float(api_output_price_per_million)
        ) / 1_000_000.0

    measured = [record for record in archived if record.measurement is not None]
    finalized = [
        record for record in archived if record.final_measurement is not None
    ]
    sample_seconds = sum(
        sum(float(value) for value in measurement.samples_ms) / 1000.0
        for record in archived
        for measurement in (record.measurement, record.final_measurement)
        if measurement is not None
    )
    correctness_cases = sum(
        int(measurement.metrics.get("case_count", 0) or 0)
        for record in archived
        for measurement in (record.measurement, record.final_measurement)
        if measurement is not None
    )
    stage_copy = {
        str(stage): {str(name): float(value) for name, value in metrics.items()}
        for stage, metrics in stage_timings.items()
    }
    evaluator_seconds = sum(
        float(stage_copy.get(stage, {}).get("seconds", 0.0))
        for stage in ("model", "measure", "profile", "final")
    )
    api_seconds = sum(
        float(stage_copy.get(stage, {}).get("seconds", 0.0))
        for stage in ("api_preflight", "api_generate", "api_repair")
    )

    return {
        "api": {
            "preflight_attempts": int(preflight_calls),
            "provider_request_attempts": int(provider_api_requests),
            "logical_generator_calls": int(generator_calls),
            "repair_calls": int(repair_calls),
            "usage": dict(generator_usage),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_price_per_million_usd": api_input_price_per_million,
            "output_price_per_million_usd": api_output_price_per_million,
            "estimated_cost_usd": estimated_cost,
            "wall_seconds": api_seconds,
        },
        "evaluator": {
            "stage_timings": stage_copy,
            "total_wall_seconds": evaluator_seconds,
            "model_calls": int(stage_copy.get("model", {}).get("calls", 0)),
            "measurement_calls": int(stage_copy.get("measure", {}).get("calls", 0)),
            "final_validation_calls": int(final_validation_calls),
            "ncu_profile_calls": int(profile_calls),
        },
        "hardware": {
            "cuda_event_candidate_calls": len(measured) + len(finalized),
            "reported_correctness_case_evaluations": correctness_cases,
            "ncu_profile_calls": int(profile_calls),
            "observed_kernel_sample_seconds": sample_seconds,
            "note": (
                "Observed kernel samples exclude compilation, warmup, correctness, "
                "profiler replay, and process overhead; evaluator wall time is the "
                "appropriate end-to-end cost measure."
            ),
        },
        "candidates": {
            "archived": len(archived),
            "modeled": sum(record.model is not None for record in archived),
            "measured": len(measured),
            "correct": sum(
                record.measurement is not None and record.measurement.correct
                for record in archived
            ),
            "compiled_equivalent": sum(
                record.compiled_equivalent_to is not None for record in archived
            ),
            "finalized": len(finalized),
        },
        "accounted_stage_wall_seconds": api_seconds + evaluator_seconds,
    }


def _token_count(usage: Mapping[str, float], *names: str) -> float:
    for name in names:
        value = usage.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return 0.0
