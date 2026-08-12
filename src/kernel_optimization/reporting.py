"""Human- and machine-readable research artifacts for one optimization run."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .archive import ArtifactStore
from .schema import CandidateRecord, SearchSummary, TaskSpec


_REPORT_FILENAMES = {
    "trajectory_json": "trajectory.json",
    "trajectory_csv": "trajectory.csv",
    "candidate_graph": "candidate_graph.json",
    "experiment_manifest": "experiment_manifest.json",
    "experiment_report": "experiment_report.md",
}


def expected_report_paths(root: Path) -> Dict[str, str]:
    return {name: str(root / filename) for name, filename in _REPORT_FILENAMES.items()}


def write_research_artifacts(
    store: ArtifactStore,
    task: TaskSpec,
    records: Sequence[CandidateRecord],
    summary: SearchSummary,
    run_metadata: Mapping[str, Any],
) -> Dict[str, str]:
    ordered = sorted(
        records,
        key=lambda record: (
            record.candidate.generation,
            record.candidate.candidate_id,
        ),
    )
    trajectory = [_trajectory_row(record) for record in ordered]
    store.save_json_artifact(
        _REPORT_FILENAMES["trajectory_json"], {"candidates": trajectory}
    )
    store.save_text_artifact(
        _REPORT_FILENAMES["trajectory_csv"], _trajectory_csv(trajectory)
    )
    store.save_json_artifact(
        _REPORT_FILENAMES["candidate_graph"],
        {
            "nodes": trajectory,
            "edges": [
                {
                    "source": record.candidate.parent_id,
                    "target": record.candidate.candidate_id,
                    "kind": record.candidate.lineage_kind,
                }
                for record in ordered
                if record.candidate.parent_id is not None
            ],
        },
    )
    store.save_json_artifact(
        _REPORT_FILENAMES["experiment_manifest"],
        {
            "task_id": task.task_id,
            "target": task.target,
            "workload": task.workload,
            "budget": task.budget.__dict__,
            "run": dict(run_metadata),
            "policies": {
                "selection": summary.selection_policy,
                "profile": summary.profile_policy,
                "fixed_promotions_per_round": summary.fixed_promotions_per_round,
                "compiled_deduplication": summary.compiled_deduplication,
            },
            "outcome": summary.to_dict(),
        },
    )
    store.save_text_artifact(
        _REPORT_FILENAMES["experiment_report"],
        _experiment_markdown(task, ordered, summary),
    )
    return expected_report_paths(store.root)


def _trajectory_row(record: CandidateRecord) -> Dict[str, Any]:
    model = record.model
    measurement = record.measurement
    final = record.final_measurement
    compiled_hash = model.metrics.get("compiled_source_sha256") if model else None
    compiled_identity_hash = (
        model.metrics.get("compiled_identity_sha256") if model else None
    )
    return {
        "candidate_id": record.candidate.candidate_id,
        "parent_id": record.candidate.parent_id,
        "generation": record.candidate.generation,
        "lineage_kind": record.candidate.lineage_kind,
        "repair_depth": record.candidate.repair_depth,
        "source_sha256": record.candidate.source_sha256,
        "compiled_source_sha256": compiled_hash,
        "compiled_identity_sha256": compiled_identity_hash,
        "compiled_equivalent_to": record.compiled_equivalent_to,
        "state": record.state,
        "hypothesis": record.candidate.hypothesis,
        "selection_reasons": list(record.selection_reasons),
        "decision_reason": record.decision_reason,
        "raw_model_latency_ms": model.predicted_latency_ms if model else None,
        "ranking_model_latency_ms": model.ranking_latency_ms if model else None,
        "measured_latency_ms": measurement.latency_ms if measurement else None,
        "final_latency_ms": final.latency_ms if final else None,
        "correct": measurement.correct if measurement else None,
        "final_correct": final.correct if final else None,
        "model_bottleneck": model.bottleneck if model else None,
        "profile_bottleneck": record.profile.bottleneck if record.profile else None,
        "diagnosis": record.diagnosis.category if record.diagnosis else None,
        "failure": record.failure.category if record.failure else None,
    }


def _trajectory_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    output = StringIO()
    fieldnames = list(rows[0]) if rows else list(_trajectory_row.__annotations__)
    if not rows:
        fieldnames = [
            "candidate_id",
            "parent_id",
            "generation",
            "state",
            "measured_latency_ms",
        ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        value = dict(row)
        value["selection_reasons"] = "|".join(value.get("selection_reasons") or [])
        writer.writerow(value)
    return output.getvalue()


def _experiment_markdown(
    task: TaskSpec,
    records: Sequence[CandidateRecord],
    summary: SearchSummary,
) -> str:
    ledger = summary.cost_ledger
    api = dict(ledger.get("api") or {})
    evaluator = dict(ledger.get("evaluator") or {})
    hardware = dict(ledger.get("hardware") or {})
    lines = [
        "# Optimization Experiment Report",
        "",
        "## Configuration",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| Task | `%s` |" % task.task_id,
        "| Selection policy | `%s` |" % summary.selection_policy,
        "| Profile policy | `%s` |" % summary.profile_policy,
        "| Fixed promotions per round | %s |"
        % _format(summary.fixed_promotions_per_round),
        "| Compiled-code deduplication | %s |"
        % ("enabled" if summary.compiled_deduplication else "disabled"),
        "| Completed rounds | %d |" % summary.completed_rounds,
        "",
        "## Outcome",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        "| Seed latency | %s ms |" % _format(summary.seed_latency_ms),
        "| Best final latency | %s ms |" % _format(summary.best_latency_ms),
        "| Speedup over seed | %sx |" % _format(summary.speedup_over_seed),
        "| Search-best latency | %s ms |"
        % _format(summary.search_best_latency_ms),
        "| Generated candidates | %d |" % summary.generated_candidates,
        "| Measured candidates | %d |" % summary.measured_candidates,
        "| Compiled-equivalent candidates skipped | %d |"
        % summary.compiled_equivalent_candidates,
        "",
        "## Cost Ledger",
        "",
        "| Cost | Value |",
        "| --- | ---: |",
        "| Hosted API logical calls | %s |"
        % _format(api.get("logical_generator_calls")),
        "| Hosted API request attempts | %s |"
        % _format(api.get("provider_request_attempts")),
        "| Hosted API wall time | %s s |" % _format(api.get("wall_seconds")),
        "| Input tokens | %s |" % _format(api.get("input_tokens")),
        "| Output tokens | %s |" % _format(api.get("output_tokens")),
        "| Estimated API cost | %s USD |"
        % _format(api.get("estimated_cost_usd")),
        "| Evaluator wall time | %s s |"
        % _format(evaluator.get("total_wall_seconds")),
        "| CUDA Event candidate calls | %s |"
        % _format(hardware.get("cuda_event_candidate_calls")),
        "| NCU calls | %s |" % _format(hardware.get("ncu_profile_calls")),
        "| Final validation calls | %d |" % summary.final_validation_calls,
        "",
        "## Round Trajectory",
        "",
        "| Round | Best measured candidate | Latency (ms) |",
        "| ---: | --- | ---: |",
    ]
    best = None
    max_generation = max((record.candidate.generation for record in records), default=0)
    for generation in range(max_generation + 1):
        eligible = [
            record
            for record in records
            if record.candidate.generation <= generation
            and record.is_measured_correct
        ]
        if eligible:
            best = min(
                eligible,
                key=lambda record: (
                    float(record.measurement.latency_ms),
                    record.candidate.candidate_id,
                ),
            )
        if best is not None:
            lines.append(
                "| %d | `%s` | %s |"
                % (
                    generation,
                    best.candidate.candidate_id,
                    _format(best.measurement.latency_ms),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The exported kernel passed fresh final validation. Model predictions are "
            "used only for allocation of the hardware budget; CUDA Event measurements "
            "rank the search beam, and held-out final measurements choose the export.",
            "",
            "Detailed per-candidate evidence is available in `trajectory.csv`, "
            "`trajectory.json`, and `candidate_graph.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "%.6g" % value
    return str(value)
