"""English-only prompts for hosted source-code candidate generation."""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence

from .prompt_compression import compact_failure_context, compress_prompt_context
from .schema import Candidate, TaskSpec


SYSTEM_PROMPT = """You are a senior GPU kernel optimization engineer.

You propose complete replacement source files for an automated,
correctness-gated optimization system. You do not execute code, fabricate
measured results, or modify the reference implementation, evaluator, workload,
or external interface. Treat fields under observed_evidence as hardware facts.
Treat fields under predicted_evidence as analytical model predictions that may
be wrong.

When shared_evidence_memory is present, cite the lesson IDs that motivated each
candidate and state which recommendation or failed hypothesis is being acted
on. For repair requests, make the smallest useful semantics-preserving change
that addresses the classified failure before attempting additional tuning.

The candidate source is untrusted and will be parsed, compiled, checked against
the reference, and benchmarked independently. Return only valid JSON matching
the requested schema. Do not include Markdown, comments outside JSON, patches,
or additional top-level keys.
"""


def build_optimization_prompt(
    task: TaskSpec,
    parent: Candidate,
    evidence: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    count: int,
) -> str:
    """Build one source optimization request with explicit evidence provenance."""

    context = compress_prompt_context(evidence, history)
    request = {
        "objective": (
            "Propose diverse complete kernel source files that preserve semantics "
            "and may reduce measured latency on the target GPU."
        ),
        "candidate_count": count,
        "task": {
            "task_id": task.task_id,
            "description": task.description,
            "reference_semantics": task.reference,
            "entrypoint": task.entrypoint,
            "language": task.language,
            "target": task.target,
            "workload": task.workload,
            "constraints": task.constraints,
        },
        "parent": {
            "candidate_id": parent.candidate_id,
            "generation": parent.generation,
            "source_name": parent.source_name,
            "source_sha256": parent.source_sha256,
            "hypothesis": parent.hypothesis,
            "source_code": parent.source_code,
        },
        "observed_evidence": context.evidence.get("observed"),
        "predicted_evidence": context.evidence.get("predicted"),
        "model_trust": context.evidence.get("model_trust"),
        "calibration_state": context.evidence.get("calibration"),
        "shared_evidence_memory": context.evidence.get("shared_memory", []),
        "recent_history": context.history,
        "context_provenance": context.provenance,
        "rules": [
            "Return the complete replacement content of the kernel source file.",
            "Preserve mathematical semantics, entrypoint, and external interface.",
            "Do not modify or reproduce the evaluator or reference implementation.",
            "Do not add file, network, shell, subprocess, or environment access.",
            "Do not claim that predicted metrics were measured on hardware.",
            "Make each candidate materially different and explain one primary hypothesis.",
            "List the shared evidence lesson IDs used by each candidate in metadata.evidence_ids.",
            "Use only APIs available to the input source and declared task environment.",
        ],
        "response_schema": {
            "candidates": [
                {
                    "hypothesis": "string",
                    "source_code": "complete Python source as a JSON string",
                    "expected_effect": {
                        "latency": "increase|decrease|unknown",
                        "bottleneck": "string",
                        "reason": "string",
                    },
                    "metadata": {
                        "strategy": "string",
                        "changed_regions": ["string"],
                        "evidence_ids": ["string"],
                        "applied_recommendations": ["string"],
                    },
                }
            ]
        },
    }
    return json.dumps(request, indent=2, sort_keys=True)


def build_repair_prompt(
    task: TaskSpec,
    failed: Candidate,
    failure: Dict[str, Any],
    evidence: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
) -> str:
    """Build a bounded repair request around one archived failed source."""

    context = compress_prompt_context(evidence, history)
    request = {
        "mode": "repair",
        "objective": (
            "Return one complete replacement source that fixes the classified "
            "failure while preserving semantics and the external interface."
        ),
        "candidate_count": 1,
        "task": {
            "task_id": task.task_id,
            "description": task.description,
            "reference_semantics": task.reference,
            "entrypoint": task.entrypoint,
            "language": task.language,
            "target": task.target,
            "workload": task.workload,
            "constraints": task.constraints,
        },
        "failed_candidate": {
            "candidate_id": failed.candidate_id,
            "parent_id": failed.parent_id,
            "generation": failed.generation,
            "repair_depth": failed.repair_depth,
            "source_name": failed.source_name,
            "source_sha256": failed.source_sha256,
            "hypothesis": failed.hypothesis,
            "source_code": failed.source_code,
        },
        "classified_failure": compact_failure_context(failure),
        "shared_evidence_memory": context.evidence.get("shared_memory", []),
        "model_trust": context.evidence.get("model_trust"),
        "calibration_state": context.evidence.get("calibration"),
        "recent_history": context.history,
        "context_provenance": context.provenance,
        "rules": [
            "Return exactly one complete Python source file, not a patch.",
            "Fix the classified failure before applying unrelated optimizations.",
            "Preserve mathematical semantics, entrypoint, and external interface.",
            "Do not modify or reproduce the evaluator or reference implementation.",
            "Do not add file, network, shell, subprocess, or environment access.",
            "Use the smallest change that can be independently compiled and checked.",
            "Cite relevant shared lesson IDs in metadata.evidence_ids.",
        ],
        "response_schema": {
            "candidates": [
                {
                    "hypothesis": "string describing the repair",
                    "source_code": "complete repaired Python source as a JSON string",
                    "expected_effect": {
                        "failure": "fixed|unknown",
                        "latency": "increase|decrease|unknown",
                        "reason": "string",
                    },
                    "metadata": {
                        "strategy": "repair",
                        "changed_regions": ["string"],
                        "evidence_ids": ["string"],
                        "applied_recommendations": ["string"],
                    },
                }
            ]
        },
    }
    return json.dumps(request, indent=2, sort_keys=True)
