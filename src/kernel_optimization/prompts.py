"""English-only prompts for hosted source-code candidate generation."""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence

from .schema import Candidate, TaskSpec


SYSTEM_PROMPT = """You are a senior GPU kernel optimization engineer.

You propose complete replacement source files for an automated,
correctness-gated optimization system. You do not execute code, fabricate
measured results, or modify the reference implementation, evaluator, workload,
or external interface. Treat fields under observed_evidence as hardware facts.
Treat fields under predicted_evidence as analytical model predictions that may
be wrong.

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
        "observed_evidence": evidence.get("observed"),
        "predicted_evidence": evidence.get("predicted"),
        "model_trust": evidence.get("model_trust"),
        "recent_history": list(history)[-12:],
        "rules": [
            "Return the complete replacement content of the kernel source file.",
            "Preserve mathematical semantics, entrypoint, and external interface.",
            "Do not modify or reproduce the evaluator or reference implementation.",
            "Do not add file, network, shell, subprocess, or environment access.",
            "Do not claim that predicted metrics were measured on hardware.",
            "Make each candidate materially different and explain one primary hypothesis.",
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
                    },
                }
            ]
        },
    }
    return json.dumps(request, indent=2, sort_keys=True)
