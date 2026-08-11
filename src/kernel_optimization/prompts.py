"""English-only prompts for hosted candidate generation."""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence

from .schema import Candidate, TaskSpec


SYSTEM_PROMPT = """You are a senior GPU kernel optimization engineer.

You propose candidate edits for an automated, correctness-gated optimization
system. You do not execute code, fabricate measured results, or modify the
reference implementation and evaluator. Treat fields under observed_evidence
as hardware facts. Treat fields under predicted_evidence as analytical model
predictions that may be wrong.

Return only valid JSON matching the requested schema. Do not include Markdown,
explanations outside JSON, comments, or additional keys.
"""


def build_optimization_prompt(
    task: TaskSpec,
    parent: Candidate,
    evidence: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    count: int,
) -> str:
    """Build a structured optimization request with explicit evidence provenance."""

    request = {
        "objective": "Propose diverse, legal edits that may reduce measured kernel latency.",
        "candidate_count": count,
        "task": {
            "task_id": task.task_id,
            "description": task.description,
            "reference": task.reference,
            "search_space": {
                name: list(values) for name, values in task.search_space.items()
            },
        },
        "parent": parent.to_dict(),
        "observed_evidence": evidence.get("observed"),
        "predicted_evidence": evidence.get("predicted"),
        "model_trust": evidence.get("model_trust"),
        "recent_history": list(history)[-12:],
        "rules": [
            "Preserve mathematical semantics and the external kernel interface.",
            "Use only parameter names and values declared in search_space.",
            "Do not claim that predicted metrics were measured on hardware.",
            "Prefer distinct hypotheses instead of repeating equivalent parameter sets.",
            "Each proposal must change at least one parameter.",
        ],
        "response_schema": {
            "candidates": [
                {
                    "hypothesis": "string",
                    "parameter_updates": {"declared_parameter_name": "declared_value"},
                    "expected_effect": {
                        "latency": "increase|decrease|unknown",
                        "reason": "string",
                    },
                    "source_patch": None,
                    "metadata": {"strategy": "string"},
                }
            ]
        },
    }
    return json.dumps(request, indent=2, sort_keys=True)

