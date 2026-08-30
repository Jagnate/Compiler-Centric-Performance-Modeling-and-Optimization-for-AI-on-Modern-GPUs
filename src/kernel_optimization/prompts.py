"""English-only prompts for hosted source-code candidate generation."""

from __future__ import annotations

import json
from typing import Any, Dict, Sequence

from .prompt_compression import (
    COMPRESSION_VERSION,
    NO_COMPRESSION_POLICY,
    compact_failure_context,
    compress_prompt_context,
)
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
on. Strategy assignments are coverage priors, not an exhaustive whitelist of
legal implementations. Compatible transformations may be combined. For repair
requests, make the smallest useful semantics-preserving change that addresses
the classified failure before attempting additional tuning.

The candidate source is untrusted and will be parsed, compiled, checked against
the reference, checked for parent-relative AST novelty, and benchmarked
independently. A structural strategy must contain a concrete structural source
change; a description alone is not evidence. Every candidate object must contain
a non-empty source_code with the complete Python file. Never omit source_code or
replace it with a patch, placeholder, summary, or unchanged marker. If response
space is insufficient, return fewer fully materialized candidates rather than
metadata-only candidates. Return only valid JSON matching the requested schema.
Do not include Markdown, comments outside JSON, patches, or additional top-level
keys.
"""


STRATEGY_PLANNER_SYSTEM_PROMPT = """You plan one round of GPU kernel search.

Choose how many candidate slots to allocate to optimization directions before
any source candidates are generated. Base the decision on the current kernel,
observed hardware evidence, analytical predictions, prior strategy outcomes,
and remaining search budget. No strategy is mandatory, including parameter
tuning. Do not optimize for a particular benchmark name or assume that a fixed
portfolio is universally best.

Use known strategy IDs when they fit. You may propose a new short strategy ID;
the controller will route it through open structural exploration. Balance
exploitation of measured improvements with exploration when evidence is weak or
the search has plateaued. Return only the requested JSON plan. Do not return
source code, Markdown, or additional top-level keys.
"""


def build_strategy_planning_prompt(
    task: TaskSpec,
    parent: Candidate,
    planning_context: Dict[str, Any],
    strategies: Sequence[Dict[str, Any]],
    count: int,
    round_number: int,
) -> str:
    """Build a compact strategy-neutral allocation request for one round."""

    task_payload = {
        "task_id": task.task_id,
        "description": task.description,
        "reference_semantics": task.reference,
        "entrypoint": task.entrypoint,
        "target": task.target,
        "workload": task.workload,
        "constraints": task.constraints,
        "remaining_rounds_including_this_one": (
            task.budget.rounds - round_number + 1
        ),
    }
    _add_baseline_protocol(task_payload, task)
    request = {
        "objective": (
            "Allocate this round's candidate-generation slots to the most useful "
            "optimization directions for the current kernel and evidence."
        ),
        "round": round_number,
        "candidate_count": count,
        "task": task_payload,
        "current_best": {
            "candidate_id": parent.candidate_id,
            "generation": parent.generation,
            "source_name": parent.source_name,
            "source_sha256": parent.source_sha256,
            "hypothesis": parent.hypothesis,
            "source_code": parent.source_code,
        },
        "planning_evidence": planning_context,
        "available_strategies": list(strategies),
        "controller_constraints": {
            "exact_total_slots": count,
            "minimum_distinct_directions_when_possible": min(2, count),
            "maximum_share_for_one_direction": (
                "two thirds of slots, rounded up"
            ),
            "parameter_tuning_is_optional": True,
            "unknown_strategy_behavior": (
                "A new strategy ID is allowed and will be normalized to the open "
                "structural lane while preserving the proposed direction."
            ),
        },
        "rules": [
            "Allocation counts must sum exactly to candidate_count.",
            "Give every allocation item a concrete evidence-backed reason.",
            "Use mode exploit for a direction supported by outcomes and explore for uncertainty reduction.",
            "Allocate zero slots to any strategy, including parameter-tuning, when the evidence does not justify it.",
            "Do not force broad coverage when focused exploitation is better, but retain at least two directions when candidate_count permits.",
            "Do not invent measurements or treat analytical predictions as hardware observations.",
        ],
        "response_schema": {
            "allocation": [
                {
                    "strategy_id": "known strategy ID or a new short ID",
                    "count": "positive integer",
                    "mode": "exploit|explore",
                    "reason": "evidence-backed allocation rationale",
                    "evidence": ["specific evidence field, outcome, or uncertainty"],
                    "objective": "optional concrete direction for generated candidates",
                }
            ],
            "round_rationale": "short explanation of the allocation as a whole",
            "confidence": "number from 0 to 1",
        },
    }
    return json.dumps(request, indent=2, sort_keys=True)


def build_optimization_prompt(
    task: TaskSpec,
    parent: Candidate,
    evidence: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    count: int,
    metadata_compression_policy: str = COMPRESSION_VERSION,
) -> str:
    """Build one source optimization request with explicit evidence provenance."""

    context = compress_prompt_context(
        evidence, history, policy=metadata_compression_policy
    )
    generation_request = dict(evidence.get("generation_request") or {})
    strategy_assignments = list(
        generation_request.get("strategy_assignments") or []
    )
    discovered_strategy_memory = list(
        generation_request.get("discovered_strategy_memory") or []
    )
    strategy_rules = []
    if strategy_assignments:
        strategy_rules = [
            "Return exactly one candidate for each strategy assignment, in the listed order.",
            (
                "Copy the assigned slot and strategy ID exactly into "
                "metadata.strategy_slot and metadata.strategy_id."
            ),
            (
                "When structural_required is true, change executable AST "
                "structure; numeric defaults, threads, num_stages, comments, "
                "formatting, and local renaming alone are invalid."
            ),
            (
                "Implement the assigned transformation in source_code rather "
                "than merely describing it in the hypothesis or metadata."
            ),
            (
                "Treat each assigned strategy as the primary coverage target, not "
                "as a whitelist; combine compatible transformations when useful."
            ),
            (
                "For open-structural-exploration, either discover a transformation "
                "not adequately represented by the named lanes or deepen a useful "
                "strategy in discovered_strategy_memory. Name it in "
                "metadata.discovered_strategy and list overlaps in "
                "metadata.related_existing_strategies."
            ),
        ]
    task_payload = {
        "task_id": task.task_id,
        "description": task.description,
        "reference_semantics": task.reference,
        "entrypoint": task.entrypoint,
        "language": task.language,
        "target": task.target,
        "workload": task.workload,
        "constraints": task.constraints,
    }
    _add_baseline_protocol(task_payload, task)
    request = {
        "objective": (
            "Propose diverse complete kernel source files that preserve semantics "
            "and may reduce measured latency on the target GPU."
        ),
        "candidate_count": count,
        "task": task_payload,
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
        "strategy_assignments": strategy_assignments,
        "discovered_strategy_memory": discovered_strategy_memory,
        "rules": [
            "Return the complete replacement content of the kernel source file.",
            "Every candidate must contain a non-empty source_code with the complete Python file.",
            "Never replace source_code with a patch, placeholder, summary, omission, or unchanged marker.",
            "If response space is insufficient, return fewer complete candidates instead of metadata-only candidates.",
            "Preserve mathematical semantics, entrypoint, and external interface.",
            "Do not modify or reproduce the evaluator or reference implementation.",
            "Do not add file, network, shell, subprocess, or environment access.",
            "Do not claim that predicted metrics were measured on hardware.",
            "Make each candidate materially different and explain one primary hypothesis.",
            "List the shared evidence lesson IDs used by each candidate in metadata.evidence_ids.",
            "Use only APIs available to the input source and declared task environment.",
        ]
        + strategy_rules,
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
                        "strategy_slot": "exact assigned slot",
                        "strategy_id": "exact assigned strategy ID",
                        "discovered_strategy": (
                            "required short strategy name for the open exploration "
                            "slot; otherwise null"
                        ),
                        "related_existing_strategies": ["strategy ID"],
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
    metadata_compression_policy: str = COMPRESSION_VERSION,
) -> str:
    """Build a bounded repair request around one archived failed source."""

    context = compress_prompt_context(
        evidence, history, policy=metadata_compression_policy
    )
    task_payload = {
        "task_id": task.task_id,
        "description": task.description,
        "reference_semantics": task.reference,
        "entrypoint": task.entrypoint,
        "language": task.language,
        "target": task.target,
        "workload": task.workload,
        "constraints": task.constraints,
    }
    _add_baseline_protocol(task_payload, task)
    request = {
        "mode": "repair",
        "objective": (
            "Return one complete replacement source that fixes the classified "
            "failure while preserving semantics and the external interface."
        ),
        "candidate_count": 1,
        "task": task_payload,
        "failed_candidate": {
            "candidate_id": failed.candidate_id,
            "parent_id": failed.parent_id,
            "generation": failed.generation,
            "repair_depth": failed.repair_depth,
            "source_name": failed.source_name,
            "source_sha256": failed.source_sha256,
            "hypothesis": failed.hypothesis,
            "source_code": failed.source_code,
            "assigned_strategy": {
                "strategy_slot": failed.proposal_metadata.get("strategy_slot"),
                "strategy_id": failed.proposal_metadata.get("strategy_id"),
                "structural_required": failed.proposal_metadata.get(
                    "structural_required"
                ),
                "novelty_parent_id": failed.proposal_metadata.get(
                    "novelty_parent_id"
                ),
                "discovered_strategy": failed.proposal_metadata.get(
                    "discovered_strategy"
                ),
                "related_existing_strategies": failed.proposal_metadata.get(
                    "related_existing_strategies"
                ),
            },
        },
        "classified_failure": (
            dict(failure)
            if metadata_compression_policy == NO_COMPRESSION_POLICY
            else compact_failure_context(failure)
        ),
        "shared_evidence_memory": context.evidence.get("shared_memory", []),
        "model_trust": context.evidence.get("model_trust"),
        "calibration_state": context.evidence.get("calibration"),
        "recent_history": context.history,
        "context_provenance": context.provenance,
        "rules": [
            "Return exactly one complete Python source file, not a patch.",
            "The candidate must contain non-empty source_code; never omit it or use a placeholder.",
            "Fix the classified failure before applying unrelated optimizations.",
            "Preserve mathematical semantics, entrypoint, and external interface.",
            "Do not modify or reproduce the evaluator or reference implementation.",
            "Do not add file, network, shell, subprocess, or environment access.",
            "Use the smallest change that can be independently compiled and checked.",
            "Cite relevant shared lesson IDs in metadata.evidence_ids.",
            (
                "Preserve the assigned strategy and make its required structural "
                "change concrete after fixing syntax or compilation."
            ),
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
                        "strategy_slot": "the failed candidate's assigned slot",
                        "strategy_id": "the failed candidate's assigned strategy ID",
                        "discovered_strategy": (
                            "preserve the failed open candidate's discovered strategy"
                        ),
                        "related_existing_strategies": ["strategy ID"],
                        "changed_regions": ["string"],
                        "evidence_ids": ["string"],
                        "applied_recommendations": ["string"],
                    },
                }
            ]
        },
    }
    return json.dumps(request, indent=2, sort_keys=True)


def _add_baseline_protocol(payload: Dict[str, Any], task: TaskSpec) -> None:
    """Expose only the compact model-facing portion of a style preset."""

    protocol = task.metadata.get("baseline_style_protocol")
    if isinstance(protocol, dict):
        payload["optimization_protocol"] = dict(protocol)
