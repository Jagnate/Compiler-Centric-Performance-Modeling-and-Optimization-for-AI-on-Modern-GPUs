"""Auditable style presets for related kernel-optimization systems.

These presets intentionally reproduce control-flow characteristics, not the
original systems or their published results.  Every run keeps this project's
source format, evaluator contract, correctness checks, and artifact schema.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, FrozenSet, Mapping, Tuple

from .schema import TaskSpec


BASELINE_STYLE_NAMES = (
    "native",
    "kernelagent",
    "kernelevolve",
    "kernelbench",
    "avo",
    "tilefoundry",
)


@dataclass(frozen=True)
class BaselineStylePreset:
    """One canonical style-emulation policy bundle."""

    name: str
    label: str
    description: str
    evaluation_policy: str
    tir_evidence_policy: str
    selection_policy: str
    profile_policy: str
    compiled_deduplication: bool
    structural_search_policy: str
    strategy_allocation_policy: str
    default_agent_workers: int
    budget_overrides: Tuple[Tuple[str, int], ...] = ()
    prompt_guidance: Tuple[str, ...] = ()
    reference_basis: Tuple[str, ...] = ()
    unsupported_capabilities: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "evaluation_policy": self.evaluation_policy,
            "tir_evidence_policy": self.tir_evidence_policy,
            "selection_policy": self.selection_policy,
            "profile_policy": self.profile_policy,
            "compiled_deduplication": self.compiled_deduplication,
            "structural_search_policy": self.structural_search_policy,
            "strategy_allocation_policy": self.strategy_allocation_policy,
            "default_agent_workers": self.default_agent_workers,
            "budget_overrides": dict(self.budget_overrides),
            "prompt_guidance": list(self.prompt_guidance),
            "reference_basis": list(self.reference_basis),
            "unsupported_capabilities": list(self.unsupported_capabilities),
        }


@dataclass(frozen=True)
class BaselineStyleResolution:
    """Effective CLI and task configuration after applying one preset."""

    preset: BaselineStylePreset
    task: TaskSpec
    agent_workers: int
    evaluation_policy: str
    tir_evidence_policy: str
    selection_policy: str
    profile_policy: str
    compiled_deduplication: bool
    structural_search_policy: str
    strategy_allocation_policy: str
    explicit_overrides: Mapping[str, Any]
    original_budget: Mapping[str, Any]

    @property
    def emulated(self) -> bool:
        return self.preset.name != "native"

    @property
    def canonical(self) -> bool:
        return not self.explicit_overrides

    def to_dict(self) -> Dict[str, Any]:
        mode = "style-emulation" if self.emulated else "native"
        return {
            "name": self.preset.name,
            "label": self.preset.label,
            "mode": mode,
            "canonical": self.canonical,
            "claim": (
                "Control-flow style emulation only; this is not an exact "
                "reimplementation and does not reproduce published results."
                if self.emulated
                else "This project's native optimization policy."
            ),
            "description": self.preset.description,
            "reference_basis": list(self.preset.reference_basis),
            "resolved_cli_controls": {
                "agent_workers": self.agent_workers,
                "evaluation_policy": self.evaluation_policy,
                "tir_evidence_policy": self.tir_evidence_policy,
                "selection_policy": self.selection_policy,
                "profile_policy": self.profile_policy,
                "compiled_deduplication": self.compiled_deduplication,
                "structural_search_policy": self.structural_search_policy,
                "strategy_allocation_policy": self.strategy_allocation_policy,
                "budget": dict(self.task.budget.__dict__),
            },
            "original_budget": dict(self.original_budget),
            "preset_budget_overrides": dict(self.preset.budget_overrides),
            "explicit_overrides": dict(self.explicit_overrides),
            "unsupported_capabilities": list(
                self.preset.unsupported_capabilities
            ),
        }


_PRESETS = {
    "native": BaselineStylePreset(
        name="native",
        label="Native TileSight-guided search",
        description=(
            "Adaptive-fidelity search with TileSight/TIR evidence, AI-planned "
            "strategy allocation, AST novelty enforcement, CUDA Event promotion, "
            "and milestone NCU profiling."
        ),
        evaluation_policy="tilesight",
        tir_evidence_policy="auto",
        selection_policy="adaptive",
        profile_policy="milestone",
        compiled_deduplication=True,
        structural_search_policy="enforce",
        strategy_allocation_policy="ai-planned",
        default_agent_workers=1,
        reference_basis=("This repository's native controller",),
    ),
    "kernelagent": BaselineStylePreset(
        name="kernelagent",
        label="KernelAgent-style hardware-guided orchestration",
        description=(
            "Hardware-guided branches use real NCU feedback, an AI "
            "strategy planner, and a multi-parent measured beam."
        ),
        evaluation_policy="ncu",
        tir_evidence_policy="hidden",
        selection_policy="measure-all",
        profile_policy="every-round",
        compiled_deduplication=False,
        structural_search_policy="observe",
        strategy_allocation_policy="ai-planned",
        default_agent_workers=1,
        prompt_guidance=(
            "Act as one hardware-guided optimization worker in an orchestrated search.",
            "Ground optimization directions in observed NCU and measured latency evidence.",
            "Implement the assigned branch while keeping it meaningfully distinct "
            "from sibling branches.",
            "Prefer bottleneck-directed source changes over blind parameter sweeps.",
        ),
        reference_basis=(
            "KernelAgent hardware-guided multi-agent workflow",
            "KernelAgent beam/greedy search and NCU feedback",
        ),
        unsupported_capabilities=(
            "specialized profiler, bottleneck, optimizer, and reflector agent roles",
            "KernelAgent's Triton-specific rule base and databases",
            "exact beam scoring, prompts, and worker orchestration",
        ),
    ),
    "kernelevolve": BaselineStylePreset(
        name="kernelevolve",
        label="KernelEvolve-style persistent evolutionary search",
        description=(
            "A persistent candidate graph and shared evidence memory guide open "
            "variation across a measured beam, with NCU as the hardware oracle."
        ),
        evaluation_policy="ncu",
        tir_evidence_policy="hidden",
        selection_policy="measure-all",
        profile_policy="every-round",
        compiled_deduplication=False,
        structural_search_policy="observe",
        strategy_allocation_policy="unconstrained",
        default_agent_workers=1,
        prompt_guidance=(
            "Act as a universal kernel variation operator over a persistent candidate lineage.",
            "Use measured outcomes, failures, and shared lessons as search memory.",
            "Explore distinct implementation families as well as local refinements.",
            "Combine compatible parameter, layout, data-movement, and algorithmic "
            "changes when justified.",
        ),
        reference_basis=(
            "KernelEvolve persistent candidate graph and context memory",
            "KernelEvolve universal operator and hardware evaluator loop",
        ),
        unsupported_capabilities=(
            "external Context Memory and Deep Search knowledge retrieval",
            "the original parent-selection or evolutionary policy",
            "cross-task warm starts, dispatch guards, and production packaging",
        ),
    ),
    "kernelbench": BaselineStylePreset(
        name="kernelbench",
        label="KernelBench-style iterative G+E refinement",
        description=(
            "Best-of-k iterative refinement follows compiler, correctness, and "
            "CUDA Event execution feedback along one incumbent lineage."
        ),
        evaluation_policy="cuda-event",
        tir_evidence_policy="hidden",
        selection_policy="measure-all",
        profile_policy="none",
        compiled_deduplication=False,
        structural_search_policy="off",
        strategy_allocation_policy="unconstrained",
        default_agent_workers=1,
        budget_overrides=(("beam_width", 1),),
        prompt_guidance=(
            "Follow iterative generation-plus-execution-feedback refinement.",
            "Use compiler errors, correctness outcomes, and CUDA Event latency "
            "from prior attempts.",
            "Generate independent best-of-k refinements of the current incumbent.",
            "Do not assume profiler or analytical-model evidence is available.",
        ),
        reference_basis=(
            "KernelBench iterative G+E feedback experiment",
            "KernelBench deterministic correctness and CUDA Event evaluator",
        ),
        unsupported_capabilities=(
            "the KernelBench task corpus and PyTorch ModelNew contract",
            "its exact one-shot, repeated-sampling, and ten-step protocols",
            "CPU precompile and multi-GPU benchmark infrastructure",
        ),
    ),
    "avo": BaselineStylePreset(
        name="avo",
        label="AVO-style committed-lineage optimization",
        description=(
            "Open plan-and-edit variations compete around one committed incumbent; "
            "correctness, latency, and NCU evidence determine non-regressing updates."
        ),
        evaluation_policy="ncu",
        tir_evidence_policy="hidden",
        selection_policy="measure-all",
        profile_policy="every-round",
        compiled_deduplication=False,
        structural_search_policy="off",
        strategy_allocation_policy="unconstrained",
        default_agent_workers=1,
        budget_overrides=(("beam_width", 1),),
        prompt_guidance=(
            "Perform a focused plan-and-edit variation around the committed incumbent.",
            "Use observed correctness, latency, compiler, and NCU feedback from the lineage.",
            "Prefer a coherent optimization hypothesis that can be independently evaluated.",
            "Retain failed attempts as evidence and avoid repeating disproven changes.",
        ),
        reference_basis=(
            "AVO autonomous variation operator",
            "AVO single committed lineage and non-regression gate",
        ),
        unsupported_capabilities=(
            "interactive coding-agent tools inside one variation step",
            "AVO's supervisor, knowledge base, and git workflow",
            "exact acceptance, planning, and stagnation policies",
        ),
    ),
    "tilefoundry": BaselineStylePreset(
        name="tilefoundry",
        label="TileFoundry-style external coding-agent workflow",
        description=(
            "An unconstrained coding-agent-style rewrite loop works against a "
            "fixed semantic contract and CUDA Event evaluator without TileSight or NCU."
        ),
        evaluation_policy="cuda-event",
        tir_evidence_policy="hidden",
        selection_policy="measure-all",
        profile_policy="none",
        compiled_deduplication=False,
        structural_search_policy="observe",
        strategy_allocation_policy="unconstrained",
        default_agent_workers=1,
        budget_overrides=(("beam_width", 1),),
        prompt_guidance=(
            "Act like an external coding agent working against an immutable "
            "semantic and runtime contract.",
            "Freely rewrite, fuse, specialize, change layouts, or reorganize the "
            "kernel when semantics are preserved.",
            "Use target and workload facts plus deterministic correctness and benchmark feedback.",
            "Treat static structure observations as diagnostics, not as measured "
            "profiler evidence.",
        ),
        reference_basis=(
            "TileFoundry external coding-agent workflow",
            "TileFoundry semantic contract and deterministic tool surface",
        ),
        unsupported_capabilities=(
            "authored HIR, runtime twins, real weights, and full-model integration",
            "TileFoundry analyze/schedule tools and backend-specific worktrees",
            "sub-agent tool use, checkpoint validation, and end-to-end decoding",
        ),
    ),
}


def get_baseline_style(name: str) -> BaselineStylePreset:
    """Return one named preset with a useful error for programmatic callers."""

    normalized = str(name).strip().lower()
    try:
        return _PRESETS[normalized]
    except KeyError as error:
        raise ValueError(
            "unknown baseline style %r; choose one of %s"
            % (name, ", ".join(BASELINE_STYLE_NAMES))
        ) from error


def resolve_baseline_style(
    *,
    name: str,
    task: TaskSpec,
    agent_workers: int,
    evaluation_policy: str,
    tir_evidence_policy: str,
    selection_policy: str,
    profile_policy: str,
    compiled_deduplication: bool,
    structural_search_policy: str,
    strategy_allocation_policy: str,
    explicit_fields: FrozenSet[str] = frozenset(),
) -> BaselineStyleResolution:
    """Apply a style preset while preserving explicitly requested CLI overrides."""

    preset = get_baseline_style(name)
    requested = {
        "agent_workers": agent_workers,
        "evaluation_policy": evaluation_policy,
        "tir_evidence_policy": tir_evidence_policy,
        "selection_policy": selection_policy,
        "profile_policy": profile_policy,
        "compiled_deduplication": compiled_deduplication,
        "structural_search_policy": structural_search_policy,
        "strategy_allocation_policy": strategy_allocation_policy,
    }
    canonical = {
        "agent_workers": preset.default_agent_workers,
        "evaluation_policy": preset.evaluation_policy,
        "tir_evidence_policy": preset.tir_evidence_policy,
        "selection_policy": preset.selection_policy,
        "profile_policy": preset.profile_policy,
        "compiled_deduplication": preset.compiled_deduplication,
        "structural_search_policy": preset.structural_search_policy,
        "strategy_allocation_policy": preset.strategy_allocation_policy,
    }

    if preset.name == "native":
        effective = requested
        explicit_overrides = {
            field: {
                "preset": canonical[field],
                "effective": requested[field],
            }
            for field in sorted(explicit_fields)
            if field in requested and requested[field] != canonical[field]
        }
        effective_task = task
    else:
        effective = dict(canonical)
        explicit_overrides = {}
        for field in sorted(explicit_fields):
            if field not in requested:
                continue
            effective[field] = requested[field]
            if requested[field] != canonical[field]:
                explicit_overrides[field] = {
                    "preset": canonical[field],
                    "effective": requested[field],
                }

        budget = replace(task.budget, **dict(preset.budget_overrides))
        evaluator = dict(task.evaluator)
        runtime = dict(evaluator.get("runtime") or {})
        if effective["evaluation_policy"] == "ncu":
            # Preserve the profiler-rich related-system protocol. The native
            # TileSight milestone path uses the smaller targeted metric set.
            runtime["ncu_set"] = "full"
            evaluator["runtime"] = runtime
        metadata = dict(task.metadata)
        metadata["baseline_style_protocol"] = {
            "name": preset.name,
            "label": preset.label,
            "mode": "style-emulation",
            "guidance": list(preset.prompt_guidance),
            "feedback": {
                "evaluation_policy": effective["evaluation_policy"],
                "tir_evidence_policy": effective["tir_evidence_policy"],
                "profile_policy": effective["profile_policy"],
            },
            "search": {
                "beam_width": budget.beam_width,
                "proposals_per_round": budget.proposals_per_round,
                "structural_search_policy": effective[
                    "structural_search_policy"
                ],
                "strategy_allocation_policy": effective[
                    "strategy_allocation_policy"
                ],
            },
        }
        effective_task = replace(
            task,
            budget=budget,
            evaluator=evaluator,
            metadata=metadata,
        )

    return BaselineStyleResolution(
        preset=preset,
        task=effective_task,
        agent_workers=int(effective["agent_workers"]),
        evaluation_policy=str(effective["evaluation_policy"]),
        tir_evidence_policy=str(effective["tir_evidence_policy"]),
        selection_policy=str(effective["selection_policy"]),
        profile_policy=str(effective["profile_policy"]),
        compiled_deduplication=bool(effective["compiled_deduplication"]),
        structural_search_policy=str(effective["structural_search_policy"]),
        strategy_allocation_policy=str(effective["strategy_allocation_policy"]),
        explicit_overrides=explicit_overrides,
        original_budget=dict(task.budget.__dict__),
    )
