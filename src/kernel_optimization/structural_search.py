"""Deterministic structural strategies and parent-aware AST novelty checks."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .schema import CandidateProposal, TaskSpec, source_digest


_TUNING_NAME = re.compile(
    r"(^|_)(block|tile|chunk|warp|thread|stage|vector|split|group|pipeline|"
    r"unroll|swizzle|cluster|cta|mma)(_|$)",
    re.IGNORECASE,
)
_TUNING_KEYWORDS = {
    "block_m",
    "block_n",
    "block_k",
    "num_stages",
    "num_warps",
    "stages",
    "threads",
    "vector_size",
    "vectorize",
}


@dataclass(frozen=True)
class StructuralStrategy:
    """One portfolio lane assigned to a generated source candidate."""

    strategy_id: str
    title: str
    structural_required: bool
    objective: str
    prohibited_shortcut: str
    family_hint: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyAssignment:
    """Stable round-local binding between a response slot and a strategy."""

    slot: str
    ordinal: int
    strategy_id: str
    title: str
    structural_required: bool
    objective: str
    prohibited_shortcut: str
    family_hint: str
    selection_reason: str
    evidence_terms: List[str]
    planning_mode: str = "explore"
    planner_objective: Optional[str] = None
    planner_reported_strategy: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["strategy_slot"] = result.pop("slot")
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrategyAssignment":
        data = dict(value)
        slot = data.pop("strategy_slot", None)
        if slot is None:
            slot = data.pop("slot", "")
        else:
            data.pop("slot", None)
        data["slot"] = str(slot)
        data["ordinal"] = int(data.get("ordinal", 0))
        data["structural_required"] = bool(data.get("structural_required"))
        data["evidence_terms"] = [
            str(item) for item in data.get("evidence_terms", [])
        ]
        return cls(**data)


@dataclass(frozen=True)
class StrategyPlan:
    """Auditable result of fixed, AI-planned, or fallback slot allocation."""

    round_number: int
    policy: str
    source: str
    requested_count: int
    assignments: List[StrategyAssignment]
    raw_plan: Dict[str, Any] = field(default_factory=dict)
    overrides: List[str] = field(default_factory=list)
    fallback_reason: Optional[str] = None
    round_rationale: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for assignment in self.assignments:
            counts[assignment.strategy_id] = counts.get(assignment.strategy_id, 0) + 1
        return {
            "round": self.round_number,
            "policy": self.policy,
            "source": self.source,
            "requested_count": self.requested_count,
            "allocation": [
                {"strategy_id": strategy_id, "count": count}
                for strategy_id, count in counts.items()
            ],
            "assignments": [item.to_dict() for item in self.assignments],
            "raw_plan": dict(self.raw_plan),
            "overrides": list(self.overrides),
            "fallback_reason": self.fallback_reason,
            "round_rationale": self.round_rationale,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StrategyPlan":
        return cls(
            round_number=int(value.get("round", value.get("round_number", 0))),
            policy=str(value.get("policy", "unknown")),
            source=str(value.get("source", "unknown")),
            requested_count=int(value.get("requested_count", 0)),
            assignments=[
                StrategyAssignment.from_dict(item)
                for item in value.get("assignments", [])
                if isinstance(item, Mapping)
            ],
            raw_plan=dict(value.get("raw_plan") or {}),
            overrides=[str(item) for item in value.get("overrides", [])],
            fallback_reason=(
                str(value["fallback_reason"])
                if value.get("fallback_reason") is not None
                else None
            ),
            round_rationale=(
                str(value["round_rationale"])
                if value.get("round_rationale") is not None
                else None
            ),
            confidence=_optional_confidence(value.get("confidence")),
        )


class StructuralStrategyPortfolio:
    """Mix a parameter baseline, open exploration, and evidence-ranked lanes."""

    _KNOWN_STRATEGY_IDS = (
        "memory-layout",
        "data-movement",
        "execution-mapping",
        "pipeline-structure",
        "work-decomposition",
    )

    _EVIDENCE_TERMS = {
        "parameter-tuning": (
            "tile",
            "block",
            "stage",
            "thread",
            "parameter",
            "baseline",
        ),
        "open-structural-exploration": (
            "unknown",
            "new direction",
            "novel",
            "plateau",
            "unexplored",
        ),
        "memory-layout": (
            "bank conflict",
            "shared memory",
            "smem",
            "layout",
            "swizzle",
            "alignment",
            "cache hit",
        ),
        "data-movement": (
            "dram",
            "ddr",
            "bandwidth",
            "memory traffic",
            "global read",
            "global write",
            "l2",
            "copy",
        ),
        "execution-mapping": (
            "occupancy",
            "register",
            "warp",
            "thread",
            "tensor core",
            "cuda core",
            "parallelism",
        ),
        "pipeline-structure": (
            "pipeline",
            "latency hiding",
            "stall",
            "dependency",
            "async",
            "stage",
        ),
        "work-decomposition": (
            "launch",
            "wave",
            "grid",
            "persistent",
            "recompute",
            "fusion",
            "underutilization",
        ),
    }

    _STRATEGIES: Tuple[Tuple[str, str, bool, str, str], ...] = (
        (
            "parameter-tuning",
            "Existing schedule parameter tuning",
            False,
            (
                "Tune the existing tile, block, thread, warp, vector, or stage "
                "parameters without claiming a structural rewrite."
            ),
            "Do not describe numeric-only edits as a layout or algorithm change.",
        ),
        (
            "open-structural-exploration",
            "Open structural exploration",
            True,
            (
                "Discover and implement a semantics-preserving structural "
                "optimization that is not adequately covered by the known strategy "
                "portfolio, or deepen a previously successful discovered strategy."
            ),
            (
                "Do not return a renamed known strategy, metadata-only claim, dead "
                "code, or numeric-only tuning."
            ),
        ),
        (
            "memory-layout",
            "Memory layout and storage organization",
            True,
            (
                "Change the physical shared/register storage organization, such as "
                "padding, indexing, transposition, swizzling, or allocation shape."
            ),
            "Changing only block sizes, threads, or stage counts is invalid.",
        ),
        (
            "data-movement",
            "Data movement and vectorization",
            True,
            (
                "Change how data is cooperatively loaded, stored, prefetched, "
                "vectorized, or moved between global, shared, and fragment storage."
            ),
            "Changing only copy tile extents through function defaults is invalid.",
        ),
        (
            "execution-mapping",
            "Thread, warp, and tensor-core mapping",
            True,
            (
                "Change work ownership or execution mapping across CTAs, warps, "
                "threads, reductions, or tensor-core policies."
            ),
            "Changing only the thread-count default is invalid.",
        ),
        (
            "pipeline-structure",
            "Producer-consumer pipeline structure",
            True,
            (
                "Change operation placement or producer-consumer overlap, including "
                "prefetch placement, prologue/steady-state/epilogue organization, "
                "or staging dataflow."
            ),
            "Changing only num_stages is invalid.",
        ),
        (
            "work-decomposition",
            "Work decomposition and fusion",
            True,
            (
                "Change the decomposition of useful work, for example persistence, "
                "split work, fusion, epilogue organization, or recomputation."
            ),
            "Changing only tile dimensions is invalid.",
        ),
    )

    def plan(
        self,
        task: TaskSpec,
        round_number: int,
        count: int,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> List[StrategyAssignment]:
        if round_number <= 0:
            raise ValueError("round_number must be positive")
        if count < 0:
            raise ValueError("count cannot be negative")
        family = _kernel_family(task)
        strategies = {
            item[0]: StructuralStrategy(
                strategy_id=item[0],
                title=item[1],
                structural_required=item[2],
                objective=item[3],
                prohibited_shortcut=item[4],
                family_hint=_family_hint(family, item[0]),
            )
            for item in self._STRATEGIES
        }
        ranked_known = self._rank_known_strategies(
            strategies,
            evidence or {},
            round_number,
            task.budget.random_seed,
        )
        selected: List[Tuple[StructuralStrategy, str, List[str]]] = []
        if count == 1:
            strategy_id = (
                "parameter-tuning"
                if round_number % 2 == 1
                else "open-structural-exploration"
            )
            reason = (
                "alternating-single-slot-parameter-baseline"
                if strategy_id == "parameter-tuning"
                else "alternating-single-slot-open-exploration"
            )
            selected.append((strategies[strategy_id], reason, []))
        elif count >= 2:
            selected.extend(
                [
                    (
                        strategies["parameter-tuning"],
                        "reserved-parameter-baseline",
                        [],
                    ),
                    (
                        strategies["open-structural-exploration"],
                        "reserved-open-exploration",
                        [],
                    ),
                ]
            )
        known_index = 0
        while len(selected) < count:
            strategy, terms = ranked_known[known_index % len(ranked_known)]
            selected.append(
                (
                    strategy,
                    (
                        "evidence-prioritized"
                        if terms
                        else "rotating-known-strategy-coverage"
                    ),
                    terms,
                )
            )
            known_index += 1

        assignments = []
        for ordinal, (strategy, selection_reason, evidence_terms) in enumerate(
            selected
        ):
            assignments.append(
                StrategyAssignment(
                    slot="r%03d-s%02d-%s"
                    % (round_number, ordinal, strategy.strategy_id),
                    ordinal=ordinal,
                    strategy_id=strategy.strategy_id,
                    title=strategy.title,
                    structural_required=strategy.structural_required,
                    objective=strategy.objective,
                    prohibited_shortcut=strategy.prohibited_shortcut,
                    family_hint=strategy.family_hint,
                    selection_reason=selection_reason,
                    evidence_terms=evidence_terms,
                )
            )
        return assignments

    def strategy_catalog(self, task: TaskSpec) -> List[Dict[str, Any]]:
        """Return the complete strategy vocabulary exposed to the AI planner."""

        family = _kernel_family(task)
        return [
            StructuralStrategy(
                strategy_id=item[0],
                title=item[1],
                structural_required=item[2],
                objective=item[3],
                prohibited_shortcut=item[4],
                family_hint=_family_hint(family, item[0]),
            ).to_dict()
            for item in self._STRATEGIES
        ]

    def fixed_plan(
        self,
        task: TaskSpec,
        round_number: int,
        count: int,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> StrategyPlan:
        """Wrap the legacy deterministic portfolio for reproducible ablations."""

        assignments = self.plan(task, round_number, count, evidence)
        return StrategyPlan(
            round_number=round_number,
            policy="fixed",
            source="fixed-portfolio",
            requested_count=count,
            assignments=assignments,
            round_rationale="Legacy deterministic strategy coverage portfolio.",
        )

    def fallback_plan(
        self,
        task: TaskSpec,
        round_number: int,
        count: int,
        evidence: Optional[Mapping[str, Any]],
        reason: str,
        raw_plan: Optional[Mapping[str, Any]] = None,
    ) -> StrategyPlan:
        """Build a deterministic, strategy-neutral plan after planner failure."""

        assignments, overrides = self._normalize_items(
            task=task,
            round_number=round_number,
            count=count,
            raw_items=[],
            evidence=evidence or {},
        )
        return StrategyPlan(
            round_number=round_number,
            policy="ai-planned",
            source="deterministic-fallback",
            requested_count=count,
            assignments=assignments,
            raw_plan=dict(raw_plan or {}),
            overrides=overrides,
            fallback_reason=reason,
            round_rationale=(
                "The hosted planner was unavailable or invalid, so the controller "
                "used evidence-ranked balanced coverage without reserving any "
                "particular strategy."
            ),
        )

    def normalize_ai_plan(
        self,
        task: TaskSpec,
        round_number: int,
        count: int,
        raw_plan: Mapping[str, Any],
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> StrategyPlan:
        """Validate an AI allocation and enforce only strategy-neutral bounds."""

        if not isinstance(raw_plan, Mapping):
            return self.fallback_plan(
                task,
                round_number,
                count,
                evidence,
                "planner response is not a JSON object",
            )
        raw_items = raw_plan.get("allocation")
        if not isinstance(raw_items, list) or not raw_items:
            return self.fallback_plan(
                task,
                round_number,
                count,
                evidence,
                "planner response has no non-empty allocation list",
                raw_plan,
            )
        usable_items = [
            item
            for item in raw_items
            if isinstance(item, Mapping)
            and str(item.get("strategy_id") or "").strip()
            and str(item.get("reason") or "").strip()
            and isinstance(item.get("count"), int)
            and not isinstance(item.get("count"), bool)
            and int(item.get("count")) > 0
        ]
        if not usable_items:
            return self.fallback_plan(
                task,
                round_number,
                count,
                evidence,
                "planner allocation contained no usable entries",
                raw_plan,
            )
        assignments, overrides = self._normalize_items(
            task=task,
            round_number=round_number,
            count=count,
            raw_items=raw_items,
            evidence=evidence or {},
        )
        return StrategyPlan(
            round_number=round_number,
            policy="ai-planned",
            source="hosted-ai-planner",
            requested_count=count,
            assignments=assignments,
            raw_plan=dict(raw_plan),
            overrides=overrides,
            round_rationale=_bounded_text(raw_plan.get("round_rationale"), 1000),
            confidence=_optional_confidence(raw_plan.get("confidence")),
        )

    def _normalize_items(
        self,
        *,
        task: TaskSpec,
        round_number: int,
        count: int,
        raw_items: Sequence[Any],
        evidence: Mapping[str, Any],
    ) -> Tuple[List[StrategyAssignment], List[str]]:
        if round_number <= 0:
            raise ValueError("round_number must be positive")
        if count < 0:
            raise ValueError("count cannot be negative")
        if count == 0:
            return [], []

        strategies = {
            item["strategy_id"]: StructuralStrategy(**item)
            for item in self.strategy_catalog(task)
        }
        cap = max(1, int(math.ceil(2.0 * count / 3.0)))
        expanded: List[Dict[str, Any]] = []
        overrides: List[str] = []
        for index, raw in enumerate(raw_items):
            if not isinstance(raw, Mapping):
                overrides.append("ignored allocation item %d because it is not an object" % index)
                continue
            reported_id = _bounded_text(raw.get("strategy_id"), 160)
            reason = _bounded_text(raw.get("reason"), 600)
            if not reported_id or not reason:
                overrides.append(
                    "ignored allocation item %d because strategy_id or reason is missing"
                    % index
                )
                continue
            requested = raw.get("count", 0)
            if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
                overrides.append(
                    "ignored allocation item %d because count is not a positive integer"
                    % index
                )
                continue
            strategy_id = reported_id
            planner_reported_strategy = None
            if strategy_id not in strategies:
                planner_reported_strategy = strategy_id
                strategy_id = "open-structural-exploration"
                overrides.append(
                    "mapped unknown planner strategy %r to open-structural-exploration"
                    % reported_id
                )
            mode = str(raw.get("mode") or "explore").strip().lower()
            if mode not in {"exploit", "explore"}:
                overrides.append(
                    "normalized invalid mode %r for strategy %s to explore"
                    % (mode, reported_id)
                )
                mode = "explore"
            evidence_value = raw.get("evidence") or []
            if isinstance(evidence_value, str):
                evidence_terms = [evidence_value[:200]]
            elif isinstance(evidence_value, Sequence):
                evidence_terms = [
                    str(item)[:200] for item in evidence_value[:8] if str(item).strip()
                ]
            else:
                evidence_terms = []
            objective = _bounded_text(raw.get("objective"), 600)
            for _ in range(requested):
                expanded.append(
                    {
                        "strategy_id": strategy_id,
                        "reported_strategy": planner_reported_strategy,
                        "reason": reason,
                        "mode": mode,
                        "evidence_terms": evidence_terms,
                        "objective": objective,
                    }
                )

        accepted: List[Dict[str, Any]] = []
        strategy_counts: Counter[str] = Counter()
        for item in expanded:
            strategy_id = item["strategy_id"]
            if len(accepted) >= count:
                overrides.append("trimmed planner allocation to the requested candidate count")
                break
            if strategy_counts[strategy_id] >= cap:
                overrides.append(
                    "capped strategy %s at %d of %d slots"
                    % (strategy_id, cap, count)
                )
                continue
            accepted.append(item)
            strategy_counts[strategy_id] += 1

        fallback_order = self._strategy_neutral_order(
            strategies, evidence, round_number, task.budget.random_seed
        )
        while len(accepted) < count:
            strategy = next(
                (
                    item
                    for item in fallback_order
                    if strategy_counts[item.strategy_id] < cap
                ),
                fallback_order[len(accepted) % len(fallback_order)],
            )
            accepted.append(
                self._fallback_item(strategy, evidence, "controller-count-fill")
            )
            strategy_counts[strategy.strategy_id] += 1
            overrides.append(
                "filled an unallocated slot with evidence-ranked strategy %s"
                % strategy.strategy_id
            )

        minimum_directions = min(2, count)
        if len(strategy_counts) < minimum_directions and accepted:
            replacement = next(
                (item for item in fallback_order if item.strategy_id not in strategy_counts),
                None,
            )
            duplicate_index = next(
                (
                    index
                    for index in range(len(accepted) - 1, -1, -1)
                    if strategy_counts[accepted[index]["strategy_id"]] > 1
                ),
                None,
            )
            if replacement is not None and duplicate_index is not None:
                previous = accepted[duplicate_index]["strategy_id"]
                strategy_counts[previous] -= 1
                accepted[duplicate_index] = self._fallback_item(
                    replacement, evidence, "controller-diversity-bound"
                )
                strategy_counts[replacement.strategy_id] += 1
                overrides.append(
                    "replaced one %s slot with %s to preserve two search directions"
                    % (previous, replacement.strategy_id)
                )

        assignments = []
        for ordinal, item in enumerate(accepted[:count]):
            strategy = strategies[item["strategy_id"]]
            objective = item.get("objective") or strategy.objective
            reported = item.get("reported_strategy")
            if reported:
                objective = (
                    "%s Planner-proposed open direction: %s."
                    % (objective, reported)
                )
            assignments.append(
                StrategyAssignment(
                    slot="r%03d-s%02d-%s"
                    % (round_number, ordinal, strategy.strategy_id),
                    ordinal=ordinal,
                    strategy_id=strategy.strategy_id,
                    title=strategy.title,
                    structural_required=strategy.structural_required,
                    objective=objective,
                    prohibited_shortcut=strategy.prohibited_shortcut,
                    family_hint=strategy.family_hint,
                    selection_reason=item["reason"],
                    evidence_terms=list(item["evidence_terms"]),
                    planning_mode=item["mode"],
                    planner_objective=item.get("objective"),
                    planner_reported_strategy=reported,
                )
            )
        return assignments, _unique_strings(overrides)

    def _strategy_neutral_order(
        self,
        strategies: Mapping[str, StructuralStrategy],
        evidence: Mapping[str, Any],
        round_number: int,
        random_seed: int,
    ) -> List[StructuralStrategy]:
        text = _evidence_text(evidence)
        identifiers = list(strategies)
        rotation = (round_number - 1 + random_seed) % len(identifiers)
        outcomes = evidence.get("strategy_outcomes")
        outcome_map = outcomes if isinstance(outcomes, Mapping) else {}
        ranked = []
        for index, strategy_id in enumerate(identifiers):
            terms = self._EVIDENCE_TERMS.get(strategy_id, ())
            evidence_score = sum(term in text for term in terms)
            outcome = outcome_map.get(strategy_id)
            improvement = 0.0
            if isinstance(outcome, Mapping):
                value = outcome.get("best_relative_improvement_vs_parent")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    improvement = float(value)
            tie_break = (index - rotation) % len(identifiers)
            ranked.append((strategy_id, improvement, evidence_score, tie_break))
        ranked.sort(key=lambda item: (-item[1], -item[2], item[3], item[0]))
        return [strategies[item[0]] for item in ranked]

    def _fallback_item(
        self,
        strategy: StructuralStrategy,
        evidence: Mapping[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        text = _evidence_text(evidence)
        terms = sorted(
            term
            for term in self._EVIDENCE_TERMS.get(strategy.strategy_id, ())
            if term in text
        )
        return {
            "strategy_id": strategy.strategy_id,
            "reported_strategy": None,
            "reason": reason,
            "mode": "explore",
            "evidence_terms": terms,
            "objective": None,
        }

    def _rank_known_strategies(
        self,
        strategies: Mapping[str, StructuralStrategy],
        evidence: Mapping[str, Any],
        round_number: int,
        random_seed: int,
    ) -> List[Tuple[StructuralStrategy, List[str]]]:
        text = _evidence_text(evidence)
        rotation = (round_number - 1 + random_seed) % len(
            self._KNOWN_STRATEGY_IDS
        )
        ranked = []
        for index, strategy_id in enumerate(self._KNOWN_STRATEGY_IDS):
            terms = sorted(
                {
                    term
                    for term in self._EVIDENCE_TERMS[strategy_id]
                    if term in text
                }
            )
            tie_break = (index - rotation) % len(self._KNOWN_STRATEGY_IDS)
            ranked.append((strategies[strategy_id], terms, tie_break))
        ranked.sort(key=lambda item: (-len(item[1]), item[2], item[0].strategy_id))
        return [(item[0], item[1]) for item in ranked]


def bind_strategy_assignments(
    proposals: Sequence[CandidateProposal],
    assignments: Sequence[StrategyAssignment],
) -> List[CandidateProposal]:
    """Bind model responses to requested slots while preserving reported metadata."""

    if not assignments:
        return list(proposals)
    by_slot = {item.slot: item for item in assignments}
    unused = [item.slot for item in assignments]
    bound = []
    for proposal in proposals:
        metadata = dict(proposal.metadata)
        reported_slot = str(metadata.get("strategy_slot") or "").strip() or None
        reported_id = str(
            metadata.get("strategy_id") or metadata.get("strategy") or ""
        ).strip() or None
        if reported_slot in by_slot and reported_slot in unused:
            assignment = by_slot[reported_slot]
            unused.remove(reported_slot)
            binding = "reported-slot"
        elif unused:
            selected_slot = unused.pop(0)
            assignment = by_slot[selected_slot]
            binding = "response-order"
        else:
            break
        related_value = metadata.get("related_existing_strategies") or []
        if isinstance(related_value, str):
            related = [related_value]
        elif isinstance(related_value, Sequence):
            related = [str(item) for item in related_value]
        else:
            related = []
        discovered = str(metadata.get("discovered_strategy") or "").strip()[:160]
        metadata.update(
            {
                "reported_strategy_slot": reported_slot,
                "reported_strategy_id": reported_id,
                "strategy_slot": assignment.slot,
                "strategy_id": assignment.strategy_id,
                "requested_strategy_title": assignment.title,
                "structural_required": assignment.structural_required,
                "strategy_binding": binding,
                "strategy_selection_reason": assignment.selection_reason,
                "strategy_evidence_terms": list(assignment.evidence_terms),
                "strategy_planning_mode": assignment.planning_mode,
                "strategy_planner_objective": assignment.planner_objective,
                "strategy_planner_reported_strategy": (
                    assignment.planner_reported_strategy
                ),
                "discovered_strategy": discovered or None,
                "related_existing_strategies": [item[:80] for item in related[:8]],
            }
        )
        bound.append(
            CandidateProposal(
                hypothesis=proposal.hypothesis,
                source_code=proposal.source_code,
                expected_effect=proposal.expected_effect,
                metadata=metadata,
            )
        )
    return bound


@dataclass(frozen=True)
class SourceNoveltyReport:
    """AST-level delta between one candidate and its measured-beam parent."""

    classification: str
    structural_change: bool
    meaningful_change: bool
    changed_tuning_parameters: List[str]
    added_call_targets: List[str]
    removed_call_targets: List[str]
    changed_control_flow: bool
    changed_allocations: bool
    structural_signals: List[str]
    parent_source_sha256: str
    candidate_source_sha256: str
    parent_ast_sha256: str
    candidate_ast_sha256: str
    parent_structural_ast_sha256: str
    candidate_structural_ast_sha256: str
    parent_ast_nodes: int
    candidate_ast_nodes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SourceNoveltyAnalyzer:
    """Classify parameter-only edits separately from structural source changes."""

    def analyze(
        self,
        parent_source: str,
        candidate_source: str,
        entrypoint: str,
    ) -> SourceNoveltyReport:
        parent_tree = ast.parse(parent_source)
        candidate_tree = ast.parse(candidate_source)
        parent_tuning = _extract_tuning_values(parent_tree, entrypoint)
        candidate_tuning = _extract_tuning_values(candidate_tree, entrypoint)
        tuning_changes = _mapping_changes(parent_tuning, candidate_tuning)

        parent_ast = _canonical_dump(parent_tree, entrypoint, normalize_tuning=False)
        candidate_ast = _canonical_dump(
            candidate_tree, entrypoint, normalize_tuning=False
        )
        parent_structural = _canonical_dump(
            parent_tree, entrypoint, normalize_tuning=True
        )
        candidate_structural = _canonical_dump(
            candidate_tree, entrypoint, normalize_tuning=True
        )

        exact_same = source_digest(parent_source) == source_digest(candidate_source)
        if exact_same:
            classification = "identical"
        elif parent_structural != candidate_structural:
            classification = "structural"
        elif tuning_changes:
            classification = "parameter-only"
        else:
            classification = "format-or-rename-only"

        parent_calls = _call_targets(parent_tree)
        candidate_calls = _call_targets(candidate_tree)
        added_calls, removed_calls = _counter_delta(parent_calls, candidate_calls)
        parent_control = _control_flow(parent_tree)
        candidate_control = _control_flow(candidate_tree)
        parent_allocations = _allocation_signatures(parent_tree, entrypoint)
        candidate_allocations = _allocation_signatures(candidate_tree, entrypoint)
        signals = []
        if added_calls or removed_calls:
            signals.append("call-graph")
        if parent_control != candidate_control:
            signals.append("control-flow")
        if parent_allocations != candidate_allocations:
            signals.append("allocation-or-layout")
        if classification == "structural" and not signals:
            signals.append("dataflow-or-expression")

        return SourceNoveltyReport(
            classification=classification,
            structural_change=classification == "structural",
            meaningful_change=classification in {"parameter-only", "structural"},
            changed_tuning_parameters=tuning_changes,
            added_call_targets=added_calls,
            removed_call_targets=removed_calls,
            changed_control_flow=parent_control != candidate_control,
            changed_allocations=parent_allocations != candidate_allocations,
            structural_signals=signals,
            parent_source_sha256=source_digest(parent_source),
            candidate_source_sha256=source_digest(candidate_source),
            parent_ast_sha256=_digest(parent_ast),
            candidate_ast_sha256=_digest(candidate_ast),
            parent_structural_ast_sha256=_digest(parent_structural),
            candidate_structural_ast_sha256=_digest(candidate_structural),
            parent_ast_nodes=sum(1 for _ in ast.walk(parent_tree)),
            candidate_ast_nodes=sum(1 for _ in ast.walk(candidate_tree)),
        )


def _kernel_family(task: TaskSpec) -> str:
    explicit = str(task.metadata.get("kernel_family") or "").lower()
    text = " ".join((explicit, task.task_id.lower(), task.description.lower()))
    if "attention" in text:
        return "flash-attention"
    if "matmul" in text or "gemm" in text:
        return "matmul"
    if "rmsnorm" in text or "rms_norm" in text or "rms norm" in text:
        return "rmsnorm"
    if "conv2d" in text or "convolution" in text:
        return "conv2d"
    return "generic"


def _family_hint(family: str, strategy_id: str) -> str:
    hints = {
        "matmul": {
            "parameter-tuning": (
                "Explore compatible CTA tiles, K tiles, thread counts, and stage depths."
            ),
            "open-structural-exploration": (
                "Look beyond the named lanes for a concrete GEMM schedule, dataflow, "
                "or decomposition change supported by APIs already available in the "
                "source environment."
            ),
            "memory-layout": (
                "Consider A/B shared-memory padding or layout changes that preserve "
                "the GEMM contract and use only available TileLang APIs."
            ),
            "data-movement": (
                "Consider cooperative/vectorized A/B loads, copy placement, and the "
                "epilogue write path."
            ),
            "execution-mapping": (
                "Consider warp ownership, tensor-core policy, and CTA-to-output mapping."
            ),
            "pipeline-structure": (
                "Consider moving or restructuring A/B prefetch and GEMM staging, not "
                "merely changing num_stages."
            ),
            "work-decomposition": (
                "Consider persistent output tiles, split work, or epilogue fusion when "
                "expressible without changing the entrypoint."
            ),
        },
        "flash-attention": {
            "parameter-tuning": (
                "Explore compatible query/key tiles, thread counts, and stage depths."
            ),
            "open-structural-exploration": (
                "Look beyond the named lanes for a concrete attention dataflow, "
                "online-reduction, or decomposition change supported by APIs already "
                "available in the source environment."
            ),
            "memory-layout": (
                "Consider Q/K/V or score shared/fragment layouts, padding, and reduction "
                "storage organization."
            ),
            "data-movement": (
                "Consider Q/K/V staging, vectorized copies, and output writeback traffic."
            ),
            "execution-mapping": (
                "Consider row ownership, warp mapping, reduction mapping, and GEMM policy."
            ),
            "pipeline-structure": (
                "Consider K/V prefetch placement and overlap around online softmax, not "
                "merely changing num_stages."
            ),
            "work-decomposition": (
                "Consider online-softmax organization, recomputation, fusion, or persistent "
                "query tiles while preserving exact semantics."
            ),
        },
        "rmsnorm": {
            "parameter-tuning": (
                "Explore compatible row tiles, hidden-dimension chunks, thread counts, "
                "and vector widths."
            ),
            "open-structural-exploration": (
                "Look beyond the named lanes for a concrete RMSNorm reduction, dataflow, "
                "or fusion change supported by APIs already available in the source "
                "environment."
            ),
            "memory-layout": (
                "Consider fragment/shared-memory organization, padding, and intermediate "
                "reduction storage while preserving the normalization contract."
            ),
            "data-movement": (
                "Consider coalesced or vectorized X, weight, and output accesses and avoid "
                "redundant rereads across reduction and normalization passes."
            ),
            "execution-mapping": (
                "Consider how rows and hidden-dimension reduction chunks map across CTAs, "
                "warps, and threads."
            ),
            "pipeline-structure": (
                "Consider overlap between reduction chunks, weight loads, and output "
                "production rather than only changing a tile parameter."
            ),
            "work-decomposition": (
                "Consider one-pass versus two-pass reduction, split reductions, or fused "
                "epilogues while preserving exact RMSNorm semantics."
            ),
        },
        "conv2d": {
            "parameter-tuning": (
                "Explore compatible CTA M/N/K tiles, thread counts, and stage depths for "
                "the convolution shape."
            ),
            "open-structural-exploration": (
                "Look beyond the named lanes for a direct-convolution, implicit-GEMM, "
                "dataflow, or decomposition change supported by APIs already available "
                "in the source environment."
            ),
            "memory-layout": (
                "Consider NHWC/HWIO access order, im2col/shared-memory organization, "
                "padding, and swizzles while preserving the convolution contract."
            ),
            "data-movement": (
                "Consider cooperative or vectorized activation/filter staging, im2col "
                "traffic, reuse, and the output write path."
            ),
            "execution-mapping": (
                "Consider CTA and warp ownership of output spatial positions and channels, "
                "including tensor-core mapping."
            ),
            "pipeline-structure": (
                "Consider overlap among im2col generation, filter staging, and GEMM "
                "execution rather than merely changing num_stages."
            ),
            "work-decomposition": (
                "Consider spatial/channel decomposition, direct versus implicit-GEMM "
                "organization, fusion, or persistent output tiles without changing semantics."
            ),
        },
    }
    return hints.get(family, {}).get(
        strategy_id,
        "Apply this strategy only with APIs already visible in the source or task environment.",
    )


def _canonical_dump(
    tree: ast.AST, entrypoint: str, *, normalize_tuning: bool
) -> str:
    # Parse again so transformations never mutate the tree used for diagnostics.
    transformed = ast.parse(ast.unparse(tree))
    transformed = _DocstringStripper().visit(transformed)
    if normalize_tuning:
        transformed = _TuningNormalizer(entrypoint).visit(transformed)
    transformed = _AlphaCanonicalizer(entrypoint, _imported_names(transformed)).visit(
        transformed
    )
    ast.fix_missing_locations(transformed)
    return ast.dump(transformed, annotate_fields=True, include_attributes=False)


def _evidence_text(value: Any) -> str:
    """Flatten bounded scalar evidence into deterministic matching text."""

    parts: List[str] = []

    def visit(item: Any, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(item, Mapping):
            for key in sorted(item, key=lambda current: str(current)):
                parts.append(str(key).lower())
                visit(item[key], depth + 1)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for child in list(item)[:64]:
                visit(child, depth + 1)
        elif item is not None:
            parts.append(str(item).lower())

    visit(value, 0)
    return " ".join(parts)


class _DocstringStripper(ast.NodeTransformer):
    def _strip(self, node: Any) -> Any:
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = body[1:]
        return node

    visit_Module = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip


class _TuningNormalizer(ast.NodeTransformer):
    def __init__(self, entrypoint: str) -> None:
        self.entrypoint = entrypoint

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        if node.name == self.entrypoint:
            node.args.defaults = [_tuning_sentinel() for _ in node.args.defaults]
            node.args.kw_defaults = [
                _tuning_sentinel() if value is not None else None
                for value in node.args.kw_defaults
            ]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if any(_is_tuning_target(target) for target in node.targets) and _is_literal(
            node.value
        ):
            node.value = _tuning_sentinel()
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        self.generic_visit(node)
        if (
            node.value is not None
            and _is_tuning_target(node.target)
            and _is_literal(node.value)
        ):
            node.value = _tuning_sentinel()
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        for keyword in node.keywords:
            if keyword.arg in _TUNING_KEYWORDS:
                keyword.value = _tuning_sentinel()
        return node


class _AlphaCanonicalizer(ast.NodeTransformer):
    """Canonicalize local identifiers while preserving API and attribute names."""

    def __init__(self, entrypoint: str, preserved_names: Sequence[str]) -> None:
        self.entrypoint = entrypoint
        self.preserved = set(preserved_names)
        self.mapping: Dict[str, str] = {}

    def _name(self, value: str) -> str:
        if value == self.entrypoint or value in self.preserved:
            return value
        if value not in self.mapping:
            self.mapping[value] = "_v%d" % len(self.mapping)
        return self.mapping[value]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name != self.entrypoint:
            node.name = self._name(node.name)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        if node.name != self.entrypoint:
            node.name = self._name(node.name)
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._name(node.arg)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._name(node.id)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Name):
            # Builtins and direct callable APIs remain semantically visible.
            node.args = [self.visit(item) for item in node.args]
            node.keywords = [self.visit(item) for item in node.keywords]
            return node
        return self.generic_visit(node)


def _extract_tuning_values(tree: ast.AST, entrypoint: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entrypoint:
            positional = list(node.args.posonlyargs) + list(node.args.args)
            defaults = list(node.args.defaults)
            for argument, value in zip(positional[-len(defaults) :], defaults):
                result["default:%s" % argument.arg] = _expression(value)
            for argument, value in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if value is not None:
                    result["default:%s" % argument.arg] = _expression(value)

    assignment_counts: Counter[str] = Counter()
    call_counts: Counter[Tuple[str, str]] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                name = _target_name(target)
                if name and _TUNING_NAME.search(name) and value is not None and _is_literal(value):
                    index = assignment_counts[name]
                    assignment_counts[name] += 1
                    result["assignment:%s#%d" % (name, index)] = _expression(value)
        if isinstance(node, ast.Call):
            target = _call_target(node.func)
            for keyword in node.keywords:
                if keyword.arg in _TUNING_KEYWORDS:
                    key = (target, str(keyword.arg))
                    index = call_counts[key]
                    call_counts[key] += 1
                    result["call:%s.%s#%d" % (target, keyword.arg, index)] = _expression(
                        keyword.value
                    )
    return result


def _mapping_changes(parent: Mapping[str, str], candidate: Mapping[str, str]) -> List[str]:
    changes = []
    for key in sorted(set(parent) | set(candidate)):
        before = parent.get(key, "<missing>")
        after = candidate.get(key, "<missing>")
        if before != after:
            changes.append("%s: %s -> %s" % (key, before, after))
    return changes


def _call_targets(tree: ast.AST) -> Counter[str]:
    return Counter(
        _call_target(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    )


def _call_target(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_target(node.value)
        return "%s.%s" % (prefix, node.attr) if prefix else node.attr
    return type(node).__name__


def _counter_delta(
    parent: Counter[str], candidate: Counter[str]
) -> Tuple[List[str], List[str]]:
    added = _expand_counter(candidate - parent)
    removed = _expand_counter(parent - candidate)
    return added, removed


def _expand_counter(counter: Counter[str]) -> List[str]:
    values = []
    for name in sorted(counter):
        count = counter[name]
        values.append(name if count == 1 else "%s x%d" % (name, count))
    return values[:24]


def _control_flow(tree: ast.AST) -> Counter[str]:
    kinds = (
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.If,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    return Counter(type(node).__name__ for node in ast.walk(tree) if isinstance(node, kinds))


def _allocation_signatures(tree: ast.AST, entrypoint: str) -> Counter[str]:
    canonical = ast.parse(ast.unparse(tree))
    canonical = _TuningNormalizer(entrypoint).visit(canonical)
    signatures = Counter()
    for node in ast.walk(canonical):
        if not isinstance(node, ast.Call):
            continue
        target = _call_target(node.func)
        if "alloc" not in target.lower():
            continue
        signatures[
            "%s(%s)" % (
                target,
                ",".join(
                    [ast.dump(item, include_attributes=False) for item in node.args]
                    + [
                        "%s=%s"
                        % (item.arg, ast.dump(item.value, include_attributes=False))
                        for item in node.keywords
                    ]
                ),
            )
        ] += 1
    return signatures


def _target_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_tuning_target(node: ast.AST) -> bool:
    name = _target_name(node)
    return bool(name and _TUNING_NAME.search(name))


def _is_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    return isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant)


def _expression(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node, include_attributes=False)


def _tuning_sentinel() -> ast.Constant:
    return ast.Constant(value="__KERNEL_OPT_TUNING_VALUE__")


def _imported_names(tree: ast.AST) -> List[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.extend(alias.asname or alias.name for alias in node.names)
    return names


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bounded_text(value: Any, limit: int) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] if text else None


def _optional_confidence(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def _unique_strings(values: Sequence[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
