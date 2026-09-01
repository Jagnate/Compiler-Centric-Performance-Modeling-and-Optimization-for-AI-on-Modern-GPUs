"""Structural strategy portfolio and AST novelty regression tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kernel_optimization.archive import ArtifactStore
from kernel_optimization.controller import OptimizationController
from kernel_optimization.cli import build_parser
from kernel_optimization.prompts import build_optimization_prompt
from kernel_optimization.schema import (
    BudgetConfig,
    Candidate,
    CandidateProposal,
    CandidateRecord,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    TIRAnalysis,
    TaskSpec,
)
from kernel_optimization.structural_search import (
    SourceNoveltyAnalyzer,
    StructuralStrategyPortfolio,
    bind_strategy_assignments,
)


PARENT = '''"""Seed."""
from tilelang import language as T

def make_kernel(block_m: int = 64, block_k: int = 32, num_stages: int = 2):
    @T.prim_func
    def main(a):
        shared = T.alloc_shared((block_m, block_k), "float16")
        for ko in T.Pipelined(4, num_stages=num_stages):
            T.copy(a[ko * block_k], shared)
        return shared
    return main
'''

PARAMETER_ONLY = PARENT.replace("block_m: int = 64", "block_m: int = 128").replace(
    "num_stages: int = 2", "num_stages: int = 3"
)

STRUCTURAL = PARENT.replace(
    "T.alloc_shared((block_m, block_k), \"float16\")",
    "T.alloc_shared((block_m, block_k + 1), \"float16\")",
)

RENAME_ONLY = (
    PARENT.replace('"""Seed."""', '"""Different prose."""')
    .replace("shared =", "staged_tile =")
    .replace(", shared)", ", staged_tile)")
    .replace("return shared", "return staged_tile")
)


class StructuralSearchTests(unittest.TestCase):
    def test_cli_keeps_tilesight_structural_search_defaults(self) -> None:
        args = build_parser().parse_args(["--source", "kernel.py", "--task", "task.json"])
        self.assertEqual(args.structural_search_policy, "enforce")
        self.assertEqual(args.strategy_allocation_policy, "ai-planned")

    def test_hardware_adaptive_plan_exploits_real_cuda_event_reward(self) -> None:
        task = _task("hardware-adaptive", proposals=6)
        plan = StructuralStrategyPortfolio().hardware_adaptive_plan(
            task,
            round_number=3,
            count=6,
            evidence={
                "strategy_outcomes": {
                    "data-movement": {
                        "generated": 4,
                        "measured_correct": 4,
                        "measured_improvements": 3,
                        "best_relative_improvement_vs_parent": 0.2,
                        "mean_relative_improvement_vs_parent": 0.1,
                    },
                    "memory-layout": {
                        "generated": 3,
                        "measured_correct": 3,
                        "measured_improvements": 0,
                        "best_relative_improvement_vs_parent": -0.02,
                        "mean_relative_improvement_vs_parent": -0.05,
                    },
                }
            },
        )

        identifiers = [item.strategy_id for item in plan.assignments]
        self.assertEqual(len(identifiers), 6)
        self.assertEqual(identifiers[0], "unrestricted-measured-search")
        self.assertGreater(
            identifiers.count("data-movement"),
            identifiers.count("memory-layout"),
        )
        self.assertEqual(plan.source, "measured-reward-ucb")

    def test_hardware_adaptive_search_batches_a_multi_parent_archive(self) -> None:
        source = "VALUE = 0\n\ndef kernel(x):\n    return x\n"
        task = _task(
            "batched-archive",
            proposals=2,
            entrypoint="kernel",
            rounds=2,
            beam_width=2,
        )
        generator = _BatchedMeasuredGenerator()
        backend = _MeasuredTIRBackend()
        with tempfile.TemporaryDirectory(prefix="batched-archive-") as directory:
            controller = OptimizationController(
                task=task,
                source_code=source,
                source_name="kernel.py",
                generator=generator,
                backend=backend,
                store=ArtifactStore(Path(directory)),
                evaluation_policy="cuda-event",
                tir_evidence_policy="visible",
                selection_policy="measure-all",
                profile_policy="none",
                structural_search_policy="observe",
                strategy_allocation_policy="hardware-adaptive",
            )
            summary = controller.run()

        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(len(generator.calls[0]["parent_archive"]), 0)
        self.assertEqual(len(generator.calls[1]["parent_archive"]), 1)
        self.assertEqual(summary.generator_calls, 2)
        self.assertEqual(summary.planner_calls, 0)
        self.assertEqual(summary.measured_candidates, 5)
        self.assertEqual(backend.model_calls, 0)
        self.assertEqual(backend.tir_calls, 3)

    def test_ai_plan_can_allocate_zero_parameter_slots(self) -> None:
        task = _task("zero-parameter", proposals=6)
        plan = StructuralStrategyPortfolio().normalize_ai_plan(
            task,
            round_number=1,
            count=6,
            raw_plan={
                "allocation": [
                    {
                        "strategy_id": "data-movement",
                        "count": 4,
                        "mode": "exploit",
                        "reason": "Observed memory traffic dominates.",
                        "evidence": ["profile.ddr_util"],
                    },
                    {
                        "strategy_id": "pipeline-structure",
                        "count": 2,
                        "mode": "explore",
                        "reason": "Overlap is still uncertain.",
                        "evidence": ["model.diagnostics"],
                    },
                ],
                "round_rationale": "Focus on traffic and overlap.",
                "confidence": 0.8,
            },
        )

        identifiers = [item.strategy_id for item in plan.assignments]
        self.assertEqual(len(identifiers), 6)
        self.assertNotIn("parameter-tuning", identifiers)
        self.assertEqual(identifiers.count("data-movement"), 4)
        self.assertEqual(identifiers.count("pipeline-structure"), 2)
        self.assertEqual(plan.source, "hosted-ai-planner")

    def test_ai_plan_enforces_generic_diversity_and_share_bounds(self) -> None:
        task = _task("generic-bounds", proposals=6)
        plan = StructuralStrategyPortfolio().normalize_ai_plan(
            task,
            round_number=1,
            count=6,
            raw_plan={
                "allocation": [
                    {
                        "strategy_id": "data-movement",
                        "count": 12,
                        "mode": "exploit",
                        "reason": "The planner wants a focused round.",
                        "evidence": ["profile.ddr_util"],
                    }
                ]
            },
        )

        identifiers = [item.strategy_id for item in plan.assignments]
        self.assertEqual(len(identifiers), 6)
        self.assertGreaterEqual(len(set(identifiers)), 2)
        self.assertLessEqual(identifiers.count("data-movement"), 4)
        self.assertTrue(any("capped strategy" in item for item in plan.overrides))

    def test_portfolio_reserves_open_search_and_rotates_known_lanes(self) -> None:
        task = _task("matmul-portfolio", proposals=6)
        assignments = StructuralStrategyPortfolio().plan(task, 1, 6)

        self.assertEqual(
            [item.strategy_id for item in assignments],
            [
                "parameter-tuning",
                "open-structural-exploration",
                "memory-layout",
                "data-movement",
                "execution-mapping",
                "pipeline-structure",
            ],
        )
        self.assertFalse(assignments[0].structural_required)
        self.assertTrue(all(item.structural_required for item in assignments[1:]))
        self.assertIn("GEMM", assignments[1].family_hint)
        self.assertEqual(assignments[1].selection_reason, "reserved-open-exploration")
        next_round_ids = {
            item.strategy_id
            for item in StructuralStrategyPortfolio().plan(task, 2, 6)
        }
        self.assertIn("work-decomposition", next_round_ids)
        self.assertNotIn("memory-layout", next_round_ids)

    def test_small_portfolio_keeps_parameter_and_open_slots(self) -> None:
        task = _task("rotation", proposals=2)
        portfolio = StructuralStrategyPortfolio()
        self.assertEqual(
            [item.strategy_id for item in portfolio.plan(task, 2, 2)],
            ["parameter-tuning", "open-structural-exploration"],
        )

    def test_known_lanes_are_ranked_from_current_bottleneck_evidence(self) -> None:
        task = _task("evidence-ranking", proposals=6)
        assignments = StructuralStrategyPortfolio().plan(
            task,
            1,
            6,
            evidence={
                "diagnosis": {
                    "summary": (
                        "Low occupancy from register pressure limits tensor core "
                        "parallelism."
                    )
                }
            },
        )

        self.assertEqual(assignments[2].strategy_id, "execution-mapping")
        self.assertEqual(assignments[2].selection_reason, "evidence-prioritized")
        self.assertIn("occupancy", assignments[2].evidence_terms)

    def test_response_order_binds_missing_strategy_metadata(self) -> None:
        task = _task("binding", proposals=2)
        assignments = StructuralStrategyPortfolio().plan(task, 1, 2)
        proposals = [
            CandidateProposal("First.", PARAMETER_ONLY),
            CandidateProposal("Second.", STRUCTURAL),
        ]

        bound = bind_strategy_assignments(proposals, assignments)

        self.assertEqual(bound[0].metadata["strategy_id"], "parameter-tuning")
        self.assertEqual(
            bound[1].metadata["strategy_id"], "open-structural-exploration"
        )
        self.assertEqual(bound[1].metadata["strategy_binding"], "response-order")

    def test_ast_novelty_separates_parameters_structure_and_renames(self) -> None:
        analyzer = SourceNoveltyAnalyzer()

        parameter = analyzer.analyze(PARENT, PARAMETER_ONLY, "make_kernel")
        structural = analyzer.analyze(PARENT, STRUCTURAL, "make_kernel")
        rename = analyzer.analyze(PARENT, RENAME_ONLY, "make_kernel")

        self.assertEqual(parameter.classification, "parameter-only")
        self.assertFalse(parameter.structural_change)
        self.assertTrue(
            any("default:block_m" in item for item in parameter.changed_tuning_parameters)
        )
        self.assertEqual(structural.classification, "structural")
        self.assertTrue(structural.changed_allocations)
        self.assertIn("allocation-or-layout", structural.structural_signals)
        self.assertEqual(rename.classification, "format-or-rename-only")
        self.assertFalse(rename.meaningful_change)

    def test_prompt_contains_exact_strategy_contract(self) -> None:
        task = _task("prompt", proposals=1)
        parent = Candidate.seed(task, PARENT, "kernel.py")
        assignments = StructuralStrategyPortfolio().plan(task, 1, 2)
        prompt = json.loads(
            build_optimization_prompt(
                task,
                parent,
                {
                    "generation_request": {
                        "round": 1,
                        "strategy_assignments": [
                            item.to_dict() for item in assignments
                        ],
                        "discovered_strategy_memory": [
                            {
                                "discovered_strategy": "persistent-query-tiles",
                                "status": "measured-improvement",
                            }
                        ],
                    }
                },
                [],
                2,
            )
        )

        self.assertEqual(
            prompt["strategy_assignments"][1]["strategy_slot"], assignments[1].slot
        )
        self.assertIn("metadata.strategy_slot", " ".join(prompt["rules"]))
        self.assertIn("not as a whitelist", " ".join(prompt["rules"]))
        self.assertEqual(
            prompt["discovered_strategy_memory"][0]["discovered_strategy"],
            "persistent-query-tiles",
        )
        metadata = prompt["response_schema"]["candidates"][0]["metadata"]
        self.assertIn("strategy_id", metadata)
        self.assertIn("discovered_strategy", metadata)

    def test_enforce_policy_rejects_false_structural_claim_before_model(self) -> None:
        source = (
            "def make_kernel(block_m: int = 64):\n"
            "    values = [0] * block_m\n"
            "    return values\n"
        )
        task = _task("controller-enforce", proposals=2, entrypoint="make_kernel")
        generator = _ParameterOnlyGenerator(source)
        backend = _SimpleBackend()

        with tempfile.TemporaryDirectory(prefix="structural-controller-") as directory:
            output = Path(directory)
            controller = OptimizationController(
                task=task,
                source_code=source,
                source_name="kernel.py",
                generator=generator,
                backend=backend,
                store=ArtifactStore(output),
                selection_policy="measure-all",
                profile_policy="none",
                structural_search_policy="enforce",
            )
            summary = controller.run()

            rejected = [
                item
                for item in controller.records.values()
                if item.state == "strategy-invalid"
            ]
            self.assertEqual(len(rejected), 1)
            self.assertEqual(
                rejected[0].failure.category, "structural-strategy-mismatch"
            )
            self.assertEqual(backend.model_calls, 2)  # seed + parameter lane only
            self.assertEqual(summary.best_latency_ms, 1.0)
            self.assertEqual(summary.parameter_only_candidates, 2)
            self.assertEqual(summary.strategy_rejected_candidates, 1)
            self.assertEqual(
                [
                    item["strategy_id"]
                    for item in generator.requests[0]["strategy_assignments"]
                ],
                ["parameter-tuning", "open-structural-exploration"],
            )
            trajectory = json.loads((output / "trajectory.json").read_text())
            rejected_row = next(
                item
                for item in trajectory["candidates"]
                if item["state"] == "strategy-invalid"
            )
            self.assertEqual(rejected_row["novelty_classification"], "parameter-only")

    def test_successful_open_strategy_is_visible_to_the_next_round(self) -> None:
        source = (
            "def make_kernel(block_m: int = 64):\n"
            "    values = [0] * block_m\n"
            "    return values\n"
        )
        task = _task(
            "open-memory",
            proposals=2,
            entrypoint="make_kernel",
            rounds=2,
        )
        generator = _OpenMemoryGenerator(source)
        backend = _OpenMemoryBackend()

        with tempfile.TemporaryDirectory(prefix="open-memory-") as directory:
            controller = OptimizationController(
                task=task,
                source_code=source,
                source_name="kernel.py",
                generator=generator,
                backend=backend,
                store=ArtifactStore(Path(directory)),
                selection_policy="measure-all",
                profile_policy="none",
                structural_search_policy="enforce",
            )
            summary = controller.run()
            report = (Path(directory) / "experiment_report.md").read_text()
            self.assertIn("single-trip-owner-loop", report)
            trajectory = json.loads(
                (Path(directory) / "trajectory.json").read_text()
            )
            self.assertTrue(
                any(
                    item["discovered_strategy"] == "single-trip-owner-loop"
                    for item in trajectory["candidates"]
                )
            )

        memory = generator.requests[1]["discovered_strategy_memory"]
        self.assertEqual(memory[0]["discovered_strategy"], "single-trip-owner-loop")
        self.assertEqual(memory[0]["status"], "measured-improvement")
        self.assertGreater(memory[0]["relative_improvement_vs_parent"], 0.0)
        self.assertEqual(summary.open_exploration_candidates, 1)
        self.assertEqual(summary.discovered_strategy_count, 1)

    def test_open_structural_candidate_must_name_its_strategy(self) -> None:
        source = (
            "def make_kernel(block_m: int = 64):\n"
            "    values = [0] * block_m\n"
            "    return values\n"
        )
        task = _task("open-name", proposals=2, entrypoint="make_kernel")

        with tempfile.TemporaryDirectory(prefix="open-name-") as directory:
            controller = OptimizationController(
                task=task,
                source_code=source,
                source_name="kernel.py",
                generator=_UnnamedOpenGenerator(source),
                backend=_OpenMemoryBackend(),
                store=ArtifactStore(Path(directory)),
                selection_policy="measure-all",
                profile_policy="none",
                structural_search_policy="enforce",
            )
            controller.run()

        rejected = [
            item
            for item in controller.records.values()
            if item.state == "strategy-invalid"
        ]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(
            rejected[0].failure.category, "open-strategy-metadata-missing"
        )

    def test_ai_planner_outcomes_feed_the_next_round(self) -> None:
        source = (
            "def make_kernel(block_m: int = 64):\n"
            "    values = [0] * block_m\n"
            "    return values\n"
        )
        task = _task(
            "ai-outcomes",
            proposals=2,
            entrypoint="make_kernel",
            rounds=2,
        )
        generator = _AiPlanningGenerator(source)

        with tempfile.TemporaryDirectory(prefix="ai-planning-") as directory:
            output = Path(directory)
            controller = OptimizationController(
                task=task,
                source_code=source,
                source_name="kernel.py",
                generator=generator,
                backend=_OpenMemoryBackend(),
                store=ArtifactStore(output),
                selection_policy="measure-all",
                profile_policy="none",
                structural_search_policy="off",
                strategy_allocation_policy="ai-planned",
            )
            summary = controller.run()
            plans = json.loads((output / "strategy_plans.json").read_text())

        self.assertEqual(summary.planner_calls, 2)
        self.assertEqual(summary.planner_fallbacks, 0)
        self.assertEqual(len(generator.planning_requests), 2)
        prior = generator.planning_requests[1]["strategy_outcomes"]
        self.assertGreater(
            prior["open-structural-exploration"][
                "best_relative_improvement_vs_parent"
            ],
            0.0,
        )
        self.assertIn("realized_outcomes", plans["rounds"][0])
        self.assertEqual(plans["rounds"][0]["source"], "hosted-ai-planner")

    def test_evidence_stable_strategy_plan_is_reused_for_three_round_window(self) -> None:
        source = (
            "def make_kernel(block_m: int = 64):\n"
            "    values = [0] * block_m\n"
            "    return values\n"
        )
        task = _task(
            "ai-plan-reuse",
            proposals=2,
            entrypoint="make_kernel",
            rounds=3,
        )
        generator = _AiPlanningGenerator(source)

        with tempfile.TemporaryDirectory(prefix="ai-plan-reuse-") as directory:
            output = Path(directory)
            controller = OptimizationController(
                task=task,
                source_code=source,
                source_name="kernel.py",
                generator=generator,
                backend=_StablePlanningBackend(),
                store=ArtifactStore(output),
                selection_policy="measure-all",
                profile_policy="none",
                structural_search_policy="off",
                strategy_allocation_policy="ai-planned",
                strategy_plan_interval_rounds=3,
            )
            summary = controller.run()
            plans = json.loads((output / "strategy_plans.json").read_text())

        self.assertEqual(summary.planner_calls, 1)
        self.assertEqual(len(generator.planning_requests), 1)
        self.assertEqual(
            [item["source"] for item in plans["rounds"]],
            ["hosted-ai-planner", "reused-ai-plan", "reused-ai-plan"],
        )
        self.assertEqual(
            [item["raw_plan"]["planned_round"] for item in plans["rounds"]],
            [1, 1, 1],
        )

    def test_unmeasured_strategy_outcomes_are_explicitly_unknown(self) -> None:
        source = "def make_kernel():\n    return 1\n"
        task = _task("unmeasured-outcome", proposals=1)
        seed = Candidate.seed(task, source, "kernel.py")
        proposal = Candidate.from_proposal(
            task,
            seed,
            CandidateProposal(
                "Try a layout.",
                source + "\nLAYOUT = 1\n",
                metadata={"strategy_id": "memory-layout"},
            ),
            generation=1,
        )

        with tempfile.TemporaryDirectory(prefix="unmeasured-outcome-") as directory:
            controller = OptimizationController(
                task=task,
                source_code=source,
                source_name="kernel.py",
                generator=_ParameterOnlyGenerator(source),
                backend=_SimpleBackend(),
                store=ArtifactStore(Path(directory)),
                profile_policy="none",
            )
            controller.records = {
                seed.candidate_id: CandidateRecord(
                    candidate=seed,
                    model=ModelEvaluation(valid=True, predicted_latency_ms=1.0),
                    measurement=Measurement(correct=True, latency_ms=1.0),
                ),
                proposal.candidate_id: CandidateRecord(
                    candidate=proposal,
                    model=ModelEvaluation(valid=True, predicted_latency_ms=0.5),
                    state="modeled-not-promoted",
                ),
            }

            outcome = controller._strategy_outcome_summary()["memory-layout"]

        self.assertEqual(outcome["unmeasured_compiled"], 1)
        self.assertEqual(outcome["hardware_coverage"], 0.0)
        self.assertEqual(outcome["evidence_status"], "no-hardware-evidence")
        self.assertFalse(outcome["negative_evidence_supported"])

    def test_reuse_is_rejected_after_repeated_strategy_failures(self) -> None:
        source = "def make_kernel():\n    return 1\n"
        task = _task("material-strategy-failure", proposals=1)

        with tempfile.TemporaryDirectory(prefix="material-strategy-failure-") as directory:
            controller = OptimizationController(
                task=task,
                source_code=source,
                source_name="kernel.py",
                generator=_ParameterOnlyGenerator(source),
                backend=_SimpleBackend(),
                store=ArtifactStore(Path(directory)),
                profile_policy="none",
                strategy_plan_interval_rounds=3,
            )
            origin = {
                "planning_evidence": {
                    "search_state": {
                        "profile_calls": 0,
                        "model_screening_mode": "evidence-warmup",
                    },
                    "strategy_outcomes": {},
                    "current_best": {
                        "candidate_id": "seed",
                        "measured_latency_ms": 1.0,
                    },
                }
            }
            current = {
                "search_state": {
                    "profile_calls": 0,
                    "model_screening_mode": "evidence-warmup",
                },
                "strategy_outcomes": {
                    "memory-layout": {
                        "failure_categories": {"compile-or-lowering": 2},
                        "negative_evidence_supported": False,
                    }
                },
                "current_best": {
                    "candidate_id": "seed",
                    "measured_latency_ms": 1.0,
                },
            }

            changed = controller._strategy_evidence_changed_materially(
                origin, current
            )

        self.assertTrue(changed)

    def test_planner_failure_uses_fallback_without_stopping_search(self) -> None:
        source = (
            "def make_kernel(block_m: int = 64):\n"
            "    values = [0] * block_m\n"
            "    return values\n"
        )
        task = _task("planner-fallback", proposals=2, entrypoint="make_kernel")
        generator = _FailingPlanningGenerator(source)

        with tempfile.TemporaryDirectory(prefix="planner-fallback-") as directory:
            output = Path(directory)
            controller = OptimizationController(
                task=task,
                source_code=source,
                source_name="kernel.py",
                generator=generator,
                backend=_SimpleBackend(),
                store=ArtifactStore(output),
                selection_policy="measure-all",
                profile_policy="none",
                structural_search_policy="off",
                strategy_allocation_policy="ai-planned",
            )
            summary = controller.run()
            plans = json.loads((output / "strategy_plans.json").read_text())

        self.assertEqual(summary.planner_calls, 1)
        self.assertEqual(summary.planner_fallbacks, 1)
        self.assertEqual(plans["rounds"][0]["source"], "deterministic-fallback")
        self.assertEqual(len(plans["rounds"][0]["assignments"]), 2)
        self.assertEqual(
            len(
                {
                    item["strategy_id"]
                    for item in plans["rounds"][0]["assignments"]
                }
            ),
            2,
        )

    def test_resume_reuses_archived_plan_without_another_planner_call(self) -> None:
        source = (
            "def make_kernel(block_m: int = 64):\n"
            "    values = [0] * block_m\n"
            "    return values\n"
        )
        task = _task("planner-resume", proposals=1, entrypoint="make_kernel")
        planner_calls = []

        with tempfile.TemporaryDirectory(prefix="planner-resume-") as directory:
            output = Path(directory)
            first = OptimizationController(
                task=task,
                source_code=source,
                source_name="kernel.py",
                generator=_ResumePlanningGenerator(
                    source, planner_calls, fail_generation=True
                ),
                backend=_SimpleBackend(),
                store=ArtifactStore(output),
                profile_policy="none",
                structural_search_policy="off",
                strategy_allocation_policy="ai-planned",
            )
            with self.assertRaisesRegex(RuntimeError, "generation interrupted"):
                first.run()

            resumed = OptimizationController(
                task=task,
                source_code=source,
                source_name="kernel.py",
                generator=_ResumePlanningGenerator(
                    source, planner_calls, fail_generation=False
                ),
                backend=_SimpleBackend(),
                store=ArtifactStore(output),
                profile_policy="none",
                structural_search_policy="off",
                strategy_allocation_policy="ai-planned",
                resume=True,
            )
            summary = resumed.run()

        self.assertEqual(planner_calls, [1])
        self.assertTrue(summary.resumed)
        self.assertEqual(summary.planner_calls, 1)


class _ParameterOnlyGenerator:
    last_call_metadata = {}
    last_exchange = {}

    def __init__(self, source: str) -> None:
        self.source = source
        self.requests = []

    def generate(self, task, parent, evidence, history, count):
        del task, parent, history
        self.requests.append(dict(evidence["generation_request"]))
        return [
            CandidateProposal(
                "Tune the existing block size.",
                self.source.replace("= 64", "= 128"),
            ),
            CandidateProposal(
                "Claim a layout rewrite but only tune the block size.",
                self.source.replace("= 64", "= 256"),
                metadata={
                    "discovered_strategy": "persistent-block-ownership",
                    "related_existing_strategies": ["work-decomposition"],
                },
            ),
        ][:count]


class _SimpleBackend:
    def __init__(self) -> None:
        self.model_calls = 0

    def model(self, task, candidate):
        del task
        self.model_calls += 1
        return ModelEvaluation(valid=True, predicted_latency_ms=1.0)

    def measure(self, task, candidate):
        del task
        latency = 1.0 if "= 128" in candidate.source_code else 2.0
        return Measurement(correct=True, latency_ms=latency, samples_ms=[latency])

    def profile(self, task, candidate):
        del task, candidate
        return ProfileEvaluation(bottleneck="unused")


class _BatchedMeasuredGenerator:
    last_call_metadata = {}
    last_exchange = {}

    def __init__(self) -> None:
        self.calls = []
        self.next_value = 1

    def generate(self, task, parent, evidence, history, count):
        del task, history
        request = dict(evidence["generation_request"])
        self.calls.append(request)
        parent_ids = [parent.candidate_id] + [
            item["candidate_id"] for item in request["parent_archive"]
        ]
        proposals = []
        for index in range(count):
            value = self.next_value
            self.next_value += 1
            proposals.append(
                CandidateProposal(
                    "Measured archive proposal %d." % value,
                    "VALUE = %d\n\ndef kernel(x):\n    return x\n" % value,
                    metadata={"parent_id": parent_ids[index % len(parent_ids)]},
                )
            )
        return proposals


class _MeasuredTIRBackend:
    def __init__(self) -> None:
        self.model_calls = 0
        self.tir_calls = 0

    def analyze_tir(self, task, candidate):
        del task
        self.tir_calls += 1
        return TIRAnalysis(
            valid=True,
            features={
                "threads_per_block": 128,
                "structural_fingerprint": candidate.source_sha256,
            },
        )

    def model(self, task, candidate):
        del task, candidate
        self.model_calls += 1
        raise AssertionError("TileSight model must not run")

    def measure(self, task, candidate):
        del task
        value = int(candidate.source_code.splitlines()[0].split("=")[1])
        latency = 10.0 - value
        return Measurement(correct=True, latency_ms=latency, samples_ms=[latency])

    def profile(self, task, candidate):
        del task, candidate
        raise AssertionError("NCU must not run during native search")


class _OpenMemoryGenerator:
    last_call_metadata = {}
    last_exchange = {}

    def __init__(self, source: str) -> None:
        self.source = source
        self.requests = []

    def generate(self, task, parent, evidence, history, count):
        del task, parent, history
        request = dict(evidence["generation_request"])
        self.requests.append(request)
        if len(self.requests) > 1:
            return []
        structural_source = self.source.replace(
            "    return values\n",
            "    for marker in range(1):\n"
            "        values = values\n"
            "    return values\n",
        )
        return [
            CandidateProposal(
                "Tune the parameter baseline.",
                self.source.replace("= 64", "= 128"),
            ),
            CandidateProposal(
                "Use a single-trip ownership loop as an open structural trial.",
                structural_source,
                metadata={
                    "discovered_strategy": "single-trip-owner-loop",
                    "related_existing_strategies": ["work-decomposition"],
                },
            ),
        ][:count]


class _OpenMemoryBackend(_SimpleBackend):
    def measure(self, task, candidate):
        del task
        if "for marker in range(1)" in candidate.source_code:
            latency = 0.8
        elif "= 128" in candidate.source_code:
            latency = 1.0
        else:
            latency = 2.0
        return Measurement(correct=True, latency_ms=latency, samples_ms=[latency])


class _StablePlanningBackend(_SimpleBackend):
    def measure(self, task, candidate):
        del task, candidate
        return Measurement(correct=True, latency_ms=1.0, samples_ms=[1.0])


class _UnnamedOpenGenerator(_OpenMemoryGenerator):
    def generate(self, task, parent, evidence, history, count):
        del task, parent, history
        self.requests.append(dict(evidence["generation_request"]))
        structural_source = self.source.replace(
            "    return values\n",
            "    for marker in range(1):\n"
            "        values = values\n"
            "    return values\n",
        )
        return [
            CandidateProposal(
                "Tune the parameter baseline.",
                self.source.replace("= 64", "= 128"),
            ),
            CandidateProposal(
                "Make an unnamed structural change.",
                structural_source,
            ),
        ][:count]


class _AiPlanningGenerator:
    last_call_metadata = {"attempts": 1}
    last_exchange = {}

    def __init__(self, source: str) -> None:
        self.source = source
        self.planning_requests = []
        self.generation_requests = []
        self.variant = 0

    def plan_strategies(
        self, task, parent, planning_context, strategies, count, round_number
    ):
        del task, parent, strategies, count, round_number
        self.planning_requests.append(dict(planning_context))
        return {
            "allocation": [
                {
                    "strategy_id": "parameter-tuning",
                    "count": 1,
                    "mode": "exploit",
                    "reason": "Retain a local schedule trial.",
                    "evidence": ["current_best"],
                },
                {
                    "strategy_id": "open-structural-exploration",
                    "count": 1,
                    "mode": "explore",
                    "reason": "Test a new ownership structure.",
                    "evidence": ["strategy_outcomes"],
                },
            ],
            "round_rationale": "Balance one local and one open trial.",
            "confidence": 0.7,
        }

    def generate(self, task, parent, evidence, history, count):
        del task, history
        request = dict(evidence["generation_request"])
        self.generation_requests.append(request)
        proposals = []
        for assignment in request["strategy_assignments"][:count]:
            self.variant += 1
            source = parent.source_code
            metadata = {}
            if assignment["strategy_id"] == "parameter-tuning":
                source = source.replace("= 64", "= 128")
            else:
                if "for marker in range(1)" not in source:
                    source = source.replace(
                        "    return values\n",
                        "    for marker in range(1):\n"
                        "        values = values\n"
                        "    return values\n",
                    )
                metadata = {
                    "discovered_strategy": "single-trip-owner-loop",
                    "related_existing_strategies": ["work-decomposition"],
                }
            source += "\nPLANNER_VARIANT_%d = %d\n" % (
                self.variant,
                self.variant,
            )
            proposals.append(
                CandidateProposal(
                    "Execute the assigned AI-planned direction.",
                    source,
                    metadata=metadata,
                )
            )
        return proposals


class _FailingPlanningGenerator(_ParameterOnlyGenerator):
    def plan_strategies(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("planner unavailable")


class _ResumePlanningGenerator:
    last_call_metadata = {"attempts": 1}
    last_exchange = {}

    def __init__(self, source, planner_calls, fail_generation):
        self.source = source
        self.planner_calls = planner_calls
        self.fail_generation = fail_generation

    def plan_strategies(
        self, task, parent, planning_context, strategies, count, round_number
    ):
        del task, parent, planning_context, strategies, count
        self.planner_calls.append(round_number)
        return {
            "allocation": [
                {
                    "strategy_id": "parameter-tuning",
                    "count": 1,
                    "mode": "exploit",
                    "reason": "Test one local schedule value.",
                    "evidence": ["current_best"],
                }
            ],
            "round_rationale": "Use the only available slot locally.",
            "confidence": 0.5,
        }

    def generate(self, task, parent, evidence, history, count):
        del task, parent, evidence, history, count
        if self.fail_generation:
            raise RuntimeError("generation interrupted")
        return [
            CandidateProposal(
                "Tune the local block size.",
                self.source.replace("= 64", "= 128"),
            )
        ]


def _task(
    task_id: str,
    proposals: int,
    entrypoint: str = "make_kernel",
    rounds: int = 1,
    beam_width: int = 1,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        description="Optimize a TileLang matmul kernel.",
        reference="Preserve the source semantics.",
        entrypoint=entrypoint,
        metadata={"kernel_family": "matmul"},
        budget=BudgetConfig(
            rounds=rounds,
            proposals_per_round=proposals,
            beam_width=beam_width,
            min_promotions_per_round=1,
            max_promotions_per_round=max(1, proposals),
            max_repairs_per_round=0,
            final_validation_candidates=0,
        ),
    )


if __name__ == "__main__":
    unittest.main()
