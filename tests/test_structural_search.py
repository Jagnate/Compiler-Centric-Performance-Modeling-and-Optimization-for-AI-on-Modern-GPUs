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
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
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
    def test_cli_enforces_structural_search_by_default(self) -> None:
        args = build_parser().parse_args(["--source", "kernel.py", "--task", "task.json"])
        self.assertEqual(args.structural_search_policy, "enforce")

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


def _task(
    task_id: str,
    proposals: int,
    entrypoint: str = "make_kernel",
    rounds: int = 1,
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
            beam_width=1,
            min_promotions_per_round=1,
            max_promotions_per_round=max(1, proposals),
            max_repairs_per_round=0,
            final_validation_candidates=0,
        ),
    )


if __name__ == "__main__":
    unittest.main()
