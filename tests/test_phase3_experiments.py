"""Equal-budget policies, compiled deduplication, ledgers, and reports."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from kernel_optimization.archive import ArtifactStore
from kernel_optimization.controller import OptimizationController
from kernel_optimization.costs import build_cost_ledger
from kernel_optimization.milestones import create_profile_policy
from kernel_optimization.schema import (
    BudgetConfig,
    Candidate,
    CandidateProposal,
    CandidateRecord,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    TaskSpec,
)
from kernel_optimization.selection import create_selection_policy
from kernel_optimization.trust import TrustTracker


SOURCE = "VALUE = %d\n\ndef kernel(x):\n    return x\n"


class PhaseThreeExperimentTests(unittest.TestCase):
    def test_equal_budget_policies_select_the_requested_count(self) -> None:
        task = _task()
        seed = Candidate.seed(task, SOURCE % 0, "kernel.py")
        modeled = []
        for value in range(1, 6):
            candidate = Candidate.from_proposal(
                task,
                seed,
                CandidateProposal("Candidate %d." % value, SOURCE % value),
                generation=1,
            )
            modeled.append(
                CandidateRecord(
                    candidate=candidate,
                    model=ModelEvaluation(
                        valid=True,
                        predicted_latency_ms=float(6 - value),
                        confidence="medium",
                    ),
                )
            )
        for name in ("adaptive", "model-top", "random"):
            policy = create_selection_policy(name, task.budget, 3)
            selected = policy.select(task, modeled, [], TrustTracker())
            self.assertEqual(len(selected), 3, name)
        measure_all = create_selection_policy("measure-all", task.budget, 3)
        self.assertEqual(
            len(measure_all.select(task, modeled, [], TrustTracker())), 5
        )

    def test_profile_none_skips_seed_and_rounds(self) -> None:
        policy = create_profile_policy("none", _task().budget)
        self.assertFalse(policy.profile_seed)
        self.assertIsNone(policy.decide(1, None, None, [], []))

    def test_every_round_profile_chooses_best_new_measurement(self) -> None:
        task = _task()
        seed = CandidateRecord(
            candidate=Candidate.seed(task, SOURCE % 0, "kernel.py"),
            measurement=Measurement(correct=True, latency_ms=3.0),
        )
        candidate = CandidateRecord(
            candidate=Candidate.from_proposal(
                task,
                seed.candidate,
                CandidateProposal("Faster candidate.", SOURCE % 1),
                generation=1,
            ),
            measurement=Measurement(correct=True, latency_ms=2.0),
        )
        policy = create_profile_policy("every-round", task.budget)
        policy.mark_profiled(seed, 0)
        decision = policy.decide(1, seed, candidate, [candidate], [seed, candidate])
        self.assertEqual(decision.candidate_id, candidate.candidate.candidate_id)
        self.assertEqual(decision.reason, "every-round")

    def test_compiled_equivalent_candidate_is_not_measured(self) -> None:
        task = _task()
        backend = _DedupBackend()
        with tempfile.TemporaryDirectory(prefix="phase3-dedup-") as directory:
            output = Path(directory)
            controller = OptimizationController(
                task=task,
                source_code=SOURCE % 0,
                source_name="kernel.py",
                generator=_TwoCandidateGenerator(),
                backend=backend,
                store=ArtifactStore(output),
                selection_policy="model-top",
                profile_policy="none",
                fixed_promotions_per_round=2,
                preflight_calls=1,
                preflight_usage={"prompt_tokens": 2, "completion_tokens": 1},
                preflight_elapsed_seconds=0.25,
                api_input_price_per_million=1.0,
                api_output_price_per_million=2.0,
            )
            summary = controller.run()

            self.assertEqual(backend.measure_calls, 2)
            self.assertEqual(summary.measured_candidates, 2)
            self.assertEqual(summary.compiled_equivalent_candidates, 1)
            self.assertEqual(summary.profile_calls, 0)
            self.assertEqual(summary.selection_policy, "model-top")
            self.assertEqual(summary.profile_policy, "none")
            equivalent = [
                record
                for record in controller.records.values()
                if record.compiled_equivalent_to is not None
            ]
            self.assertEqual(len(equivalent), 1)
            self.assertEqual(equivalent[0].state, "compiled-equivalent")
            for filename in (
                "trajectory.json",
                "trajectory.csv",
                "candidate_graph.json",
                "experiment_manifest.json",
                "experiment_report.md",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            trajectory = json.loads((output / "trajectory.json").read_text())
            self.assertEqual(len(trajectory["candidates"]), 3)
            self.assertAlmostEqual(
                summary.cost_ledger["api"]["estimated_cost_usd"], 11e-6
            )
            self.assertEqual(
                summary.cost_ledger["api"]["provider_request_attempts"], 2
            )
            self.assertGreaterEqual(
                summary.cost_ledger["api"]["wall_seconds"], 0.25
            )

    def test_cost_ledger_separates_samples_from_wall_time(self) -> None:
        task = _task()
        candidate = Candidate.seed(task, SOURCE % 0, "kernel.py")
        record = CandidateRecord(
            candidate=candidate,
            model=ModelEvaluation(valid=True, predicted_latency_ms=1.0),
            measurement=Measurement(
                correct=True,
                latency_ms=1.0,
                samples_ms=[1.0, 2.0],
                metrics={"case_count": 3},
            ),
        )
        ledger = build_cost_ledger(
            [record],
            stage_timings={"model": {"calls": 1, "failures": 0, "seconds": 2}},
            generator_usage={},
            preflight_calls=1,
            provider_api_requests=1,
            generator_calls=0,
            repair_calls=0,
            profile_calls=0,
            final_validation_calls=0,
        )
        self.assertEqual(ledger["evaluator"]["total_wall_seconds"], 2.0)
        self.assertEqual(ledger["hardware"]["observed_kernel_sample_seconds"], 0.003)
        self.assertIn("exclude compilation", ledger["hardware"]["note"])

    def test_suite_command_carries_equal_budget_policy(self) -> None:
        module = _load_suite_module()
        parsed = module.build_parser().parse_args(
            ["--source", "kernel.py", "--task", "task.json", "--output-root", "out"]
        )
        self.assertEqual(parsed.profile_policy, "every-round")
        self.assertEqual(parsed.strategy_allocation_policy, "ai-planned")
        command = module.build_run_command(
            source=Path("kernel.py"),
            task=Path("task.json"),
            output=Path("results/adaptive"),
            policy="adaptive",
            promotions_per_round=4,
            profile_policy="milestone",
        )
        self.assertIn("--selection-policy", command)
        self.assertEqual(
            command[command.index("--strategy-allocation-policy") + 1],
            "ai-planned",
        )
        self.assertEqual(command[command.index("--promotions-per-round") + 1], "4")


class _TwoCandidateGenerator:
    last_call_metadata = {
        "attempts": 1,
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    last_exchange = {}

    def generate(self, task, parent, evidence, history, count):
        del task, parent, evidence, history
        return [
            CandidateProposal("First compiled schedule.", SOURCE % 1),
            CandidateProposal("Equivalent compiled schedule.", SOURCE % 2),
        ][:count]


class _DedupBackend:
    def __init__(self):
        self.measure_calls = 0

    def model(self, task, candidate):
        del task
        value = int(candidate.source_code.splitlines()[0].split("=")[1])
        return ModelEvaluation(
            valid=True,
            predicted_latency_ms=3.0 - value,
            confidence="high",
            metrics={
                "compiled_source_sha256": "seed" if value == 0 else "same-binary"
            },
        )

    def measure(self, task, candidate):
        del task
        self.measure_calls += 1
        value = int(candidate.source_code.splitlines()[0].split("=")[1])
        latency = 3.0 - value
        return Measurement(
            correct=True,
            latency_ms=latency,
            samples_ms=[latency],
            metrics={"case_count": 1},
        )

    def profile(self, task, candidate):
        del task, candidate
        return ProfileEvaluation(bottleneck="unused")


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="phase3-test",
        description="Test experiment infrastructure.",
        reference="Return x.",
        entrypoint="kernel",
        budget=BudgetConfig(
            rounds=1,
            proposals_per_round=2,
            beam_width=1,
            min_promotions_per_round=1,
            max_promotions_per_round=2,
            max_repairs_per_round=0,
            final_validation_candidates=0,
        ),
    )


def _load_suite_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "run_experiment_suite.py"
    spec = importlib.util.spec_from_file_location("experiment_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
