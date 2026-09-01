"""Equal-budget policies, compiled deduplication, ledgers, and reports."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from kernel_optimization.archive import ArtifactStore
from kernel_optimization.controller import (
    OptimizationController,
    resolve_evaluation_policies,
)
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
    TIRAnalysis,
    TaskSpec,
)
from kernel_optimization.selection import create_selection_policy
from kernel_optimization.trust import TrustTracker


SOURCE = "VALUE = %d\n\ndef kernel(x):\n    return x\n"


class PhaseThreeExperimentTests(unittest.TestCase):
    def test_evaluation_policy_resolution_is_orthogonal_and_explicit(self) -> None:
        resolved = resolve_evaluation_policies(
            evaluation_policy="ncu",
            tir_evidence_policy="auto",
            selection_policy="adaptive",
            profile_policy="milestone",
            compiled_deduplication=True,
        )
        self.assertEqual(resolved["tir_evidence_policy"], "hidden")
        self.assertEqual(resolved["selection_policy"], "measure-all")
        self.assertEqual(resolved["profile_policy"], "every-candidate")
        self.assertFalse(resolved["compiled_deduplication"])
        self.assertEqual(resolved["requested_selection_policy"], "adaptive")
        measured_tir = resolve_evaluation_policies(
            evaluation_policy="cuda-event",
            tir_evidence_policy="visible",
            selection_policy="adaptive",
            profile_policy="milestone",
            compiled_deduplication=True,
        )
        self.assertEqual(measured_tir["tir_evidence_policy"], "visible")
        self.assertEqual(measured_tir["selection_policy"], "measure-all")
        self.assertEqual(measured_tir["profile_policy"], "none")

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

    def test_milestone_profile_obeys_spacing_for_ordinary_improvements(self) -> None:
        task = _task()
        seed = CandidateRecord(
            candidate=Candidate.seed(task, SOURCE % 0, "kernel.py"),
            model=ModelEvaluation(valid=True, predicted_latency_ms=10.0),
            measurement=Measurement(correct=True, latency_ms=10.0),
        )
        candidate = CandidateRecord(
            candidate=Candidate.from_proposal(
                task,
                seed.candidate,
                CandidateProposal("Ten percent faster.", SOURCE % 1),
                generation=1,
            ),
            model=ModelEvaluation(valid=True, predicted_latency_ms=9.0),
            measurement=Measurement(correct=True, latency_ms=9.0),
        )
        policy = create_profile_policy("milestone", task.budget)
        policy.mark_profiled(seed, 0)

        self.assertIsNone(
            policy.decide(1, seed, candidate, [candidate], [seed, candidate])
        )
        decision = policy.decide(
            task.budget.ncu_min_interval_rounds,
            seed,
            candidate,
            [candidate],
            [seed, candidate],
        )
        self.assertEqual(decision.candidate_id, candidate.candidate.candidate_id)
        self.assertEqual(decision.reason, "meaningful-measured-improvement")

    def test_milestone_profiles_breakthrough_without_waiting_for_spacing(self) -> None:
        task = _task()
        seed = CandidateRecord(
            candidate=Candidate.seed(task, SOURCE % 0, "kernel.py"),
            model=ModelEvaluation(valid=True, predicted_latency_ms=10.0),
            measurement=Measurement(correct=True, latency_ms=10.0),
        )
        candidate = CandidateRecord(
            candidate=Candidate.from_proposal(
                task,
                seed.candidate,
                CandidateProposal("Twenty percent faster.", SOURCE % 1),
                generation=1,
            ),
            model=ModelEvaluation(valid=True, predicted_latency_ms=8.0),
            measurement=Measurement(correct=True, latency_ms=8.0),
        )
        policy = create_profile_policy("milestone", task.budget)
        policy.mark_profiled(seed, 0)

        decision = policy.decide(1, seed, candidate, [candidate], [seed, candidate])

        self.assertEqual(decision.candidate_id, candidate.candidate.candidate_id)
        self.assertEqual(decision.reason, "breakthrough-measured-improvement")

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
            self.assertEqual(summary.evaluation_policy, "tilesight")
            self.assertEqual(summary.requested_tir_evidence_policy, "auto")
            self.assertEqual(summary.tir_evidence_policy, "visible")
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
                "incumbent_history.jsonl",
                "incumbent_history.csv",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            trajectory = json.loads((output / "trajectory.json").read_text())
            self.assertEqual(len(trajectory["candidates"]), 3)
            report = (output / "experiment_report.md").read_text(encoding="utf-8")
            self.assertIn("Incumbent snapshot interval", report)
            self.assertIn("incumbent_history.csv", report)
            self.assertIn("Exported best latency", report)
            self.assertIn("Separate held-out final validation was not run", report)
            self.assertNotIn("passed fresh final validation", report)
            self.assertAlmostEqual(
                summary.cost_ledger["api"]["estimated_cost_usd"], 11e-6
            )
            self.assertEqual(
                summary.cost_ledger["api"]["provider_request_attempts"], 2
            )
            self.assertGreaterEqual(
                summary.cost_ledger["api"]["wall_seconds"], 0.25
            )
            self.assertEqual(summary.incumbent_snapshot_interval_seconds, 300.0)

    def test_cuda_event_policy_skips_tilesight_and_measures_every_candidate(self) -> None:
        backend = _AblationBackend()
        with tempfile.TemporaryDirectory(prefix="phase3-cuda-event-") as directory:
            controller = OptimizationController(
                task=_task(),
                source_code=SOURCE % 0,
                source_name="kernel.py",
                generator=_TwoCandidateGenerator(),
                backend=backend,
                store=ArtifactStore(Path(directory)),
                selection_policy="model-top",
                profile_policy="every-round",
                evaluation_policy="cuda-event",
                tir_evidence_policy="auto",
                strategy_allocation_policy="unconstrained",
            )
            summary = controller.run()

            self.assertEqual(backend.model_calls, 0)
            self.assertEqual(backend.measure_calls, 3)
            self.assertEqual(backend.profile_calls, 0)
            self.assertEqual(summary.modeled_candidates, 0)
            self.assertEqual(summary.measured_candidates, 3)
            self.assertEqual(summary.evaluation_policy, "cuda-event")
            self.assertEqual(summary.tir_evidence_policy, "hidden")
            self.assertEqual(summary.requested_selection_policy, "model-top")
            self.assertEqual(summary.selection_policy, "measure-all")
            self.assertEqual(summary.requested_profile_policy, "every-round")
            self.assertEqual(summary.profile_policy, "none")
            self.assertTrue(summary.requested_compiled_deduplication)
            self.assertFalse(summary.compiled_deduplication)
            self.assertTrue(
                all(
                    record.diagnosis.source == "deterministic-cuda-events"
                    for record in controller.records.values()
                    if record.is_measured_correct
                )
            )
            snapshots = [
                json.loads(line)
                for line in (Path(directory) / "incumbent_history.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                snapshots[-1]["policies"]["evaluation_policy"], "cuda-event"
            )

    def test_cuda_event_search_exposes_static_tir_without_model_screening(self) -> None:
        backend = _AblationBackend()
        generator = _CapturingGenerator()
        with tempfile.TemporaryDirectory(prefix="phase3-measured-tir-") as directory:
            controller = OptimizationController(
                task=_task(),
                source_code=SOURCE % 0,
                source_name="kernel.py",
                generator=generator,
                backend=backend,
                store=ArtifactStore(Path(directory)),
                evaluation_policy="cuda-event",
                tir_evidence_policy="visible",
                structural_search_policy="observe",
                strategy_allocation_policy="unconstrained",
            )
            summary = controller.run()

        self.assertEqual(backend.model_calls, 0)
        self.assertEqual(backend.tir_calls, 1)
        self.assertEqual(summary.modeled_candidates, 0)
        evidence = generator.calls[0]["evidence"]
        self.assertIsNone(evidence["predicted"])
        self.assertTrue(evidence["tir"]["valid"])
        self.assertEqual(
            evidence["tir"]["features"]["threads_per_block"], 128
        )

    def test_ncu_policy_profiles_every_correct_candidate_without_tilesight(self) -> None:
        backend = _AblationBackend()
        with tempfile.TemporaryDirectory(prefix="phase3-ncu-") as directory:
            controller = OptimizationController(
                task=_task(),
                source_code=SOURCE % 0,
                source_name="kernel.py",
                generator=_TwoCandidateGenerator(),
                backend=backend,
                store=ArtifactStore(Path(directory)),
                evaluation_policy="ncu",
                tir_evidence_policy="auto",
                strategy_allocation_policy="unconstrained",
            )
            summary = controller.run()

            self.assertEqual(backend.model_calls, 0)
            self.assertEqual(backend.measure_calls, 3)
            self.assertEqual(backend.profile_calls, 3)
            self.assertEqual(summary.profile_calls, 3)
            self.assertEqual(summary.profile_policy, "every-candidate")
            self.assertTrue(
                all(
                    record.profile is not None and record.profile.valid
                    for record in controller.records.values()
                    if record.is_measured_correct
                )
            )
            self.assertTrue(
                all(
                    record.diagnosis.source == "deterministic-ncu"
                    for record in controller.records.values()
                    if record.is_measured_correct
                )
            )

    def test_hidden_tir_evidence_is_not_exposed_to_generation_prompts(self) -> None:
        backend = _AblationBackend()
        generator = _CapturingGenerator()
        with tempfile.TemporaryDirectory(prefix="phase3-hidden-tir-") as directory:
            controller = OptimizationController(
                task=_task(),
                source_code=SOURCE % 0,
                source_name="kernel.py",
                generator=generator,
                backend=backend,
                store=ArtifactStore(Path(directory)),
                selection_policy="measure-all",
                profile_policy="none",
                compiled_deduplication=False,
                evaluation_policy="tilesight",
                tir_evidence_policy="hidden",
                strategy_allocation_policy="unconstrained",
            )
            summary = controller.run()

            self.assertEqual(backend.model_calls, 3)
            self.assertEqual(summary.modeled_candidates, 3)
            evidence = generator.calls[0]["evidence"]
            history = generator.calls[0]["history"]
            self.assertIsNone(evidence["predicted"])
            self.assertIsNone(evidence["model_trust"])
            self.assertIsNone(evidence["calibration"])
            self.assertIsNone(evidence["observed"]["diagnosis"])
            self.assertTrue(evidence["shared_memory"])
            lesson = evidence["shared_memory"][0]
            self.assertEqual(lesson["bottleneck"], "unknown")
            self.assertEqual(lesson["recommended_actions"], [])
            self.assertNotIn(
                "raw_predicted_latency_ms", lesson["supporting_evidence"]
            )
            self.assertIsNone(history[0]["predicted_latency_ms"])
            self.assertIsNone(history[0]["diagnosis"])
            prompt_payload = json.dumps(generator.calls[0], sort_keys=True)
            self.assertNotIn("deterministic-tilesight", prompt_payload)

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
            stage_timings={
                "tir": {"calls": 1, "failures": 0, "seconds": 0.5},
                "model": {"calls": 1, "failures": 0, "seconds": 2},
            },
            generator_usage={},
            preflight_calls=1,
            provider_api_requests=1,
            generator_calls=0,
            repair_calls=0,
            profile_calls=0,
            final_validation_calls=0,
        )
        self.assertEqual(ledger["evaluator"]["total_wall_seconds"], 2.5)
        self.assertEqual(ledger["evaluator"]["tir_analysis_calls"], 1)
        self.assertEqual(ledger["hardware"]["observed_kernel_sample_seconds"], 0.003)
        self.assertIn("exclude compilation", ledger["hardware"]["note"])

    def test_suite_command_carries_equal_budget_policy(self) -> None:
        module = _load_suite_module()
        parsed = module.build_parser().parse_args(
            ["--source", "kernel.py", "--task", "task.json", "--output-root", "out"]
        )
        self.assertEqual(parsed.profile_policy, "every-round")
        self.assertEqual(parsed.evaluation_policy, "tilesight")
        self.assertEqual(parsed.tir_evidence_policy, "auto")
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
        self.assertEqual(
            command[
                command.index("--incumbent-snapshot-interval-seconds") + 1
            ],
            "300.0",
        )
        self.assertEqual(
            command[command.index("--evaluation-policy") + 1], "tilesight"
        )
        self.assertEqual(
            command[command.index("--tir-evidence-policy") + 1], "auto"
        )


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


class _AblationBackend:
    def __init__(self):
        self.model_calls = 0
        self.measure_calls = 0
        self.profile_calls = 0
        self.tir_calls = 0

    def analyze_tir(self, task, candidate):
        del task, candidate
        self.tir_calls += 1
        return TIRAnalysis(
            valid=True,
            features={"threads_per_block": 128, "structural_fingerprint": "tir"},
        )

    def model(self, task, candidate):
        del task
        self.model_calls += 1
        value = int(candidate.source_code.splitlines()[0].split("=")[1])
        return ModelEvaluation(
            valid=True,
            predicted_latency_ms=3.0 - value,
            bottleneck="tensor-core",
            confidence="high",
            metrics={"tensor_util": 0.75},
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
        )

    def profile(self, task, candidate):
        del task, candidate
        self.profile_calls += 1
        return ProfileEvaluation(
            bottleneck="tensor-core",
            metrics={"tensor_util": 80.0, "achieved_occupancy": 0.6},
        )


class _CapturingGenerator(_TwoCandidateGenerator):
    def __init__(self):
        self.calls = []

    def generate(self, task, parent, evidence, history, count):
        self.calls.append({"evidence": evidence, "history": history})
        return super().generate(task, parent, evidence, history, count)


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
