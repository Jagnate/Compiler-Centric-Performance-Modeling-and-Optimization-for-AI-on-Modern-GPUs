"""End-to-end tests for the optimization controller."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kernel_optimization.archive import ArtifactStore
from kernel_optimization.backends.mock import MockPerformanceBackend
from kernel_optimization.controller import OptimizationController
from kernel_optimization.generators.mock import DeterministicMockGenerator
from kernel_optimization.schema import (
    Candidate,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    TaskSpec,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class OptimizationControllerTests(unittest.TestCase):
    def test_mock_search_improves_seed_and_preserves_trust_boundary(self) -> None:
        task = TaskSpec.from_json_file(REPOSITORY_ROOT / "examples" / "mock_task.json")
        backend = MockPerformanceBackend(task.evaluator)

        with tempfile.TemporaryDirectory(prefix="kernel-opt-controller-") as directory:
            output = Path(directory)
            controller = OptimizationController(
                task=task,
                generator=DeterministicMockGenerator(),
                backend=backend,
                store=ArtifactStore(output),
            )
            summary = controller.run()

            self.assertLess(summary.best_latency_ms, summary.seed_latency_ms)
            self.assertGreater(summary.speedup_over_seed, 1.0)
            self.assertEqual(summary.completed_rounds, task.budget.rounds)
            self.assertEqual(summary.modeled_candidates, backend.model_calls)
            self.assertEqual(summary.measured_candidates, backend.measure_calls)
            self.assertEqual(summary.profile_calls, backend.profile_calls)
            self.assertLessEqual(
                summary.profile_calls,
                task.budget.rounds + 1,
                "The seed and at most one candidate per round may be profiled.",
            )
            self.assertTrue(all(record.is_measured_correct for record in controller.beam))
            self.assertTrue(
                any(
                    record.state == "correctness-failed"
                    for record in controller.records.values()
                ),
                "The fixture should prove that incorrect candidates are archived.",
            )

            summary_file = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            state_file = json.loads(
                (output / "state.json").read_text(encoding="utf-8")
            )
            events = [
                json.loads(line)
                for line in (output / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(summary_file["best_candidate_id"], summary.best_candidate_id)
            self.assertEqual(state_file["best_candidate_id"], summary.best_candidate_id)
            self.assertEqual(
                sum(event["event"] == "candidate_profiled" for event in events),
                summary.profile_calls - 1,
            )
            self.assertEqual(
                len(list((output / "candidates").glob("*.json"))),
                summary.generated_candidates,
            )

    def test_seed_correctness_failure_stops_before_profiling(self) -> None:
        task = TaskSpec(
            task_id="bad-seed",
            description="Reject an invalid initial kernel.",
            reference="The seed must pass correctness.",
            base_parameters={"tile": 1},
            search_space={"tile": (1, 2)},
        )
        backend = _FailingSeedBackend()

        with tempfile.TemporaryDirectory(prefix="kernel-opt-bad-seed-") as directory:
            controller = OptimizationController(
                task=task,
                generator=DeterministicMockGenerator(),
                backend=backend,
                store=ArtifactStore(Path(directory)),
            )
            with self.assertRaisesRegex(RuntimeError, "failed correctness"):
                controller.run()

            self.assertEqual(backend.profile_calls, 0)
            candidate_files = list((Path(directory) / "candidates").glob("*.json"))
            self.assertEqual(len(candidate_files), 1)
            record = json.loads(candidate_files[0].read_text(encoding="utf-8"))
            self.assertEqual(record["state"], "seed-correctness-failed")


class _FailingSeedBackend:
    def __init__(self) -> None:
        self.profile_calls = 0

    def model(self, task: TaskSpec, candidate: Candidate) -> ModelEvaluation:
        del task, candidate
        return ModelEvaluation(valid=True, predicted_latency_ms=1.0)

    def measure(self, task: TaskSpec, candidate: Candidate) -> Measurement:
        del task, candidate
        return Measurement(correct=False, error="Reference mismatch.")

    def profile(self, task: TaskSpec, candidate: Candidate) -> ProfileEvaluation:
        del task, candidate
        self.profile_calls += 1
        return ProfileEvaluation(bottleneck="unknown")


if __name__ == "__main__":
    unittest.main()

