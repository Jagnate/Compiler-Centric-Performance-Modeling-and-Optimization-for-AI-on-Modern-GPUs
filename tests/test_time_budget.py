"""Search wall-clock budget and candidate-graph bound tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kernel_optimization.archive import ArtifactStore
from kernel_optimization.cli import build_parser
from kernel_optimization.controller import OptimizationController
from kernel_optimization.schema import (
    BudgetConfig,
    CandidateProposal,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    TaskSpec,
)


SOURCE = "def kernel(x):\n    return x\n"
PROPOSAL = "TILE = 2\n\ndef kernel(x):\n    return x\n"


class SearchTimeBudgetTests(unittest.TestCase):
    def test_cli_time_budget_is_disabled_by_default(self) -> None:
        args = build_parser().parse_args(
            ["--source", "kernel.py", "--task", "task.json"]
        )
        self.assertEqual(args.max_search_seconds, 0.0)
        self.assertFalse(args.search_until_time_budget)

    def test_stops_after_inflight_generation_and_exports_seed(self) -> None:
        task = _budget_task("time-budget-generation")
        clock = _Clock()
        generator = _SlowGenerator(clock)
        with tempfile.TemporaryDirectory(prefix="kernel-time-budget-") as directory:
            store = ArtifactStore(Path(directory))
            controller = OptimizationController(
                task=task,
                source_code=SOURCE,
                source_name="kernel.py",
                generator=generator,
                backend=_Backend(),
                store=store,
                structural_search_policy="off",
                strategy_allocation_policy="unconstrained",
                incumbent_snapshot_interval_seconds=0,
                max_search_seconds=5,
                time_budget_clock=clock,
            )

            summary = controller.run()
            state = store.load_state()
            manifest = json.loads(
                (Path(directory) / "experiment_manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(generator.calls, 1)
        self.assertEqual(summary.completed_rounds, 0)
        self.assertEqual(summary.best_latency_ms, 3.0)
        self.assertEqual(summary.termination_reason, "time-budget")
        self.assertTrue(summary.time_budget_exhausted)
        self.assertEqual(summary.max_search_seconds, 5.0)
        self.assertEqual(summary.generated_candidates, 2)
        self.assertEqual(summary.modeled_candidates, 1)
        self.assertEqual(summary.candidate_graph_upper_bound, 4)
        self.assertEqual(state["termination_reason"], "time-budget")
        self.assertEqual(state["active_round"], 1)
        self.assertEqual(state["candidate_graph_upper_bound"], 4)
        self.assertEqual(
            manifest["outcome"]["termination_reason"], "time-budget"
        )

    def test_exports_a_newly_measured_incumbent_when_budget_expires(self) -> None:
        task = _budget_task("time-budget-measurement")
        clock = _Clock()
        generator = _SlowGenerator(clock, delay_seconds=0)
        with tempfile.TemporaryDirectory(prefix="kernel-time-budget-") as directory:
            controller = OptimizationController(
                task=task,
                source_code=SOURCE,
                source_name="kernel.py",
                generator=generator,
                backend=_DelayedMeasurementBackend(clock),
                store=ArtifactStore(Path(directory)),
                structural_search_policy="off",
                strategy_allocation_policy="unconstrained",
                incumbent_snapshot_interval_seconds=0,
                max_search_seconds=5,
                time_budget_clock=clock,
            )

            summary = controller.run()
            best_source = Path(summary.best_source_path).read_text(encoding="utf-8")

        self.assertEqual(summary.termination_reason, "time-budget")
        self.assertEqual(summary.completed_rounds, 0)
        self.assertEqual(summary.best_latency_ms, 1.0)
        self.assertIn("TILE = 2", best_source)

    def test_fixed_time_mode_continues_after_an_empty_candidate_round(self) -> None:
        task = _budget_task("fixed-time-empty-round")
        clock = _Clock()
        generator = _EmptySlowGenerator(clock, delay_seconds=3)
        with tempfile.TemporaryDirectory(prefix="kernel-time-budget-") as directory:
            root = Path(directory)
            controller = OptimizationController(
                task=task,
                source_code=SOURCE,
                source_name="kernel.py",
                generator=generator,
                backend=_Backend(),
                store=ArtifactStore(root),
                structural_search_policy="off",
                strategy_allocation_policy="unconstrained",
                incumbent_snapshot_interval_seconds=0,
                max_search_seconds=5,
                search_until_time_budget=True,
                time_budget_clock=clock,
            )

            summary = controller.run()
            events = [
                json.loads(line)["event"]
                for line in (root / "events.jsonl").read_text().splitlines()
            ]

        self.assertEqual(generator.calls, 2)
        self.assertEqual(summary.completed_rounds, 1)
        self.assertEqual(summary.termination_reason, "time-budget")
        self.assertTrue(summary.time_budget_exhausted)
        self.assertEqual(summary.best_latency_ms, 3.0)
        self.assertIn("empty_round_continued", events)

    def test_resumed_time_budget_counts_prior_search_elapsed_time(self) -> None:
        clock = _Clock()
        with tempfile.TemporaryDirectory(prefix="kernel-time-budget-") as directory:
            controller = OptimizationController(
                task=_budget_task("cumulative-time-budget"),
                source_code=SOURCE,
                source_name="kernel.py",
                generator=_EmptySlowGenerator(clock, delay_seconds=1),
                backend=_Backend(),
                store=ArtifactStore(Path(directory)),
                incumbent_snapshot_interval_seconds=0,
                max_search_seconds=5,
                search_until_time_budget=True,
                time_budget_clock=clock,
            )
            controller._time_budget_started_at = clock()
            controller._prior_search_elapsed_seconds = 4.0
            clock.advance(2.0)

            exhausted = controller._time_budget_exhausted(2, "resume-test")

        self.assertTrue(exhausted)
        self.assertEqual(controller._search_elapsed_seconds(), 6.0)

    def test_rejects_invalid_time_budget(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kernel-time-budget-") as directory:
            with self.assertRaisesRegex(ValueError, "max_search_seconds"):
                OptimizationController(
                    task=_minimal_task(),
                    source_code=SOURCE,
                    source_name="kernel.py",
                    generator=_SlowGenerator(_Clock()),
                    backend=_Backend(),
                    store=ArtifactStore(Path(directory)),
                    max_search_seconds=-1,
                )
            with self.assertRaisesRegex(ValueError, "positive max_search_seconds"):
                OptimizationController(
                    task=_minimal_task(),
                    source_code=SOURCE,
                    source_name="kernel.py",
                    generator=_SlowGenerator(_Clock()),
                    backend=_Backend(),
                    store=ArtifactStore(Path(directory) / "fixed-time"),
                    search_until_time_budget=True,
                )


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class _SlowGenerator:
    def __init__(self, clock: _Clock, delay_seconds: float = 6.0) -> None:
        self.clock = clock
        self.delay_seconds = float(delay_seconds)
        self.calls = 0

    def generate(self, task, parent, evidence, history, count):
        del task, parent, evidence, history, count
        self.calls += 1
        self.clock.advance(self.delay_seconds)
        return [
            CandidateProposal(
                hypothesis="Change the tile marker.",
                source_code=PROPOSAL,
            )
        ]


class _EmptySlowGenerator:
    def __init__(self, clock: _Clock, delay_seconds: float) -> None:
        self.clock = clock
        self.delay_seconds = float(delay_seconds)
        self.calls = 0

    def generate(self, task, parent, evidence, history, count):
        del task, parent, evidence, history, count
        self.calls += 1
        self.clock.advance(self.delay_seconds)
        return []


class _Backend:
    def model(self, task, candidate):
        del task, candidate
        return ModelEvaluation(
            valid=True,
            predicted_latency_ms=3.0,
            bottleneck="test",
        )

    def measure(self, task, candidate):
        del task, candidate
        return Measurement(correct=True, latency_ms=3.0, samples_ms=[3.0])

    def profile(self, task, candidate):
        del task, candidate
        return ProfileEvaluation(valid=True, bottleneck="test")


class _DelayedMeasurementBackend(_Backend):
    def __init__(self, clock: _Clock) -> None:
        self.clock = clock

    def measure(self, task, candidate):
        del task
        if "TILE = 2" in candidate.source_code:
            self.clock.advance(6.0)
            latency = 1.0
        else:
            latency = 3.0
        return Measurement(correct=True, latency_ms=latency, samples_ms=[latency])


def _minimal_task() -> TaskSpec:
    return TaskSpec(
        task_id="time-budget-validation",
        description="validate time budget",
        reference="kernel returns its input",
        entrypoint="kernel",
    )


def _budget_task(task_id: str) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        description="Exercise a soft controller deadline.",
        reference="kernel returns its input",
        entrypoint="kernel",
        budget=BudgetConfig(
            rounds=3,
            proposals_per_round=1,
            beam_width=1,
            min_promotions_per_round=1,
            max_promotions_per_round=1,
            max_repairs_per_round=0,
        ),
    )


if __name__ == "__main__":
    unittest.main()
