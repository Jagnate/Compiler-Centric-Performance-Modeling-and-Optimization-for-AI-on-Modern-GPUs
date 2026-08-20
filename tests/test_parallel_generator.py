"""Parallel hosted-generation worker tests."""

from __future__ import annotations

import threading
import unittest

from kernel_optimization.cli import build_parser
from kernel_optimization.generators.parallel import ParallelCandidateGenerator
from kernel_optimization.schema import Candidate, CandidateProposal, TaskSpec


SOURCE = "def kernel():\n    return 0\n"


class ParallelCandidateGeneratorTests(unittest.TestCase):
    def test_cli_defaults_to_one_worker_and_accepts_override(self) -> None:
        default = build_parser().parse_args(
            ["--source", "kernel.py", "--task", "task.json"]
        )
        configured = build_parser().parse_args(
            [
                "--source",
                "kernel.py",
                "--task",
                "task.json",
                "--agent-workers",
                "2",
            ]
        )
        self.assertEqual(default.agent_workers, 1)
        self.assertEqual(configured.agent_workers, 2)

    def test_parallel_workers_split_slots_and_merge_in_stable_order(self) -> None:
        task, parent = _task_and_parent()
        barrier = threading.Barrier(2)
        calls = []
        calls_lock = threading.Lock()

        def factory():
            return _RecordingGenerator(barrier, calls, calls_lock)

        generator = ParallelCandidateGenerator(factory, max_workers=2)
        evidence = _evidence(5)
        proposals = generator.generate(task, parent, evidence, [], count=5)

        self.assertEqual(sorted(item["count"] for item in calls), [2, 3])
        self.assertEqual(
            [item.metadata["strategy_slot"] for item in proposals],
            ["slot-0", "slot-1", "slot-2", "slot-3", "slot-4"],
        )
        self.assertEqual(generator.last_call_metadata["active_workers"], 2)
        self.assertEqual(generator.last_call_metadata["successful_workers"], 2)
        self.assertEqual(generator.last_call_metadata["failed_workers"], 0)
        self.assertEqual(generator.last_call_metadata["attempts"], 2)
        self.assertEqual(
            generator.last_call_metadata["usage"]["prompt_tokens"], 5.0
        )

    def test_one_worker_preserves_single_request_behavior(self) -> None:
        task, parent = _task_and_parent()
        calls = []

        def factory():
            return _RecordingGenerator(None, calls, threading.Lock())

        generator = ParallelCandidateGenerator(factory, max_workers=1)
        proposals = generator.generate(task, parent, _evidence(4), [], count=4)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["count"], 4)
        self.assertEqual(len(proposals), 4)
        self.assertEqual(generator.last_call_metadata["kind"], "generate")

    def test_partial_worker_failure_keeps_successful_candidates(self) -> None:
        task, parent = _task_and_parent()
        generator = ParallelCandidateGenerator(
            lambda: _SlotFailingGenerator("slot-1"), max_workers=2
        )

        proposals = generator.generate(task, parent, _evidence(2), [], count=2)

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].metadata["strategy_slot"], "slot-0")
        self.assertEqual(generator.last_call_metadata["successful_workers"], 1)
        self.assertEqual(generator.last_call_metadata["failed_workers"], 1)
        errors = [
            item["error"]
            for item in generator.last_call_metadata["worker_calls"]
            if item["error"] is not None
        ]
        self.assertEqual(errors[0]["type"], "RuntimeError")

    def test_all_worker_failures_raise_and_preserve_metadata(self) -> None:
        task, parent = _task_and_parent()
        generator = ParallelCandidateGenerator(
            lambda: _AlwaysFailingGenerator(), max_workers=2
        )

        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            generator.generate(task, parent, _evidence(2), [], count=2)

        self.assertEqual(generator.last_call_metadata["successful_workers"], 0)
        self.assertEqual(generator.last_call_metadata["failed_workers"], 2)


class _RecordingGenerator:
    def __init__(self, barrier, calls, calls_lock) -> None:
        self.barrier = barrier
        self.calls = calls
        self.calls_lock = calls_lock
        self.last_call_metadata = {}
        self.last_exchange = {}

    def generate(self, task, parent, evidence, history, count):
        del task, parent, history
        slots = [
            item["slot"]
            for item in evidence["generation_request"]["strategy_assignments"]
        ]
        with self.calls_lock:
            self.calls.append({"count": count, "slots": slots})
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        self.last_call_metadata = {
            "kind": "generate",
            "attempts": 1,
            "usage": {"prompt_tokens": count, "completion_tokens": count * 2},
        }
        self.last_exchange = {"request_slots": slots}
        return [
            CandidateProposal(
                hypothesis="Optimize %s." % slot,
                source_code=SOURCE.replace("0", str(index + 1)),
                metadata={"strategy_slot": slot},
            )
            for index, slot in enumerate(slots)
        ]


class _SlotFailingGenerator:
    def __init__(self, failing_slot) -> None:
        self.failing_slot = failing_slot
        self.last_call_metadata = {}
        self.last_exchange = {}

    def generate(self, task, parent, evidence, history, count):
        del task, parent, history, count
        slot = evidence["generation_request"]["strategy_assignments"][0]["slot"]
        self.last_call_metadata = {
            "kind": "generate",
            "attempts": 1,
            "usage": {"prompt_tokens": 1},
        }
        self.last_exchange = {"slot": slot}
        if slot == self.failing_slot:
            raise RuntimeError("worker failed")
        return [
            CandidateProposal(
                hypothesis="Successful worker.",
                source_code=SOURCE,
                metadata={"strategy_slot": slot},
            )
        ]


class _AlwaysFailingGenerator:
    last_call_metadata = {"kind": "generate", "attempts": 1}
    last_exchange = {}

    def generate(self, task, parent, evidence, history, count):
        del task, parent, evidence, history, count
        raise RuntimeError("worker failed")


def _evidence(count):
    return {
        "generation_request": {
            "strategy_assignments": [
                {"slot": "slot-%d" % index, "strategy_id": "test"}
                for index in range(count)
            ]
        }
    }


def _task_and_parent():
    task = TaskSpec(
        task_id="parallel-test",
        description="Test parallel generation.",
        reference="tests.reference",
        entrypoint="kernel",
    )
    return task, Candidate.seed(task, SOURCE, "kernel.py")


if __name__ == "__main__":
    unittest.main()
