"""Integration tests for the subprocess evaluator boundary."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

from kernel_optimization.backends.command import CommandBackend
from kernel_optimization.factory import create_backend
from kernel_optimization.schema import Candidate, TaskSpec


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPOSITORY_ROOT / "examples"


class CommandBackendTests(unittest.TestCase):
    def test_external_process_supports_all_fidelity_stages(self) -> None:
        task = TaskSpec.from_json_file(EXAMPLES / "command_task.json")
        backend = CommandBackend(
            command=[sys.executable, str(EXAMPLES / "mock_external_evaluator.py")],
            working_directory=REPOSITORY_ROOT,
            timeout_seconds=30,
        )
        candidate = Candidate.seed(task)

        modeled = backend.model(task, candidate)
        measured = backend.measure(task, candidate)
        profiled = backend.profile(task, candidate)

        self.assertTrue(modeled.valid)
        self.assertGreater(modeled.predicted_latency_ms, 0)
        self.assertTrue(measured.correct)
        self.assertGreater(measured.latency_ms, 0)
        self.assertEqual(measured.metrics["measurement_source"], "deterministic-mock")
        self.assertTrue(profiled.metrics["mock_ncu"])

    def test_factory_resolves_working_directory_relative_to_task_file(self) -> None:
        task = TaskSpec.from_json_file(EXAMPLES / "command_task.json")

        backend = create_backend(task, EXAMPLES)

        self.assertIsInstance(backend, CommandBackend)
        self.assertEqual(backend.working_directory, REPOSITORY_ROOT)


if __name__ == "__main__":
    unittest.main()

