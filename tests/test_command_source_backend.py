"""Integration tests for the source-file subprocess evaluator boundary."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

from kernel_optimization.backends.command import CommandBackend
from kernel_optimization.factory import create_backend
from kernel_optimization.schema import Candidate, TaskSpec


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "contract_evaluator.py"


class CommandSourceBackendTests(unittest.TestCase):
    def test_external_process_receives_materialized_source_for_all_stages(self) -> None:
        task = TaskSpec(
            task_id="command-source-test",
            description="Exercise the source command contract.",
            reference="Return the input.",
            entrypoint="kernel",
        )
        candidate = Candidate.seed(
            task,
            "def kernel(x):\n    return x\n",
            "kernel.py",
        )
        backend = CommandBackend(
            command=[sys.executable, str(FIXTURE)],
            working_directory=REPOSITORY_ROOT,
            timeout_seconds=30,
        )

        modeled = backend.model(task, candidate)
        measured = backend.measure(task, candidate)
        profiled = backend.profile(task, candidate)

        self.assertTrue(modeled.valid)
        self.assertTrue(modeled.metrics["source_path_exists"])
        self.assertTrue(measured.correct)
        self.assertEqual(measured.metrics["measurement_source"], "test-contract")
        self.assertEqual(profiled.metrics["profile_source"], "test-contract")

    def test_example_task_resolves_adapter_from_repository_root(self) -> None:
        task_path = REPOSITORY_ROOT / "examples" / "tilelang_matmul_task.json"
        task = TaskSpec.from_json_file(task_path)

        backend = create_backend(task, task_path.parent)

        self.assertIsInstance(backend, CommandBackend)
        self.assertEqual(backend.working_directory, REPOSITORY_ROOT)
        self.assertEqual(backend.environment["PYTHONPATH"], "../TileSight")


if __name__ == "__main__":
    unittest.main()
