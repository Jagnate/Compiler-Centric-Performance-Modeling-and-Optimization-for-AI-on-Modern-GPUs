"""Integration tests for the source-file subprocess evaluator boundary."""

from __future__ import annotations

import json
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock

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

        analyzed = backend.analyze_tir(task, candidate)
        modeled = backend.model(task, candidate)
        measured = backend.measure(task, candidate)
        profiled = backend.profile(task, candidate)
        finalized = backend.finalize(task, candidate)

        self.assertTrue(analyzed.valid)
        self.assertTrue(analyzed.features["source_path_exists"])
        self.assertTrue(modeled.valid)
        self.assertTrue(modeled.metrics["source_path_exists"])
        self.assertTrue(measured.correct)
        self.assertEqual(measured.metrics["measurement_source"], "test-contract")
        self.assertEqual(profiled.metrics["profile_source"], "test-contract")
        self.assertTrue(finalized.correct)
        self.assertEqual(finalized.metrics["stage"], "final")

    def test_example_task_resolves_adapter_from_repository_root(self) -> None:
        task_path = REPOSITORY_ROOT / "examples" / "tilelang_matmul_task.json"
        task = TaskSpec.from_json_file(task_path)

        backend = create_backend(task, task_path.parent)

        self.assertIsInstance(backend, CommandBackend)
        self.assertEqual(backend.working_directory, REPOSITORY_ROOT)
        self.assertEqual(backend.environment["PYTHONPATH"], "../TileSight")
        self.assertTrue(backend.persistent_process)
        self.assertEqual(backend.worker_max_requests, 32)

    def test_persistent_worker_reuses_one_process_and_archives_attempts(self) -> None:
        task = TaskSpec(
            task_id="persistent-command-test",
            description="Reuse one warmed evaluator process.",
            reference="Return the input.",
            entrypoint="kernel",
        )
        candidate = Candidate.seed(task, "def kernel(x):\n    return x\n", "kernel.py")
        with tempfile.TemporaryDirectory(prefix="persistent-command-") as directory:
            backend = CommandBackend(
                command=[sys.executable, str(FIXTURE)],
                working_directory=REPOSITORY_ROOT,
                artifact_directory=Path(directory),
                persistent_process=True,
                worker_max_requests=8,
            )
            try:
                analyzed = backend.analyze_tir(task, candidate)
                modeled = backend.model(task, candidate)
                measured = backend.measure(task, candidate)
                profiled = backend.profile(task, candidate)
                finalized = backend.finalize(task, candidate)
            finally:
                backend.close()

            pids = {
                analyzed.features["worker_pid"],
                modeled.metrics["worker_pid"],
                measured.metrics["worker_pid"],
                profiled.metrics["worker_pid"],
            }
            self.assertEqual(len(pids), 1)
            self.assertEqual(analyzed.features["worker_request_index"], 1)
            self.assertEqual(modeled.metrics["worker_request_index"], 2)
            self.assertEqual(measured.metrics["worker_request_index"], 3)
            self.assertEqual(profiled.metrics["worker_request_index"], 4)
            self.assertEqual(finalized.metrics["stage"], "final")
            self.assertIsNone(backend._worker)
            attempts = sorted(Path(directory).glob("*/*/attempt.json"))
            self.assertEqual(len(attempts), 5)
            metadata = [json.loads(path.read_text()) for path in attempts]
            self.assertEqual(
                [item["execution_mode"] for item in metadata].count(
                    "persistent-worker"
                ),
                4,
            )
            self.assertEqual(
                [item["execution_mode"] for item in metadata].count("one-shot"),
                1,
            )
            persistent_metadata = [
                item
                for item in metadata
                if item["execution_mode"] == "persistent-worker"
            ]
            self.assertTrue(
                all(item["worker_start_count"] == 1 for item in persistent_metadata)
            )

    def test_persistent_worker_restarts_at_the_request_bound(self) -> None:
        task = TaskSpec(
            task_id="bounded-persistent-command-test",
            description="Bound evaluator process lifetime.",
            reference="Return the input.",
            entrypoint="kernel",
        )
        candidate = Candidate.seed(task, "def kernel(x):\n    return x\n", "kernel.py")
        backend = CommandBackend(
            command=[sys.executable, str(FIXTURE)],
            working_directory=REPOSITORY_ROOT,
            persistent_process=True,
            worker_max_requests=2,
        )
        try:
            modeled = backend.model(task, candidate)
            measured = backend.measure(task, candidate)
            profiled = backend.profile(task, candidate)
        finally:
            backend.close()

        self.assertEqual(modeled.metrics["worker_pid"], measured.metrics["worker_pid"])
        self.assertNotEqual(modeled.metrics["worker_pid"], profiled.metrics["worker_pid"])
        self.assertEqual(profiled.metrics["worker_request_index"], 1)

    def test_persistent_worker_recovers_after_a_candidate_crashes_it(self) -> None:
        task = TaskSpec(
            task_id="crashing-persistent-command-test",
            description="Restart after an evaluator process crash.",
            reference="Return the input.",
            entrypoint="kernel",
        )
        crashing = Candidate.seed(
            task,
            "# KERNEL_OPT_TEST_WORKER_EXIT\ndef kernel(x):\n    return x\n",
            "kernel.py",
        )
        healthy = Candidate.seed(task, "def kernel(x):\n    return x\n", "kernel.py")
        with tempfile.TemporaryDirectory(prefix="crashing-command-") as directory:
            backend = CommandBackend(
                command=[sys.executable, str(FIXTURE)],
                working_directory=REPOSITORY_ROOT,
                artifact_directory=Path(directory),
                persistent_process=True,
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "worker exited"):
                    backend.model(task, crashing)
                modeled = backend.model(task, healthy)
                metadata = dict(backend.last_stage_metadata)
            finally:
                backend.close()

            self.assertTrue(modeled.valid)
            self.assertEqual(metadata["worker_start_count"], 2)
            failed_attempt = next(
                Path(directory).glob(crashing.candidate_id + "/model_*/attempt.json")
            )
            failure = json.loads(failed_attempt.read_text())
            self.assertEqual(failure["status"], "failed")

    def test_factory_isolates_tilelang_cache_inside_run_output(self) -> None:
        task_path = REPOSITORY_ROOT / "examples" / "tilelang_matmul_task.json"
        task = TaskSpec.from_json_file(task_path)

        with tempfile.TemporaryDirectory(prefix="kernel-cache-output-") as directory:
            artifacts = Path(directory) / "evaluator_attempts"
            backend = create_backend(
                task,
                task_path.parent,
                artifact_directory=artifacts,
            )

            expected = Path(directory) / ".tilelang_cache"
            self.assertEqual(
                Path(backend.environment["TILELANG_CACHE_DIR"]), expected
            )
            self.assertEqual(
                Path(backend.environment["TILELANG_TMP_DIR"]), expected / "tmp"
            )
            self.assertTrue((expected / "tmp").is_dir())

    def test_sensitive_environment_is_removed_and_attempt_is_archived(self) -> None:
        task = TaskSpec(
            task_id="command-environment-test",
            description="Verify the evaluator environment boundary.",
            reference="Return the input.",
            entrypoint="kernel",
        )
        candidate = Candidate.seed(
            task,
            "def kernel(x):\n    return x\n",
            "kernel.py",
        )
        with tempfile.TemporaryDirectory(prefix="kernel-command-artifacts-") as directory:
            backend = CommandBackend(
                command=[sys.executable, str(FIXTURE)],
                working_directory=REPOSITORY_ROOT,
                environment={"SAFE_TEST_ENV": "forwarded"},
                artifact_directory=Path(directory),
            )
            with mock.patch.dict(
                os.environ,
                {"TEST_API_KEY": "must-not-leak", "SAFE_PARENT_VALUE": "kept"},
            ):
                modeled = backend.model(task, candidate)

            self.assertFalse(modeled.metrics["secret_visible"])
            self.assertEqual(modeled.metrics["safe_environment"], "forwarded")
            attempts = list(Path(directory).glob("*/model_*"))
            self.assertEqual(len(attempts), 1)
            self.assertTrue((attempts[0] / "request.json").is_file())
            self.assertTrue((attempts[0] / "response.json").is_file())
            self.assertTrue((attempts[0] / "stdout.log").is_file())
            self.assertIn(
                "TEST_API_KEY",
                backend.last_stage_metadata["removed_sensitive_environment_names"],
            )


if __name__ == "__main__":
    unittest.main()
