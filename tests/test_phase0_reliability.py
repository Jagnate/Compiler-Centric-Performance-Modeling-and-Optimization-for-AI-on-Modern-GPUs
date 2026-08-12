"""Failure trace and checkpoint-resume tests for the reliable run foundation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from kernel_optimization.archive import ArtifactStore
from kernel_optimization.controller import OptimizationController
from kernel_optimization.schema import (
    BudgetConfig,
    Candidate,
    CandidateProposal,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    TaskSpec,
)


class PhaseZeroReliabilityTests(unittest.TestCase):
    def test_api_archive_redacts_secrets_but_keeps_token_accounting(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kernel-api-archive-") as directory:
            store = ArtifactStore(Path(directory))
            path = store.save_api_call(
                "test",
                {
                    "api_key": "secret",
                    "usage": {"prompt_tokens": 12, "completion_tokens": 7},
                },
                {"request": {"max_completion_tokens": 100}},
            )
            value = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(value["metadata"]["api_key"], "<redacted>")
        self.assertEqual(value["metadata"]["usage"]["prompt_tokens"], 12)
        self.assertEqual(
            value["exchange"]["request"]["max_completion_tokens"], 100
        )

    def test_interrupted_generation_resumes_without_repeating_seed(self) -> None:
        task = TaskSpec(
            task_id="resume-test",
            description="Exercise a round-boundary recovery.",
            reference="Return the input.",
            entrypoint="kernel",
            constraints={"required_fragments": ["def kernel"]},
            budget=BudgetConfig(
                rounds=1,
                proposals_per_round=1,
                beam_width=1,
                min_promotions_per_round=1,
                max_promotions_per_round=1,
            ),
        )
        backend = _CountingBackend()
        source = "VALUE = 0\n\ndef kernel(x):\n    return x\n"

        with tempfile.TemporaryDirectory(prefix="kernel-resume-test-") as directory:
            store = ArtifactStore(Path(directory))
            first = OptimizationController(
                task=task,
                source_code=source,
                source_name="kernel.py",
                generator=_FailingGenerator(),
                backend=backend,
                store=store,
            )
            with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
                first.run()

            self.assertEqual(backend.model_calls, 1)
            self.assertEqual(backend.measure_calls, 1)
            self.assertEqual(backend.profile_calls, 1)
            self.assertTrue((Path(directory) / "failure.json").is_file())

            resumed = OptimizationController(
                task=task,
                source_code=source,
                source_name="kernel.py",
                generator=_WorkingGenerator(),
                backend=backend,
                store=ArtifactStore(Path(directory)),
                resume=True,
            )
            summary = resumed.run()

            self.assertTrue(summary.resumed)
            self.assertEqual(summary.completed_rounds, 1)
            self.assertEqual(summary.generator_calls, 2)
            self.assertEqual(backend.model_calls, 2)
            self.assertEqual(backend.measure_calls, 2)
            events = [
                json.loads(line)
                for line in (Path(directory) / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertTrue(all("timestamp" in event for event in events))
            self.assertTrue(any(event["event"] == "run_failed" for event in events))
            self.assertTrue(any(event["event"] == "run_completed" for event in events))
            self.assertGreater(len(list((Path(directory) / "checkpoints").glob("*.json"))), 2)
            histories = list((Path(directory) / "candidates").glob("*/history/*.json"))
            self.assertGreater(len(histories), 2)


class _FailingGenerator:
    last_call_metadata = {}
    last_exchange = {}

    def generate(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("intentional interruption")


class _WorkingGenerator:
    last_call_metadata = {"usage": {"prompt_tokens": 5, "completion_tokens": 5}}
    last_exchange = {"request": {"test": True}, "response": {"test": True}}

    def generate(self, task, parent, evidence, history, count):
        del task, parent, evidence, history, count
        return [
            CandidateProposal(
                hypothesis="Use the faster test marker.",
                source_code="VALUE = 1\n\ndef kernel(x):\n    return x\n",
            )
        ]


class _CountingBackend:
    def __init__(self) -> None:
        self.model_calls = 0
        self.measure_calls = 0
        self.profile_calls = 0

    def model(self, task: TaskSpec, candidate: Candidate) -> ModelEvaluation:
        del task
        self.model_calls += 1
        latency = 2.0 if "VALUE = 0" in candidate.source_code else 1.0
        return ModelEvaluation(valid=True, predicted_latency_ms=latency)

    def measure(self, task: TaskSpec, candidate: Candidate) -> Measurement:
        del task
        self.measure_calls += 1
        latency = 2.0 if "VALUE = 0" in candidate.source_code else 1.0
        return Measurement(correct=True, latency_ms=latency, samples_ms=[latency])

    def profile(self, task: TaskSpec, candidate: Candidate) -> ProfileEvaluation:
        del task, candidate
        self.profile_calls += 1
        return ProfileEvaluation(bottleneck="test")


if __name__ == "__main__":
    unittest.main()
