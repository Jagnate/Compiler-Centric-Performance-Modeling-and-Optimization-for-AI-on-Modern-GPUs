"""End-to-end source optimization tests using a fake hosted API transport."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from kernel_optimization.archive import ArtifactStore
from kernel_optimization.controller import OptimizationController
from kernel_optimization.generators.api import ApiGeneratorConfig, OpenAICompatibleGenerator
from kernel_optimization.schema import (
    BudgetConfig,
    Candidate,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    TaskSpec,
)


SOURCE_TEMPLATE = "TILE = %d\n\ndef kernel(x):\n    return x\n"


def tile_value(source_code: str) -> int:
    match = re.search(r"^TILE = (\d+)$", source_code, re.MULTILINE)
    if match is None:
        raise ValueError("source has no TILE marker")
    return int(match.group(1))


class SourceOptimizationControllerTests(unittest.TestCase):
    def test_api_sources_are_filtered_measured_and_exported(self) -> None:
        task = TaskSpec(
            task_id="source-controller-test",
            description="Optimize a test source through the complete controller.",
            reference="kernel(x) returns x.",
            entrypoint="kernel",
            constraints={"required_fragments": ["def kernel"]},
            budget=BudgetConfig(
                rounds=2,
                proposals_per_round=4,
                beam_width=2,
                min_promotions_per_round=4,
                max_promotions_per_round=4,
                ncu_improvement_threshold=0.05,
                random_seed=3,
            ),
        )
        transport = _SourceTransport()
        generator = OpenAICompatibleGenerator(
            ApiGeneratorConfig(
                api_url="https://provider.example/v1/chat/completions",
                model="test-model",
                api_key_environment_variable="TEST_KERNEL_API_KEY",
            ),
            transport=transport,
        )
        backend = _SourceBackend()
        original_source = SOURCE_TEMPLATE % 0

        with tempfile.TemporaryDirectory(prefix="kernel-source-controller-") as directory:
            output = Path(directory)
            controller = OptimizationController(
                task=task,
                source_code=original_source,
                source_name="kernel.py",
                generator=generator,
                backend=backend,
                store=ArtifactStore(output),
                run_metadata={"generator": {"model": "test-model"}},
            )
            with mock.patch.dict(os.environ, {"TEST_KERNEL_API_KEY": "test-secret"}):
                summary = controller.run()

            self.assertEqual(summary.best_latency_ms, 1.0)
            self.assertEqual(summary.seed_latency_ms, 3.0)
            self.assertEqual(summary.speedup_over_seed, 3.0)
            self.assertEqual(
                Path(summary.best_source_path).read_text(encoding="utf-8"),
                SOURCE_TEMPLATE % 2,
            )
            self.assertEqual(original_source, SOURCE_TEMPLATE % 0)
            self.assertTrue(all(record.is_measured_correct for record in controller.beam))
            self.assertTrue(
                any(record.state == "correctness-failed" for record in controller.records.values())
            )
            self.assertEqual(summary.profile_calls, backend.profile_calls)
            self.assertLessEqual(summary.profile_calls, task.budget.rounds + 1)
            self.assertGreaterEqual(summary.generator_calls, task.budget.rounds)
            self.assertEqual(
                summary.generator_usage["prompt_tokens"],
                10.0 * summary.generator_calls,
            )
            self.assertGreater(summary.elapsed_seconds, 0.0)

            events = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(event["event"] == "proposal_rejected" for event in events))
            self.assertTrue(any(event["event"] == "candidate_deduplicated" for event in events))
            archived_task = json.loads(
                (output / "task.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                archived_task["run"]["generator"]["model"], "test-model"
            )
            candidate_sources = list((output / "candidates").glob("*/kernel.py"))
            self.assertEqual(len(candidate_sources), summary.generated_candidates)


class _SourceTransport:
    def __call__(self, url, headers, payload, timeout):
        del url, headers, timeout
        request = json.loads(payload["messages"][1]["content"])
        parent_value = tile_value(request["parent"]["source_code"])
        count = int(request["candidate_count"])
        if parent_value == 0:
            sources = [
                SOURCE_TEMPLATE % 1,
                SOURCE_TEMPLATE % 2,
                SOURCE_TEMPLATE % 3,
                "TILE = 4\n\ndef removed_entrypoint(x):\n    return x\n",
            ]
        elif parent_value == 2:
            sources = [SOURCE_TEMPLATE % 4, SOURCE_TEMPLATE % 0]
        else:
            sources = [SOURCE_TEMPLATE % 2, SOURCE_TEMPLATE % 3]
        candidates = [
            {
                "hypothesis": "Try source schedule marker %d." % index,
                "source_code": source,
                "expected_effect": {"latency": "unknown", "reason": "test"},
                "metadata": {"strategy": "test-source-generation"},
            }
            for index, source in enumerate(sources[:count])
        ]
        return {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [{"message": {"content": json.dumps({"candidates": candidates})}}]
        }


class _SourceBackend:
    def __init__(self) -> None:
        self.model_calls = 0
        self.measure_calls = 0
        self.profile_calls = 0

    def model(self, task: TaskSpec, candidate: Candidate) -> ModelEvaluation:
        del task
        self.model_calls += 1
        value = tile_value(candidate.source_code)
        measured = abs(value - 2) + 1.0
        predicted = measured * (1.15 if value == 1 else 0.95)
        return ModelEvaluation(
            valid=True,
            predicted_latency_ms=predicted,
            bottleneck="tensor-core",
            confidence="medium",
        )

    def measure(self, task: TaskSpec, candidate: Candidate) -> Measurement:
        del task
        self.measure_calls += 1
        value = tile_value(candidate.source_code)
        if value == 3:
            return Measurement(correct=False, error="Intentional test mismatch.")
        latency = abs(value - 2) + 1.0
        return Measurement(correct=True, latency_ms=latency, samples_ms=[latency])

    def profile(self, task: TaskSpec, candidate: Candidate) -> ProfileEvaluation:
        del task, candidate
        self.profile_calls += 1
        return ProfileEvaluation(
            bottleneck="tensor-core",
            metrics={"profile_source": "test-fake"},
        )


if __name__ == "__main__":
    unittest.main()
