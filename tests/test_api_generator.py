"""Tests for hosted candidate generation without network access."""

from __future__ import annotations

import json
import os
import re
import unittest
from unittest import mock

from kernel_optimization.generators.api import (
    ApiGeneratorConfig,
    OpenAICompatibleGenerator,
)
from kernel_optimization.schema import Candidate, TaskSpec


def make_task() -> TaskSpec:
    return TaskSpec(
        task_id="api-test",
        description="Optimize a tiled matrix multiplication schedule.",
        reference="C = A @ B with float32 accumulation.",
        base_parameters={"num_stages": 3, "threads": 128},
        search_space={"num_stages": (1, 2, 3, 4), "threads": (64, 128, 256)},
    )


class ApiGeneratorTests(unittest.TestCase):
    def test_fake_transport_receives_provenance_prompt_and_parses_json_fence(self) -> None:
        captured = {}

        def transport(url, headers, payload, timeout):
            captured.update(
                {"url": url, "headers": headers, "payload": payload, "timeout": timeout}
            )
            content = {
                "candidates": [
                    {
                        "hypothesis": "Reduce stages to lower register pressure.",
                        "parameter_updates": {"num_stages": 2},
                        "expected_effect": {
                            "latency": "decrease",
                            "reason": "Potentially higher occupancy.",
                        },
                        "source_patch": None,
                        "metadata": {"strategy": "occupancy"},
                    },
                    {
                        "hypothesis": "Try a wider thread block.",
                        "parameters": {"threads": 256},
                    },
                ]
            }
            return {
                "choices": [
                    {"message": {"content": "```json\n%s\n```" % json.dumps(content)}}
                ]
            }

        generator = OpenAICompatibleGenerator(
            ApiGeneratorConfig(
                api_url="https://provider.example/v1/chat/completions",
                model="test-model",
                api_key_environment_variable="TEST_KERNEL_API_KEY",
                timeout_seconds=17.0,
            ),
            transport=transport,
        )
        task = make_task()
        parent = Candidate.seed(task)
        evidence = {
            "observed": {"measurement": {"latency_ms": 0.4}},
            "predicted": {"predicted_latency_ms": 0.3},
            "model_trust": {"score": 0.5},
        }

        with mock.patch.dict(os.environ, {"TEST_KERNEL_API_KEY": "test-secret"}):
            proposals = generator.generate(task, parent, evidence, [], count=2)

        self.assertEqual([item.parameter_updates for item in proposals], [
            {"num_stages": 2},
            {"threads": 256},
        ])
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-secret")
        self.assertEqual(captured["timeout"], 17.0)
        messages = captured["payload"]["messages"]
        request = json.loads(messages[1]["content"])
        self.assertEqual(
            request["observed_evidence"]["measurement"]["latency_ms"], 0.4
        )
        self.assertEqual(
            request["predicted_evidence"]["predicted_latency_ms"], 0.3
        )
        self.assertIn("may be wrong", messages[0]["content"])
        self.assertIsNone(re.search(r"[\u4e00-\u9fff]", messages[0]["content"]))
        self.assertIsNone(re.search(r"[\u4e00-\u9fff]", messages[1]["content"]))

    def test_missing_api_key_fails_before_transport(self) -> None:
        generator = OpenAICompatibleGenerator(
            ApiGeneratorConfig(
                api_url="https://provider.example/v1/chat/completions",
                model="test-model",
                api_key_environment_variable="DEFINITELY_MISSING_KERNEL_API_KEY",
            ),
            transport=lambda *args: self.fail("transport must not be called"),
        )
        task = make_task()

        with mock.patch.dict(
            os.environ, {"DEFINITELY_MISSING_KERNEL_API_KEY": ""}, clear=False
        ):
            with self.assertRaisesRegex(RuntimeError, "is not set"):
                generator.generate(task, Candidate.seed(task), {}, [], count=1)

    def test_malformed_response_is_rejected(self) -> None:
        generator = OpenAICompatibleGenerator(
            ApiGeneratorConfig(
                api_url="https://provider.example/v1/chat/completions",
                model="test-model",
                api_key_environment_variable="TEST_KERNEL_API_KEY",
            ),
            transport=lambda *args: {"choices": [{"message": {"content": "{}"}}]},
        )
        task = make_task()

        with mock.patch.dict(os.environ, {"TEST_KERNEL_API_KEY": "test-secret"}):
            with self.assertRaisesRegex(ValueError, "candidates list"):
                generator.generate(task, Candidate.seed(task), {}, [], count=1)


if __name__ == "__main__":
    unittest.main()

