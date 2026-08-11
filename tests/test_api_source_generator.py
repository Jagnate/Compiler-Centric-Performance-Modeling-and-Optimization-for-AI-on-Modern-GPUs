"""Tests for hosted source generation without network access."""

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
        task_id="api-source-test",
        description="Optimize a TileLang kernel source.",
        reference="Preserve the reference output.",
        entrypoint="make_kernel",
        target={"architecture": "rtx3090"},
        workload={"shape": [2048, 2048]},
    )


class ApiSourceGeneratorTests(unittest.TestCase):
    def test_fake_transport_receives_source_and_parses_complete_replacement(self) -> None:
        captured = {}

        def transport(url, headers, payload, timeout):
            captured.update(
                {"url": url, "headers": headers, "payload": payload, "timeout": timeout}
            )
            content = {
                "candidates": [
                    {
                        "hypothesis": "Use a deeper software pipeline.",
                        "source_code": (
                            "```python\ndef make_kernel():\n"
                            "    stages = 4\n    return stages\n```"
                        ),
                        "expected_effect": {
                            "latency": "decrease",
                            "bottleneck": "memory-latency",
                            "reason": "Increase overlap.",
                        },
                        "metadata": {"strategy": "pipeline"},
                    }
                ]
            }
            return {
                "id": "response-test",
                "model": "resolved-test-model",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "```json\n%s\n```" % json.dumps(content)},
                    }
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
        parent = Candidate.seed(
            task,
            "def make_kernel():\n    return 1\n",
            "kernel.py",
        )
        evidence = {
            "observed": {"measurement": {"latency_ms": 0.4}},
            "predicted": {"predicted_latency_ms": 0.3},
            "model_trust": {"score": 0.5},
        }

        with mock.patch.dict(os.environ, {"TEST_KERNEL_API_KEY": "test-secret"}):
            proposals = generator.generate(task, parent, evidence, [], count=1)

        self.assertEqual(
            proposals[0].source_code,
            "def make_kernel():\n    stages = 4\n    return stages\n",
        )
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-secret")
        request = json.loads(captured["payload"]["messages"][1]["content"])
        self.assertEqual(request["parent"]["source_code"], parent.source_code)
        self.assertNotIn("search_space", request["task"])
        self.assertEqual(request["task"]["target"]["architecture"], "rtx3090")
        self.assertIn("complete replacement", request["rules"][0])
        self.assertEqual(generator.last_call_metadata["response_id"], "response-test")
        self.assertEqual(generator.last_call_metadata["usage"]["prompt_tokens"], 100)
        prompts = "\n".join(
            message["content"] for message in captured["payload"]["messages"]
        )
        self.assertIsNone(re.search(r"[\u4e00-\u9fff]", prompts))

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
        parent = Candidate.seed(task, "def make_kernel():\n    return 1\n", "kernel.py")

        with mock.patch.dict(
            os.environ, {"DEFINITELY_MISSING_KERNEL_API_KEY": ""}, clear=False
        ):
            with self.assertRaisesRegex(RuntimeError, "is not set"):
                generator.generate(task, parent, {}, [], count=1)


if __name__ == "__main__":
    unittest.main()
