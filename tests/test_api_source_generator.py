"""Tests for hosted source generation without network access."""

from __future__ import annotations

import json
import os
import re
import unittest
from unittest import mock

from kernel_optimization.generators.api import (
    ApiGeneratorConfig,
    HostedApiError,
    OpenAICompatibleGenerator,
    _request_too_large,
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
    def test_strategy_planner_uses_small_budget_and_returns_allocation(self) -> None:
        captured = {}

        def transport(url, headers, payload, timeout):
            del url, headers, timeout
            captured.update(payload)
            return {
                "id": "planner-test",
                "model": "resolved-test-model",
                "usage": {"prompt_tokens": 80, "completion_tokens": 40},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "allocation": [
                                        {
                                            "strategy_id": "data-movement",
                                            "count": 4,
                                            "mode": "exploit",
                                            "reason": "Measured bandwidth pressure.",
                                            "evidence": ["profile.ddr_util"],
                                        },
                                        {
                                            "strategy_id": "pipeline-structure",
                                            "count": 2,
                                            "mode": "explore",
                                            "reason": "Overlap remains uncertain.",
                                            "evidence": ["model.diagnostics"],
                                        },
                                    ],
                                    "round_rationale": "Exploit traffic reduction.",
                                    "confidence": 0.75,
                                }
                            )
                        },
                    }
                ],
            }

        generator = OpenAICompatibleGenerator(
            ApiGeneratorConfig(
                api_url="https://provider.example/v1/chat/completions",
                model="test-model",
                api_key_environment_variable="TEST_KERNEL_API_KEY",
                max_output_tokens=8000,
                planner_max_output_tokens=1500,
            ),
            transport=transport,
        )
        task = make_task()
        parent = Candidate.seed(
            task, "def make_kernel():\n    return 1\n", "kernel.py"
        )
        strategies = [
            {
                "strategy_id": "data-movement",
                "title": "Data movement",
                "structural_required": True,
            }
        ]
        with mock.patch.dict(os.environ, {"TEST_KERNEL_API_KEY": "test-secret"}):
            plan = generator.plan_strategies(
                task,
                parent,
                {"strategy_outcomes": {}},
                strategies,
                count=6,
                round_number=1,
            )

        self.assertEqual(plan["allocation"][0]["count"], 4)
        self.assertEqual(captured["max_completion_tokens"], 1500)
        request = json.loads(captured["messages"][1]["content"])
        self.assertEqual(request["candidate_count"], 6)
        self.assertTrue(request["controller_constraints"]["parameter_tuning_is_optional"])
        self.assertIn("source_code", request["current_best"])
        self.assertEqual(generator.last_call_metadata["kind"], "plan")
        planner_prompts = "\n".join(
            message["content"] for message in captured["messages"]
        )
        self.assertIsNone(re.search(r"[\u4e00-\u9fff]", planner_prompts))

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

    def test_partially_malformed_response_keeps_alias_source_candidate(self) -> None:
        calls = []

        def transport(url, headers, payload, timeout):
            del url, headers, timeout
            calls.append(payload)
            return {
                "id": "partial-response",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidates": [
                                        {
                                            "hypothesis": "Description only.",
                                            "metadata": {"strategy": "invalid"},
                                        },
                                        {
                                            "rationale": "Use a smaller tile.",
                                            "code": (
                                                "```python\ndef make_kernel():\n"
                                                "    return 2\n```"
                                            ),
                                            "strategy_id": "parameter-tuning",
                                        },
                                    ]
                                }
                            )
                        }
                    }
                ],
            }

        generator = OpenAICompatibleGenerator(
            ApiGeneratorConfig(
                api_url="https://provider.example/v1/chat/completions",
                model="test-model",
                api_key_environment_variable="TEST_KERNEL_API_KEY",
            ),
            transport=transport,
        )
        task = make_task()
        parent = Candidate.seed(
            task, "def make_kernel():\n    return 1\n", "kernel.py"
        )
        with mock.patch.dict(os.environ, {"TEST_KERNEL_API_KEY": "test-secret"}):
            proposals = generator.generate(task, parent, {}, [], count=2)

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].hypothesis, "Use a smaller tile.")
        self.assertEqual(
            proposals[0].source_code, "def make_kernel():\n    return 2\n"
        )
        self.assertEqual(
            proposals[0].metadata["strategy_id"], "parameter-tuning"
        )
        parsing = generator.last_call_metadata["candidate_parsing"]
        self.assertEqual(parsing["accepted_candidates"], 1)
        self.assertEqual(parsing["rejected_candidates"], 1)
        self.assertEqual(
            parsing["normalizations"][0]["source_field"], "candidate.code"
        )

    def test_missing_sources_trigger_one_accounted_schema_recovery(self) -> None:
        payloads = []

        def transport(url, headers, payload, timeout):
            del url, headers, timeout
            payloads.append(payload)
            if len(payloads) == 1:
                candidates = [
                    {
                        "hypothesis": "Metadata-only response.",
                        "expected_effect": {"latency": "decrease"},
                    }
                ]
                response_id = "missing-source"
            else:
                candidates = [
                    {
                        "hypothesis": "Recovered full source.",
                        "source_code": "def make_kernel():\n    return 3\n",
                    }
                ]
                response_id = "schema-recovered"
            return {
                "id": response_id,
                "usage": {"prompt_tokens": 40, "completion_tokens": 20},
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"candidates": candidates})
                        }
                    }
                ],
            }

        generator = OpenAICompatibleGenerator(
            ApiGeneratorConfig(
                api_url="https://provider.example/v1/chat/completions",
                model="test-model",
                api_key_environment_variable="TEST_KERNEL_API_KEY",
            ),
            transport=transport,
        )
        task = make_task()
        parent = Candidate.seed(
            task, "def make_kernel():\n    return 1\n", "kernel.py"
        )
        with mock.patch.dict(os.environ, {"TEST_KERNEL_API_KEY": "test-secret"}):
            proposals = generator.generate(task, parent, {}, [], count=6)

        self.assertEqual(len(payloads), 2)
        recovery_request = json.loads(payloads[1]["messages"][1]["content"])
        self.assertEqual(recovery_request["candidate_count"], 1)
        self.assertIn("response_recovery", recovery_request)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].source_code.splitlines()[-1], "    return 3")
        metadata = generator.last_call_metadata
        self.assertEqual(metadata["attempts"], 2)
        self.assertEqual(metadata["response_validation_attempts"], 2)
        self.assertEqual(metadata["response_ids"], ["missing-source", "schema-recovered"])
        self.assertEqual(metadata["usage"]["prompt_tokens"], 80)
        self.assertEqual(metadata["usage"]["completion_tokens"], 40)
        self.assertEqual(
            metadata["candidate_parsing_attempts"][0]["rejected_candidates"], 1
        )
        self.assertEqual(
            metadata["candidate_parsing_attempts"][1]["accepted_candidates"], 1
        )

    def test_two_metadata_only_responses_fail_with_parse_diagnostics(self) -> None:
        calls = []

        def transport(url, headers, payload, timeout):
            del url, headers, timeout
            calls.append(payload)
            return {
                "id": "still-missing-%d" % len(calls),
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidates": [
                                        {"hypothesis": "Still no source."}
                                    ]
                                }
                            )
                        }
                    }
                ],
            }

        generator = OpenAICompatibleGenerator(
            ApiGeneratorConfig(
                api_url="https://provider.example/v1/chat/completions",
                model="test-model",
                api_key_environment_variable="TEST_KERNEL_API_KEY",
            ),
            transport=transport,
        )
        task = make_task()
        parent = Candidate.seed(
            task, "def make_kernel():\n    return 1\n", "kernel.py"
        )
        with mock.patch.dict(os.environ, {"TEST_KERNEL_API_KEY": "test-secret"}):
            with self.assertRaisesRegex(ValueError, "schema-recovery"):
                generator.generate(task, parent, {}, [], count=6)

        self.assertEqual(len(calls), 2)
        metadata = generator.last_call_metadata
        self.assertEqual(metadata["response_validation_attempts"], 2)
        self.assertEqual(metadata["usage"]["prompt_tokens"], 20)
        self.assertEqual(
            metadata["candidate_parsing"]["rejections"][0]["index"], 0
        )
        self.assertEqual(
            len(generator.last_exchange["response_validation_attempts"]), 2
        )

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

    def test_retryable_failure_is_retried_and_archived_in_metadata(self) -> None:
        calls = []
        sleeps = []

        def transport(url, headers, payload, timeout):
            del url, headers, payload, timeout
            calls.append(1)
            if len(calls) == 1:
                raise HostedApiError(
                    "temporary rate limit",
                    status_code=429,
                    error_code="rate_limit_exceeded",
                    retryable=True,
                    retry_after_seconds=0.25,
                )
            return {
                "id": "retry-success",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidates": [
                                        {
                                            "hypothesis": "Retry succeeded.",
                                            "source_code": "def make_kernel():\n    return 2\n",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ],
            }

        generator = OpenAICompatibleGenerator(
            ApiGeneratorConfig(
                api_url="https://provider.example/v1/chat/completions",
                model="test-model",
                api_key_environment_variable="TEST_KERNEL_API_KEY",
            ),
            transport=transport,
            sleeper=sleeps.append,
        )
        task = make_task()
        parent = Candidate.seed(task, "def make_kernel():\n    return 1\n", "kernel.py")
        with mock.patch.dict(os.environ, {"TEST_KERNEL_API_KEY": "test-secret"}):
            proposals = generator.generate(task, parent, {}, [], count=1)

        self.assertEqual(len(proposals), 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(sleeps, [0.25])
        self.assertEqual(generator.last_call_metadata["attempts"], 2)

    def test_nonretryable_quota_failure_stops_immediately(self) -> None:
        calls = []

        def transport(*args):
            calls.append(args)
            raise HostedApiError(
                "insufficient quota",
                status_code=429,
                error_code="insufficient_quota",
                retryable=False,
            )

        generator = OpenAICompatibleGenerator(
            ApiGeneratorConfig(
                api_url="https://provider.example/v1/chat/completions",
                model="test-model",
                api_key_environment_variable="TEST_KERNEL_API_KEY",
            ),
            transport=transport,
            sleeper=lambda delay: self.fail("must not sleep"),
        )
        with mock.patch.dict(os.environ, {"TEST_KERNEL_API_KEY": "test-secret"}):
            with self.assertRaisesRegex(HostedApiError, "insufficient quota"):
                generator.preflight()
        self.assertEqual(len(calls), 1)
        self.assertEqual(generator.last_call_metadata["error"]["error_code"], "insufficient_quota")

    def test_preflight_uses_a_small_request(self) -> None:
        captured = {}

        def transport(url, headers, payload, timeout):
            del url, headers, timeout
            captured.update(payload)
            return {"id": "preflight-ok", "choices": [{"message": {"content": "{}"}}]}

        generator = OpenAICompatibleGenerator(
            ApiGeneratorConfig(
                api_url="https://provider.example/v1/chat/completions",
                model="test-model",
                api_key_environment_variable="TEST_KERNEL_API_KEY",
                max_output_tokens=12000,
            ),
            transport=transport,
        )
        with mock.patch.dict(os.environ, {"TEST_KERNEL_API_KEY": "test-secret"}):
            metadata = generator.preflight()

        self.assertEqual(metadata["kind"], "preflight")
        self.assertLessEqual(captured["max_completion_tokens"], 128)
        self.assertNotIn("temperature", captured)

    def test_oversized_token_request_is_not_treated_as_transient(self) -> None:
        body = json.dumps(
            {
                "error": {
                    "message": (
                        "Request too large for this model. The input or output "
                        "tokens must be reduced in order to run successfully."
                    ),
                    "type": "tokens",
                    "code": "rate_limit_exceeded",
                }
            }
        )
        self.assertTrue(_request_too_large(body))
        self.assertFalse(_request_too_large('{"error":{"message":"Try later"}}'))


if __name__ == "__main__":
    unittest.main()
