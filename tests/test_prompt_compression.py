"""Bounded prompt context and local request-budget tests."""

from __future__ import annotations

from copy import deepcopy
import json
import os
import unittest
from unittest import mock

from kernel_optimization.generators.api import (
    ApiGeneratorConfig,
    OpenAICompatibleGenerator,
)
from kernel_optimization.prompt_compression import PromptBudgetError
from kernel_optimization.prompts import build_optimization_prompt
from kernel_optimization.schema import Candidate, TaskSpec


def make_task() -> TaskSpec:
    return TaskSpec(
        task_id="prompt-compression-test",
        description="Optimize a complete TileLang source file.",
        reference="Preserve the mathematical result.",
        entrypoint="make_kernel",
        target={"architecture": "rtx3090"},
        workload={"factory_arguments": {"m": 2048, "n": 2048, "k": 2048}},
    )


class PromptCompressionTests(unittest.TestCase):
    def test_huge_raw_evidence_becomes_a_bounded_key_metric_prompt(self) -> None:
        task = make_task()
        source = "def make_kernel():\n    return 1\n"
        parent = Candidate.seed(task, source, "kernel.py")
        sentinel = "RAW_NCU_SENTINEL_" + ("x" * 20000)
        evidence = _huge_evidence(sentinel)
        history = _huge_history(sentinel)
        original_evidence = deepcopy(evidence)
        original_history = deepcopy(history)

        prompt = build_optimization_prompt(task, parent, evidence, history, count=2)
        request = json.loads(prompt)

        self.assertLess(len(prompt), 50000)
        self.assertEqual(request["parent"]["source_code"], source)
        self.assertEqual(
            request["observed_evidence"]["profile"]["metrics"]["ddr_util"],
            81.5,
        )
        self.assertEqual(
            request["predicted_evidence"]["metrics"]["registers_per_thread"],
            144,
        )
        self.assertNotIn("RAW_NCU_SENTINEL_" + ("x" * 1000), prompt)
        self.assertNotIn("raw_metrics", prompt)
        self.assertNotIn("captured_passes", prompt)
        self.assertEqual(len(request["shared_evidence_memory"]), 8)
        self.assertEqual(len(request["recent_history"]), 8)
        provenance = request["context_provenance"]
        self.assertEqual(provenance["compression_version"], "key-metrics-v1")
        self.assertGreater(
            provenance["raw_context_characters"],
            provenance["compressed_context_characters"] * 10,
        )
        self.assertEqual(evidence, original_evidence)
        self.assertEqual(history, original_history)

    def test_local_input_budget_fails_before_network_access(self) -> None:
        observed_sizes = []
        generator = OpenAICompatibleGenerator(
            ApiGeneratorConfig(
                api_url="https://provider.example/v1/chat/completions",
                model="test-model",
                api_key_environment_variable="TEST_KERNEL_API_KEY",
                max_input_tokens=20,
            ),
            transport=lambda *args: self.fail("transport must not be called"),
            request_observer=observed_sizes.append,
        )
        task = make_task()
        parent = Candidate.seed(
            task, "def make_kernel():\n    return 1\n", "kernel.py"
        )

        with mock.patch.dict(os.environ, {"TEST_KERNEL_API_KEY": "test-secret"}):
            with self.assertRaisesRegex(PromptBudgetError, "local.*budget"):
                generator.generate(task, parent, {}, [], count=1)

        self.assertEqual(generator.total_api_requests, 0)
        self.assertEqual(generator.last_call_metadata["attempts"], 0)
        self.assertEqual(
            generator.last_call_metadata["error"]["error_code"],
            "local_prompt_budget_exceeded",
        )
        self.assertGreater(observed_sizes[0]["estimated_input_tokens"], 20)
        self.assertEqual(observed_sizes[0]["max_input_tokens"], 20)


def _huge_evidence(sentinel: str):
    raw_metrics = {"metric_%05d" % index: float(index) for index in range(5000)}
    raw_metrics["sentinel"] = sentinel
    lessons = []
    for index in range(20):
        lessons.append(
            {
                "lesson_id": "lesson-%02d" % index,
                "round_number": index,
                "candidate_id": "candidate-%02d" % index,
                "parent_id": "parent-%02d" % index,
                "kind": "hardware-profile",
                "bottleneck": "compute-tensor",
                "observed_fact": "Profiled candidate %d" % index,
                "hypothesis_tested": "Tune a tile shape.",
                "result": "Latency improved.",
                "confidence": "high",
                "supporting_evidence": {
                    "profile_metrics": {
                        "ddr_util": 81.5,
                        "registers_per_thread": 144,
                        "raw_metrics": raw_metrics,
                    },
                    "measured_latency_ms": 0.3,
                },
                "source_regions": ["make_kernel"],
                "recommended_actions": ["Reduce register pressure."],
            }
        )
    return {
        "observed": {
            "measurement": {
                "correct": True,
                "latency_ms": 0.3,
                "samples_ms": [0.29, 0.3, 0.31],
                "metrics": {
                    "measurement_source": "cuda-events",
                    "primary_case_id": "primary",
                    "statistics": {
                        "sample_count": 3,
                        "coefficient_of_variation": 0.02,
                    },
                    "cases": [
                        {
                            "case_id": "primary",
                            "factory_arguments": {"m": 2048, "n": 2048, "k": 2048},
                            "runtime": {
                                "benchmark_latency_ms": 0.3,
                                "environment": {"compiler_log": sentinel},
                            },
                        }
                    ],
                },
            },
            "profile": {
                "valid": True,
                "bottleneck": "dram-bandwidth",
                "metrics": {
                    "ddr_util": 81.5,
                    "l2_hit_rate": 0.7,
                    "registers_per_thread": 144,
                    "raw_metrics": raw_metrics,
                    "raw_units": {key: "%" for key in raw_metrics},
                },
            },
            "diagnosis": {
                "category": "occupancy-limited",
                "confidence": "high",
                "summary": "Register pressure limits occupancy.",
                "recommendations": ["Reduce live accumulator state."],
                "metrics": {"achieved_occupancy_percent": 25.0},
            },
        },
        "predicted": {
            "valid": True,
            "predicted_latency_ms": 0.2,
            "bottleneck": "tensor-core",
            "confidence": "medium",
            "metrics": {
                "registers_per_thread": 144,
                "shared_memory_per_block": 98304,
                "tensor_util": 69.0,
                "captured_passes": [sentinel] * 20,
                "resource_provenance": {"generated_source": sentinel},
            },
        },
        "model_trust": {"score": 0.5, "measured_samples": 4},
        "calibration": {
            "policy": "median-scale",
            "observation_count": 10,
            "groups": {
                "group-%02d" % index: {"samples": index, "median_scale": 1.1}
                for index in range(20)
            },
        },
        "shared_memory": lessons,
    }


def _huge_history(sentinel: str):
    return [
        {
            "candidate_id": "candidate-%02d" % index,
            "parent_id": "parent-%02d" % index,
            "generation": index,
            "hypothesis": "Try schedule %d" % index,
            "state": "static-invalid" if index % 2 else "measured",
            "predicted_latency_ms": 0.4,
            "measured_latency_ms": 0.5,
            "failure": {
                "stage": "compile",
                "category": "layout-inference",
                "message": sentinel,
                "diagnostics": [sentinel] * 20,
                "details": {"compiler_dump": sentinel},
            },
        }
        for index in range(30)
    ]


if __name__ == "__main__":
    unittest.main()
