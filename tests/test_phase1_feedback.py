"""Tests for repair, global evidence, diagnosis, and search-local calibration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from kernel_optimization.archive import ArtifactStore
from kernel_optimization.calibration import LatencyCalibrator
from kernel_optimization.controller import OptimizationController
from kernel_optimization.diagnosis import BottleneckAnalyzer, FailureClassifier
from kernel_optimization.evidence import GlobalEvidenceMemory
from kernel_optimization.generators.api import ApiGeneratorConfig, OpenAICompatibleGenerator
from kernel_optimization.schema import (
    BudgetConfig,
    Candidate,
    CandidateProposal,
    CandidateRecord,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    TaskSpec,
)


def make_task(**budget_overrides) -> TaskSpec:
    budget = {
        "rounds": 1,
        "proposals_per_round": 2,
        "beam_width": 2,
        "min_promotions_per_round": 2,
        "max_promotions_per_round": 2,
        "max_repairs_per_round": 2,
        "max_repair_depth": 2,
        "calibration_min_samples": 2,
    }
    budget.update(budget_overrides)
    return TaskSpec(
        task_id="phase1-feedback-test",
        description="Exercise feedback-driven source repair.",
        reference="Return the input.",
        entrypoint="kernel",
        target={"architecture": "test-gpu"},
        constraints={"required_fragments": ["def kernel"]},
        budget=BudgetConfig(**budget),
        metadata={"kernel_family": "test-family"},
    )


class RepairFeedbackTests(unittest.TestCase):
    def test_static_and_correctness_failures_are_repaired_with_lineage(self) -> None:
        task = make_task()
        generator = _RepairingGenerator()
        backend = _RepairBackend()
        seed_source = "VALUE = 0\n\ndef kernel(x):\n    return x\n"

        with tempfile.TemporaryDirectory(prefix="kernel-phase1-") as directory:
            output = Path(directory)
            controller = OptimizationController(
                task=task,
                source_code=seed_source,
                source_name="kernel.py",
                generator=generator,
                backend=backend,
                store=ArtifactStore(output),
            )
            summary = controller.run()

            self.assertEqual(summary.repair_calls, 2)
            self.assertEqual(summary.repair_candidates, 2)
            self.assertEqual(summary.best_latency_ms, 0.5)
            self.assertGreaterEqual(summary.evidence_lessons, 4)
            repairs = [
                item
                for item in controller.records.values()
                if item.candidate.lineage_kind == "repair"
            ]
            self.assertEqual(len(repairs), 2)
            self.assertTrue(all(item.candidate.repair_depth == 1 for item in repairs))
            self.assertTrue(
                all(item.candidate.parent_id in controller.records for item in repairs)
            )
            self.assertTrue(
                any(
                    item.model
                    and item.model.calibration.get("applied")
                    for item in repairs
                    if item.is_measured_correct
                )
            )
            failures = [
                item.failure.category
                for item in controller.records.values()
                if item.failure is not None
            ]
            self.assertIn("interface-contract", failures)
            self.assertIn("correctness", failures)
            memory = json.loads(
                (output / "evidence_memory.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(memory["lessons"]), summary.evidence_lessons)
            self.assertTrue(generator.repair_evidence)
            self.assertTrue(generator.repair_evidence[0]["shared_memory"])


class DiagnosisTests(unittest.TestCase):
    def test_resource_pressure_and_low_occupancy_are_classified(self) -> None:
        task = make_task(max_repairs_per_round=0)
        candidate = Candidate.seed(task, "def kernel(x):\n    return x\n", "kernel.py")
        record = CandidateRecord(
            candidate=candidate,
            model=ModelEvaluation(
                valid=True,
                predicted_latency_ms=1.0,
                bottleneck="tensor-core",
                metrics={"registers_per_thread": 160, "tensor_util": 45.0},
            ),
            measurement=Measurement(correct=True, latency_ms=1.2),
            profile=ProfileEvaluation(
                bottleneck="occupancy",
                metrics={
                    "achieved_occupancy": 0.25,
                    "registers_per_thread": 160,
                    "tensor_util": 48.0,
                },
            ),
        )
        diagnosis = BottleneckAnalyzer().diagnose(record)
        self.assertEqual(diagnosis.category, "occupancy-limited")
        self.assertEqual(diagnosis.confidence, "high")
        self.assertIn("high registers per thread", diagnosis.limiting_factors)

    def test_failure_classifier_preserves_actionable_layout_diagnostic(self) -> None:
        model = ModelEvaluation(
            valid=False,
            diagnostics=["InternalError: Layout infer conflict between fragments"],
        )
        failure = FailureClassifier.model(model)
        self.assertEqual(failure.category, "layout-inference")
        self.assertIn("Layout infer conflict", failure.message)

    def test_raw_bank_conflict_count_is_normalized_before_classification(self) -> None:
        task = make_task(max_repairs_per_round=0)
        candidate = Candidate.seed(task, "def kernel(x):\n    return x\n", "kernel.py")
        record = CandidateRecord(
            candidate=candidate,
            model=ModelEvaluation(valid=True, predicted_latency_ms=1.0),
            measurement=Measurement(correct=True, latency_ms=1.0),
            profile=ProfileEvaluation(
                bottleneck="dram-bandwidth",
                metrics={
                    "ddr_util": 0.92,
                    "l2_util": 0.35,
                    "smem_util": 0.10,
                    "raw_metrics": {
                        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": 4737,
                        "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum": 480126,
                    },
                },
            ),
        )

        diagnosis = BottleneckAnalyzer().diagnose(record)

        self.assertEqual(diagnosis.category, "memory-dram")
        self.assertAlmostEqual(
            diagnosis.metrics["bank_conflict_indicator"],
            480126.0 / (480126.0 - 4737.0),
        )
        self.assertAlmostEqual(
            diagnosis.metrics["bank_conflict_fraction"], 4737.0 / 480126.0
        )

    def test_reported_transaction_amplification_can_identify_conflicts(self) -> None:
        task = make_task(max_repairs_per_round=0)
        candidate = Candidate.seed(task, "def kernel(x):\n    return x\n", "kernel.py")
        record = CandidateRecord(
            candidate=candidate,
            model=ModelEvaluation(valid=True, predicted_latency_ms=1.0),
            measurement=Measurement(correct=True, latency_ms=1.0),
            profile=ProfileEvaluation(
                bottleneck="shared-memory",
                metrics={
                    "shared_transactions_per_request": 1.5,
                    "smem_util": 0.40,
                },
            ),
        )

        diagnosis = BottleneckAnalyzer().diagnose(record)

        self.assertEqual(diagnosis.category, "shared-memory-bank-conflict")

    def test_disk_quota_failure_is_not_repairable_correctness(self) -> None:
        failure = FailureClassifier.measurement(
            Measurement(
                correct=False,
                error="RuntimeError: [Errno 122] Disk quota exceeded: cache.cubin",
            )
        )

        self.assertEqual(failure.category, "infrastructure-storage")
        self.assertFalse(failure.retryable)
        self.assertTrue(failure.details["infrastructure_failure"])

    def test_profiler_permission_failure_is_infrastructure(self) -> None:
        failure = FailureClassifier.profile(
            "ERR_NVGPUCTRPERM: permission to access NVIDIA GPU performance counters"
        )

        self.assertEqual(failure.category, "infrastructure-profiler-permission")
        self.assertEqual(failure.stage, "profile")
        self.assertFalse(failure.retryable)


class InfrastructureCircuitBreakerTests(unittest.TestCase):
    def test_repeated_storage_failures_stop_without_llm_repair(self) -> None:
        task = make_task(
            proposals_per_round=2,
            min_promotions_per_round=2,
            max_promotions_per_round=2,
            max_repairs_per_round=2,
        )
        generator = _InfrastructureGenerator()
        backend = _InfrastructureBackend()

        with tempfile.TemporaryDirectory(prefix="kernel-infrastructure-") as directory:
            controller = OptimizationController(
                task=task,
                source_code="VALUE = 0\n\ndef kernel(x):\n    return x\n",
                source_name="kernel.py",
                generator=generator,
                backend=backend,
                store=ArtifactStore(Path(directory)),
            )

            with self.assertRaisesRegex(RuntimeError, "consecutive infrastructure-storage"):
                controller.run()

        self.assertEqual(generator.repair_calls, 0)
        failures = [
            record
            for record in controller.records.values()
            if record.state == "infrastructure-failed"
        ]
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(not record.failure.retryable for record in failures))


class CalibrationAndMemoryTests(unittest.TestCase):
    def test_regime_calibration_keeps_raw_and_adds_robust_prediction(self) -> None:
        task = make_task(max_repairs_per_round=0)
        calibrator = LatencyCalibrator(min_samples=2)
        for index, (raw, measured) in enumerate(((1.0, 2.0), (2.0, 4.0))):
            candidate = Candidate._create(
                task=task,
                parent_id=None,
                generation=index,
                source_name="kernel.py",
                source_code="VALUE = %d\n\ndef kernel(x):\n    return x\n" % index,
                hypothesis="calibration sample",
                expected_effect={},
                proposal_metadata={},
                lineage_kind="proposal",
                repair_depth=0,
            )
            record = CandidateRecord(
                candidate=candidate,
                model=ModelEvaluation(
                    valid=True,
                    predicted_latency_ms=raw,
                    bottleneck="tensor-core",
                    metrics={
                        "registers_per_thread": 64,
                        "shared_memory_per_block": 16384,
                    },
                ),
                measurement=Measurement(correct=True, latency_ms=measured),
            )
            calibrator.observe(task, record)

        raw_model = ModelEvaluation(
            valid=True,
            predicted_latency_ms=3.0,
            bottleneck="tensor-core",
            metrics={
                "registers_per_thread": 64,
                "shared_memory_per_block": 16384,
            },
        )
        calibrated = calibrator.apply(task, raw_model)
        self.assertEqual(calibrated.predicted_latency_ms, 3.0)
        self.assertEqual(calibrated.calibrated_latency_ms, 6.0)
        self.assertEqual(calibrated.ranking_latency_ms, 6.0)
        self.assertEqual(calibrated.calibration["source"], "exact-regime")

    def test_profile_lesson_is_visible_to_an_unrelated_parent(self) -> None:
        task = make_task(max_repairs_per_round=0)
        seed = Candidate.seed(task, "def kernel(x):\n    return x\n", "kernel.py")
        profiled = CandidateRecord(
            candidate=seed,
            model=ModelEvaluation(valid=True, predicted_latency_ms=1.0),
            measurement=Measurement(correct=True, latency_ms=1.1),
            profile=ProfileEvaluation(
                bottleneck="dram-bandwidth", metrics={"ddr_util": 80.0}
            ),
        )
        profiled.diagnosis = BottleneckAnalyzer().diagnose(profiled)
        other_candidate = Candidate._create(
            task=task,
            parent_id=None,
            generation=1,
            source_name="kernel.py",
            source_code="VALUE = 8\n\ndef kernel(x):\n    return x\n",
            hypothesis="unrelated parent",
            expected_effect={},
            proposal_metadata={},
            lineage_kind="proposal",
            repair_depth=0,
        )
        other = CandidateRecord(candidate=other_candidate, diagnosis=profiled.diagnosis)
        memory = GlobalEvidenceMemory()
        lesson = memory.ingest(profiled, None, round_number=0)

        visible = memory.for_prompt(other)
        self.assertEqual(visible[0]["lesson_id"], lesson.lesson_id)
        self.assertEqual(visible[0]["kind"], "hardware-profile")


class ApiRepairPromptTests(unittest.TestCase):
    def test_repair_prompt_contains_failure_source_and_shared_lessons(self) -> None:
        captured = {}

        def transport(url, headers, payload, timeout):
            del url, headers, timeout
            captured.update(json.loads(payload["messages"][1]["content"]))
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidates": [
                                        {
                                            "hypothesis": "Restore the entrypoint.",
                                            "source_code": "def kernel(x):\n    return x\n",
                                            "metadata": {
                                                "strategy": "repair",
                                                "evidence_ids": ["lesson-1"],
                                            },
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

        generator = OpenAICompatibleGenerator(
            ApiGeneratorConfig(
                api_url="https://provider.example/v1/chat/completions",
                model="test-model",
                api_key_environment_variable="TEST_KERNEL_API_KEY",
            ),
            transport=transport,
        )
        task = make_task(max_repairs_per_round=0)
        failed = Candidate.seed(
            task, "def removed(x):\n    return x\n", "kernel.py"
        )
        with mock.patch.dict(os.environ, {"TEST_KERNEL_API_KEY": "test-secret"}):
            proposal = generator.repair(
                task,
                failed,
                {"failure": {"category": "interface-contract"}},
                {"shared_memory": [{"lesson_id": "lesson-1"}]},
                [],
            )

        self.assertEqual(proposal.metadata["evidence_ids"], ["lesson-1"])
        self.assertEqual(captured["mode"], "repair")
        self.assertEqual(
            captured["classified_failure"]["failure"]["category"],
            "interface-contract",
        )
        self.assertIn("def removed", captured["failed_candidate"]["source_code"])
        prompts = json.dumps(captured)
        self.assertFalse(any("\u4e00" <= char <= "\u9fff" for char in prompts))


class _InfrastructureGenerator:
    last_call_metadata = {}
    last_exchange = {}

    def __init__(self) -> None:
        self.repair_calls = 0

    def generate(self, task, parent, evidence, history, count):
        del task, parent, evidence, history
        return [
            CandidateProposal(
                hypothesis="Exercise infrastructure failure %d." % value,
                source_code="VALUE = %d\n\ndef kernel(x):\n    return x\n" % value,
            )
            for value in range(1, count + 1)
        ]

    def repair(self, *args, **kwargs):
        del args, kwargs
        self.repair_calls += 1
        raise AssertionError("infrastructure failures must not request source repair")


class _InfrastructureBackend:
    def model(self, task: TaskSpec, candidate: Candidate) -> ModelEvaluation:
        del task, candidate
        return ModelEvaluation(valid=True, predicted_latency_ms=1.0)

    def measure(self, task: TaskSpec, candidate: Candidate) -> Measurement:
        del task
        if "VALUE = 0" in candidate.source_code:
            return Measurement(correct=True, latency_ms=1.0, samples_ms=[1.0])
        return Measurement(
            correct=False,
            error="RuntimeError: [Errno 122] Disk quota exceeded: cache.cubin",
        )

    def profile(self, task: TaskSpec, candidate: Candidate) -> ProfileEvaluation:
        del task, candidate
        return ProfileEvaluation(bottleneck="test", metrics={"cuda_util": 0.5})


class _RepairingGenerator:
    last_call_metadata = {}
    last_exchange = {}

    def __init__(self) -> None:
        self.repair_evidence = []

    def generate(self, task, parent, evidence, history, count):
        del task, parent, evidence, history, count
        return [
            CandidateProposal(
                hypothesis="Remove the entrypoint to trigger static repair.",
                source_code="VALUE = 9\n\ndef removed(x):\n    return x\n",
            ),
            CandidateProposal(
                hypothesis="Use an intentionally incorrect schedule marker.",
                source_code="VALUE = 3\n\ndef kernel(x):\n    return x\n",
            ),
        ]

    def repair(self, task, failed, failure, evidence, history):
        del task, history
        self.repair_evidence.append(evidence)
        stage = failure["failure"]["stage"]
        value = 1 if stage == "static" else 2
        return CandidateProposal(
            hypothesis="Repair %s failure." % stage,
            source_code="VALUE = %d\n\ndef kernel(x):\n    return x\n" % value,
            metadata={
                "strategy": "repair",
                "evidence_ids": [
                    item["lesson_id"] for item in evidence["shared_memory"][:2]
                ],
            },
        )


class _RepairBackend:
    def model(self, task: TaskSpec, candidate: Candidate) -> ModelEvaluation:
        del task
        value = _value(candidate.source_code)
        latency = {0: 2.0, 1: 1.0, 2: 0.4, 3: 0.8}[value]
        return ModelEvaluation(
            valid=True,
            predicted_latency_ms=latency,
            bottleneck="tensor-core",
            confidence="medium",
            metrics={
                "tensor_util": 55.0,
                "registers_per_thread": 64,
                "shared_memory_per_block": 16384,
            },
        )

    def measure(self, task: TaskSpec, candidate: Candidate) -> Measurement:
        del task
        value = _value(candidate.source_code)
        if value == 3:
            return Measurement(correct=False, error="Correctness mismatch at output 0")
        latency = {0: 2.0, 1: 1.0, 2: 0.5}[value]
        return Measurement(correct=True, latency_ms=latency, samples_ms=[latency])

    def profile(self, task: TaskSpec, candidate: Candidate) -> ProfileEvaluation:
        del task, candidate
        return ProfileEvaluation(
            bottleneck="tensor-core",
            metrics={"tensor_util": 65.0, "achieved_occupancy": 0.5},
        )


def _value(source_code: str) -> int:
    for line in source_code.splitlines():
        if line.startswith("VALUE = "):
            return int(line.split("=", 1)[1])
    raise ValueError("source has no VALUE marker")


if __name__ == "__main__":
    unittest.main()
