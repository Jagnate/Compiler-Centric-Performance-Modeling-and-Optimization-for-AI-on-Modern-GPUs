"""Generic workload runtime, held-out final gate, and robust timing tests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from examples import tilesight_kernel_evaluator as evaluator
from kernel_optimization.archive import ArtifactStore
from kernel_optimization.controller import OptimizationController
from kernel_optimization.prompts import build_optimization_prompt
from kernel_optimization.schema import (
    BudgetConfig,
    Candidate,
    CandidateProposal,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    TaskSpec,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class GenericEvaluatorTests(unittest.TestCase):
    def test_targeted_ncu_profile_avoids_the_full_metric_set(self) -> None:
        arguments = evaluator._ncu_collection_arguments(
            {"ncu_set": "tilesight-targeted"}, {}
        )

        self.assertEqual(arguments[0], "--metrics")
        metrics = arguments[1].split(",")
        self.assertIn("gpu__time_duration.sum", metrics)
        self.assertIn("launch__registers_per_thread", metrics)
        self.assertIn(
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", metrics
        )
        self.assertIn(
            "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed",
            metrics,
        )
        self.assertNotIn("--set", arguments)

    def test_explicit_ncu_set_and_metric_overrides_remain_supported(self) -> None:
        self.assertEqual(
            evaluator._ncu_collection_arguments({"ncu_set": "full"}, {}),
            ["--set", "full"],
        )
        self.assertEqual(
            evaluator._ncu_collection_arguments(
                {"ncu_metrics": ["metric.a", "metric.a", "metric.b"]}, {}
            ),
            ["--metrics", "metric.a,metric.b"],
        )

    def test_all_workloads_share_one_evaluator_runtime(self) -> None:
        tasks = {
            name: TaskSpec.from_json_file(
                REPOSITORY_ROOT / "examples" / f"tilelang_{name}_task.json"
            )
            for name in (
                "matmul",
                "flash_attention",
                "rms_norm",
                "fused_add_rms_norm",
                "conv2d",
            )
        }
        commands = {tuple(task.evaluator["command"]) for task in tasks.values()}
        plugins = {
            task.evaluator["runtime"]["plugin"] for task in tasks.values()
        }
        self.assertEqual(len(commands), 1)
        self.assertEqual(
            list(commands.pop()),
            ["python3", "examples/tilesight_kernel_evaluator.py"],
        )
        self.assertEqual(len(plugins), len(tasks))
        schedule_arguments = {
            "block_m",
            "block_n",
            "block_k",
            "block_rows",
            "block_hidden",
            "num_stages",
            "threads",
        }
        for task in tasks.values():
            self.assertGreaterEqual(
                len(task.evaluator["runtime"]["final_cases"]), 1
            )
            arguments = set(task.workload["factory_arguments"])
            self.assertTrue(arguments.isdisjoint(schedule_arguments))
            self.assertFalse(task.evaluator["runtime"]["model_collect_ptxas"])
            self.assertTrue(task.evaluator["persistent_process"])
            self.assertEqual(task.evaluator["worker_max_requests"], 32)
            self.assertEqual(
                task.evaluator["runtime"]["ncu_set"],
                "tilesight-targeted",
            )

    def test_held_out_cases_are_not_in_generation_prompt(self) -> None:
        task = TaskSpec.from_json_file(
            REPOSITORY_ROOT / "examples" / "tilelang_flash_attention_task.json"
        )
        source = (
            REPOSITORY_ROOT / "examples" / "tilelang_flash_attention_kernel.py"
        ).read_text(encoding="utf-8")
        seed = Candidate.seed(task, source, "tilelang_flash_attention_kernel.py")
        prompt = build_optimization_prompt(task, seed, {}, [], count=1)
        self.assertIn("seq_len", prompt)
        self.assertNotIn("heldout-causal-b1-h4-s512-d64", prompt)
        self.assertNotIn("examples/workloads/flash_attention.py", prompt)

    def test_case_resolution_merges_primary_and_held_out_arguments(self) -> None:
        request = _runtime_request(Path("unused.py"), "", final_repeats=3)
        search, primary = evaluator.resolve_cases(request, "measure")
        final, final_primary = evaluator.resolve_cases(request, "final")
        self.assertEqual(primary["case_id"], "primary")
        self.assertEqual(final_primary["case_id"], "primary")
        self.assertEqual(len(search), 2)
        self.assertEqual(len(final), 3)
        held_out = next(item for item in final if item["held_out"])
        self.assertEqual(held_out["factory_arguments"]["m"], 1536)
        self.assertEqual(held_out["factory_arguments"]["n"], 2048)

    def test_measurement_runs_multi_case_correctness_and_repeated_timing(self) -> None:
        source = (
            "def make_kernel(m=1, n=1, k=1):\n"
            "    return {'m': m, 'n': n, 'k': k}\n"
        )
        with tempfile.TemporaryDirectory(prefix="generic-evaluator-") as directory:
            source_path = Path(directory) / "candidate.py"
            source_path.write_text(source, encoding="utf-8")
            request = _runtime_request(source_path, source, measurement_repeats=3)
            runner = _FakeValidationRunner([1.2, 1.0, 1.1])
            response = evaluator.measure_response(
                request,
                final=False,
                validation_runner=runner,
                environment_resolver=_fake_environment,
            )

        self.assertTrue(response["correct"])
        self.assertEqual(response["latency_ms"], 1.1)
        self.assertEqual(response["samples_ms"], [1.2, 1.0, 1.1])
        self.assertEqual(response["metrics"]["case_count"], 2)
        self.assertEqual(response["metrics"]["held_out_case_count"], 0)
        self.assertEqual(response["metrics"]["statistics"]["sample_count"], 3)
        correctness_calls = [item for item in runner.calls if item["check_correctness"]]
        benchmark_calls = [item for item in runner.calls if item["benchmark"]]
        self.assertEqual(len(correctness_calls), 2)
        self.assertEqual(len(benchmark_calls), 3)

    def test_model_stage_reports_tilesight_resources_and_compiled_hash(self) -> None:
        source = (
            "def make_kernel(m=1, n=1, k=1):\n"
            "    return {'m': m, 'n': n, 'k': k}\n"
        )
        with tempfile.TemporaryDirectory(prefix="generic-model-") as directory:
            source_path = Path(directory) / "candidate.py"
            source_path.write_text(source, encoding="utf-8")
            request = _runtime_request(source_path, source)
            model_options = {}

            def modeler(program, arch, **kwargs):
                model_options.update(kwargs)
                return _fake_modeler(program, arch, **kwargs)

            response = evaluator.model_response(
                request,
                modeler=modeler,
                environment_resolver=_fake_environment,
            )

        self.assertTrue(response["valid"])
        self.assertEqual(response["predicted_latency_ms"], 0.75)
        self.assertEqual(response["bottleneck"], "tensor-core")
        self.assertEqual(response["metrics"]["registers_per_thread"], 72.0)
        self.assertEqual(response["metrics"]["threads_per_block"], 128)
        self.assertFalse(model_options["collect_ptxas"])
        self.assertFalse(response["metrics"]["model_collect_ptxas"])
        self.assertEqual(response["metrics"]["register_source"], "tir-estimate")
        self.assertEqual(response["metrics"]["resource_fidelity"], "fast-screening")
        self.assertEqual(response["confidence"], "medium")
        self.assertEqual(
            response["metrics"]["compiled_source_sha256"],
            hashlib.sha256(b"compiled cuda source").hexdigest(),
        )
        self.assertIsNotNone(response["metrics"]["compiled_identity_sha256"])

    def test_tir_stage_compacts_structure_without_latency_prediction(self) -> None:
        resource = _Object(
            global_read_bytes=4096,
            global_write_bytes=2048,
            l2_read_bytes=4096,
            l2_write_bytes=2048,
            smem_read_bytes=8192,
            smem_write_bytes=4096,
            tensor_flops=262144,
            cuda_flops=0,
            sfu_ops=0,
            integer_ops=32,
            reduction_ops=0,
            sync_ops=1,
        )
        operation = _Object(
            name="gemm",
            kind="tensor",
            resources=resource,
            pipeline_stage=1,
            pipeline_order=2,
            is_async=False,
            reads=[_Object(scope="shared")],
            writes=[_Object(scope="fragment")],
            dependencies=[],
            loop_carried_dependencies=[],
        )
        loop = _FakeTIRLoop(operation)
        program = _FakeTIRProgram(loop, operation)

        features, diagnostics = evaluator._compact_tir_features(program)

        self.assertEqual(features["threads_per_block"], 128)
        self.assertEqual(features["operation_kind_counts"], {"tensor": 1})
        self.assertEqual(
            features["resource_totals_per_source_iteration"]["tensor_flops"],
            262144.0,
        )
        self.assertIn("structural_fingerprint", features)
        self.assertNotIn("predicted_latency_ms", features)
        self.assertEqual(diagnostics, [])

    def test_measurement_uses_one_compile_for_all_benchmark_repeats(self) -> None:
        source = (
            "def make_kernel(m=1, n=1, k=1):\n"
            "    return {'m': m, 'n': n, 'k': k}\n"
        )
        with tempfile.TemporaryDirectory(prefix="generic-evaluator-") as directory:
            source_path = Path(directory) / "candidate.py"
            source_path.write_text(source, encoding="utf-8")
            request = _runtime_request(source_path, source, measurement_repeats=3)
            request["_persistent_worker"] = True
            runner = _BatchedValidationRunner([1.2, 1.0, 1.1])
            response = evaluator.measure_response(
                request,
                final=False,
                validation_runner=runner,
                environment_resolver=_fake_environment,
            )

        self.assertTrue(response["correct"])
        self.assertEqual(response["samples_ms"], [1.2, 1.0, 1.1])
        self.assertEqual(response["latency_ms"], 1.1)
        self.assertFalse(response["metrics"]["fresh_process"])
        benchmark_calls = [item for item in runner.calls if item["benchmark"]]
        self.assertEqual(len(benchmark_calls), 1)
        self.assertEqual(benchmark_calls[0]["benchmark_repeats"], 3)

    def test_final_stage_adds_held_out_case_and_fresh_statistics(self) -> None:
        source = (
            "def make_kernel(m=1, n=1, k=1):\n"
            "    return {'m': m, 'n': n, 'k': k}\n"
        )
        with tempfile.TemporaryDirectory(prefix="generic-final-") as directory:
            source_path = Path(directory) / "candidate.py"
            source_path.write_text(source, encoding="utf-8")
            request = _runtime_request(source_path, source, final_repeats=4)
            runner = _FakeValidationRunner([2.0, 1.8, 1.9, 2.1])
            response = evaluator.measure_response(
                request,
                final=True,
                validation_runner=runner,
                environment_resolver=_fake_environment,
            )

        self.assertTrue(response["correct"])
        self.assertEqual(response["metrics"]["stage"], "final")
        self.assertTrue(response["metrics"]["fresh_process"])
        self.assertEqual(response["metrics"]["case_count"], 3)
        self.assertEqual(response["metrics"]["held_out_case_count"], 1)
        self.assertEqual(response["metrics"]["statistics"]["sample_count"], 4)
        self.assertEqual(len([call for call in runner.calls if call["check_correctness"]]), 3)

    def test_robust_statistics_rejects_invalid_samples(self) -> None:
        value = evaluator.robust_statistics([1.0, 1.1, 20.0])
        self.assertEqual(value["median_ms"], 1.1)
        self.assertAlmostEqual(value["median_absolute_deviation_ms"], 0.1)
        with self.assertRaisesRegex(ValueError, "finite positive"):
            evaluator.robust_statistics([1.0, 0.0])


class FreshFinalGateTests(unittest.TestCase):
    def test_search_winner_that_fails_held_out_case_is_not_exported(self) -> None:
        task = TaskSpec(
            task_id="final-gate-test",
            description="Verify fresh final fallback.",
            reference="Return the input.",
            entrypoint="kernel",
            constraints={"required_fragments": ["def kernel"]},
            budget=BudgetConfig(
                rounds=1,
                proposals_per_round=2,
                beam_width=2,
                min_promotions_per_round=2,
                max_promotions_per_round=2,
                max_repairs_per_round=0,
                final_validation_candidates=2,
            ),
        )
        source = "VALUE = 0\n\ndef kernel(x):\n    return x\n"
        with tempfile.TemporaryDirectory(prefix="final-gate-") as directory:
            controller = OptimizationController(
                task=task,
                source_code=source,
                source_name="kernel.py",
                generator=_TwoCandidateGenerator(),
                backend=_FinalGateBackend(),
                store=ArtifactStore(Path(directory)),
            )
            summary = controller.run()

            best_source = Path(summary.best_source_path).read_text(encoding="utf-8")
            report = (Path(directory) / "experiment_report.md").read_text(
                encoding="utf-8"
            )

        self.assertNotEqual(summary.search_best_candidate_id, summary.best_candidate_id)
        self.assertEqual(summary.search_best_latency_ms, 0.5)
        self.assertEqual(summary.best_latency_ms, 1.4)
        self.assertEqual(summary.final_seed_latency_ms, 2.0)
        self.assertEqual(summary.final_validation_calls, 3)
        self.assertEqual(summary.final_validation_passes, 2)
        self.assertIn("VALUE = 2", best_source)
        self.assertIn("passed separate final validation", report)


@dataclass
class _FakeRuntimeResult:
    correctness: str
    benchmark_latency_ms: float | None
    benchmark_samples_ms: list[float] | None = None
    kernel_name: str = "fake"

    def to_dict(self):
        return {
            "correctness": self.correctness,
            "benchmark_latency_ms": self.benchmark_latency_ms,
            "benchmark_samples_ms": self.benchmark_samples_ms or [],
            "kernel_name": self.kernel_name,
        }


class _FakeValidationRunner:
    def __init__(self, samples):
        self.samples = iter(samples)
        self.calls = []

    def __call__(self, program, target, **kwargs):
        del program, target
        self.calls.append(dict(kwargs))
        latency = next(self.samples) if kwargs["benchmark"] else None
        return _FakeRuntimeResult(
            correctness="passed" if kwargs["check_correctness"] else "not-run",
            benchmark_latency_ms=latency,
        )


class _BatchedValidationRunner:
    def __init__(self, samples):
        self.samples = [float(item) for item in samples]
        self.calls = []

    def __call__(self, program, target, **kwargs):
        del program, target
        self.calls.append(dict(kwargs))
        if not kwargs["benchmark"]:
            return _FakeRuntimeResult("passed", None)
        repeats = int(kwargs.get("benchmark_repeats", 1))
        samples = self.samples[:repeats]
        return _FakeRuntimeResult(
            correctness="passed" if kwargs["check_correctness"] else "not-run",
            benchmark_latency_ms=sorted(samples)[len(samples) // 2],
            benchmark_samples_ms=samples,
        )


def _runtime_request(
    source_path: Path,
    source: str,
    measurement_repeats: int = 2,
    final_repeats: int = 3,
):
    return {
        "task": {
            "task_id": "generic-runtime-test",
            "entrypoint": "make_kernel",
            "target": {"architecture": "test", "tilelang_target": "test"},
            "workload": {
                "factory_arguments": {"m": 2048, "n": 2048, "k": 2048},
                "output_indices": [2],
                "warmup_ms": 1.0,
                "rep_ms": 2.0,
            },
            "evaluator": {
                "runtime": {
                    "plugin": str(REPOSITORY_ROOT / "examples" / "workloads" / "matmul.py"),
                    "measurement_repeats": measurement_repeats,
                    "final_repeats": final_repeats,
                    "search_cases": [
                        {"case_id": "primary"},
                        {
                            "case_id": "public-small",
                            "factory_arguments": {"m": 1024, "n": 1024, "k": 1024},
                        },
                    ],
                    "final_cases": [
                        {
                            "case_id": "heldout-rectangular",
                            "factory_arguments": {"m": 1536},
                        }
                    ],
                }
            },
        },
        "candidate": {"candidate_id": "candidate-test"},
        "source": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        },
    }


def _fake_environment(request):
    del request
    return object(), "fake-target", {"device_name": "fake-gpu"}


class _Object:
    def __init__(self, **values):
        self.__dict__.update(values)


class _FakeTIRLoop:
    def __init__(self, operation) -> None:
        self.name = "ko"
        self.extent = 4
        self.pipeline_depth = 2
        self.schedule_policy = "annotated"
        self.loop_carried_dependencies = []
        self.operation = operation

    def walk_loops(self):
        return iter([self])


class _FakeTIRProgram:
    def __init__(self, loop, operation) -> None:
        self.symbol = "matmul"
        self.grid_shape = (16, 16, 1)
        self.threads_per_block = 128
        self.warps_per_block = 4
        self.smem_footprint = 32768
        self.reg_footprint = 64
        self.buffers = {
            "a": _Object(
                name="a",
                scope="global",
                shape=(2048, 2048),
                dtype="float16",
                size_bytes=8388608,
                is_parameter=True,
            )
        }
        self.root = loop
        self._operation = operation
        self.diagnostics = []

    def walk_operations(self):
        return iter([self._operation])


def _fake_modeler(program, arch, **kwargs):
    del program, arch, kwargs
    metrics = _Object(
        latency=0.00075,
        ddr_util=0.2,
        l2_hit_rate=0.5,
        l2_util=0.3,
        smem_util=0.4,
        tensor_util=0.7,
        cuda_util=0.1,
        sfu_util=0.05,
        smem_footprint=32768,
        reg_footprint=72,
    )
    extracted_program = _Object(
        diagnostics=[],
        metadata={},
        threads_per_block=128,
        symbol="fake",
    )
    model_input = _Object(
        grids=(16, 16, 1),
        tiles_per_sm=4,
        logical_global_write_io=8192,
        l2_write_io=8192,
        ddr_write_io=8192,
        spill_load_io=0,
        spill_store_io=0,
        provenance={"registers": "tir-estimate"},
    )
    result = _Object(
        metrics=metrics,
        program=extracted_program,
        model_input=model_input,
    )
    lowered = _Object(kernel_source="compiled cuda source")
    snapshots = _Object(before={"LowerTileOp": 1}, after={"LowerTileOp": 1})
    return result, lowered, snapshots


class _TwoCandidateGenerator:
    last_call_metadata = {}
    last_exchange = {}

    def generate(self, task, parent, evidence, history, count):
        del task, parent, evidence, history, count
        return [
            CandidateProposal(
                hypothesis="Fast but semantically narrow.",
                source_code="VALUE = 1\n\ndef kernel(x):\n    return x\n",
            ),
            CandidateProposal(
                hypothesis="General and moderately fast.",
                source_code="VALUE = 2\n\ndef kernel(x):\n    return x\n",
            ),
        ]


class _FinalGateBackend:
    def model(self, task, candidate):
        del task
        value = _source_value(candidate.source_code)
        return ModelEvaluation(
            valid=True,
            predicted_latency_ms={0: 2.0, 1: 0.5, 2: 1.0}[value],
        )

    def measure(self, task, candidate):
        del task
        value = _source_value(candidate.source_code)
        latency = {0: 2.0, 1: 0.5, 2: 1.0}[value]
        return Measurement(correct=True, latency_ms=latency, samples_ms=[latency])

    def profile(self, task, candidate):
        del task, candidate
        return ProfileEvaluation(bottleneck="test")

    def finalize(self, task, candidate):
        del task
        value = _source_value(candidate.source_code)
        if value == 1:
            return Measurement(correct=False, error="Held-out case mismatch")
        latency = {0: 2.0, 2: 1.4}[value]
        return Measurement(
            correct=True,
            latency_ms=latency,
            samples_ms=[latency - 0.01, latency, latency + 0.01],
            metrics={"held_out_cases": 1},
        )


def _source_value(source):
    for line in source.splitlines():
        if line.startswith("VALUE = "):
            return int(line.split("=", 1)[1])
    raise ValueError("missing VALUE marker")


if __name__ == "__main__":
    unittest.main()
