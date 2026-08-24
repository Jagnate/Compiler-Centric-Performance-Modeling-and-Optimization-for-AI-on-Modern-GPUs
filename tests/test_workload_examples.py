"""Contracts and optional TileLang smoke tests for the example workloads."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest

from examples import tilesight_kernel_evaluator as evaluator
from kernel_optimization.schema import TaskSpec
from kernel_optimization.structural_search import StructuralStrategyPortfolio


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TILELANG_AVAILABLE = importlib.util.find_spec("tilelang") is not None
SCHEDULE_ARGUMENTS = {
    "block_m",
    "block_n",
    "block_k",
    "block_rows",
    "block_hidden",
    "num_stages",
    "threads",
}


def _load_task(name: str) -> TaskSpec:
    return TaskSpec.from_json_file(
        REPOSITORY_ROOT / "examples" / f"tilelang_{name}_task.json"
    )


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _factory_parameters(source: str, entrypoint: str) -> set[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == entrypoint:
                return {argument.arg for argument in node.args.args}
    raise AssertionError(f"entrypoint {entrypoint!r} is missing")


class WorkloadContractTests(unittest.TestCase):
    def test_all_task_shapes_match_seed_factory_signatures(self) -> None:
        for name in ("matmul", "flash_attention", "rms_norm", "conv2d"):
            with self.subTest(workload=name):
                task = _load_task(name)
                source_path = (
                    REPOSITORY_ROOT / "examples" / f"tilelang_{name}_kernel.py"
                )
                parameters = _factory_parameters(
                    source_path.read_text(encoding="utf-8"), task.entrypoint
                )
                workload_arguments = set(task.workload["factory_arguments"])
                self.assertTrue(workload_arguments <= parameters)
                self.assertTrue(workload_arguments.isdisjoint(SCHEDULE_ARGUMENTS))

    def test_new_workloads_exercise_multiple_input_shapes(self) -> None:
        expected_changes = {
            "rms_norm": {"rows", "hidden_size"},
            "conv2d": {
                "batch",
                "in_height",
                "in_width",
                "in_channels",
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
            },
        }
        for name, expected in expected_changes.items():
            with self.subTest(workload=name):
                task = _load_task(name)
                request = {"task": task.to_dict()}
                search, primary = evaluator.resolve_cases(request, "measure")
                final, final_primary = evaluator.resolve_cases(request, "final")
                self.assertEqual(primary["case_id"], final_primary["case_id"])
                self.assertGreaterEqual(len(search), 2)
                self.assertGreater(len(final), len(search))
                changed = {
                    key
                    for case in final[1:]
                    for key, value in case["factory_arguments"].items()
                    if value != primary["factory_arguments"].get(key)
                }
                self.assertTrue(expected <= changed)

    def test_new_workload_plugins_validate_every_configured_case(self) -> None:
        for name in ("rms_norm", "conv2d"):
            with self.subTest(workload=name):
                task = _load_task(name)
                request = {"task": task.to_dict()}
                cases, _primary = evaluator.resolve_cases(request, "final")
                plugin = _load_module(
                    REPOSITORY_ROOT / "examples" / "workloads" / f"{name}.py",
                    f"test_workload_{name}",
                )
                for case in cases:
                    plugin.validate_case(case, task.to_dict())
                    self.assertEqual(plugin.output_indices(case, task.to_dict()), [2])
                    self.assertTrue(callable(plugin.reference_program(case, task.to_dict())))

    def test_invalid_rms_norm_and_conv2d_shapes_are_rejected(self) -> None:
        rms_task = _load_task("rms_norm").to_dict()
        rms_plugin = _load_module(
            REPOSITORY_ROOT / "examples" / "workloads" / "rms_norm.py",
            "test_invalid_rms_norm",
        )
        with self.assertRaisesRegex(ValueError, "positive integer hidden_size"):
            rms_plugin.validate_case(
                {
                    "case_id": "invalid-rms",
                    "factory_arguments": {
                        "rows": 8,
                        "hidden_size": 0,
                        "epsilon": 1e-6,
                    },
                },
                rms_task,
            )

        conv_task = _load_task("conv2d").to_dict()
        conv_plugin = _load_module(
            REPOSITORY_ROOT / "examples" / "workloads" / "conv2d.py",
            "test_invalid_conv2d",
        )
        invalid_conv = dict(conv_task["workload"]["factory_arguments"])
        invalid_conv.update({"in_height": 2, "in_width": 2, "kernel_size": 7})
        with self.assertRaisesRegex(ValueError, "empty output"):
            conv_plugin.validate_case(
                {"case_id": "invalid-conv", "factory_arguments": invalid_conv},
                conv_task,
            )

    def test_new_workloads_receive_family_specific_search_guidance(self) -> None:
        portfolio = StructuralStrategyPortfolio()
        for name, marker in (("rms_norm", "RMSNorm"), ("conv2d", "convolution")):
            with self.subTest(workload=name):
                catalog = portfolio.strategy_catalog(_load_task(name))
                hints = " ".join(item["family_hint"] for item in catalog)
                self.assertIn(marker, hints)
                self.assertNotIn(
                    "Apply this strategy only with APIs already visible", hints
                )


@unittest.skipUnless(TILELANG_AVAILABLE, "TileLang is not installed")
class RealTileLangSeedSmokeTests(unittest.TestCase):
    def test_constructs_rms_norm_primfunc(self) -> None:
        module = _load_module(
            REPOSITORY_ROOT / "examples" / "tilelang_rms_norm_kernel.py",
            "tilelang_rms_norm_seed_smoke",
        )
        program = module.make_rms_norm_program(
            rows=8,
            hidden_size=256,
            block_rows=1,
            block_hidden=256,
            threads=128,
        )
        self.assertIsNotNone(program)

    def test_constructs_conv2d_primfunc(self) -> None:
        module = _load_module(
            REPOSITORY_ROOT / "examples" / "tilelang_conv2d_kernel.py",
            "tilelang_conv2d_seed_smoke",
        )
        program = module.make_conv2d_program(
            batch=1,
            in_height=8,
            in_width=8,
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding=1,
            block_m=64,
            block_n=64,
            block_k=32,
            num_stages=2,
            threads=128,
        )
        self.assertIsNotNone(program)


if __name__ == "__main__":
    unittest.main()
