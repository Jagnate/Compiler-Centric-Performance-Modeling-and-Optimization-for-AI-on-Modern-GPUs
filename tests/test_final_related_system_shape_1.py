"""Tests for the native-plus-proxies shape-family-1 experiment."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "examples"))

import run_final_related_system_baselines as baseline_driver
import run_final_related_system_shape_1 as shape_driver


class FinalRelatedSystemShapeOneTests(unittest.TestCase):
    def test_wrapper_selects_six_styles_and_requested_output(self) -> None:
        forwarded = shape_driver.build_forwarded_arguments([])
        args = baseline_driver.build_parser().parse_args(forwarded)
        treatments = baseline_driver.selected_treatments(
            args.only_kernel,
            args.only_style,
            include_native=args.include_native,
        )

        self.assertEqual(len(treatments), 30)
        self.assertEqual(
            [item.style.key for item in treatments[:6]],
            [
                "native",
                "kernelagent",
                "kernelevolve",
                "kernelbench",
                "avo",
                "tilefoundry",
            ],
        )
        self.assertEqual(
            baseline_driver._default_output_root(
                baseline_driver._builtin_workload_suite(args.shape_config)
            ).parts[-2:],
            ("final_eval", "related_system_baselines_30m_special"),
        )
        self.assertIsNone(args.output_root)
        self.assertEqual(args.shape_config, "special")
        self.assertIsNone(args.workload_suite)
        self.assertEqual(args.budget_mode, "fixed-time")
        self.assertEqual(args.measurement_repeats, 3)

    def test_special_primary_shape_differs_from_basic_and_large(self) -> None:
        configs = [
            baseline_driver._builtin_workload_suite(name)
            for name in ("basic", "large", "special")
        ]
        for kernel in baseline_driver.KERNELS:
            shapes = {
                json.dumps(
                    config["workloads"][kernel.key]["factory_arguments"],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for config in configs
            }
            self.assertEqual(
                len(shapes), 3, "%s repeats a configured shape" % kernel.key
            )

    def test_materialized_tasks_validate_and_isolate_profile_directories(self) -> None:
        suite = baseline_driver._builtin_workload_suite("special")
        treatments = baseline_driver.selected_treatments(
            only_kernels=["matmul"],
            only_styles=["native", "kernelagent"],
            include_native=True,
        )
        base_path = baseline_driver.KERNELS[0].task_path(REPOSITORY_ROOT)
        original = base_path.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "shape-one"
            paths, payloads = baseline_driver._prepare_treatment_tasks(
                REPOSITORY_ROOT,
                output_root,
                treatments,
                suite,
                dry_run=False,
            )

            self.assertEqual(len(paths), 2)
            native_task = _read_json(paths[("matmul", "native")])
            proxy_task = _read_json(paths[("matmul", "kernelagent")])
            self.assertEqual(
                native_task["workload"]["factory_arguments"],
                {"m": 4096, "n": 1024, "k": 4096},
            )
            self.assertNotEqual(
                native_task["evaluator"]["runtime"]["profile_directory"],
                proxy_task["evaluator"]["runtime"]["profile_directory"],
            )
            self.assertEqual(
                Path(native_task["evaluator"]["working_directory"]),
                REPOSITORY_ROOT,
            )
            self.assertEqual(
                payloads["matmul"]["metadata"]["shape_suite"]["base_task_id"],
                "tilelang_matmul_2048_rtx3090",
            )
            self.assertEqual(base_path.read_text(encoding="utf-8"), original)

    def test_all_materialized_cases_pass_workload_contract_validation(self) -> None:
        suite = baseline_driver._builtin_workload_suite("special")
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "shape-one"
            treatments = baseline_driver.selected_treatments(
                only_styles=["native"], include_native=True
            )
            _paths, payloads = baseline_driver._prepare_treatment_tasks(
                REPOSITORY_ROOT,
                output_root,
                treatments,
                suite,
                dry_run=False,
            )
            for kernel in baseline_driver.KERNELS:
                task = payloads[kernel.key]
                plugin_name = task["evaluator"]["runtime"]["plugin"]
                module_name = plugin_name[:-3].replace("/", ".")
                plugin = importlib.import_module(module_name)
                primary = dict(task["workload"]["factory_arguments"])
                cases = (
                    list(task["evaluator"]["runtime"]["search_cases"])
                    + list(task["evaluator"]["runtime"]["final_cases"])
                )
                for raw_case in cases:
                    case = dict(raw_case)
                    arguments = dict(primary)
                    arguments.update(dict(case.get("factory_arguments") or {}))
                    case["factory_arguments"] = arguments
                    plugin.validate_case(case, task)

    def test_single_cell_dry_run_uses_materialized_task_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "not-created"
            output = io.StringIO()
            with redirect_stdout(output):
                code = shape_driver.main(
                    [
                        "--dry-run",
                        "--api-url",
                        "https://api.example/v1/chat/completions",
                        "--api-model",
                        "test-model",
                        "--output-root",
                        str(output_root),
                        "--only-kernel",
                        "flash_attention",
                        "--only-style",
                        "native",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertFalse(output_root.exists())
            rendered = output.getvalue()
            self.assertIn("Selected 1 treatments", rendered)
            self.assertIn("--baseline-style native", rendered)
            self.assertIn("_tasks/flash_attention/native.json", rendered)

    def test_shape_suite_report_records_native_and_all_configured_cases(self) -> None:
        suite = baseline_driver._builtin_workload_suite("special")
        treatments = baseline_driver.selected_treatments(include_native=True)
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            paths, payloads = baseline_driver._prepare_treatment_tasks(
                REPOSITORY_ROOT,
                output_root,
                treatments,
                suite,
                dry_run=True,
            )
            workloads = baseline_driver._load_workload_metadata(
                REPOSITORY_ROOT,
                treatments,
                task_payloads=payloads,
                task_paths=paths,
            )
            baseline_driver._write_reports(
                output_root,
                treatments,
                workloads,
                [],
                1800.0,
                suite,
            )

            report = _read_json(output_root / "suite_summary.json")
            markdown = (output_root / "suite_summary.md").read_text(
                encoding="utf-8"
            )
            self.assertEqual(report["planned_treatments"], 30)
            self.assertEqual(report["candidate_graph_upper_bound"], 990)
            self.assertEqual(report["styles"][0], "native")
            self.assertEqual(
                report["workload_suite"]["suite_id"],
                "related_system_special_shapes",
            )
            self.assertIn("Configured Shape Family", markdown)
            self.assertIn("final-b1-h16-s3072-d64-causal", markdown)
            self.assertIn("Exported best (ms)", markdown)
            self.assertIn("Final checks", markdown)
            self.assertIn("Search (s)", markdown)
            self.assertIn("Final (s)", markdown)
            self.assertIn("Total (s)", markdown)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
