"""Tests for the 5 x 6 related-system baseline suite driver."""

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

import run_final_related_system_baselines as suite

from kernel_optimization.baseline_styles import BASELINE_STYLE_NAMES


class FinalRelatedSystemBaselineTests(unittest.TestCase):
    def test_default_defines_five_kernels_by_six_styles(self) -> None:
        treatments = suite.selected_treatments()
        self.assertEqual(len(treatments), 30)
        self.assertEqual(
            tuple(kernel.key for kernel in suite.KERNELS),
            (
                "matmul",
                "rms_norm",
                "conv2d",
                "flash_attention",
                "fused_add_rms_norm",
            ),
        )
        self.assertEqual(
            tuple(style.key for style in suite.RELATED_STYLES),
            BASELINE_STYLE_NAMES[1:],
        )
        self.assertEqual(
            tuple(style.key for style in suite.ALL_STYLES),
            BASELINE_STYLE_NAMES,
        )
        self.assertEqual(
            [item.style.key for item in treatments[:6]],
            list(BASELINE_STYLE_NAMES),
        )

    def test_can_exclude_native_for_proxy_only_matrix(self) -> None:
        treatments = suite.selected_treatments(include_native=False)
        self.assertEqual(len(treatments), 25)
        self.assertTrue(
            all(treatment.style.key != "native" for treatment in treatments)
        )

    def test_command_uses_original_task_and_canonical_style_controls(self) -> None:
        treatment = suite.selected_treatments(
            only_kernels=["matmul"], only_styles=["kernelagent"]
        )[0]
        command = suite.build_command(
            treatment=treatment,
            output=Path("results/matmul/kernelagent"),
            max_search_seconds=1800.0,
            snapshot_interval_seconds=300.0,
            api_url="https://api.example/v1/chat/completions",
            api_model="test-model",
            api_key_env="KERNEL_OPT_API_KEY",
            search_until_time_budget=True,
            resume=True,
            repository_root=REPOSITORY_ROOT,
        )

        self.assertEqual(
            Path(command[command.index("--source") + 1]).name,
            "tilelang_matmul_kernel.py",
        )
        self.assertEqual(
            Path(command[command.index("--task") + 1]).name,
            "tilelang_matmul_task.json",
        )
        self.assertEqual(
            command[command.index("--baseline-style") + 1], "kernelagent"
        )
        self.assertEqual(
            command[command.index("--max-search-seconds") + 1], "1800.0"
        )
        self.assertNotIn("--agent-workers", command)
        self.assertNotIn("--evaluation-policy", command)
        self.assertNotIn("--strategy-allocation-policy", command)
        self.assertIn("--search-until-time-budget", command)
        self.assertIn("--resume", command)

    def test_filters_keep_deterministic_kernel_major_order(self) -> None:
        treatments = suite.selected_treatments(
            only_kernels=["conv2d", "flash_attention"],
            only_styles=["kernelbench", "avo"],
        )
        self.assertEqual(
            [(item.kernel.key, item.style.key) for item in treatments],
            [
                ("conv2d", "kernelbench"),
                ("conv2d", "avo"),
                ("flash_attention", "kernelbench"),
                ("flash_attention", "avo"),
            ],
        )

    def test_basic_suite_materializes_fixed_shapes_and_equal_measurement_budget(
        self,
    ) -> None:
        workload_suite = suite._builtin_workload_suite("basic")
        treatments = suite.selected_treatments(only_styles=["kernelagent"])
        expected_shapes = {
            "matmul": {"m": 1024, "n": 1024, "k": 1024},
            "rms_norm": {
                "rows": 4096,
                "hidden_size": 4096,
                "epsilon": 1e-06,
            },
            "conv2d": {
                "batch": 16,
                "in_height": 56,
                "in_width": 56,
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "stride": 1,
                "dilation": 1,
                "padding": 1,
            },
            "flash_attention": {
                "batch": 1,
                "heads": 8,
                "seq_len": 1024,
                "dim": 64,
                "is_causal": False,
            },
            "fused_add_rms_norm": {
                "rows": 4096,
                "hidden_size": 4096,
                "epsilon": 1e-06,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            paths, payloads = suite._prepare_treatment_tasks(
                REPOSITORY_ROOT,
                Path(temporary),
                treatments,
                workload_suite,
                dry_run=False,
                measurement_repeats=3,
                round_ceiling=128,
            )

            self.assertEqual(set(payloads), set(expected_shapes))
            for kernel, expected in expected_shapes.items():
                task = payloads[kernel]
                runtime = task["evaluator"]["runtime"]
                self.assertTrue(task["evaluator"]["persistent_process"])
                self.assertEqual(task["evaluator"]["worker_max_requests"], 32)
                self.assertEqual(task["workload"]["factory_arguments"], expected)
                self.assertEqual(task["budget"]["rounds"], 128)
                self.assertEqual(runtime["measurement_repeats"], 3)
                self.assertEqual(len(runtime["search_cases"]), 1)
                self.assertEqual(len(runtime["final_cases"]), 1)
                self.assertNotEqual(
                    runtime["search_cases"][0]["case_id"],
                    runtime["final_cases"][0]["case_id"],
                )
                plugin = importlib.import_module(
                    runtime["plugin"][:-3].replace("/", ".")
                )
                for raw_case in runtime["search_cases"] + runtime["final_cases"]:
                    case = dict(raw_case)
                    case["factory_arguments"] = dict(expected)
                    plugin.validate_case(case, task)
                task_path = paths[(kernel, "kernelagent")]
                self.assertEqual(json.loads(task_path.read_text()), task)

    def test_dry_run_selects_one_treatment_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "not-created"
            rendered = io.StringIO()
            with redirect_stdout(rendered):
                code = suite.main(
                    [
                        "--dry-run",
                        "--api-url",
                        "https://api.example/v1/chat/completions",
                        "--api-model",
                        "test-model",
                        "--output-root",
                        str(output_root),
                        "--only-kernel",
                        "fused_add_rms_norm",
                        "--only-style",
                        "tilefoundry",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertFalse(output_root.exists())
            output = rendered.getvalue()
            self.assertIn("Selected 1 treatments", output)
            self.assertIn("_tasks/fused_add_rms_norm/tilefoundry.json", output)
            self.assertIn("--baseline-style tilefoundry", output)
            self.assertIn("--search-until-time-budget", output)
            self.assertIn("Agent workers: 1", output)

    def test_default_output_and_time_cap_are_bounded(self) -> None:
        args = suite.build_parser().parse_args([])
        workload_suite = suite._builtin_workload_suite(args.shape_config)
        self.assertEqual(
            suite._default_output_root(workload_suite).parts[-2:],
            ("final_eval", "related_system_baselines_30m_basic"),
        )
        self.assertIsNone(args.output_root)
        self.assertEqual(args.max_search_seconds, 1800.0)
        self.assertEqual(args.budget_mode, "fixed-time")
        self.assertEqual(args.measurement_repeats, 3)
        self.assertEqual(args.time_budget_round_ceiling, 128)
        self.assertEqual(args.shape_config, "basic")
        self.assertIsNone(args.workload_suite)
        self.assertTrue(args.include_native)
        self.assertFalse(
            suite.build_parser().parse_args(["--exclude-native"]).include_native
        )

    def test_three_named_shape_configs_have_distinct_outputs(self) -> None:
        self.assertEqual(tuple(suite.SHAPE_CONFIGS), ("basic", "standard", "shape1"))
        outputs = {
            suite._default_output_root(suite._builtin_workload_suite(name)).name
            for name in suite.SHAPE_CONFIGS
        }
        self.assertEqual(
            outputs,
            {
                "related_system_baselines_30m_basic",
                "related_system_baselines_30m_standard",
                "related_system_baselines_30m_shape1",
            },
        )

    def test_each_named_config_uses_one_shape_for_search_and_final(self) -> None:
        for config_name in suite.SHAPE_CONFIGS:
            workload_suite = suite._builtin_workload_suite(config_name)
            for kernel in suite.KERNELS:
                workload = workload_suite["workloads"][kernel.key]
                search_cases = workload["search_cases"]
                final_cases = workload["final_cases"]
                self.assertEqual(len(search_cases), 1)
                self.assertEqual(len(final_cases), 1)
                self.assertNotEqual(
                    search_cases[0]["case_id"], final_cases[0]["case_id"]
                )
                primary = dict(workload["factory_arguments"])
                search_shape = dict(primary)
                search_shape.update(
                    dict(search_cases[0].get("factory_arguments") or {})
                )
                final_shape = dict(primary)
                final_shape.update(
                    dict(final_cases[0].get("factory_arguments") or {})
                )
                self.assertEqual(search_shape, final_shape)

    def test_all_named_shape_cases_satisfy_kernel_contracts(self) -> None:
        treatments = suite.selected_treatments(
            only_styles=["native"], include_native=True
        )
        with tempfile.TemporaryDirectory() as temporary:
            for config_name in suite.SHAPE_CONFIGS:
                workload_suite = suite._builtin_workload_suite(config_name)
                _paths, payloads = suite._prepare_treatment_tasks(
                    REPOSITORY_ROOT,
                    Path(temporary) / config_name,
                    treatments,
                    workload_suite,
                    dry_run=True,
                    measurement_repeats=3,
                    round_ceiling=128,
                )
                for kernel in suite.KERNELS:
                    task = payloads[kernel.key]
                    runtime = task["evaluator"]["runtime"]
                    plugin = importlib.import_module(
                        runtime["plugin"][:-3].replace("/", ".")
                    )
                    primary = dict(task["workload"]["factory_arguments"])
                    for raw_case in (
                        list(runtime["search_cases"])
                        + list(runtime["final_cases"])
                    ):
                        case = dict(raw_case)
                        arguments = dict(primary)
                        arguments.update(dict(case.get("factory_arguments") or {}))
                        case["factory_arguments"] = arguments
                        plugin.validate_case(case, task)

    def test_current_fixed_time_summary_requires_separate_timing_fields(self) -> None:
        legacy = {
            "elapsed_seconds": 1801.0,
            "termination_reason": "time-budget",
            "time_budget_exhausted": True,
            "final_validation_calls": 0,
        }
        current = {
            **legacy,
            "search_elapsed_seconds": 1801.0,
            "final_validation_seconds": 12.0,
            "total_elapsed_seconds": 1813.0,
            "final_validation_calls": 2,
        }

        self.assertFalse(
            suite._summary_uses_current_timing_protocol(
                legacy,
                budget_mode="fixed-time",
                expected_final_validations=1,
            )
        )
        self.assertTrue(
            suite._summary_uses_current_timing_protocol(
                current,
                budget_mode="fixed-time",
                expected_final_validations=1,
            )
        )

    def test_builtin_basic_can_resume_legacy_json_provenance(self) -> None:
        treatment = suite.selected_treatments(
            only_kernels=["matmul"], only_styles=["native"]
        )
        legacy_suite = suite._load_workload_suite(
            REPOSITORY_ROOT / "examples" / "related_system_basic_shapes.json"
        )
        builtin_suite = suite._builtin_workload_suite("basic")
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            paths, _payloads = suite._prepare_treatment_tasks(
                REPOSITORY_ROOT,
                output_root,
                treatment,
                legacy_suite,
                dry_run=False,
                measurement_repeats=3,
                round_ceiling=128,
            )
            archived_task = json.loads(
                paths[("matmul", "native")].read_text(encoding="utf-8")
            )

            suite._prepare_treatment_tasks(
                REPOSITORY_ROOT,
                output_root,
                treatment,
                builtin_suite,
                dry_run=False,
                measurement_repeats=3,
                round_ceiling=128,
            )
            changed = json.loads(json.dumps(archived_task))
            changed["workload"]["factory_arguments"]["m"] = 2048
            with self.assertRaisesRegex(SystemExit, "choose a fresh output root"):
                suite._write_immutable_json(
                    paths[("matmul", "native")], changed
                )

    def test_fixed_time_completion_requires_time_budget_termination(self) -> None:
        valid = {
            "termination_reason": "time-budget",
            "time_budget_exhausted": True,
        }
        early = {
            "termination_reason": "round-ceiling-before-time-budget",
            "time_budget_exhausted": False,
        }

        self.assertEqual(
            suite._completed_treatment_status(
                valid, budget_mode="fixed-time", reused=False
            ),
            "completed",
        )
        self.assertEqual(
            suite._completed_treatment_status(
                early, budget_mode="fixed-time", reused=False
            ),
            "ended-early",
        )


if __name__ == "__main__":
    unittest.main()
