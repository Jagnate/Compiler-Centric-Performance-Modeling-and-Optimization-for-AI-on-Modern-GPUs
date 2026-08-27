"""Tests for the final Fused Add + RMSNorm ablation driver."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "examples"))

import run_final_fused_add_rms_norm_ablation as fused_ablation
import run_final_matmul_ablation as shared_driver


class FinalFusedAddRmsNormAblationTests(unittest.TestCase):
    def test_injects_fused_workload_and_report_defaults(self) -> None:
        arguments = fused_ablation.build_forwarded_arguments([])
        parsed = shared_driver.build_parser().parse_args(arguments)

        self.assertEqual(
            parsed.source.name, "tilelang_fused_add_rms_norm_kernel.py"
        )
        self.assertEqual(
            parsed.task.name, "tilelang_fused_add_rms_norm_task.json"
        )
        self.assertEqual(
            parsed.output_root, fused_ablation.DEFAULT_OUTPUT_ROOT
        )
        self.assertEqual(
            parsed.output_root.parts[-2:],
            ("final_eval", "fused_ablation"),
        )
        self.assertEqual(
            parsed.report_title, "Final Fused Add + RMSNorm Ablation"
        )

    def test_user_arguments_override_defaults_and_keep_four_treatments(self) -> None:
        custom_output = Path("custom-fused-output")
        arguments = fused_ablation.build_forwarded_arguments(
            ["--output-root", str(custom_output), "--agent-workers", "2"]
        )
        parsed = shared_driver.build_parser().parse_args(arguments)

        self.assertEqual(parsed.output_root, custom_output)
        self.assertEqual(parsed.agent_workers, 2)
        self.assertEqual(
            [item.directory for item in shared_driver.TREATMENTS],
            [
                "01_full_system",
                "02_no_tir",
                "03_ncu_only",
                "04_ai_unconstrained",
            ],
        )

    def test_direct_dry_run_uses_shared_controller(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            return_code = fused_ablation.main(
                [
                    "--dry-run",
                    "--api-url",
                    "https://api.example/v1/chat/completions",
                    "--api-model",
                    "test-model",
                    "--only",
                    "01_full_system",
                ]
            )

        self.assertEqual(return_code, 0)
        rendered = output.getvalue()
        self.assertIn("tilelang_fused_add_rms_norm_kernel.py", rendered)
        self.assertIn("tilelang_fused_add_rms_norm_task.json", rendered)
        self.assertIn("01_full_system", rendered)


if __name__ == "__main__":
    unittest.main()
