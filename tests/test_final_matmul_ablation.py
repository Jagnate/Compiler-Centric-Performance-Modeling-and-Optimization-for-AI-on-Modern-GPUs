"""Tests for the final Matmul ablation experiment driver."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "examples"))

import run_final_matmul_ablation as ablation


class FinalMatmulAblationTests(unittest.TestCase):
    def test_defines_four_orthogonal_treatments(self) -> None:
        treatments = {item.directory: item for item in ablation.TREATMENTS}
        self.assertEqual(
            list(treatments),
            [
                "01_full_system",
                "02_no_tir",
                "03_ncu_only",
                "04_ai_unconstrained",
            ],
        )
        self.assertEqual(treatments["01_full_system"].tir_evidence_policy, "visible")
        self.assertEqual(treatments["02_no_tir"].tir_evidence_policy, "hidden")
        self.assertEqual(treatments["03_ncu_only"].evaluation_policy, "ncu")
        self.assertEqual(
            treatments["04_ai_unconstrained"].strategy_allocation_policy,
            "unconstrained",
        )

    def test_command_keeps_common_controls_and_can_resume(self) -> None:
        treatment = ablation.TREATMENTS[2]
        command = ablation.build_command(
            treatment=treatment,
            source=Path("seed.py"),
            task=Path("task.json"),
            output=Path("result"),
            agent_workers=2,
            snapshot_interval_seconds=300.0,
            api_url="https://api.example/v1/chat/completions",
            api_model="test-model",
            api_key_env="KERNEL_OPT_API_KEY",
            resume=True,
        )
        self.assertEqual(command[command.index("--evaluation-policy") + 1], "ncu")
        self.assertEqual(command[command.index("--selection-policy") + 1], "adaptive")
        self.assertEqual(
            command[command.index("--structural-search-policy") + 1], "enforce"
        )
        self.assertEqual(command[command.index("--agent-workers") + 1], "2")
        self.assertIn("--resume", command)

    def test_matmul_report_title_remains_the_default(self) -> None:
        parsed = ablation.build_parser().parse_args([])
        self.assertEqual(parsed.report_title, "Final Matmul Ablation")


if __name__ == "__main__":
    unittest.main()
