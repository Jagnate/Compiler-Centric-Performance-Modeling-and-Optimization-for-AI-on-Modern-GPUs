"""Regression tests for auditable related-system style presets."""

from __future__ import annotations

import json
from dataclasses import replace
import unittest

from kernel_optimization.baseline_styles import (
    BASELINE_STYLE_NAMES,
    get_baseline_style,
    resolve_baseline_style,
)
from kernel_optimization.cli import _explicit_baseline_fields, build_parser
from kernel_optimization.prompts import build_optimization_prompt
from kernel_optimization.schema import BudgetConfig, Candidate, TaskSpec


class BaselineStyleTests(unittest.TestCase):
    def test_cli_defaults_to_unchanged_native_style(self) -> None:
        args = build_parser().parse_args(
            ["--source", "kernel.py", "--task", "task.json"]
        )
        self.assertEqual(args.baseline_style, "native")
        self.assertEqual(args.evaluation_policy, "tilesight")
        self.assertEqual(args.strategy_allocation_policy, "ai-planned")

    def test_all_documented_styles_are_registered(self) -> None:
        self.assertEqual(
            BASELINE_STYLE_NAMES,
            (
                "native",
                "kernelagent",
                "kernelevolve",
                "kernelbench",
                "avo",
                "tilefoundry",
            ),
        )
        for name in BASELINE_STYLE_NAMES:
            self.assertEqual(get_baseline_style(name).name, name)

    def test_native_style_preserves_task_and_requested_policies(self) -> None:
        task = _task()
        resolved = _resolve(
            "native",
            task,
            evaluation_policy="cuda-event",
            explicit_fields=frozenset({"evaluation_policy"}),
        )

        self.assertIs(resolved.task, task)
        self.assertEqual(resolved.evaluation_policy, "cuda-event")
        self.assertEqual(resolved.task.budget.beam_width, 3)
        self.assertNotIn("baseline_style_protocol", resolved.task.metadata)
        self.assertFalse(resolved.emulated)

    def test_kernelagent_uses_single_agent_ncu_guided_beam(self) -> None:
        task = replace(
            _task(),
            evaluator={"runtime": {"ncu_set": "tilesight-targeted"}},
        )
        resolved = _resolve("kernelagent", task)

        self.assertEqual(resolved.agent_workers, 1)
        self.assertEqual(resolved.evaluation_policy, "ncu")
        self.assertEqual(resolved.tir_evidence_policy, "hidden")
        self.assertEqual(resolved.strategy_allocation_policy, "ai-planned")
        self.assertEqual(resolved.structural_search_policy, "observe")
        self.assertEqual(resolved.task.budget.beam_width, 3)
        self.assertEqual(resolved.task.evaluator["runtime"]["ncu_set"], "full")
        self.assertTrue(resolved.canonical)

    def test_persistent_and_single_incumbent_topologies_are_distinct(self) -> None:
        task = _task()
        evolve = _resolve("kernelevolve", task)
        kernelbench = _resolve("kernelbench", task)
        avo = _resolve("avo", task)
        tilefoundry = _resolve("tilefoundry", task)

        self.assertEqual(evolve.task.budget.beam_width, 3)
        for resolved in (kernelbench, avo, tilefoundry):
            self.assertEqual(resolved.task.budget.beam_width, 1)
            self.assertEqual(resolved.task.budget.rounds, 4)
            self.assertEqual(resolved.task.budget.proposals_per_round, 6)
        self.assertEqual(kernelbench.evaluation_policy, "cuda-event")
        self.assertEqual(avo.evaluation_policy, "ncu")
        self.assertEqual(tilefoundry.evaluation_policy, "cuda-event")

    def test_explicit_override_is_applied_and_audited(self) -> None:
        resolved = _resolve(
            "kernelagent",
            _task(),
            agent_workers=2,
            explicit_fields=frozenset({"agent_workers"}),
        )

        self.assertEqual(resolved.agent_workers, 2)
        self.assertFalse(resolved.canonical)
        self.assertEqual(
            resolved.explicit_overrides["agent_workers"],
            {"preset": 1, "effective": 2},
        )
        manifest = resolved.to_dict()
        self.assertEqual(manifest["mode"], "style-emulation")
        self.assertIn("not an exact reimplementation", manifest["claim"])

    def test_style_protocol_is_visible_in_generation_prompt(self) -> None:
        resolved = _resolve("tilefoundry", _task())
        parent = Candidate.seed(resolved.task, "def kernel():\n    return 1\n", "k.py")
        prompt = json.loads(
            build_optimization_prompt(
                resolved.task,
                parent,
                evidence={},
                history=[],
                count=2,
            )
        )

        protocol = prompt["task"]["optimization_protocol"]
        self.assertEqual(protocol["name"], "tilefoundry")
        self.assertEqual(protocol["mode"], "style-emulation")
        self.assertTrue(protocol["guidance"])
        self.assertNotIn("unsupported_capabilities", protocol)

    def test_explicit_option_detection_supports_equals_form(self) -> None:
        fields = _explicit_baseline_fields(
            [
                "--baseline-style",
                "kernelagent",
                "--agent-workers=1",
                "--evaluation-policy",
                "cuda-event",
                "--no-compiled-dedup",
            ]
        )
        self.assertEqual(
            fields,
            frozenset(
                {
                    "agent_workers",
                    "evaluation_policy",
                    "compiled_deduplication",
                }
            ),
        )


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="baseline-style-test",
        description="test kernel",
        reference="return the same values",
        entrypoint="make_kernel",
        budget=BudgetConfig(rounds=4, proposals_per_round=6, beam_width=3),
    )


def _resolve(
    name: str,
    task: TaskSpec,
    *,
    agent_workers: int = 1,
    evaluation_policy: str = "tilesight",
    explicit_fields: frozenset[str] = frozenset(),
):
    return resolve_baseline_style(
        name=name,
        task=task,
        agent_workers=agent_workers,
        evaluation_policy=evaluation_policy,
        tir_evidence_policy="auto",
        selection_policy="adaptive",
        profile_policy="milestone",
        compiled_deduplication=True,
        structural_search_policy="enforce",
        strategy_allocation_policy="ai-planned",
        explicit_fields=explicit_fields,
    )


if __name__ == "__main__":
    unittest.main()
