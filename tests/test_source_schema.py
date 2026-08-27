"""Tests for source candidates and static source validation."""

from __future__ import annotations

from pathlib import Path
import unittest

from kernel_optimization.schema import Candidate, CandidateProposal, TaskSpec
from kernel_optimization.source_validation import SourceValidationError, SourceValidator


def make_task() -> TaskSpec:
    return TaskSpec(
        task_id="source-schema-test",
        description="Optimize a Python kernel source.",
        reference="Return the same value as the reference implementation.",
        entrypoint="kernel",
        constraints={
            "required_fragments": ["def kernel"],
            "forbidden_fragments": ["subprocess"],
        },
    )


class SourceCandidateTests(unittest.TestCase):
    def test_proposal_parser_reports_missing_source_code_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty source_code"):
            CandidateProposal.from_dict(
                {"hypothesis": "Describe an optimization without implementing it."}
            )

    def test_candidate_identity_depends_on_complete_source_not_parent(self) -> None:
        task = make_task()
        seed = Candidate.seed(task, "def kernel(x):\n    return x\n", "kernel.py")
        first = Candidate.from_proposal(
            task,
            seed,
            CandidateProposal(
                hypothesis="Use one expression.",
                source_code="def kernel(x):\n    return x + 0\n",
            ),
            generation=1,
        )
        other_parent = Candidate.from_proposal(
            task,
            seed,
            CandidateProposal(
                hypothesis="Use a local value.",
                source_code="def kernel(x):\n    y = x\n    return y\n",
            ),
            generation=1,
        )
        same_source = Candidate.from_proposal(
            task,
            other_parent,
            CandidateProposal(
                hypothesis="Reach the same implementation by another path.",
                source_code=first.source_code,
            ),
            generation=2,
        )

        self.assertEqual(first.candidate_id, same_source.candidate_id)
        self.assertEqual(first.source_sha256, same_source.source_sha256)

    def test_source_name_is_reduced_to_a_safe_basename(self) -> None:
        task = make_task()
        candidate = Candidate.seed(
            task,
            "def kernel(x):\n    return x\n",
            "../../kernel.py",
        )

        self.assertEqual(candidate.source_name, "kernel.py")


class SourceValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = make_task()
        self.validator = SourceValidator()

    def test_accepts_valid_python_with_preserved_entrypoint(self) -> None:
        self.validator.validate(
            self.task,
            "def kernel(x):\n    return x\n",
            "kernel.py",
        )

    def test_rejects_syntax_error(self) -> None:
        with self.assertRaisesRegex(SourceValidationError, "valid Python syntax"):
            self.validator.validate(self.task, "def kernel(:\n", "kernel.py")

    def test_rejects_removed_entrypoint(self) -> None:
        with self.assertRaisesRegex(SourceValidationError, "entrypoint"):
            self.validator.validate(
                self.task,
                "def different(x):\n    return x\n# def kernel\n",
                "kernel.py",
            )

    def test_rejects_entrypoint_moved_inside_another_function(self) -> None:
        source = (
            "def wrapper():\n"
            "    def kernel(x):\n"
            "        return x\n"
            "    return kernel\n"
        )
        with self.assertRaisesRegex(SourceValidationError, "entrypoint"):
            self.validator.validate(self.task, source, "kernel.py")

    def test_included_tilelang_sources_match_their_task_contracts(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        examples = (
            ("tilelang_matmul_task.json", "tilelang_matmul_kernel.py"),
            (
                "tilelang_flash_attention_task.json",
                "tilelang_flash_attention_kernel.py",
            ),
            ("tilelang_rms_norm_task.json", "tilelang_rms_norm_kernel.py"),
            (
                "tilelang_fused_add_rms_norm_task.json",
                "tilelang_fused_add_rms_norm_kernel.py",
            ),
            ("tilelang_conv2d_task.json", "tilelang_conv2d_kernel.py"),
        )
        for task_name, source_name in examples:
            with self.subTest(task=task_name):
                task = TaskSpec.from_json_file(
                    repository_root / "examples" / task_name
                )
                source_path = repository_root / "examples" / source_name
                self.validator.validate(
                    task,
                    source_path.read_text(encoding="utf-8"),
                    source_path.name,
                )

    def test_rejects_forbidden_fragment(self) -> None:
        with self.assertRaisesRegex(SourceValidationError, "forbidden fragment"):
            self.validator.validate(
                self.task,
                "import subprocess\ndef kernel(x):\n    return x\n",
                "kernel.py",
            )


if __name__ == "__main__":
    unittest.main()
