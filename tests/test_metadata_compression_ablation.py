"""One-command metadata-compression treatment and aggregation tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from examples import run_metadata_compression_ablation as ablation
from kernel_optimization.archive import ArtifactStore
from kernel_optimization.controller import OptimizationController
from kernel_optimization.schema import (
    Candidate,
    CandidateProposal,
    CandidateRecord,
    TaskSpec,
)


class MetadataCompressionAblationTests(unittest.TestCase):
    def test_defaults_select_the_stable_matmul_workload(self) -> None:
        args = ablation.build_parser().parse_args([])

        self.assertEqual(args.source, "examples/tilelang_matmul_kernel.py")
        self.assertEqual(args.task, "examples/tilelang_matmul_task.json")
        self.assertEqual(
            args.output, "results/final_eval/metadata_compression_matmul"
        )

    def test_none_controller_history_keeps_all_raw_candidate_metadata(self) -> None:
        task = TaskSpec(
            task_id="raw-history-test",
            description="Inspect uncompressed history.",
            reference="Preserve output.",
            entrypoint="kernel",
        )
        seed = Candidate.seed(
            task, "def kernel(x):\n    return x\n", "kernel.py"
        )
        records = {}
        parent = seed
        for generation in range(25):
            candidate = (
                seed
                if generation == 0
                else Candidate.from_proposal(
                    task,
                    parent,
                    CandidateProposal(
                        hypothesis="candidate %d" % generation,
                        source_code=(
                            "MARKER = %d\n\ndef kernel(x):\n    return x\n"
                            % generation
                        ),
                        expected_effect={"raw": generation},
                        metadata={"full_metadata": "kept-%d" % generation},
                    ),
                    generation=generation,
                )
            )
            records[candidate.candidate_id] = CandidateRecord(candidate=candidate)
            parent = candidate

        with tempfile.TemporaryDirectory(prefix="raw-history-controller-") as directory:
            controller = OptimizationController(
                task=task,
                source_code=seed.source_code,
                source_name=seed.source_name,
                generator=object(),
                backend=object(),
                store=ArtifactStore(Path(directory)),
                metadata_compression_policy="none",
            )
            controller.records = records
            history = controller._history()

        self.assertEqual(len(history), 25)
        self.assertIn("candidate", history[-1])
        self.assertEqual(
            history[-1]["candidate"]["proposal_metadata"]["full_metadata"],
            "kept-24",
        )
        self.assertNotIn("source_code", history[-1]["candidate"])

    def test_command_controls_the_context_policy_and_sampling_interval(self) -> None:
        command = ablation.build_treatment_command(
            source=Path("/repo/kernel.py"),
            task=Path("/repo/task.json"),
            output=Path("/repo/results/compressed"),
            policy="key-metrics-v1",
            snapshot_interval_seconds=300.0,
            api_max_input_tokens=60000,
            resume=False,
            forwarded=["--max-search-seconds", "1800"],
        )

        self.assertIn("--metadata-compression-policy", command)
        self.assertEqual(
            command[command.index("--metadata-compression-policy") + 1],
            "key-metrics-v1",
        )
        self.assertEqual(
            command[command.index("--incumbent-snapshot-interval-seconds") + 1],
            "300.0",
        )
        self.assertEqual(command[-2:], ["--max-search-seconds", "1800"])

    def test_exports_call_round_and_time_token_tables(self) -> None:
        with tempfile.TemporaryDirectory(prefix="compression-ablation-") as directory:
            root = Path(directory)
            _write_treatment(root / "compressed", "key-metrics-v1", 120, 30)
            _write_treatment(
                root / "uncompressed",
                "none",
                900,
                35,
                prompt_budget_failure=True,
            )

            summaries = ablation._write_comparison_artifacts(root)

            self.assertEqual(summaries[0]["status"], "completed")
            self.assertEqual(summaries[1]["status"], "prompt-budget-exceeded")
            with (root / "token_usage_by_call.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                calls = list(csv.DictReader(handle))
            with (root / "token_usage_by_time.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                times = list(csv.DictReader(handle))

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["actual_input_tokens"], "120.0")
            self.assertEqual(calls[1]["error_code"], "local_prompt_budget_exceeded")
            self.assertEqual(len(times), 2)
            report = (root / "compression_comparison.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Per-Round Growth", report)
            self.assertIn("prompt-budget-exceeded", report)


def _write_treatment(
    root: Path,
    policy: str,
    input_tokens: int,
    output_tokens: int,
    *,
    prompt_budget_failure: bool = False,
) -> None:
    api = root / "api_calls"
    api.mkdir(parents=True)
    error = (
        {
            "error_code": "local_prompt_budget_exceeded",
            "message": "prompt exceeded the local budget",
        }
        if prompt_budget_failure
        else None
    )
    provider = {
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "prompt_size": {
            "estimated_input_tokens": 61000
            if prompt_budget_failure
            else input_tokens,
            "max_output_tokens": 12000,
        },
    }
    if error:
        provider["error"] = error
    (api / "0001_round-001-generate.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "kind": "round-001-generate",
                "metadata": {"round": 1, "provider": provider},
                "exchange": {},
            }
        ),
        encoding="utf-8",
    )
    state = {
        "completed_round": 1,
        "best_latency_ms": 1.0,
        "generator_usage": provider["usage"],
    }
    (root / "state.json").write_text(json.dumps(state), encoding="utf-8")
    if prompt_budget_failure:
        (root / "failure.json").write_text(
            json.dumps(
                {
                    "error_type": "PromptBudgetError",
                    "message": "compressed API prompt exceeded local budget",
                }
            ),
            encoding="utf-8",
        )
    else:
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "completed_rounds": 1,
                    "best_latency_ms": 1.0,
                    "speedup_over_seed": 1.1,
                }
            ),
            encoding="utf-8",
        )
    with (root / "incumbent_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "reason",
                "scheduled_elapsed_seconds",
                "elapsed_seconds",
                "completed_round",
                "active_round",
                "api_input_tokens",
                "api_output_tokens",
                "api_total_tokens",
                "incumbent_latency_ms",
                "speedup_over_seed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "reason": "round-completed",
                "elapsed_seconds": 100,
                "completed_round": 1,
                "api_input_tokens": input_tokens,
                "api_output_tokens": output_tokens,
                "api_total_tokens": input_tokens + output_tokens,
                "incumbent_latency_ms": 1.0,
                "speedup_over_seed": 1.1,
            }
        )


if __name__ == "__main__":
    unittest.main()
