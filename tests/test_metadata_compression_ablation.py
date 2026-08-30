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
from kernel_optimization.factory import create_backend
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
        self.assertEqual(args.compressed_api_max_input_tokens, 60000)
        self.assertEqual(args.uncompressed_api_max_input_tokens, 1000000)
        self.assertEqual(args.compressed_context_target_tokens, 60000)
        self.assertEqual(args.rounds, 8)

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
            compressed_context_target_tokens=60000,
            metadata_history_limit=64,
            metadata_lesson_limit=64,
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
        self.assertEqual(
            command[command.index("--compressed-context-target-tokens") + 1],
            "60000",
        )
        self.assertEqual(
            command[command.index("--metadata-history-limit") + 1], "64"
        )
        self.assertEqual(command[command.index("--agent-workers") + 1], "1")
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
            self.assertEqual(summaries[1]["status"], "local-limit-exceeded")
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
            self.assertIn("local-limit-exceeded", report)
            self.assertTrue((root / "token_usage_by_5min.csv").is_file())

    def test_provider_request_limit_is_an_expected_terminal_boundary(self) -> None:
        value = ablation._classify_limit_error(
            {
                "status_code": 429,
                "error_code": "rate_limit_exceeded",
                "response_body": (
                    "Request too large for model on tokens per min (TPM): "
                    "Limit 200000, Requested 201696. The input or output tokens "
                    "must be reduced."
                ),
            }
        )

        self.assertEqual(value["kind"], "provider_token_limit_exceeded")
        self.assertEqual(value["limit_tokens"], 200000)
        self.assertEqual(value["requested_tokens"], 201696)

    def test_provider_limit_failure_is_archived_as_expected_status(self) -> None:
        with tempfile.TemporaryDirectory(prefix="provider-limit-") as directory:
            root = Path(directory)
            api = root / "api_calls"
            api.mkdir()
            error = {
                "status_code": 429,
                "error_code": "rate_limit_exceeded",
                "response_body": (
                    "Request too large on tokens per min: Limit 200000, "
                    "Requested 240000."
                ),
            }
            (api / "0001_round-003-generate-failed.json").write_text(
                json.dumps(
                    {
                        "kind": "round-003-generate-failed",
                        "metadata": {
                            "round": 3,
                            "provider": {
                                "prompt_size": {"estimated_input_tokens": 240000},
                                "error": error,
                            },
                        },
                        "exchange": {"error": error},
                    }
                ),
                encoding="utf-8",
            )
            (root / "task.json").write_text(
                json.dumps(
                    {
                        "run": {
                            "generator": {
                                "max_input_tokens": 1000000,
                                "compressed_context_target_tokens": None,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "failure.json").write_text(
                json.dumps({"error_type": "HostedApiError", "message": "429"}),
                encoding="utf-8",
            )

            calls = ablation._load_api_calls(root, "uncompressed", "none")
            summary = ablation._summarize_treatment(
                root, "uncompressed", "none", calls
            )

            self.assertEqual(calls[0]["limit_kind"], "provider_token_limit_exceeded")
            self.assertEqual(summary["status"], "provider-limit-exceeded")
            self.assertEqual(summary["provider_limit_tokens"], 200000)
            self.assertEqual(summary["provider_requested_tokens"], 240000)

    def test_five_minute_rows_report_incremental_and_cumulative_tokens(self) -> None:
        snapshots = [
            {
                "treatment": "compressed",
                "metadata_compression_policy": "key-metrics-v1",
                "reason": "interval",
                "scheduled_elapsed_seconds": "300",
                "elapsed_seconds": "300.1",
                "api_input_tokens": "100",
                "api_output_tokens": "20",
                "api_total_tokens": "120",
                "completed_round": "0",
                "active_round": "1",
            },
            {
                "treatment": "compressed",
                "metadata_compression_policy": "key-metrics-v1",
                "reason": "interval",
                "scheduled_elapsed_seconds": "600",
                "elapsed_seconds": "600.1",
                "api_input_tokens": "350",
                "api_output_tokens": "50",
                "api_total_tokens": "400",
                "completed_round": "1",
                "active_round": "2",
            },
        ]

        rows = ablation._five_minute_rows(
            snapshots, snapshot_interval_seconds=300.0
        )

        self.assertEqual(rows[0]["window_label"], "0-5 min")
        self.assertEqual(rows[0]["tokens_in_window"], 120.0)
        self.assertEqual(rows[1]["window_label"], "5-10 min")
        self.assertEqual(rows[1]["tokens_in_window"], 280.0)
        self.assertEqual(rows[1]["cumulative_total_tokens"], 400.0)

    def test_materializes_same_eight_round_task_for_both_treatments(self) -> None:
        with tempfile.TemporaryDirectory(prefix="compression-task-") as directory:
            root = Path(directory)
            source = root / "task.json"
            source.write_text(
                json.dumps({"task_id": "x", "budget": {"rounds": 4}}),
                encoding="utf-8",
            )

            path = ablation._materialize_ablation_task(source, root, 8)
            value = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(value["budget"]["rounds"], 8)
            self.assertEqual(
                value["metadata"]["metadata_compression_ablation"]["treatments"],
                ["compressed", "uncompressed"],
            )

    def test_materialized_task_preserves_original_evaluator_working_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="compression-task-path-") as directory:
            repository = Path(directory) / "repository"
            examples = repository / "examples"
            output = repository / "results" / "final_eval" / "compression"
            examples.mkdir(parents=True)
            output.mkdir(parents=True)
            source = examples / "task.json"
            source.write_text(
                json.dumps(
                    {
                        "task_id": "path-test",
                        "description": "Preserve evaluator path semantics.",
                        "reference": "Return the expected output.",
                        "entrypoint": "kernel",
                        "budget": {"rounds": 4},
                        "evaluator": {
                            "type": "command",
                            "command": ["python3", "examples/evaluator.py"],
                            "working_directory": "..",
                        },
                    }
                ),
                encoding="utf-8",
            )

            path = ablation._materialize_ablation_task(source, output, 8)
            value = json.loads(path.read_text(encoding="utf-8"))
            backend = create_backend(TaskSpec.from_json_file(path), path.parent)

            self.assertEqual(
                Path(value["evaluator"]["working_directory"]),
                repository.resolve(),
            )
            self.assertEqual(backend.working_directory, repository.resolve())

    def test_upgrades_legacy_relative_evaluator_path_for_resume(self) -> None:
        with tempfile.TemporaryDirectory(prefix="compression-task-resume-") as directory:
            repository = Path(directory) / "repository"
            examples = repository / "examples"
            output = repository / "results" / "compression"
            examples.mkdir(parents=True)
            output.mkdir(parents=True)
            source = examples / "task.json"
            task = {
                "task_id": "resume-path-test",
                "budget": {"rounds": 4},
                "evaluator": {
                    "type": "command",
                    "command": ["python3", "examples/evaluator.py"],
                    "working_directory": "..",
                },
            }
            source.write_text(json.dumps(task), encoding="utf-8")
            legacy = dict(task)
            legacy["budget"] = {"rounds": 8}
            legacy["metadata"] = {
                "metadata_compression_ablation": {
                    "source_task": str(source.resolve()),
                    "rounds": 8,
                    "treatments": ["compressed", "uncompressed"],
                }
            }
            (output / "ablation_task.json").write_text(
                json.dumps(legacy, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output / "uncompressed").mkdir()

            path = ablation._materialize_ablation_task(source, output, 8)
            value = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(
                Path(value["evaluator"]["working_directory"]),
                repository.resolve(),
            )


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
