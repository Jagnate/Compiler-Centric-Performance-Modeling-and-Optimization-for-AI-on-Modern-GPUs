"""Wall-clock incumbent snapshot persistence and selection tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import time
import unittest

from kernel_optimization.archive import ArtifactStore
from kernel_optimization.incumbent_tracking import PeriodicIncumbentRecorder
from kernel_optimization.schema import (
    Candidate,
    CandidateProposal,
    CandidateRecord,
    Measurement,
    ModelEvaluation,
    ProfileEvaluation,
    TaskSpec,
)


SOURCE = "VALUE = %d\n\ndef kernel(x):\n    return x\n"


class IncumbentTrackingTests(unittest.TestCase):
    def test_background_thread_records_fixed_intervals(self) -> None:
        task = _task()
        seed = Candidate.seed(task, SOURCE % 0, "kernel.py")
        with tempfile.TemporaryDirectory(prefix="incumbent-periodic-") as directory:
            store = ArtifactStore(Path(directory))
            store.initialize(task, seed)
            store.save_candidate(_record(seed, measured=2.0, predicted=2.2))
            store.save_state(_state("seed_completed", seed.candidate_id))
            recorder = PeriodicIncumbentRecorder(store, interval_seconds=0.02)
            recorder.start()
            deadline = time.monotonic() + 1.0
            rows = []
            while time.monotonic() < deadline:
                if recorder.jsonl_path.is_file():
                    rows = [
                        json.loads(line)
                        for line in recorder.jsonl_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                    ]
                if len([row for row in rows if row["reason"] == "interval"]) >= 2:
                    break
                time.sleep(0.005)
            recorder.stop()

        interval_rows = [row for row in rows if row["reason"] == "interval"]
        self.assertGreaterEqual(len(interval_rows), 2)
        self.assertAlmostEqual(interval_rows[0]["scheduled_elapsed_seconds"], 0.02)
        self.assertAlmostEqual(interval_rows[1]["scheduled_elapsed_seconds"], 0.04)
        self.assertGreaterEqual(interval_rows[0]["elapsed_seconds"], 0.02)

    def test_records_verified_incumbent_metrics_on_event_and_interval(self) -> None:
        task = _task()
        seed = Candidate.seed(task, SOURCE % 0, "kernel.py")
        clock = _FakeClock(100.0)
        with tempfile.TemporaryDirectory(prefix="incumbent-history-") as directory:
            store = ArtifactStore(Path(directory))
            store.initialize(task, seed)
            seed_record = _record(seed, measured=3.0, predicted=2.8)
            store.save_candidate(seed_record)
            store.save_state(_state("seed_completed", seed.candidate_id))

            recorder = PeriodicIncumbentRecorder(
                store, interval_seconds=300.0, clock=clock
            )
            recorder.start(started_at=clock())

            faster = Candidate.from_proposal(
                task,
                seed,
                CandidateProposal("Use a faster schedule.", SOURCE % 1),
                generation=1,
            )
            faster_record = _record(
                faster,
                measured=1.5,
                predicted=2.0,
                profile=True,
            )
            store.save_candidate(faster_record)

            incorrect = Candidate.from_proposal(
                task,
                seed,
                CandidateProposal("Invalid but model-favored schedule.", SOURCE % 2),
                generation=1,
            )
            store.save_candidate(
                CandidateRecord(
                    candidate=incorrect,
                    model=ModelEvaluation(valid=True, predicted_latency_ms=0.1),
                    measurement=Measurement(correct=False, error="mismatch"),
                )
            )
            store.save_state(_state("measured", seed.candidate_id))
            clock.value = 250.0
            changed = recorder.record_if_changed("incumbent-updated")
            self.assertIsNotNone(changed)
            self.assertIsNone(recorder.record_if_changed("incumbent-updated"))

            clock.value = 400.0
            interval = recorder._record(
                reason="interval",
                scheduled_elapsed_seconds=300.0,
                changed_only=False,
            )
            recorder.stop()

            rows = [
                json.loads(line)
                for line in recorder.jsonl_path.read_text(encoding="utf-8").splitlines()
            ]
            with recorder.csv_path.open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual([row["sequence"] for row in rows], [1, 2, 3])
        self.assertEqual(rows[0]["incumbent"]["candidate_id"], seed.candidate_id)
        self.assertEqual(changed["incumbent"]["candidate_id"], faster.candidate_id)
        self.assertEqual(interval["reason"], "interval")
        self.assertEqual(interval["scheduled_elapsed_seconds"], 300.0)
        self.assertEqual(interval["incumbent_latency_ms"], 1.5)
        self.assertEqual(interval["speedup_over_seed"], 2.0)
        self.assertEqual(
            interval["incumbent"]["model"]["metrics"]["registers_per_thread"],
            64,
        )
        self.assertEqual(
            interval["incumbent"]["profile"]["metrics"]["dram_utilization"],
            71.0,
        )
        self.assertEqual(len(csv_rows), 3)
        self.assertEqual(csv_rows[-1]["candidate_id"], faster.candidate_id)
        self.assertEqual(csv_rows[-1]["registers_per_thread"], "64")
        self.assertEqual(csv_rows[-1]["reason"], "interval")
        self.assertEqual(interval["api_usage"]["input_tokens"], 100.0)
        self.assertEqual(interval["api_usage"]["output_tokens"], 50.0)
        self.assertEqual(interval["api_usage"]["total_tokens"], 150.0)
        self.assertEqual(csv_rows[-1]["api_total_tokens"], "150.0")
        self.assertEqual(
            csv_rows[-1]["metadata_compression_policy"], "key-metrics-v1"
        )

    def test_resume_continues_sequence_and_active_elapsed_time(self) -> None:
        task = _task()
        seed = Candidate.seed(task, SOURCE % 0, "kernel.py")
        with tempfile.TemporaryDirectory(prefix="incumbent-resume-") as directory:
            store = ArtifactStore(Path(directory))
            store.initialize(task, seed)
            store.save_candidate(_record(seed, measured=4.0, predicted=3.5))
            store.save_state(_state("seed_completed", seed.candidate_id))

            first_clock = _FakeClock(10.0)
            first = PeriodicIncumbentRecorder(
                store, interval_seconds=300.0, clock=first_clock
            )
            first.start(started_at=first_clock())
            first_clock.value = 130.0
            first.record_now("run-failed")
            first.stop()

            resumed_clock = _FakeClock(500.0)
            resumed = PeriodicIncumbentRecorder(
                store, interval_seconds=300.0, clock=resumed_clock
            )
            resumed.start(started_at=resumed_clock(), resumed=True)
            resumed_clock.value = 680.0
            snapshot = resumed.record_now("manual-checkpoint")
            resumed.stop()

            rows = [
                json.loads(line)
                for line in resumed.jsonl_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([row["sequence"] for row in rows], [1, 2, 3, 4])
        self.assertEqual(rows[2]["reason"], "run-resumed")
        self.assertAlmostEqual(rows[2]["elapsed_seconds"], 120.0)
        self.assertAlmostEqual(snapshot["elapsed_seconds"], 300.0)

    def test_final_phase_uses_official_finalized_candidate(self) -> None:
        task = _task()
        seed = Candidate.seed(task, SOURCE % 0, "kernel.py")
        faster = Candidate.from_proposal(
            task,
            seed,
            CandidateProposal("Fast on the search shape.", SOURCE % 1),
            generation=1,
        )
        with tempfile.TemporaryDirectory(prefix="incumbent-final-") as directory:
            store = ArtifactStore(Path(directory))
            store.initialize(task, seed)
            seed_record = _record(seed, measured=3.0, predicted=3.0)
            seed_record.final_measurement = Measurement(
                correct=True, latency_ms=2.0, samples_ms=[2.0]
            )
            faster_record = _record(faster, measured=1.0, predicted=1.0)
            faster_record.final_measurement = Measurement(
                correct=True, latency_ms=2.5, samples_ms=[2.5]
            )
            store.save_candidate(seed_record)
            store.save_candidate(faster_record)
            store.save_state(_state("finalized", seed.candidate_id))

            clock = _FakeClock(0.0)
            recorder = PeriodicIncumbentRecorder(
                store, interval_seconds=300.0, clock=clock
            )
            recorder.start(started_at=clock())
            snapshot = recorder.record_now("run-completed")
            recorder.stop()

        self.assertEqual(snapshot["selection_basis"], "final-measurement")
        self.assertEqual(snapshot["incumbent"]["candidate_id"], seed.candidate_id)
        self.assertEqual(snapshot["incumbent_latency_ms"], 2.0)
        self.assertEqual(snapshot["speedup_over_seed"], 1.0)

    def test_zero_interval_disables_all_output(self) -> None:
        task = _task()
        seed = Candidate.seed(task, SOURCE % 0, "kernel.py")
        with tempfile.TemporaryDirectory(prefix="incumbent-disabled-") as directory:
            store = ArtifactStore(Path(directory))
            store.initialize(task, seed)
            recorder = PeriodicIncumbentRecorder(store, interval_seconds=0)
            recorder.start()
            self.assertIsNone(recorder.record_now("manual"))
            recorder.stop()
            self.assertFalse(recorder.jsonl_path.exists())
            self.assertFalse(recorder.csv_path.exists())


class _FakeClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="incumbent-test",
        description="Test periodic incumbent records.",
        reference="Return the input.",
        entrypoint="kernel",
    )


def _record(
    candidate: Candidate,
    *,
    measured: float,
    predicted: float,
    profile: bool = False,
) -> CandidateRecord:
    return CandidateRecord(
        candidate=candidate,
        state="measured-beam",
        model=ModelEvaluation(
            valid=True,
            predicted_latency_ms=predicted,
            calibrated_latency_ms=predicted * 1.1,
            bottleneck="tensor-core",
            confidence="high",
            metrics={
                "ddr_util": 0.2,
                "l2_hit_rate": 0.6,
                "tensor_util": 0.8,
                "registers_per_thread": 64,
                "shared_memory_per_block": 32768,
                "tiles_per_sm": 3,
            },
        ),
        measurement=Measurement(
            correct=True,
            latency_ms=measured,
            samples_ms=[measured],
            metrics={"case_count": 2},
        ),
        profile=(
            ProfileEvaluation(
                bottleneck="dram-bandwidth",
                metrics={"dram_utilization": 71.0},
                report_path="profile.ncu-rep",
            )
            if profile
            else None
        ),
    )


def _state(phase: str, best_candidate_id: str):
    return {
        "schema_version": 1,
        "phase": phase,
        "completed_round": 0,
        "active_round": None,
        "beam": [best_candidate_id],
        "best_candidate_id": best_candidate_id,
        "profile_calls": 1,
        "generator_calls": 2,
        "planner_calls": 1,
        "api_request_attempts": 3,
        "generator_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
        "experiment": {
            "metadata_compression_policy": "key-metrics-v1",
        },
    }


if __name__ == "__main__":
    unittest.main()
