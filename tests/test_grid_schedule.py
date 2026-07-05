from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
import unittest

from perf_model.hardware import load_hardware_config
from perf_model.tile_centric_gemm import simulate, take_spread_first_batches
from perf_model.tir_parser import infer_gemm_model, parse_facts


ROOT = Path(__file__).resolve().parents[1]


def current_gemm():
    tir_path = ROOT / "tir_dump/gemm/20_device_tir.py"
    hardware_path = ROOT / "hardware_configs/hardware_sm86_default.json"
    text = tir_path.read_text(encoding="utf-8")
    hardware = load_hardware_config(hardware_path)
    facts = parse_facts(text)
    model = infer_gemm_model(
        text,
        facts,
        hardware.tensor_peak_tflops,
        hardware.ddr_bandwidth_gbs,
    )
    return model, hardware


class GridScheduleTests(unittest.TestCase):
    def test_tail_residency_spreads_before_filling_second_slot(self) -> None:
        pending = deque((index, 0, index) for index in range(100))
        batches = take_spread_first_batches(
            pending,
            list(range(82)),
            resident_capacity=2,
        )
        residency = Counter(len(batch) for batch in batches.values())

        self.assertEqual(residency, Counter({1: 64, 2: 18}))
        self.assertFalse(pending)

    def test_current_small_grid_uses_one_cta_on_64_sms(self) -> None:
        model, hardware = current_gemm()
        result = simulate(model, hardware, mode="detailed")
        initial_events = [event for event in result.schedule_events if event.start_us == 0.0]

        self.assertEqual(result.schedule_model, "dynamic_rolling_sm_batches_v1")
        self.assertEqual(len(initial_events), 64)
        self.assertTrue(all(event.resident_ctas_on_sm == 1 for event in initial_events))
        self.assertEqual(sorted(event.sm_id for event in initial_events), list(range(64)))
        self.assertTrue(all(event.dynamic_rate_updates > 4 for event in initial_events))
        self.assertAlmostEqual(
            result.aggregate_utilization.sm_activity,
            64 / hardware.num_sms,
        )
        self.assertAlmostEqual(
            result.aggregate_utilization.theoretical_occupancy,
            result.resident_ctas_per_sm
            * model.warps_per_cta
            / (hardware.max_threads_per_sm / 32),
        )
        self.assertAlmostEqual(
            result.aggregate_utilization.achieved_occupancy,
            64
            * model.warps_per_cta
            / (hardware.num_sms * (hardware.max_threads_per_sm / 32)),
        )
        self.assertTrue(
            0.0 < result.aggregate_utilization.tensor_core_utilization <= 1.0
        )

    def test_fast_mode_uses_cohorts_without_schedule_events(self) -> None:
        model, hardware = current_gemm()
        result = simulate(model, hardware, mode="fast")

        self.assertEqual(result.schedule_model, "analytical_wave_cohorts_v1")
        self.assertFalse(result.schedule_events)
        self.assertEqual(len(result.fast_cohorts), 1)
        self.assertEqual(result.fast_cohorts[0].sm_count, 64)
        self.assertEqual(result.fast_cohorts[0].ctas_per_sm, 1)
        self.assertAlmostEqual(
            result.aggregate_utilization.sm_activity,
            64 / hardware.num_sms,
        )

    def test_next_ctas_start_at_the_earliest_available_completion(self) -> None:
        model, hardware = current_gemm()
        grid_y = 32
        m = model.block_M * grid_y
        num_ctas = model.grid_x * grid_y
        larger_model = replace(
            model,
            M=m,
            grid_y=grid_y,
            num_ctas=num_ctas,
            flops=2 * m * model.N * model.K,
        )

        result = simulate(larger_model, hardware, mode="detailed")
        initial_events = [event for event in result.schedule_events if event.start_us == 0.0]
        later_events = [event for event in result.schedule_events if event.start_us > 0.0]
        first_initial_finish = min(event.end_us for event in initial_events)

        self.assertTrue(later_events)
        self.assertAlmostEqual(
            min(event.start_us for event in later_events),
            first_initial_finish,
        )
        scheduled_ctas = sorted(
            cta_index
            for event in result.schedule_events
            for cta_index in event.cta_indices
        )
        self.assertEqual(scheduled_ctas, list(range(num_ctas)))


if __name__ == "__main__":
    unittest.main()
