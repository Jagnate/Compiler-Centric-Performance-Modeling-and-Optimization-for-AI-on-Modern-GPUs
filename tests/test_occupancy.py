from __future__ import annotations

import unittest
from pathlib import Path

from perf_model.models import HardwareConfig
from perf_model.occupancy import estimate_occupancy
from perf_model.tir_parser import parse_facts


ROOT = Path(__file__).resolve().parents[1]


def sm86_hardware() -> HardwareConfig:
    return HardwareConfig(
        num_sms=82,
        smem_per_sm_bytes=102400,
        max_threads_per_sm=1536,
        cta_limit_per_sm=16,
        registers_per_sm=65536,
        register_allocation_unit_regs=256,
        l2_capacity_bytes=6291456,
        tensor_peak_tflops=119.48,
        l2_bandwidth_gbs=2149.89,
        ddr_bandwidth_gbs=936.096,
        smem_bandwidth_gbs=14934.90,
        tensor_efficiency=0.75,
        l2_efficiency=0.75,
        ddr_efficiency=0.75,
        smem_efficiency=0.70,
    )


class OccupancyTests(unittest.TestCase):
    def test_gemm_register_and_shared_memory_limits_match_ncu(self) -> None:
        result = estimate_occupancy(
            hw=sm86_hardware(),
            threads_per_cta=128,
            warps_per_cta=4,
            dynamic_shared_bytes_per_cta=49152,
            registers_per_thread=217,
        )

        self.assertEqual(result.allocated_registers_per_cta, 28672)
        self.assertEqual(result.by_shared_memory, 2)
        self.assertEqual(result.by_threads, 12)
        self.assertEqual(result.by_registers, 2)
        self.assertEqual(result.resident_ctas_per_sm, 2)
        self.assertEqual(result.limiting_resources, ["shared_memory", "registers"])

    def test_unknown_register_usage_is_explicitly_unmodeled(self) -> None:
        result = estimate_occupancy(
            hw=sm86_hardware(),
            threads_per_cta=128,
            warps_per_cta=4,
            dynamic_shared_bytes_per_cta=49152,
            registers_per_thread=None,
        )

        self.assertIsNone(result.by_registers)
        self.assertIsNone(result.allocated_registers_per_cta)
        self.assertEqual(result.resident_ctas_per_sm, 2)
        self.assertEqual(result.limiting_resources, ["shared_memory"])

    def test_tir_static_register_estimate_is_used_by_default(self) -> None:
        text = (ROOT / "tir_dump/gemm/20_device_tir.py").read_text(encoding="utf-8")
        facts = parse_facts(text)

        self.assertEqual(facts.registers_per_thread, 171)
        self.assertEqual(facts.registers_per_thread_source, "tir_static_estimate")
        self.assertEqual(facts.register_estimate.explicit_local_registers, 161)
        self.assertEqual(
            facts.register_estimate.local_buffer_registers,
            {
                "C_local": 128,
                "C_local_cast": 1,
                "A_local": 16,
                "B_local": 16,
            },
        )

        occupancy = estimate_occupancy(
            hw=sm86_hardware(),
            threads_per_cta=facts.threads_per_cta,
            warps_per_cta=facts.warps_per_cta,
            dynamic_shared_bytes_per_cta=facts.dynamic_shared_bytes_per_cta,
            registers_per_thread=facts.registers_per_thread,
        )
        self.assertEqual(occupancy.by_registers, 2)

    def test_compiled_override_is_optional_validation_input(self) -> None:
        text = (ROOT / "tir_dump/gemm/20_device_tir.py").read_text(encoding="utf-8")
        facts = parse_facts(text, registers_per_thread_override=217)

        self.assertEqual(facts.registers_per_thread, 217)
        self.assertEqual(facts.registers_per_thread_source, "compiled_override")
        self.assertEqual(facts.register_estimate.estimated_registers_per_thread, 171)

    def test_impossible_resource_usage_fails_instead_of_forcing_one_cta(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero resident CTAs"):
            estimate_occupancy(
                hw=sm86_hardware(),
                threads_per_cta=1024,
                warps_per_cta=32,
                dynamic_shared_bytes_per_cta=0,
                registers_per_thread=255,
            )


if __name__ == "__main__":
    unittest.main()
