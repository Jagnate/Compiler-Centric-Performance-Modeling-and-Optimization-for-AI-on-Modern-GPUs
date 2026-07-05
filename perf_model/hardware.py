from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import HardwareConfig


# Editable placeholder hardware profile. Treat these as modeling assumptions, not
# calibrated measurements; pass --hardware-config to swap in measured platform data.
DEFAULT_HARDWARE = HardwareConfig(
    num_sms=84,
    smem_per_sm_bytes=100 * 1024,
    max_threads_per_sm=1536,
    cta_limit_per_sm=16,
    registers_per_sm=65536,
    register_allocation_unit_regs=256,
    l2_capacity_bytes=6 * 1024 * 1024,
    tensor_peak_tflops=125.0,
    l2_bandwidth_gbs=3000.0,
    ddr_bandwidth_gbs=760.0,
    smem_bandwidth_gbs=12000.0,
    tensor_efficiency=0.75,
    l2_efficiency=0.75,
    ddr_efficiency=0.75,
    smem_efficiency=0.70,
)


def load_hardware_config(path: Path | None = None) -> HardwareConfig:
    # A JSON hardware file has the same keys as HardwareConfig, which makes sweeps
    # over peak compute, bandwidth, cache size, and efficiency factors straightforward.
    if path is None:
        return DEFAULT_HARDWARE

    data = json.loads(path.read_text(encoding="utf-8"))
    return HardwareConfig(**data)


def write_default_hardware_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(DEFAULT_HARDWARE), indent=2), encoding="utf-8")
