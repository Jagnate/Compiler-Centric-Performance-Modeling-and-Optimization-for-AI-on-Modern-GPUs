from __future__ import annotations

import math

from .models import HardwareConfig, OccupancyLimits


WARP_SIZE = 32


def estimate_occupancy(
    *,
    hw: HardwareConfig,
    threads_per_cta: int,
    warps_per_cta: int,
    dynamic_shared_bytes_per_cta: int,
    registers_per_thread: int | None,
) -> OccupancyLimits:
    """Compute resource-limited CTA residency for one SM."""
    if threads_per_cta <= 0 or warps_per_cta <= 0:
        raise ValueError("CTA thread and warp counts must be positive.")

    by_shared_memory = None
    if dynamic_shared_bytes_per_cta > 0:
        by_shared_memory = hw.smem_per_sm_bytes // dynamic_shared_bytes_per_cta

    by_threads = hw.max_threads_per_sm // threads_per_cta

    allocated_registers_per_cta = None
    by_registers = None
    if registers_per_thread is not None:
        if registers_per_thread <= 0:
            raise ValueError("registers_per_thread must be positive when provided.")
        registers_per_warp = registers_per_thread * WARP_SIZE
        allocation_unit = hw.register_allocation_unit_regs
        if allocation_unit <= 0:
            raise ValueError("register_allocation_unit_regs must be positive.")
        allocated_registers_per_warp = (
            math.ceil(registers_per_warp / allocation_unit) * allocation_unit
        )
        allocated_registers_per_cta = allocated_registers_per_warp * warps_per_cta
        by_registers = hw.registers_per_sm // allocated_registers_per_cta

    limits = {
        "threads": by_threads,
        "architecture": hw.cta_limit_per_sm,
    }
    if by_shared_memory is not None:
        limits["shared_memory"] = by_shared_memory
    if by_registers is not None:
        limits["registers"] = by_registers

    resident = min(limits.values())
    if resident < 1:
        limiting = [name for name, value in limits.items() if value == resident]
        raise ValueError(
            "Kernel resource usage exceeds one-SM capacity; "
            f"zero resident CTAs from {', '.join(limiting)}."
        )

    limiting_resources = [name for name, value in limits.items() if value == resident]
    return OccupancyLimits(
        resident_ctas_per_sm=resident,
        by_shared_memory=by_shared_memory,
        by_threads=by_threads,
        by_registers=by_registers,
        by_architecture=hw.cta_limit_per_sm,
        allocated_registers_per_cta=allocated_registers_per_cta,
        limiting_resources=limiting_resources,
    )
