from __future__ import annotations

import math
import re

from .models import RegisterEstimate


REGISTER_BITS = 32
POINTER_BITS = 64


def dtype_bits(dtype: str) -> int:
    match = re.fullmatch(r"(?:float|bfloat|int|uint)(\d+)", dtype)
    if match:
        return int(match.group(1))
    if dtype == "bool":
        return 1
    raise ValueError(f"Unsupported local-buffer dtype for register estimation: {dtype!r}")


def estimate_registers_from_tir(text: str) -> RegisterEstimate:
    """Estimate per-thread register pressure from lowered GEMM TIR.

    Explicit thread-local buffers are converted to 32-bit register equivalents.
    Repeated allocations with the same name represent reused lexical storage, so
    only the largest instance is retained. Pointer, launch-index, and serial-loop
    state add the persistent scalar registers visible in TIR. Backend liveness
    optimization and address temporaries are intentionally outside this estimate.
    """
    local_buffers: dict[str, int] = {}
    allocation_pattern = re.compile(
        r'^\s*([A-Za-z_]\w*)\s*=\s*T\.allocate\(\[(\d+)\],\s*"([^"]+)",\s*'
        r'"(local(?:\.[^"]*)?)"\)',
        re.MULTILINE,
    )
    for name, element_count, dtype, _scope in allocation_pattern.findall(text):
        registers = math.ceil(int(element_count) * dtype_bits(dtype) / REGISTER_BITS)
        local_buffers[name] = max(local_buffers.get(name, 0), registers)

    explicit_local_registers = sum(local_buffers.values())

    global_pointer_count = len(
        re.findall(r'[A-Za-z_]\w*:\s*T\.handle\("[^"]+",\s*"global"\)', text)
    )
    kernel_pointer_registers = global_pointer_count * (POINTER_BITS // REGISTER_BITS)

    launch_indices = {
        name
        for name, extent in re.findall(
            r'([A-Za-z_]\w*)\s*=\s*T\.launch_thread\("[^"]+",\s*(\d+)\)',
            text,
        )
        if int(extent) > 1
    }
    launch_index_registers = len(launch_indices)

    serial_loops = set(
        re.findall(r"for\s+([A-Za-z_]\w*)\s+in\s+T\.serial\(", text)
    )
    serial_loop_registers = len(serial_loops)

    estimated = (
        explicit_local_registers
        + kernel_pointer_registers
        + launch_index_registers
        + serial_loop_registers
    )
    if estimated <= 0:
        raise ValueError("Could not infer any register-resident state from TIR.")

    return RegisterEstimate(
        explicit_local_registers=explicit_local_registers,
        kernel_pointer_registers=kernel_pointer_registers,
        launch_index_registers=launch_index_registers,
        serial_loop_registers=serial_loop_registers,
        estimated_registers_per_thread=estimated,
        local_buffer_registers=local_buffers,
        method="tir_static_estimate",
        confidence="heuristic",
    )
