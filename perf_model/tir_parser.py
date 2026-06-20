from __future__ import annotations

import math
import re

from .models import GemmModel, TirFacts


# The dumped TIR is a Python script, but the structural metadata we need appears as
# stable text fragments in TileLang output. These helpers keep the static extractor
# small while making each inference pattern explicit and easy to replace later.
def search_int(pattern: str, text: str, default: int | None = None) -> int | None:
    match = re.search(pattern, text)
    if not match:
        return default
    return int(match.group(1))


def search_str(pattern: str, text: str, default: str | None = None) -> str | None:
    match = re.search(pattern, text)
    if not match:
        return default
    return match.group(1)


def dtype_nbytes(dtype: str) -> int:
    if dtype in {"float16", "bfloat16", "int16", "uint16"}:
        return 2
    if dtype in {"float32", "int32", "uint32"}:
        return 4
    if dtype in {"float64", "int64", "uint64"}:
        return 8
    if dtype in {"int8", "uint8", "bool"}:
        return 1
    raise ValueError(f"Unknown dtype byte width for {dtype!r}")


def parse_facts(text: str) -> TirFacts:
    # Facts are direct properties of the generated kernel: launch shape, shared
    # memory allocation, pipeline markers, and textual PTX intrinsic counts.
    kernel_name = search_str(r"def\s+([A-Za-z_]\w*)\(", text, "unknown_kernel")
    arch = search_str(r'"arch":\s*"([^"]+)"', text)
    grid_x = search_int(r'"blockIdx\.x":\s*(\d+)', text, 1)
    grid_y = search_int(r'"blockIdx\.y":\s*(\d+)', text, 1)
    threads = search_int(r'"threadIdx\.x":\s*(\d+)', text, 1)
    shared = search_int(r'"dyn_shared_memory_buf":\s*(\d+)', text, 0)
    stages = search_int(r'"tl\.pipeline_mvb_num_stages",\s*(\d+)', text)
    c_elements = search_int(r"C_1\s*=\s*T\.decl_buffer\(\((\d+),\),\s*\"float16\"", text)

    assert kernel_name is not None
    assert grid_x is not None
    assert grid_y is not None
    assert threads is not None
    assert shared is not None

    return TirFacts(
        kernel_name=kernel_name,
        arch=arch,
        grid_x=grid_x,
        grid_y=grid_y,
        threads_per_cta=threads,
        warps_per_cta=math.ceil(threads / 32),
        num_ctas=grid_x * grid_y,
        dynamic_shared_bytes_per_cta=shared,
        pipeline_stages=stages,
        c_elements=c_elements,
        textual_ptx_cp_async=text.count("T.ptx_cp_async("),
        textual_ptx_ldmatrix=text.count("T.ptx_ldmatrix("),
        textual_ptx_mma=text.count("T.ptx_mma("),
        textual_ptx_commit_group=text.count("T.ptx_commit_group("),
        textual_ptx_wait_group=text.count("T.ptx_wait_group("),
        textual_tvm_storage_sync=text.count("T.tvm_storage_sync("),
    )


def infer_gemm_model(
    text: str,
    facts: TirFacts,
    peak_tflops: float | None,
    bandwidth_gbs: float | None,
) -> GemmModel:
    notes: list[str] = []

    # Recover the GEMM problem and CTA tile shape from address expressions in TIR.
    # This is conservative: if a pattern is missing, we fail loudly instead of
    # silently producing a misleading performance point.
    dtype_a = search_str(r"A:\s*T\.handle\(\"([^\"]+)\"", text, "float16")
    dtype_b = search_str(r"B:\s*T\.handle\(\"([^\"]+)\"", text, "float16")
    dtype_c = search_str(r"C:\s*T\.handle\(\"([^\"]+)\"", text, "float16")
    assert dtype_a is not None and dtype_b is not None and dtype_c is not None

    bytes_a = dtype_nbytes(dtype_a)
    bytes_b = dtype_nbytes(dtype_b)
    bytes_c = dtype_nbytes(dtype_c)

    block_n = search_int(r"\+\s*bx\s*\*\s*(\d+)", text)
    if block_n is None:
        raise ValueError("Could not infer block_N from bx * <stride> in TIR.")

    n = block_n * facts.grid_x
    if facts.c_elements is None:
        raise ValueError("Could not infer C element count from C_1 decl_buffer.")
    if facts.c_elements % n != 0:
        raise ValueError(f"C element count {facts.c_elements} is not divisible by inferred N {n}.")

    m = facts.c_elements // n
    if m % facts.grid_y != 0:
        raise ValueError(f"Inferred M {m} is not divisible by grid_y {facts.grid_y}.")
    block_m = m // facts.grid_y

    block_k = search_int(r"\+\s*k\s*\*\s*(\d+)\s*\+\s*thread_binding\s*%\s*4", text)
    if block_k is None:
        b_k_stride = search_int(r"B,\s*k\s*\*\s*(\d+)\s*\+", text)
        block_k = b_k_stride // n if b_k_stride is not None else None
    if block_k is None:
        raise ValueError("Could not infer block_K from k * <stride> in TIR.")

    serial_k = search_int(r"for\s+k\s+in\s+T\.serial\((\d+),", text)
    if serial_k is None:
        raise ValueError("Could not infer pipelined serial K loop count.")

    k_tiles = serial_k + 2
    notes.append("K tile count inferred as T.serial trip count + 2 pipeline prologue/epilogue tiles.")

    k = block_k * k_tiles
    flops = 2 * m * n * k

    read_a_per_cta = block_m * k * bytes_a
    read_b_per_cta = k * block_n * bytes_b
    write_c_per_cta = block_m * block_n * bytes_c
    read_tiled = facts.num_ctas * (read_a_per_cta + read_b_per_cta)
    write_tiled = facts.num_ctas * write_c_per_cta
    bytes_tiled = read_tiled + write_tiled

    bytes_ideal = m * k * bytes_a + k * n * bytes_b + m * n * bytes_c
    ai_tiled = flops / bytes_tiled
    ai_ideal = flops / bytes_ideal

    mma_shape = search_str(r'T\.ptx_mma\("[^"]+",\s*"([^"]+)"', text)
    mma_flops = None
    mma_total = None
    mma_per_cta = None
    if mma_shape:
        # PTX MMA shape lets us estimate instruction counts for sanity checking.
        mma_match = re.match(r"m(\d+)n(\d+)k(\d+)", mma_shape)
        if mma_match:
            mma_m, mma_n, mma_k = (int(x) for x in mma_match.groups())
            mma_flops = 2 * mma_m * mma_n * mma_k
            mma_total = flops // mma_flops
            mma_per_cta = mma_total // facts.num_ctas

    memory_roof = None
    bound = None
    if bandwidth_gbs is not None:
        memory_roof = ai_tiled * bandwidth_gbs / 1000.0
    if peak_tflops is not None and memory_roof is not None:
        bound = min(peak_tflops, memory_roof)
    elif peak_tflops is not None:
        bound = peak_tflops
    elif memory_roof is not None:
        bound = memory_roof

    return GemmModel(
        kernel_name=facts.kernel_name,
        arch=facts.arch,
        M=m,
        N=n,
        K=k,
        block_M=block_m,
        block_N=block_n,
        block_K=block_k,
        k_tiles=k_tiles,
        grid_x=facts.grid_x,
        grid_y=facts.grid_y,
        num_ctas=facts.num_ctas,
        threads_per_cta=facts.threads_per_cta,
        warps_per_cta=facts.warps_per_cta,
        dynamic_shared_bytes_per_cta=facts.dynamic_shared_bytes_per_cta,
        pipeline_stages=facts.pipeline_stages,
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        dtype_c=dtype_c,
        bytes_a_per_element=bytes_a,
        bytes_b_per_element=bytes_b,
        bytes_c_per_element=bytes_c,
        flops=flops,
        dram_read_bytes_tiled=read_tiled,
        dram_write_bytes_tiled=write_tiled,
        dram_bytes_tiled=bytes_tiled,
        dram_bytes_ideal=bytes_ideal,
        arithmetic_intensity_tiled=ai_tiled,
        arithmetic_intensity_ideal=ai_ideal,
        mma_shape=mma_shape,
        mma_flops_per_instruction=mma_flops,
        mma_instructions_total_est=mma_total,
        mma_instructions_per_cta_est=mma_per_cta,
        roofline_bound_tflops=bound,
        memory_roof_tflops=memory_roof,
        compute_roof_tflops=peak_tflops,
        notes=notes,
    )
