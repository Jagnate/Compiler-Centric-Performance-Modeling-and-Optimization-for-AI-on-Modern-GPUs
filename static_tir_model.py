#!/usr/bin/env python3
"""Static performance model extractor for a lowered TileLang GEMM TIR file.

This first version is intentionally small and transparent. It targets the
lowered GEMM TIR shape produced in tir_dump/gemm/20_device_tir.py and emits a
JSON/CSV record plus an optional roofline plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_TIR = Path("tir_dump/gemm/20_device_tir.py")


@dataclass
class TirFacts:
    kernel_name: str
    arch: str | None
    grid_x: int
    grid_y: int
    threads_per_cta: int
    warps_per_cta: int
    num_ctas: int
    dynamic_shared_bytes_per_cta: int
    pipeline_stages: int | None
    c_elements: int | None
    textual_ptx_cp_async: int
    textual_ptx_ldmatrix: int
    textual_ptx_mma: int
    textual_ptx_commit_group: int
    textual_ptx_wait_group: int
    textual_tvm_storage_sync: int


@dataclass
class GemmModel:
    kernel_name: str
    arch: str | None
    M: int
    N: int
    K: int
    block_M: int
    block_N: int
    block_K: int
    k_tiles: int
    grid_x: int
    grid_y: int
    num_ctas: int
    threads_per_cta: int
    warps_per_cta: int
    dynamic_shared_bytes_per_cta: int
    pipeline_stages: int | None
    dtype_a: str
    dtype_b: str
    dtype_c: str
    bytes_a_per_element: int
    bytes_b_per_element: int
    bytes_c_per_element: int
    flops: int
    dram_read_bytes_tiled: int
    dram_write_bytes_tiled: int
    dram_bytes_tiled: int
    dram_bytes_ideal: int
    arithmetic_intensity_tiled: float
    arithmetic_intensity_ideal: float
    mma_shape: str | None
    mma_flops_per_instruction: int | None
    mma_instructions_total_est: int | None
    mma_instructions_per_cta_est: int | None
    roofline_bound_tflops: float | None
    memory_roof_tflops: float | None
    compute_roof_tflops: float | None
    notes: list[str]


@dataclass
class PipelineLatencyModel:
    num_sms: int
    smem_per_sm_bytes: int
    max_threads_per_sm: int
    cta_limit_per_sm: int
    resident_ctas_per_sm: int
    active_ctas_per_wave: int
    waves: int
    tensor_peak_tflops: float
    load_bandwidth_gbs: float
    store_bandwidth_gbs: float
    tensor_efficiency: float
    load_efficiency: float
    store_efficiency: float
    effective_tensor_peak_tflops: float
    effective_load_bandwidth_gbs: float
    effective_store_bandwidth_gbs: float
    prologue_k_tiles: int
    steady_k_tiles: int
    epilogue_k_tiles: int
    load_bytes_per_k_tile_per_cta: int
    compute_flops_per_k_tile_per_cta: int
    store_bytes_per_cta: int
    load_us_per_k_tile: float
    compute_us_per_k_tile: float
    store_us_per_cta: float
    estimated_cta_us: float
    estimated_kernel_us: float
    estimated_kernel_ms: float
    estimated_tflops: float
    steady_state_bottleneck: str
    formula: str


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

    # In this lowered TileLang pipeline, T.serial(30) is the steady-state loop,
    # with 2 prologue tiles and 2 explicit epilogue compute blocks. Therefore
    # total K tiles are serial_k + 2 for this GEMM.
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


def estimate_pipeline_latency(
    model: GemmModel,
    num_sms: int,
    smem_per_sm_bytes: int,
    max_threads_per_sm: int,
    cta_limit_per_sm: int,
    tensor_peak_tflops: float,
    load_bandwidth_gbs: float,
    store_bandwidth_gbs: float,
    tensor_efficiency: float,
    load_efficiency: float,
    store_efficiency: float,
) -> PipelineLatencyModel:
    resident_by_smem = smem_per_sm_bytes // model.dynamic_shared_bytes_per_cta
    resident_by_threads = max_threads_per_sm // model.threads_per_cta
    resident_ctas = max(1, min(resident_by_smem, resident_by_threads, cta_limit_per_sm))
    active_ctas_per_wave = min(model.num_ctas, num_sms * resident_ctas)
    waves = math.ceil(model.num_ctas / active_ctas_per_wave)

    effective_tensor_peak = tensor_peak_tflops * tensor_efficiency
    effective_load_bandwidth = load_bandwidth_gbs * load_efficiency
    effective_store_bandwidth = store_bandwidth_gbs * store_efficiency

    per_cta_tensor_flops_per_s = effective_tensor_peak * 1e12 / active_ctas_per_wave
    per_cta_load_bytes_per_s = effective_load_bandwidth * 1e9 / active_ctas_per_wave
    per_cta_store_bytes_per_s = effective_store_bandwidth * 1e9 / active_ctas_per_wave

    load_bytes_per_k = (
        model.block_M * model.block_K * model.bytes_a_per_element
        + model.block_K * model.block_N * model.bytes_b_per_element
    )
    compute_flops_per_k = 2 * model.block_M * model.block_N * model.block_K
    store_bytes = model.block_M * model.block_N * model.bytes_c_per_element

    load_us = load_bytes_per_k / per_cta_load_bytes_per_s * 1e6
    compute_us = compute_flops_per_k / per_cta_tensor_flops_per_s * 1e6
    store_us = store_bytes / per_cta_store_bytes_per_s * 1e6

    prologue_tiles = min(max((model.pipeline_stages or 1) - 1, 0), model.k_tiles)
    steady_tiles = max(model.k_tiles - prologue_tiles, 0)
    epilogue_tiles = prologue_tiles

    estimated_cta_us = (
        prologue_tiles * load_us
        + steady_tiles * max(load_us, compute_us)
        + epilogue_tiles * compute_us
        + store_us
    )
    estimated_kernel_us = waves * estimated_cta_us
    estimated_kernel_ms = estimated_kernel_us / 1000.0
    estimated_tflops = model.flops / (estimated_kernel_ms * 1e9)

    if compute_us > load_us * 1.05:
        bottleneck = "compute"
    elif load_us > compute_us * 1.05:
        bottleneck = "load"
    else:
        bottleneck = "balanced"

    return PipelineLatencyModel(
        num_sms=num_sms,
        smem_per_sm_bytes=smem_per_sm_bytes,
        max_threads_per_sm=max_threads_per_sm,
        cta_limit_per_sm=cta_limit_per_sm,
        resident_ctas_per_sm=resident_ctas,
        active_ctas_per_wave=active_ctas_per_wave,
        waves=waves,
        tensor_peak_tflops=tensor_peak_tflops,
        load_bandwidth_gbs=load_bandwidth_gbs,
        store_bandwidth_gbs=store_bandwidth_gbs,
        tensor_efficiency=tensor_efficiency,
        load_efficiency=load_efficiency,
        store_efficiency=store_efficiency,
        effective_tensor_peak_tflops=effective_tensor_peak,
        effective_load_bandwidth_gbs=effective_load_bandwidth,
        effective_store_bandwidth_gbs=effective_store_bandwidth,
        prologue_k_tiles=prologue_tiles,
        steady_k_tiles=steady_tiles,
        epilogue_k_tiles=epilogue_tiles,
        load_bytes_per_k_tile_per_cta=load_bytes_per_k,
        compute_flops_per_k_tile_per_cta=compute_flops_per_k,
        store_bytes_per_cta=store_bytes,
        load_us_per_k_tile=load_us,
        compute_us_per_k_tile=compute_us,
        store_us_per_cta=store_us,
        estimated_cta_us=estimated_cta_us,
        estimated_kernel_us=estimated_kernel_us,
        estimated_kernel_ms=estimated_kernel_ms,
        estimated_tflops=estimated_tflops,
        steady_state_bottleneck=bottleneck,
        formula="P*load + (K_tiles-P)*max(load, compute) + P*compute + store",
    )


def write_csv(path: Path, record: dict[str, Any]) -> None:
    flat = {}
    for key, value in record.items():
        flat[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)


def maybe_write_roofline(
    path: Path,
    model: GemmModel,
    peak_tflops: float | None,
    bandwidth_gbs: float | None,
    measured_ms: float | None,
    measured_traffic_model: str,
    pipeline_model: PipelineLatencyModel | None,
    estimated_traffic_model: str,
) -> bool:
    if peak_tflops is None or bandwidth_gbs is None:
        return False

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        svg_path = path.with_suffix(".svg")
        write_roofline_svg(
            svg_path,
            model,
            peak_tflops,
            bandwidth_gbs,
            measured_ms,
            measured_traffic_model,
            pipeline_model,
            estimated_traffic_model,
        )
        print(f"matplotlib/numpy not available; wrote SVG roofline fallback to {svg_path}.")
        return False

    x = np.logspace(-1, 4, 400)
    y = np.minimum(peak_tflops, x * bandwidth_gbs / 1000.0)
    tiled_ai = model.arithmetic_intensity_tiled
    ideal_ai = model.arithmetic_intensity_ideal
    tiled_bound = min(peak_tflops, tiled_ai * bandwidth_gbs / 1000.0)
    ideal_bound = min(peak_tflops, ideal_ai * bandwidth_gbs / 1000.0)

    plt.figure(figsize=(7, 5))
    plt.loglog(x, y, label="Roofline bound")
    plt.axhline(peak_tflops, linestyle="--", alpha=0.5, label="Compute peak")
    plt.axvline(tiled_ai, linestyle="--", alpha=0.25)
    plt.axvline(ideal_ai, linestyle="--", alpha=0.25)
    plt.scatter(
        [tiled_ai],
        [tiled_bound],
        s=84,
        facecolors="none",
        edgecolors="#f97316",
        linewidths=2,
        label="Static bound (CTA global traffic)",
    )
    plt.scatter(
        [ideal_ai],
        [ideal_bound],
        s=84,
        facecolors="none",
        edgecolors="#16a34a",
        linewidths=2,
        label="Static bound (ideal HBM traffic)",
    )
    if measured_ms is not None:
        measured_ai = tiled_ai if measured_traffic_model == "tiled" else ideal_ai
        measured_tflops = model.flops / (measured_ms * 1e9)
        plt.scatter([measured_ai], [measured_tflops], s=86, color="#dc2626", label="Measured kernel")
    if pipeline_model is not None:
        estimated_ai = tiled_ai if estimated_traffic_model == "tiled" else ideal_ai
        plt.scatter(
            [estimated_ai],
            [pipeline_model.estimated_tflops],
            s=90,
            color="#7c3aed",
            marker="D",
            label="Estimated pipeline model",
        )
    plt.xlabel("Arithmetic intensity (FLOP/byte)")
    plt.ylabel("Performance (TFLOP/s)")
    plt.title("Static Roofline Model")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return True


def write_roofline_svg(
    path: Path,
    model: GemmModel,
    peak_tflops: float,
    bandwidth_gbs: float,
    measured_ms: float | None,
    measured_traffic_model: str,
    pipeline_model: PipelineLatencyModel | None,
    estimated_traffic_model: str,
) -> None:
    width = 760
    height = 520
    left = 82
    right = 24
    top = 32
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min, x_max = 0.1, 10000.0
    y_min = 0.1
    y_max = max(peak_tflops * 1.5, 10.0)

    def lx(v: float) -> float:
        return math.log10(v)

    def sx(v: float) -> float:
        return left + (lx(v) - lx(x_min)) / (lx(x_max) - lx(x_min)) * plot_w

    def sy(v: float) -> float:
        return top + (lx(y_max) - lx(v)) / (lx(y_max) - lx(y_min)) * plot_h

    ridge_ai = peak_tflops * 1000.0 / bandwidth_gbs
    tiled_ai = model.arithmetic_intensity_tiled
    ideal_ai = model.arithmetic_intensity_ideal
    tiled_bound = min(peak_tflops, tiled_ai * bandwidth_gbs / 1000.0)
    ideal_bound = min(peak_tflops, ideal_ai * bandwidth_gbs / 1000.0)

    mem_x0 = x_min
    mem_y0 = max(y_min, mem_x0 * bandwidth_gbs / 1000.0)
    mem_x1 = min(ridge_ai, x_max)
    mem_y1 = min(peak_tflops, mem_x1 * bandwidth_gbs / 1000.0)
    comp_x0 = max(ridge_ai, x_min)
    comp_x1 = x_max

    def line(x1: float, y1: float, x2: float, y2: float, color: str, dash: str = "") -> str:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="2.4"{dash_attr}/>'
        )

    grid_lines = []
    labels = []
    for exp in range(-1, 5):
        v = 10.0**exp
        x = sx(v)
        grid_lines.append(line(x, top, x, top + plot_h, "#dddddd", "3 5"))
        labels.append(f'<text x="{x:.1f}" y="{height - 42}" text-anchor="middle" font-size="12">{v:g}</text>')
    y_exp_min = math.floor(lx(y_min))
    y_exp_max = math.ceil(lx(y_max))
    for exp in range(y_exp_min, y_exp_max + 1):
        v = 10.0**exp
        if y_min <= v <= y_max:
            y = sy(v)
            grid_lines.append(line(left, y, left + plot_w, y, "#dddddd", "3 5"))
            labels.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{v:g}</text>')

    roof = [
        line(sx(mem_x0), sy(mem_y0), sx(mem_x1), sy(mem_y1), "#2563eb"),
        line(sx(comp_x0), sy(peak_tflops), sx(comp_x1), sy(peak_tflops), "#2563eb"),
        line(sx(tiled_ai), top, sx(tiled_ai), top + plot_h, "#f97316", "4 5"),
        line(sx(ideal_ai), top, sx(ideal_ai), top + plot_h, "#16a34a", "4 5"),
    ]
    measured_svg = ""
    measured_text = "measured point: pass --measured-ms to draw actual performance"
    if measured_ms is not None:
        measured_ai = tiled_ai if measured_traffic_model == "tiled" else ideal_ai
        measured_tflops = model.flops / (measured_ms * 1e9)
        measured_svg = (
            f'<circle cx="{sx(measured_ai):.1f}" cy="{sy(measured_tflops):.1f}" r="6" fill="#dc2626"/>'
            f'<text x="{sx(measured_ai) + 10:.1f}" y="{sy(measured_tflops) - 8:.1f}" '
            f'font-size="13" font-family="Arial">measured</text>'
        )
        measured_text = f"measured={measured_tflops:.2f} TFLOP/s using {measured_traffic_model} AI"
    estimated_svg = ""
    estimated_text = "estimated point: pipeline latency model disabled"
    if pipeline_model is not None:
        estimated_ai = tiled_ai if estimated_traffic_model == "tiled" else ideal_ai
        estimated_svg = (
            f'<rect x="{sx(estimated_ai) - 5:.1f}" y="{sy(pipeline_model.estimated_tflops) - 5:.1f}" '
            f'width="10" height="10" fill="#7c3aed" transform="rotate(45 {sx(estimated_ai):.1f} {sy(pipeline_model.estimated_tflops):.1f})"/>'
            f'<text x="{sx(estimated_ai) + 12:.1f}" y="{sy(pipeline_model.estimated_tflops) + 4:.1f}" '
            f'font-size="13" font-family="Arial">estimated</text>'
        )
        estimated_text = (
            f"estimated={pipeline_model.estimated_tflops:.2f} TFLOP/s, "
            f"{pipeline_model.estimated_kernel_ms:.4f} ms, steady={pipeline_model.steady_state_bottleneck}"
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.1f}" y="22" text-anchor="middle" font-size="18" font-family="Arial">Static Roofline Model</text>
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#333"/>
  {"".join(grid_lines)}
  {"".join(roof)}
  <circle cx="{sx(tiled_ai):.1f}" cy="{sy(tiled_bound):.1f}" r="6" fill="white" stroke="#f97316" stroke-width="2"/>
  <text x="{sx(tiled_ai) + 10:.1f}" y="{sy(tiled_bound) - 8:.1f}" font-size="13" font-family="Arial">CTA global bound</text>
  <circle cx="{sx(ideal_ai):.1f}" cy="{sy(ideal_bound):.1f}" r="6" fill="white" stroke="#16a34a" stroke-width="2"/>
  <text x="{sx(ideal_ai) + 10:.1f}" y="{sy(ideal_bound) + 18:.1f}" font-size="13" font-family="Arial">ideal HBM bound</text>
  {measured_svg}
  {estimated_svg}
  <text x="{width / 2:.1f}" y="{height - 14}" text-anchor="middle" font-size="14" font-family="Arial">Arithmetic intensity (FLOP/byte)</text>
  <text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle" font-size="14" font-family="Arial">Performance bound (TFLOP/s)</text>
  {"".join(labels)}
  <text x="{left + 12}" y="{top + 22}" font-size="13" font-family="Arial">peak={peak_tflops:g} TFLOP/s, bandwidth={bandwidth_gbs:g} GB/s</text>
  <text x="{left + 12}" y="{top + 42}" font-size="13" font-family="Arial">CTA global AI={tiled_ai:.2f}, ideal HBM AI={ideal_ai:.2f}</text>
  <text x="{left + 12}" y="{top + 62}" font-size="13" font-family="Arial">{measured_text}</text>
  <text x="{left + 12}" y="{top + 82}" font-size="13" font-family="Arial">{estimated_text}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tir", nargs="?", type=Path, default=DEFAULT_TIR)
    parser.add_argument("--out-dir", type=Path, default=Path("model_out"))
    parser.add_argument("--peak-tflops", type=float, default=125.0, help="Hardware FP16 tensor-core peak in TFLOP/s.")
    parser.add_argument("--bandwidth-gbs", type=float, default=760.0, help="Hardware DRAM bandwidth in GB/s.")
    parser.add_argument(
        "--load-bandwidth-gbs",
        type=float,
        default=3000.0,
        help="Estimated effective global/L2-to-shared load path bandwidth in GB/s.",
    )
    parser.add_argument(
        "--store-bandwidth-gbs",
        type=float,
        default=None,
        help="Estimated store bandwidth in GB/s. Defaults to --bandwidth-gbs.",
    )
    parser.add_argument("--num-sms", type=int, default=84, help="Number of SMs for the assumed GPU.")
    parser.add_argument("--smem-per-sm-kib", type=float, default=100.0, help="Shared memory capacity per SM in KiB.")
    parser.add_argument("--max-threads-per-sm", type=int, default=1536)
    parser.add_argument("--cta-limit-per-sm", type=int, default=16)
    parser.add_argument("--tensor-efficiency", type=float, default=0.75)
    parser.add_argument("--load-efficiency", type=float, default=0.75)
    parser.add_argument("--store-efficiency", type=float, default=0.65)
    parser.add_argument("--measured-ms", type=float, default=None, help="Measured kernel latency in milliseconds.")
    parser.add_argument(
        "--measured-traffic-model",
        choices=["ideal", "tiled"],
        default="ideal",
        help="AI x-position for measured point. Use ideal for HBM roofline, tiled for CTA global-load model.",
    )
    parser.add_argument(
        "--estimated-traffic-model",
        choices=["ideal", "tiled"],
        default="ideal",
        help="AI x-position for the estimated pipeline point.",
    )
    args = parser.parse_args()

    text = args.tir.read_text(encoding="utf-8")
    facts = parse_facts(text)
    model = infer_gemm_model(text, facts, args.peak_tflops, args.bandwidth_gbs)
    pipeline_model = estimate_pipeline_latency(
        model=model,
        num_sms=args.num_sms,
        smem_per_sm_bytes=int(args.smem_per_sm_kib * 1024),
        max_threads_per_sm=args.max_threads_per_sm,
        cta_limit_per_sm=args.cta_limit_per_sm,
        tensor_peak_tflops=args.peak_tflops,
        load_bandwidth_gbs=args.load_bandwidth_gbs,
        store_bandwidth_gbs=args.store_bandwidth_gbs or args.bandwidth_gbs,
        tensor_efficiency=args.tensor_efficiency,
        load_efficiency=args.load_efficiency,
        store_efficiency=args.store_efficiency,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tir_file": str(args.tir),
        "facts_from_tir": asdict(facts),
        "gemm_model": asdict(model),
        "pipeline_latency_model": asdict(pipeline_model),
    }

    json_path = args.out_dir / "gemm_static_model.json"
    csv_path = args.out_dir / "gemm_static_model.csv"
    roofline_path = args.out_dir / "gemm_roofline.png"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_record = asdict(model)
    csv_record.update({f"pipeline_{key}": value for key, value in asdict(pipeline_model).items()})
    write_csv(csv_path, csv_record)
    plotted = maybe_write_roofline(
        roofline_path,
        model,
        args.peak_tflops,
        args.bandwidth_gbs,
        args.measured_ms,
        args.measured_traffic_model,
        pipeline_model,
        args.estimated_traffic_model,
    )

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    if plotted:
        print(f"Wrote {roofline_path}")

    print()
    print("Static GEMM summary")
    print(f"  kernel: {model.kernel_name}")
    print(f"  arch: {model.arch}")
    print(f"  shape: M={model.M}, N={model.N}, K={model.K}")
    print(f"  tile: block_M={model.block_M}, block_N={model.block_N}, block_K={model.block_K}")
    print(f"  launch: grid=({model.grid_x}, {model.grid_y}), threads/CTA={model.threads_per_cta}")
    print(f"  shared memory/CTA: {model.dynamic_shared_bytes_per_cta} B")
    print(f"  FLOPs: {model.flops:,}")
    print(f"  tiled DRAM bytes: {model.dram_bytes_tiled:,}")
    print(f"  tiled AI: {model.arithmetic_intensity_tiled:.2f} FLOP/byte")
    print(f"  ideal AI: {model.arithmetic_intensity_ideal:.2f} FLOP/byte")
    if model.roofline_bound_tflops is not None:
        print(f"  roofline bound: {model.roofline_bound_tflops:.2f} TFLOP/s")
    print()
    print("Estimated pipeline latency")
    print(f"  resident CTAs/SM: {pipeline_model.resident_ctas_per_sm}")
    print(f"  active CTAs/wave: {pipeline_model.active_ctas_per_wave}")
    print(f"  waves: {pipeline_model.waves}")
    print(f"  load per K tile: {pipeline_model.load_us_per_k_tile:.4f} us")
    print(f"  compute per K tile: {pipeline_model.compute_us_per_k_tile:.4f} us")
    print(f"  store per CTA: {pipeline_model.store_us_per_cta:.4f} us")
    print(f"  steady bottleneck: {pipeline_model.steady_state_bottleneck}")
    print(f"  estimated latency: {pipeline_model.estimated_kernel_ms:.4f} ms")
    print(f"  estimated throughput: {pipeline_model.estimated_tflops:.2f} TFLOP/s")


if __name__ == "__main__":
    main()
