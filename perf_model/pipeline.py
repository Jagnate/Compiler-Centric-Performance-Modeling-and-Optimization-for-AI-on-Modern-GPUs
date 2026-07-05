from __future__ import annotations

import math

from .models import GemmModel, HardwareConfig, PipelineLatencyModel
from .occupancy import estimate_occupancy


def fixed_pipeline_cta_latency_us(
    prologue_tiles: int,
    steady_tiles: int,
    epilogue_tiles: int,
    load_us: float,
    compute_us: float,
    store_us: float,
) -> float:
    # Fixed baseline model:
    #   fill loads + overlapped steady K tiles + drain compute + final store.
    return (
        prologue_tiles * load_us
        + steady_tiles * max(load_us, compute_us)
        + epilogue_tiles * compute_us
        + store_us
    )


def estimate_simple_pipeline_latency(
    model: GemmModel,
    hw: HardwareConfig,
    load_bandwidth_gbs: float | None = None,
    store_bandwidth_gbs: float | None = None,
) -> PipelineLatencyModel:
    # This is the compact baseline used by the static roofline script: one CTA
    # latency envelope multiplied by the number of CTA waves on the GPU.
    occupancy = estimate_occupancy(
        hw=hw,
        threads_per_cta=model.threads_per_cta,
        warps_per_cta=model.warps_per_cta,
        dynamic_shared_bytes_per_cta=model.dynamic_shared_bytes_per_cta,
        registers_per_thread=model.registers_per_thread,
    )
    resident_ctas = occupancy.resident_ctas_per_sm
    active_ctas_per_wave = min(model.num_ctas, hw.num_sms * resident_ctas)
    waves = math.ceil(model.num_ctas / active_ctas_per_wave)

    load_bw = load_bandwidth_gbs if load_bandwidth_gbs is not None else hw.l2_bandwidth_gbs
    store_bw = store_bandwidth_gbs if store_bandwidth_gbs is not None else hw.ddr_bandwidth_gbs

    effective_tensor_peak = hw.tensor_peak_tflops * hw.tensor_efficiency
    effective_load_bandwidth = load_bw * hw.l2_efficiency
    effective_store_bandwidth = store_bw * hw.ddr_efficiency

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

    # Software pipeline approximation: pay load-only fill, overlapped steady
    # iterations, compute-only drain, then the final C store.
    prologue_tiles = min(max((model.pipeline_stages or 1) - 1, 0), model.k_tiles)
    steady_tiles = max(model.k_tiles - prologue_tiles, 0)
    epilogue_tiles = prologue_tiles

    estimated_cta_us = fixed_pipeline_cta_latency_us(
        prologue_tiles=prologue_tiles,
        steady_tiles=steady_tiles,
        epilogue_tiles=epilogue_tiles,
        load_us=load_us,
        compute_us=compute_us,
        store_us=store_us,
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
        num_sms=hw.num_sms,
        smem_per_sm_bytes=hw.smem_per_sm_bytes,
        max_threads_per_sm=hw.max_threads_per_sm,
        cta_limit_per_sm=hw.cta_limit_per_sm,
        resident_ctas_per_sm=resident_ctas,
        occupancy=occupancy,
        active_ctas_per_wave=active_ctas_per_wave,
        waves=waves,
        tensor_peak_tflops=hw.tensor_peak_tflops,
        load_bandwidth_gbs=load_bw,
        store_bandwidth_gbs=store_bw,
        tensor_efficiency=hw.tensor_efficiency,
        load_efficiency=hw.l2_efficiency,
        store_efficiency=hw.ddr_efficiency,
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
    )
