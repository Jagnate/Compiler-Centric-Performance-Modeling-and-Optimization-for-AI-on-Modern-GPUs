from __future__ import annotations

import csv
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .models import GemmModel, HardwareConfig, ResourceVectorUs, SimulationResult, WaveResult
from .tile_cache import TileLRU


# This simulator is inspired by the referenced tile-centric performance modeling
# paper: model hardware-level CTA waves and software pipeline fill/steady/drain
# separately, then combine them into an end-to-end kernel latency estimate.
def cta_launch_order(model: GemmModel) -> list[tuple[int, int]]:
    # TileLang GEMM launch order is approximated as row-major CTA coordinates.
    return [(by, bx) for by in range(model.grid_y) for bx in range(model.grid_x)]


def chunked(items: list[tuple[int, int]], chunk_size: int) -> Iterable[list[tuple[int, int]]]:
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def resident_ctas_per_sm(model: GemmModel, hw: HardwareConfig) -> int:
    # Occupancy is limited by dynamic shared memory, thread slots, and the
    # architecture's maximum resident CTA count.
    by_smem = hw.smem_per_sm_bytes // model.dynamic_shared_bytes_per_cta
    by_threads = hw.max_threads_per_sm // model.threads_per_cta
    return max(1, min(by_smem, by_threads, hw.cta_limit_per_sm))


def per_cta_rates(hw: HardwareConfig, active_sms: int, active_ctas: int) -> dict[str, float]:
    # Peak resources are shared by active CTAs in the current wave. Tail waves get
    # proportionally less chip-level bandwidth/compute because fewer SMs are active.
    active_fraction = active_sms / hw.num_sms
    return {
        "tc_flops_s": hw.tensor_peak_tflops * hw.tensor_efficiency * 1e12 * active_fraction / active_ctas,
        "l2_bytes_s": hw.l2_bandwidth_gbs * hw.l2_efficiency * 1e9 * active_fraction / active_ctas,
        "ddr_bytes_s": hw.ddr_bandwidth_gbs * hw.ddr_efficiency * 1e9 * active_fraction / active_ctas,
        "smem_bytes_s": hw.smem_bandwidth_gbs * hw.smem_efficiency * 1e9 * active_fraction / active_ctas,
    }


def bottleneck_name(v: ResourceVectorUs) -> str:
    values = {"tc": v.tc, "smem": v.smem, "l2": v.l2, "ddr": v.ddr}
    return max(values, key=values.get)


def fixed_wave_latency_us(
    load_vec: ResourceVectorUs,
    compute_vec: ResourceVectorUs,
    store_vec: ResourceVectorUs,
    k_tiles: int,
    effective_depth: int,
) -> tuple[float, float, float, float, float]:
    # Fixed tile-centric wave model:
    #   prologue load fill + overlapped steady K tiles + compute drain + C store.
    fill_iters = min(effective_depth, k_tiles)
    steady_iters = max(k_tiles - effective_depth, 0)
    steady_vec = load_vec.add(compute_vec)
    t_pro = fill_iters * load_vec.steady_time()
    t_steady = steady_vec.steady_time()
    t_epi = fill_iters * compute_vec.steady_time()
    t_store = store_vec.steady_time()
    latency = t_pro + steady_iters * t_steady + t_epi + t_store
    return t_pro, t_steady, t_epi, t_store, latency


def simulate_wave(
    wave_index: int,
    wave_ctas: list[tuple[int, int]],
    model: GemmModel,
    hw: HardwareConfig,
    lru: TileLRU,
    resident: int,
) -> WaveResult:
    active_ctas = len(wave_ctas)
    active_sms = math.ceil(active_ctas / resident)
    rates = per_cta_rates(hw, active_sms=active_sms, active_ctas=active_ctas)

    a_tile_bytes = model.block_M * model.block_K * model.bytes_a_per_element
    b_tile_bytes = model.block_K * model.block_N * model.bytes_b_per_element
    load_bytes_per_k_per_cta = a_tile_bytes + b_tile_bytes
    compute_flops_per_k_per_cta = 2 * model.block_M * model.block_N * model.block_K
    store_bytes_per_cta = model.block_M * model.block_N * model.bytes_c_per_element

    # Count A/B tile accesses and classify them as L2 hits or HBM misses with a
    # small logical LRU model. Stores are counted separately after the K loop.
    l2_bytes = 0
    ddr_bytes = 0
    hit_bytes = 0

    for by, bx in wave_ctas:
        for k in range(model.k_tiles):
            for key, size in ((("A", by, k), a_tile_bytes), (("B", k, bx), b_tile_bytes)):
                l2_bytes += size
                if lru.access(key, size):
                    hit_bytes += size
                else:
                    ddr_bytes += size

    total_load_bytes = active_ctas * model.k_tiles * load_bytes_per_k_per_cta
    l2_hit_rate = hit_bytes / total_load_bytes if total_load_bytes else 0.0

    l2_bytes_per_k_per_cta = l2_bytes / active_ctas / model.k_tiles
    ddr_bytes_per_k_per_cta = ddr_bytes / active_ctas / model.k_tiles

    # Per-K resource vectors are measured in microseconds. Resource lanes can
    # overlap, so max(tc, smem, l2, ddr) is the local steady time.
    load_vec = ResourceVectorUs(
        smem=load_bytes_per_k_per_cta / rates["smem_bytes_s"] * 1e6,
        l2=l2_bytes_per_k_per_cta / rates["l2_bytes_s"] * 1e6,
        ddr=ddr_bytes_per_k_per_cta / rates["ddr_bytes_s"] * 1e6,
    )
    compute_vec = ResourceVectorUs(tc=compute_flops_per_k_per_cta / rates["tc_flops_s"] * 1e6)
    store_vec = ResourceVectorUs(
        l2=store_bytes_per_cta / rates["l2_bytes_s"] * 1e6,
        ddr=store_bytes_per_cta / rates["ddr_bytes_s"] * 1e6,
    )

    effective_depth = max(0, (model.pipeline_stages or 1) * resident - 1)
    # Hardware-level resident CTAs extend the software pipeline window: while one
    # CTA drains, other CTAs on the same SM can keep issuing middle K tiles.
    steady_vec = load_vec.add(compute_vec)
    t_pro, t_steady, t_epi, t_store, latency = fixed_wave_latency_us(
        load_vec=load_vec,
        compute_vec=compute_vec,
        store_vec=store_vec,
        k_tiles=model.k_tiles,
        effective_depth=effective_depth,
    )

    return WaveResult(
        wave_index=wave_index,
        ctas=active_ctas,
        active_sms=active_sms,
        resident_ctas_per_sm=resident,
        effective_depth=effective_depth,
        l2_hit_rate=l2_hit_rate,
        load_l2_bytes=int(l2_bytes),
        load_ddr_bytes=int(ddr_bytes),
        store_ddr_bytes=active_ctas * store_bytes_per_cta,
        load_vector_us_per_k=load_vec,
        compute_vector_us_per_k=compute_vec,
        store_vector_us=store_vec,
        t_pro_us=t_pro,
        t_steady_us=t_steady,
        t_epi_us=t_epi,
        t_store_us=t_store,
        latency_us=latency,
        steady_bottleneck=bottleneck_name(steady_vec),
    )


def simulate(model: GemmModel, hw: HardwareConfig) -> SimulationResult:
    resident = resident_ctas_per_sm(model, hw)
    full_capacity = hw.num_sms * resident
    order = cta_launch_order(model)
    lru = TileLRU(hw.l2_capacity_bytes)

    # Split the full grid into hardware waves. Each wave may have a different L2
    # hit rate and active-SM count, especially the final tail wave.
    waves = [
        simulate_wave(i, wave, model, hw, lru, resident)
        for i, wave in enumerate(chunked(order, full_capacity))
    ]

    total_latency_us = sum(w.latency_us for w in waves)
    total_latency_ms = total_latency_us / 1000.0
    estimated_tflops = model.flops / (total_latency_ms * 1e9)
    total_l2 = sum(w.load_l2_bytes for w in waves)
    total_ddr = sum(w.load_ddr_bytes for w in waves)
    overall_l2_hit = 1.0 - total_ddr / total_l2 if total_l2 else 0.0
    full_waves = model.num_ctas // full_capacity
    tail_ctas = model.num_ctas % full_capacity

    return SimulationResult(
        model_name="tile_centric_gemm_v1",
        kernel_name=model.kernel_name,
        arch=model.arch,
        M=model.M,
        N=model.N,
        K=model.K,
        block_M=model.block_M,
        block_N=model.block_N,
        block_K=model.block_K,
        k_tiles=model.k_tiles,
        grid_x=model.grid_x,
        grid_y=model.grid_y,
        num_ctas=model.num_ctas,
        pipeline_stages=model.pipeline_stages or 1,
        resident_ctas_per_sm=resident,
        full_wave_capacity_ctas=full_capacity,
        full_waves=full_waves,
        tail_ctas=tail_ctas,
        total_latency_us=total_latency_us,
        total_latency_ms=total_latency_ms,
        estimated_tflops=estimated_tflops,
        overall_l2_hit_rate=overall_l2_hit,
        hardware=hw,
        waves=waves,
    )


def write_wave_csv(path: Path, waves: list[WaveResult]) -> None:
    rows: list[dict[str, Any]] = []
    for wave in waves:
        row = asdict(wave)
        for key in ("load_vector_us_per_k", "compute_vector_us_per_k", "store_vector_us"):
            vec = row.pop(key)
            for resource, value in vec.items():
                row[f"{key}_{resource}"] = value
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
