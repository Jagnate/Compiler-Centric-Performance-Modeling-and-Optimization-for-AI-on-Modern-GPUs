#!/usr/bin/env python3
"""A small TPerf-like analytical simulator for the lowered TileLang GEMM TIR.

This is deliberately GEMM-specific. It uses the same high-level ideas as the
TPerf paper:

1. Treat a CTA tile as the modeling unit.
2. Convert each repeated K tile into a resource vector.
3. Combine software pipeline depth with resident CTAs per SM:
       depth = num_stages * resident_ctas_per_sm - 1
4. Evaluate prologue / steady / epilogue for each wave.
5. Estimate L2/DDR split with a lightweight tile-level LRU reuse model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from static_tir_model import GemmModel, infer_gemm_model, parse_facts


DEFAULT_TIR = Path("tir_dump/gemm/20_device_tir.py")


@dataclass
class HardwareConfig:
    num_sms: int
    smem_per_sm_bytes: int
    max_threads_per_sm: int
    cta_limit_per_sm: int
    l2_capacity_bytes: int
    tensor_peak_tflops: float
    l2_bandwidth_gbs: float
    ddr_bandwidth_gbs: float
    smem_bandwidth_gbs: float
    tensor_efficiency: float
    l2_efficiency: float
    ddr_efficiency: float
    smem_efficiency: float


@dataclass
class ResourceVectorUs:
    tc: float = 0.0
    smem: float = 0.0
    l2: float = 0.0
    ddr: float = 0.0

    def steady_time(self) -> float:
        return max(self.tc, self.smem, self.l2, self.ddr)

    def add(self, other: "ResourceVectorUs") -> "ResourceVectorUs":
        return ResourceVectorUs(
            tc=self.tc + other.tc,
            smem=self.smem + other.smem,
            l2=self.l2 + other.l2,
            ddr=self.ddr + other.ddr,
        )


@dataclass
class WaveResult:
    wave_index: int
    ctas: int
    active_sms: int
    resident_ctas_per_sm: int
    effective_depth: int
    l2_hit_rate: float
    load_l2_bytes: int
    load_ddr_bytes: int
    store_ddr_bytes: int
    load_vector_us_per_k: ResourceVectorUs
    compute_vector_us_per_k: ResourceVectorUs
    store_vector_us: ResourceVectorUs
    t_pro_us: float
    t_steady_us: float
    t_epi_us: float
    t_store_us: float
    latency_us: float
    steady_bottleneck: str


@dataclass
class SimulationResult:
    model_name: str
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
    pipeline_stages: int
    resident_ctas_per_sm: int
    full_wave_capacity_ctas: int
    full_waves: int
    tail_ctas: int
    total_latency_us: float
    total_latency_ms: float
    estimated_tflops: float
    overall_l2_hit_rate: float
    formula: str
    hardware: HardwareConfig
    waves: list[WaveResult]


class TileLRU:
    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self.used_bytes = 0
        self.cache: OrderedDict[tuple[Any, ...], int] = OrderedDict()

    def access(self, key: tuple[Any, ...], size: int) -> bool:
        if key in self.cache:
            self.cache.move_to_end(key)
            return True

        while self.cache and self.used_bytes + size > self.capacity_bytes:
            _, evicted_size = self.cache.popitem(last=False)
            self.used_bytes -= evicted_size

        if size <= self.capacity_bytes:
            self.cache[key] = size
            self.used_bytes += size
        return False


def cta_launch_order(model: GemmModel) -> list[tuple[int, int]]:
    # CUDA's blockIdx.x is the fastest-changing grid dimension in the usual
    # linear launch order. The TIR maps bx -> N tile and by -> M tile.
    return [(by, bx) for by in range(model.grid_y) for bx in range(model.grid_x)]


def chunked(items: list[tuple[int, int]], chunk_size: int) -> Iterable[list[tuple[int, int]]]:
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def resident_ctas_per_sm(model: GemmModel, hw: HardwareConfig) -> int:
    by_smem = hw.smem_per_sm_bytes // model.dynamic_shared_bytes_per_cta
    by_threads = hw.max_threads_per_sm // model.threads_per_cta
    return max(1, min(by_smem, by_threads, hw.cta_limit_per_sm))


def per_cta_rates(hw: HardwareConfig, active_sms: int, active_ctas: int) -> dict[str, float]:
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

    load_vec = ResourceVectorUs(
        smem=load_bytes_per_k_per_cta / rates["smem_bytes_s"] * 1e6,
        l2=l2_bytes_per_k_per_cta / rates["l2_bytes_s"] * 1e6,
        ddr=ddr_bytes_per_k_per_cta / rates["ddr_bytes_s"] * 1e6,
    )
    compute_vec = ResourceVectorUs(
        tc=compute_flops_per_k_per_cta / rates["tc_flops_s"] * 1e6,
    )
    store_vec = ResourceVectorUs(
        l2=store_bytes_per_cta / rates["l2_bytes_s"] * 1e6,
        ddr=store_bytes_per_cta / rates["ddr_bytes_s"] * 1e6,
    )

    effective_depth = max(0, (model.pipeline_stages or 1) * resident - 1)
    repeated = model.k_tiles
    steady_vec = load_vec.add(compute_vec)
    t_steady = steady_vec.steady_time()

    fill_iters = min(effective_depth, repeated)
    t_pro = fill_iters * load_vec.steady_time()
    t_epi = fill_iters * compute_vec.steady_time()
    steady_iters = max(repeated - effective_depth, 0)
    t_store = store_vec.steady_time()
    latency = t_pro + steady_iters * t_steady + t_epi + t_store

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
        model_name="tperf_like_gemm_v1",
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
        formula="T = T_pro + max(K_tiles - d, 0) * T_steady + T_epi + T_store, d = stages * resident_ctas_per_SM - 1",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tir", nargs="?", type=Path, default=DEFAULT_TIR)
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--num-sms", type=int, default=84)
    parser.add_argument("--smem-per-sm-kib", type=float, default=100.0)
    parser.add_argument("--max-threads-per-sm", type=int, default=1536)
    parser.add_argument("--cta-limit-per-sm", type=int, default=16)
    parser.add_argument("--l2-capacity-mib", type=float, default=6.0)
    parser.add_argument("--tensor-peak-tflops", type=float, default=125.0)
    parser.add_argument("--l2-bandwidth-gbs", type=float, default=3000.0)
    parser.add_argument("--ddr-bandwidth-gbs", type=float, default=760.0)
    parser.add_argument("--smem-bandwidth-gbs", type=float, default=12000.0)
    parser.add_argument("--tensor-efficiency", type=float, default=0.75)
    parser.add_argument("--l2-efficiency", type=float, default=0.75)
    parser.add_argument("--ddr-efficiency", type=float, default=0.75)
    parser.add_argument("--smem-efficiency", type=float, default=0.70)
    args = parser.parse_args()

    text = args.tir.read_text(encoding="utf-8")
    facts = parse_facts(text)
    gemm = infer_gemm_model(text, facts, args.tensor_peak_tflops, args.ddr_bandwidth_gbs)
    hw = HardwareConfig(
        num_sms=args.num_sms,
        smem_per_sm_bytes=int(args.smem_per_sm_kib * 1024),
        max_threads_per_sm=args.max_threads_per_sm,
        cta_limit_per_sm=args.cta_limit_per_sm,
        l2_capacity_bytes=int(args.l2_capacity_mib * 1024 * 1024),
        tensor_peak_tflops=args.tensor_peak_tflops,
        l2_bandwidth_gbs=args.l2_bandwidth_gbs,
        ddr_bandwidth_gbs=args.ddr_bandwidth_gbs,
        smem_bandwidth_gbs=args.smem_bandwidth_gbs,
        tensor_efficiency=args.tensor_efficiency,
        l2_efficiency=args.l2_efficiency,
        ddr_efficiency=args.ddr_efficiency,
        smem_efficiency=args.smem_efficiency,
    )

    result = simulate(gemm, hw)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "gemm_tperf_like_model.json"
    wave_csv_path = args.out_dir / "gemm_tperf_like_waves.csv"
    json_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    write_wave_csv(wave_csv_path, result.waves)

    print(f"Wrote {json_path}")
    print(f"Wrote {wave_csv_path}")
    print()
    print("Tile-Centric GEMM simulation")
    print(f"  kernel: {result.kernel_name} ({result.arch})")
    print(f"  GEMM: M={result.M}, N={result.N}, K={result.K}")
    print(f"  CTA tile: {result.block_M}x{result.block_N}x{result.block_K}")
    print(f"  CTAs: {result.num_ctas}, resident CTAs/SM: {result.resident_ctas_per_sm}")
    print(f"  effective depth: {result.pipeline_stages} * {result.resident_ctas_per_sm} - 1 = {result.waves[0].effective_depth}")
    print(f"  full wave capacity: {result.full_wave_capacity_ctas} CTAs")
    print(f"  full waves: {result.full_waves}, tail CTAs: {result.tail_ctas}")
    print(f"  estimated L2 hit rate: {result.overall_l2_hit_rate * 100:.2f}%")
    print(f"  estimated latency: {result.total_latency_ms:.4f} ms")
    print(f"  estimated throughput: {result.estimated_tflops:.2f} TFLOP/s")
    print()
    print("Wave breakdown")
    for wave in result.waves:
        print(
            f"  wave {wave.wave_index}: ctas={wave.ctas}, active_sms={wave.active_sms}, "
            f"L2_hit={wave.l2_hit_rate * 100:.2f}%, bottleneck={wave.steady_bottleneck}, "
            f"latency={wave.latency_us:.3f} us"
        )


if __name__ == "__main__":
    main()
