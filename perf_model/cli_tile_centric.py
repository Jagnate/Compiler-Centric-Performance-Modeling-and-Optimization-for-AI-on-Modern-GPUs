from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .hardware import load_hardware_config
from .tile_centric_gemm import simulate, write_wave_csv
from .tir_parser import infer_gemm_model, parse_facts


DEFAULT_TIR = Path("tir_dump/gemm/20_device_tir.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile-centric GEMM latency simulator.")
    parser.add_argument("tir", nargs="?", type=Path, default=DEFAULT_TIR)
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--hardware-config", type=Path, default=None)
    args = parser.parse_args()

    # CLI wiring: parse the dumped TIR, infer GEMM metadata, then run the
    # wave-level latency simulator with the selected hardware assumptions.
    hw = load_hardware_config(args.hardware_config)
    text = args.tir.read_text(encoding="utf-8")
    facts = parse_facts(text)
    gemm = infer_gemm_model(text, facts, hw.tensor_peak_tflops, hw.ddr_bandwidth_gbs)
    result = simulate(gemm, hw)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "gemm_tile_centric_model.json"
    wave_csv_path = args.out_dir / "gemm_tile_centric_waves.csv"
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
