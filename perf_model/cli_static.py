from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .hardware import load_hardware_config
from .io_utils import write_single_record_csv
from .pipeline import estimate_simple_pipeline_latency
from .roofline import write_roofline_plot
from .tir_parser import infer_gemm_model, parse_facts


DEFAULT_TIR = Path("tir_dump/gemm/20_device_tir.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Static GEMM TIR extractor and roofline model.")
    parser.add_argument("tir", nargs="?", type=Path, default=DEFAULT_TIR)
    parser.add_argument("--out-dir", type=Path, default=Path("model_out"))
    parser.add_argument("--hardware-config", type=Path, default=None)
    parser.add_argument(
        "--compiled-registers-per-thread",
        "--registers-per-thread",
        dest="registers_per_thread",
        type=int,
        default=None,
        help="Optional compiled value used to validate/override the TIR static estimate.",
    )
    parser.add_argument("--measured-ms", type=float, default=None)
    parser.add_argument("--measured-traffic-model", choices=["ideal", "tiled"], default="ideal")
    parser.add_argument("--estimated-traffic-model", choices=["ideal", "tiled"], default="ideal")
    args = parser.parse_args()

    # Static path: extract compiler-visible GEMM facts, create a compact pipeline
    # estimate, and draw a roofline plot with optional measured latency overlay.
    hw = load_hardware_config(args.hardware_config)
    text = args.tir.read_text(encoding="utf-8")
    facts = parse_facts(text, registers_per_thread_override=args.registers_per_thread)
    model = infer_gemm_model(text, facts, hw.tensor_peak_tflops, hw.ddr_bandwidth_gbs)
    pipeline_model = estimate_simple_pipeline_latency(model, hw)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "gemm_static_model.json"
    csv_path = args.out_dir / "gemm_static_model.csv"
    roofline_path = args.out_dir / "gemm_roofline.png"

    payload = {
        # Keep raw facts and derived model side by side so later analysis can
        # distinguish direct TIR observations from modeling assumptions.
        "tir_file": str(args.tir),
        "facts_from_tir": asdict(facts),
        "gemm_model": asdict(model),
        "pipeline_latency_model": asdict(pipeline_model),
        "hardware": asdict(hw),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_record = asdict(model)
    csv_record.update({f"pipeline_{key}": value for key, value in asdict(pipeline_model).items()})
    write_single_record_csv(csv_path, csv_record)
    plotted = write_roofline_plot(
        roofline_path,
        model,
        hw.tensor_peak_tflops,
        hw.ddr_bandwidth_gbs,
        measured_ms=args.measured_ms,
        measured_traffic_model=args.measured_traffic_model,
        pipeline_model=pipeline_model,
        estimated_traffic_model=args.estimated_traffic_model,
    )

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {roofline_path if plotted else roofline_path.with_suffix('.svg')}")
    print()
    print("Static GEMM summary")
    print(f"  kernel: {model.kernel_name}")
    print(f"  shape: M={model.M}, N={model.N}, K={model.K}")
    print(f"  tile: block_M={model.block_M}, block_N={model.block_N}, block_K={model.block_K}")
    print(
        f"  registers/thread: {model.registers_per_thread} "
        f"({model.registers_per_thread_source})"
    )
    print(
        "  CTA residency limits: "
        f"smem={pipeline_model.occupancy.by_shared_memory or 'n/a'}, "
        f"threads={pipeline_model.occupancy.by_threads}, "
        f"registers={pipeline_model.occupancy.by_registers or 'unknown'}, "
        f"architecture={pipeline_model.occupancy.by_architecture}"
    )
    print(f"  FLOPs: {model.flops:,}")
    print(f"  tiled AI: {model.arithmetic_intensity_tiled:.2f} FLOP/byte")
    print(f"  ideal AI: {model.arithmetic_intensity_ideal:.2f} FLOP/byte")
    print(f"  simple pipeline latency: {pipeline_model.estimated_kernel_ms:.4f} ms")
    print(f"  simple pipeline throughput: {pipeline_model.estimated_tflops:.2f} TFLOP/s")
