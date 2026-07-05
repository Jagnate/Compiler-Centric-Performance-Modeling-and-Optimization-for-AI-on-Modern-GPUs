from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .hardware import load_hardware_config
from .tile_centric_gemm import simulate, write_schedule_csv
from .tir_parser import infer_gemm_model, parse_facts


DEFAULT_TIR = Path("tir_dump/gemm/20_device_tir.py")


def print_metric_section(
    title: str,
    rows: list[tuple[str, str, str]],
) -> None:
    """Render a compact table similar to Nsight Compute's CLI output."""
    separator = "-" * 88
    print()
    print(f"Section: {title}")
    print(separator)
    print(f"{'Metric Name':<40} {'Metric Unit':>14} {'Metric Value':>30}")
    print(separator)
    for name, unit, value in rows:
        print(f"{name:<40} {unit:>14} {value:>30}")
    print(separator)


def optional_limit(value: int | None) -> str:
    return str(value) if value is not None else "N/A"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile-centric GEMM latency simulator.")
    parser.add_argument("tir", nargs="?", type=Path, default=DEFAULT_TIR)
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--hardware-config", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["fast", "detailed"],
        default="fast",
        help="Fast analytical cohorts or detailed rolling dynamic-rate simulation.",
    )
    parser.add_argument(
        "--dump-schedule",
        action="store_true",
        help="Write detailed per-SM schedule events to JSON and CSV.",
    )
    parser.add_argument(
        "--compiled-registers-per-thread",
        "--registers-per-thread",
        dest="registers_per_thread",
        type=int,
        default=None,
        help="Optional compiled value used to validate/override the TIR static estimate.",
    )
    args = parser.parse_args()
    if args.dump_schedule and args.mode != "detailed":
        parser.error("--dump-schedule requires --mode detailed")

    # CLI wiring: parse the dumped TIR, infer GEMM metadata, then run the
    # wave-level latency simulator with the selected hardware assumptions.
    hw = load_hardware_config(args.hardware_config)
    text = args.tir.read_text(encoding="utf-8")
    facts = parse_facts(text, registers_per_thread_override=args.registers_per_thread)
    gemm = infer_gemm_model(text, facts, hw.tensor_peak_tflops, hw.ddr_bandwidth_gbs)
    result = simulate(gemm, hw, mode=args.mode)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "gemm_tile_centric_model.json"
    schedule_csv_path = args.out_dir / "gemm_tile_centric_schedule.csv"
    payload = asdict(result)
    payload["schedule_event_count"] = len(result.schedule_events)
    if not args.dump_schedule:
        payload["schedule_events"] = []
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.dump_schedule:
        write_schedule_csv(schedule_csv_path, result.schedule_events)

    utilization = result.aggregate_utilization
    memory_throughput = max(
        utilization.smem_utilization,
        utilization.l2_utilization,
        utilization.hbm_utilization,
    )
    print_metric_section(
        "GPU Speed Of Light Throughput",
        [
            ("Duration", "us", f"{result.total_latency_us:.3f}"),
            (
                "Compute (SM) Throughput",
                "%",
                f"{utilization.tensor_core_utilization * 100:.2f}",
            ),
            ("Memory Throughput", "%", f"{memory_throughput * 100:.2f}"),
            ("DRAM Throughput", "%", f"{utilization.hbm_utilization * 100:.2f}"),
            ("L2 Cache Throughput", "%", f"{utilization.l2_utilization * 100:.2f}"),
            (
                "Shared Memory Throughput",
                "%",
                f"{utilization.smem_utilization * 100:.2f}",
            ),
            ("Estimated L2 Hit Rate", "%", f"{result.overall_l2_hit_rate * 100:.2f}"),
            ("Estimated Throughput", "TFLOP/s", f"{result.estimated_tflops:.3f}"),
        ],
    )
    print_metric_section(
        "Launch Statistics",
        [
            ("Kernel Name", "", result.kernel_name),
            ("Architecture", "", result.arch or "unknown"),
            ("Problem Size", "M x N x K", f"{result.M} x {result.N} x {result.K}"),
            (
                "CTA Tile",
                "M x N x K",
                f"{result.block_M} x {result.block_N} x {result.block_K}",
            ),
            ("Grid Size", "block", str(result.num_ctas)),
            ("Block Size", "thread", str(gemm.threads_per_cta)),
            ("Registers Per Thread", "register", str(gemm.registers_per_thread)),
            (
                "Dynamic Shared Memory Per Block",
                "byte",
                str(gemm.dynamic_shared_bytes_per_cta),
            ),
            ("Pipeline Stages", "stage", str(result.pipeline_stages)),
            (
                "Waves Per SM",
                "wave",
                f"{result.num_ctas / result.full_wave_capacity_ctas:.3f}",
            ),
        ],
    )
    print_metric_section(
        "Occupancy",
        [
            ("Block Limit SM", "block", str(result.occupancy.by_architecture)),
            (
                "Block Limit Registers",
                "block",
                optional_limit(result.occupancy.by_registers),
            ),
            (
                "Block Limit Shared Mem",
                "block",
                optional_limit(result.occupancy.by_shared_memory),
            ),
            ("Block Limit Warps", "block", str(result.occupancy.by_threads)),
            (
                "Theoretical Blocks Per SM",
                "block",
                str(result.resident_ctas_per_sm),
            ),
            (
                "Theoretical Occupancy",
                "%",
                f"{utilization.theoretical_occupancy * 100:.2f}",
            ),
            (
                "Achieved Occupancy",
                "%",
                f"{utilization.achieved_occupancy * 100:.2f}",
            ),
            ("SM Activity", "%", f"{utilization.sm_activity * 100:.2f}"),
            (
                "Resident CTA Slot Utilization",
                "%",
                f"{utilization.resident_cta_slot_utilization * 100:.2f}",
            ),
        ],
    )

    scheduler_rows = [
        ("Simulation Mode", "", args.mode),
        ("Schedule Model", "", result.schedule_model),
        ("Full Wave Capacity", "CTA", str(result.full_wave_capacity_ctas)),
        ("Full Waves", "wave", str(result.full_waves)),
        ("Tail CTAs", "CTA", str(result.tail_ctas)),
    ]
    if args.mode == "fast":
        cohort_text = ", ".join(
            f"{cohort.wave_kind}:{cohort.sm_count} SM x {cohort.ctas_per_sm} CTA"
            for cohort in result.fast_cohorts
        )
        scheduler_rows.append(("Analytical Cohorts", "", cohort_text or "none"))
    else:
        initial_events = [event for event in result.schedule_events if event.start_us == 0.0]
        initial_residency: dict[int, int] = {}
        for event in initial_events:
            initial_residency[event.resident_ctas_on_sm] = (
                initial_residency.get(event.resident_ctas_on_sm, 0) + 1
            )
        residency_text = ", ".join(
            f"{sm_count} SM(s) x {ctas} CTA(s)"
            for ctas, sm_count in sorted(initial_residency.items())
        )
        scheduler_rows.extend(
            [
                ("Initial Residency", "", residency_text),
                ("Schedule Events", "event", str(len(result.schedule_events))),
                (
                    "Dynamic Rate Updates",
                    "update",
                    str(
                        sum(
                            event.dynamic_rate_updates
                            for event in result.schedule_events
                        )
                    ),
                ),
            ]
        )
    print_metric_section("Scheduler Statistics", scheduler_rows)

    print()
    print(f"Output: {json_path}")
    if args.dump_schedule:
        print(f"Schedule: {schedule_csv_path}")
