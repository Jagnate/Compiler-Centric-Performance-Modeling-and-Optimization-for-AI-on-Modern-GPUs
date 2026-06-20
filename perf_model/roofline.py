from __future__ import annotations

import math
from pathlib import Path

from .models import GemmModel, PipelineLatencyModel


def write_roofline_plot(
    path: Path,
    model: GemmModel,
    peak_tflops: float,
    bandwidth_gbs: float,
    measured_ms: float | None = None,
    measured_traffic_model: str = "ideal",
    pipeline_model: PipelineLatencyModel | None = None,
    estimated_traffic_model: str = "ideal",
) -> bool:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        # Keep the script useful on minimal environments by emitting a dependency-
        # free SVG when matplotlib/numpy are not installed.
        write_roofline_svg(
            path.with_suffix(".svg"),
            model,
            peak_tflops,
            bandwidth_gbs,
            measured_ms,
            measured_traffic_model,
            pipeline_model,
            estimated_traffic_model,
        )
        return False

    x = np.logspace(-1, 4, 400)
    y = np.minimum(peak_tflops, x * bandwidth_gbs / 1000.0)
    tiled_ai = model.arithmetic_intensity_tiled
    ideal_ai = model.arithmetic_intensity_ideal
    tiled_bound = min(peak_tflops, tiled_ai * bandwidth_gbs / 1000.0)
    ideal_bound = min(peak_tflops, ideal_ai * bandwidth_gbs / 1000.0)

    # Hollow points are static upper bounds; filled points are latency-derived
    # positions and are the ones to compare against real performance.
    plt.figure(figsize=(7, 5))
    plt.loglog(x, y, label="Roofline bound")
    plt.axhline(peak_tflops, linestyle="--", alpha=0.5, label="Compute peak")
    plt.axvline(tiled_ai, linestyle="--", alpha=0.25)
    plt.axvline(ideal_ai, linestyle="--", alpha=0.25)
    plt.scatter([tiled_ai], [tiled_bound], s=84, facecolors="none", edgecolors="#f97316", linewidths=2, label="CTA global bound")
    plt.scatter([ideal_ai], [ideal_bound], s=84, facecolors="none", edgecolors="#16a34a", linewidths=2, label="Ideal HBM bound")
    if measured_ms is not None:
        measured_ai = tiled_ai if measured_traffic_model == "tiled" else ideal_ai
        measured_tflops = model.flops / (measured_ms * 1e9)
        plt.scatter([measured_ai], [measured_tflops], s=86, color="#dc2626", label="Measured kernel")
    if pipeline_model is not None:
        estimated_ai = tiled_ai if estimated_traffic_model == "tiled" else ideal_ai
        plt.scatter([estimated_ai], [pipeline_model.estimated_tflops], s=90, color="#7c3aed", marker="D", label="Estimated pipeline")
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

    def line(x1: float, y1: float, x2: float, y2: float, color: str, dash: str = "") -> str:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2.4"{dash_attr}/>'

    ridge_ai = peak_tflops * 1000.0 / bandwidth_gbs
    tiled_ai = model.arithmetic_intensity_tiled
    ideal_ai = model.arithmetic_intensity_ideal
    tiled_bound = min(peak_tflops, tiled_ai * bandwidth_gbs / 1000.0)
    ideal_bound = min(peak_tflops, ideal_ai * bandwidth_gbs / 1000.0)

    mem_x1 = min(ridge_ai, x_max)
    comp_x0 = max(ridge_ai, x_min)
    grid_lines = []
    labels = []
    for exp in range(-1, 5):
        v = 10.0**exp
        x = sx(v)
        grid_lines.append(line(x, top, x, top + plot_h, "#dddddd", "3 5"))
        labels.append(f'<text x="{x:.1f}" y="{height - 42}" text-anchor="middle" font-size="12">{v:g}</text>')
    for exp in range(math.floor(lx(y_min)), math.ceil(lx(y_max)) + 1):
        v = 10.0**exp
        if y_min <= v <= y_max:
            y = sy(v)
            grid_lines.append(line(left, y, left + plot_w, y, "#dddddd", "3 5"))
            labels.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{v:g}</text>')

    measured_svg = ""
    measured_text = "measured point: pass --measured-ms to draw actual performance"
    if measured_ms is not None:
        measured_ai = tiled_ai if measured_traffic_model == "tiled" else ideal_ai
        measured_tflops = model.flops / (measured_ms * 1e9)
        measured_svg = f'<circle cx="{sx(measured_ai):.1f}" cy="{sy(measured_tflops):.1f}" r="6" fill="#dc2626"/>'
        measured_text = f"measured={measured_tflops:.2f} TFLOP/s using {measured_traffic_model} AI"

    estimated_svg = ""
    estimated_text = "estimated point: pipeline latency model disabled"
    if pipeline_model is not None:
        estimated_ai = tiled_ai if estimated_traffic_model == "tiled" else ideal_ai
        estimated_svg = (
            f'<rect x="{sx(estimated_ai) - 5:.1f}" y="{sy(pipeline_model.estimated_tflops) - 5:.1f}" '
            f'width="10" height="10" fill="#7c3aed" transform="rotate(45 {sx(estimated_ai):.1f} {sy(pipeline_model.estimated_tflops):.1f})"/>'
        )
        estimated_text = f"estimated={pipeline_model.estimated_tflops:.2f} TFLOP/s, {pipeline_model.estimated_kernel_ms:.4f} ms"

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.1f}" y="22" text-anchor="middle" font-size="18" font-family="Arial">Static Roofline Model</text>
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#333"/>
  {"".join(grid_lines)}
  {line(sx(x_min), sy(max(y_min, x_min * bandwidth_gbs / 1000.0)), sx(mem_x1), sy(min(peak_tflops, mem_x1 * bandwidth_gbs / 1000.0)), "#2563eb")}
  {line(sx(comp_x0), sy(peak_tflops), sx(x_max), sy(peak_tflops), "#2563eb")}
  {line(sx(tiled_ai), top, sx(tiled_ai), top + plot_h, "#f97316", "4 5")}
  {line(sx(ideal_ai), top, sx(ideal_ai), top + plot_h, "#16a34a", "4 5")}
  <circle cx="{sx(tiled_ai):.1f}" cy="{sy(tiled_bound):.1f}" r="6" fill="white" stroke="#f97316" stroke-width="2"/>
  <circle cx="{sx(ideal_ai):.1f}" cy="{sy(ideal_bound):.1f}" r="6" fill="white" stroke="#16a34a" stroke-width="2"/>
  {measured_svg}
  {estimated_svg}
  <text x="{width / 2:.1f}" y="{height - 14}" text-anchor="middle" font-size="14" font-family="Arial">Arithmetic intensity (FLOP/byte)</text>
  <text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle" font-size="14" font-family="Arial">Performance (TFLOP/s)</text>
  {"".join(labels)}
  <text x="{left + 12}" y="{top + 22}" font-size="13" font-family="Arial">peak={peak_tflops:g} TFLOP/s, bandwidth={bandwidth_gbs:g} GB/s</text>
  <text x="{left + 12}" y="{top + 42}" font-size="13" font-family="Arial">CTA global AI={tiled_ai:.2f}, ideal HBM AI={ideal_ai:.2f}</text>
  <text x="{left + 12}" y="{top + 62}" font-size="13" font-family="Arial">{measured_text}</text>
  <text x="{left + 12}" y="{top + 82}" font-size="13" font-family="Arial">{estimated_text}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")
