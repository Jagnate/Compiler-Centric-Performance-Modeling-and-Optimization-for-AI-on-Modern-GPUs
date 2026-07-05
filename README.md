# Compiler-Centric-Performance-Modeling-and-Optimization-for-AI-on-Modern-GPUs
This is my MSc Computing Individual Project at Imperial College London, supervised by Dr. Hongxiang Fan and Zhiwen Mo, exploring compiler-centric performance modeling and optimization for AI on modern GPUs.

## Current GEMM Modeling Scripts

The code reads a dumped TileLang/TVM TIR file, extracts static GEMM facts, and
builds simple roofline and tile-centric latency estimates.

```bash
python3 static_tir_model.py --hardware-config hardware_configs/hardware_sm86_default.json
python3 tile_centric_model.py \
  --hardware-config hardware_configs/hardware_sm86_default.json \
  --mode fast
```

The hardware JSON is a modeling input. Edit or duplicate it when you want to
try another GPU, measured bandwidth, measured peak throughput, or different
efficiency assumptions.

The model estimates register pressure directly from TIR-visible local buffers,
kernel pointers, launch indices, and serial-loop state. No register input is
required for normal static prediction.

The estimate is heuristic because backend liveness optimization and address
temporaries are applied after TIR. A compiled value from `ptxas` or NCU can
optionally validate it:

```bash
python3 tile_centric_model.py \
  --hardware-config hardware_configs/hardware_sm86_default.json \
  --compiled-registers-per-thread 217
```

The tile-centric simulator has two execution modes:

- `fast` (default) evaluates analytical full-wave and tail-wave residency
  cohorts. It avoids materializing per-CTA events and is intended for rapid
  design-space exploration.
- `detailed` uses a rolling SM scheduler and a logical L2 LRU. It spreads CTAs
  across available SMs, dispatches new batches when an SM becomes free, and
  dynamically reallocates Tensor Core, shared-memory, L2, and HBM rates at
  each resource-completion or pipeline-phase event.

The detailed simulator is event-driven rather than cycle-accurate. Its cost
therefore scales with CTA batches and resource transitions, not GPU cycles.
Use `--dump-schedule` only when a per-SM JSON/CSV trace is needed:

```bash
python3 tile_centric_model.py \
  --hardware-config hardware_configs/hardware_sm86_default.json \
  --mode detailed \
  --dump-schedule
```

This writes `out/gemm_tile_centric_model.json` and
`out/gemm_tile_centric_schedule.csv`. Without `--dump-schedule`, the detailed
simulation still runs, but the potentially large event list is omitted from
the JSON output.

Both modes report aggregate utilization over the complete kernel time window:
SM activity, resident CTA-slot utilization, and TC/SMEM/L2/HBM utilization.
Resource utilization is normalized against the configured physical peak so
the percentages are closer to NCU's peak-relative metrics. Theoretical and
time-weighted achieved occupancy are reported separately. Detailed mode
integrates the rolling schedule events; fast mode integrates its analytical
full/tail cohorts.

## Recursive Pipeline Envelope

The pipeline implementation is inspired by the paper's recursive
prologue-steady-epilogue analysis. A GEMM is represented as a node tree:

```text
tile grid
  -> full/tail wave or rolling scheduler cohort
    -> CTA sequence
      -> pipelined K loop
        -> load A/B action
        -> MMA compute action
      -> store C action
```

Each pipeline loop is evaluated after its children. Its effective depth is
`software stages * resident tiles per SM - 1`; short loops clamp the steady
iteration count to zero. Nested pipelines retain the child envelope's boundary
critical path when their parent composes it. Fast mode evaluates the recursive
tree with fixed per-resource times. Detailed mode uses the same tree to produce
resource-work phases, then advances those phases with dynamic TC/SMEM/L2/HBM
rate updates under the rolling SM scheduler.

The complete tree is exported as `pipeline_structure` in the model JSON.
