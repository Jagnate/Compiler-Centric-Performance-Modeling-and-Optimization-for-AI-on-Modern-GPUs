# Compiler-Centric-Performance-Modeling-and-Optimization-for-AI-on-Modern-GPUs
This is my MSc Computing Individual Project at Imperial College London, supervised by Dr. Hongxiang Fan and Zhiwen Mo, exploring compiler-centric performance modeling and optimization for AI on modern GPUs.

## Current GEMM Modeling Scripts

The code reads a dumped TileLang/TVM TIR file, extracts static GEMM facts, and
builds simple roofline and tile-centric latency estimates.

```bash
python3 static_tir_model.py --hardware-config hardware_configs/hardware_sm86_default.json
python3 tile_centric_model.py --hardware-config hardware_configs/hardware_sm86_default.json
```

The hardware JSON is a modeling input. Edit or duplicate it when you want to
try another GPU, measured bandwidth, measured peak throughput, or different
efficiency assumptions.
