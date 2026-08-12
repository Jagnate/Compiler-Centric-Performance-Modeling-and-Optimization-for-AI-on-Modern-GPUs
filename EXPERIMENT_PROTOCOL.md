# Experimental Protocol

This document defines the Phase 0-3 protocol for evaluating model-guided GPU
kernel generation. It applies unchanged to the included Matmul and Flash
Attention workloads.

## Claim Under Test

The implementation does not claim that an analytical model replaces execution.
It tests whether TileSight can allocate a limited hardware-evaluation budget
better than model-top-only or random selection while preserving correctness and
final measured performance.

The primary comparison is:

| Strategy | Proposal budget | CUDA promotions | NCU policy | Final gate |
| --- | ---: | ---: | --- | --- |
| adaptive | equal | equal cap | every round | equal |
| model-top | equal | equal cap | every round | equal |
| random | equal | equal cap | every round | equal |

`measure-all` is not an equal-budget strategy. It is an optional high-cost
reference that measures every model-valid, non-equivalent candidate.

## Fidelity Pipeline

1. Validate the complete Python source and immutable task invariants.
2. Lower every valid candidate and run the TileSight TIR model.
3. Deduplicate candidates with identical compiled CUDA source and launch identities.
4. Use the selected policy to allocate a fixed number of CUDA measurements.
5. Run public multi-case correctness before timing.
6. Update the measured beam using CUDA Event latency only.
7. Collect NCU according to the configured profile policy.
8. Feed failures, measurements, model error, diagnosis, and NCU evidence into
   later hosted-model requests.
9. Recompile the seed and finalists in fresh processes, run held-out cases, and
   export only the lowest-median passing finalist.

TileSight prediction never directly determines the exported source.

## Required Runs

Run the offline suite first:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -t . -v
```

Validate both seed kernels on the actual GPU before spending API budget:

```bash
PYTHONPATH=src python3 examples/validate_seed_kernel.py \
  --source examples/tilelang_matmul_kernel.py \
  --task examples/tilelang_matmul_task.json \
  --output results/matmul_seed_validation

PYTHONPATH=src python3 examples/validate_seed_kernel.py \
  --source examples/tilelang_flash_attention_kernel.py \
  --task examples/tilelang_flash_attention_task.json \
  --output results/flash_attention_seed_validation
```

Run equal-budget suites:

```bash
PYTHONPATH=src python3 examples/run_experiment_suite.py \
  --source examples/tilelang_matmul_kernel.py \
  --task examples/tilelang_matmul_task.json \
  --output-root results/matmul_policy_suite \
  --promotions-per-round 4

PYTHONPATH=src python3 examples/run_experiment_suite.py \
  --source examples/tilelang_flash_attention_kernel.py \
  --task examples/tilelang_flash_attention_task.json \
  --output-root results/flash_attention_policy_suite \
  --promotions-per-round 4
```

Repeat each suite with fresh output roots. Record hosted model identifier,
sampling temperature, API prices, TileLang/TVM/TileSight revisions, CUDA stack,
GPU model, clocks/power mode, and competing system load.

## Metrics

Report at least:

- held-out final latency and speedup over the freshly measured seed;
- number of generated, modeled, compiled-equivalent, measured, correct, and
  finalized candidates;
- model raw and calibrated ranking error and direction agreement;
- hosted-model logical calls, provider request attempts, input/output tokens,
  and estimated API cost when prices are supplied;
- evaluator wall time by model, measure, profile, and final stage;
- CUDA candidate calls and NCU calls;
- failure and repair categories;
- complete candidate trajectory and lineage graph.

Do not use the sum of CUDA Event latency samples as total GPU experiment time.
It excludes compiler, correctness, warmup, profiler replay, and process costs.

## Interpretation Rules

- Compare strategies under identical configured promotion and profile budgets,
  and report actual calls when invalid or equivalent pools consume less.
- Treat `measure-all` separately because it has a larger hardware budget.
- Use multiple trials because hosted-model sampling and machine noise are not
  fully controlled by the controller random seed.
- Separate search-best timing from fresh final timing.
- Count a failed compiler, correctness, timing, or NCU attempt as consumed cost.
- Inspect `trajectory.csv` before attributing a result to the analytical model;
  proposal quality and source-pool differences are independent factors.
- A kernel that fails any held-out case is not a successful optimization even
  if its public-case latency is lower.

## Current Scope

Phase 0 provides reliable preflight, retries, durable artifacts, progress, and
resume. Phase 1 closes the repair, diagnosis, evidence-memory, and local
calibration loop. Phase 2 provides generic Matmul and Flash Attention workload
plugins, multi-case correctness, robust timing, and a held-out final gate. Phase
3 provides controlled selection/profile policies, compiled-code deduplication,
cost accounting, candidate graphs, and comparable experiment reports.

The current framework is a single-controller sequential reference
implementation. Parallel candidate workers, cross-run retrieval memory,
provider-side deterministic sampling guarantees, and broader kernel families
are future work and should not be implied by Phase 0-3 results.
