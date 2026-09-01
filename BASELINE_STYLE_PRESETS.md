# Related-System Style Presets

## Scope

`--baseline-style` provides controlled, auditable approximations of five related
kernel-generation workflows. A preset reuses this repository's TileLang source
contract, hosted model, correctness evaluator, CUDA measurement implementation,
artifact store, and controller. It changes only the search protocol and feedback
policy.

These runs are **style emulations**, not reimplementations. They cannot reproduce
the original papers' published results because their task corpora, prompts,
knowledge bases, source languages, profilers, runtimes, and orchestration code are
not copied here. Reports preserve this distinction explicitly.

## Preset Matrix

| Style | Search topology | Candidate feedback | Direction control | AST policy | Default API workers |
| --- | --- | --- | --- | --- | ---: |
| `native` | Task beam | TileSight for all, bounded CUDA Event, milestone NCU | AI-planned | enforce | 1 |
| `kernelagent` | Task beam | CUDA Event + NCU for every correct candidate | AI-planned hardware branches | observe | 1 |
| `kernelevolve` | Persistent task beam and shared evidence memory | CUDA Event + NCU for every correct candidate | unconstrained variation | observe | 1 |
| `kernelbench` | One incumbent, best-of-k iterative refinement | compiler, correctness, CUDA Event | unconstrained G+E refinement | off | 1 |
| `avo` | One committed incumbent, best-of-k variation | compiler, correctness, CUDA Event + NCU | unconstrained plan/edit variation | off | 1 |
| `tilefoundry` | One incumbent controlled by an external-agent-style loop | semantic checks + CUDA Event | unconstrained whole-source rewrite | observe | 1 |

All presets preserve `rounds` and `proposals_per_round` from the task JSON. This
keeps the model proposal budget comparable. The three single-incumbent styles set
only `beam_width=1`; the two population styles retain the task's beam width.

All five related-system presets hide TIR evidence and disable TileSight. This is
intentional: otherwise a baseline would silently inherit the main contribution.
The `native` style is the only canonical preset that uses TileSight/TIR.

## Why Each Mapping Exists

### Measured-Archive Native Cell

The final related-system runner intentionally overrides the canonical `native`
preset for its `native` output cell. That treatment uses CUDA Event latency as
the only search reward, extracts bounded source-level TIR only for next-round
parents, allocates strategies from measured reward plus uncertainty, and submits
all archive parents in one generation request. The fastest candidate is always
retained; remaining positions prefer competitive strategy and AST/TIR diversity.
One unrestricted slot keeps the strategy vocabulary open-ended. This override is
scoped to `run_final_related_system_baselines.py`; ordinary CLI runs and other
experiment scripts keep the TileSight-guided canonical native preset.

### KernelAgent

The proxy retains the recognizable hardware-guided orchestration loop: measured
hardware profiles, an AI direction planner, multiple optimization branches, and a
beam of viable parents. One API agent generates those branches sequentially by
default to bound provider cost. NCU is collected for every correctness-passing
candidate because the local controller does not implement KernelAgent's
specialized profile selection and rule database.

Missing pieces include its Triton-specific agents, diagnosis rules, databases,
reflection protocol, and exact beam scorer.

### KernelEvolve

The proxy uses the existing persistent candidate graph, multiple beam parents,
failure history, and global evidence lessons as evolutionary memory. Candidate
generation is unconstrained so the model acts as a universal variation operator.
NCU is the available target-hardware oracle.

Missing pieces include external Context Memory, Deep Search/RAG, the original
parent-selection policy, cross-task warm starts, and production dispatch logic.

### KernelBench

KernelBench is primarily a benchmark and evaluator, not one autonomous search
algorithm. This preset represents its iterative **G+E** treatment: the model sees
the previous kernel plus compiler, correctness, and execution feedback. A
single incumbent receives best-of-k refinements each round, and every candidate
is timed with CUDA Event.

It does not reproduce the PyTorch `ModelNew` task corpus, the paper's exact
ten-step protocol, or its distributed precompile/evaluation infrastructure.

### AVO

The proxy represents one committed lineage. Each round proposes variations around
the current measured incumbent; correctness and latency gates prevent a worse
candidate from replacing the best, while failures remain in search memory. NCU
provides the closest available profile-tool feedback.

It does not reproduce AVO's interactive coding-agent tool loop, supervisor,
knowledge base, git workflow, or exact stagnation policy.

### TileFoundry

TileFoundry is an agent-facing compiler/workbench; its external coding agent
performs the real search. The proxy therefore gives one incumbent to an
unconstrained whole-source rewrite loop, preserves a fixed semantic contract,
and uses deterministic correctness plus CUDA Event timing without an internal
profiler.

It does not provide authored HIR, runtime twins, real model weights, model-level
fusion, TileFoundry's analyze/schedule tools, or end-to-end decoding.

## Running One Style

Use the same source, task, model, GPU, and measurement settings for every run;
change only the style and output directory:

```bash
PYTHONPATH=src python3 -m kernel_optimization.cli \
  --source examples/tilelang_matmul_kernel.py \
  --task examples/tilelang_matmul_task.json \
  --baseline-style kernelagent \
  --output results/baselines/matmul_kernelagent
```

Valid names are:

```text
native
kernelagent
kernelevolve
kernelbench
avo
tilefoundry
```

Replace the source, task, style, and output path for another kernel or treatment.
Every preset defaults to one API generation worker. Parallel generation remains
available only when the user explicitly passes `--agent-workers N`; the report
marks that as a noncanonical override. Any explicitly supplied style-controlled
policy similarly overrides the preset and is recorded.

## Running the Full Matrix

The repository provides a final-evaluation runner for the measured-archive
treatment and five related-system styles across all five existing kernel tasks:

```bash
PYTHONPATH=src python3 examples/run_final_related_system_baselines.py
```

This is a 30-treatment, single-agent, sequential-GPU suite. It materializes one
derived task per cell without modifying the original task files. The default
basic-shape output layout is
`results/final_eval/related_system_baselines_15m_basic_2048_measured_archive/<kernel>/<style>/`,
plus aggregate JSON and Markdown reports. A 900-second search cap applies
independently to each treatment. The same command is resumable: compatible
completed summaries are reused and a nonempty partial treatment receives
`--resume`.

Use `--dry-run` to audit commands, or repeat `--only-kernel` and `--only-style`
to select cells. The five related styles remain canonical single-agent presets.
Only the `native` cell receives the explicit measured-archive controls described
above, and its manifest records those overrides as noncanonical.

## What Is Recorded

Every run stores the following in `task.json`, `summary.json`,
`experiment_manifest.json`, and `experiment_report.md` as applicable:

- requested style, human-readable label, and `style-emulation` status;
- canonical preset versus explicit overrides;
- original and effective task budgets;
- requested/effective evaluator, TIR, selection, profiling, deduplication,
  structural-search, and strategy-allocation policies;
- default or overridden agent-worker count;
- unsupported original-system capabilities;
- candidate graph, strategy plans, measurements, profiles, API cost, GPU time,
  wall time, and periodic incumbent history.

The compact style protocol is also included in English generation, planning, and
repair prompts. Unsupported-capability disclaimers stay in artifacts rather than
the model prompt.

## Comparison Protocol

For a proposal-budget comparison, keep task `rounds`, `proposals_per_round`, API
model, temperature, source, shape, correctness cases, GPU, and measurement repeats
fixed. Report both best latency and total search cost.

For a time-budget comparison, use `incumbent_history.csv` at common wall-clock
cutoffs. NCU-heavy styles intentionally spend far more GPU time per candidate.
The runner's measured-archive cell measures every candidate with CUDA Event but
avoids per-candidate NCU, TileSight modeling, and a separate hosted planning
request. Comparing only the same number of rounds would hide that systems-level
difference.

To stop each treatment automatically at the same controller budget, add for
example `--max-search-seconds 3600`. The limit is checked before expensive stages
and between candidates. One already-running API/compiler/GPU operation is allowed
to finish so artifacts remain consistent. All presets use one API agent unless an
explicit `--agent-workers` override is supplied.

The checked-in `run_final_related_system_baselines.py` protocol additionally uses
`--search-until-time-budget` and a deliberately high safety round ceiling. This
prevents an empty generation round or a short task round budget from ending one
style before the common 900-second cutoff. Resumed runs count previously
checkpointed search time, and only `termination_reason=time-budget` is accepted as
a valid equal-time result. The suite fixes `measurement_repeats=3` and uses one
basic shape per kernel for every style.

Do not label these numbers as "KernelAgent results" or "AVO results." Use names
such as **KernelAgent-style proxy in our common harness**.
