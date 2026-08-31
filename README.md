# Compiler-Centric Performance Modeling and Optimization for AI on Modern GPUs

This repository is an MSc Computing Individual Project at Imperial College
London, supervised by Dr. Hongxiang Fan and Zhiwen Mo. It investigates whether
compiler-visible analytical modeling can reduce the hardware cost of iterative
GPU kernel optimization.

The current system accepts a GPU kernel source file as its primary input. A
hosted language-model API proposes complete replacement implementations in an
open-ended code space. Before generation, a compact AI planning request allocates
the current round across parameter, memory, data-movement, execution, pipeline,
decomposition, or newly discovered directions using TileSight, NCU, and prior
strategy outcomes. The controller applies strategy-neutral count, diversity,
and concentration bounds; no direction, including parameter tuning, is
mandatory. A parent-relative AST check then verifies whether each claimed
structural candidate actually changes executable structure.
TileSight evaluates accepted candidates cheaply, while correctness checks, CUDA
Event timing, and selective NCU profiling provide progressively more expensive
hardware evidence.

## Research Question

> Under equal wall-clock, GPU-time, profiling-call, and hosted-model budgets,
> can analytical-model-guided adaptive evaluation find equally fast or faster
> kernels with fewer expensive hardware evaluations?

Each evidence source has a distinct role:

```text
Hosted LLM       proposes complete candidate kernel sources
Strategy planner allocates directions from current evidence and outcomes
Static validator rejects malformed or contract-breaking Python
AST novelty      rejects parameter-only claims in structural lanes
TileSight        predicts latency, resources, utilization, and bottlenecks
Correctness      protects mathematical semantics
CUDA Events      provide the measured latency used to rank the official beam
NCU              provides milestone-level diagnosis and calibration
```

The hosted model and TileSight are proposal and ranking tools. Neither is the
final judge of correctness or measured performance.

## Source-First Interface

The user supplies two inputs:

1. a TileLang Python source file containing the kernel factory;
2. a JSON task contract describing semantics, entrypoint, workload, target,
   evaluation command, and budgets.

The source file is the primary optimization object. The task JSON does not
declare a finite parameter search space. The hosted model may change tiling,
thread and warp mappings, software pipelines, memory layouts, vectorization,
fusion structure, TileLang primitives, and other implementation details.

The input source is never modified. Every generated implementation is stored as
an immutable source candidate, identified by its complete source SHA-256 digest.

## Search Loop

```text
Measured source beam ------------------------------------------+
   |                                                           |
   v                                                           |
Hosted API plans the round's strategy allocation               |
   |  controller normalizes count, diversity, and share        |
   v                                                           |
Hosted API proposes complete replacement source files          |
   |                                                           |
   v                                                           |
Python syntax and task-invariant validation                     |
   |                                                           |
   v                                                           |
Parent-relative AST novelty and strategy validation             |
   |  parameter lane or verified structural transformation     |
   v                                                           |
TileSight model for every valid candidate                       |
   |                                                           |
   v                                                           |
Compiled-code equivalence deduplication                         |
   |                                                           |
   v                                                           |
Adaptive promotion                                             |
   |  model top + low confidence + source diversity + audit    |
   v                                                           |
Compile + reference correctness + CUDA Event measurement       |
   |                                                           |
   +---- incorrect candidate -> archived failure               |
   |                                                           |
   v                                                           |
Measured beam update -------------------------------------------+
   |
   +---- milestone trigger -> NCU -> next-round observed evidence
```

Only correctness-passing candidates with measured latency can enter the beam.
Model-only candidates remain predicted evidence and can never become the final
answer.

The open-ended source space is not enumerated. The hosted model acts as a
proposal policy and samples a small number of promising transformations each
round. The controller uses adaptive-fidelity evaluation to decide where scarce
GPU and NCU calls should be spent. The normalized plan constrains proposal
intent, not implementation syntax or the set of legal transformations. The
model still writes complete source, may combine strategy families, and can name
a previously unrepresented transformation through open structural exploration.

## Wall-Clock Incumbent History

Every optimization run records the best correctness-verified, CUDA-Event-measured
kernel available at fixed wall-clock intervals. The default interval is 300 seconds:

```text
results/<run>/incumbent_history.jsonl   complete nested evidence per snapshot
results/<run>/incumbent_history.csv     flat metrics for time-to-quality plots
```

The recorder runs in a background thread, so a long hosted-API, compiler, CUDA, or
NCU call does not postpone the five-minute sampling decision. It reads only atomic
candidate records already written by the controller and never ranks an unmeasured
TileSight prediction as the incumbent. Each JSONL record includes the candidate and
source identity, measured and final latency, full TileSight model metrics, CUDA
measurement metrics, available NCU profile metrics, diagnosis, speedup over the
seed, search phase, round, beam, and cumulative call/candidate counters. Candidate
source remains in its immutable `candidates/<id>/` directory and is referenced by
path rather than duplicated every five minutes.

Rows with `reason=interval` are the fixed-time observations intended for equal-time
comparisons. Additional event rows mark run start, seed completion, incumbent
changes, final re-ranking, completion, failure, and resume. On resume, sequence
numbers and active elapsed time continue from the existing history; time while the
process is stopped is not counted.

Change the interval explicitly when needed:

```bash
PYTHONPATH=src python3 -m kernel_optimization.cli \
  --source examples/tilelang_matmul_kernel.py \
  --task examples/tilelang_matmul_task.json \
  --output results/matmul_timed_run \
  --incumbent-snapshot-interval-seconds 300
```

Use `--incumbent-snapshot-interval-seconds 0` only when periodic artifacts must be
disabled. Keep the same nonzero interval across compared methods.

## Requirements

Controller machine:

- Python 3.10 or newer;
- network access to an OpenAI-compatible chat-completions endpoint;
- no additional Python runtime dependencies.

GPU evaluator machine:

- TileLang and its bundled TVM;
- TileSight with the TIR interface;
- CUDA and a supported GPU;
- NCU for milestone profiling;
- PyTorch for the included workload reference checks.

Install the controller in editable mode:

```bash
python3 -m pip install -e .
```

Installation is optional. Commands can use `PYTHONPATH=src` instead.

## Included Workloads

The repository contains:

```text
examples/tilelang_matmul_kernel.py             API-editable Matmul seed
examples/tilelang_matmul_task.json             Matmul optimization contract
examples/tilelang_flash_attention_kernel.py    API-editable Flash Attention seed
examples/tilelang_flash_attention_task.json    Flash Attention contract
examples/tilelang_rms_norm_kernel.py           API-editable weighted RMSNorm seed
examples/tilelang_rms_norm_task.json           RMSNorm optimization contract
examples/tilelang_fused_add_rms_norm_kernel.py API-editable fused Add + RMSNorm seed
examples/tilelang_fused_add_rms_norm_task.json fused Add + RMSNorm contract
examples/tilelang_conv2d_kernel.py             API-editable NHWC Conv2D seed
examples/tilelang_conv2d_task.json             Conv2D optimization contract
examples/tilesight_kernel_evaluator.py         shared TileSight/CUDA/NCU runtime
examples/workloads/matmul.py                   immutable Matmul semantics
examples/workloads/flash_attention.py          immutable attention semantics
examples/workloads/rms_norm.py                 immutable weighted RMSNorm semantics
examples/workloads/fused_add_rms_norm.py       immutable fused Add + RMSNorm semantics
examples/workloads/conv2d.py                   immutable NHWC/HWIO Conv2D semantics
```

All five example tasks target an RTX 3090. Matmul uses a 2048 x 2048 x 2048
FP16 primary workload. Flash Attention uses B=1, H=32, S=1024, D=64 FP16 BSHD
input. Weighted RMSNorm and fused Add + RMSNorm use 8192 rows with hidden size
4096. The fused workload returns both the normalized tensor and the FP16
residual sum. Conv2D uses NHWC N=32, H=W=56, C=64 input and an HWIO 3 x 3 x 64
x 128 filter.

The editable seeds are deliberately under-tuned, correctness-first baselines.
They retain a stable Tensor Core or reduction algorithm while exposing ordinary
schedule and data-movement opportunities. They are not intentionally invalid:
an optimization run should improve implementation quality rather than spend its
budget repairing the starting source.

The Matmul seed specifically uses a validated 128 x 128 x 32, three-stage
schedule. On the target RTX 3090 this is intentionally resource-heavy, while
remaining correct and executable, and leaves occupancy and tile-shape
opportunities for the optimizer.

The Flash Attention seed follows the official TileLang BSHD online-softmax
dataflow, including staging each output tile's loop-invariant Q data once before
the K/V loop. Its workload shape and factory interface are adapted for this
system, while the 64 x 64 one-stage schedule remains a conservative starting
configuration.

The fused Add + RMSNorm seed uses one row per CTA, 64 threads, and 128-element
hidden chunks. Its first pass writes `Z = X + residual`; its normalization pass
then deliberately reloads both inputs and recomputes Z. This is valid on an
RTX 3090 but exposes measurable retention, vectorization, reduction-mapping,
and dual-output writeback opportunities without making the seed invalid.

Input shapes are semantic task data rather than a fixed schedule search space.
Change `workload.factory_arguments` in the selected task JSON to set the primary
shape. Each `search_cases` or `final_cases` entry inherits that primary shape
and may override only the dimensions it needs. Keep at least one public shape
and one held-out final shape when testing generalization. Schedule defaults such
as block tiles, stage count, and thread count stay in the Python seed so the
hosted model can modify them as implementation choices. Edit the architecture
and TileLang target together when running on another GPU.

Before spending hosted-API credits, validate each seed through the exact
generic evaluator used by search:

```bash
PYTHONPATH=src python3 examples/validate_seed_kernel.py \
  --source examples/tilelang_matmul_kernel.py \
  --task examples/tilelang_matmul_task.json \
  --output results/matmul_seed_validation

PYTHONPATH=src python3 examples/validate_seed_kernel.py \
  --source examples/tilelang_flash_attention_kernel.py \
  --task examples/tilelang_flash_attention_task.json \
  --output results/flash_attention_seed_validation

PYTHONPATH=src python3 examples/validate_seed_kernel.py \
  --source examples/tilelang_rms_norm_kernel.py \
  --task examples/tilelang_rms_norm_task.json \
  --output results/rms_norm_seed_validation

PYTHONPATH=src python3 examples/validate_seed_kernel.py \
  --source examples/tilelang_fused_add_rms_norm_kernel.py \
  --task examples/tilelang_fused_add_rms_norm_task.json \
  --output results/fused_add_rms_norm_seed_validation

PYTHONPATH=src python3 examples/validate_seed_kernel.py \
  --source examples/tilelang_conv2d_kernel.py \
  --task examples/tilelang_conv2d_task.json \
  --output results/conv2d_seed_validation
```

Add `--profile` to any command for one NCU collection. Without that flag the
smoke run performs TileSight modeling, public multi-case correctness and CUDA
Event timing, then a fresh held-out correctness and robust timing pass.

Each run automatically places TileLang's persistent and temporary compiler
cache under `<output>/.tilelang_cache`. Explicit `TILELANG_CACHE_DIR` and
`TILELANG_TMP_DIR` environment values still take precedence. Storage quota,
profiler-permission, and unavailable-device failures are archived as
infrastructure failures, never sent to source repair, and open a circuit breaker
after two consecutive occurrences.

## Hosted API Configuration

The generator uses an OpenAI-compatible chat-completions endpoint through the
Python standard library. No local model deployment or provider SDK is required.

```bash
export KERNEL_OPT_API_URL="https://provider.example/v1/chat/completions"
export KERNEL_OPT_API_MODEL="provider-model-id"
export KERNEL_OPT_API_KEY="secret"
```

Run source optimization from the repository root:

```bash
PYTHONPATH=src python3 -m kernel_optimization.cli \
  --source examples/tilelang_matmul_kernel.py \
  --task examples/tilelang_matmul_task.json \
  --output results/matmul_api_run
```

Run Flash Attention through the same controller and evaluator by changing only
the source, task, and output paths:

```bash
PYTHONPATH=src python3 -m kernel_optimization.cli \
  --source examples/tilelang_flash_attention_kernel.py \
  --task examples/tilelang_flash_attention_task.json \
  --output results/flash_attention_api_run
```

RMSNorm, fused Add + RMSNorm, and Conv2D use the same command shape:

```bash
PYTHONPATH=src python3 -m kernel_optimization.cli \
  --source examples/tilelang_rms_norm_kernel.py \
  --task examples/tilelang_rms_norm_task.json \
  --output results/rms_norm_api_run

PYTHONPATH=src python3 -m kernel_optimization.cli \
  --source examples/tilelang_fused_add_rms_norm_kernel.py \
  --task examples/tilelang_fused_add_rms_norm_task.json \
  --output results/fused_add_rms_norm_api_run

PYTHONPATH=src python3 -m kernel_optimization.cli \
  --source examples/tilelang_conv2d_kernel.py \
  --task examples/tilelang_conv2d_task.json \
  --output results/conv2d_api_run
```

Candidate generation uses one hosted-model agent by default. Increase the
bounded worker pool when the provider quota can sustain concurrent requests:

```bash
PYTHONPATH=src python3 -m kernel_optimization.cli \
  --source examples/tilelang_matmul_kernel.py \
  --task examples/tilelang_matmul_task.json \
  --output results/matmul_two_agents \
  --agent-workers 2
```

`--agent-workers` controls concurrent source-generation calls, not concurrent
GPU measurements. For six planned slots, two workers receive balanced groups of
three slots, use isolated API clients and prompt contexts, and run at the same
time. Results are merged in deterministic slot order before validation. The
round planner, repair queue, TileSight evaluation, correctness, CUDA Event
timing, and NCU remain under the single controller; in particular, hardware
evaluation is serialized on one GPU to avoid measurement interference.

The default `--agent-workers 1` preserves the original single-request path. A
worker failure is archived in the aggregate API record; successful sibling
results can continue the round, while failure of every worker stops at the
normal resumable checkpoint. Parallel calls repeat some source and evidence
context and reserve provider capacity independently, so they consume more TPM
than one request returning the same number of candidates. Start with two
workers and reduce to one when approaching the provider's token-per-minute
limit. `task.json`, `experiment_manifest.json`, and `experiment_report.md`
record the configured worker count, while `api_calls/*.json` records active workers,
per-worker strategy slots, usage, failures, and exchanges.

The default experiment uses `--selection-policy adaptive`,
`--profile-policy milestone`, `--strategy-allocation-policy ai-planned`,
`--structural-search-policy enforce`, and compiled-code deduplication. To give
promotion ablations the same CUDA
measurement cap per round, add for example:

```bash
--promotions-per-round 4
```

### Orthogonal evidence and evaluator ablations

Three experiment-level controls separate AI guidance, candidate evaluation,
and search-direction allocation. They belong on the CLI rather than in the
task JSON because they change the experimental treatment, not kernel semantics:

```bash
--evaluation-policy tilesight|cuda-event|ncu
--tir-evidence-policy auto|visible|hidden
--strategy-allocation-policy ai-planned|fixed|unconstrained
```

### Related-system style presets

Use one auditable preset when comparing the common harness with the control-flow
style of a related system:

```bash
--baseline-style native|kernelagent|kernelevolve|kernelbench|avo|tilefoundry
```

For example:

```bash
PYTHONPATH=src python3 -m kernel_optimization.cli \
  --source examples/tilelang_matmul_kernel.py \
  --task examples/tilelang_matmul_task.json \
  --baseline-style kernelagent \
  --output results/baselines/matmul_kernelagent
```

The five related-system presets disable TileSight/TIR guidance so they do not
inherit the main system's model. They map the shared controller onto NCU-guided
multi-branch search, persistent evolution, iterative G+E refinement, committed
single-lineage variation, or an external coding-agent-style loop. Task rounds and
proposal counts remain unchanged; single-incumbent styles set `beam_width=1`.
All presets default to one API generation worker; parallelism is enabled only by
an explicit `--agent-workers` override.

These are style emulations in a common TileLang harness, not copies of the
original systems and not claims to reproduce their published results. The exact
mapping, missing capabilities, override rules, and fair-comparison protocol are
documented in [BASELINE_STYLE_PRESETS.md](BASELINE_STYLE_PRESETS.md). Every run
records the preset and any explicit overrides in its manifest and Markdown report.

Run the complete 5 x 6 related-system comparison with one command:

```bash
PYTHONPATH=src python3 examples/run_final_related_system_baselines.py
```

The runner executes 30 treatments sequentially on one GPU: GEMM, RMSNorm,
Conv2D, Flash Attention, and fused Add + RMSNorm times this project's native
full system and the five non-native styles. It uses each preset's canonical
single-agent mode and materializes the checked-in
`examples/related_system_basic_shapes.json` workload family. The fixed
primary/final shapes are GEMM `1024 x 1024 x 1024`, RMSNorm and fused norm
`4096 x 4096`, Conv2D `N16 H56 W56 C64 F128 K3`, and Flash Attention
`B1 H8 S1024 D64`. Base task files are not modified. Results go to
`results/final_eval/related_system_baselines_30m_basic/<kernel>/<style>/`; aggregate
`suite_summary.json` and `suite_summary.md` files are updated after every run.

Each treatment uses exactly the same fixed-time protocol: one agent,
`measurement_repeats=3`, a 1,800-second controller budget, and a 128-round safety
ceiling that is intentionally unreachable during a normal 30-minute hosted-model
run. `--search-until-time-budget` keeps searching after an empty candidate round.
The native system uses TIR-estimated registers during broad TileSight screening,
so it does not invoke PTXAS for every generated candidate. Exact register and
hardware evidence is refreshed by seed and milestone NCU profiles. Each promoted
candidate is compiled once and all CUDA Event timing repeats reuse that compiled
kernel; `measurement_repeats=3` therefore means three measurements, not three
compilations.
At the first safe checkpoint after the deadline, the controller exports the best
verified incumbent and records `termination_reason=time-budget`. The full 30-cell
matrix therefore has a configured search budget of 15 GPU-hours plus bounded
in-flight-call overshoot. A result that ends at the round ceiling or any other
early condition is marked `ended-early`, not silently accepted as fair.

Re-running the command reuses valid completed directories and resumes the first
partial one. Useful controls are:

```bash
# Inspect all 30 commands without API or GPU work.
PYTHONPATH=src python3 examples/run_final_related_system_baselines.py --dry-run

# Debug one matrix cell before starting the full suite.
PYTHONPATH=src python3 examples/run_final_related_system_baselines.py \
  --only-kernel matmul --only-style kernelagent

# Run only the five related-system proxy styles for the old 5 x 5 matrix.
PYTHONPATH=src python3 examples/run_final_related_system_baselines.py \
  --exclude-native

# Reproduce the old round-budget semantics explicitly.
PYTHONPATH=src python3 examples/run_final_related_system_baselines.py \
  --budget-mode rounds
```

The default fail-fast behavior prevents one API or profiler outage from cascading
through the matrix. Add `--continue-on-error` to collect independent failures and
continue; the suite still exits nonzero when any cell failed.

Run the second workload family, including the native full system, with:

```bash
PYTHONPATH=src python3 examples/run_final_related_system_shape_1.py
```

This executes 30 treatments: five kernels times `native` plus the five proxy
styles. The checked-in `examples/related_system_shape_1.json` changes the complete
primary/public/held-out workload family without modifying any base task. Effective
tasks and per-treatment NCU reports are materialized below
`results/final_eval/related_system_baselines_shape_1/`. All cases are distinct
from the first suite. The shape rationale and first-suite findings are documented
in [SHAPE_1_EXPERIMENT.md](SHAPE_1_EXPERIMENT.md).

### Cost and graph bounds

The task budget already bounds candidate-graph growth. The maximum number of
stored candidate nodes is:

```text
1 + rounds * (proposals_per_round + max_repairs_per_round)
```

For example, four rounds with six proposals and two repairs per round can store
at most 33 nodes. Deduplication and failed API returns normally make the graph
smaller.

Add a soft controller wall-clock limit when experiments must stop automatically:

```bash
--max-search-seconds 3600
```

The default is `0`, which disables the limit and preserves previous behavior.
The timer includes seed evaluation, generation, modeling, correctness, timing,
profiling, repairs, and final validation after the controller starts. It does not
kill an API, compiler, CUDA Event, or NCU call in flight; that atomic operation
finishes, then the controller writes a resumable checkpoint and exports the best
correctness-verified incumbent available at that point. Consequently, wall time
can exceed the limit by one in-flight operation. `summary.json` and the Markdown
report record the limit, termination reason, graph upper bound, and whether the
budget was exhausted.

`--evaluation-policy tilesight` preserves the adaptive-fidelity pipeline:
TileSight models every static-valid candidate, the promotion policy chooses a
bounded hardware subset, CUDA Event checks correctness and latency, and NCU is
used according to `--profile-policy`.

`--evaluation-policy cuda-event` never calls `backend.model`. Every
static-valid candidate is compiled, checked, and timed with CUDA Event. The
effective selection policy becomes `measure-all`, NCU becomes `none`, and
compiled-identity deduplication is disabled because that identity is emitted by
the TileSight model stage.

`--evaluation-policy ncu` also skips TileSight and sends every static-valid
candidate through correctness and CUDA Event timing. Every correctness-passing
candidate is then profiled with NCU. Candidate ranking continues to use the
clean CUDA Event latency; NCU supplies feedback rather than replacing the
ranking measurement. This is the highest-cost evaluator baseline.

`--tir-evidence-policy hidden` is a guidance ablation, not an evaluator
ablation. With `evaluation-policy=tilesight`, TileSight can still rank
candidates internally, but successful model predictions, TIR-derived
diagnoses, calibration/trust values, and TIR-derived shared lessons are removed
from AI planning, generation, repair history, and subsequent-round prompts.
Measured CUDA Event outcomes, NCU evidence, and actionable compiler/runtime
failures remain visible. `auto` resolves to `visible` with TileSight and
`hidden` otherwise. Explicit `visible` is rejected when TileSight is disabled.

The following treatments isolate one variable at a time:

```bash
# Main system
--evaluation-policy tilesight --tir-evidence-policy visible \
  --strategy-allocation-policy ai-planned

# Same evaluator and search control, but no TIR/TileSight guidance to the AI
--evaluation-policy tilesight --tir-evidence-policy hidden \
  --strategy-allocation-policy ai-planned

# No TileSight; all candidates use CUDA Event
--evaluation-policy cuda-event --tir-evidence-policy auto \
  --strategy-allocation-policy ai-planned

# No TileSight; all correct candidates also use NCU
--evaluation-policy ncu --tir-evidence-policy auto \
  --strategy-allocation-policy ai-planned

# AI chooses all search directions; retain the same AST validity gate
--evaluation-policy tilesight --tir-evidence-policy visible \
  --strategy-allocation-policy unconstrained
```

Run all four treatments sequentially for Fused Add + RMSNorm with one command:

```bash
PYTHONPATH=src python3 examples/run_final_fused_add_rms_norm_ablation.py \
  --agent-workers 1
```

The default output root is
`results/final_eval/fused_ablation`, with one numbered subdirectory per
treatment plus aggregate `ablation_summary.json` and `ablation_summary.md`
reports. Re-running the command reuses completed treatments and resumes an
interrupted treatment. Use `--dry-run` to inspect all commands, or select a
subset by repeating `--only`, for example:

```bash
PYTHONPATH=src python3 examples/run_final_fused_add_rms_norm_ablation.py \
  --only 01_full_system \
  --only 03_ncu_only
```

For a literal no-novelty-gate AI baseline, additionally set
`--structural-search-policy off`. The recommended strategy ablation leaves it
at `enforce`, so only direction allocation changes and all groups keep the same
no-op/false-claim filter.

Every checkpoint, summary, experiment manifest, Markdown report, and incumbent
snapshot records requested and effective policies. This matters because the
CUDA Event and NCU baselines intentionally override model-based selection,
milestone profiling, and compiled-code deduplication.

Strategy allocation and AST novelty are independent policies. Allocation has
three reproducible modes:

```bash
--strategy-allocation-policy ai-planned    # default evidence-guided allocation
--strategy-allocation-policy fixed         # legacy 1 parameter + 1 open + known lanes
--strategy-allocation-policy unconstrained # one-stage free candidate generation
```

The AI planner receives the current complete kernel, target/workload, bounded
TileSight and NCU evidence, model trust, prior allocations, discovered
strategies, and per-strategy compile/correctness/latency outcomes. Its counts
may assign zero slots to parameter tuning. The controller then enforces exactly
the requested total, at least two directions when the round has multiple slots,
and a two-thirds maximum share for any one direction. Unknown strategy names
are retained as proposed directions and routed through open structural
exploration. Invalid output or a planner/API failure produces an archived,
deterministic evidence-ranked fallback and does not stop candidate generation.

AST novelty has three modes:

```bash
--structural-search-policy enforce  # reject false structural claims before modeling
--structural-search-policy observe  # record AST classifications without rejecting
--structural-search-policy off      # skip parent-relative AST novelty checks
```

In `enforce` mode, changing only existing numeric defaults, thread counts, or
`num_stages` is valid only for the `parameter-tuning` lane. Structural lanes
must change executable AST after docstrings, formatting, local renaming, and
existing tuning values are normalized. The archived candidate metadata records
the assigned strategy, AST hashes, changed tuning values, and structural
signals. This is a novelty gate, not a proof that the transformation is correct
or fast; the existing correctness and hardware stages remain authoritative.

The legacy fixed allocation remains available for ablations. With six proposals
it uses:

```text
1 parameter-tuning baseline
1 open-structural-exploration candidate
4 known structural strategies ranked by current bottleneck evidence
```

The open slot is deliberately not a catch-all parameter candidate. It must name
the proposed transformation in `metadata.discovered_strategy`, explain overlaps
with known lanes in `metadata.related_existing_strategies`, and pass the same AST
structural gate. Measured outcomes from discovered strategies are summarized in
subsequent prompts, allowing later rounds to deepen successful ideas. Strategy
assignments are coverage priors rather than a whitelist, so every candidate may
combine compatible transformations.

Planning uses a separate small output budget, 2,000 tokens by default, while
source generation keeps the larger candidate budget:

```bash
--api-planner-max-output-tokens 2000 --api-max-output-tokens 12000
```

The CLI first sends a very small API preflight request. Authentication, model
access, quota, and endpoint failures are therefore detected before TileLang
compilation, CUDA timing, or seed NCU profiling begins. Progress is flushed to
stderr for every API, model, measurement, profile, and round boundary.

Before each generation or repair call, the controller deterministically
compresses accumulated TileSight, NCU, diagnosis, lesson, and history data into
a bounded key-metric view. Complete source code, task semantics, and the
response contract remain lossless. Full raw evidence stays in the artifact
directory and the prompt records hashes and compression counts for provenance.
The progress stream reports a conservative input-token estimate before network
access. By default, requests estimated above 60,000 input tokens fail locally
instead of consuming an oversized provider request. The budgets can be changed
explicitly:

```bash
--api-max-input-tokens 60000 --api-max-output-tokens 12000
```

The input count is an intentionally conservative dependency-free estimate;
provider usage metadata remains authoritative after a successful call.

### Metadata compression ablation

The default full-system `key-metrics-v1` behavior is unchanged: eight compact
history records, eight compact lessons, no context-filling target, and a 60,000
token local safety guard. The ablation driver opts into a larger compact-record
reservoir and a deterministic budget packer without changing ordinary runs.

A controlled experiment may select either prompt policy directly:

```bash
--metadata-compression-policy key-metrics-v1
--metadata-compression-policy none
```

`none` includes complete accumulated candidate-record metadata, relevant run
lessons, raw TileSight/NCU metadata, and full failure diagnostics. Historical
source files remain excluded. The current parent record is also removed from
history because its source and current evidence are already present in dedicated
fields; this canonical deduplication prevents the first request from containing
the same record twice without compressing earlier candidates. The strategy
planner keeps its separately bounded context in both treatments.

Run both treatments with one command:

```bash
PYTHONPATH=src python3 examples/run_metadata_compression_ablation.py \
  --snapshot-interval-seconds 300 \
  --output results/final_eval/metadata_compression_matmul_v2 \
  -- \
  --max-search-seconds 7200
```

The driver creates one derived eight-round task and runs one agent per treatment.
Its defaults intentionally use different terminal boundaries:

```text
compressed:
  compact history limit       64 records
  compact lesson limit        64 lessons
  compressed context target   60,000 estimated input tokens
  local hard guard            60,000 estimated input tokens

uncompressed:
  full canonical history      grows each round
  local probe guard           1,000,000 estimated input tokens
  expected terminal boundary  provider TPM or context-window rejection
```

The compressed packer removes the oldest compact history and lowest-priority
lessons only after the complete request would exceed its target. Current source,
task semantics, current evidence, and the response schema are never trimmed.
The uncompressed local guard is deliberately high so a provider error such as
`Limit 200000, Requested 201696` can be archived as the experimental endpoint.
The exact provider boundary depends on the model, organization, project, and
current rate-limit tier. This experiment is expected to make one rejected
provider request; that rejected request has no successful-call token usage.

Override the experiment boundaries when needed:

```bash
--compressed-api-max-input-tokens 60000 \
--compressed-context-target-tokens 60000 \
--uncompressed-api-max-input-tokens 1000000 \
--rounds 8
```

The legacy `--api-max-input-tokens` option sets the same local guard for both
treatments. It is retained for reproducibility but should not be used when the
goal is to observe the provider boundary.

The driver defaults to Matmul. Flash Attention has a larger source and often
uses more hosted-model tokens, but source length and repair traffic are fixed
prompt overheads that confound a metadata-compression result. Matmul compiles
reliably, usually completes more rounds, and exposes a cleaner contrast between
bounded compressed history and growing raw candidate metadata. `--source` and
`--task` remain available for a secondary workload.

The API URL, model, and key use the same `KERNEL_OPT_API_*` environment variables
as the main CLI. Arguments after `--` are forwarded to both treatments. The
driver reuses completed treatment directories and supports `--resume` for an
interrupted experiment.

The comparison directory contains:

```text
compressed/                    key-metrics-v1 run artifacts
uncompressed/                  no-compression run artifacts
token_usage_by_call.csv        actual and estimated tokens per API call
token_usage_by_round.csv       provider token usage grouped by search round
token_usage_by_time.csv        round and fixed wall-clock snapshots
token_usage_by_5min.csv        new and cumulative tokens per five-minute window
compression_comparison.json    machine-readable treatment summary
compression_comparison.md      human-readable comparison
```

The Markdown report starts with the five-minute table: `Tokens in window` is new
provider usage since the preceding snapshot, while `Cumulative` is total usage
since that treatment began. It then reports the maximum estimated input request
for each round against the applicable boundary. The expected result is bounded
compressed per-call context near the local target and steeper uncompressed
growth ending at `provider-limit-exceeded`. If a user explicitly lowers the
uncompressed local guard, `local-limit-exceeded` is reported instead.

Transient rate-limit, connection, and server failures use bounded exponential
retry. Quota exhaustion and other permanent client errors fail immediately.
Provider requests and responses are archived without authorization headers or
API-key values.

Resume an interrupted artifact directory without repeating completed stages:

```bash
PYTHONPATH=src python3 -m kernel_optimization.cli \
  --source examples/tilelang_matmul_kernel.py \
  --task examples/tilelang_matmul_task.json \
  --output results/matmul_api_run \
  --resume
```

`--resume` verifies the task and source digest, reconstructs candidate records,
the measured beam, model trust, NCU milestone state, counters, and the active
round, then continues from the latest durable checkpoint. Use
`--skip-api-preflight` only when the endpoint does not support a minimal
chat-completions request.

The CLI has no mock or parameter-search mode. A hosted API and a command
evaluator are required for a production run.

The endpoint must accept an OpenAI-compatible `messages` request and return
`choices[0].message.content`. The response content must be JSON:

```json
{
  "candidates": [
    {
      "hypothesis": "Reduce register pressure by changing the pipeline structure.",
      "source_code": "from tilelang import language as T\n\ndef make_matmul_program(...):\n    ...\n",
      "expected_effect": {
        "latency": "decrease",
        "bottleneck": "register-pressure",
        "reason": "Higher occupancy may improve latency hiding."
      },
      "metadata": {
        "strategy": "pipeline",
        "strategy_slot": "r001-s04-pipeline-structure",
        "strategy_id": "pipeline-structure",
        "discovered_strategy": null,
        "related_existing_strategies": ["data-movement"],
        "changed_regions": ["shared-memory staging loop"]
      }
    }
  ]
}
```

`source_code` is the complete replacement file, not a patch. Returning complete
source avoids ambiguous patch application and makes each candidate independently
reproducible.

All prompts are English. They explicitly separate observed hardware evidence
from analytical predictions and forbid modifications to the evaluator,
reference implementation, workload, and external interface.

## Task Contract

The task JSON defines constraints around the open source space:

```json
{
  "task_id": "tilelang_matmul_2048_rtx3090",
  "description": "Optimize a TileLang FP16 matrix multiplication kernel.",
  "reference": "Compute C = A @ transpose(B).",
  "entrypoint": "make_matmul_program",
  "language": "python",
  "target": {
    "architecture": "rtx3090",
    "tilelang_target": "cuda -arch=sm_86"
  },
  "workload": {
    "factory_arguments": {"m": 2048, "n": 2048, "k": 2048},
    "output_indices": [2]
  },
  "constraints": {
    "preserve_entrypoint": true,
    "max_source_bytes": 200000,
    "required_fragments": ["from tilelang import language as T"],
    "forbidden_fragments": ["subprocess", "os.system"]
  },
  "budget": {},
  "evaluator": {
    "type": "command",
    "command": ["python3", "examples/tilesight_kernel_evaluator.py"],
    "working_directory": "..",
    "timeout_seconds": 1800,
    "environment": {"PYTHONPATH": "../TileSight"},
    "runtime": {
      "plugin": "examples/workloads/matmul.py",
      "measurement_repeats": 2,
      "final_repeats": 5,
      "search_cases": [{"case_id": "primary"}],
      "final_cases": [{
        "case_id": "heldout-rectangular",
        "factory_arguments": {"m": 3072, "n": 1024, "k": 2048}
      }]
    }
  }
}
```

Human configuration controls semantics, safety boundaries, target conditions,
test cases, and cost budgets. It does not enumerate implementation choices.
Only `target`, `workload`, and `constraints` enter generation prompts;
`evaluator.runtime.final_cases` remain held out from the hosted model.

## Evaluator Contract

The controller remains independent from TileLang, TVM, CUDA, TileSight, and NCU
imports. For each fidelity stage it invokes:

```text
<configured command>
  --stage model|measure|profile|final
  --request /temporary/request.json
  --response /temporary/response.json
```

The request contains candidate metadata and a temporary materialized source:

```json
{
  "stage": "model",
  "task": {},
  "candidate": {
    "candidate_id": "...",
    "source_name": "tilelang_matmul_kernel.py",
    "source_sha256": "..."
  },
  "source": {
    "path": "/temporary/tilelang_matmul_kernel.py",
    "filename": "tilelang_matmul_kernel.py",
    "sha256": "..."
}
```

### Model Stage

The adapter should compile or lower the candidate without launching the GPU,
run the TileSight TIR interface, and return:

```json
{
  "valid": true,
  "predicted_latency_ms": 0.18,
  "bottleneck": "tensor-core",
  "confidence": "medium",
  "metrics": {},
  "diagnostics": []
}
```

Expected candidate compilation failures must return `valid=false` with
diagnostics and process exit code zero. Nonzero exit codes are reserved for
evaluator infrastructure failures.

### Measure Stage

The adapter compiles the selected candidate, compares it with an immutable
reference implementation, and uses CUDA Event timing only after correctness
passes:

```json
{
  "correct": true,
  "latency_ms": 0.16,
  "samples_ms": [0.159, 0.160, 0.161],
  "metrics": {"measurement_source": "cuda-events"},
  "error": null
}
```

### Profile Stage

The adapter runs NCU only for a selected milestone candidate and retains both
the original report and exported CSV:

```json
{
  "bottleneck": "register-pressure",
  "valid": true,
  "metrics": {
    "achieved_occupancy": 0.5,
    "tensor_util": 60.0,
    "ddr_util": 35.0
  },
  "report_path": "/absolute/path/to/candidate.ncu-rep",
  "error": null
}
```

A failed NCU attempt returns `valid=false`. It is counted as an attempted
profile call but is not treated as fresh profiling evidence, so a later
milestone can retry.

### Final Stage

After all search rounds, the seed and top measured beam candidates are compiled
again in fresh evaluator processes. The final stage runs every public case,
adds held-out cases that were absent from the API prompt, and collects several
independent primary-case timing samples. It records median, mean, min, max,
standard deviation, median absolute deviation, and coefficient of variation.
The exported source is the lowest-median finalist that passes every case. A
candidate that won the search timing but fails held-out correctness cannot be
exported; the controller falls back to another validated finalist or the seed.

The shared evaluator implements all four stages against TileSight's public TIR
interface and runtime validation helpers. Workload plugins contain only the
immutable reference, output indices, and case validation. Schedule parameters
remain defaults in candidate source, so tile sizes, stages, threads, mappings,
and algorithmic structure stay inside the LLM optimization space. NCU runs in a
child process distinct from TileSight ranking and CUDA Event measurement.

## Adaptive Promotion

The controller maintains:

- a predicted pool containing every statically and model-valid source;
- a measured beam containing only correctness-passing hardware measurements.

Online trust combines mean absolute relative model error and agreement between
predicted and measured optimization directions. Low trust moves real evaluation
toward the configured maximum. High trust moves it toward the minimum.

Each promoted batch combines:

- model-top exploitation;
- a low-confidence audit candidate;
- a source-diverse candidate measured by line-sequence distance;
- deterministic random-audit candidates.

This policy preserves a route to discover candidates that an imperfect model
misranks.

The CLI also exposes `model-top` and deterministic `random` policies for
equal-budget ablations. `measure-all` is a deliberately more expensive upper
baseline and ignores the fixed promotion count. Different source candidates
whose model stage emits the same compiled execution identity are represented in
the trajectory but only the first is eligible for CUDA measurement. The identity
combines generated CUDA source, target, grid, threads, and shared memory. Disable
this conservative cost optimization with `--no-compiled-dedup`.

## Equal-Budget Experiment Suites

Run all three comparable promotion policies for Matmul:

```bash
PYTHONPATH=src python3 examples/run_experiment_suite.py \
  --source examples/tilelang_matmul_kernel.py \
  --task examples/tilelang_matmul_task.json \
  --output-root results/matmul_policy_suite \
  --promotions-per-round 4
```

Run the same protocol for Flash Attention:

```bash
PYTHONPATH=src python3 examples/run_experiment_suite.py \
  --source examples/tilelang_flash_attention_kernel.py \
  --task examples/tilelang_flash_attention_task.json \
  --output-root results/flash_attention_policy_suite \
  --promotions-per-round 4
```

Use `--dry-run` to inspect commands without API or GPU work. The default suite
runs `adaptive,model-top,random`; `--policies` may select a subset or include
`measure-all`. It uses `every-round` NCU so each strategy has the same configured
profile allocation, and holds proposal count, rounds, final validation, and the
per-round promotion cap fixed. Invalid or compiled-equivalent source pools can
consume less than that cap, so reports retain actual call counts. Hosted model
sampling can still produce different source pools, so publish repeated trials
rather than treating one run as a statistically complete comparison.

The suite uses AI-planned strategy allocation by default. Add
`--strategy-allocation-policy fixed` or `--strategy-allocation-policy
unconstrained` to run allocation ablations while keeping the promotion policy
and hardware budget unchanged.

The suite also forwards `--evaluation-policy` and `--tir-evidence-policy`.
When `evaluation-policy` is `cuda-event` or `ncu`, every requested promotion
policy resolves to `measure-all`; use `--policies adaptive` for a single
evaluator-baseline run instead of repeating equivalent selection treatments.

For a no-NCU ablation, add `--profile-policy none`. To study event-triggered NCU,
add `--profile-policy milestone`. Optional
`--api-input-price-per-million` and `--api-output-price-per-million` values add
an estimated hosted-model cost to reports; they never affect selection.

## Feedback, Repair, And Calibration

Failures are evidence rather than discarded rows. Static validation, compiler
or lowering failures, runtime failures, and correctness mismatches are assigned
stable categories with the original concise diagnostics. Each failed proposal
may enter a bounded repair chain controlled by:

```json
{
  "max_repairs_per_round": 2,
  "max_repair_depth": 2
}
```

A repair API call receives the complete failed source, classified failure,
deterministic diagnosis, recent candidate history, and shared run evidence. It
must return one complete replacement source. Repair candidates have explicit
`lineage_kind=repair`, `repair_depth`, and a parent pointing to the failed
candidate. Repair API calls and materialized sources are counted separately in
the summary and remain subject to syntax, compiler, correctness, and timing
gates.

The deterministic diagnosis layer normalizes common TileSight and NCU fields
and reports a category, confidence, supporting evidence, limiting factors, and
actionable recommendations. It recognizes resource-limited occupancy,
register spills, shared-memory bank conflicts, launch underfill, DRAM/L2/shared
memory pressure, Tensor/CUDA/SFU compute pressure, and generally low
utilization. Raw bank-conflict event counts are converted to transaction
amplification with their shared-wavefront denominator before thresholding; they
are never compared directly with a ratio threshold. A raw maximum-utilization
label is still retained in the backend
response, but the API receives the structured interpretation.

Every correctness outcome, failure, and successful NCU milestone creates a
run-level evidence lesson. Lessons include an ID, observed fact, supporting
metrics, tested hypothesis, result, confidence, source regions, and recommended
next action. Relevant lessons are sent to every beam parent, including evidence
from candidates that did not enter that parent lineage. Generated sources are
asked to cite their motivating lesson IDs in `metadata.evidence_ids`.

Latency calibration is deliberately outside TileSight. The controller keeps
the raw analytical latency and optionally adds a calibrated latency for search
ranking. Scale factors use a robust median of measured/raw ratios grouped by
architecture, kernel family, analytical bottleneck, register regime, and
shared-memory regime. Calibration is disabled until the configured minimum
sample count is available, is clamped conservatively, records dispersion, and
downgrades confidence when the regime is unstable. This avoids embedding a
single Matmul or Flash Attention observation into a global TileSight formula.

## NCU Milestones

Under the default `milestone` policy, the input source is profiled once. The
controller then profiles at most one new candidate per round when it observes:

- a meaningful measured improvement;
- disagreement between predicted and measured direction;
- low model confidence;
- stale hardware evidence;
- a search plateau.

CUDA Event timing ranks candidates. NCU explains bottlenecks and informs later
API proposals.

## Artifacts

Every run writes:

```text
<output>/
  task.json
  environment.json
  evidence_memory.json
  events.jsonl
  state.json
  failure.json                  # present after a failed attempt
  summary.json
  experiment_manifest.json
  experiment_report.md
  trajectory.json
  trajectory.csv
  candidate_graph.json
  strategy_plans.json
  best_candidate.json
  best_kernel.py
  api_calls/
    0001_preflight.json
    0002_round-001-plan.json
    0003_round-001-generate.json
  checkpoints/
    00001_round_none_seed_completed.json
    ...
  evaluator_attempts/
    <candidate-id>/
      model_001/{request.json,response.json,stdout.log,attempt.json,...}
  candidates/
    <candidate-id>/
      record.json
      <original-source-name>.py
      history/
        0001_generated.json
        ...
```

Candidate records retain lineage, source hash, optimization hypothesis, model
evidence, raw and calibrated predictions, structured diagnosis, classified
failure context, correctness and timing results, optional NCU evidence, selection
reasons, and final state. Failed and non-promoted sources remain available for
analysis. `task.json` also records the hosted model, endpoint, sampling
temperature, framework version, and API-key environment-variable name, but
never the API key value. Evaluator subprocesses also remove key, token, secret,
password, and credential environment variables before generated code is
loaded. `summary.json` records API call count and accumulates
numeric token-usage fields returned by the provider, alongside model, hardware,
NCU call counts, per-stage wall-clock time, resume status, and total controller
wall-clock time. `events.jsonl`, checkpoints, candidate histories, API calls,
and evaluator attempts form a timestamped audit trail rather than a final-only
summary.

The cost ledger keeps evaluator wall time separate from the sum of observed
kernel latency samples. The latter excludes compilation, correctness, warmup,
process startup, and NCU replay and therefore must not be reported as total GPU
experiment cost. `trajectory.*` provides one row per generated or repaired
candidate, while `candidate_graph.json` preserves proposal and repair edges.
`experiment_report.md` is the concise result intended for inspection, and the
JSON/CSV artifacts support later statistical analysis.

`strategy_plans.json` keeps the raw planner response, normalized slot-level
assignments, controller overrides, fallback reason, planning evidence, and
realized compiler/correctness/performance outcomes for each round. These records
make `ai-planned`, `fixed`, and `unconstrained` allocation directly auditable.

## Offline Tests

Tests use a fake API transport and test-only evaluator classes. They make no
network calls, import no TileLang or TileSight modules, and require no GPU:

```bash
python3 -m unittest discover -s tests -t . -v
```

Coverage includes:

- complete-source API response parsing;
- English prompt and evidence provenance;
- source identity and immutable candidate materialization;
- Python syntax, entrypoint, required-fragment, and forbidden-fragment checks;
- subprocess source-file contract for all four fidelity stages;
- adaptive promotion and source diversity;
- rejection of invalid and incorrect generated kernels;
- measured-beam integrity and `best_kernel.py` export.
- retryable versus permanent hosted-API failures and API preflight;
- secret removal at the evaluator process boundary;
- persistent evaluator request/response/log artifacts;
- interruption recovery without repeating seed model, timing, or NCU work.
- bounded static/compiler/correctness repair with explicit lineage;
- cross-parent evidence propagation and lesson citation fields;
- deterministic bottleneck categories and recommendations;
- regime-local robust calibration with raw prediction preservation.
- one generic evaluator contract shared by Matmul and Flash Attention;
- public multi-case and held-out final correctness;
- repeated timing statistics and fresh-process final selection;
- rejection of a fast search winner when held-out semantics fail.
- equal-budget adaptive, model-top, and random promotion policies;
- optional no-NCU and every-round NCU allocation policies;
- compiled-code equivalence deduplication before hardware measurement;
- API/evaluator/hardware cost accounting and trajectory reports;
- reproducible Matmul and Flash Attention experiment-suite commands.
- small-budget strategy-planner request parsing and English-only prompts;
- zero-parameter AI allocations plus generic diversity/share normalization;
- planner-failure fallback and next-round strategy outcome attribution.

## Security Boundary

API-generated code is untrusted. Static syntax and fragment checks are useful
filters, not a security sandbox. The command evaluator must run compilation and
execution with appropriate process, filesystem, network, GPU, and time limits
for the deployment environment. The editable source must remain separate from
the immutable evaluator and correctness reference.

The research trust invariant is:

> If a prediction is wrong, selection must retain a path to discover the error;
> hardware evidence must influence later rounds; and the final source must
> always be correctness-passing and measured on the target GPU.
