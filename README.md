# Compiler-Centric Performance Modeling and Optimization for AI on Modern GPUs

This repository is an MSc Computing Individual Project at Imperial College
London, supervised by Dr. Hongxiang Fan and Zhiwen Mo. It investigates whether
compiler-visible analytical modeling can reduce the hardware cost of iterative
GPU kernel optimization.

The current system accepts a GPU kernel source file as its primary input. A
hosted language-model API proposes complete replacement implementations in an
implicit, open-ended code space. TileSight evaluates all statically valid
candidates cheaply, while correctness checks, CUDA Event timing, and selective
NCU profiling provide progressively more expensive hardware evidence.

## Research Question

> Under equal wall-clock, GPU-time, profiling-call, and hosted-model budgets,
> can analytical-model-guided adaptive evaluation find equally fast or faster
> kernels with fewer expensive hardware evaluations?

Each evidence source has a distinct role:

```text
Hosted LLM       proposes complete candidate kernel sources
Static validator rejects malformed or contract-breaking Python
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
Hosted API proposes complete replacement source files          |
   |                                                           |
   v                                                           |
Python syntax and task-invariant validation                     |
   |                                                           |
   v                                                           |
TileSight model for every valid candidate                       |
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
GPU and NCU calls should be spent.

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
- PyTorch for the included matmul reference check.

Install the controller in editable mode:

```bash
python3 -m pip install -e .
```

Installation is optional. Commands can use `PYTHONPATH=src` instead.

## Included Matmul Example

The repository contains:

```text
examples/tilelang_matmul_kernel.py   API-editable input kernel
examples/tilelang_matmul_task.json   immutable optimization contract
examples/tilesight_matmul_adapter.py TileSight/CUDA/NCU evaluator
```

The task targets an RTX 3090 with a 2048 x 2048 x 2048 FP16 matmul workload.
Edit the target and workload fields when running on another GPU or shape.

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

The CLI first sends a very small API preflight request. Authentication, model
access, quota, and endpoint failures are therefore detected before TileLang
compilation, CUDA timing, or seed NCU profiling begins. Progress is flushed to
stderr for every API, model, measurement, profile, and round boundary.

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
    "command": ["python3", "examples/tilesight_matmul_adapter.py"],
    "working_directory": "..",
    "timeout_seconds": 1800,
    "environment": {"PYTHONPATH": "../TileSight"}
  }
}
```

Human configuration controls semantics, safety boundaries, target conditions,
and cost budgets. It does not enumerate implementation choices.

## Evaluator Contract

The controller remains independent from TileLang, TVM, CUDA, TileSight, and NCU
imports. For each fidelity stage it invokes:

```text
<configured command>
  --stage model|measure|profile
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

The included matmul adapter implements all three stages against TileSight's TIR
interface and runtime validation helpers. It runs NCU in a child process so the
profiled launch is distinct from model ranking and CUDA Event measurement.

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
utilization. A raw maximum-utilization label is still retained in the backend
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

The input source is profiled once. The controller then profiles at most one new
candidate per round when it observes:

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
  best_candidate.json
  best_kernel.py
  api_calls/
    0001_preflight.json
    0002_round-001-generate.json
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
- subprocess source-file contract for all three fidelity stages;
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
