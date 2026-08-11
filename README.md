# Compiler-Centric Performance Modeling and Optimization for AI on Modern GPUs

This repository is an MSc Computing Individual Project at Imperial College
London, supervised by Dr. Hongxiang Fan and Zhiwen Mo. It explores how a
compiler-visible analytical performance model can reduce the cost of iterative
GPU kernel optimization.

The current code is an initial, dependency-free optimization framework. It does
not contain the previous standalone performance model. TileSight is intended to
be connected as an external performance backend on the GPU server.

## Research Goal

The core question is:

> Under equal wall-clock, GPU-time, profiling-call, and language-model budgets,
> can analytical-model-guided adaptive evaluation find equally fast or faster
> kernels with fewer expensive hardware evaluations?

The framework treats each source of evidence differently:

```text
Hosted LLM       proposes optimization hypotheses and candidate edits
TileSight        predicts latency, resources, utilization, and bottlenecks
Correctness      protects mathematical semantics
CUDA Events      provide the measured latency used to rank the official beam
NCU              provides milestone-level hardware diagnosis and calibration
```

Neither the LLM nor the analytical model is trusted as the final judge.

## Current Status

Implemented:

- versioned JSON task specifications;
- immutable content-derived candidate identities;
- deterministic mock and hosted API candidate generators;
- low-cost model, real-measurement, and expensive-profile backend stages;
- adaptive promotion based on online model trust;
- model-top, low-confidence, diversity, and random-audit selection;
- a correctness-gated measured beam;
- event-triggered NCU milestone decisions;
- atomic candidate artifacts, event logs, state snapshots, and final summaries;
- an in-process deterministic mock backend;
- a subprocess JSON backend for TileSight or another compiler runtime;
- a local command-backend example;
- unit tests that require no GPU, network, TileLang, or hosted API.

Not implemented yet:

- the server-side TileSight adapter for a real TileLang workload;
- source-patch application and compilation sandboxing;
- resume from an interrupted state snapshot;
- multi-shape kernel portfolios and guarded dispatch;
- a dashboard or distributed job queue.

## Architecture

```text
TaskSpec
   |
   v
Measured parent beam ------------------------------+
   |                                               |
   v                                               |
Candidate generator                                |
   |  deterministic mock or hosted API             |
   v                                               |
All generated candidates                           |
   |                                               |
   v                                               |
Low-cost performance model                         |
   |                                               |
   v                                               |
Adaptive selection                                 |
   |  model-top + uncertainty + diversity + audit  |
   v                                               |
Compile / correctness / CUDA Event measurement     |
   |                                               |
   +---- incorrect candidates -> archived failure  |
   |                                               |
   v                                               |
Measured beam update -------------------------------+
   |
   +---- milestone trigger -> NCU profile -> next-round evidence
```

Only candidates that pass correctness and receive a measured latency can enter
the official beam. Model-only candidates remain in the predicted pool.

## Requirements

- Python 3.10 or newer;
- no runtime Python dependencies for mock or command modes;
- a GPU/compiler environment only when a real evaluator is connected;
- network access and a hosted API key only for API generation mode.

Install in editable mode:

```bash
python3 -m pip install -e .
```

Installation is optional during development. Commands below can instead use
`PYTHONPATH=src`.

## Local Mock Run

The mock task contains a hidden deterministic performance surface and a
deliberately biased analytical model. It validates the complete optimization
loop without a GPU or API.

```bash
PYTHONPATH=src python3 -m kernel_optimization.cli \
  --task examples/mock_task.json \
  --generator mock \
  --output results/mock_run
```

Expected behavior:

- the initial seed is modeled, checked, measured, and profiled;
- each round generates local schedule mutations;
- all candidates receive a cheap model prediction;
- an adaptive subset receives a correctness check and measured latency;
- only correct measured candidates update the beam;
- milestone candidates receive a mock NCU profile;
- `summary.json` reports a speedup over the seed.

Use a fresh output directory for every run. The CLI refuses to mix a new run
with an existing non-empty artifact directory.

## Subprocess Evaluator Contract

The command backend keeps this controller independent from TileLang, TVM, CUDA,
and TileSight imports. The included example runs the same three-stage protocol
in a separate Python process:

```bash
PYTHONPATH=src python3 -m kernel_optimization.cli \
  --task examples/command_task.json \
  --generator mock \
  --output results/command_run
```

For each stage, the controller invokes:

```text
<configured command>
  --stage model|measure|profile
  --request /temporary/request.json
  --response /temporary/response.json
```

The request contains:

```json
{
  "stage": "model",
  "task": {},
  "candidate": {}
}
```

The response for `model` must contain fields accepted by `ModelEvaluation`:

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

The response for `measure` must contain fields accepted by `Measurement`:

```json
{
  "correct": true,
  "latency_ms": 0.16,
  "samples_ms": [0.159, 0.160, 0.161],
  "metrics": {
    "measurement_source": "cuda-events"
  },
  "error": null
}
```

The response for `profile` must contain fields accepted by
`ProfileEvaluation`:

```json
{
  "bottleneck": "register-pressure",
  "metrics": {
    "achieved_occupancy": 0.5,
    "compute_sol_pct": 60.0,
    "memory_sol_pct": 35.0
  },
  "report_path": "/absolute/path/to/report.ncu-rep"
}
```

The process must exit with code zero and write the response file. Standard
output is captured and included in an exception if the process fails.

## Connecting TileSight on the GPU Server

Implement a server-side adapter with the same command contract. It should:

### `model` stage

1. construct or render the TileLang candidate;
2. call the TileSight TileLang/TIR interface;
3. initially use `collect_ptxas=False` for the cheapest tier;
4. return model latency, utilization, resources, diagnostics, provenance, and
   an overall confidence label;
5. avoid launching the GPU.

### `measure` stage

1. compile the selected candidate and collect ptxas resources;
2. compare it with an immutable reference implementation;
3. return `correct=false` immediately on a numerical failure;
4. otherwise run a stable CUDA Event benchmark and return raw samples plus a
   summary latency.

### `profile` stage

1. run NCU only for the selected milestone candidate;
2. parse the relevant counters into a stable evidence schema;
3. retain the original `.ncu-rep` and exported CSV;
4. return the structured counters and report path.

The adapter can import TileSight and TileLang on the server while the controller
remains import-independent. Replace the `evaluator` section of a task file with:

```json
{
  "type": "command",
  "command": ["python3", "path/to/tilesight_adapter.py"],
  "working_directory": "path/to/server/workspace",
  "timeout_seconds": 600
}
```

## Hosted API Generator

The repository includes an OpenAI-compatible chat-completions adapter implemented
with the Python standard library. No model deployment or additional SDK is
required.

Set the endpoint, model, and API key:

```bash
export KERNEL_OPT_API_URL="https://provider.example/v1/chat/completions"
export KERNEL_OPT_API_MODEL="provider-model-id"
export KERNEL_OPT_API_KEY="secret"
```

Run API generation against the local mock evaluator first:

```bash
PYTHONPATH=src python3 -m kernel_optimization.cli \
  --task examples/mock_task.json \
  --generator api \
  --output results/api_mock_run
```

The endpoint must accept an OpenAI-compatible `messages` request and return
`choices[0].message.content`. The model is instructed to return JSON only:

```json
{
  "candidates": [
    {
      "hypothesis": "Reduce pipeline depth to lower register pressure.",
      "parameter_updates": {
        "num_stages": 2
      },
      "expected_effect": {
        "latency": "decrease",
        "reason": "Higher occupancy may improve latency hiding."
      },
      "source_patch": null,
      "metadata": {
        "strategy": "occupancy"
      }
    }
  ]
}
```

All system and user prompts are written in English. Prompts explicitly label
hardware observations separately from TileSight predictions so the model cannot
mistake an analytical estimate for a measured fact.

The hosted model remains an untrusted candidate generator. Its output is
schema-checked, constrained by the declared search space, deduplicated, and then
sent through the independent evaluator.

## Adaptive Promotion

The controller maintains two frontiers:

- **Predicted pool**: every candidate with a valid low-cost model result;
- **Measured beam**: only correctness-passing candidates with real latency.

Online trust combines:

- mean absolute relative model error;
- agreement between predicted and measured optimization directions.

Low trust increases the number of real evaluations. High trust moves the count
toward the configured minimum. Each promoted batch mixes:

- model-top exploitation;
- a low-confidence audit candidate;
- a parameter-diverse candidate;
- deterministic random-audit candidates.

This prevents an imperfect model from permanently hiding every candidate it
misranks.

## NCU Milestones

The controller profiles at most one new candidate per round. Triggers include:

- the initial seed;
- a meaningful measured improvement;
- a model-versus-measurement direction disagreement;
- a low-confidence measured candidate;
- stale profiling evidence;
- a search plateau.

CUDA Event timing calibrates latency ranking. NCU calibrates the bottleneck
explanation. Numerical correctness remains a separate gate.

## Artifacts

Each run writes:

```text
<output>/
  task.json
  events.jsonl
  state.json
  summary.json
  candidates/
    <candidate-id>.json
```

Candidate records contain lineage, parameters, hypothesis, model evidence,
measurement, optional profile evidence, selection reasons, and decision state.
Failed and non-promoted candidates are retained for analysis.

## Tests

Run the complete local suite:

```bash
python3 -m unittest discover -s tests -t . -v
```

The tests use fake API transport and deterministic evaluators. They make no
network calls and require no GPU.

## Suggested Development Order

1. Validate offline selection policies using existing GEMM and FlashAttention
   sweep reports.
2. Add the real TileSight command adapter for one TileLang GEMM task.
3. Run deterministic online parameter search on the server.
4. Run the hosted API generator against the mock evaluator.
5. Combine hosted API generation with the real TileSight adapter.
6. Add source-patch rendering and compile/correctness repair.
7. Extend from one fixed shape to validated multi-shape kernel portfolios.

## Trust Boundary

The design does not require the language model or TileSight to be perfectly
accurate. It requires failures to remain observable and recoverable:

> If a prediction is wrong, the selection policy must retain a path to discover
> the error; hardware evidence must influence later rounds; and the final best
> candidate must always be correctness-passing and measured on the target GPU.
