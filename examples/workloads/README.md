# Workload Plugin Contract

The generic evaluator imports the plugin path declared at
`task.evaluator.runtime.plugin`. A plugin owns immutable workload semantics and
must define:

```python
def reference_program(case, task):
    """Return a callable consumed by TileLang profiler.assert_allclose."""

def output_indices(case, task):
    """Return the candidate kernel output argument indices."""
```

A plugin may also define:

```python
def validate_case(case, task): ...
def build_program(candidate_factory, case, task): ...
```

Without `build_program`, the evaluator invokes the task entrypoint with
`case["factory_arguments"]`. Schedule choices should therefore remain defaults
inside candidate source. Cases should pass semantic shape or mode arguments,
not fixed tile, stage, or thread values that would remove those choices from
the optimization space.

The hosted model sees `task.workload` but not `task.evaluator`, so
`runtime.final_cases` act as held-out final checks. This is an evaluation
boundary rather than a hostile-code sandbox: generated code still requires OS
process isolation in adversarial deployments.
