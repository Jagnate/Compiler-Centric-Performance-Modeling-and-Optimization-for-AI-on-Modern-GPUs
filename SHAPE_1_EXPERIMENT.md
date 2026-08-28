# Shape Family 1 Experiment

## Motivation from the First 25 Runs

The first related-system suite used the original primary workloads and produced
the following signals:

- GEMM separated the proxy styles well: final speedups ranged from 1.202x to
  1.344x and every treatment completed four rounds.
- RMSNorm search repeatedly found approximately 0.161 ms candidates from a
  0.231 ms seed, but three styles fell back to the seed after held-out final
  validation. The next family should stress reduction-shape generalization.
- Conv2D remained discriminative, with 1.034x to 1.349x final speedups, although
  NCU-heavy treatments often approached or exhausted the time budget.
- Flash Attention was only 0.14-0.16 ms and improved by at most 1.034x. Four of
  five proxy treatments exhausted 1,800 seconds, so more NCU is not the useful
  fix; the primary kernel needs a stronger timing signal.
- Fused Add + RMSNorm converged to essentially the same 1.51x result in all five
  styles. A different reduction width tests whether that transformation is
  structural or specialized to hidden size 4096.

## Controlled Shape Changes

| Kernel | First primary | Shape-1 primary | Design control |
| --- | --- | --- | --- |
| GEMM | M=N=K=2048 | M=4096, N=1024, K=4096 | Tall 4:1 output, about 2x the original arithmetic |
| RMSNorm | rows=8192, hidden=4096 | rows=16384, hidden=2048 | Same 33.6M elements, half reduction width |
| Conv2D | N32, 56x56, C64, F128, K3 | N32, 28x28, C128, F256, K3 | Same approximate MAC count, channel-heavy regime |
| Flash Attention | B1, H32, S1024, D64, noncausal | B1, H32, S2048, D64, causal | About 2x useful attention-pair work after causal masking |
| Fused Add + RMSNorm | rows=8192, hidden=4096 | rows=16384, hidden=2048 | Same elements as the first primary, different reduction width |

Every primary, public-search, and held-out shape in
`examples/related_system_shape_1.json` is distinct from every corresponding
case in the first suite. RMSNorm and fused Add + RMSNorm intentionally share the
same shape family so their standalone and fused behavior can be compared without
changing tensor size.

## Treatments and Budget

Shape family 1 adds the native full system to the five proxy styles:

```text
5 kernels x (native + kernelagent + kernelevolve + kernelbench + avo + tilefoundry)
= 30 treatments
```

All treatments retain one API worker, four rounds, six proposals per round, up
to two repairs per round, and a 1,800-second soft search cap. The candidate-graph
upper bound is 30 x 33 = 990 nodes. The configured wall-clock upper bound is 15
search hours plus at most one in-flight API/compiler/GPU operation per treatment.

Run the complete experiment from the repository root:

```bash
PYTHONPATH=src python3 examples/run_final_related_system_shape_1.py
```

Results are written to:

```text
results/final_eval/related_system_baselines_shape_1/
```

The runner materializes immutable effective tasks under `_tasks/`, isolates NCU
reports by treatment, updates aggregate `suite_summary.json` and
`suite_summary.md`, reuses completed runs, and resumes a partial treatment.

Use fixed-time rows from each `incumbent_history.csv` for the main time-to-quality
comparison. Final summaries are also useful, but round completion is not a fair
cost unit because NCU-heavy styles spend much more wall time per candidate.
