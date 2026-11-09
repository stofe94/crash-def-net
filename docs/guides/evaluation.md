# Evaluate and compare

Three tools read finished runs and rollouts and write a new folder in `evaluations/`;
none of them overwrites an earlier evaluation:

| Tool | Question it answers |
|---|---|
| `utils/evaluate_run.py` | How did this one run train, and how does it do over the test geometries? |
| `utils/compare_paper.py` | How do the rollouts compare with the published values of the papers? |
| `utils/check_protocol.py` | Were two runs trained under the comparison protocol, and which one is better per trajectory? |

The metrics they report: [rollout and metrics](../concepts/metrics.md).

## Evaluate a run

```bash
python utils/evaluate_run.py rollouts/<rollout>     # the rollout and its training run
python utils/evaluate_run.py checkpoints/<run>      # without a rollout: test_metrics.json
```

Evaluates one run on its own, without the paper: the training, the single-step test
errors and the rollout with its spread over the trajectories. A rollout can also start the
evaluation itself ([rollout in a config file](rollout.md#define-a-rollout-in-a-config-file)).
Writes a new folder `evaluations/<YYYYMMDD_HHMMSS>_<run>_evaluation/`, or into the new or
empty folder given with `--output-dir`:

| File | Content |
|---|---|
| `01_training.png` | losses, val rollout, generalization gap over the epochs, selected epoch |
| `02_error_per_step.png` | error over the raw frames of the run, from the first predicted frame on (grey: before it): mean, best and worst trajectory, RMS displacement of the ground truth (rollout folder only) |
| `03_trajectories.png` | every metric pooled, as median, best and worst trajectory |
| `evaluation.md` | run with peak memory, findings, tables of training, single-step errors, rollout metrics and every trajectory |
| `trajectories.csv`, `evaluation.json` | the values, in SI units |

The findings are measurements, not verdicts: end of the training, scatter of the
selection criterion, generalization gap, single-step test error against the val loss,
pooled error against the median and trajectories far above it, error growth, error
relative to the displacement. A checkpoint folder alone uses `test_metrics.json`
(rollout metrics per trajectory, no curves per step).

## Compare with the papers

```bash
python utils/compare_paper.py                      # every finished rollout in rollouts/
python utils/compare_paper.py rollouts/<a> rollouts/<b> --exclude test_traj0149
```

Compares the rollout metrics of the rollouts with those of MGN-T (arXiv v4, Tab. 3 on
p. 7) and MeshGraphNets (Pfaff et al. 2021, Tab. 1 on p. 7, with the standard error over the 100
test trajectories; the MGN-T paper repeats these values without it in Tab. 2) and writes
a new folder `evaluations/<YYYYMMDD_HHMMSS>_paper_comparison/`:

| File | Content |
|---|---|
| `01_absolute_errors.png` | RMSE-1 and RMSE-all of position and stress next to the paper |
| `02_relative_errors.png` | R-RMSE of position and stress |
| `03_error_per_step.png` | error over the raw frames of the run, from the first predicted frame on (grey: before it) |
| `04_error_per_trajectory.png` | RMSE-all over the trajectories of every rollout: all pooled, median, best and worst trajectory; paper mean as reference line |
| `05_table_metrics.png`, `05b_table_compact.png`, `06_table_conditions.png` | tables as images: all metrics, summary with the multiple of the paper, conditions of each run |
| `comparison.md`, `metrics.csv`, `metrics.json` | report with all tables and figures, raw values |

Rollouts that are still running are listed but not evaluated. `--exclude` adds rows
without the given trajectories (e.g. an outlier), pooled like the paper. The rows with
all trajectories always remain. The numbers of a small run are not comparable with the
papers ([why](../results/deforming-plate.md#why-the-numbers-are-not-comparable-with-the-paper)).

## Check the comparison protocol

```bash
python utils/check_protocol.py rollouts/<baseline rollout> rollouts/<new rollout>
```

Checks every rollout against the protocol config (dataset fingerprint, seed, budget,
recipe; only the declared variations may differ) and compares the runs with the first
one, the reference, trajectory by trajectory. How to run a comparison and read the
result: [comparison protocol](../guides/comparison-protocol.md).
