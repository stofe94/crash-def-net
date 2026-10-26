# Rollout and evaluation

## Rollout

```bash
python predict/rollout.py --checkpoint checkpoints/<run>/best_model.pt \
    --data-dir data/<dataset> --split test \
    [--max-sequences N] [--max-steps N] [--no-export-vtu] [--no-one-step] [--device cuda]
    [--lazy | --no-lazy]
```

`--lazy` reads the frames from `dataset.db` on access instead of loading the split
(default: as in the training). The normalizer comes from the checkpoint; checkpoints
written before it was stored there use `normalizer.json` of `--data-dir`.

The model is applied autoregressively to its own predictions, for every trajectory of
the split:

1. Step 0 takes the ground truth as the initial state.
2. Positions: xₜ₊₁ = xₜ + ẋ (predicted) for the free nodes. Actuator and clamped nodes
   follow their prescribed motion.
3. The world-space part of the mesh edges and all world (contact) edges are rebuilt
   from the predicted positions.
4. The model predicts velocity and stress. The stress is an output only and is not
   fed back. With a history export the `*_prev` columns are replaced by the previous
   prediction.

Errors are measured on the nodes in `loss_mask` from step 1 on. The rollout checks that
the dataset has the same feature layout as the model, and warns if its export settings
(e.g. the stride) differ from those of the training dataset.

### Output

Every call writes into a folder of its own and never overwrites an earlier rollout:
`rollouts/<YYYYMMDD_HHMMSS>_<training run>_<checkpoint>_<split>/`
(`best_model.pt` appears as `best_model_ep<epoch>`). A folder given with
`--output-dir` must be new or empty.

| File | Content |
|---|---|
| `rollout_info.json` | traceability: checkpoint (path, SHA-256, epoch, training config and its source), dataset (path, content fingerprint, normalizer, split, check against the training dataset), options, command line, git commit, package versions, status, duration, metrics |
| `<split>_rollout_summary.json` | paper metrics, normalized MSE, RMSE / MAE per target, position RMSE and RMS displacement of the ground truth ([internal metrics](#internal-metrics)), per step, for every trajectory and as their mean |
| `rollout.log` | log of the run |
| `<split>/<trajectory>/tNNNN.vtu` | per step: predicted mesh with prediction, ground truth and error as point data (needs pyvista; skip with `--no-export-vtu`) |
| `<split>/<trajectory>/<trajectory>.vtu.series` | opens the whole trajectory in ParaView as a time series (time = rollout step) |

### Visualizing in ParaView

Open `<split>/<trajectory>/<trajectory>.vtu.series`. The mesh points are the predicted positions. The point
data contains `world_pos_gt` (ground-truth positions), `position_error`, and
`<field>_pred` / `<field>_gt` / `<field>_error` for the velocity components and the
stress. Colouring by `position_error` or `stress_error` shows where the prediction
deviates.

### Animation

```bash
python utils/animate_rollout.py rollouts/<a> [rollouts/<b> ...] --trajectory test_traj0136
```

Renders one trajectory as an animated GIF: the ground truth (FEM) next to the prediction
of every given rollout, coloured by von Mises stress on one scale, the actuator in grey
(node types from the rollout's `dataset.db`). The rollouts need the `.vtu` export and
must come from the same dataset with the same steps; the first one also provides the
ground truth. Writes `rollout_<trajectory>.gif` into a new folder
`evaluations/<YYYYMMDD_HHMMSS>_animation/`, or to the file given with `--output`.

| Option | Default | |
|---|---|---|
| `--trajectory` | first trajectory of the split | trajectory to render |
| `--labels` | architecture and parameter count | panel titles, one per rollout |
| `--stress-max` | 99.9th percentile of the ground-truth stress, rounded up to 50 kPa | upper end of the colour scale in kPa |
| `--output` | new folder in `evaluations/` | GIF file |

Rendering runs off-screen in software (Mesa), so it needs neither a GPU nor a display
and gives the same frames on every machine; 98 steps with three panels take about 20 s.
Needs `pyvista`.

## Metrics

### Paper metrics

The rollout computes the metrics of the MGN-T paper (Iparraguirre et al., arXiv v4, App. A on p. 12) for
the position q (m) and the von Mises stress σ (Pa). They are stored as `paper_metrics`
in the rollout summary, in `rollout_info.json` and in `test_metrics.json`:

| Metric | Meaning |
|---|---|
| RMSE-1 | one step from the ground-truth state (`--no-one-step` skips it) |
| RMSE-all | over the whole rollout |
| R-RMSE | as above, error divided by the infinity norm of the trajectory (largest \|q\| component or \|σ\|) |

As in the paper, the squared error is averaged over all nodes, components, steps and
trajectories before taking the root: one RMSE over all trajectories together
(**pooled**). A trajectory with a large error weighs more than in the median: RMSEs of
2, 2 and 8 mm on three equal trajectories give √((2² + 2² + 8²) / 3) = 4.9 mm pooled and
2.0 mm as median. Median, best and worst first take one RMSE per trajectory, then the
median, the smallest and the largest of them: here 2.0, 2 and 8 mm. `_se` is the standard
error over the trajectories. The log shows the
units of the paper's tables (q ×10⁻³ m, σ ×10³ Pa, R-RMSE in %).

### Internal metrics

Used for model selection and logged during training: `rollout mse` (normalized MSE) and
`pos rmse` (length of the position error vector per node, mean of the per-step RMSEs;
by the vector length about √3 times q RMSE-all, not comparable with the paper).

`displacement_rms_per_step` in the rollout summary is the RMS displacement of the ground
truth since step 1, the last exact positions, with the nodes and norm of `pos rmse`: the
position error of a prediction in which nothing moves. Position error over displacement
is the error as a share of the deformation.

## Evaluating a run

```bash
python utils/evaluate_run.py rollouts/<rollout>     # the rollout and its training run
python utils/evaluate_run.py checkpoints/<run>      # without a rollout: test_metrics.json
```

Evaluates one run on its own, without the paper: the training, the single-step test
errors and the rollout with its spread over the trajectories. Writes a new folder
`evaluations/<YYYYMMDD_HHMMSS>_<run>_evaluation/`, or into the new or empty folder given
with `--output-dir`:

| File | Content |
|---|---|
| `01_training.png` | losses, val rollout, generalization gap over the epochs, selected epoch |
| `02_error_per_step.png` | error over the rollout: mean, best and worst trajectory, RMS displacement of the ground truth (rollout folder only) |
| `03_trajectories.png` | every metric pooled, as median, best and worst trajectory |
| `evaluation.md` | run with peak memory, findings, tables of training, single-step errors, rollout metrics and every trajectory |
| `trajectories.csv`, `evaluation.json` | the values, in SI units |

The findings are measurements, not verdicts: end of the training, scatter of the
selection criterion, generalization gap, single-step test error against the val loss,
pooled error against the median and trajectories far above it, error growth, error
relative to the displacement. A checkpoint folder alone uses `test_metrics.json`
(paper metrics per trajectory, no curves per step).

## Comparison with the paper

```bash
python utils/compare_paper.py                      # every finished rollout in rollouts/
python utils/compare_paper.py rollouts/<a> rollouts/<b> --exclude test_traj0149
```

Sets the paper metrics of the rollouts against MGN-T (arXiv v4, Tab. 3 on p. 7) and
MeshGraphNets (Pfaff et al. 2021, Tab. 1 on p. 7, with the standard error over the 100
test trajectories; the MGN-T paper repeats these values without it in Tab. 2) and writes
a new folder `evaluations/<YYYYMMDD_HHMMSS>_paper_comparison/`:

| File | Content |
|---|---|
| `01_absolute_errors.png` | RMSE-1 and RMSE-all of position and stress next to the paper |
| `02_relative_errors.png` | R-RMSE of position and stress |
| `03_error_per_step.png` | error over the rollout steps |
| `04_error_per_trajectory.png` | RMSE-all over the trajectories of every rollout: all pooled, median, best and worst trajectory; paper mean as reference line |
| `05_table_metrics.png`, `05b_table_compact.png`, `06_table_conditions.png` | tables as images: all metrics, summary with the multiple of the paper, conditions of each run |
| `comparison.md`, `metrics.csv`, `metrics.json` | report with all tables and figures, raw values |

Rollouts that are still running are listed but not evaluated. `--exclude` adds rows
without the given trajectories (e.g. an outlier), pooled like the paper. The rows with
all trajectories always remain.
