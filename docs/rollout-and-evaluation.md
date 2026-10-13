# Rollout and evaluation

## Rollout

```bash
python predict/rollout.py --checkpoint checkpoints/<run>/best_model.pt \
    --data-dir data/<dataset> --split test \
    [--max-sequences N] [--max-steps N] [--no-export-vtu] [--no-one-step] [--device cuda]
```

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
| `rollout_info.json` | traceability: checkpoint (path, SHA-256, modification time, epoch, full training config), dataset (path, SHA-256 of `dataset.db` and `normalizer.json`, split, trajectories, check against the training dataset), options, command line, git commit including uncommitted changes, package versions, status, duration and metrics |
| `<split>_rollout_summary.json` | paper metrics, normalized MSE, RMSE / MAE per target and position RMSE, each per step and per trajectory |
| `rollout.log` | log of the run |
| `<split>/<trajectory>/tNNNN.vtu` | per step: predicted mesh with prediction, ground truth and error as point data (needs pyvista; skip with `--no-export-vtu`) |
| `<split>/<trajectory>/<trajectory>.vtu.series` | opens the whole trajectory in ParaView as a time series (time = rollout step) |

### Visualizing in ParaView

Open `<split>/<trajectory>/<trajectory>.vtu.series`. The mesh points are the predicted positions. The point
data contains `world_pos_gt` (ground-truth positions), `position_error`, and
`<field>_pred` / `<field>_gt` / `<field>_error` for the velocity components and the
stress. Colouring by `position_error` or `stress_error` shows where the prediction
deviates.

## Metrics

### Paper metrics

The rollout computes the metrics of the MGN-T paper (Iparraguirre et al., Tab. 4/5) for
the position q (m) and the von Mises stress σ (Pa). They are stored as `paper_metrics`
in the rollout summary, in `rollout_info.json` and in `test_metrics.json`:

| Metric | Meaning |
|---|---|
| RMSE-1 | one step from the ground-truth state (`--no-one-step` skips it) |
| RMSE-all | over the whole rollout |
| R-RMSE | as above, error divided by the infinity norm of the trajectory (largest \|q\| component or \|σ\|) |

As in the paper, the squared error is averaged over all nodes, the components of q,
all steps and all trajectories before taking the root. `_se` is the standard error over
the trajectories. The log shows the values in the units of the paper's tables
(q ×10⁻³ m, σ ×10³ Pa, R-RMSE in %).

Paper values for `deforming_plate` (400 steps, stride 1): MGN-T q RMSE-all
3.16 ± 0.25 ×10⁻³ m, σ 5.83 ± 0.34 ×10³ Pa; MeshGraphNet q 15.1 ×10⁻³ m. With a
different stride or fewer steps the numbers are only partly comparable.

### Internal metrics

Used for model selection and logged during training: `rollout mse` (normalized MSE) and
`pos rmse` (length of the position error vector per node, free and clamped nodes, mean
of the per-step RMSEs). `pos rmse` is therefore larger than q RMSE-all (among other
things by a factor of ≈ √3, since it uses the vector length instead of the component
mean) and not directly comparable with the paper.

## Comparison with the paper

```bash
python utils/compare_paper.py                      # every finished rollout in rollouts/
python utils/compare_paper.py rollouts/<a> rollouts/<b> --exclude test_traj0038
```

Sets the paper metrics of the rollouts against Tables 4/5 of the MGN-T paper and writes
a new folder `evaluations/<YYYYMMDD_HHMMSS>_paper_comparison/`:

| File | Content |
|---|---|
| `01_absolute_errors.png` | RMSE-1 and RMSE-all of position and stress next to the paper |
| `02_relative_errors.png` | R-RMSE of position and stress |
| `03_error_per_step.png` | error over the rollout steps |
| `04_error_per_trajectory.png` | RMSE-all per trajectory, paper mean as reference line |
| `05_table_metrics.png`, `05b_table_compact.png`, `06_table_conditions.png` | tables as images: all metrics, summary with the multiple of the paper, conditions of each run |
| `comparison.md`, `metrics.csv`, `metrics.json` | report with all tables and figures, raw values |

Rollouts that are still running are listed but not evaluated. `--exclude` adds rows
without the given trajectories (e.g. an outlier), pooled like the paper. The rows with
all trajectories always remain.
