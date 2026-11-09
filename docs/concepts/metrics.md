# Rollout and metrics

A model predicts one time step. In use, it has to carry a whole load case: every
prediction becomes the input of the next step, and errors accumulate. The rollout
measures this, during training on the val split (rollout validation) and afterwards on
the test split ([roll out a model](../guides/rollout.md)).

## The autoregressive rollout

The model is applied to its own predictions, for every trajectory of the split:

1. The initial state is the ground truth of one frame: by default the first frame of the
   dataset, the undeformed state (with a history export together with its initial velocity,
   [optional history](graphs.md#optional-history-history-1-not-in-the-paper)); for a model
   trained on a part of the frames the first of them, and with
   [`--start-frame`](../guides/rollout.md) any other. Every later frame of the trajectory is
   predicted, up to the last.
2. Positions: xₜ₊₁ = xₜ + ẋ (predicted) for the free nodes. Actuator and clamped nodes
   follow their prescribed motion. Free rigid parts move as rigid bodies, by the rigid
   motion closest to their predicted velocities
   ([free rigid parts](graphs.md#free-rigid-parts)).
3. The world-space part of the mesh edges and all world (contact) edges are rebuilt
   from the predicted positions.
4. The model predicts velocity and stress. The stress is an output only and is not
   fed back. With a history export the `*_prev` columns are replaced by the previous
   prediction.

Errors are measured on the nodes in `loss_mask` at every predicted frame. Step t is the
frame t steps after the initial state, with the fields predicted for it.

## Rollout metrics

The rollout computes the metrics of the MGN-T paper (Iparraguirre et al., arXiv v4,
App. A on p. 12) for the position q (m) and the von Mises stress σ (Pa); for data
without a stress field only for q. They are stored as `rollout_metrics` in the rollout
summary, in `rollout_info.json` and in `test_metrics.json`:

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
error over the trajectories. The log shows the units of the paper's tables
(q ×10⁻³ m, σ ×10³ Pa, R-RMSE in %).

## Relative L2 position error

The rollout also computes the relative L2 position error as defined for the crash studies
of Nabian et al. (arXiv:2510.15201) and Akhare et al. (arXiv:2605.27758): per step and
trajectory the norm of the position error over the norm of the ground-truth positions,
both over the nodes in `loss_mask` and their coordinates,

‖x_pred − x_true‖ / ‖x_true‖,

then the mean over the trajectories per step (`position_rel_l2_per_step`) and the mean of
these over the steps (`position_rel_l2`, the value in the tables of these papers). It is
stored in the rollout summary and in `rollout_info.json`; the evaluation report shows it in
units of 10⁻³.

The ground-truth positions are absolute coordinates, so the metric depends on the origin
of the coordinate system and on the size of the structure: the same error in mm gives a
smaller value far from the origin. Values compare only on data in the same coordinates,
with the same steps and nodes. Nodes of an actuator, such as an impactor, are not counted.

## Internal metrics

Used for model selection and logged during training: `rollout mse` (normalized MSE) and
`pos rmse` (length of the position error vector per node, mean of the per-step RMSEs;
by the vector length about √3 times q RMSE-all, not comparable with the paper). The
rollout summary also has the RMSE and MAE of every target column (`rmse_per_field`,
`mae_per_field`), whatever result fields the data has.

`displacement_rms_per_step` in the rollout summary is the RMS displacement of the ground
truth since the initial state (the last exact positions), over the same nodes and with the
same vector norm as `pos rmse`: the position error of a prediction in which nothing moves.
The position error divided by it is the error as a share of the deformation.
