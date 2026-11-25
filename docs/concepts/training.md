# Training

What a training run does: the two training methods, then what every optimizer step does.
How to start one and what it writes: [train a model](../guides/training.md); every setting:
[configuration reference](../reference/configuration.md).

## Training methods

Both methods train the step from t to t + 1 that the rollout repeats, and every step gets
its contact edges from the geometry it starts from. They differ in the state the model gets
as input, in what the loss covers and in how far the gradients reach. The config chooses
the method ([choose the training method](../guides/training.md#choose-the-training-method));
scheduled sampling and rollout training exclude each other.

| | One-step training | Rollout training |
|---|---|---|
| Unit of the training | one sample: the state at t and its targets at t + 1 | a window of up to `steps` consecutive samples of a trajectory |
| Input | the state of the simulation, with noise; with scheduled sampling partly the model's own prediction | the first step: the state of the simulation, with noise; every further step: the model's prediction of the step before |
| Loss | the one step | every step of the window |
| Gradients | through the one step | through the whole window |
| An epoch | every sample once; samples / `batch_size` optimizer steps | every sample once; windows / `batch_size` optimizer steps (a batch counts windows) |
| GPU memory | one step | about one step (activation checkpointing), plus the inputs of the steps |
| Settings | `[noise]`, optionally `[scheduled_sampling]` | `[rollout_training]`, `[noise]` |
| Data | any export; scheduled sampling needs the history layout | the history layout (`--history 1`) |

### One-step training

The default, and the training of MeshGraphNets and MGN-T: every sample is a graph of the
state at t from the simulation, its targets the velocity to t + 1 and the result fields at
t + 1, and the loss covers this one step.

- **Position noise** (`[noise] world_noise_std`, MeshGraphNets paper: 3·10⁻³): drawn anew
  for every batch and added to the current positions of the free nodes. As in that paper, the
  next state stays unchanged, so the velocity target is corrected by the noise. `velocity_prev`
  (if present) and all edges, including the contact edges, are rebuilt consistently.
  This teaches the model to recover from its own errors during a rollout.
- **Stress noise** (`[noise] stress_noise_std`, history layout only):
  noise on `stress_prev` of the nodes in the loss; the target stays, so the model learns to
  correct a previous value that drifts.
- **Scheduled sampling** (optional, needs the history layout; after Bengio et al., 2015): the
  prediction for the previous sample replaces `*_prev` on a fraction of the nodes, and the
  current positions move consistently. The share of the nodes that keep the ground truth
  falls linearly from `start_ratio` to `end_ratio` over `warmup_epochs` and then stays
  there. With `steps` > 1
  the model rolls out from that many samples back, the predicted state of every step
  feeding the next on all nodes, so that the inputs carry the drift of a short rollout. The
  share then applies to whole samples: a sample takes the rolled-out state on all its nodes
  or keeps the ground truth, so that its drift stays coherent as in a rollout. These
  predictions carry no gradients: the loss still covers the one step. Each further step
  costs a forward pass and a sample more per batch.

### Rollout training

`[rollout_training]` trains on windows of up to `steps` consecutive
samples of a trajectory instead of single samples:

- **Windows:** every epoch each trajectory is cut into windows at a random phase, so that
  every sample is trained once per epoch and takes every position in its window over the
  epochs. A window shorter than `steps` occurs at the start and the end of a trajectory.
- **Steps of a window:** the first starts from the state of the simulation, with the noise
  of one-step training. Every further step starts from the model's prediction of the step
  before: it is written into the `*_prev` inputs, the positions move with it, and all edges,
  the contact edges included, are rebuilt from the predicted geometry, as in the rollout.
- **Loss:** every step of the window counts, every term once over the batch. The velocity
  target of a step, corrected by the displacement of its start, makes its loss the position
  error after the step.
- **Gradients** flow through the whole window, so the model also learns how an error in one
  step affects the steps that follow, not only how to recover from it.
- **Memory:** every step but the last is recomputed in the backward pass (activation
  checkpointing). The GPU memory stays near that of one step, for one forward pass more per
  step.
- **Batch and optimizer steps:** `batch_size` counts windows. A step thus covers about
  `steps` times as many samples as one of one-step training with the same `batch_size`, and
  an epoch has correspondingly fewer optimizer steps.
- **Recorded:** `train_loss` is the loss of the steps from the ground truth, the first of
  every window, comparable to `val_loss`; `train_rollout_loss` the loss over all steps, the
  one that is optimized ([history.json](../reference/file-formats.md#historyjson)).
- `steps = 1` is one-step training.

## Loss and optimizer step

- **Loss:** MSE (or MAE / Huber / smooth L1) on the normalized targets, only on the
  nodes in `loss_mask` (all except the actuator: the nodes of the deformable bodies, over
  which the MGN-T paper averages its loss), without targets a node
  [does not have](graphs.md#general-meshes-and-run-parameters); with rollout training on
  every step of a window ([training methods](#training-methods)).
- **Additional loss terms** (optional, `[loss]`, weight 0 = off), squared errors in units of
  the velocity change per step of the targets, added with their weights:
    - `edge_stretch_weight`: the length change of every mesh edge between two nodes of
      `loss_mask` in the step, prediction against ground truth. A rotation of the edge, as in
      folding, leaves its length unchanged; stretching of the predicted mesh does not.
    - `rigid_body_weight`: the mean velocity of every free rigid body, the translation the
      rollout gives it. A rigid body holds few nodes and so counts little in the loss on the
      nodes, however much of the mass it carries.

  `train_loss` stays the loss on the targets, comparable to `val_loss`; the terms are
  recorded unweighted (`train_edge_stretch_loss`, `train_rigid_body_loss`).
- **Learning rate:** `exponential` decays per optimizer step from `learning_rate` to
  `min_lr` over `decay_steps` (MeshGraphNets paper: 1e-4 → 1e-6 over 5 M steps; MGN-T
  paper: `cosine` from 1e-4 to 1e-6).
  `decay_steps = "auto"` = epochs × steps per epoch, so the LR reaches `min_lr` exactly
  at the end of the run. Alternatives: `plateau`, `cosine`, `none`.
- **Gradient clipping** (`max_grad_norm`).
- **Micro-batches** (`micro_batch_size`, a machine setting): `batch_size /
  micro_batch_size` forward passes make one optimizer step, with the loss normalized
  over the whole batch, so the step equals that of one batch of `batch_size` while only
  one micro-batch is in memory.

## Evaluations

Evaluations during the training measure how the model carries whole trajectories, select
the model by it and can stop the training early. With rollout validation an evaluation runs
every `every_n_epochs` epochs (and in the first and last epoch), else every epoch. An
evaluation records the selection criteria:

| Criterion | |
|---|---|
| `val_loss` | single-step loss on the val split |
| `rollout_mse` | normalized MSE of an autoregressive rollout of the val split |
| `rollout_pos_rmse` | position error of that rollout, mean over the steps |
| `rollout_pos_rmse_final` | position error at the last rollout step |

The rollout criteria need rollout validation. Each evaluation also records the train
loss without noise (on 5 % of the train split, at least 200 samples) and the
generalization gap `max(1, val / train)`, and the rollout validation the length error of the
mesh edges (`rollout_edge_rmse`, mean over the steps) and, with free rigid bodies, their
position and velocity errors at the last step; none of these takes part in selection or
stopping.

## Model selection and early stopping

- **Model selection** (`[training.early_stopping] criteria`): every evaluation is scored
  by its *worst* criterion, the relative excess over that criterion's lowest value; the
  evaluation with the lowest score is selected, close to the best in all criteria at
  once (with one criterion: its minimum). Weights of evaluations that can still be
  selected are kept in `candidates/` until the run ends; then the selected model is
  restored and evaluated on the test split. The `plateau` scheduler follows the first
  criterion.
- **Early stopping** (`enabled`, `patience`, `min_epochs`): after at least `min_epochs`
  epochs, a run stops when no criterion has reached a new low in the last `patience`
  evaluations; the log shows "n/patience evaluations without a new low". With
  `enabled = false` all epochs run and the selection still applies.
- **Limits:** only evaluated epochs are candidates; on a small val split the criteria
  scatter by several % between evaluations, so the selected model need not be the best
  on the test split. Adapt criteria, `patience` and `min_epochs` to how much they
  scatter on your data.
