# Training

What a training run does, in the order of a step. How to start one and what it writes:
[train a model](../guides/training.md); every setting:
[configuration reference](../reference/configuration.md).

## Loss and optimizer step

- **Loss:** MSE (or MAE / Huber / smooth L1) on the normalized targets, only on the
  nodes in `loss_mask` (all except the actuator), without targets a node
  [does not have](graphs.md#general-meshes-and-run-parameters).
- **Position noise** (`[noise] world_noise_std`, paper: 3·10⁻³): drawn anew for every
  batch and added to the current positions of the free nodes. As in the paper, the next
  state stays unchanged, so the velocity target is corrected by the noise. `velocity_prev`
  (if present) and all edges, including the contact edges, are rebuilt consistently.
  This teaches the model to recover from its own errors during a rollout.
- **Learning rate:** `exponential` decays per optimizer step from `learning_rate` to
  `min_lr` over `decay_steps` (paper: 1e-4 → 1e-6 over 5 M steps).
  `decay_steps = "auto"` = epochs × steps per epoch, so the LR reaches `min_lr` exactly
  at the end of the run. Alternatives: `plateau`, `cosine`, `none`.
- **Gradient clipping** (`max_grad_norm`, not in the paper).
- **Micro-batches** (`micro_batch_size`, a machine setting): `batch_size /
  micro_batch_size` forward passes make one optimizer step, with the loss normalized
  over the whole batch, so the step equals that of one batch of `batch_size` while only
  one micro-batch is in memory.
- **Scheduled sampling** (optional, needs the history layout): the prediction for the
  previous sample replaces `*_prev` on a fraction of the nodes, and the current positions
  move consistently. The share of the nodes that keep the ground truth falls linearly from
  `start_ratio` to `end_ratio` over `warmup_epochs` and then stays there.

## Evaluations

With rollout validation an evaluation runs every `every_n_epochs` epochs (and in the
first and last epoch), else every epoch. An evaluation records the selection criteria:

| Criterion | |
|---|---|
| `val_loss` | single-step loss on the val split |
| `rollout_mse` | normalized MSE of an autoregressive rollout of the val split |
| `rollout_pos_rmse` | position error of that rollout, mean over the steps |
| `rollout_pos_rmse_final` | position error at the last rollout step |

The rollout criteria need rollout validation. Each evaluation also records the train
loss without noise (on 5 % of the train split, at least 200 samples) and the
generalization gap `max(1, val / train)`; neither takes part in selection or stopping.

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

## Deviations from the paper

Deliberate differences (quick test and extensions):

- Quick test (`examples/quick_start/train.toml`): stride 10 instead of 1, and MGN-T with widths
  32-16 and 1 + 1 message passing steps instead of 2 + 2, so that it trains on a CPU
- Loss on all nodes except the actuator (the paper does not specify this)
- Gradient clipping, early stopping and rollout validation
- MGN-T training recipe: the [comparison protocol](../guides/comparison-protocol.md) fixes
  one recipe for all architectures instead of the
  [paper's](models.md#meshgraphnet-transformer-mgn_t)
- Optional: history (`--history 1`), `residual_target`, scheduled sampling, noise on
  `stress_prev`
