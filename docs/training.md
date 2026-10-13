# Training

```bash
python train/train.py --config config.example.toml [--epochs N] [--device cuda]
```

Command-line arguments override the TOML only when they are given explicitly:
`--data-dir`, `--checkpoint-dir`, `--epochs`, `--batch-size`, `--learning-rate`,
`--device` (`auto` | `cpu` | `cuda`), `--seed`, `--num-workers`, `--resume`
(`--help` for details). Unknown TOML keys are reported at start-up, so typos do not go
unnoticed.

## Configs

[`config.example.toml`](../config.example.toml) is a complete, commented example: a small
MeshGraphNet that trains on a CPU in minutes (~4–5 min per epoch on the small dataset)
and runs every stage of the pipeline. It contains the settings of both architectures;
switching to MGN-T is one line:

```toml
[model]
architecture = "mgn_t"
```

For own experiments, copy the example (e.g. to `configs/my_run.toml`) and adapt it. On
the way to the paper setting, typically:

- `[data] data_dir`: a larger export (see [data.md](data.md)); `lazy = true` for splits
  that do not fit into RAM
- `[model.meshgraphnet]`: paper size `latent_size = 128`, `num_message_passing_steps = 15`
- `[training]`: more `epochs`, `device = "cuda"`, `num_dataloader_workers` 2–4 on a GPU
  machine
- `[training.rollout_validation]`: `every_n_epochs` and `max_sequences` so that a
  validation rollout stays cheap compared with an epoch
- `[training.early_stopping]`: `patience` and `min_epochs` matching the run length
- `[checkpointing] checkpoint_dir`: one folder per run

## What happens during training

- **Loss:** MSE (or MAE / Huber / smooth L1) on the normalized targets, only on the
  nodes in `loss_mask` (all except the actuator).
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
- **Scheduled sampling** (optional, needs the history layout): the prediction for the
  previous sample replaces `*_prev` on a fraction of the nodes, and the current positions
  move consistently.
- **Rollout validation:** model selection and early stopping use the error of a full
  autoregressive rollout on the val split rather than the single-step val loss.
  `selection_metric` chooses the quantity: normalized MSE (`mse`, default), position RMSE
  over all steps (`pos_rmse`) or at the last step (`pos_rmse_final`). The MSE easily
  misses a drift that accumulates over many steps; the position error does not.
- **Early stopping:** stops when the selection metric stagnates, or when the
  train/val gap grows without a val gain. At the end the best model is restored and
  evaluated on the test split.

## Output

Everything goes into `[checkpointing].checkpoint_dir`:

| File | Content |
|---|---|
| `best_model.pt` | best model by the selection metric |
| `checkpoint_epoch_N.pt` | every `save_every` epochs |
| `history.json` | per epoch: losses, rollout metrics, LR, duration, `is_best` |
| `train_<time>.log`, `train_<time>_config.json` | log and the resolved config |
| `test_metrics.json` | single-step test metrics (normalized MSE/MAE, RMSE/MAE per target in physical units) and a full test rollout with the paper metrics |

Loss curves: `python utils/plot_losses.py --run-dir checkpoints/<run>` writes
`loss_curves.png` into the run folder.

To continue a run: `--resume checkpoints/<run>/checkpoint_epoch_N.pt`.

## Configuration reference

| Section | Important keys |
|---|---|
| `[data]` | `data_dir`, `num_dataloader_workers`, `lazy`, `cache_frames` |
| `[model]` | `architecture` (`meshgraphnet` \| `mgn_t`), `residual_target` |
| `[model.meshgraphnet]` | `latent_size` (paper 128), `num_mlp_layers` (2), `num_message_passing_steps` (paper 15), `aggregation` (`sum` \| `mean`) |
| `[model.mgn_t]` | `latent_size`, `token_dim`, `num_tokens`, `num_heads`, `num_transformer_blocks`, `num_pre_mp_steps`, `num_post_mp_steps`, `pe_frequencies`, `ffn_ratio`, `tau_init` (see [models.md](models.md)) |
| `[training]` | `epochs`, `batch_size` (paper 2), `learning_rate` (1e-4), `weight_decay`, `max_grad_norm`, `seed`, `device` |
| `[training.scheduler]` | `type` (`exponential` \| `plateau` \| `cosine` \| `none`), `min_lr`, `decay_steps` (number or `"auto"`), `patience` / `factor` (plateau) |
| `[training.rollout_validation]` | `enabled`, `every_n_epochs`, `max_sequences`, `selection_metric` (`mse` \| `pos_rmse` \| `pos_rmse_final`) |
| `[training.early_stopping]` | `enabled`, `patience`, `min_delta`, `min_epochs`, `smoothing_window`, `efficiency_window`, `efficiency_threshold`, `gap_weight` (`patience` and the windows count evaluations, `min_epochs` counts epochs) |
| `[loss]` | `type` (`mse` \| `mae` \| `huber` \| `smooth_l1`) |
| `[noise]` | `world_noise_std` (3e-3), `stress_noise_std` (history only) |
| `[scheduled_sampling]` | `enabled`, `start_ratio`, `end_ratio`, `warmup_epochs` (history only) |
| `[checkpointing]` | `checkpoint_dir`, `save_every`, `log_every`, `resume` |

Older key names (`[training.scheduler] lr`, `[training.early_stopping] lambda`,
`[rollout] scheduled_sampling*`) are still read, and older checkpoints can be loaded.

## Deviations from the paper

Deliberate differences (quick test and extensions):

- Quick test: stride 10 instead of 1, and `latent_size` 64 / 8 steps instead of
  128 / 15, so that it trains on a CPU
- Loss on all nodes except the actuator (the paper does not specify this)
- Gradient clipping, early stopping and rollout validation
- Optional: history (`--history 1`), `residual_target`, scheduled sampling, noise on
  `stress_prev`
