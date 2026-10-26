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

[`config.example.toml`](../config.example.toml) is a complete, commented example: MGN-T
in a minimal size (~50 k parameters) that trains on a CPU in about two and a half minutes
(~30 s per epoch on the small dataset) and runs every stage of the pipeline. It contains the
settings of both architectures; switching to MeshGraphNets is one line (~2 min 30 s per
epoch on the CPU listed in [installation.md](installation.md#requirements)):

```toml
[model]
architecture = "meshgraphnets"
```

For own experiments, copy the example (e.g. to `configs/my_run.toml`) and adapt it. On
the way to the paper setting, typically:

- `[data] data_dir`: a larger export (see [data.md](data.md)); `lazy = true` for splits
  that do not fit into RAM (static graph data then up to `static_cache_mb` per process,
  see [data.md](data.md#large-datasets))
- `[model.mgn_t]`: paper values `num_pre_mp_steps = 2`, `num_post_mp_steps = 2` and wider
  latents (e.g. `latent_size = 64`, `token_dim = 32`, see [models.md](models.md));
  `[model.meshgraphnets]`: paper size `latent_size = 128`, `num_message_passing_steps = 15`
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
- **Micro-batches** (`micro_batch_size`, a machine setting): `batch_size /
  micro_batch_size` forward passes make one optimizer step, with the loss normalized
  over the whole batch, so the step equals that of one batch of `batch_size` while only
  one micro-batch is in memory.
- **Scheduled sampling** (optional, needs the history layout): the prediction for the
  previous sample replaces `*_prev` on a fraction of the nodes, and the current positions
  move consistently.
- **Evaluations:** with rollout validation every `every_n_epochs` epochs (and in the
  first and last epoch), else every epoch. An evaluation records the selection
  criteria:

  | Criterion | |
  |---|---|
  | `val_loss` | single-step loss on the val split |
  | `rollout_mse` | normalized MSE of an autoregressive rollout of the val split |
  | `rollout_pos_rmse` | position error of that rollout, mean over the steps |
  | `rollout_pos_rmse_final` | position error at the last rollout step |

  The rollout criteria need rollout validation. Each evaluation also records the train
  loss without noise (on 5 % of the train split, at least 200 samples) and the
  generalization gap `max(1, val / train)`; neither takes part in selection or stopping.
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
  scatter on your data. A resumed run (`--resume`) starts the selection anew.

## Output

Everything goes into `[checkpointing].checkpoint_dir`:

| File | Content |
|---|---|
| `best_model.pt` | selected model; like every checkpoint with the training config, the dataset metadata and the normalizer statistics the weights belong to |
| `checkpoint_epoch_N.pt` | every `save_every` epochs |
| `candidates/` | with more than one criterion: weights of the evaluations that can still be selected, removed at the regular end of the run |
| `history.json` | per epoch: losses, rollout metrics, LR, duration, `selected`; at evaluations also `selection_score`, `train_loss_clean`, `gap_ratio`; `peak_memory` of the run so far in GiB (RAM of the main process and the largest worker, on a GPU allocated and reserved memory) and the `machine`; the last record notes `end` (`early_stopping` or `epochs`), an aborted run has none |
| `train_<time>.log`, `train_<time>_config.json` | log and the resolved config with its source TOML file, SHA-256 and command-line overrides, only the settings in effect |
| `test_metrics.json` | single-step test metrics (normalized MSE/MAE, RMSE/MAE per target in physical units) and a full test rollout with the paper metrics |

Loss curves: `python utils/plot_losses.py --run-dir checkpoints/<run>` writes
`loss_curves.png` into the run folder.

To continue a run: `--resume checkpoints/<run>/checkpoint_epoch_N.pt`. It continues with
the normalizer stored in the checkpoint (a warning names a `normalizer.json` that differs,
e.g. after a new export).

## Configuration reference

| Section | Important keys |
|---|---|
| `[data]` | `data_dir`, `num_dataloader_workers`, `lazy`, `cache_frames`, `static_cache_mb` (lazy: static graph data per process, MB; default 1024) |
| `[model]` | `architecture` (`meshgraphnets` \| `mgn_t`), `residual_target`, `world_edges` (`all` \| `contact` \| `self_contact_aware`, see [data.md](data.md#world-edge-variants-model-world_edges)), `world_edge_exclude_distance` (2.0, `self_contact_aware`), `world_edge_exclude_hops` (configs up to v0.3.0) |
| `[model.meshgraphnets]` | `latent_size` (paper 128), `num_mlp_layers` (2), `num_message_passing_steps` (paper 15), `aggregation` (`sum` \| `mean`) |
| `[model.mgn_t]` | `latent_size`, `token_dim`, `num_tokens`, `num_heads`, `num_transformer_blocks`, `num_pre_mp_steps`, `num_post_mp_steps`, `pe_frequencies`, `ffn_ratio`, `tau_init`, `transformer_nodes` (`structure` \| `all`), `num_mlp_layers` and `aggregation` (one setting shared with `[model.meshgraphnets]`) (see [models.md](models.md)) |
| `[training]` | `epochs`, `batch_size` (MeshGraphNets paper 2, MGN-T paper 32), `micro_batch_size` (default `batch_size`; divides it), `learning_rate` (1e-4), `weight_decay`, `max_grad_norm`, `seed`, `device` |
| `[training.scheduler]` | `type` (`exponential` \| `plateau` \| `cosine` \| `none`), `min_lr`, `decay_steps` (number or `"auto"`), `patience` / `factor` (plateau) |
| `[training.rollout_validation]` | `enabled`, `every_n_epochs`, `max_sequences` |
| `[training.early_stopping]` | `enabled`, `patience` (evaluations), `min_epochs`, `criteria` (default all: `val_loss`, `rollout_mse`, `rollout_pos_rmse`, `rollout_pos_rmse_final`) |
| `[loss]` | `type` (`mse` \| `mae` \| `huber` \| `smooth_l1`) |
| `[noise]` | `world_noise_std` (paper 3e-3, default 0), `stress_noise_std` (history only) |
| `[scheduled_sampling]` | `enabled`, `start_ratio`, `end_ratio`, `warmup_epochs` (history only) |
| `[checkpointing]` | `checkpoint_dir`, `save_every`, `log_every`, `resume` |

Configs and checkpoints of earlier releases still load: older key names are translated,
keys without effect are listed in a note, and `world_edge_exclude_hops` keeps counting
mesh edges (a config sets it or `world_edge_exclude_distance`, not both).

## Deviations from the paper

Deliberate differences (quick test and extensions):

- Quick test (`config.example.toml`): stride 10 instead of 1, and MGN-T with widths
  32-16 and 1 + 1 message passing steps instead of 2 + 2, so that it trains on a CPU
- Loss on all nodes except the actuator (the paper does not specify this)
- Gradient clipping, early stopping and rollout validation
- MGN-T training recipe: the paper trains 25 epochs with batch 32 (snapshots of one
  trajectory) and a cosine schedule from 1e-4 to 1e-6 (arXiv v4, Sec. 5.1); the
  [comparison protocol](comparison-protocol.md) fixes one recipe for all architectures
- Optional: history (`--history 1`), `residual_target`, scheduled sampling, noise on
  `stress_prev`
