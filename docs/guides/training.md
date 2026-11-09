# Train a model

On the command line, or without typing commands in the [CrashDefNet app](app.md), which runs
the same tool with the configuration, architecture and epochs you choose there.

```bash
python train/train.py --config examples/quick_start/train.toml [--epochs N] [--device cuda]
```

Command-line options override the TOML only when they are given explicitly
([command line reference](../reference/cli.md#train)). What happens during a run (noise,
validation, model selection, early stopping): [training](../concepts/training.md).

## Configs

[`examples/quick_start/train.toml`](https://github.com/stofe94/crash-def-net/blob/main/examples/quick_start/train.toml)
is a complete, commented example: MGN-T in a minimal size (~50 k parameters) that trains
on a CPU and runs every stage of the pipeline. It contains the settings of both
architectures; switching to MeshGraphNets is one line (times of both:
[installation](../getting-started/installation.md#requirements)):

```toml
[model]
architecture = "meshgraphnets"
```

or, without editing the file, on the command line:
`python train/train.py --config examples/quick_start/train.toml --architecture meshgraphnets`.

For own experiments, copy the example (e.g. to `configs/my_run.toml`) and adapt it. On
the way to the paper setting, typically:

- `[data] data_dir`: a larger export (see [export](export.md)); `lazy = true` for splits
  that do not fit into RAM (static graph data then up to `static_cache_mb` per process,
  see [scaling](../concepts/scaling.md#loading-during-training))
- `[model.mgn_t]`: the paper size of the results, `latent_size = 80`, `token_dim = 48`,
  `num_tokens = 16`, `num_pre_mp_steps = 2`, `num_post_mp_steps = 2`
  ([configuration](../results/deforming-plate.md#mgn-t-configuration));
  `[model.meshgraphnets]`: paper size `latent_size = 128`, `num_message_passing_steps = 15`
- `[training]`: more `epochs`, `device = "cuda"`, `num_dataloader_workers` 2–4 on a GPU
  machine
- `[training.rollout_validation]`: `every_n_epochs` and `max_sequences` so that a
  validation rollout stays cheap compared with an epoch
- `[training.early_stopping]`: `patience` and `min_epochs` matching the run length
- `[checkpointing] checkpoint_dir`: one folder per run

Every key with its default: [configuration reference](../reference/configuration.md).

## Choose the features

An export holds everything a run could use, and the config chooses from it by name, so
comparing feature sets needs no new export. All three keys are optional; without them a run
uses everything:

```toml
[data]
local_features  = ["thickness", "yield_stress"]   # static node features
global_features = ["impact_velocity"]             # run parameters
target_fields   = ["stress", "plastic_strain"]    # result fields, besides the velocity
```

`examples/quick_start/train.toml` sets them for `deforming_plate`, which has the stress as its result field
and neither static node features nor run parameters; for other data, change or remove them.

| Key | What it chooses | Names in `metadata.json` | With `[]` |
|---|---|---|---|
| `local_features` | inputs per node | `static_feature_names` | no static node features |
| `global_features` | inputs per graph (`u`) | `global_feature_names` | no run parameters and no [global block](../concepts/models.md#global-block-run-parameters) |
| `target_fields` | outputs per node | `target_feature_names`, after the velocity | the velocity only |

- The names are those of the columns: a vector field has one per component, e.g.
  `initial_velocity_x`. A name the export does not have stops the run, and the message lists
  the names it has. Features of the export that a run leaves out are named in a warning at its
  start.
- With `--history 1`, the `<name>_prev` input of a result field follows that field.
- The node type, `bc_velocity`, the velocity history (`--history`), rigid walls and the
  velocity target are always in: they are the state and the boundary conditions of a step.
- The checkpoint stores the selection, so the rollout uses it by itself. `--resume` stops if
  the config selects other features than the checkpoint.
- The [comparison protocol](../guides/comparison-protocol.md) fixes the features:
  `check_protocol.py` reports a selection as a violation.

## Choose the frames

An export holds the whole run from its first frame; the config can train on a part of it,
e.g. from the second frame on, without a new export:

```toml
[data]
start_frame = 1     # raw frame of the run: the first current frame of the samples
end_frame   = 100   # raw frame of the run: the last target
```

- Both are raw frames of the run and must be frames of the dataset (with `--time-stride 8`:
  0, 8, 16, ...). Without them a run uses all frames.
- With `--history 1` the inputs of the first sample come from the frame before it, as in
  every other sample.
- The rollouts of the training, for the validation and the test, start and end there as
  well, and so does [`predict/rollout.py`](rollout.md) unless it is given other frames.
- The statistics of the inputs that change over a run (`bc_velocity`, the `*_prev` inputs)
  and of the targets are fitted anew on these frames of the train split when the run starts,
  in one pass over them, so that e.g. a jump at the first frame left out does not set the
  scale; static node features, run parameters, walls and edges keep those of the export. The
  checkpoint stores them.

## Output

Everything goes into `[checkpointing].checkpoint_dir`:

| File | Content |
|---|---|
| `best_model.pt` | selected model; like every checkpoint with the config, the metadata and the normalizer its weights belong to |
| `checkpoint_epoch_N.pt` | every `save_every` epochs |
| `candidates/` | with more than one criterion: weights of the evaluations that can still be selected, removed at the regular end of the run |
| `history.json` | per epoch: losses, rollout metrics, learning rate, duration, selection, peak memory; the last record notes how the run ended |
| `train_<time>.log`, `train_<time>_config.json` | log (with the peak memory of the run at its end) and the resolved config |
| `test_metrics.json` | single-step test metrics and a full test rollout with the rollout metrics |

Details of these files: [file formats](../reference/file-formats.md#training-run).

Loss curves: `python utils/plot_losses.py --run-dir checkpoints/<run>` writes
`loss_curves.png` into the run folder.

## Continue a run

`--resume checkpoints/<run>/checkpoint_epoch_N.pt` continues a run. It uses the
normalizer stored in the checkpoint (a warning names a `normalizer.json` that differs,
e.g. after a new export). A resumed run starts the model selection anew.

## Next steps

- [Roll out the model](rollout.md) on the test split.
- [Evaluate the run](evaluation.md#evaluate-a-run) on its own.
