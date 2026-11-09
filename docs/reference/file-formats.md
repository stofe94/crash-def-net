# File formats

What every [stage](../concepts/overview.md) writes and the next one reads. All JSON files
are plain UTF-8. Dataset, checkpoints and `.vtu` files keep the units of the raw data (m and
Pa for `deforming_plate`), the metrics are in SI ([units](../guides/export.md#units));
velocities are per time step of the dataset (one frame = `time_stride` raw steps).

## Raw data

| File | Written by | Content |
|---|---|---|
| `meta.json` | DeepMind; the converters | `collision_radius` (radius of the contact edges), `trajectory_length` (raw steps per run); for TFRecords also the field layout |
| `<split>.tfrecord` | DeepMind | one record per trajectory: `mesh_pos`, `world_pos`, `stress`, `node_type`, `cells` |
| `<run>/<name>.d3plot` …, `<run>/<name>_0000.rad` | OpenRadioss, animation files converted to d3plot; `utils/remesh_pole.py` | one run with its starter deck ([OpenRadioss runs](../guides/openradioss.md)) |
| `<run>/d3plot`, `<run>/*.k` | LS-DYNA; `tfrecord_to_d3plot.py` | one run ([LS-DYNA runs](../guides/ls-dyna.md)) |
| `<run>/*.vtu` … | ParaView, any VTK writer; `tfrecord_to_vtk.py` | one run, one file per state ([VTK time series](../guides/vtk.md)) |
| `<run>/globals.json` | the user | run parameters, e.g. `{"impact_velocity": 15.6}`; the same names in every run |
| `<source>/conversion.json` | the converters | source TFRecord and representation of the converted runs |

## Dataset folder

Written by [`export.py`](cli.md#export) into `--out-dir`; read by training, rollout and
every evaluation tool.

### dataset.db

SQLite, schema `DB_SCHEMA` in
[`dataset.py`](https://github.com/stofe94/crash-def-net/blob/main/dataset_preprocessor/dataset.py).
Arrays are stored as raw NumPy buffers with the dtype and shape given below, for a
trajectory with N nodes.

| Table | One row per | Columns |
|---|---|---|
| `info` | setting | `key`, `value` (text, JSON for lists): `format` (`trajectories_v4`), `fingerprint`, `contact_radius`, `num_steps`, `time_stride`, `frames` (the raw frames selected), `history`, `prev_stress`, with history `first_frame` (0: the first frame is a sample, from `start_velocity`), `raw_format`, `sources`, `splits`, `edge_stats_stride`, `global_names`, `static_names`; with result fields other than the stress alone `field_names` (missing: only `stress`); with components `component_names` and, with contact rules, `contact_pairs`; with walls `max_walls`; with cylinder walls `wall_columns` (7: point on the axis, unit axis, radius; missing: 6, point and unit normal); if nodes lack a result field `field_gaps`; with run parameters `global_input` (`graph`: the input `u` of a graph); with `--select-seed` the seed (`select_seed`); for VTK raw data the reader options (`vtk_options`), for LS-DYNA runs those given (`d3plot_options`: `fields`, `static_fields`, `free_rigid_parts`, `exclude_parts`, `components`, `no_deck_walls`); with `--global-features` and `--globals-file` their values (`global_features`, `globals_file`), with `--config` its content (`export_config`); units other than m and Pa (`length_unit`, `stress_unit`); with free rigid parts their component names (`rigid_bodies`) |
| `meshes` | distinct mesh | `sha256`, `num_nodes`, `num_cells`, `num_edges`, `cells` (int32, tetrahedra or padded with −1), `cell_types` (uint8 VTK types, NULL for tetrahedra), `mesh_edges` (int32 (2, E), both directions) |
| `trajectories` | trajectory | `split`, `sequence_id` (e.g. `test_traj0010`), `source` (raw run), `mesh_id`, `num_nodes`, `num_frames`, `node_type` (uint8 (N), raw values 0 / 1 / 3), `mesh_pos` (float32 (N, 3)), `globals`, `node_static` (NaN: the node has no value), `walls` (optional), `sha256`, `components` (uint32 (N), optional: the component of every node), `start_velocity` (float32 (N, 3), with history: the motion over one frame interval before the first frame at the initial velocity, the velocity input of the first frame) |
| `frames` | selected frame | `trajectory_id`, `frame` (0 … `num_frames` − 1), `world_pos` (float32 (N, 3)), `stress` (float32 (N, F): the result fields, in `deforming_plate` F = 1, the stress; NaN: the node has no value) |
| `moments` | feature group | running statistics of the train split, for `--resume` |

The `fingerprint` is a SHA-256 over the `info` settings and the hashes of every
trajectory and mesh: it identifies the content without hashing the whole file. Rollouts
record it, and the protocol check compares it.

### metadata.json

The feature layout and the export settings; models are sized from it, with the
[features a run selects](../guides/training.md#choose-the-features).

| Key | Content |
|---|---|
| `node_feature_dims`, `edge_feature_dims`, `world_edge_feature_dims`, `target_dim` | input and output widths of the model |
| `node_feature_names`, `edge_feature_names`, `world_edge_feature_names`, `target_feature_names` | column names, e.g. `bc_velocity_x`, `mesh_dist`, `stress` |
| `history`, `prev_stress` | the optional history layout (`--history 1`) |
| `first_frame` | with history: 0, the first frame of a run is a sample too, and a rollout starts from it alone |
| `contact_radius` | radius of the contact edges |
| `num_steps`, `time_stride` | frames per trajectory and raw steps between them |
| `splits` | per split: `start`, `count`; with `--select-seed` or one source per split also `source` and `indices` (positions in the raw data) |
| `sample_counts` | training samples per split |
| `db_format`, `dataset`, `loss_mask`, `raw_dir`, `sources` | format tag and provenance; `dataset`: `deforming_plate` for the TFRecord, else the name of the export folder |
| `raw_format` | only for d3plot and VTK exports |
| `length_unit`, `stress_unit` | the units of the raw data (`--length-unit`, `--stress-unit`); only when they are not m and Pa ([units](../guides/export.md#units)) |
| `global_feature_names`, `global_feature_dims` | run parameters and their number: the width of the global input `u`; only when the runs have them |
| `component_names`, `contact_pairs` | the components of the runs and the contact rules between them; only when the runs have components |
| `rigid_bodies` | the free rigid parts (`--free-rigid-parts`), whose motion the model predicts and the rollout keeps rigid; only when there are any |
| `static_feature_names`, `max_walls`, `wall_distance_cutoff`, `edge_stats_stride`, `select_seed`, `export_config` | only when they apply |

### normalizer.json

Mean and standard deviation per column of every feature group, fitted on the train
split. The format is unchanged since the first export:

```json
{"method": "standardize",
 "stats": {"node_features": {"mean": [...], "std": [...]},
           "target": {...}, "mesh_edge_features": {...}, "world_edge_features": {...}}}
```

With history a group `target_delta` is added, with run parameters a group
`global_features` (one value per train trajectory). Every checkpoint stores a copy of the
columns its run selects, which rollout and resumed training use, so a model keeps its scale
after a new export.

## Training run

Written by [`train.py`](cli.md#train) into `[checkpointing] checkpoint_dir`
([output](../guides/training.md#output)).

### Checkpoints

`best_model.pt` and `checkpoint_epoch_N.pt` are `torch.save` dictionaries;
`train.trainer.load_trained_model` rebuilds the model from them.

| Key | Content |
|---|---|
| `model_state_dict` | the weights |
| `config` | the training config in effect (`TrainingConfig.to_dict`): only the settings that take effect, with the source TOML file, its SHA-256 and the command-line overrides |
| `metadata` | `metadata.json` of the training dataset, with the [selected features](../guides/training.md#choose-the-features) only |
| `normalizer` | the statistics the weights belong to, of the selected features only |
| `epoch` | completed epochs; in `best_model.pt` the selected epoch |
| `history` | the records of `history.json` up to this epoch |
| `optimizer_state_dict`, `scheduler_state_dict` | for `--resume`; `None` when `best_model.pt` holds the weights of an earlier evaluation |
| `best_val_loss` | selection score of the selected evaluation (0 = best in all criteria) |

The attribute names of the model's modules are `state_dict` keys: renaming them breaks
existing checkpoints.

### history.json

A list with one record per epoch:

| Key | Content |
|---|---|
| `epoch`, `train_loss`, `val_loss`, `learning_rate`, `epoch_time_seconds` | always |
| `val_rollout_mse`, `val_rollout_pos_rmse`, `val_rollout_pos_rmse_final` | at evaluations with rollout validation |
| `train_loss_clean`, `gap_ratio` | at evaluations: train loss without noise, `max(1, val / train)` |
| `selection_score` | at evaluations: the worst relative excess over the criteria's lowest values |
| `selected` | `true` at the selected epoch (the weights of `best_model.pt`) |
| `is_best` | the evaluation was the selected one when it was recorded |
| `peak_memory` | peak so far in GiB: `ram_main`, `ram_worker` (the largest worker; with workers, on Linux), `gpu_allocated`, `gpu_reserved` (on a GPU), and `machine` |
| `end` | last record only: `early_stopping` or `epochs`; an aborted run has none |

### test_metrics.json

The selected model on the test split: `mse_normalized`, `mae_normalized`,
`rmse_per_field` and `mae_per_field` (single step, in SI: velocity in m per frame, stress in
Pa, other result fields in the units of the data), and `rollout` with
`rollout_metrics`, `mean_mse_normalized`, `position_rmse`, `position_rmse_final`,
`rmse_per_field` and the rollout metrics of every trajectory (`sequences`).

## Rollout folder

Written by [`rollout.py`](cli.md#rollout) ([output](../guides/rollout.md#output)).

### rollout_info.json

The call, model, data, settings and environment of the rollout:

| Key | Content |
|---|---|
| `status`, `started`, `finished`, `duration_seconds`, `command` | the call |
| `model` | `checkpoint` (path, size, modification time, SHA-256), `training_run`, `epoch`, `architecture`, `parameters`, `training_config` |
| `data` | `data_dir`, `dataset_db` (with `fingerprint`), `normalizer` (with SHA-256) and `normalizer_source`, `split`, `max_sequences`, `start_frame` and `end_frame` (the raw frames the rollout starts and ends with: given, or those of the training; null: the whole trajectory), `max_steps`, `metadata`, `matches_training_dataset`, `differences_to_training_dataset` |
| `settings` | `device`, `export_vtu`, `one_step`, `config_file` (the file of `--config`, if any), `evaluate` |
| `environment` | `git` (`commit`, `branch`, `uncommitted_changes`), `host`, versions of Python, PyTorch and PyG |
| `results` | `summary_file`, `initial_frame`, `sequences`, `rollout_metrics`, `mean_mse_normalized`, `mean_position_rmse`, `final_position_rmse`, `position_rel_l2`, `rmse_per_field` |

### Rollout summary

`<split>_rollout_summary.json` holds the errors of the rollout, as the mean over the
trajectories and in `sequences` for every trajectory, in SI whatever the units of the data
([units](../guides/export.md#units)): positions and velocities in m, the stress in Pa, other
result fields in the units of the data:

| Key | Content |
|---|---|
| `rollout_metrics` | `q_rmse_1`, `q_rmse_all`, `q_r_rmse_1`, `q_r_rmse_all` and the same for `stress`, each with `_se` (standard error over the trajectories); `units` (q in m, stress in Pa, R-RMSE as fraction); `num_trajectories` |
| `mean_mse_normalized`, `mse_normalized_per_step` | normalized MSE |
| `position_rmse_per_step` | length of the position error vector per node |
| `displacement_rms_per_step` | RMS displacement of the ground truth since the initial state: the error of a prediction in which nothing moves |
| `position_rel_l2_per_step`, `position_rel_l2` | relative L2 position error per step and its mean over the steps ([definition](../concepts/metrics.md#relative-l2-position-error)) |
| `rmse_per_field`, `mae_per_field`, `rmse_per_field_per_step` | per target column |
| `initial_frame`, `frames` | the raw frame of the run the rollout starts from, and the raw frames from the first frame of the dataset to the last predicted one |
| `sequences.<trajectory>` | the same per trajectory, plus `num_predicted_steps`, `final_position_rmse`, `final_rmse_per_field` |

### .vtu time series

`<split>/<trajectory>/tNNNN.vtu`, one file per frame from the first frame of the dataset,
named by its raw frame of the run, and `<trajectory>.vtu.series` for ParaView (time = raw
frame). The frames up to the initial state are ground truth, with prediction = ground
truth and error 0; every later file is a rollout step. Points are the predicted positions; point data:
`world_pos_gt`, `position_error`, and `<field>_pred`, `<field>_gt`, `<field>_error` for
`velocity_x/y/z` and every result field; cells with their VTK types. In the units of the
data.
