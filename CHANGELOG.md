# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [0.3.0] — 2026-10-22

CrashDefNet reads crash data: besides DeepMind's TFRecords, LS-DYNA runs and VTK time
series, with general meshes, crash boundary conditions and an export that scales to
vehicle-crash datasets. A comparison protocol makes new architectures comparable with the
MeshGraphNets baseline, and MGN-T follows the current version of its paper (contact and
self-contact edges, global stage over the simulated structure only).

### Added

- **LS-DYNA runs as a second raw format:** `python -m dataset_preprocessor.tfrecord_to_d3plot`
  converts a TFRecord with all frames into LS-DYNA runs (`d3plot` via lasso-python plus a
  `model.k` with parts, rigid actuator and clamped nodes), checked by reading them back
  and readable by ParaView; `export.py --raw-format d3plot` builds the dataset from them
  (`d3plot_reader.py`).
  Positions, mesh and node types match the TFRecord export; the stress passes through
  element tensors and is smoothed. See [docs/data.md](docs/data.md).
- **General LS-DYNA reader** (`d3plot_reader.py`): shells, solids, thick shells and beams
  in mixed models, user node and part IDs, rigid parts (also rigid shells without state
  data), von Mises stress, plastic strain and thickness at the nodes, selected states
  with their times; keyword decks with includes, long and free format and node sets.
  Checked on real LS-DYNA files.
- **VTK time series as third raw format:** `python -m dataset_preprocessor.tfrecord_to_vtk`
  converts a TFRecord into VTK time series (one `.vtu` or legacy `.vtk` per state plus
  `.vtu.series`, checked by reading back); `export.py --raw-format vtk` builds the dataset
  from them (`vtk_reader.py`) — identical to the TFRecord export. The reader takes
  `.vtu` / `.vtp` / `.vtk` runs ordered by `.pvd`, `*.series` or file numbers, several parts
  per state, all linear cell types (voxels, polygons, quadratic cells converted), scalar or
  tensor stress in point or cell data, optional displacement field, node type and static
  fields, `globals.json`. Field names via `--stress-field`, `--node-type-field`,
  `--static-fields`, `--displacement-field`.
  Crash runs exported with ParaView's LS-DYNA reader are read as well: `.pvd` with its
  files in subfolders and multiblock `.vtm` (one block per part); the copies of shared
  nodes in several parts are merged by node ID (`--node-id-field`, default the global ID
  field such as `UserID`) and checked by position; degenerate cells as LS-DYNA writes them
  become triangles, tetrahedra, pyramids and wedges; node types from parts
  (`--actuator-parts`, `--clamped-parts`, `--part-field`); fields missing in some parts
  give 0 there, vector fields one static feature per component. Checked against the
  d3plot of a crash-like model with shared nodes, shells, hexahedra and an impactor.
- **General meshes and run parameters in the dataset:** lines, triangles, quadrilaterals,
  tetrahedra, pyramids, wedges and hexahedra, also mixed (VTK cell types; mesh edges per
  cell type; `.vtu` export with the cell types), and run parameters from `globals.json`
  as constant node features `global_<name>`. `export.py --contact-radius` for raw data
  without `meta.json`. Such datasets have DB format v3; tetrahedral datasets without
  parameters (all TFRecord exports) stay format v2 and unchanged. Checked end to end on
  real LS-DYNA runs with shells and hexahedra.
- **Crash boundary conditions and properties:** prescribed motion
  (`*BOUNDARY_PRESCRIBED_MOTION_*`) marks nodes as actuators (motion taken as given, like
  rigid parts); static node features `thickness` and `initial_velocity_x/y/z`
  (`*INITIAL_VELOCITY*`); stationary planar rigid walls (`*RIGIDWALL_PLANAR`) as node
  features signed distance to the nearest wall and its normal, rebuilt with the world
  edges. Part sets in keyword decks.
- **Large datasets (vehicle crash).** The export reads LS-DYNA and VTK runs in blocks of
  frames (`--block-mb`), commits every run with the running statistics and continues an
  interrupted export with `--resume` (same result as an uninterrupted one), stores runs
  on the same mesh (DoE) once with their mesh edges, and can fit the edge statistics on
  every k-th sample (`--edge-stats-stride`). Lazy loading keeps the static graph data of
  recently used trajectories up to `[data] static_cache_mb` per process instead of all of
  them. See [docs/data.md](docs/data.md#large-datasets), including the outlook on
  training on subgraphs for full-vehicle meshes.
- **World edge variants** (`[model] world_edges`): `all` (paper, default), `contact`
  (only between different bodies) and `self_contact_aware` (contact between bodies plus
  self-contact, without near neighbours within `world_edge_exclude_hops` mesh edges).
  The variant is applied consistently in the dataset, position noise and rollout and is
  stored with the checkpoint; see [docs/data.md](docs/data.md).
- **Comparison protocol** for new architectures against the MeshGraphNets baseline on the
  `deforming_plate` subset at stride 8 ([docs/comparison-protocol.md](docs/comparison-protocol.md)):
  fixed dataset, budget (batch 2, 120 epochs) and training recipe in
  `protocols/deforming_plate_s8_t40.toml`; architecture and world edges are declared
  variations, dataset, seed and model selection are fixed, and every model trains the
  full budget without early stopping.
  `utils/check_protocol.py` checks rollouts against it (training config, selected model,
  dataset file, trajectories) and compares them with the baseline per trajectory.
- **Rollout animation** (`utils/animate_rollout.py`): animated GIF of one trajectory, the
  ground truth next to the predictions of one or more rollouts, coloured by von Mises
  stress on one scale, the actuator in grey. Rendered off-screen in software, so it needs
  neither a GPU nor a display and gives the same frames on every machine
  ([docs/rollout-and-evaluation.md](docs/rollout-and-evaluation.md#animation)).
- **Dataset fingerprint:** `dataset.db` stores a content fingerprint, computed while
  exporting; rollouts, `check_protocol.py` and `compare_paper.py` compare it instead of
  hashing the whole file (minutes at a few hundred GB). Older exports keep the SHA-256.
- **Normalizer in the checkpoint:** rollout and `--resume` use the statistics the
  weights were trained with, also after a new export; a different `normalizer.json` is
  reported. Older checkpoints read `normalizer.json` as before.
- **End of training in the history:** the last record notes `end` (`early_stopping` or
  `epochs`), and `selected` flags the selected epoch. `check_protocol.py` reports runs
  that ended without it (aborted) and uses `selected` for the evaluated model.
- **Train loss without noise and the generalization gap:** at every evaluation the
  single-step loss on fixed train samples without noise (`train_loss_clean`) and
  `gap_ratio = max(1, val / train)` are recorded, logged and plotted; they take no part
  in selection or stopping.

### Changed

- **MGN-T: global stage over the simulated structure only** (`[model.mgn_t]
  transformer_nodes = "structure"`, as in the paper, arXiv v4 Sec. 3.3): the actuator
  takes no tokens and no longer shapes the positional encoding; it acts through the
  contact edges of the message passing stages. `"all"` keeps every node, as before;
  checkpoints without the key load with it and give the same results.
- **Model selection and early stopping are one method with criteria**
  (`[training.early_stopping] criteria`, default all: `val_loss`, `rollout_mse`,
  `rollout_pos_rmse`, `rollout_pos_rmse_final`). Every evaluation is scored by its
  worst criterion, the relative excess over the lowest value so far; the selected model
  is close to the best in all criteria at once. An earlier evaluation can be selected
  again; the weights of those still selectable are kept in `candidates/` until the run
  ends. A run stops after `patience` evaluations without a new low in any criterion; the
  history notes `stop_reason`. At the end the log lists every criterion of the selected
  model next to its lowest value. One criterion gives the former selection by that
  metric. Replaces `selection_metric` (older configs and checkpoints are translated:
  `mse` → `["rollout_mse"]`, …) and the former early-stopping settings `min_delta`,
  `smoothing_window`, `efficiency_window`, `efficiency_threshold` and `gap_weight`, which
  are read without effect. The rules are tuned to `deforming_plate` with a small val
  split; [docs/training.md](docs/training.md) asks to check and adapt them for other
  data.
- **Example config trains MGN-T** in a minimal size (widths 32-16, 1 + 1 message passing
  steps, 2 transformer blocks, 16 tokens, ~50 k parameters) with contact edges between
  bodies and for self-contact: an epoch of the CPU quick test takes ~25 s instead of
  ~4.5 min. For MeshGraphNets set `[model] architecture = "meshgraphnets"`.
- **Architecture name `meshgraphnets`** (was `meshgraphnet`), also the config section
  `[model.meshgraphnets]` and the module `models/meshgraphnets.py`. Configs and
  checkpoints with the old name are translated when read and keep working.
- **Results** ([docs/results.md](docs/results.md), README): MGN-T trained with the data,
  budget and recipe of the comparison protocol in the configuration of its paper (arXiv v4: 2 + 2 message passing
  steps, 16 tokens, contact and self-contact edges, global stage over the structure;
  0.51 M parameters), compared with the MeshGraphNets baseline; the values the paper
  leaves open are documented, followed by a short discussion; a scientific evaluation is
  planned for later releases. New figures and animation, made with
  `utils/animate_rollout.py`.
- **MGN-T paper cited as arXiv:2601.23177v4:** table, section and page numbers refer
  to version 4, which renumbers the tables of v3, replaces BSMS-GNN by EAGLE among the
  baselines and states the deforming_plate setup (16 tokens, 25 epochs, batch 32, τ₀ =
  1.0). The MGN-T values on deforming_plate are the same in both versions; the widths
  64-32-64 and 128 tokens of the defaults remain from v3 (Tab. 2, its Pi-beam example).
  The MeshGraphNets reference comes from its source, Pfaff et al. 2021 (Tab. 1 on
  p. 7), now with its standard error over the test trajectories (RMSE-1 0.25 ± 0.05,
  RMSE-all 15.1 ± 4.0 × 10⁻³ m).
- **Stored training config names its source and lists only the settings in effect:**
  `train_<time>_config.json`, the checkpoint and `rollout_info.json` record the TOML
  file (`config_file`, `config_file_sha256`) and the command-line overrides
  (`config_overrides`), and the training and rollout logs show them. Settings without
  effect are left out: those of the other architecture, of the schedulers not selected
  and of disabled options (rollout validation, early stopping, scheduled sampling, lazy
  loading cache, `world_edge_exclude_hops` without `self_contact_aware`). Older
  checkpoints load as before.
- **Scalable world edges:** the radius search computes distances directly (dense matrix up
  to 2,500 nodes per graph, cell list above) instead of `torch.cdist`'s matrix-product
  form; it scales to crash meshes. Pairs within float32 rounding of the radius can differ
  from before (on `deforming_plate` at most 5·10⁻⁷ from the radius; world edge statistics
  change in the 6th digit).
- **`dataset.db` format v4:** tables `meshes` (cells and mesh edges, once per mesh) and
  `moments` (statistics for `--resume`), per trajectory its source run and a content
  hash. Formats v1–v3 still load; `db_format` no longer counts as a difference to the
  training dataset in the rollout check. Samples and `normalizer.json` are identical to
  those of the previous export.
- **Contact search where it is used:** with position noise the train loader no longer
  builds world edges that every batch rebuilds on the training device anyway, and the
  rollout builds them on its device. Results unchanged (bit-identical in single-threaded
  runs).
- **Mesh edges ~30× faster** with a sixth of the memory (deduplicated as one integer key
  per edge); same edges.
- **Test evaluation sums its errors per batch** instead of keeping all predictions;
  single-step test metrics change in the 8th digit (float64 sums).
- **SQLite:** read-only connections use `immutable=1` (no locking; no export may write a
  `dataset.db` that is being read); runs with frames of 1 MB and more get 64 KB pages
  (writing ~3× faster).

### Fixed

- **Rollout from the command line ignored lazy loading** and always loaded the whole
  split into memory; it now follows the training config (`--lazy` / `--no-lazy`).

## [0.2.0] — 2026-10-13

Developed and validated on DeepMind's `deforming_plate` benchmark.

### Added

- **Data pipeline:** streaming export of the DeepMind TFRecord data into a compact
  SQLite dataset (`dataset.db`, `normalizer.json`, `metadata.json`) with train/val/test
  split, selectable stride and number of frames; in-memory or lazy loading; one graph
  per time step with mesh and world (contact) edges; optional history features.
- **Models:** MeshGraphNets (Pfaff et al., ICLR 2021) with mesh and world edges, and
  MeshGraphNet-Transformer (Iparraguirre et al., 2026) with physics attention,
  implemented from the paper with all assumptions documented; optional residual target.
- **Training:** single-step training with consistent position noise, exponential /
  plateau / cosine LR schedules (`decay_steps = "auto"`), gradient clipping, rollout
  validation with selectable selection metric (`mse`, `pos_rmse`, `pos_rmse_final`),
  early stopping, resumable checkpoints, test metrics including a full test rollout.
- **Rollout:** autoregressive rollout with contact edges rebuilt each step; metrics of
  the MGN-T paper (RMSE-1, RMSE-all, R-RMSE); `.vtu` export with `.vtu.series` for
  ParaView; one folder per call with `rollout_info.json` for traceability.
- **Evaluation:** `utils/compare_paper.py` compares rollouts with the tables of the
  MGN-T paper (figures, tables, Markdown report, CSV/JSON); `utils/plot_losses.py`.
- **Example config:** `config.example.toml`, a commented CPU quick test that covers
  both architectures.
- **Docker:** GPU training image (CUDA 12.4) with docker compose.
- **Documentation:** README, `docs/` (installation, data, models, adding an
  architecture, training, rollout and evaluation, Docker, first results); a disclaimer
  for the results: they are predictions of learned models, no substitute for validated
  crash simulation or physical testing, and the software comes without warranty.

[Unreleased]: https://github.com/stofe94/crash-def-net/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/stofe94/crash-def-net/releases/tag/v0.3.0
[0.2.0]: https://github.com/stofe94/crash-def-net/releases/tag/v0.2.0
