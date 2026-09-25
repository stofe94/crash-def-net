# Data

## The benchmark

DeepMind's `deforming_plate` (Pfaff et al., 2021) contains finite element simulations in
which an actuator impacts and deforms a hyperelastic plate: a simplified crash scenario
with contact and large deformations. Each trajectory has its own
geometry and tetrahedral mesh (~900–1,600 nodes) and 400 time steps. Per node and
step it provides the world position and the von Mises stress, plus a node type:

| Node type | Raw value | Meaning |
|---|---|---|
| normal | 0 | free plate nodes; their motion is predicted |
| actuator | 1 | scripted obstacle; motion is prescribed |
| clamped | 3 | fixed boundary nodes |

## Export

[`dataset_preprocessor/export.py`](../dataset_preprocessor/export.py) reads the raw
TFRecord once and writes three files into `--out-dir`:

| File | Content |
|---|---|
| `dataset.db` | SQLite: table `trajectories` (mesh per trajectory: node types, rest positions, tetrahedra) and table `frames` (`world_pos` / `stress` per selected frame) |
| `normalizer.json` | mean / std per feature group, fitted on the train split |
| `metadata.json` | feature layout, dimensions, splits, contact radius, export settings |

```bash
python -m dataset_preprocessor.export [options]      # --help lists all options
```

| Option | Default | Meaning |
|---|---|---|
| `--raw-dir` | `datasets/deforming_plate/raw_dataset` | folder with `meta.json` and the TFRecord |
| `--source-tfrecord` | `test.tfrecord` | TFRecord to read (100 trajectories; `train.tfrecord`: 1,000) |
| `--out-dir` | `data/deforming_plate_s10_small` | output folder (= `[data].data_dir` of the config) |
| `--n-train` / `--n-val` / `--n-test` | 8 / 2 / 2 | trajectories per split; disjoint, contiguous ranges |
| `--num-steps` | 40 | frames per trajectory |
| `--time-stride` | 10 | raw frames between two selected frames (paper: 1) |
| `--history` | 0 | 1 = append `*_prev` columns (see below) |
| `--no-prev-stress` | – | with `--history 1`: only `velocity_prev` |
| `--no-overwrite` | – | stop if `dataset.db` already exists |

Relative paths are resolved against the repository root, so the working directory
does not matter.

### Dataset variants

| Folder | Command options | Size | Used for |
|---|---|---|---|
| `deforming_plate_s10_small` | (defaults) 8/2/2 traj., 40 frames, stride 10 | ~10 MB | quick test (`config.example.toml`) |
| `deforming_plate_s8_t40` | `--n-train 28 --n-val 8 --n-test 4 --num-steps 50 --time-stride 8` | ~44 MB | first results ([results.md](results.md)) |
| `deforming_plate_s4` | `--n-train 80 --n-val 10 --n-test 10 --num-steps 100 --time-stride 4` | ~215 MB | all 100 trajectories of `test.tfrecord` |
| as in the paper | `--num-steps 400 --time-stride 1` | ~0.7 GB (100 traj.) | paper setting |

The export processes one trajectory at a time and folds it into the statistics
immediately (numerically stable pairwise update). It needs ~0.5 GB of RAM regardless of
the dataset size.

## Loading during training

[`DeformingPlateDataset`](../dataset_preprocessor/dataset.py) builds one graph per time
step when a sample is requested.

- **In memory (default).** The frames of a split are loaded into RAM once (~8 MB per
  trajectory at 400 frames). DataLoader workers (`num_dataloader_workers`) share them
  through `/dev/shm`; each worker costs only ~0.3 GB.
- **Lazy (`[data] lazy = true`).** The frames stay in the database. Each process opens
  its own read-only connection and reads only the 2–3 frames of a sample (optionally
  cached with `cache_frames`). The speed is the same, and memory does not depend on the
  dataset size. Use this for splits that do not fit into RAM or `/dev/shm`.

## Graph per time step t

Features follow MeshGraphNets (Pfaff et al., 2021, App. A.1):

| | Features | Dim. |
|---|---|---|
| Nodes `x` | one-hot node type (normal / actuator / clamped) + next actuator velocity xₜ₊₁ − xₜ (0 elsewhere) | 6 |
| Mesh edges `edge_attr` | uᵢⱼ, ‖uᵢⱼ‖ (rest configuration) + xᵢⱼ, ‖xᵢⱼ‖ (current configuration) | 8 |
| World edges `world_edge_attr` | xᵢⱼ, ‖xᵢⱼ‖ for all node pairs with ‖xᵢ − xⱼ‖ < 0.03 that are not mesh neighbours | 4 |
| Target `y` | velocity xₜ₊₁ − xₜ (3) + stress σₜ₊₁ (1) | 4 |
| `loss_mask` | all nodes except the actuator | – |

Relative vectors point from sender to receiver. Mesh edges come from the six edges of
every tetrahedron, in both directions. World edges model contact: they connect nodes
that are close in the deformed configuration but not neighbours in the mesh, and they
are rebuilt whenever the positions change (noise during training, every rollout step).

All feature groups are standardized with the statistics in `normalizer.json`. The
one-hot columns keep mean 0 / std 1 and pass through unchanged.

## Optional history (`--history 1`, not in the paper)

Appends `velocity_x/y/z_prev` (= xₜ − xₜ₋₁) and `stress_prev` (= σₜ) to the node
features (10 instead of 6 columns; 9 with `--no-prev-stress`). Only with this layout do
`residual_target`, `[scheduled_sampling]` and `stress_noise_std` take effect. The first
frame of each trajectory has no predecessor and is dropped as a sample. The layout is
stored in `metadata.json` and the database; model, training and rollout pick it up
automatically.
