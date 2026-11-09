# Export a dataset

The exporter turns simulation runs into a training dataset: one folder with three files
that training, rollout and every evaluation tool read. It reads these raw formats:

| Raw format | Status | Guide |
|---|---|---|
| DeepMind TFRecord (default) | standard: the `deforming_plate` benchmark | this page |
| OpenRadioss runs (`d3plot` converted from the animation files, with starter deck) | early stage, verified on a real crash example, a [bumper beam against a pole](../results/bumper-beam.md) | [OpenRadioss runs](openradioss.md) |
| LS-DYNA runs (`d3plot` with keyword deck) | early stage, checked on runs converted from `deforming_plate` | [LS-DYNA runs](ls-dyna.md) |
| VTK time series (`.vtu`, `.vtp`, `.vtk`) | early stage, checked on runs converted from `deforming_plate` | [VTK time series](vtk.md) |

The TFRecord is the format of the benchmark and of the comparison with the papers. The
d3plot and VTK readers take general meshes, parts, rigid bodies and boundary conditions
([general meshes](../concepts/graphs.md#general-meshes-and-run-parameters)). What the
dataset contains and how a graph is built from it: [data and graphs](../concepts/graphs.md).

## Export the benchmark

Needs `meta.json` and `test.tfrecord` of DeepMind's `deforming_plate` in
`datasets/deforming_plate/raw_dataset/` ([installation](../getting-started/installation.md#raw-data)).

```bash
python -m dataset_preprocessor.export                 # small test set, ~3 s
python -m dataset_preprocessor.export --help          # all options
```

Without options the exporter writes the small dataset of the quick start: 8 / 2 / 2
trajectories of `test.tfrecord`, 40 frames each at stride 10, into
`data/deforming_plate_s10_small/`. Relative paths are resolved against the repository
root, so the working directory does not matter. Every option:
[command line reference](../reference/cli.md#export).

The output folder (`--out-dir`, the `[data] data_dir` of a training config) holds:

| File | Content |
|---|---|
| `dataset.db` | SQLite: every distinct mesh once, the static data of every trajectory, `world_pos` / `stress` of every selected frame |
| `normalizer.json` | mean / std per feature group, fitted on the train split |
| `metadata.json` | feature layout, dimensions, splits, contact radius, export settings |

Their contents: [file formats](../reference/file-formats.md#dataset-folder).

## Choose trajectories and frames

A trajectory is one simulation run: one geometry with its own mesh, followed over the
load case. The splits hold whole trajectories, so val and test geometries are never
seen in training. If two splits would hold the same run, by its path or by its content,
the export stops.

- **One source** (default `--sources test.tfrecord`, 100 trajectories): the
  splits are disjoint, contiguous ranges, `--n-train` first, then `--n-val` and
  `--n-test`.
- **DeepMind's original splits:** three names, one per split,
  `--sources train.tfrecord valid.tfrecord test.tfrecord`; each split takes its
  trajectories from its own file (`train.tfrecord`: 1,200 trajectories).
- **Random selection:** `--select-seed 42` draws the trajectories of each split at
  random instead of taking the first ones; `metadata.json` lists the drawn ones
  (`splits.<split>.indices`).
- **Frames:** `--num-steps` frames per trajectory, `--time-stride` raw frames apart
  (paper: every frame), always from the first frame of the run on. A trajectory gives
  `num_steps - 1` training samples, one per pair of consecutive frames, also with
  `--history 1`; a training can use a part of them
  ([choose the frames](training.md#choose-the-frames)).

### Dataset variants

| Folder | Command options | Size | Used for |
|---|---|---|---|
| `deforming_plate_s10_small` | (defaults) 8/2/2 traj., 40 frames, stride 10 | ~11 MB | quick start ([`examples/quick_start/`](https://github.com/stofe94/crash-def-net/tree/main/examples/quick_start)) |
| `deforming_plate_s4` | `--n-train 80 --n-val 10 --n-test 10 --num-steps 100 --time-stride 4` | ~225 MB | all 100 trajectories of `test.tfrecord` |
| `deforming_plate_s4_t150` | `--sources train.tfrecord valid.tfrecord test.tfrecord --select-seed 42 --n-train 100 --n-val 30 --n-test 20 --num-steps 100 --time-stride 4` | ~330 MB | DeepMind's original splits, trajectories drawn at random: [comparison protocol](../guides/comparison-protocol.md), [results](../results/deforming-plate.md) |
| as in the paper | `--num-steps 400 --time-stride 1` | ~0.85 GB (100 traj.) | paper setting |

A dataset exported with `--history 1` also gives the model the previous velocity and
stress as inputs ([optional history](../concepts/graphs.md#optional-history-history-1-not-in-the-paper));
`residual_target`, scheduled sampling and stress noise need it.

## Define the dataset in a config file

Simulation output often holds more than a training needs: display models, auxiliary parts,
sections. A config file defines the dataset before it is generated: which components it
holds, what each of them is, and which features it has. Only what it lists goes in. It works
for LS-DYNA, OpenRadioss and VTK raw data. A runnable example on real crash data:
[`examples/bumper_beam/export.toml`](https://github.com/stofe94/crash-def-net/blob/main/examples/bumper_beam/export.toml)
([tutorial](../getting-started/bumper-beam.md#3-dataset)).

```bash
python -m dataset_preprocessor.export --config export.toml --list-parts   # the parts of a run
python -m dataset_preprocessor.export --config export.toml
```

```toml
[raw]
format  = "d3plot"
dir     = "my_runs"
sources = ["train_runs", "train_runs", "test_runs"]   # one, or one per split

[dataset]
out_dir   = "data/my_crash"
n_train   = 110
n_val     = 14
n_test    = 7
num_steps = 11
history   = 1

[units]
length = "mm"
stress = "GPa"

[components]                   # name or pattern = role
"crash_box_*"  = "deformable"
"beam_*"       = "deformable"
"impactor"     = "actuator"
"barrier"      = "free_rigid"

[contact]
radius = 30
# pairs = [["beam_*", "beam_*"], ["beam_*", "barrier"]]

[features]
local   = ["thickness"]        # static node features
global  = ["impact_velocity"]  # run parameters, one input per graph
targets = ["plastic_strain"]   # result fields besides the von Mises stress
```

| Role | Meaning | Motion in training and rollout |
|---|---|---|
| `deformable` | the structure that deforms | predicted by the model |
| `actuator` | rigid or moving, e.g. an impactor | given by the run |
| `fixed` | held in place, e.g. a fixed barrier | stands still (clamped) |
| `free_rigid` | rigid, its motion results from the crash, e.g. a barrier with mass | predicted by the model; the rollout keeps the part rigid |

- **Components:** a part that no pattern matches is left out, with the nodes only it holds;
  nodal fields are averaged without it. Parts are named as `--list-parts` shows them. The
  export lists the parts of the dataset with their roles and those left out; a pattern
  without a match or a part with two roles stops it.
- **Roles and the raw data:** the roles replace the rigid parts of the decks, with a warning
  where a deck calls a part rigid that is deformable here. A `free_rigid` part keeps all its
  nodes, also those it shares with other parts. Constraints on single nodes stay (deck, or
  the node type field of VTK files).
- **Thickness:** if `local` names `thickness`, every shell of a deformable part needs one
  (from the d3plot, `*SECTION_SHELL` or `/PROP/SHELL`), or the export stops and names the
  part. Solids and beams have none, so [their nodes have no value](../concepts/graphs.md#general-meshes-and-run-parameters).
- **Run parameters:** a `globals.json` in every run folder, or `globals_file` with all runs
  by the name of their folder (`{"run1": {"impact_velocity": 15.6}, ...}`).
- **Rigid walls** of the decks are used without being listed; `use_deck_walls = false` under
  `[contact]` turns that off.
- **The config stays with the dataset** in `metadata.json` and `info`. `--resume` refuses a
  changed one. Options on the command line override the file, and unknown keys stop the
  export.

A training then chooses among these features
([choose the features](training.md#choose-the-features)). Every key:
[configuration reference](../reference/configuration.md#export-config).

## Units

The exporter keeps the values of the raw data in their units: `deforming_plate` is in
m and Pa, LS-DYNA crash models mostly in mm and MPa (mm, t, s). `--length-unit` (`m`,
`cm`, `mm`) and `--stress-unit` (`Pa`, `kPa`, `MPa`, `GPa`) name them; `metadata.json`
records them as `length_unit` / `stress_unit` when they are not m and Pa.

- **In the units of the data:** the stored values, the model inputs and targets, the
  contact radius (`--contact-radius`), the position noise of the training
  (`world_noise_std`) and the `.vtu` files of a rollout. Choose the noise for the motion
  per frame of the data: the paper value 3e-3 m is too large for a crash run whose nodes
  move a few mm per frame.
- **In SI:** the metrics of the rollout and of the test evaluation (positions and
  velocities in m, the stress in Pa), so that they read alike for every dataset. Other
  result fields, e.g. the plastic strain, keep the units of the data.
- **Shown** in the logs, the reports of [evaluation and data exploration](evaluation.md)
  and the animation: lengths in mm, the stress in kPa for data in Pa or kPa, as in the
  paper tables, else in the unit of the data, e.g. MPa.

## Large and interrupted exports

The export processes one run at a time, in blocks of frames, and folds it into the
statistics immediately (numerically stable pairwise update). Its memory depends neither
on the number of runs nor on their length (~0.5 GB for `deforming_plate`).

- **Continue an interrupted export** with the same options and `--resume`: every run is
  committed together with the statistics, so the export goes on after the last complete
  trajectory.
- **Thin out the contact search** of the edge statistics with `--edge-stats-stride k`:
  only every k-th sample of a trajectory enters them. This is the most expensive part of
  an export of large meshes.
- **Block size:** d3plot and VTK runs are read in blocks of about `--block-mb` MB
  (default 1024), so a run need not fit into memory.
- **Shared meshes:** runs on the same mesh, such as the runs of a design study, store it
  once.
- An existing `dataset.db` is replaced; `--no-overwrite` stops instead. A `dataset.db`
  must not be written while a training or rollout reads it.

How the data layout scales to crash datasets of a few hundred GB:
[scaling](../concepts/scaling.md).

## Next steps

- [Train a model](training.md) on the new dataset (`[data] data_dir` or `--data-dir`).
- [Explore the dataset](explore-data.md): how far val and test lie from what the training
  sees.
