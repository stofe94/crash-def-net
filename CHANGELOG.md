# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version numbers follow
[semantic versioning](https://semver.org/) as a guideline
([versioning](https://stofe94.github.io/crash-def-net/about/#versioning)).

## [1.0.0] — 2026-11-09

Version 1.0.0: the whole chain from raw simulation data to evaluated predictions is in
place, with MeshGraphNets and MGN-T. Built up on DeepMind's `deforming_plate` benchmark,
CrashDefNet now takes general crash data and is verified on a first real crash example:
MGN-T predicts OpenRadioss runs of a bumper beam against a pole over the whole impact.
CrashDefNet is research software, made to be tested on industrial problems by users with
expertise in simulation and machine learning, who verify its predictions for their
application
([disclaimer](https://stofe94.github.io/crash-def-net/about/#disclaimer)).

It gets a documentation website with tutorials, guides, concepts and a reference generated
from the code, and two ways to try it without setup effort: the CrashDefNet app, a graphical
interface in the browser that leads through a first run without typing commands, at an early
stage and to be developed further, and the quick start on Google Colab without installation.
A new tool explores an exported dataset split against split, and
CrashDefNet now runs on Windows without WSL2 as well, tested on the CPU. From this release on,
it is licensed under the GNU AGPL-3.0.

General crash data goes through every stage: run parameters, any result fields, parts with
contact rules, material properties, free rigid parts and the units of the data, from
LS-DYNA and VTK raw data, and the config chooses which features a training uses. A config
file defines a dataset before it is generated: its components and what each of them is, and
its features. Converted OpenRadioss runs can be used as they are: the d3plot reader takes
the model definition, which such a d3plot does not hold, from the starter deck next to it.
These are at an early stage of development. The chain from OpenRadioss runs is verified on
the bumper beam, with results and a tutorial; the rest is checked on converted and synthetic
runs.

### Added

- **Train and roll out on a part of the frames:** `[data] start_frame` and `end_frame` of the
  training config choose the raw frames of the run a training uses, without a new export
  ([choose the frames](https://stofe94.github.io/crash-def-net/guides/training/#choose-the-frames));
  its rollouts start and end there as well, and the statistics of the node features and
  targets are fitted anew on these frames. `predict/rollout.py --start-frame` and
  `--end-frame` roll a model out from and to other frames of the dataset, so one model can be
  rolled out from different points in time
  ([roll out a model](https://stofe94.github.io/crash-def-net/guides/rollout/)).
- **Documentation website**, [stofe94.github.io/crash-def-net](https://stofe94.github.io/crash-def-net/),
  built from `docs/` and published with every release. Getting started: quick start (app,
  Colab or command line), installation and two tutorials, a first training and a bumper
  beam against a pole from OpenRadioss runs, with their configs in `examples/`. Results per
  dataset, `deforming_plate` and the bumper beam. Guides for every task, among them the
  comparison of the two architectures that reproduces the results on `deforming_plate`, and
  the comparison protocol; concepts
  (how CrashDefNet works, data and graphs, models, training, rollout and metrics,
  scaling), and a reference: every command line option
  and config key with its default, generated from the code, every file format between
  the stages, the project structure and a glossary. Built
  locally with `pip install -e ".[docs]"` and `zensical serve -o`.
- **Quick start on Google Colab**: `examples/quick_start/crashdefnet-quickstart.ipynb` runs export, training and rollout
  on Colab's free GPU and shows the prediction for an unseen plate as an animation, in
  under 10 minutes and without installation. Linked from the README.
- **CrashDefNet app** (early stage): `python utils/app.py` opens a graphical interface as a page
  in the browser, without typing commands. Made mainly for the quick start, it leads through
  the raw data (six example plates show what it holds), a dataset, the training with live
  progress and loss curve, the prediction of unseen simulations and the result: errors per
  simulation and an animation. It can already be used beyond it: it takes any
  configuration (`examples/quick_start/train.toml`, `configs/`, `configs/train/`, `protocols/`), lists every dataset, trained
  model and rollout, also of runs from the command line, and shows the stress in the unit of
  the data. Every step runs the command line tool and shows its log; the page is served on
  127.0.0.1 only and needs no further package. Later versions will develop it further. See
  [the app](https://stofe94.github.io/crash-def-net/guides/app/).
- **Data exploration** of an export, split against split: `utils/explore_data.py`
  describes every trajectory by mesh, loading and response (e.g. actuator travel,
  contact pairs, displacement, stress, smallest element volume), checks the data
  (non-finite values, inverted elements, identical trajectories, the largest normalized
  target), and shows how far val and test lie from train: descriptors per split,
  Kolmogorov–Smirnov distances, distance to the nearest train trajectory, targets as the
  model sees them. `--rollout` sets the error of every trajectory against the
  descriptors. Writes to `evaluations/<time>_<dataset>_data_exploration/`. See
  [data exploration](https://stofe94.github.io/crash-def-net/guides/explore-data/).
- **Feature selection in the config:** `[data] local_features`, `global_features` and
  `target_fields` choose by name which static node features, run parameters and result
  fields of an export a training uses, so comparing feature sets needs no new export.
  Without them a run uses everything, as before; `examples/quick_start/train.toml` names those of
  `deforming_plate`, and a warning names the features of an export that a run leaves out. See
  [training](https://stofe94.github.io/crash-def-net/guides/training/#choose-the-features).
- **Global block for run parameters:** datasets whose runs have run parameters
  (`globals.json`, e.g. the impact velocity) give them to the models as a global input
  per graph, and both architectures then use full GraphNet blocks that read and update
  a global latent. Without run parameters the models are unchanged. Exports from before
  this version keep the run parameters as node feature columns, and models trained on
  them load as before. See
  [models](https://stofe94.github.io/crash-def-net/concepts/models/#global-block-run-parameters).
- **Result fields besides the stress:** the targets are the velocity and the result
  fields of the data, any number of them under their own names (e.g. stress and plastic
  strain) or none; with `--history 1` each is also an input `<name>_prev`. VTK raw data
  name them with `--fields`, and `--stress-field ''` leaves out the stress; LS-DYNA runs
  add the effective plastic strain and the shell thickness with `--fields plastic_strain
  thickness`. Rollout,
  evaluation, data exploration and animation follow the fields of the data; the paper
  metrics of the stress are reported when there is one. Exports of `deforming_plate` stay
  as they were, byte for byte. See
  [data and graphs](https://stofe94.github.io/crash-def-net/concepts/graphs/#general-meshes-and-run-parameters).
- **Components and contact rules:** the parts of LS-DYNA and VTK raw data become
  components, any number under their own names (LS-DYNA: the part titles), stored with
  the export. `export.py --contact-pairs`
  (e.g. `plate:impactor plate:plate`, also as patterns) lists the pairs of components that can touch; the
  world edges then only join nodes of these pairs, in training, noise and rollout alike.
  The models see roles and properties of the nodes, not component names. See
  [components and contact rules](https://stofe94.github.io/crash-def-net/concepts/graphs/#components-and-contact-rules).
- **Free rigid parts:** `export.py --free-rigid-parts NAME` lets the model predict the
  motion of rigid parts whose motion results from the crash, such as a barrier with mass
  or a pole that gives way, instead of taking it as given. The rollout moves each of them
  as a rigid body, by the rigid motion closest to the prediction, so it keeps its shape.
  Without the option rigid parts are actuators, as before. See
  [free rigid parts](https://stofe94.github.io/crash-def-net/concepts/graphs/#free-rigid-parts).
- **Material properties of LS-DYNA runs:** `export.py --static-fields` chooses the static
  node features of LS-DYNA runs as well: shell thickness, initial velocity and, from the
  keyword deck, density, Young's modulus, Poisson's ratio and yield stress of the common
  crash materials (`*MAT_024`, `*MAT_123`, `*MAT_003`, …), averaged at the nodes. See
  [LS-DYNA runs](https://stofe94.github.io/crash-def-net/guides/ls-dyna/#the-d3plot-reader).
- **Units of the raw data:** `export.py --length-unit` / `--stress-unit` (e.g. `mm` and
  `MPa` for LS-DYNA models) record the units in `metadata.json`. Data, model, contact
  radius and position noise stay in these units; the metrics of rollout and test are in
  SI, so they read alike for every dataset. Logs, reports and animation show lengths in
  mm and the stress in kPa as the paper tables, or for data in MPa in MPa. See
  [units](https://stofe94.github.io/crash-def-net/guides/export/#units).
- **Export config** ([define the dataset in a config file](https://stofe94.github.io/crash-def-net/guides/export/#define-the-dataset-in-a-config-file)):
  `export.py --config FILE` defines a dataset before it is generated. `[components]` lists
  what goes in, by name or pattern, each with a role (`deformable`, `actuator`, `fixed`,
  `free_rigid`); every other part is left out. `--list-parts` shows the parts of a run. If
  `thickness` is asked for, a shell of a deformable part without one stops the export.
  `examples/bumper_beam/export.toml` is a runnable example.
- **Rollout config** ([define a rollout in a config file](https://stofe94.github.io/crash-def-net/guides/rollout/#define-a-rollout-in-a-config-file)):
  `rollout.py --config FILE` takes the settings of a rollout from a file, and `[evaluation]`
  or `--evaluate` runs `evaluate_run.py` on the new rollout in the same go. `[animation]` or
  `--animate` renders one trajectory, by default the one at the median of the position error.
  `examples/quick_start/rollout.toml` is a runnable example.
- **Relative L2 position error** ([rollout and metrics](https://stofe94.github.io/crash-def-net/concepts/metrics/#relative-l2-position-error)):
  the rollout also computes ‖x_pred − x_true‖ / ‖x_true‖ per step, as in published crash
  studies, in the rollout summary and `rollout_info.json`, and `evaluate_run.py` reports it.
  It is taken on absolute coordinates and so depends on their origin.
- **Run parameters from one file** (`--globals-file`) and a choice among them
  (`--global-features`).
- `*SECTION_SHELL` gives the shell thickness where a d3plot holds only zeros.
- **OpenRadioss starter decks** ([OpenRadioss runs](https://stofe94.github.io/crash-def-net/guides/openradioss/)):
  a `*_0000.rad` next to a d3plot is read without any option. Shell thickness from
  `/PROP/SHELL` as static node feature, parts titled like a `/RWALL` or `/RBODY` as rigid
  parts, clamped nodes from `/BCS`, and a mesh edge from every node tied by `/INTER/TYPE2`
  to the nearest node of its surface.
- **Cylinder walls:** stationary `/RWALL/CYL` as rigid walls in the node features, with the
  distance to the surface and the radial normal. Their rows in `dataset.db` have 7 columns
  (`wall_columns` in `info`); other exports stay as they are.
- **`--exclude-parts`** for d3plot raw data: leaves parts out of the export, e.g. display
  models written as shells, with the nodes only they hold. Nodal fields are averaged
  without them. Without the option every part stays.
- **`utils/remesh_pole.py`** ([a pole as contact partner](https://stofe94.github.io/crash-def-net/guides/openradioss/#a-pole-as-contact-partner)):
  writes the runs of a bumper beam against a pole as a second raw dataset, with a meshed
  cylinder surface in place of the display model of the pole, or as a new part where the
  conversion left the display out, so that the pole acts through contact edges. All else stays
  identical, and every run is read back and compared.
- **Bumper beam against a pole** ([results](https://stofe94.github.io/crash-def-net/results/bumper-beam/),
  [tutorial](https://stofe94.github.io/crash-def-net/getting-started/bumper-beam/)):
  MGN-T on the OpenRadioss runs of BumperBeamCrashExample, predicted over the whole impact
  on 7 unseen runs, trained with scheduled sampling and rolled out from 10 ms. The tutorial
  goes from the download through the meshed pole, the export and the training to the
  rollout, with the configs in `examples/bumper_beam/`.
- **`train.py --architecture`** switches the architecture of a config from the command
  line, e.g. `examples/quick_start/train.toml` to MeshGraphNets.

### Changed

- **Rollouts start from the undeformed state, also with `--history 1`.** The first frame of
  a run is a sample as well: its velocity input is the initial velocity of the deck over one
  frame interval (`*INITIAL_VELOCITY*`, now also `/INIVEL/TRA` of OpenRadioss starter decks;
  a tied node takes that of its surface), at rest without one. A rollout takes only this
  frame from the ground truth and predicts every later one; before, it started from the first
  two frames, of which the second can already be deformed. The database stores it per run
  (`start_velocity`, `info.first_frame`). Exports without history stay as they were; history
  exports up to v0.4.0 load and roll out as before, from the first two frames
  ([optional history](https://stofe94.github.io/crash-def-net/concepts/graphs/#optional-history-history-1-not-in-the-paper)).
- **Results from the first frame of the run, named by raw frame:** the `.vtu` files of a
  rollout are named by their raw frame of the run and begin with the first frame of the
  dataset, the given frames up to the initial state with error 0. The error curves of
  `evaluate_run.py` and `compare_paper.py` run over the raw frames as well, from the first
  predicted frame on, as do the curves over the frames of `explore_data.py` and the
  animation. The metrics are unchanged, over the predicted steps.
- **License:** from this release on, CrashDefNet is licensed under the GNU AGPL-3.0. Releases up
  to v0.4.0 stay under the Apache License 2.0.
- **Example configs in `examples/`:** `config.example.toml` is now
  `examples/quick_start/train.toml`, next to the export and rollout config and the Colab
  notebook of the quick start; every example, from the quick start to the tutorials, is a folder with
  `export.toml`, `train.toml` and `rollout.toml`. `train.py` without `--config`, the app,
  the Docker image and the Colab notebook use the new path.
- **`metadata.json`:** `dataset` names the export folder for LS-DYNA and VTK exports, also in
  the footer of the animation; TFRecord exports stay `deforming_plate`.
- **Verification status** in the documentation: verified on `deforming_plate`. OpenRadioss
  runs are at an early stage and verified on a real crash example, a bumper beam against a
  pole; LS-DYNA keyword decks, run parameters, further result fields, contact rules and the
  VTK reader are checked on synthetic and converted runs. See
  [verification status](https://stofe94.github.io/crash-def-net/concepts/graphs/#general-meshes-and-run-parameters).
- **Development direction** in the documentation: CrashDefNet is built for autoregressive
  prediction, with contact found anew in the predicted geometry at every step
  ([why autoregressive](https://stofe94.github.io/crash-def-net/concepts/overview/#why-autoregressive)),
  and refinements of the data structures and architectures aim at modelling the physics of crash deformation in more detail,
  not at a generic data level that takes any data column; every input stands for a
  physical quantity. CrashDefNet is developed further specifically for crash, not as a
  general physics AI framework. See
  [purpose](https://stofe94.github.io/crash-def-net/purpose/) and
  [design principles](https://stofe94.github.io/crash-def-net/concepts/overview/#design-principles).
- **Versioning:** the version numbers follow semantic versioning as a guideline: compatibility
  is aimed for, not guaranteed, and this changelog names what changes. See
  [versioning](https://stofe94.github.io/crash-def-net/about/#versioning).
- **README:** what CrashDefNet is and the highlights; the installation, quick start,
  first results, background, roadmap, Docker, project structure and references moved to
  the website. The pages of `docs/` are its source, in folders
  by kind (getting started, guides, concepts, reference, results, about).
- **Command line help:** every option of `train.py` and `rollout.py` has a help text,
  and `export.py --help` names all three raw formats.
- **General names for general data.** New exports name their format `trajectories_v4` and
  their sources `sources` (`deforming_plate_trajectories_v4`, `source_tfrecord`), rollouts and
  trainings their metrics `rollout_metrics` (`paper_metrics`), the export option is
  `--sources`, the dataset class `TrajectoryDataset`. The earlier names keep working: older
  exports, checkpoints and rollouts load, `--source-tfrecord` and `DeformingPlateDataset`
  remain, and the same export keeps its fingerprint. Scripts that read these files need the
  new keys, and earlier versions do not read new exports.
- **`self_contact_aware`: one list of excluded pairs for the runs of a DoE.** Runs on the same
  mesh whose edges keep their lengths, e.g. if only an impactor sits elsewhere, now share it
  instead of computing and holding one each. This shortens the start of a training and needs
  far less memory, also in the cache of `lazy` and in `/dev/shm`. The graphs stay the same.
- **Contact search faster:** the world edges drop the excluded pairs (the mesh edges, or
  with `self_contact_aware` the pairs connected through the material) by binary search in
  the sorted pairs within the radius instead of `torch.isin`, which sorted both sets on
  every call. On the CPU this step is 2–6× faster, from the plate of `deforming_plate` to a
  bumper beam; the data exploration of a shell model with 19,000 nodes and 51 frames per
  run takes 4.3 instead of 12 s per run. Same edges.
- **A feature or result a node does not have is no longer 0**, e.g. the thickness at solids
  and beams, a material property at rigid parts or the stress at beams. Such a node has no
  value in the dataset and counts in no statistics; the model gets a neutral value and a
  column `<name>_given`. A target it does not have counts neither in the loss nor in any
  metric, and the rollout leaves it empty in its `.vtu` files. Exports of models that mix
  such parts change and need a new training; others stay as they are.
- **The same run in two splits stops the export**, also if the paths differ, as the content
  of the runs is compared. In one split it gives a warning.
- The d3plot reader finds `<name>.d3plot` next to `d3plot`, tells parts that share an ID
  apart by their titles and leaves out shells without area with a warning, instead of
  stopping the export.

### Fixed

- **The rollout predicts every frame after the initial state.** Its first step took the next
  frame from the ground truth instead of the model, and the positions of the last frame were
  not formed, so two frames of every trajectory went unmeasured. On a trajectory of few
  frames this changes the metrics noticeably, on `deforming_plate` a little. In the `.vtu`
  files every step now shows one frame with the fields predicted for it; before, the
  predicted fields belonged to the next frame. Rollouts from before have one step less, which
  `check_protocol.py` reports against a newer one: roll the reference out again.
- **Windows without WSL2:** training and rollout stopped at start-up, because the trainer
  imported the Unix-only module `resource`; the peak memory now comes from the Windows
  API there. `tzdata` is installed on Windows for the time zone of the timestamps.
  Configs, JSON files, keyword decks and reports are read and written as UTF-8 on every
  system, not in the Windows code page. `check_protocol.py`, `compare_paper.py`,
  `evaluate_run.py` and `explore_data.py` no longer fail at the end when their output is
  redirected there: characters outside the Windows code page, such as σ, are replaced.
  `environment.yml` installs numpy from pip, because a conda build next to PyTorch stopped
  the process there with two OpenMP runtimes (`OMP: Error #15`).
  The whole test suite now also runs on Windows without WSL2 (CPU) before every release;
  see
  [installation](https://stofe94.github.io/crash-def-net/getting-started/installation/#windows-without-wsl2).
- A dataset in a folder whose path contains `#` or `%` could not be read: the read-only
  connection opened a different file.
- The steps of the comparison protocol copied the protocol config into `configs/`, which
  a fresh clone does not have; they now create the folder first.
- `utils/animate_rollout.py` printed a deprecation warning with pyvista 0.49; it now
  reads the cell offsets from VTK, which works with every supported pyvista version.
- `utils/evaluate_run.py` on a rollout folder showed the trajectories per split as
  `?/?/?`: `rollout_info.json` keeps no split counts, so they now come from the
  dataset's `metadata.json`, or from the checkpoint if the rollout ran on the training
  dataset.

## [0.4.0] — 2026-10-26

The comparison protocol now draws its trajectories from DeepMind's original train,
validation and test splits, and the first results are new on it: MGN-T and MeshGraphNets
with the same contact edges on 20 unseen test geometries. A new tool evaluates every run
on its own, and training, rollout and contact search handle larger splits.

### Added

- **Evaluation of a single run** without the paper: `utils/evaluate_run.py` takes a
  rollout or checkpoint folder and writes the training curves, the error over the
  rollout (mean, best and worst trajectory), every metric over the trajectories (all
  pooled, median, best, worst), the single-step errors per target and measured findings
  to `evaluations/<time>_<run>_evaluation/`. See
  [docs/rollout-and-evaluation.md](https://github.com/stofe94/crash-def-net/blob/v0.4.0/docs/rollout-and-evaluation.md#evaluating-a-run).
- **Export from DeepMind's original splits:** `export.py --source-tfrecord` takes three
  names, one per split (`train.tfrecord valid.tfrecord test.tfrecord`), and
  `--select-seed` draws the trajectories of each split at random instead of taking the
  first ones; `metadata.json` lists the drawn trajectories. Exports without these
  options are unchanged. See [docs/data.md](https://github.com/stofe94/crash-def-net/blob/v0.4.0/docs/data.md).
- **Peak memory of every training run:** every record of `history.json` holds the peak
  RAM of the main process and of the largest DataLoader worker, on a GPU the memory
  allocated and reserved by PyTorch, and the machine; the training log and
  `utils/evaluate_run.py` show it. See [docs/training.md](https://github.com/stofe94/crash-def-net/blob/v0.4.0/docs/training.md#output).
- **Micro-batches** (`[training] micro_batch_size`, a machine setting): gradient
  accumulation over micro-batches with the loss normalized over the whole batch, so
  the optimizer step equals that of `batch_size` while only one micro-batch is in
  memory. Large models and large samples keep the recipe of a comparison.
- **Displacement as the scale of the position error:** the rollout summary stores the RMS
  displacement of the ground truth since step 1, the position error of a prediction in
  which nothing moves, and every trajectory keeps its errors per step.
- **DOI on Zenodo** for all versions,
  [10.5281/zenodo.23575635](https://doi.org/10.5281/zenodo.23575635): badge and BibTeX
  in the README, `doi` in `CITATION.cff`.

### Changed

- **Comparison protocol on DeepMind's original splits**
  (`protocols/deforming_plate_s4_t150.toml`): 100 / 30 / 20 trajectories drawn from the
  train / valid / test splits, stride 4, batch 8, 60 epochs, learning rate 2e-4 → 1e-6
  along a cosine, rollout validation of all 30 val trajectories every 4 epochs. It
  replaces the protocol on `test.tfrecord` (see Removed); `check_protocol.py` checks
  against it by default. New architectures are compared with a comparable MeshGraphNets
  baseline with `self_contact_aware` edges, since the paper's world edges (`all`, the
  setting of the protocol config) do not scale to crash meshes; the published results of
  the MeshGraphNets paper remain the original reference. See
  [docs/comparison-protocol.md](https://github.com/stofe94/crash-def-net/blob/v0.4.0/docs/comparison-protocol.md#baselines).
- **First results on the new protocol:** MGN-T and MeshGraphNets in paper size as the
  comparable baseline, both with `self_contact_aware` contact edges, on 20 test
  trajectories, with the GPU memory of both runs and the median trajectory next to the
  pooled metrics ([docs/results.md](https://github.com/stofe94/crash-def-net/blob/v0.4.0/docs/results.md)).
- **Paired comparison as counts:** `check_protocol.py` reports the difference on every
  trajectory and how often each model has the lower error, without a verdict.
- **Paper comparison over the trajectories:** `04_error_per_trajectory.png` shows for
  every rollout all trajectories pooled, the median, the best and the worst trajectory
  instead of one bar per trajectory. A run whose world edges differ from its
  architecture's paper carries them in its name, e.g. "MGN (self_contact_aware)", and the
  conditions table lists the contact edges.
- **`self_contact_aware` measures the path through the material:** pairs whose
  shortest path along the mesh edges of the rest state is shorter than
  `world_edge_exclude_distance` × r_W (default 2) get no world edge. Counting mesh edges
  (`world_edge_exclude_hops`) let neighbours in the material through where a mesh is
  locally refined; in 20 of 100 trajectories of DeepMind's train split these were up to
  112,000 world edges per trajectory. Configs and checkpoints that set
  `world_edge_exclude_hops` (up to v0.3.0) keep counting edges.
- **Rollouts in batches:** trajectories of equal length are rolled out together, as one
  batch of graphs with one model call per step (up to 100,000 nodes); the errors are
  summed on the device and read once. Metrics as before up to rounding.
- **Training steps without reads from the GPU:** the loss stays on the device and is
  read once per epoch, the loss mask multiplies instead of indexing, and the contact
  search and MGN-T's slicing and position encoding take the graphs of a batch as blocks
  of consecutive nodes instead of one mask per graph. Predictions are bitwise the same,
  gradients the same up to rounding.
- **Train loss without noise on fewer samples:** the generalization gap is computed on
  5 % of the train split (evenly spread, at least 200 samples) instead of as many
  samples as the val split, which made it half the cost of every evaluation on large
  val splits. It takes no part in model selection.

### Removed

- The protocol `deforming_plate_s8_t40` on 40 trajectories of `test.tfrecord` at
  stride 8 and its results; both remain in v0.3.0.

### Fixed

- The documentation gave 1,000 trajectories for DeepMind's `train.tfrecord`; it holds
  1,200.
- DataLoader workers stopped with "Too many open files" on splits of more than about
  120 trajectories: every shared tensor kept a file descriptor open. They now receive
  the tensors through files in shared memory.

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
  element tensors and is smoothed. See [docs/data.md](https://github.com/stofe94/crash-def-net/blob/v0.3.0/docs/data.md).
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
  them. See [docs/data.md](https://github.com/stofe94/crash-def-net/blob/v0.3.0/docs/data.md#large-datasets), including the outlook on
  training on subgraphs for full-vehicle meshes.
- **World edge variants** (`[model] world_edges`): `all` (paper, default), `contact`
  (only between different bodies) and `self_contact_aware` (contact between bodies plus
  self-contact, without near neighbours within `world_edge_exclude_hops` mesh edges).
  The variant is applied consistently in the dataset, position noise and rollout and is
  stored with the checkpoint; see [docs/data.md](https://github.com/stofe94/crash-def-net/blob/v0.3.0/docs/data.md).
- **Comparison protocol** for new architectures against the MeshGraphNets baseline on the
  `deforming_plate` subset at stride 8 ([docs/comparison-protocol.md](https://github.com/stofe94/crash-def-net/blob/v0.3.0/docs/comparison-protocol.md)):
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
  ([docs/rollout-and-evaluation.md](https://github.com/stofe94/crash-def-net/blob/v0.3.0/docs/rollout-and-evaluation.md#animation)).
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
  split; [docs/training.md](https://github.com/stofe94/crash-def-net/blob/v0.3.0/docs/training.md) asks to check and adapt them for other
  data.
- **Example config trains MGN-T** in a minimal size (widths 32-16, 1 + 1 message passing
  steps, 2 transformer blocks, 16 tokens, ~50 k parameters) with contact edges between
  bodies and for self-contact: an epoch of the CPU quick test takes ~25 s instead of
  ~4.5 min. For MeshGraphNets set `[model] architecture = "meshgraphnets"`.
- **Architecture name `meshgraphnets`** (was `meshgraphnet`), also the config section
  `[model.meshgraphnets]` and the module `models/meshgraphnets.py`. Configs and
  checkpoints with the old name are translated when read and keep working.
- **Results** ([docs/results.md](https://github.com/stofe94/crash-def-net/blob/v0.3.0/docs/results.md), README): MGN-T trained with the data,
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

The first release: the workflow, built up and validated on DeepMind's `deforming_plate`
benchmark.

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

[Unreleased]: https://github.com/stofe94/crash-def-net/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/stofe94/crash-def-net/releases/tag/v1.0.0
[0.4.0]: https://github.com/stofe94/crash-def-net/releases/tag/v0.4.0
[0.3.0]: https://github.com/stofe94/crash-def-net/releases/tag/v0.3.0
[0.2.0]: https://github.com/stofe94/crash-def-net/releases/tag/v0.2.0
