# Data

CrashDefNet turns simulation runs into a training dataset. It reads three raw formats:

| Raw format | Typical source | Details |
|---|---|---|
| DeepMind TFRecord (default) | the `deforming_plate` benchmark | [The benchmark](#the-benchmark), [Export](#export) |
| LS-DYNA runs (`d3plot` with keyword deck) | crash simulations | [Second raw format](#second-raw-format-ls-dyna-runs-d3plot) |
| VTK time series (`.vtu`, `.vtp`, `.vtk`) | ParaView and other solvers or post-processors | [Third raw format](#third-raw-format-vtk-time-series-vtu--vtp--vtk) |

The TFRecord is the format of the benchmark and of the comparison with the papers.

**For research, also on industrial data.** Besides benchmarks, CrashDefNet is compatible
with the data of industrial crash simulations, so it can serve research close to
industrial applications as well. The LS-DYNA reader takes shells, solids, thick shells
and beams in mixed meshes, rigid parts, prescribed motion, initial velocities,
thicknesses and rigid walls from the d3plot and its keyword deck; the VTK reader takes
general linear cells, parts and fields, e.g. of files exported by ParaView's LS-DYNA
reader ([general meshes and run parameters](#general-meshes-and-run-parameters)).

Large datasets and meshes need only modest hardware: the export reads every run in
blocks of frames, so a run need not fit into memory, and with lazy loading the training
reads only the frames of the current sample, so its memory does not depend on the size
of the dataset. Vehicle-crash datasets of a few hundred GB work this way, an interrupted
export continues where it stopped, and the contact search handles crash meshes ([Large
datasets](#large-datasets)). The limit today is the GPU memory of one sample: with
MeshGraphNets in paper size about 240,000 shell nodes fit on an 80 GB GPU; training on
subgraphs, together with multi-GPU support, is planned for full-vehicle meshes
([outlook](#outlook-training-on-subgraphs-and-multiple-gpus)).

The basic idea stays the same: simple, and quickly adapted to new data and new models.
The predictions are research results, no substitute for validated crash simulation or
physical testing (see the [disclaimer](../README.md#disclaimer)).

## The benchmark

DeepMind's `deforming_plate` (Pfaff et al., 2021) contains finite element simulations in
which an actuator impacts and deforms a hyperelastic plate: a simplified crash scenario
with contact and large deformations. Each trajectory has its own
geometry and tetrahedral mesh (~800–1,900 nodes in `test.tfrecord`) and 400 time steps. Per node and
step it provides the world position and the von Mises stress, plus a node type:

| Node type | Raw value | Meaning |
|---|---|---|
| normal | 0 | free plate nodes; their motion is predicted |
| actuator | 1 | scripted obstacle; motion is prescribed |
| clamped | 3 | fixed boundary nodes |

The dataset is published by DeepMind with the
[MeshGraphNets release](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
and is not part of this repository; it is downloaded directly from DeepMind (see
[Installation](installation.md#raw-data)). DeepMind states no explicit license for the
datasets (the Apache 2.0 license of that repository covers its code), so the terms of
the source apply, in particular for commercial use.

## Export

[`dataset_preprocessor/export.py`](../dataset_preprocessor/export.py) reads the raw
TFRecord once and writes three files into `--out-dir`:

| File | Content |
|---|---|
| `dataset.db` | SQLite: table `meshes` (every distinct mesh once: cells, optional cell types, mesh edges), table `trajectories` (per trajectory: source run, mesh, node types, rest positions, optional run parameters, static node features and walls, content hash), table `frames` (`world_pos` / `stress` per selected frame) and table `moments` (running statistics of the train split, for `--resume`) |
| `normalizer.json` | mean / std per feature group, fitted on the train split |
| `metadata.json` | feature layout, dimensions, splits, contact radius, export settings |

```bash
python -m dataset_preprocessor.export [options]      # --help lists all options
```

| Option | Default | Meaning |
|---|---|---|
| `--raw-format` | `tfrecord` | `d3plot`: read the LS-DYNA runs instead, stress targets are smoothed ([below](#second-raw-format-ls-dyna-runs-d3plot)); `vtk`: read VTK time series ([below](#third-raw-format-vtk-time-series-vtu--vtp--vtk)) |
| `--raw-dir` | `datasets/deforming_plate/raw_dataset` | folder with `meta.json` and the TFRecord (`d3plot`: `datasets/deforming_plate/raw_d3plot`, `vtk`: `datasets/deforming_plate/raw_vtk`) |
| `--contact-radius` | from `meta.json` | world edge radius; required for LS-DYNA runs without `meta.json` |
| `--source-tfrecord` | `test.tfrecord` | TFRecord to read (100 trajectories; `train.tfrecord`: 1,000) |
| `--out-dir` | `data/deforming_plate_s10_small` | output folder (= `[data].data_dir` of the config) |
| `--n-train` / `--n-val` / `--n-test` | 8 / 2 / 2 | trajectories per split; disjoint, contiguous ranges |
| `--num-steps` | 40 | frames per trajectory |
| `--time-stride` | 10 | raw frames between two selected frames (paper: 1) |
| `--history` | 0 | 1 = append `*_prev` columns (see below) |
| `--no-prev-stress` | – | with `--history 1`: only `velocity_prev` |
| `--no-overwrite` | – | stop if `dataset.db` already exists |
| `--resume` | off | continue an interrupted export in `--out-dir` ([large datasets](#large-datasets)) |
| `--edge-stats-stride` | 1 | fit the edge statistics on every k-th sample only ([large datasets](#large-datasets)) |
| `--block-mb` | 1024 | LS-DYNA / VTK: read the frames of a run in blocks of about this size |

Relative paths are resolved against the repository root, so the working directory
does not matter.

### Dataset variants

| Folder | Command options | Size | Used for |
|---|---|---|---|
| `deforming_plate_s10_small` | (defaults) 8/2/2 traj., 40 frames, stride 10 | ~11 MB | quick test (`config.example.toml`) |
| `deforming_plate_s8_t40` | `--n-train 28 --n-val 8 --n-test 4 --num-steps 50 --time-stride 8` | ~48 MB | first results ([results.md](results.md)) |
| `deforming_plate_s4` | `--n-train 80 --n-val 10 --n-test 10 --num-steps 100 --time-stride 4` | ~225 MB | all 100 trajectories of `test.tfrecord` |
| as in the paper | `--num-steps 400 --time-stride 1` | ~0.85 GB (100 traj.) | paper setting |

The export processes one run at a time, in blocks of frames, and folds it into the
statistics immediately (numerically stable pairwise update). Its memory depends neither
on the number of runs nor on their length (~0.5 GB for `deforming_plate`); see
[large datasets](#large-datasets).

## Loading during training

[`DeformingPlateDataset`](../dataset_preprocessor/dataset.py) builds one graph per time
step when a sample is requested.

- **In memory (default).** The frames and the static graph data of a split are loaded
  into RAM once (~8 MB per trajectory at 400 frames). DataLoader workers
  (`num_dataloader_workers`) share them through `/dev/shm`; each worker costs only
  ~0.3 GB.
- **Lazy (`[data] lazy = true`).** The frames stay in the database. Each process opens
  its own read-only connection and reads only the 2–3 frames of a sample (optionally
  cached with `cache_frames`). The static graph data of a trajectory (mesh edges, world
  edge selection, rest positions) are built when it is first needed and kept up to
  `static_cache_mb` (default 1024) per process, least recently used first out. The speed
  is the same, and memory does not depend on the dataset size. Use this for splits that
  do not fit into RAM or `/dev/shm`.

Trajectories on the same mesh share its mesh edges and world edge selection in memory.
With position noise (`world_noise_std > 0`) the train loader leaves out the contact
search: every train batch gets its world edges anew on the training device anyway, so
the search in the loader would be wasted. The rollout builds them on its device as well.
The results are the same as before.

## Large datasets

The data layout targets crash data far beyond `deforming_plate`: full-vehicle runs with
millions of nodes and DoE studies with many runs, in a `dataset.db` of up to a few
hundred GB, without extreme RAM. A frame costs 16 bytes per node (`world_pos` and
`stress`, float32 — as precise as the d3plot itself; less precise positions would
corrupt the velocity targets, which are differences of positions):

| Model | Nodes | per run (100 frames) | runs in 300 GB |
|---|---|---|---|
| component | 50,000 | 80 MB | ~3,700 |
| front end | 500,000 | 0.8 GB | ~370 |
| full vehicle | 3–5 million | 5–8 GB | ~40–60 |

**Export** (format v4 of `dataset.db`):

- *Blocks.* LS-DYNA and VTK runs are read, written and folded into the statistics in
  blocks of frames (`--block-mb`, default 1024; the LS-DYNA block size is estimated from
  the d3plot header). A run therefore need not fit into memory; small runs are one block,
  and the result does not depend on the block size.
- *Resume.* Every run is committed together with the running statistics. An export that
  stops (crash, out of time on a cluster) continues with `--resume` and the same options
  after its last complete run, with the same result as an uninterrupted export; other
  options are refused. `dataset.db` counts as finished only once `info.format` is
  written, so an unfinished file is not loaded by mistake.
- *Shared meshes.* Runs on the same mesh (same cells, e.g. the runs of a DoE) store it
  and its mesh edges once (table `meshes`); in memory they share mesh edges and world
  edge selection. The mesh edges are computed during the export, not at every start of
  a training.
- *Edge statistics.* The statistics of the mesh and world edges need a contact search
  per frame, the most expensive part of the export for large meshes.
  `--edge-stats-stride k` fits them on every k-th sample only (e.g. 10; on
  `deforming_plate` a stride of 5 moved their means and standard deviations by less than
  0.05 % of the standard deviation). It is recorded in
  `metadata.json`; the default 1 uses every sample.
- *Fingerprint.* `info.fingerprint` (SHA-256 over the export settings and the hashes of
  all blobs per trajectory, computed while writing) identifies the content. Rollouts and
  `utils/check_protocol.py` compare it instead of hashing the whole file, which would
  take minutes at a few hundred GB; older exports keep the SHA-256 of the file.
- *Page size.* Runs with frames of 1 MB and more (~65,000 nodes) get SQLite pages of
  64 KB: writing large frames is ~3x faster, reading as fast.

**Loading:** `lazy = true` (see [above](#loading-during-training)); every read-only
connection is opened with `immutable=1` (no locking, also on network file systems), so
no export may write into a `dataset.db` while a training or rollout reads it. Older
exports (format v1–v3) still load; continuing them needs a new export.

**Hardware.** What a large dataset needs depends on one run or one sample, not on the
size of the dataset:

| | Memory | Bounded by |
|---|---|---|
| Export | one block of frames and the static data of one run (~0.5 GB for `deforming_plate`) | `--block-mb` (default 1024) |
| Training with lazy loading | the 2–3 frames of a sample and the static graph data kept per process | `static_cache_mb` (default 1024) per process |
| Disk | 16 bytes per node and frame (table above) | – |
| GPU | the activations of one sample | the limit for full-vehicle meshes (below) |

### Outlook: training on subgraphs and multiple GPUs

The limit for full-vehicle meshes is not the database but the GPU memory of one sample.
One-step training keeps the activations of every message passing step for the backward
pass: measured for MeshGraphNets (latent 128, 15 steps) about 48 KiB per node and
54 KiB per directed edge, ~320 KiB per node of a shell mesh (4 mesh + 1 world edge per
node). An 80 GB GPU then holds ~240,000 nodes; gradient checkpointing per block and
bfloat16 raise that to ~2 million — still less than a full vehicle.

Planned for a later release: training on subgraphs with a halo, together with
multi-GPU support. The prediction of a node depends only on its neighbourhood of L
message passing steps (mesh and world edges), so a patch plus a halo of L hops, with the
loss on the patch nodes only, gives exactly the gradients of the whole graph while its
memory grows with the patch, not with the model. The world edges are found on the whole mesh first, the halo then
follows the combined graph. Gradient checkpointing and bfloat16 come with it as
options. MGN-T is not exact under this split (attention over all nodes, positional
encoding normalized per graph) and needs its own treatment.

The two belong together: the patches of a sample are independent units of work, so
data-parallel training (PyTorch DDP) distributes them over several GPUs, on one or more
machines, and combines their gradients, weighted by their share of the loss nodes, into
the gradient of the whole graph. The subgraphs bound the memory of each GPU, the number
of GPUs the time per epoch; the halo nodes, computed in more than one patch, are the
overhead.

## Graph per time step t

Features follow MeshGraphNets (Pfaff et al., 2021, App. A.1):

| | Features | Dim. |
|---|---|---|
| Nodes `x` | one-hot node type (normal / actuator / clamped) + next actuator velocity xₜ₊₁ − xₜ (0 elsewhere) | 6 |
| Mesh edges `edge_attr` | uᵢⱼ, ‖uᵢⱼ‖ (rest configuration) + xᵢⱼ, ‖xᵢⱼ‖ (current configuration) | 8 |
| World edges `world_edge_attr` | xᵢⱼ, ‖xᵢⱼ‖ for all node pairs with ‖xᵢ − xⱼ‖ < 0.03 that are not mesh neighbours (paper; other variants below) | 4 |
| Target `y` | velocity xₜ₊₁ − xₜ (3) + stress σₜ₊₁ (1) | 4 |
| `loss_mask` | all nodes except the actuator | – |

Relative vectors point from sender to receiver. Mesh edges come from the six edges of
every tetrahedron, in both directions (for other cell types see
[below](#general-meshes-and-run-parameters)). World edges model contact: they connect nodes
that are close in the deformed configuration but not neighbours in the mesh, and they
are rebuilt whenever the positions change (noise during training, every rollout step).
The radius search computes the distances directly: a dense distance matrix for graphs up
to 2,500 nodes, a cell list (spatial grid) above, with identical results; it scales to
crash meshes (300,000 nodes in ~4 s instead of a 360 GB distance matrix).

### World edge variants (`[model] world_edges`)

Which pairs within the radius r_W get a world edge is a model setting, so the variants
can be compared on the same dataset:

| `world_edges` | Pairs within r_W that get a world edge |
|---|---|
| `"all"` (default, paper) | all pairs that are not mesh neighbours |
| `"contact"` | only pairs of different bodies (connected parts of the mesh) |
| `"self_contact_aware"` | pairs more than `world_edge_exclude_hops` (default 4) mesh edges apart: contact between bodies plus self-contact, without the near neighbours inside a body |

With r_W = 0.03 and a median mesh edge length of about 0.018, the paper rule also links
nodes of the same body that are two or three mesh edges apart. On `deforming_plate`
(8 trajectories of the small export) the world edges split up into actuator ↔ actuator
50–90 % (rigid and prescribed, constant over time), plate ↔ plate 9–46 % (over 99 % of
them exist already in the rest configuration, so they are no self-contact) and
actuator ↔ plate 2–20 % (the actual contact). Actuator and plate share no mesh edge;
the actuator acts on the plate only through world edges.

- `contact` keeps only the edges between bodies. It suits bodies that touch each other
  but do not fold onto themselves; self-contact cannot be represented.
- `self_contact_aware` also keeps pairs of the same body that are far apart along the
  mesh, e.g. the folds of a crushed crash structure. On `deforming_plate`, 4 hops remove
  the near neighbours almost completely (3 hops leave up to 642 per trajectory in finely
  meshed regions), so the result there is close to `contact`. The number of hops needed
  grows with r_W / smallest edge length.

Bodies and hop neighbourhoods are computed once per mesh and stored in every sample
(`body_id`, `world_exclude_index`; with 4 hops about 100–170 k pairs per trajectory), so
position noise and rollout rebuild the same variant. The variant is part of the training
config in the checkpoint and is used by the rollout automatically. The normalizer
statistics of the world edges come from the export (all pairs) and are reused for every
variant: all world edges are relative vectors within the same radius, and their mean and
standard deviation differ between the variants by up to about 20 %.

All feature groups are standardized with the statistics in `normalizer.json`. The
one-hot columns keep mean 0 / std 1 and pass through unchanged.

## General meshes and run parameters

Raw data other than `deforming_plate` (LS-DYNA runs, see [below](#second-raw-format-ls-dyna-runs-d3plot))
may contain other cells, parameters per run, static node properties and rigid walls. The
dataset takes all of them:

| | |
|---|---|
| Cells | lines, triangles, quadrilaterals, tetrahedra, pyramids, wedges and hexahedra, also mixed, stored in VTK node order with their VTK cell types. Mesh edges come from the edges of every cell; the rollout writes the cell types into the `.vtu` files |
| Run parameters | `globals.json` next to a run's `d3plot` or VTK files, e.g. `{"impact_velocity": 15.6, "thickness_scale": 1.1}`. Every run of a dataset must have the same names. They are appended to the node features as constant columns `global_<name>`, standardized over the train trajectories; models and training adapt to the wider input automatically |
| Static node features | properties of a node that stay over the run: `thickness` (shell thickness at the first state, 0 at nodes without shells) and `initial_velocity_x/y/z` (from `*INITIAL_VELOCITY*`). Standardized over the train nodes; every run of a dataset must have the same ones |
| Rigid walls | stationary planar walls (`*RIGIDWALL_PLANAR`) as the last 4 node feature columns: signed distance to the nearest wall, clipped to ±5 × contact radius, and its unit normal (no wall: distance 5 × contact radius, normal 0). They depend on the positions and are rebuilt with the world edges (noise, scheduled sampling, every rollout step) |
| Node types | the three types stay: crash parts map onto them as deformable → normal; rigid (`*MAT_RIGID`) and prescribed motion (`*BOUNDARY_PRESCRIBED_MOTION_*`) → actuator, whose motion is taken as given input; fixed (`*BOUNDARY_SPC_*`) → clamped |

Column order of `x`: node type, `bc_velocity`, history columns, static node features, run
parameters, wall columns. `dataset.db` stores them per trajectory (`globals`,
`node_static`, `walls`; the cell types with the mesh) and their names in the `info` table
(`global_names`, `static_names`, `max_walls`); `metadata.json` lists
`global_feature_names`, `static_feature_names`, `max_walls` and `wall_distance_cutoff`.

Checked end to end on real LS-DYNA runs (16 quadrilaterals + 16 hexahedra) with a keyword
deck containing a rigid part, clamped nodes, prescribed motion on a node and a node set,
initial velocities (all nodes, overridden for one part) and two planar walls, plus run
parameters: export, training of both architectures (also with history, residual target
and scheduled sampling), rollout and `.vtu` export.

### Run parameters: current approach and open points

Run parameters enter the models only as constant node feature columns; the models process
node and edge features and have no input of their own for global quantities. This is the
approach of the MeshGraphNets paper (wind speed in FlagDynamic, App. A.1) and suffices as
long as global quantities are inputs only. `deforming_plate` has no run parameters, so the
current datasets do not use them.

Open, to be integrated if global quantities become relevant (e.g. crash datasets with
varied run parameters or global results):

- **Global targets** — a quantity per run and time step instead of per node, e.g. the
  acceleration at a sensor, total energy or rigid wall force. Needs a global output of the
  model (pooling over the nodes plus a decoder) and global metrics.
- **Global state in the processor** — a global latent vector updated in every message
  passing step, as in the GraphNet blocks of Battaglia et al. (2018). In MGN-T the physics
  attention tokens partly take this role.
- **Memory** — the broadcast repeats every parameter at every node; negligible for a few
  parameters, worth a dedicated input for many.

## Optional history (`--history 1`, not in the paper)

Appends `velocity_x/y/z_prev` (= xₜ − xₜ₋₁) and `stress_prev` (= σₜ) to the node
features (10 instead of 6 columns; 9 with `--no-prev-stress`). Only with this layout do
`residual_target`, `[scheduled_sampling]` and `stress_noise_std` take effect. The first
frame of each trajectory has no predecessor and is dropped as a sample. The layout is
stored in `metadata.json` and the database; model, training and rollout pick it up
automatically.

## Second raw format: LS-DYNA runs (d3plot)

The same data also exist as LS-DYNA runs, the format of crash simulations. This shows that
the pipeline works with LS-DYNA data as well, checked on data whose content is known
exactly. **The TFRecord remains the standard raw format** for training, benchmarks and the
comparison with the papers. Needs `lasso-python` (`pip install lasso-python`).

The [reader](#the-d3plot-reader) reads general LS-DYNA output (shells, solids, thick
shells, beams, user IDs, rigid parts, included keyword files), and the export takes
[general meshes and run parameters](#general-meshes-and-run-parameters), so crash runs
with shells and solids reach the dataset and the training as well.

> **The stress differs between the two raw formats.** Positions, mesh, node types and all
> input features are identical, but a dataset exported from d3plot has smoothed stress
> targets (14 % relative RMS difference on the small export, see
> [below](#why-the-stress-differs)). Do not mix the two: a model trained on one export is
> not comparable in stress with a model trained on the other, nor with the stress metrics
> of the papers when it comes from d3plot. `metadata.json` of a d3plot export records
> `raw_format: d3plot`, and the rollout warns when it differs from the training data.

**1. Convert** ([`tfrecord_to_d3plot.py`](../dataset_preprocessor/tfrecord_to_d3plot.py)):
every trajectory of a TFRecord with all frames (all 100 of `test.tfrecord`: 5 GB,
~1.5 min).

```bash
python -m dataset_preprocessor.tfrecord_to_d3plot [--source-tfrecord test.tfrecord] [--count N] [--no-overwrite]
```

```
datasets/deforming_plate/raw_d3plot/
  meta.json                        collision_radius, trajectory_length (as DeepMind's)
  test/conversion.json             source and representation
  test/traj0000/d3plot, d3plot01   geometry and states
  test/traj0000/model.k            parts and boundary conditions
```

| File | Content |
|---|---|
| `d3plot`, `d3plot01` | tetrahedra as 8-node solids (n1 n2 n3 n4 n4 n4 n4 n4); part 1 plate, part 2 actuator; state coordinates = `world_pos` of every frame; state time = frame index (quasi-static); stress as element tensors |
| `model.k` | what a d3plot does not store: `*PART`, `*MAT_RIGID` for the actuator, `*SET_NODE_LIST` + `*BOUNDARY_SPC_SET` for the clamped nodes. Not a runnable simulation input (material data unknown) |

Every run is read back with the reader below and compared with the TFRecord. A state
takes ~130 kB, ~90 % of it the 7 values per solid that the format always contains.
ParaView opens a run directly (`d3plot`, LS-DYNA reader): parts, all states, stress and
plastic strain; the part names are only in `model.k`.

**2. Export** as usual, with `--raw-format d3plot` (raw directory default:
`datasets/deforming_plate/raw_d3plot`):

```bash
python -m dataset_preprocessor.export --raw-format d3plot --out-dir data/deforming_plate_d3_s10_small
```

### The d3plot reader

[`d3plot_reader.py`](../dataset_preprocessor/d3plot_reader.py) reads one run (a folder with
a `d3plot` and optionally a keyword deck `*.k` / `*.key` / `*.dyn`) into geometry, states,
element blocks, node types and nodal result fields:

| | Supported |
|---|---|
| Elements | shells (quadrilaterals, triangles n1 n2 n3 n3), solids (hexahedra, pentahedra, tetrahedra n1 n2 n3 n4 n4 n4 n4 n4), thick shells, beams (end nodes); mixed models |
| IDs | user node and part IDs of the d3plot; the deck refers to them |
| States | all or selected states (`frames`) with their times; only the arrays needed are loaded |
| Results | von Mises stress (from the full tensor), effective plastic strain, shell thickness; reduced over the integration points (`layer_reduction`: `mean` or `max`) and averaged at the nodes. Rigid shells, for which LS-DYNA writes no state data, are left out |
| Node types | actuator = nodes of rigid parts (`*MAT_RIGID` / `*MAT_020` in the deck, or material type 20 in the d3plot) and nodes with prescribed motion (`*BOUNDARY_PRESCRIBED_MOTION_NODE` / `_SET` / `_RIGID`, option `_ID`); clamped = nodes fixed in x, y and z by `*BOUNDARY_SPC_SET` / `*BOUNDARY_SPC_NODE`; partially constrained nodes stay normal |
| Initial velocity | `*INITIAL_VELOCITY` (all nodes or a node set), `*INITIAL_VELOCITY_GENERATION` (part, part set or node set), `*INITIAL_VELOCITY_RIGID_BODY`, `*INITIAL_VELOCITY_NODE`, applied in this order (the more specific wins); rotations are ignored |
| Rigid walls | `*RIGIDWALL_PLANAR` with `_ID`, `_FINITE` (treated as infinite), `_ORTHO`, `_FORCES`; the normal points from tail to head |
| Keyword deck | fixed, long (`*KEYWORD LONG=Y`, trailing `+`) and free (comma) format; `*INCLUDE`; node and part sets `*SET_NODE` / `*SET_PART` (`_LIST`, `_GENERATE`, `_ADD`); options `_TITLE`, `_ID`, `_BIRTH_DEATH`; `*PART` with `_CONTACT`, `_PRINT`, `_ATTACHMENT_NODES` |

Checked on the converted runs and on real LS-DYNA files from the test data of
lasso-python: a shell model with user node IDs (4,915 nodes), a mixed shell / hexahedron
model with part IDs 1000–4000 and 5 / 8 integration points, a beam model and a d3plot
family split over several files.

Not supported yet: `*INCLUDE_TRANSFORM`, `*PART_INERTIA` / `*PART_COMPOSITE`, other set
types, eroded elements (`*_is_alive`), moving and non-planar rigid walls, the optional
second card of `*BOUNDARY_PRESCRIBED_MOTION`, rotational initial velocities, beam results
and femzip-compressed files. A d3plot without state coordinates (IU = 0) is rejected.

### Why the stress differs

DeepMind's data hold a von Mises stress per node; d3plot stores stress tensors per
element, as LS-DYNA does. The converter gives each tetrahedron σ = diag(s, 0, 0) with
s = mean nodal von Mises stress of its four nodes (the von Mises stress of this tensor is
exactly s). The reader computes the von Mises stress per element and averages it at each
node over the elements around it, as post-processors do. The stress thus passes node →
element mean → node mean, which smooths it.

| Small export (default options), TFRecord vs. d3plot | |
|---|---|
| splits, node types, rest positions, tetrahedra | identical |
| `world_pos` of every frame | identical |
| normalizer statistics of the input features | identical |
| stress target | 14 % relative RMS difference |
| normalizer statistics of the target | up to 6 % different |

For real LS-DYNA crash runs this averaging is the regular way to nodal stress, since they
contain element stress only.

As DeepMind states no license for the data, keep the converted files local.

## Third raw format: VTK time series (.vtu / .vtp / .vtk)

The same data also exist as VTK time series, one file per state — the ParaView
convention, which many solvers and post-processors export and which the rollout writes as
well. Unlike d3plot, VTK keeps the stress per node, so a dataset exported from these files
is **identical** to the TFRecord export. Needs `pyvista`.

**1. Convert** ([`tfrecord_to_vtk.py`](../dataset_preprocessor/tfrecord_to_vtk.py)):
every trajectory of a TFRecord with all frames (all 100 of `test.tfrecord`: 2.3 GB,
~4 min; `--format vtk` writes uncompressed legacy files, ~3× larger).

```bash
python -m dataset_preprocessor.tfrecord_to_vtk [--source-tfrecord test.tfrecord] [--count N] [--format vtu|vtk] [--no-overwrite]
```

```
datasets/deforming_plate/raw_vtk/
  meta.json                         collision_radius, trajectory_length (as DeepMind's)
  test/conversion.json              source and representation
  test/traj0000/traj0000.vtu.series order and times (ParaView opens the series)
  test/traj0000/t0000.vtu …         one state per file: points = world_pos, tetrahedra,
                                    point data node_type (0 / 1 / 3) and stress
```

**2. Export** as usual, with `--raw-format vtk` (raw directory default:
`datasets/deforming_plate/raw_vtk`):

```bash
python -m dataset_preprocessor.export --raw-format vtk --out-dir data/deforming_plate_vtk_s10_small
```

### The VTK reader

[`vtk_reader.py`](../dataset_preprocessor/vtk_reader.py) reads every folder with VTK
files as one run:

| | Supported |
|---|---|
| Files | `.vtu`, `.vtp`, legacy `.vtk` and multiblock `.vtm`; order and times from a `.pvd`, else from `*.series` files, else from the trailing number of the file names. The files may lie in subfolders, as ParaView saves a `.pvd` (`run/run.pvd` + `run/run/run_0_<t>.vtu` or `.vtm`); folders below a `.pvd`, `.series` or `.vtm` belong to it and are no runs of their own. One `.pvd` per run folder |
| Parts | several files per state (e.g. `part_a_0001.vtp`, `part_b_0001.vtp`, or `part` in the `.pvd`) and the blocks of a `.vtm` are joined into one mesh |
| Shared nodes | exports give every part its own copies of the nodes it shares with other parts (ParaView's LS-DYNA reader: one block per part). With node IDs, `--node-id-field` (default: the field the files mark as global IDs, e.g. `UserID` of ParaView's LS-DYNA reader), the copies become one node again; they must lie at the same position in every state, and the states are matched by ID. Without IDs (or `--node-id-field ''`) the parts stay separate bodies (warning) |
| Cells | lines, triangles, quadrilaterals, tetrahedra, pyramids, wedges, hexahedra; voxels / pixels reordered, polygons and triangle strips triangulated, quadratic cells linearized (their mid-side nodes stay unconnected), vertices ignored. Degenerate cells as LS-DYNA writes them, also through ParaView, become the cells they are: quadrilateral n1 n2 n3 n3 → triangle, hexahedra with repeated nodes → tetrahedron, pyramid, wedge; other cells with repeated nodes are rejected. The cells of the first state hold for all states |
| Positions | the points of every state, or with `--displacement-field U` the points plus that field (exports that keep the reference geometry); rest positions = first state |
| Node types | `--node-type-field` (default `node_type`): 0 normal, 1 actuator, 3 clamped; then the nodes of `--actuator-parts` (e.g. a rigid impactor whose motion is given) and `--clamped-parts`. Parts are named by their block in a `.vtm` (block name, else its index), else by their file (`part` of the `.pvd`, name of the `.series`, file name without the number), or per cell by `--part-field` (e.g. a part ID field). A node takes the highest type of its copies and parts (clamped > actuator > normal); without field and parts all nodes are normal |
| Fields | `--stress-field` (default `stress`): scalar, or stress tensor with 6 (xx yy zz xy yz zx) or 9 components → von Mises; cell data: von Mises per cell, averaged at the nodes over the cells of all parts. Nodes of parts without the field get 0 (warning). `--static-fields`: point or cell fields of the first state as static node features, 0 at nodes of parts without them (e.g. shell thickness at solid nodes); vectors give one feature per component (`<name>_x` / `_y` / `_z`). `globals.json` in the run folder (next to the `.pvd`): run parameters |

**LS-DYNA runs via ParaView.** Open the `d3plot` with ParaView's LS-DYNA reader and save
it as `<run>/<run>.pvd` with all time steps, one run folder each; ParaView writes one
`.vtm` per state with one block per part. Then, e.g. with the impactor as block 3:

```bash
python -m dataset_preprocessor.export --raw-format vtk --raw-dir runs_vtk --source-tfrecord none \
    --stress-field Stress --static-fields Thickness --actuator-parts 3 --contact-radius 5
```

Keep the blocks rather than applying *Merge Blocks*: the merged grid keeps only the fields
that every block has (no `Thickness` next to solids) and loses the parts. ParaView's
`Stress` of a shell is the mid surface; the inner and outer surfaces are separate fields
(the d3plot reader reduces over all integration points). VTK 9.6.2 combines two user IDs of
a 32-bit d3plot into one (newer VTK versions convert them one by one), so the
copies of a node no longer match and the reader stops with an error; check `UserID` in
ParaView, or ignore the IDs with `--node-id-field ''`.

Checked: TFRecord → VTU → `dataset.db` gives the same dataset as the TFRecord export;
synthetic runs cover `.pvd` with two shell parts, numbered files, displacement fields,
legacy `.vtk` with hexahedra and 9-component tensors, quadratic tetrahedra and polygons,
and a shell dataset through export and training. A crash-like LS-DYNA model (two shell
parts sharing an edge, hexahedra on the plate, a separate impactor, triangles, user IDs)
read with VTK's LS-DYNA reader (that of ParaView) and saved as a `.pvd` of multiblock files,
as one file per part and state, and as a merged grid gives the nodes, cells, mesh edges,
positions and nodal stress of the d3plot in all three cases, plus the shell thickness and
the impactor as actuator from the parts; the multiblock export goes through export and
training.

Not supported yet: changing topology (eroded cells), partitioned collections (`.vtpc`),
layered shell stress (one field), rigid walls and other boundary conditions beyond node
types.
