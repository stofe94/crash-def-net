# LS-DYNA runs (d3plot)

The exporter reads LS-DYNA runs: one folder per run with a `d3plot` and optionally a keyword
deck. Needs `lasso-python` (`pip install lasso-python`). The d3plot reader is the same for
[OpenRadioss runs](openradioss.md).

!!! warning "Early stage of development"
    The reading of the keyword deck is checked on runs converted from `deforming_plate`, on
    small example files and on synthetic crash runs, not yet on real LS-DYNA simulations. The
    d3plot itself is read as for OpenRadioss runs, which are verified on a real crash example
    ([verification status](../concepts/graphs.md#general-meshes-and-run-parameters)).

## Export

Own runs need no `meta.json`:

```bash
python -m dataset_preprocessor.export --raw-format d3plot --raw-dir my_runs --sources none \
    --contact-radius 5 --n-train 40 --n-val 5 --n-test 5 --num-steps 50 --time-stride 1 \
    --length-unit mm --stress-unit MPa --out-dir data/my_runs
```

A crash run with several parts, e.g. a car side against a rigid pole, with the plastic
strain as a further target, material properties as static node features and contact
rules between the parts (named by their titles):

```bash
python -m dataset_preprocessor.export --raw-format d3plot --raw-dir side_impact --sources none \
    --contact-radius 30 --n-train 40 --n-val 5 --n-test 5 --num-steps 25 --time-stride 1 \
    --history 1 --length-unit mm --stress-unit MPa --fields plastic_strain \
    --static-fields thickness initial_velocity yield_stress \
    --contact-pairs pole:door_outer pole:sill door_outer:intrusion_beam door_outer:door_outer \
    --out-dir data/side_impact
```

- Every folder below `--raw-dir` that holds a `d3plot` is one run. Sorted by path, the
  splits take contiguous ranges of them, or random ones with `--select-seed`.
- `--contact-radius` sets the radius of the contact edges in the length unit of the
  runs; without `meta.json` it is required.
- The frames are counted in states of the d3plot; `--time-stride` and `--num-steps` must
  fit into the shortest selected run.
- Run parameters go into a `globals.json` next to the `d3plot` of every run
  ([run parameters](../concepts/graphs.md#general-meshes-and-run-parameters)).
- `--length-unit` / `--stress-unit` name the units of the runs, e.g. `mm` and `MPa` for
  a model in mm, t, s. The values stay as they are; the reports convert to SI
  ([units](export.md#units)).
- `--fields` adds result fields to the von Mises stress as targets: `plastic_strain`
  (effective plastic strain) and `thickness` (shell thickness, which changes as a shell
  thins). `--static-fields` chooses the static node features: `thickness`,
  `initial_velocity` and the material properties `density`, `youngs_modulus`,
  `poisson_ratio`, `yield_stress` (default: thickness and initial velocity, where the
  runs have them). A training can then use a part of them
  ([choose the features](training.md#choose-the-features)).
- The parts become [components](../concepts/graphs.md#components-and-contact-rules),
  named by their titles; `--contact-pairs` sets which of them can touch.
- Rigid parts are actuators: their motion is taken from the run. A rigid part whose
  motion results from the crash, e.g. a barrier with mass or a pole that gives way, can
  be predicted instead: `--free-rigid-parts NAME`
  ([free rigid parts](../concepts/graphs.md#free-rigid-parts)).
- `--exclude-parts NAME` leaves parts out, named as the components: their elements and the
  nodes only they hold. Nodal fields are averaged without them; every other value stays.
- To define the parts and what each of them is before the export:
  [config file](export.md#define-the-dataset-in-a-config-file).

## The d3plot reader

[`d3plot_reader.py`](https://github.com/stofe94/crash-def-net/blob/main/dataset_preprocessor/d3plot_reader.py)
reads one run (a folder with a `d3plot` or `<name>.d3plot` and optionally a keyword deck
`*.k` / `*.key` / `*.dyn`) into geometry, states, element blocks, node types and nodal
result fields:

| | Supported |
|---|---|
| Elements | shells (quadrilaterals, triangles n1 n2 n3 n3), solids (hexahedra, pentahedra, tetrahedra n1 n2 n3 n4 n4 n4 n4 n4), thick shells, beams (end nodes); mixed models |
| IDs | user node and part IDs of the d3plot; the deck refers to them |
| States | all or selected states (`frames`) with their times; only the arrays needed are loaded |
| Results | von Mises stress (from the full tensor), effective plastic strain, shell thickness; reduced over the integration points (`layer_reduction`: `mean` or `max`) and averaged at the nodes. Rigid shells, for which LS-DYNA writes no state data, are left out. Targets: the stress, plus `--fields` |
| Parts | every part is a component, named by its title (part titles of the d3plot, else the heading of `*PART` in the deck), else by its part ID; parts that share a title get their ID appended (`title (ID)`). A node of several parts (e.g. a weld) belongs to the first part of the d3plot, nodes outside every element to the component `(no element)` |
| Material properties | from the first card of the first material of every `*MAT` keyword (also `_TITLE`, by name or number) of the types `*MAT_ELASTIC` (001), `_PLASTIC_KINEMATIC` (003), `_RIGID` (020), `_PIECEWISE_LINEAR_PLASTICITY` (024), `_PLASTICITY_WITH_DAMAGE` (081), `_SPOTWELD` (100), `_MODIFIED_PIECEWISE_LINEAR_PLASTICITY` (123): density RO, Young's modulus E, Poisson's ratio PR, and the yield stress SIGY (not for 001 and 020). Per node the mean over the elements around it whose material has the property; [no value](../concepts/graphs.md#general-meshes-and-run-parameters) at nodes without such elements (warning for parts with other material types) |
| Node types | actuator = nodes of rigid parts (`*MAT_RIGID` / `*MAT_020` in the deck, or material type 20 in the d3plot) and nodes with prescribed motion (`*BOUNDARY_PRESCRIBED_MOTION_NODE` / `_SET` / `_RIGID`, option `_ID`); clamped = nodes fixed in x, y and z by `*BOUNDARY_SPC_SET` / `*BOUNDARY_SPC_NODE`; partially constrained nodes stay normal. Rigid parts named in `--free-rigid-parts` are normal instead ([free rigid parts](../concepts/graphs.md#free-rigid-parts)) |
| Initial velocity | `*INITIAL_VELOCITY` (all nodes or a node set), `*INITIAL_VELOCITY_GENERATION` (part, part set or node set), `*INITIAL_VELOCITY_RIGID_BODY`, `*INITIAL_VELOCITY_NODE`, applied in this order (the more specific wins); rotations are ignored |
| Rigid walls | `*RIGIDWALL_PLANAR` with `_ID`, `_FINITE` (treated as infinite), `_ORTHO`, `_FORCES`; the normal points from tail to head |
| Keyword deck | fixed, long (`*KEYWORD LONG=Y`, trailing `+`) and free (comma) format; `*INCLUDE`; node and part sets `*SET_NODE` / `*SET_PART` (`_LIST`, `_GENERATE`, `_ADD`); options `_TITLE`, `_ID`, `_BIRTH_DEATH`; `*PART` with `_CONTACT`, `_PRINT`, `_ATTACHMENT_NODES` |

Checked on the OpenRadioss runs of a [bumper beam against a pole](../results/bumper-beam.md), on
the [converted runs](#test-with-converted-runs), on LS-DYNA files from the test data of lasso-python
(shells with user IDs, mixed shells and hexahedra, beams, a d3plot family) and on a
synthetic car side against a rigid pole (shell panels, beams, hexahedra welded to a
panel, part titles, materials and initial velocity in the deck), exported, trained and
rolled out end to end.

Not supported yet: `*INCLUDE_TRANSFORM`, `*PART_INERTIA` / `*PART_COMPOSITE`, other set
types, eroded elements (`*_is_alive`), moving and non-planar `*RIGIDWALL`, the optional
second card of `*BOUNDARY_PRESCRIBED_MOTION`, rotational initial velocities, beam results,
material types other than those above and femzip-compressed files. A d3plot without
state coordinates (IU = 0) is rejected.

## Test with converted runs

To test the chain against a known result, [`tfrecord_to_d3plot.py`](https://github.com/stofe94/crash-def-net/blob/main/dataset_preprocessor/tfrecord_to_d3plot.py) writes the
trajectories of a TFRecord as LS-DYNA runs below `datasets/deforming_plate/raw_d3plot/` (all
100 of `test.tfrecord`: 5 GB, ~35 s), reads every run back and compares it with the TFRecord.
`--raw-format d3plot` exports them by default:

```bash
python -m dataset_preprocessor.tfrecord_to_d3plot [--source-tfrecord test.tfrecord] [--count N] [--no-overwrite]
python -m dataset_preprocessor.export --raw-format d3plot --out-dir data/deforming_plate_d3_s10_small
```

Each run holds the tetrahedra as 8-node solids (n1 n2 n3 n4 n4 n4 n4 n4), part 1 the plate and
part 2 the actuator, the positions of every frame (state time = frame index) and the stress as
element tensors in the `d3plot`; `model.k` adds what a d3plot does not store: `*PART`,
`*MAT_RIGID` for the actuator and `*SET_NODE_LIST` + `*BOUNDARY_SPC_SET` for the clamped nodes,
not a runnable simulation input. `meta.json` (`collision_radius`, `trajectory_length`) and
`test/conversion.json` record radius, length, source and representation. ParaView opens a run
with its LS-DYNA reader. As DeepMind states no license for the data, keep the converted files
local.

Splits, node types, positions, mesh and all input features of such an export are identical to
those of the TFRecord; the stress targets are not. DeepMind's data hold a von Mises stress per
node, a d3plot stores stress per element: the converter gives each tetrahedron the mean nodal
von Mises stress of its four nodes (σ = diag(s, 0, 0)), and the reader averages the element
values at the nodes, as post-processors do and as is usual for real LS-DYNA runs.
The stress thus passes node → element mean → node mean and is smoothed: 14 % relative RMS
difference on the small export, up to 6 % in the normalizer statistics of the target. A model
trained on one of the two exports is therefore not comparable in stress with one trained on
the other, nor with the stress metrics of the papers; `metadata.json` records `raw_format`, and
the rollout warns when it differs from the training data.
