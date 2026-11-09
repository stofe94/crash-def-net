# Tutorial: bumper beam

A real crash case from start to finish: OpenRadioss runs of a bumper beam that hits a rigid
pole, from the download of the raw data through the dataset and the training of MGN-T to the
prediction of runs the model has never seen. It reproduces the
[bumper beam results](../results/bumper-beam.md).

Needs the [installation](installation.md) with `lasso-python` and `pyvista`, and
`huggingface_hub` for the download (`pip install huggingface_hub`); about 9 GB of disk space
and a GPU. The GPU times on this page were measured on an NVIDIA RTX A4000 (16 GB).

!!! warning "Early stage of development"
    The processing of OpenRadioss runs is at an early stage of development; it is verified on
    this example
    ([verification status](../concepts/graphs.md#general-meshes-and-run-parameters)).

!!! note "The data"
    [BumperBeamCrashExample](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample)
    on Hugging Face, generated with
    [OpenRadioss2PhysicsNemo](https://github.com/HoussemMouradi/OpenRadioss2PhysicsNemo). The
    decks of the runs carry the notice "Copyright Altair Engineering, CC BY-NC 4.0". CrashDefNet
    does not ship the data; the first step downloads them from their source.

## 1. Raw data

```bash
hf download AIRBORNEPANDA/BumperBeamCrashExample --repo-type dataset \
    --include "*.d3plot*" --include "*_0000.rad" \
    --local-dir datasets/BumperBeamCrashExample
```

131 runs (3.9 GB): a bumper beam with two crash boxes, mounted on a rigid body of 500 kg, hits
a rigid pole at 5 m/s. The runs differ in the thickness of the beam (1.21 to 2.0 mm) and in
the position of the pole across the beam (−480 to 492 mm); each holds 11 states at 10 ms and
13,882 nodes. The two patterns take only what the export reads:

```
datasets/BumperBeamCrashExample/RAW_DATA/
  TRAINING_DATA/Run100/Bumper_Beam_AP_meshed.d3plot, .d3plot01 ...   deformation and results
  TRAINING_DATA/Run100/Bumper_Beam_AP_meshed_0000.rad                starter deck
  ...                                                                124 runs
  VALIDATION_DATA/Run224/ ... Run230/                                7 runs
```

The d3plot files are converted from the OpenRadioss animation files and hold the mesh, the node
coordinates of every state and the element stress. The model definition comes from the starter
deck next to them: shell thickness, rigid bodies, the pole as an analytic cylinder, ties
([OpenRadioss runs](../guides/openradioss.md#the-starter-deck)).

## 2. Mesh the pole

```bash
python utils/remesh_pole.py --out-dir datasets/BumperBeamCrashExample_remeshed/RAW_DATA
```

The solver computes the impact against the analytic cylinder of the deck; the d3plot shows the
pole only as two rings of nodes about a metre above and below the beam. So that the pole acts
on the structure through [contact edges](../concepts/graphs.md#world-edge-variants-model-world_edges),
as every other contact, the script replaces this display by a meshed cylinder surface at the
same place: 10 mm edge length like the structure, 2,320 nodes, over the height the structure
reaches. It writes a second raw dataset (~50 s, 4.5 GB), reads every run back and compares it
with the original: all other nodes and results stay identical
([options](../reference/cli.md#remesh_pole)).

## 3. Dataset

The export config
[`examples/bumper_beam/export.toml`](https://github.com/stofe94/crash-def-net/blob/main/examples/bumper_beam/export.toml)
defines the dataset. The names of the parts of a run:

```bash
python -m dataset_preprocessor.export --config examples/bumper_beam/export.toml --list-parts
```

`[components]` gives each part its role; a part that is not listed is left out:

| Parts | Role | |
|---|---|---|
| `crash_box_*`, `BumperBeam_*`, `CONNECTION_SPOTWELD*` | `deformable` | the structure, with its spot welds |
| `Body_in_White` | `free_rigid` | the rigid body behind the crash boxes: the model predicts its motion, the rollout keeps it rigid ([free rigid parts](../concepts/graphs.md#free-rigid-parts)) |
| `RWALL` | `actuator` | the meshed pole, standing still |
| `New SECT 1` (not listed) | left out | shells that only display a section of the output and carry no stress |

The other settings:

- **Splits:** 110 runs of `TRAINING_DATA` for training, the next 14 for validation, the 7 runs
  of `VALIDATION_DATA` for the test.
- **States:** all 11, with the previous velocity and stress as inputs (`history = 1`); at the
  first state the previous velocity comes from the initial velocity of the deck (`/INIVEL`).
- **Units:** mm and GPa, as in the decks.
- **Contact:** a radius of 30 mm; the wall of the deck does not enter as node features, as the
  meshed pole takes its place.
- **Static node feature:** the shell thickness from the starter deck.

```bash
python -m dataset_preprocessor.export --config examples/bumper_beam/export.toml
```

The export (~3 min, ~460 MB) writes `data/bumper_beam_remeshed/` with 16,152 nodes per run.
Nodes without a shell, the ends of the spot welds and the main node of the rigid body, have no
thickness, and the pole has no stress: such a value is [missing, not 0](../concepts/graphs.md#general-meshes-and-run-parameters),
and the missing stress counts in neither loss nor metrics.

## 4. Train

```bash
python train/train.py --config examples/bumper_beam/train.toml
```

[`examples/bumper_beam/train.toml`](https://github.com/stofe94/crash-def-net/blob/main/examples/bumper_beam/train.toml)
trains MGN-T in the size of its paper (0.51 M parameters) for 80 epochs at batch 4, in about
3 h 20 min with 7.3 GB of GPU memory. Its settings for this case:

- **From 10 ms** (`[data] start_frame = 1`): the training leaves out the step from the
  undeformed state at 0 ms to the first deformed state
  ([why](../results/bumper-beam.md#setup)); the statistics of the inputs and targets are
  fitted on the frames it uses ([choose the frames](../guides/training.md#choose-the-frames)).
- **The change, not the state** (`residual_target`): the beam moves mainly by inertia, so the
  model predicts how velocity and stress change from one state to the next.
- **Contact edges** (`self_contact_aware`): between the structure and the pole and where the
  structure folds onto itself, rebuilt from the predicted geometry at every step; the
  transformer of MGN-T works on the structure only (`transformer_nodes = "structure"`).
- **Noise in mm:** 0.3 mm on the positions and 0.01 GPa on the stress. The motion across the
  beam is only about 0.5 mm per state, so the 3 mm of `deforming_plate` would cover it.
- **Scheduled sampling:** a growing share of the nodes takes its previous velocity and stress
  from the model's own prediction of the step before, as in the rollout, up to 70 % after 40
  of the 80 epochs ([training](../concepts/training.md#loss-and-optimizer-step)).
- **Model selection:** every 4 epochs the model rolls out the 14 validation runs; the best
  rollout gives `best_model.pt` ([model selection](../concepts/training.md#model-selection-and-early-stopping)).

## 5. Predict and evaluate

```bash
python predict/rollout.py --config examples/bumper_beam/rollout.toml
```

[`examples/bumper_beam/rollout.toml`](https://github.com/stofe94/crash-def-net/blob/main/examples/bumper_beam/rollout.toml)
predicts the 7 test runs from the state at 10 ms, the first frame of the training, over 9 steps
to 100 ms, each step from its own previous prediction. In the same call it evaluates the run
and renders the test run at the median of the position error (a few minutes on a CPU):

- `rollouts/<time>_bumper_beam_mgn_t_best_model_ep<epoch>_test/`: the metrics per run and a
  `.vtu` per state, from the initial one, with prediction, ground truth, error and the pole,
  for ParaView
- `evaluations/<time>_bumper_beam_mgn_t_evaluation/evaluation.md`: training, single step and
  rollout, with figures ([evaluate a run](../guides/evaluation.md#evaluate-a-run))
- `evaluations/<time>_animation/`: the animation

The [results](../results/bumper-beam.md) show what this run gives. A new training comes close
to them, not to the last digit, as training on a GPU is not bit for bit reproducible.

## Next steps

- [OpenRadioss runs](../guides/openradioss.md) and
  [export config](../guides/export.md#define-the-dataset-in-a-config-file): own runs.
- [Choose the features](../guides/training.md#choose-the-features): train on a part of the
  features of an export.
