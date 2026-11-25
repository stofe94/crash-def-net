# Predict new runs

A trained model predicts a crash from the undeformed state, step by step over the load
case. For a new design it needs no simulation result, only what defines the run at t0. `prepare.py`
checks first whether the model applies to a new run at all, compared with the runs it was
trained on, and then writes the run as prediction data; [`rollout.py`](rollout.md) predicts
it.

```bash
python predict/prepare.py --checkpoint checkpoints/<run>/best_model.pt \
    --raw-dir <folder with the new runs> --out-dir data/<name> [--check-only] [--force]
python predict/rollout.py --checkpoint checkpoints/<run>/best_model.pt \
    --data-dir data/<name> --split test --export-vtu
```

## What a new run needs

One folder per run below `--raw-dir`, with a `d3plot` that holds at least the initial state
and the deck next to it. The runs are read with the reader settings of the training data
(its `dataset.db`): the same part patterns and roles, features, units and contact settings
([OpenRadioss runs](openradioss.md), [LS-DYNA runs](ls-dyna.md)).

| Information | Where it comes from |
|---|---|
| Mesh at t0: nodes, elements, parts | the first state of the `d3plot` |
| Role of every part: deformable, free rigid body, actuator | the part patterns of the training data |
| Static node features, e.g. thickness, yield stress | the deck, as for the training data |
| Connections | ties of the deck (`/INTER/TYPE2`) |
| Initial velocity | the deck (`/INIVEL/TRA`, `*INITIAL_VELOCITY`); times the frame interval of the training data, the motion before the first frame |
| Impactor | its mesh, at rest |

The time step and the number of frames are those of the training data; `--num-steps`
chooses fewer or more frames (more: beyond the time span the model was trained on). The
frame interval comes from the raw data of the training data, or from `--frame-interval`
where these are not at hand. Prediction data is prepared from d3plot runs only.

## The applicability check

Every run is compared with the train runs of the model (the train split of the export
it was trained on, `--train-data-dir`, by default the one its training config names).
Each check is `ok`, `outside` or `error`:

- **Errors**, the model cannot take the run: other free rigid bodies, an actuator where the
  training data has none or the other way round, a static node feature or run parameter it
  does not have, no initial velocity where the train runs have one, or prescribed motion
  (actuator or clamped nodes that move in the training data, or an actuator with an
  initial velocity): it would have to be given and is not supported.
- **Ranges of the train runs**: the extent of the structure, the median length of its mesh
  edges, the initial speed and its direction, the values of every static node feature,
  position and extent of the actuator relative to the structure and the clearance between
  them at t0, and the number of frames. A value outside the range of the train runs is an
  extrapolation of the model.
- **What the model does not see**: the mass of each free rigid body and Young's modulus and
  density of the parts. They are no inputs, so the training data fixes them; compared where
  the deck states them (OpenRadioss starter decks), else `not checked`.
- **Parts**: parts that no pattern of the training data matches are left out of the
  prediction, and parts the model has not seen are named; both are to be reviewed
  (`review`), since a display model may rightly be left out, a part of the structure not.

The verdict of a run is `not applicable` (an error), `extrapolation` (a value outside) or
`within the training data`, with `parts to review` where that applies. A run with an error
is not prepared unless `--force` is given; `--check-only` writes nothing but
`applicability.json`. Within every range is no guarantee: the train runs span a region,
and a combination of values inside the ranges may still be new to the model.

## Prediction data and its rollout

The run is written as a dataset in the format of an export, in the split `test`
([file formats](../reference/file-formats.md#dataset-folder)): the initial state for every
frame, so that the actuator stays at rest, the motion before the first frame from the
initial velocity, the layout and the normalizer of the training data, and no ground truth
(`ground_truth: false` in `metadata.json`). The rollout predicts every later frame as for
any dataset, but has nothing to compare with: the rollout summary and the `.vtu` files hold
the prediction only, without errors, and evaluation and animation are left out
([output](rollout.md#output)). ParaView opens the `.vtu` series.
