---
hide:
  - toc
---

# How CrashDefNet works

CrashDefNet learns from finite element simulations how a structure deforms, and then
predicts the deformation on its own, time step by time step, on the simulation mesh. The
workflow has four stages; files on disk connect them, so every stage can be run, checked
and repeated on its own.

<div class="pipeline" markdown>

<div class="pipeline-stage" markdown>

:material-file-multiple-outline:{ .xxl }

**Raw data**

TFRecord<br>d3plot<br>VTK

</div>

<div class="pipeline-arrow"><span>export</span></div>

<div class="pipeline-stage" markdown>

:material-database-outline:{ .xxl }

**Dataset**

`dataset.db`<br>`normalizer.json`<br>`metadata.json`

</div>

<div class="pipeline-arrow"><span>train</span></div>

<div class="pipeline-stage" markdown>

:material-brain:{ .xxl }

**Training run**

`best_model.pt`<br>`history.json`

</div>

<div class="pipeline-arrow"><span>rollout</span></div>

<div class="pipeline-stage" markdown>

:material-chart-timeline-variant:{ .xxl }

**Rollout**

metrics<br>`.vtu`

</div>

<div class="pipeline-link pipeline-link-dataset"><span>explore</span></div>

<div class="pipeline-link pipeline-link-training"><span>evaluate</span></div>

<div class="pipeline-link pipeline-link-rollout"><span>evaluate · compare · check · animate</span></div>

<div class="pipeline-stage pipeline-end" markdown>

:material-chart-box-outline:{ .xxl }

**Evaluations**

reports · figures · values

</div>

</div>

| Stage | Tool | Reads | Writes |
|---|---|---|---|
| Export | `python -m dataset_preprocessor.export` | raw simulation runs | `data/<name>/`: the trajectories in a compact SQLite file, the normalization statistics, the feature layout |
| Training | `python train/train.py` | the dataset, a TOML config | `checkpoints/<run>/`: the selected model with its config and statistics, the history, test metrics |
| Rollout | `python predict/rollout.py` | a checkpoint, a dataset | `rollouts/<time>_.../`: errors per step and trajectory, `.vtu` files, `rollout_info.json` |
| Evaluation | `utils/evaluate_run.py`, `compare_paper.py`, `check_protocol.py`, `explore_data.py`, `animate_rollout.py` | runs, rollouts, datasets | `evaluations/<time>_.../`: reports, figures, values |

New designs that hold only their initial state and deck take a step before the rollout:
`predict/prepare.py` checks them against the runs a model was trained on and writes them as
a dataset that the rollout predicts ([predict new runs](../guides/prediction.md)). Every option
of every tool: [command line](../reference/cli.md); every file:
[file formats](../reference/file-formats.md).

## From simulation to graph

A simulation run is a sequence of states of one mesh. The exporter keeps the mesh once
and the node positions and result fields, such as the stress, of the selected states. When a
sample is needed, the dataset builds the graph of one time step t: the mesh nodes with their
type, the prescribed motion and, where the data have them, properties such as thickness and
material and the state before; the mesh edges with the relative positions at rest and now;
and contact edges between nodes that are close but not connected through the material. The
target is the velocity of every node to step t + 1 and its result fields there.
[Data and graphs](graphs.md)

## Learning one step

A model maps this graph to one prediction per node. Both architectures encode the
features into latent vectors, pass messages along the edges, and decode the node
latents into velocity and result fields. MeshGraphNets does this with 15 local message passing
steps; MGN-T replaces most of them with a transformer that exchanges information across
the whole structure. [Models](models.md)

Training shows the model one step at a time from the ground truth, with noise on the
positions so that it learns to correct small errors of its own, optionally with its own
predictions of the steps before as inputs; or it trains on windows of its own rollout, with
the gradients through every step. Validation rollouts during training select the model that
predicts whole trajectories best. [Training](training.md)

## Predicting a load case

The rollout starts from the ground truth of the first state and then feeds the model its
own predictions: positions advance by the predicted velocities, the contact edges are
rebuilt from the predicted geometry, and the next step is predicted. The error that
accumulates over the whole load case is what the rollout metrics measure.
[Rollout and metrics](metrics.md)

## Why autoregressive

CrashDefNet is developed specifically for autoregressive prediction. Together with the
contact modelling that goes with it, this is its basic construct and the goal of its further
development.

A crash is a process in time. What a structure does at an instant depends on how it got
there: plastic deformation is permanent, folds form one after another, and contact opens and
closes as parts meet and separate. An explicit finite element solver follows this process in
small time steps, each computed from the state before.

The autoregressive prediction has the same structure. The model learns one step, from the
state at t to the state at t + 1, on the coarser time steps of the saved states; the rollout
carries it through the whole load case on its own predictions. The final deformation is
reached through every state in between.

Contact follows from it. As every step starts from a known geometry, the contact edges are
built from that geometry: an edge appears where two bodies come close or a structure folds
onto itself, and disappears when they separate. Information crosses from one body to another
only where and when they meet, and the actuator acts on the structure only through contact
([contact edges](graphs.md#world-edge-variants-model-world_edges)).

### Time schemes

Learned crash models resolve the time dependence of a load case in three ways
([Nabian et al., 2025; Akhare et al., 2026](../about/index.md#references)):

| Scheme | The state at time t is predicted from |
|---|---|
| Autoregressive | the state one step before, step by step from the initial state |
| One-shot | the initial state, all states in one pass |
| Time-conditional | the initial state and the time t, each state on its own |

One-shot and time-conditional prediction compute every state from the initial state,
independently of the states in between. The states can be computed in parallel, and an
error in one does not carry over to the next. The path to a state is not computed: whether
a fold has formed or two parts have met by time t has to be inferred from the initial state
and t, and contact is not determined from a deformed geometry. The autoregressive prediction
computes this path, each state from the one before and contact from the geometry of that
state, as the explicit solver does. In return, an error in one state carries over into the
next. Training for the rollout addresses this.

### Training for the rollout

An autoregressive model learns from the steps of the simulation. The ways to train it differ
in the state it gets as input:

- **One step from the ground truth** (teacher forcing): every step starts from the exact
  state of the simulation. In the rollout the model starts from its own predictions, states
  it has not seen in training, and small errors can grow from step to step.
- **Noise on the inputs:** the positions get noise, and the target is corrected so that it
  leads back to the true next state; the model learns to pull small deviations back
  ([Pfaff et al., 2021](../about/index.md#references)).
- **Scheduled sampling:** part of the inputs come from the model's own predictions for the
  steps before, computed without gradients, so that they carry the drift of a short rollout.
- **Rollout training** (autoregressive rollout training): the model runs over several steps,
  up to the whole load case, on its own predictions, with the loss over every step and the
  gradients through the whole rollout (backpropagation through time). Besides recovering from
  drift, it learns how an error in one step affects the steps that follow.

All four keep the prediction causal, with contact from the geometry of every step; they
differ in how much of its own rollout the model sees during training.

### Built for the rollout

Everything in CrashDefNet is built for this prediction, and new architectures, data
structures and contact models are developed within it and judged by the rollout over the
whole load case:

- Training combines single steps from the ground truth with noise on the positions and,
  optionally, scheduled sampling over several steps; alternatively it trains on rollout
  windows with the gradients through every step. Every input state, noisy or predicted,
  gets its contact edges from its own geometry, as in the rollout
  ([training methods](training.md#training-methods)).
- Validation rollouts during training select the model that carries whole trajectories best
  ([model selection](training.md#model-selection-and-early-stopping)).
- The metrics measure the error accumulated over the whole load case
  ([rollout and metrics](metrics.md)).

## Design principles

- **Autoregressive by design.** Every stage is built for predicting a whole load case on the
  model's own predictions, with contact found anew at every step.
  [Why autoregressive](#why-autoregressive)
- **One graph builder.** Export, training, rollout and evaluation build their graphs with
  the same functions, so every stage sees the same graph.
- **Physics in the graph.** Information travels where it travels physically: along the
  material and through contact. Contact edges link bodies and the folds of a structure,
  not neighbours inside the same material, and the actuator acts on the structure only
  through contact.
- **Refined for the physics, not for any data.** Further refinements of the data
  structures, and of the architectures built on them, aim at modelling the physical
  process of crash deformation better and in more detail. A generic, statistical data
  level that takes any data column is not a goal: an input without a physical role in
  the deformation, such as an ID, a bookkeeping value or a quantity that does not affect
  the response, adds noise to the dataset, makes training needlessly stochastic and
  invites overfitting. Every input stands for a physical quantity and enters the graph
  where it acts. [General data](graphs.md#general-meshes-and-run-parameters)
- **Scalable data layout.** Export, training and rollout need memory for one run or one
  sample, not for the dataset; datasets of a few hundred GB need no extreme RAM.
  [Scaling](scaling.md)
- **Fair comparison.** Architectures are compared under a fixed protocol: same data,
  seed, budget and recipe; only the model changes.
  [Comparison protocol](../guides/comparison-protocol.md)
- **Lean code.** Plain PyTorch and PyTorch Geometric, readable end to end; a new
  architecture is one file and a few lines.
  [Add an architecture](../guides/adding-an-architecture.md)
