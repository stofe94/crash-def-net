# Glossary

Terms from finite element simulation and from machine learning as CrashDefNet uses them.

**Actuator**
: The rigid body that impacts the plate in `deforming_plate` (node type 1). Its motion is
  prescribed: the model gets its next velocity as input and predicts nothing for it. In
  LS-DYNA and OpenRadioss runs, rigid parts and nodes with prescribed motion become actuator
  nodes, as do the parts with the role `actuator` in an export config.
  [Data and graphs](../concepts/graphs.md#the-benchmark)

**Baseline (original, comparable)**
: The two MeshGraphNets references: the *original* is the published result of the
  MeshGraphNets paper (`world_edges = "all"`), the *comparable baseline* is an own
  MeshGraphNets run under the comparison protocol with `self_contact_aware`, the
  reference for new architectures.
  [Baselines](../guides/comparison-protocol.md#baselines)

**Body**
: A connected part of the mesh. `contact` world edges link only nodes of different
  bodies.

**Checkpoint**
: A saved training state (`.pt`): weights, training config, dataset metadata and
  normalizer. `best_model.pt` holds the selected model.
  [File formats](file-formats.md#checkpoints)

**Clamped node**
: A fixed boundary node (node type 3). In LS-DYNA runs: nodes fixed in x, y and z by
  `*BOUNDARY_SPC_*`, in OpenRadioss runs by `/BCS`; in an export config the parts with the
  role `fixed`.

**Comparison protocol**
: The fixed dataset, seed, budget and training recipe under which architectures are
  compared; only the model and declared variations change.
  [Comparison protocol](../guides/comparison-protocol.md)

**Component**
: A part of the raw data (LS-DYNA and OpenRadioss runs, VTK time series) under its name;
  contact rules and the roles of an export config refer to components. The models do not
  see the names. [Components](../concepts/graphs.md#components-and-contact-rules)

**Contact radius (r_W)**
: Distance below which two nodes that are not linked through the material get a world
  edge; 0.03 m in `deforming_plate` (`collision_radius` in `meta.json`).

**Criterion (selection criterion)**
: A metric recorded at every evaluation during training (`val_loss`, `rollout_mse`,
  `rollout_pos_rmse`, `rollout_pos_rmse_final`). The model selection uses all criteria in
  `criteria` at once. [Training](../concepts/training.md#evaluations)

**d3plot**
: The binary result file of LS-DYNA: geometry and states of a run, with element results.
  Read with the keyword deck (`*.k`) for parts and boundary conditions; OpenRadioss runs
  converted to d3plot with their starter deck (`*_0000.rad`).
  [OpenRadioss runs](../guides/openradioss.md), [LS-DYNA runs](../guides/ls-dyna.md)

**Early stopping**
: Ending a training run when no criterion has reached a new low for `patience`
  evaluations. [Training](../concepts/training.md#model-selection-and-early-stopping)

**Encoder, processor, decoder**
: The three stages of MeshGraphNets and MGN-T: MLPs map the features into latents, the
  processor passes messages (and, in MGN-T, runs the transformer), an MLP maps the node
  latents to the targets. [Models](../concepts/models.md)

**Evaluation**
: In training: the epochs at which the criteria are computed (every epoch, or with
  rollout validation every `every_n_epochs`). Only evaluated epochs can be selected.

**Export**
: The conversion of raw simulation data into a dataset folder (`dataset.db`,
  `normalizer.json`, `metadata.json`). [Export a dataset](../guides/export.md)

**Feature selection**
: The static node features, run parameters and result fields of an export that a run uses,
  named in `[data]` of its config. [Choose the features](../guides/training.md#choose-the-features)

**Fingerprint**
: A SHA-256 over the content of a `dataset.db` (settings and hashes of every trajectory
  and mesh). Rollouts record it; the protocol check compares it.

**Frame**
: One stored state of a trajectory. The export selects every `time_stride`-th raw frame.

**Free rigid part**
: A rigid part whose motion results from the crash, e.g. a barrier with mass: the model
  predicts its motion, and the rollout keeps it rigid.
  [Free rigid parts](../concepts/graphs.md#free-rigid-parts)

**Generalization gap**
: `max(1, val loss / train loss without noise)`, recorded at every evaluation; it takes no
  part in selection or stopping.

**History layout**
: An export with `--history 1`: the previous velocity and stress become node inputs.
  Needed for the residual target, scheduled sampling and stress noise; not in the paper.
  [Optional history](../concepts/graphs.md#optional-history-history-1-not-in-the-paper)

**Latent**
: The learned vector a model keeps per node or edge between its stages; its width is
  `latent_size`.

**Lazy loading**
: `[data] lazy = true`: the frames stay in `dataset.db` and are read per sample, so memory
  does not grow with the dataset. [Scaling](../concepts/scaling.md#loading-during-training)

**Loss mask**
: The nodes on which loss and errors are computed: all except the actuator.

**Mesh edge**
: An edge of a cell of the simulation mesh, in both directions. Its features hold the
  relative position at rest and in the current state.

**Message passing**
: One step in which every edge computes a message from its two nodes and every node
  aggregates the messages of its edges (sum by default). Information travels one edge per
  step.

**MeshGraphNets (MGN)**
: The graph network for mesh-based simulation by Pfaff et al. (ICLR 2021): 15 message
  passing steps over mesh and world edges. [Models](../concepts/models.md#meshgraphnets-meshgraphnets)

**MeshGraphNet-Transformer (MGN-T)**
: The architecture by Iparraguirre et al. (2026): local message passing, a transformer
  with physics attention over the structure, local message passing again.
  [Models](../concepts/models.md#meshgraphnet-transformer-mgn_t)

**Micro-batch**
: A part of a batch computed in one forward pass; several micro-batches make one
  optimizer step with the same result as the whole batch (`micro_batch_size`).

**Model selection**
: The choice of the evaluation whose weights become `best_model.pt`: the one closest to
  the best value in all criteria at once.

**Node type**
: Normal (0, free), actuator (1, prescribed motion) or clamped (3, fixed); one-hot encoded
  as node input.

**Normalizer**
: Mean and standard deviation per feature column, fitted on the train split; models see
  standardized inputs and predict standardized targets.
  [File formats](file-formats.md#normalizerjson)

**Physics attention, tokens**
: The attention of MGN-T (after Transolver): the nodes are softly assigned to P tokens,
  attention runs over the tokens, and the result is projected back onto the nodes. The
  cost grows linearly with the number of nodes. [Models](../concepts/models.md#meshgraphnet-transformer-mgn_t)

**Pooled**
: A metric over several trajectories computed as in the papers: squared errors averaged
  over all nodes, steps and trajectories, then the root. Trajectories with large errors
  weigh more than in the median. [Rollout metrics](../concepts/metrics.md#rollout-metrics)

**Position noise**
: Gaussian noise on the current positions of the free nodes during training
  (`world_noise_std`), with the target corrected, so that the model learns to recover from
  its own errors in a rollout.

**Residual target**
: `residual_target = true`: the model predicts the change of a target against its
  previous value instead of the value itself; needs the history layout.

**RMSE-1, RMSE-all, R-RMSE**
: The rollout metrics: root mean square error of one step from the ground truth, of the
  whole rollout, and relative to the largest value of the trajectory.
  [Rollout metrics](../concepts/metrics.md#rollout-metrics)

**Rollout (autoregressive)**
: Applying the model to its own predictions, step after step over a whole trajectory,
  starting from the ground truth at step 0. [Rollout and metrics](../concepts/metrics.md)

**Rollout validation**
: A rollout of the val split during training; its errors are selection criteria.

**Run parameters**
: Values that describe a whole run, e.g. the impact velocity, given in `globals.json`;
  the global input `u` of the models, which then have a
  [global block](../concepts/models.md#global-block-run-parameters).

**Sample**
: One training example: the graph of frame t with the targets of frame t + 1. A
  trajectory with n frames gives n − 1 samples.

**Scheduled sampling**
: Replacing the previous-state inputs of a part of the nodes by the model's own
  prediction during training; needs the history layout.

**Self-contact**
: Contact between two parts of the same body, e.g. the folds of a crushed structure.
  `self_contact_aware` world edges represent it.

**Split**
: The train, val (validation) or test part of a dataset; each holds whole trajectories.

**Stride (time stride)**
: Raw simulation steps between two exported frames (`--time-stride`); the paper uses
  every step.

**Trajectory**
: One simulation run: a geometry with its mesh, followed over the load case. In
  `deforming_plate`, every trajectory has its own geometry.

**von Mises stress**
: The scalar equivalent stress of a stress tensor; the stress target of CrashDefNet, in the
  stress unit of the data (Pa for `deforming_plate`).

**World edge (contact edge)**
: An edge between two nodes closer than the contact radius in the current state that are
  not linked through the material. Rebuilt from the positions at every step.
  [World edge variants](../concepts/graphs.md#world-edge-variants-model-world_edges)
