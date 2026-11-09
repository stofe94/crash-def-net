# Purpose

CrashDefNet is built for one application, the learned prediction of crash deformation, and
is developed further specifically for it. Its basic construct is the autoregressive
prediction, with contact found anew in the predicted geometry at every step
([why autoregressive](concepts/overview.md#why-autoregressive)). On it, data structures and
architectures are refined to model the physics of a crash in more detail
([design principles](concepts/overview.md#design-principles)). It is not meant to become a
general physics AI framework that also covers other problem classes, such as fluid dynamics.

## Highlights

- **Autoregressive by design.** The model predicts one step and carries the whole load case
  on its own predictions. Training with noise on the positions, model selection by
  validation rollouts and the metrics over the whole load case are built for it
  ([why autoregressive](concepts/overview.md#why-autoregressive),
  [rollout and metrics](concepts/metrics.md)).
- **Complete workflow.** From raw data through training with model selection on several
  criteria to rollout, rollout metrics, comparison reports, data exploration, ParaView export
  and animations ([evaluation](guides/evaluation.md),
  [data exploration](guides/explore-data.md)).
- **Quick to try, with a graphical app.** A page in the browser leads through a first run
  without typing commands; at an early stage and to be developed further, it can already
  train with any configuration ([the app](guides/app.md)). On Google Colab's free GPU the
  quick start needs no installation ([quick start](getting-started/quick-start.md)).
- **New architectures, quickly.** A new model on the same graph is one file plus a few lines
  in the config and the model factory ([guide](guides/adding-an-architecture.md)).
- **Fair comparison under a fixed protocol.** Data, seed, budget and recipe are fixed, and
  `check_protocol.py` checks every run against them. New architectures are compared with a
  MeshGraphNets baseline on the same contact edges and with the published results of the
  MeshGraphNets paper ([protocol](guides/comparison-protocol.md#baselines),
  [deforming plate results](results/deforming-plate.md)).
- **Physics in the graph.** Every input stands for a physical quantity and enters where it
  acts ([design principles](concepts/overview.md#design-principles)):
    - Contact and self-contact are rebuilt from the predicted geometry at every step, only
      where bodies touch or a structure folds onto itself
      ([contact edges](concepts/graphs.md#world-edge-variants-model-world_edges)).
    - Parts are components, with contact rules for the pairs that can touch
      ([components](concepts/graphs.md#components-and-contact-rules)).
    - Run parameters enter as a global input
      ([global block](concepts/models.md#global-block-run-parameters)), node properties such
      as thickness as static features, any result fields as targets
      ([general data](concepts/graphs.md#general-meshes-and-run-parameters)).
    - Rigid parts are given actuators or free rigid bodies that keep their shape
      ([free rigid parts](concepts/graphs.md#free-rigid-parts)).
- **MeshGraphNets and MGN-T, implemented from the papers.** The authors of MGN-T have not
  published code; CrashDefNet implements it independently from the paper and documents every
  assumption that fills a gap ([models](concepts/models.md#assumptions-where-the-paper-leaves-gaps)).
- **OpenRadioss runs and further raw formats (early stage).** The exporter reads OpenRadioss
  runs with their starter decks, verified on a real crash example, a bumper beam against a
  pole ([results](results/bumper-beam.md), [tutorial](getting-started/bumper-beam.md),
  [OpenRadioss runs](guides/openradioss.md)). LS-DYNA runs and VTK time series are read too,
  with general meshes, parts, materials, result fields and units; an export config defines
  the dataset, the training config chooses what a run uses ([LS-DYNA runs](guides/ls-dyna.md),
  [VTK time series](guides/vtk.md),
  [choose the features](guides/training.md#choose-the-features),
  [verification status](concepts/graphs.md#general-meshes-and-run-parameters)).
- **Built for large datasets.** Streaming, resumable export and lazy loading keep the memory
  independent of the dataset size, up to a few hundred GB; micro-batches train large models
  on large meshes ([scaling](concepts/scaling.md)).
- **Documented as a reference.** Tutorials, guides, concepts and a reference of every command
  line option, config key and file format ([command line](reference/cli.md),
  [configuration](reference/configuration.md), [file formats](reference/file-formats.md)).
- **Lean, open and local.** Plain PyTorch and PyTorch Geometric under AGPL-3.0, with a core
  that can be read end to end. Everything runs on your own machines, also on a CPU, so
  confidential simulation data stays on site
  ([installation](getting-started/installation.md#requirements)).
