# CrashDefNet

**Learned deformation prediction for crash, from raw simulation data to
benchmark-validated predictions.** A complete, lean PyTorch workflow for research:
dataset export, training, autoregressive rollout and evaluation against published
results. It also works with industrial crash data (LS-DYNA `d3plot`, VTK `.vtu` / `.vtp`
/ `.vtk`) and large meshes. New model architectures plug in quickly, so the latest
research can be put to work and compared under identical conditions.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)
![PyTorch + PyG](https://img.shields.io/badge/PyTorch-PyG-ee4c2c.svg)
![Version 0.3.0](https://img.shields.io/badge/version-0.3.0-lightgrey.svg)

![Rollout of an unseen test geometry: ground truth, MGN-T and MeshGraphNets](docs/assets/rollout_test_traj0037.gif)

*Autoregressive rollout on a test geometry that was not part of training. Left: FEM
ground truth from DeepMind's `deforming_plate` (Pfaff et al., 2021). Middle and right:
the two architectures in CrashDefNet. Colour: von Mises stress.*

---

## What is CrashDefNet?

CrashDefNet (**Crash Deformation Network**) is a deep-learning research tool for
**predicting structural deformation**, designed for scientific purposes and
investigations. It learns from finite element simulations and predicts, step by step on
the simulation mesh, how each node moves and which stress it carries. The full chain is
covered: from the raw simulation data through dataset export, training and
autoregressive rollout to the evaluation against published results.

Two things define it: it covers the **complete workflow** that an application needs,
and it stays **lean enough** that new model architectures can be integrated and
compared quickly. The application it is built for is **crash deformation**.

This release (v0.3.0) is developed and validated on DeepMind's open
**`deforming_plate`** dataset: an actuator impacts and deforms plates of varying
geometry. It is a simplified crash scenario with contact and large deformations, and
the approach is built to carry over to more complex crash cases. The release ships two
graph neural network architectures that run through the same pipeline and can be
compared directly:

- **MeshGraphNets (MGN)**: the original architecture for mesh-based simulation
  (Pfaff et al., ICLR 2021)
- **MeshGraphNet-Transformer (MGN-T)**: the current state of research (Iparraguirre
  et al., 2026). It combines local message passing with a physics-attention
  transformer as a global processor and targets exactly the setting of crash: impact
  dynamics with self-contact and plasticity on industrial-scale meshes

## Highlights

- **Contact and self-contact where they physically occur.** Positions follow from the
  predicted velocity, and the contact edges are rebuilt from the predicted geometry at
  every step: between bodies and where a structure folds onto itself
  (`self_contact_aware`). The contact search scales to crash meshes
  ([data.md](docs/data.md#world-edge-variants-model-world_edges)).
- **Industrial crash data.** Besides DeepMind's TFRecords, the exporter reads LS-DYNA
  runs (`d3plot` with keyword deck: mixed meshes, rigid parts, prescribed motion, initial
  velocities, rigid walls) and VTK time series (`.vtu` / `.vtp` / `.vtk`, e.g. from
  ParaView). Both readers are checked against the benchmark export
  ([data.md](docs/data.md#second-raw-format-ls-dyna-runs-d3plot)).
- **MGN-T, implemented from the paper.** The authors of MGN-T have not published code.
  CrashDefNet implements it from the paper, including the physics attention of
  Transolver / Transolver++, and documents every assumption that fills a gap
  ([models.md](docs/models.md#assumptions-where-the-paper-leaves-gaps)). It is an
  independent implementation, not affiliated with or endorsed by the authors.
- **New architectures, quickly.** A new model is one file plus a few lines in the config
  and the model factory; data, training, rollout and evaluation work unchanged
  ([guide](docs/adding-an-architecture.md)).
- **Fair comparison under a fixed protocol.** Dataset, seed, budget and training recipe
  are fixed; `check_protocol.py` checks every run against them and compares it with the
  MeshGraphNets baseline trajectory by trajectory, next to the metrics of the papers
  ([protocol](docs/comparison-protocol.md)).
- **Scales to vehicle-crash data.** The export streams and resumes, stores the runs of a
  design study on the same mesh once, and lazy loading keeps the memory independent of
  the dataset size, up to a few hundred GB ([data.md](docs/data.md#large-datasets)).
- **Complete and traceable workflow.** From raw data through training with model
  selection on several criteria to rollout, paper metrics, comparison reports, ParaView
  export and animations; every rollout records its checkpoint, dataset, config and git
  commit
  ([rollout and evaluation](docs/rollout-and-evaluation.md#output)).
- **Lean, open and local.** Plain PyTorch and PyTorch Geometric: the core framework is
  about 4,200 lines of Python, readable end to end, under Apache 2.0. Everything runs on
  your own machines, so confidential simulation data stays on site. With a small dataset
  and a reduced model, a training run takes minutes to a few hours on a CPU, so the tool
  can be tried on simple hardware ([installation](docs/installation.md#requirements)).

## Background

CrashDefNet is developed by **Felix Stocker**, a data scientist with 4.5 years of
experience in CAE and crash simulation. For one year of that, the work focused
entirely on research and development in deep learning for 3D geometries.

CrashDefNet builds on this experience as a **self-developed, independent tool:
complete in function, lean in code.**

- **Complete workflow.** Everything from the raw simulation data to benchmark-validated
  predictions and reports is part of the tool.
- **Fast integration.** A new architecture goes from the paper into a trained,
  evaluated model within one pipeline. MGN-T, published in 2026, is the first example.
- **Fair comparison.** Every architecture runs on the same data, training loop and
  metrics, next to MeshGraphNets and the numbers of the papers.
- **Full control.** Every part, from data preparation to model and evaluation, can be
  changed as needed; the code base is compact and readable.
- **Focused on crash.** The application is crash deformation, not a general-purpose
  platform.

## Roadmap

- Integrate and test further recent architectures as they are published
- More complex crash scenarios (dynamic impact, plasticity), with all architectures
  side by side
- Training on the full `deforming_plate` training split at stride 1, as in the papers
- Training on subgraphs with a halo, together with multi-GPU support, for full-vehicle
  meshes with millions of nodes: the subgraphs bound the memory of each GPU, several GPUs
  the training time (outlook in [data.md](docs/data.md#outlook-training-on-subgraphs-and-multiple-gpus))
- A published test suite for every stage of the pipeline: unit tests, equivalence of
  the raw formats, end-to-end runs and regression against reference models
- A professional documentation: how CrashDefNet works, how it is used and what it can
  do, structured as user guide and reference (e.g. a documentation website), instead of
  the Markdown pages in [`docs/`](docs/) and this README of today

## First results (preliminary)

A first comparison on a reduced setup: 28 training geometries, stride 8, 4 unseen
test geometries, one RTX A4000. Both models share the data, seed, budget and training
recipe of the [comparison protocol](docs/comparison-protocol.md), each in the
configuration of its paper: MeshGraphNets as the baseline, MGN-T as in arXiv v4
([configuration](docs/results.md#mgn-t-configuration)).

| | MGN-T | MeshGraphNets |
|---|---|---|
| Parameters | **0.51 M** | 3.85 M |
| Training time | **1 h 09 min** | 3 h 41 min |
| Position error, RMSE-all (4 test geometries) | **35.1 ± 16 mm** | 44.1 ± 21 mm |
| Position error, RMSE-all (without the outlier) | 7.0 ± 1.7 mm | **3.0 ± 0.6 mm** |
| Stress error, RMSE-all (4 test geometries) | 34.2 ± 11 kPa | **16.5 ± 4.1 kPa** |

- **Efficiency.** MGN-T has 7.5× fewer parameters and needs about 3.5× less training
  time per epoch.
- **Accuracy.** MeshGraphNets is more accurate in stress on all four test geometries
  and in position on three of them.
- **The outlier.** On the hardest geometry MGN-T has the lower position error. This
  geometry deforms by up to 490 mm, more than anything in the training data.

These runs used far less data and training than the paper, so the numbers are not a
reproduction of the paper's results. A scientific evaluation is planned for later
releases. Setup, all metrics, the comparison with the paper, a short discussion and next
steps: **[docs/results.md](docs/results.md)**.

![Error per test trajectory](docs/assets/error_per_trajectory.png)

## Installation

Requirements: Python 3.10–3.12, Linux or WSL2 (tested with Python 3.12). A GPU is
optional.

```bash
git clone https://github.com/stofe94/crash-def-net.git
cd crash-def-net

# conda (recommended)
conda env create -f environment.yml
conda activate crashdefnet

# or pip / venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # GPU (CUDA 12.6): requirements-gpu.txt
```

Raw data: [`meta.json`](https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/meta.json) and [`test.tfrecord`](https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/test.tfrecord) of
DeepMind's `deforming_plate` (~0.8 GB, 100 trajectories). For now only the test file
is used, for reasons of size (the training file holds 1,000 trajectories in 9.9 GB);
the exporter splits its 100 trajectories into separate train, validation and test
sets. The files are not part of the repository:

```bash
wget -P datasets/deforming_plate/raw_dataset \
    https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/meta.json \
    https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/test.tfrecord
```

Details, GPU setup and troubleshooting: [docs/installation.md](docs/installation.md).

## Quick start

A complete run on the CPU with a small dataset (8/2/2 trajectories) and MGN-T in a
minimal size. Each epoch takes about 25 s, the training of 5 epochs about two minutes.

```bash
# 1) Export a small dataset (~3 s, ~11 MB) -> data/deforming_plate_s10_small/
python -m dataset_preprocessor.export

# 2) Train with the example config (MGN-T; for MeshGraphNets set architecture = "meshgraphnets")
python train/train.py --config config.example.toml

# 3) Roll out the trained model on the test split -> rollouts/<time>_<run>_<checkpoint>_test/
python predict/rollout.py --checkpoint checkpoints/example_run/best_model.pt \
    --data-dir data/deforming_plate_s10_small --split test

# 4) Loss curves -> checkpoints/example_run/loss_curves.png
python utils/plot_losses.py --run-dir checkpoints/example_run

# 5) Compare all finished rollouts with the MGN-T paper -> evaluations/<time>_paper_comparison/
python utils/compare_paper.py
```

## Docker (GPU)

The image contains the code only; the dataset is mounted from the host.

```bash
cp .env.example .env          # set DATASET_DIR (and optionally CONFIG)
docker compose build
docker compose up             # trains with the config from .env on the GPU
```

Export and rollout inside the container, CUDA versions and hints for small GPUs:
[docs/docker.md](docs/docker.md).

## Documentation

| Topic | |
|---|---|
| [Installation](docs/installation.md) | conda / pip, GPU, raw data, troubleshooting |
| [Data](docs/data.md) | export, raw formats (TFRecord, LS-DYNA, VTK), dataset format, graph features, optional history |
| [Models](docs/models.md) | MeshGraphNets and MGN-T, documented assumptions |
| [Adding an architecture](docs/adding-an-architecture.md) | how a new model plugs into the pipeline |
| [Training](docs/training.md) | training loop, noise, validation, early stopping, config reference |
| [Rollout and evaluation](docs/rollout-and-evaluation.md) | rollout, paper metrics, comparison report, ParaView, animation |
| [Docker](docs/docker.md) | GPU training in a container |
| [Results](docs/results.md) | first comparison MGN vs MGN-T vs paper |
| [Comparison protocol](docs/comparison-protocol.md) | how a new architecture is compared with the MeshGraphNets baseline |

## Project structure

```
config.example.toml          example training config (CPU quick test); copy and adapt for own runs
dataset_preprocessor/
  export.py                  CLI: raw data (TFRecord, d3plot, VTK) -> dataset.db + normalizer.json + metadata.json
  dataset.py                 DeformingPlateDataset: one graph per time step
  graph_features.py          feature layout, mesh and world edges, consistent position noise
  tfrecord_to_d3plot.py      CLI: DeepMind TFRecord -> LS-DYNA runs (d3plot + model.k)
  d3plot_reader.py           LS-DYNA reader (d3plot + keyword deck), raw data for --raw-format d3plot
  tfrecord_to_vtk.py         CLI: DeepMind TFRecord -> VTK time series (.vtu / .vtk)
  vtk_reader.py              VTK time series reader (.vtu / .vtp / .vtk), raw data for --raw-format vtk
models/
  meshgraphnets.py           MeshGraphNets (mesh + world edges)
  mgn_t.py                   MeshGraphNet-Transformer (physics attention)
  residual.py                optional residual target wrapper
train/                       training CLI, config, trainer, early stopping
predict/rollout.py           autoregressive rollout, paper metrics, .vtu export
protocols/                   training configs of the comparison protocols
utils/                       normalizer, loss curves, paper comparison, protocol check,
                             rollout animation, time zone
docker/, Dockerfile,         GPU training in a container
docker-compose.yml
docs/                        documentation and figures
```

Not versioned: `datasets/` (raw data), `data/` (exports), `checkpoints/`, `rollouts/`,
`evaluations/`.

## References

- T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, P. W. Battaglia. *Learning Mesh-Based
  Simulation with Graph Networks.* ICLR 2021. [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
- M. M. Iparraguirre, I. Alfaro, D. González, E. Cueto. *MeshGraphNet-Transformer:
  Scalable Mesh-Based Learned Simulation for Solid Mechanics.* 2026.
  [arXiv:2601.23177v4](https://arxiv.org/abs/2601.23177v4)
- H. Wu, H. Luo, H. Wang, J. Wang, M. Long. *Transolver: A Fast Transformer Solver for
  PDEs on General Geometries.* ICML 2024. [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)
  (physics attention, used in the global processor of MGN-T)
- H. Luo et al. *Transolver++: An Accurate Neural Solver for PDEs on Million-Scale
  Geometries.* 2025. [arXiv:2502.02414](https://arxiv.org/abs/2502.02414)
  (eidetic states of the physics attention, used in MGN-T)
- Dataset: `deforming_plate` from DeepMind's
  [MeshGraphNets release](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
  (downloaded separately, not redistributed)

## Acknowledgements

Thanks to DeepMind for publishing MeshGraphNets together with its datasets, and to the
authors of MGN-T for the detailed description of their architecture.

## Citation

If you use CrashDefNet, please cite it (see [CITATION.cff](CITATION.cff)):

```bibtex
@software{stocker_crashdefnet_2026,
  author  = {Stocker, Felix},
  title   = {CrashDefNet: Learned Deformation Prediction on Simulation Meshes},
  year    = {2026},
  version = {0.3.0},
  url     = {https://github.com/stofe94/crash-def-net}
}
```

## Contributing

Suggestions are welcome as issues or pull requests; see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE). Copyright 2026 Felix Stocker.

## Disclaimer

CrashDefNet is a research tool. Its outputs are predictions of learned models and are
no substitute for validated crash simulation or physical testing; they must not be used
for safety-relevant design decisions without independent verification. The software is
provided "as is", without warranty of any kind; see the [License](LICENSE).

## Author

**Felix Stocker**, data scientist (CAE, crash simulation, deep learning on 3D geometries)

- E-mail: [felix.stocker@sto-eng.de](mailto:felix.stocker@sto-eng.de)
- LinkedIn: [linkedin.com/in/felixstocker](https://www.linkedin.com/in/felixstocker)
- Skill profile: [stofe94.github.io/skill-profile](https://stofe94.github.io/skill-profile/)
