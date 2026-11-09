# Installation

## Requirements

- Linux, also under WSL2, or Windows (tested: Ubuntu under WSL2 with Python 3.12; Windows
  without WSL2 on the CPU, [notes](#windows-without-wsl2))
- Python 3.10–3.12
- Optional: an NVIDIA GPU with a current driver (CUDA 12.x). Everything also runs on
  the CPU, so CrashDefNet can be tried on simple hardware: the training of the quick start
  takes about two minutes.
- Large datasets need no machine with much RAM: the export and lazy loading keep only
  one block of a run or one sample in memory ([scaling](../concepts/scaling.md)).
- OpenRadioss and LS-DYNA runs need `lasso-python`, VTK files `pyvista`
  ([dependencies](#dependencies)).
- Everything runs locally: once installed, export, training, rollout and evaluation
  need no network connection, cloud service or account, so simulation data never
  leaves your machines.

Times in this documentation that name no GPU were measured on a 6-core desktop CPU
(AMD Ryzen 5 5600GE, 5 GB RAM under WSL2):

| Config (quick start dataset of `deforming_plate`, 8/2/2 trajectories) | Parameters | Per epoch | 5 epochs | Memory |
|---|---|---|---|---|
| `examples/quick_start/train.toml` (MGN-T, minimal size) | 50 k | ~22 s | ~2 min | 0.6 GB |
| same with `architecture = "meshgraphnets"` (latent 64, 8 steps) | 0.53 M | ~2 min | ~10 min | 1.2 GB |

## Get the code

```bash
git clone https://github.com/stofe94/crash-def-net.git
cd crash-def-net
```

## Python environment

### conda (recommended)

```bash
conda env create -f environment.yml
conda activate crashdefnet
```

`environment.yml` installs Python from conda-forge and everything with compiled
extensions (numpy, PyTorch, PyG, pyvista, matplotlib) from pip. Mixing conda and pip
builds of such packages leads to libstdc++ conflicts on Linux/WSL and to two OpenMP
runtimes on Windows.

For a GPU, change the index URL in `environment.yml` from `.../whl/cpu` to
`.../whl/cu126` (or the CUDA version of your driver) before creating the environment.

### pip / venv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU
pip install -r requirements-gpu.txt      # GPU, CUDA 12.6 (other versions: change the index URL)
pip install -e ".[dev]"                  # optional: package install + pytest, ruff
```

### Windows without WSL2

Before every release the whole test suite also runs on Windows without WSL2: export of
every raw format, training, rollout, evaluation, animation and the app, on
the CPU. A GPU under Windows is not tested.

### Dependencies

| Package | Purpose |
|---|---|
| `torch`, `torch-geometric` | models, training, graph batching |
| `numpy`, `tfrecord` | reading DeepMind's raw data during export |
| `toml`, `tqdm` | configs, progress bars |
| `tzdata` (Windows only) | time zone of the timestamps; Linux has it built in |
| `pyvista` (optional) | `.vtu` export of the rollout for ParaView, rollout animation; third raw format: VTK time series ([guide](../guides/vtk.md)) |
| `matplotlib` (optional) | loss curves, comparison with the paper, rollout animation |
| `lasso-python` (optional) | d3plot raw format: [OpenRadioss runs](../guides/openradioss.md) and [LS-DYNA runs](../guides/ls-dyna.md) |
| `zensical` (optional) | building the documentation website locally: `pip install -e ".[docs]"`, then `zensical serve -o` |

## GPU machines and virtual machines

CrashDefNet is ready to train on any machine with an NVIDIA GPU, also on a rented virtual
machine with the hardware configuration a run needs. The [Docker image](../guides/docker.md)
holds CUDA, PyTorch and PyG, so the machine itself needs only the NVIDIA driver and Docker.

## Raw data

CrashDefNet uses DeepMind's `deforming_plate` dataset from the
[MeshGraphNets release](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets).
It is not part of the repository and comes under DeepMind's terms, not CrashDefNet's
license ([details](../concepts/graphs.md#the-benchmark)). The example config and the quick start
need only [`meta.json`](https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/meta.json) and [`test.tfrecord`](https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/test.tfrecord)
(~0.8 GB, 100 trajectories); the exporter splits its trajectories into separate train,
validation and test sets. The folder address itself cannot be browsed; only the links to
the files work.

```bash
wget -P datasets/deforming_plate/raw_dataset \
    https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/meta.json \
    https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/test.tfrecord
```

The [comparison protocol](../guides/comparison-protocol.md) and the [results](../results/deforming-plate.md) take each
split from DeepMind's original one and also need the training split
([`train.tfrecord`](https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/train.tfrecord), 1,200 trajectories, ~9.9 GB) and
[`valid.tfrecord`](https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/valid.tfrecord) (100 trajectories, ~0.8 GB):
`--sources train.tfrecord valid.tfrecord test.tfrecord`, with `--select-seed`
drawing the trajectories at random ([options](../guides/export.md#choose-trajectories-and-frames)).

## Quick start on the command line

The whole workflow on a small example, one command after the other: export a dataset, train
a model, roll it out on plates it has never seen, and look at the result. The setup is the
absolute minimum to show that the workflow runs (8 training plates, every 10th time step, a
model with 50,000 parameters) and is far from a real training, which usable results need
([first training](first-training.md), [deforming plate results](../results/deforming-plate.md)).

Needs `meta.json` and `test.tfrecord` from the [raw data](#raw-data). The training of 5 epochs
takes about two minutes. To only check the installation, the first two commands are
enough, the second with `--epochs 1` (~30 s).

```bash
# 1) Export a small dataset (~3 s, ~11 MB) -> data/deforming_plate_s10_small/
python -m dataset_preprocessor.export

# 2) Train with the example config (MGN-T; MeshGraphNets: add --architecture meshgraphnets)
python train/train.py --config examples/quick_start/train.toml

# 3) Roll out the trained model on the test split -> rollouts/<time>_<run>_<checkpoint>_test/
python predict/rollout.py --checkpoint checkpoints/example_run/best_model.pt \
    --data-dir data/deforming_plate_s10_small --split test

# 4) Loss curves -> checkpoints/example_run/loss_curves.png
python utils/plot_losses.py --run-dir checkpoints/example_run

# 5) Evaluate the run on its own -> evaluations/<time>_example_run_evaluation/
python utils/evaluate_run.py checkpoints/example_run

# 6) Compare all finished rollouts with the papers -> evaluations/<time>_paper_comparison/
python utils/compare_paper.py
```

### What each step does

| Step | Stage | Details |
|---|---|---|
| 1 | The exporter reads 12 of the 100 trajectories of `test.tfrecord`, every 10th frame, writes `dataset.db`, `normalizer.json` and `metadata.json`, and prints the sample counts. | [export a dataset](../guides/export.md) |
| 2 | The training follows `examples/quick_start/train.toml`: MGN-T with ~50 k parameters, position noise, a validation rollout after every epoch, model selection on four criteria. It ends with the test metrics of the selected model. | [train a model](../guides/training.md), [training](../concepts/training.md) |
| 3 | The rollout predicts the 2 test plates from the first frame on, every step from its own previous prediction, and writes the errors and one `.vtu` file per step for ParaView. | [roll out a model](../guides/rollout.md) |
| 4–5 | Loss curves and the evaluation of the run: training, error over the rollout, spread over the test plates. | [evaluate and compare](../guides/evaluation.md#evaluate-a-run) |
| 6 | The comparison of every finished rollout with the values of the papers. With this minimal setup the errors are far above them. | [compare with the papers](../guides/evaluation.md#compare-with-the-papers) |

## Next steps

- [The CrashDefNet app](../guides/app.md): `python utils/app.py` opens a page in the browser
  that leads through the whole workflow.
- [Tutorial: first training](first-training.md): a training whose result can be analysed.
- [Tutorial: bumper beam](bumper-beam.md): a real crash case.
- [How CrashDefNet works](../concepts/overview.md): the stages and the files between them.
