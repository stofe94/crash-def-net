# Installation

## Requirements

- Linux or Windows with WSL2 (tested: Ubuntu under WSL2, Python 3.12)
- Python 3.10–3.12
- Optional: an NVIDIA GPU with a current driver (CUDA 12.x). Everything also runs on
  the CPU, so CrashDefNet can be tried on simple hardware: with the small dataset of the
  quick start and a reduced model, a training run takes minutes to a few hours. Measured
  on a 6-core desktop CPU (AMD Ryzen 5 5600GE, 5 GB RAM under WSL2):

  | Config (small dataset, 8/2/2 trajectories) | Parameters | per epoch | Memory |
  |---|---|---|---|
  | `config.example.toml` (MGN-T, minimal size) | 50 k | ~30 s, 5 epochs in ~2.5 min | 0.6 GB |
  | same with `architecture = "meshgraphnets"` (latent 64, 8 steps) | 0.53 M | ~2 min 30 s, 30 epochs in ~1 h 15 min | 1.6 GB |
- Large datasets need no machine with much RAM: the export and lazy loading keep only
  one block of a run or one sample in memory ([data.md](data.md#large-datasets)).
  LS-DYNA runs need `lasso-python`, VTK files `pyvista` ([Dependencies](#dependencies)).
- Everything runs locally: once installed, export, training, rollout and evaluation
  need no network connection, cloud service or account, so simulation data never
  leaves your machines.

## Python environment

### conda (recommended)

```bash
conda env create -f environment.yml
conda activate crashdefnet
```

`environment.yml` installs Python and numpy from conda-forge and everything with C++
extensions (PyTorch, PyG, pyvista, matplotlib) from pip. Mixing conda and pip builds of
such packages leads to libstdc++ conflicts on Linux/WSL (see
[Troubleshooting](#troubleshooting)).

For a GPU, change the index URL in `environment.yml` from `.../whl/cpu` to
`.../whl/cu126` (or the CUDA version of your driver) before creating the environment.

### pip / venv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU
pip install -r requirements-gpu.txt      # GPU, CUDA 12.6 (other versions: change the index URL)
pip install -e ".[dev]"                  # optional: package install + pytest, ruff
```

### Dependencies

| Package | Purpose |
|---|---|
| `torch`, `torch-geometric` | models, training, graph batching |
| `numpy`, `tfrecord` | reading DeepMind's raw data during export |
| `toml`, `tqdm` | configs, progress bars |
| `pyvista` (optional) | `.vtu` export of the rollout for ParaView, rollout animation; third raw format: VTK time series ([data.md](data.md#third-raw-format-vtk-time-series-vtu--vtp--vtk)) |
| `matplotlib` (optional) | loss curves, comparison with the paper, rollout animation |
| `lasso-python` (optional) | second raw format: LS-DYNA runs ([data.md](data.md#second-raw-format-ls-dyna-runs-d3plot)) |

## Raw data

CrashDefNet uses DeepMind's `deforming_plate` dataset from the
[MeshGraphNets release](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets).
It is not part of the repository and comes under DeepMind's terms, not CrashDefNet's
license ([details](data.md#the-benchmark)). The example config and the quick start
need only [`meta.json`](https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/meta.json) and [`test.tfrecord`](https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/test.tfrecord)
(~0.8 GB, 100 trajectories); the exporter splits its trajectories into separate train,
validation and test sets:

```bash
wget -P datasets/deforming_plate/raw_dataset \
    https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/meta.json \
    https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/test.tfrecord
```

The [comparison protocol](comparison-protocol.md) and the [results](results.md) take each
split from DeepMind's original one and also need the training split
([`train.tfrecord`](https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/train.tfrecord), 1,200 trajectories, ~9.9 GB) and
[`valid.tfrecord`](https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/valid.tfrecord) (100 trajectories, ~0.8 GB):
`--source-tfrecord train.tfrecord valid.tfrecord test.tfrecord`, with `--select-seed`
drawing the trajectories at random ([options](data.md#export)). The folder address itself is not browsable; only the file
links work.

## Check the installation

```bash
python -m dataset_preprocessor.export          # small dataset, ~3 s
python train/train.py --config config.example.toml --epochs 1
```

The first command prints the exported sample counts. The second one trains one epoch
on the CPU (~30 s) and ends with test metrics.

## Troubleshooting

**`ImportError: ... libstdc++.so.6: version 'CXXABI_1.3.15' not found`** when matplotlib
is imported after torch. The environment mixes a conda build of matplotlib with pip
builds of torch. Fix: `conda remove matplotlib matplotlib-base` and
`pip install matplotlib`, or recreate the environment from the current
`environment.yml`.

**`ERROR: pip's dependency resolver ... lasso-python ... requires ...`** after a
`pip install`: lasso-python pins versions of packages that only its modules
`lasso.dimred` and `lasso.diffcrash` use; CrashDefNet uses only `lasso.dyna`, so the
message can be ignored. Check: `python -m dataset_preprocessor.tfrecord_to_d3plot
--count 2` writes two runs and reads them back ("read-back OK").

**`dataset.db exists`** from the exporter with `--no-overwrite`. Remove the flag or
choose another `--out-dir`.

**CUDA not used.** Check `python -c "import torch; print(torch.cuda.is_available())"`.
If it prints `False`, the CPU wheel of PyTorch is installed; reinstall it from
`requirements-gpu.txt`. `--device auto` picks the GPU when available.
