# Installation

## Requirements

- Linux or Windows with WSL2 (tested: Ubuntu under WSL2, Python 3.12)
- Python 3.10–3.12
- Optional: an NVIDIA GPU with a current driver (CUDA 12.x). Everything also runs on
  the CPU; `config.example.toml` is made for it.

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
| `pyvista` (optional) | `.vtu` export of the rollout for ParaView |
| `matplotlib` (optional) | loss curves, comparison with the paper |

## Raw data

CrashDefNet uses DeepMind's `deforming_plate` dataset from the
[MeshGraphNets release](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets).
It is not part of the repository. The example config and the results in this release
need only [`meta.json`](https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/meta.json) and [`test.tfrecord`](https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/test.tfrecord)
(~0.8 GB, 100 trajectories). The test file is used for reasons of size; the exporter
splits its trajectories into separate train, validation and test sets:

```bash
wget -P datasets/deforming_plate/raw_dataset \
    https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/meta.json \
    https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/test.tfrecord
```

The full training split ([`train.tfrecord`](https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/train.tfrecord), 1,000 trajectories,
~9.9 GB) can be exported with `--source-tfrecord train.tfrecord`. The folder address
itself is not browsable; only the file links work.

## Check the installation

```bash
python -m dataset_preprocessor.export          # small dataset, ~3 s
python train/train.py --config config.example.toml --epochs 1
```

The first command prints the exported sample counts. The second one trains one epoch
on the CPU (~4–5 min) and ends with test metrics.

## Troubleshooting

**`ImportError: ... libstdc++.so.6: version 'CXXABI_1.3.15' not found`** when matplotlib
is imported after torch. The environment mixes a conda build of matplotlib with pip
builds of torch. Fix: `conda remove matplotlib matplotlib-base` and
`pip install matplotlib`, or recreate the environment from the current
`environment.yml`.

**`dataset.db exists`** from the exporter with `--no-overwrite`. Remove the flag or
choose another `--out-dir`.

**CUDA not used.** Check `python -c "import torch; print(torch.cuda.is_available())"`.
If it prints `False`, the CPU wheel of PyTorch is installed; reinstall it from
`requirements-gpu.txt`. `--device auto` picks the GPU when available.
