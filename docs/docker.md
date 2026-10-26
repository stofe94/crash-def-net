# Docker (GPU)

The Docker image trains CrashDefNet on an NVIDIA GPU without installing CUDA, PyTorch or
PyG on the host. It contains the code only. The dataset is exported beforehand and
mounted as a volume; results are written back to the host.

## Requirements

- **NVIDIA driver** ≥ 550 (the image uses CUDA 12.4)
- **Docker with GPU support:** Docker Desktop with the WSL2 backend on Windows (GPU
  support included), or Docker Engine + NVIDIA Container Toolkit on Linux
- GPU check:
  `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
  must list the GPU

## Build

```bash
docker compose build          # image crashdefnet-deforming-plate:latest (~6 GB: CUDA + torch)
```

The image contains no vtk / pyvista. Training does not need them, and the `.vtu` export
of the rollout is switched off in the container (see below).

## Train

The dataset folder (with `dataset.db`, `metadata.json`, `normalizer.json`) is created
with the exporter (see [data.md](data.md)) and set in `.env` next to
`docker-compose.yml` (not versioned, template `.env.example`):

```bash
cp .env.example .env
# in .env, e.g.:
#   DATASET_DIR=./data/deforming_plate_s4_t150
#   CONFIG=configs/my_run.toml
docker compose up
```

Without `.env`, `./data/deforming_plate_s10_small` and `config.example.toml` are used.
A one-off override also works from the shell:
`DATASET_DIR=/path CONFIG=configs/... docker compose up`.

What the container does:

1. GPU check (name and VRAM)
2. Checks that `/data` contains the three dataset files (aborts with a hint otherwise)
3. `python train/train.py --config $CONFIG --data-dir /data --device auto`

`--device auto` overrides `[training].device` of the TOML: `cuda` in the GPU container,
`cpu` if no GPU is visible. Extra arguments are passed on to `train.py`:

```bash
docker compose run --rm train --epochs 50 --batch-size 4
```

**Results** end up on the host under `./checkpoints/<checkpoint_dir of the config>/`.
Own configs go into `configs/`; the folder is mounted, so they can be edited without a
rebuild.

> Performance on Docker Desktop / WSL2: keep the dataset in the WSL file system
> (`/home/...`) and run `docker compose` from the WSL shell. Windows drives (`/mnt/c`,
> `D:/`) are mounted much more slowly.

## Export inside the container

With the raw data under `datasets/deforming_plate/raw_dataset/` (see
[installation.md](installation.md)):

```bash
docker run --rm -v $PWD/datasets:/app/datasets -v $PWD/data:/app/data \
  --entrypoint python crashdefnet-deforming-plate:latest -m dataset_preprocessor.export \
  --out-dir data/deforming_plate_s4_t150 \
  --source-tfrecord train.tfrecord valid.tfrecord test.tfrecord --select-seed 42 \
  --n-train 100 --n-val 30 --n-test 20 --num-steps 100 --time-stride 4
```

## Rollout inside the container

```bash
docker compose run --rm --entrypoint python train predict/rollout.py \
  --checkpoint checkpoints/<run>/best_model.pt --data-dir /data --split test --no-export-vtu
```

The rollout folder appears under `./rollouts/` on the host. For `.vtu` files for
ParaView, run the rollout locally with pyvista installed.

## Moving the image to another machine

Build here, run there:

```bash
docker save crashdefnet-deforming-plate | gzip > crashdefnet-dp.tar.gz
# copy the file to the GPU machine, then:
docker load < crashdefnet-dp.tar.gz
```

Copy the dataset folder separately. Alternatively copy the repository and run
`docker compose build` on the GPU machine.

## Memory hints

- **VRAM:** the model of `config.example.toml` fits into 4 GB at batch 2. The runs of the
  [results](results.md) needed ~2 GB (MGN-T) and ~8 GB (MeshGraphNets in paper size) at
  batch 8 with `self_contact_aware`; with `world_edges = "all"` MeshGraphNets needs
  micro-batches on a 16 GB GPU ([protocol](comparison-protocol.md#training)). The trainer
  records the peak memory in `history.json`. `PYTORCH_CUDA_ALLOC_CONF` in
  `docker-compose.yml` counters fragmentation caused by the varying number of contact
  edges.
- **RAM:** the dataset is held once in shared memory (`shm_size` 2 GB); each DataLoader
  worker costs ~0.3 GB. On 4 vCPU / 8 GB RAM, 3 workers fit. For splits that do not fit
  into RAM or `/dev/shm`: `[data] lazy = true`.

## Other CUDA versions

For newer drivers, align the base image and the torch index in the `Dockerfile`, e.g.
CUDA 12.6 (driver ≥ 560): `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` and
`--index-url https://download.pytorch.org/whl/cu126`.
