# ============================================================
# CrashDefNet — GPU training image (deforming_plate)
# ============================================================
# CUDA 12.4 + cuDNN runtime on Ubuntu 22.04 (Python 3.10).
# (CUDA 12.4 has no ubuntu24.04 tag; 24.04 starts with CUDA 12.6, which
#  requires driver >= 560.)
# For newer drivers CUDA can be raised (adjust base image + torch index):
#   base:  nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
#   torch: --index-url https://download.pytorch.org/whl/cu126
#
# Host requirements: NVIDIA driver >= 550 (CUDA 12.4) and Docker with GPU
# support (Docker Desktop with the WSL2 backend, or Docker Engine + NVIDIA
# Container Toolkit on Linux).
# ============================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Europe/Berlin

# Time zone: all timestamps are written in Europe/Berlin (utils/timezone.py);
# Python needs the tz database from the tzdata package for that. TZ puts shell
# commands in the container (date, ls -l) into the same zone.

# System packages: Python and tzdata (no vtk/pyvista in the image — the .vtu
# export of the rollout runs locally, training does not need it)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev tzdata \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv (avoids PEP 668 "externally-managed-environment")
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

WORKDIR /app

# PyTorch (CUDA 12.4). torch ships its own CUDA libraries.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
      "torch>=2.4,<3.0"

# Remaining dependencies (as in requirements.txt, without the optional pyvista)
RUN pip install \
      "torch-geometric>=2.5,<3.0" tfrecord \
      "numpy>=1.24,<3.0" "toml>=0.10" "tqdm>=4.60"

# Project code only — datasets are mounted as volumes at run time (see .dockerignore)
COPY . /app

RUN chmod +x /app/docker/entrypoint.sh

# Checks the mounted dataset (/data), then starts the training.
ENTRYPOINT ["/app/docker/entrypoint.sh"]
