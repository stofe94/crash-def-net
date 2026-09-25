#!/usr/bin/env bash
# ============================================================
# Container entrypoint: checks the mounted dataset and starts the training.
# The dataset is NOT built here; it must be mounted at ${DATA_DIR} (see
# docker-compose.yml, DATASET_DIR). Extra arguments are passed on to
# train.py, e.g.:  docker compose run --rm train --epochs 50
# ============================================================
set -euo pipefail
cd /app

DATA_DIR="${DATA_DIR:-/data}"
CONFIG="${CONFIG:-config.example.toml}"
# Overrides [training].device of the TOML: "auto" = cuda if visible, else cpu.
DEVICE="${DEVICE:-auto}"

echo "[entrypoint] torch/GPU check:"
python3 - <<'PY'
import torch
print(f"  torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
else:
    print("  WARNING: no GPU visible — container started without GPU access (--gpus all)?")
PY

missing=0
for f in dataset.db metadata.json normalizer.json; do
  if [ ! -f "${DATA_DIR}/${f}" ]; then
    echo "[entrypoint] ERROR: ${DATA_DIR}/${f} is missing."
    missing=1
  fi
done
if [ "${missing}" -ne 0 ]; then
  echo "[entrypoint] No valid dataset mounted at ${DATA_DIR}."
  echo "             Set the host folder via DATASET_DIR, e.g.:"
  echo "               DATASET_DIR=/path/to/dataset docker compose up"
  exit 1
fi
echo "[entrypoint] Using the dataset at ${DATA_DIR}"

echo "[entrypoint] Starting training: python3 train/train.py --config ${CONFIG} --data-dir ${DATA_DIR} --device ${DEVICE} $*"
exec python3 train/train.py --config "${CONFIG}" --data-dir "${DATA_DIR}" --device "${DEVICE}" "$@"
