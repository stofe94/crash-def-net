"""
export.py — Export DeepMind deforming_plate trajectories into a compact dataset.

Reads the raw TFRecord once and writes into ``--out-dir``:

``dataset.db``
    SQLite: the static mesh of every trajectory (node type, mesh positions,
    tetrahedra) and world_pos / stress of every selected frame as its own row
    (schema: ``dataset.TRAJECTORY_DB_SCHEMA``).
``normalizer.json``
    mean/std per feature group, fitted on the train split.
``metadata.json``
    feature layout, dimensions, splits and contact radius.

Train/val/test are disjoint, contiguous trajectory ranges of one TFRecord
(``test.tfrecord`` holds 100 trajectories). Frames are subsampled with
``time_stride``: frames 0, stride, 2·stride, ... (paper: stride 1). Every
trajectory is written and folded into the statistics as soon as it is read,
so memory stays at about one trajectory regardless of the dataset size.

Usage (without arguments the small test set of ``EXPORT_DEFAULTS`` is written)::

    python -m dataset_preprocessor.export [--out-dir DIR] [--n-train N] ...
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):   # executed as a file, e.g. "Run Python File"
    sys.path.insert(0, str(_REPO_ROOT))

from dataset_preprocessor.dataset import TRAJECTORY_DB_FORMAT, TRAJECTORY_DB_SCHEMA  # noqa: E402
from dataset_preprocessor.graph_features import (  # noqa: E402
    MESH_EDGE_FEATURE_DIM,
    MESH_EDGE_FEATURE_NAMES,
    NODE_TYPE_ACTUATOR,
    NUM_NODE_TYPES,
    TARGET_DIM,
    TARGET_FEATURE_NAMES,
    WORLD_EDGE_FEATURE_DIM,
    WORLD_EDGE_FEATURE_NAMES,
    edge_features,
    mesh_edges_from_cells,
    node_feature_names,
    relative_features,
    trajectory_features,
)
from utils.normalizer import Normalizer  # noqa: E402

# Settings used when the module is run without arguments. Every value can be
# overridden on the command line (see --help). Relative paths are resolved
# against the repository root, so the working directory does not matter.
#
# Small test set: 8/2/2 trajectories, stride 10 -> 312/78/78 samples, ~10 MB.
EXPORT_DEFAULTS = dict(
    raw_dir="datasets/deforming_plate/raw_dataset",
    source_tfrecord="test.tfrecord",
    out_dir="data/deforming_plate_s10_small",   # == [data].data_dir in the config
    n_train=8, n_val=2, n_test=2,                  # whole trajectories (= geometries)
    num_steps=40, time_stride=10,                  # frames 0, 10, ..., 390
    history=0, prev_stress=True,                   # 0 = paper; 1 adds *_prev inputs
    overwrite=True,                                # replace an existing dataset.db
)
# Full set at stride 10 (~90 MB):
#   python -m dataset_preprocessor.export \
#       --out-dir data/deforming_plate_s10 --n-train 70 --n-val 15 --n-test 15
# All 100 trajectories at stride 4 (~215 MB, ~1 min):
#   python -m dataset_preprocessor.export --out-dir data/deforming_plate_s4 \
#       --n-train 80 --n-val 10 --n-test 10 --num-steps 100 --time-stride 4
# Every frame, as in the paper: --num-steps 400 --time-stride 1 (~0.8 GB)

# Guards against zero std (constant columns) when fitting the statistics.
_MIN_FEATURE_STD = 1e-6
_MIN_EDGE_STD = 1e-9


@dataclass
class RawTrajectory:
    """One trajectory as stored in the TFRecord, restricted to the selected frames."""

    node_type: np.ndarray   # (N,)      raw node type {0, 1, 3}
    mesh_pos: np.ndarray    # (N, 3)    float32, rest configuration
    cells: np.ndarray       # (C, 4)    int32 tetrahedra
    world_pos: np.ndarray   # (T, N, 3) float32
    stress: np.ndarray      # (T, N)    float32 von-Mises stress


# ─────────────────────────────────────────────────────────────────────────────
# Reading the raw data
# ─────────────────────────────────────────────────────────────────────────────

def read_raw_meta(raw_dir) -> dict:
    """DeepMind ``meta.json`` (field shapes, dtypes, trajectory length)."""
    with open(os.path.join(raw_dir, "meta.json")) as f:
        return json.load(f)


def read_raw_trajectories(raw_dir, tfrecord_name: str, frames: Sequence[int],
                          count: int) -> Iterator[RawTrajectory]:
    """Yield the first ``count`` trajectories of ``tfrecord_name``.

    Dynamic fields are restricted to ``frames``; static fields (mesh, node
    type, cells) are stored once per trajectory in the file.
    """
    from tfrecord.torch.dataset import TFRecordDataset

    meta = read_raw_meta(raw_dir)
    records = TFRecordDataset(os.path.join(raw_dir, tfrecord_name), None,
                              {name: "byte" for name in meta["field_names"]})
    frames = np.asarray(frames)

    def field(record, name):
        spec = meta["features"][name]
        values = np.frombuffer(record[name], dtype=getattr(np, spec["dtype"]))
        values = values.reshape(spec["shape"])
        return values[0] if spec["type"] == "static" else values[frames]

    for index, record in enumerate(records):
        if index >= count:
            break
        yield RawTrajectory(
            node_type=field(record, "node_type")[:, 0],
            mesh_pos=field(record, "mesh_pos"),
            cells=field(record, "cells"),
            world_pos=field(record, "world_pos"),
            stress=field(record, "stress")[..., 0],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

class _RunningMoments:
    """Column mean and std of many (M, D) tensors without keeping them.

    Each batch is folded in with the pairwise update of Chan et al. (float64),
    which stays accurate where the textbook E[x²] - E[x]² would cancel.
    """

    def __init__(self, dim: int):
        self.count = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.m2 = torch.zeros(dim, dtype=torch.float64)   # sum of squared deviations

    def add(self, rows: torch.Tensor) -> None:
        rows = rows.reshape(-1, self.mean.shape[0]).double()
        n = rows.shape[0]
        if n == 0:
            return
        batch_mean = rows.mean(0)
        batch_m2 = ((rows - batch_mean) ** 2).sum(0)
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * (n / total)
        self.m2 += batch_m2 + delta ** 2 * (self.count * n / total)
        self.count = total

    def mean_std(self, name: str, unbiased: bool, min_std: float):
        """(mean, std) as float32 arrays; std below ``min_std`` becomes 1."""
        if self.count < 2:
            raise ValueError(f"not enough {name} in the train split to fit statistics")
        std = torch.sqrt(self.m2 / (self.count - (1 if unbiased else 0)))
        std = torch.where(std < min_std, torch.ones_like(std), std)
        return self.mean.float().numpy(), std.float().numpy()


class NormalizerFitter:
    """Fit all normalizer groups on the train trajectories, one at a time.

    * ``node_features``: the one-hot columns get mean 0 / std 1 (pass through);
      the dynamic columns are standardized over all nodes and samples.
    * ``target`` and ``target_delta`` (history only; change w.r.t. the
      ``*_prev`` value, for ``residual_target``): only nodes in the loss —
      the actuator's prescribed velocity would otherwise dominate.
    * ``mesh_edge_features`` / ``world_edge_features``: every edge of every
      sample, since the world-space parts change with each time step.
    """

    def __init__(self, contact_radius: float, history: int = 0, prev_stress: bool = True):
        self.contact_radius = contact_radius
        self.history = history
        self.prev_stress = prev_stress
        num_dynamic = len(node_feature_names(history, prev_stress)) - NUM_NODE_TYPES
        self._dynamic = _RunningMoments(num_dynamic)
        self._target = _RunningMoments(TARGET_DIM)
        self._delta = _RunningMoments(TARGET_DIM)
        self._mesh = _RunningMoments(MESH_EDGE_FEATURE_DIM)
        self._world = _RunningMoments(WORLD_EDGE_FEATURE_DIM)

    def add(self, traj: RawTrajectory) -> None:
        """Fold the samples of one train trajectory into the statistics."""
        world_pos = torch.tensor(traj.world_pos, dtype=torch.float32)       # (T, N, 3)
        stress = torch.tensor(traj.stress, dtype=torch.float32)[..., None]  # (T, N, 1)
        actuator = torch.from_numpy(traj.node_type == NODE_TYPE_ACTUATOR)
        dynamic_inputs, targets, current_pos = trajectory_features(
            world_pos, stress, actuator, self.history, self.prev_stress)

        self._dynamic.add(dynamic_inputs)
        self._target.add(targets[:, ~actuator])
        if self.history:
            # velocity_t - velocity_{t-1} and stress_{t+1} - stress_t
            previous = torch.cat((world_pos[1:-1] - world_pos[:-2], stress[1:-1]), dim=-1)
            self._delta.add((targets - previous)[:, ~actuator])

        mesh_edge_index = mesh_edges_from_cells(traj.cells)
        mesh_space_features = relative_features(
            torch.tensor(traj.mesh_pos, dtype=torch.float32), mesh_edge_index)
        for pos in current_pos:
            mesh_edge_attr, _, world_edge_attr = edge_features(
                mesh_edge_index, mesh_space_features, pos, self.contact_radius)
            self._mesh.add(mesh_edge_attr)
            self._world.add(world_edge_attr)

    def normalizer(self) -> Normalizer:
        """The fitted statistics.

        Node and target statistics use the unbiased std, edge statistics the
        population std (as the first exporter did; the difference is negligible).
        """
        normalizer = Normalizer()
        dynamic_mean, dynamic_std = self._dynamic.mean_std("samples", True, _MIN_FEATURE_STD)
        normalizer.set_stats(
            "node_features",
            np.concatenate([np.zeros(NUM_NODE_TYPES, np.float32), dynamic_mean]),
            np.concatenate([np.ones(NUM_NODE_TYPES, np.float32), dynamic_std]))
        normalizer.set_stats("target", *self._target.mean_std("samples", True, _MIN_FEATURE_STD))
        if self.history:
            normalizer.set_stats("target_delta",
                                 *self._delta.mean_std("samples", True, _MIN_FEATURE_STD))
        normalizer.set_stats("mesh_edge_features",
                             *self._mesh.mean_std("mesh edges", False, _MIN_EDGE_STD))
        normalizer.set_stats("world_edge_features",
                             *self._world.mean_std("world edges", False, _MIN_EDGE_STD))
        return normalizer


# ─────────────────────────────────────────────────────────────────────────────
# Writing the dataset
# ─────────────────────────────────────────────────────────────────────────────

def _insert_trajectory(conn: sqlite3.Connection, index: int, split: str,
                       traj: RawTrajectory) -> None:
    """One ``trajectories`` row (static mesh) plus one ``frames`` row per frame."""
    num_frames, num_nodes = traj.world_pos.shape[:2]
    conn.execute(
        "INSERT INTO trajectories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (index, split, f"{split}_traj{index:04d}", num_nodes, num_frames, traj.cells.shape[0],
         traj.node_type.astype(np.uint8).tobytes(),
         traj.mesh_pos.astype(np.float32).tobytes(),
         traj.cells.astype(np.int32).tobytes()))
    conn.executemany(
        "INSERT INTO frames VALUES (?, ?, ?, ?)",
        ((index, frame, traj.world_pos[frame].astype(np.float32).tobytes(),
          traj.stress[frame].astype(np.float32).tobytes())
         for frame in range(num_frames)))


def write_dataset(
    raw_dir: str,
    out_dir: str,
    source_tfrecord: str = "test.tfrecord",
    n_train: int = 70,
    n_val: int = 15,
    n_test: int = 15,
    num_steps: int = 400,
    time_stride: int = 1,
    history: int = 0,
    prev_stress: bool = True,
) -> Dict[str, str]:
    """Write ``dataset.db``, ``normalizer.json`` and ``metadata.json`` to ``out_dir``.

    ``history`` / ``prev_stress`` fix the node feature layout; they are stored
    in the DB's ``info`` table and in ``metadata.json``. An existing
    ``dataset.db`` is replaced. Returns the paths of the three files.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = read_raw_meta(raw_dir)
    contact_radius = float(meta.get("collision_radius", 0.03))
    frames = [k * time_stride for k in range(num_steps)]
    samples_per_trajectory = num_steps - 1 - history
    splits = {"train": {"start": 0, "count": n_train},
              "val": {"start": n_train, "count": n_val},
              "test": {"start": n_train + n_val, "count": n_test}}
    num_trajectories = n_train + n_val + n_test

    db_path = out / "dataset.db"
    # Delete instead of reusing: SQLite keeps the old file size after DELETE.
    for path in (db_path, Path(f"{db_path}-journal")):
        path.unlink(missing_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(TRAJECTORY_DB_SCHEMA)
    conn.executemany("INSERT INTO info (key, value) VALUES (?, ?)", [
        ("format", TRAJECTORY_DB_FORMAT),
        ("contact_radius", repr(contact_radius)),
        ("time_stride", str(time_stride)),
        ("history", str(history)),
        ("prev_stress", "1" if prev_stress else "0"),
        ("num_steps", str(num_steps)),
        ("frames", json.dumps(frames)),
        ("source_tfrecord", source_tfrecord),
    ])

    fitter = NormalizerFitter(contact_radius, history, prev_stress)
    num_read = 0
    for index, traj in enumerate(read_raw_trajectories(
            raw_dir, source_tfrecord, frames, num_trajectories)):
        split = next(name for name, r in splits.items()
                     if r["start"] <= index < r["start"] + r["count"])
        _insert_trajectory(conn, index, split, traj)
        if split == "train":
            fitter.add(traj)
        num_read += 1
    conn.commit()
    conn.close()
    if num_read < num_trajectories:
        raise ValueError(f"{source_tfrecord} holds only {num_read} trajectories, "
                         f"{num_trajectories} requested")

    normalizer_path = out / "normalizer.json"
    fitter.normalizer().save(str(normalizer_path))

    names = node_feature_names(history, prev_stress)
    metadata = {
        "dataset": "deforming_plate",
        "db_format": TRAJECTORY_DB_FORMAT,
        "history": history,
        "prev_stress": bool(prev_stress and history),
        "node_feature_dims": len(names),
        "edge_feature_dims": MESH_EDGE_FEATURE_DIM,
        "world_edge_feature_dims": WORLD_EDGE_FEATURE_DIM,
        "target_dim": TARGET_DIM,
        "node_feature_names": names,
        "edge_feature_names": list(MESH_EDGE_FEATURE_NAMES),
        "world_edge_feature_names": list(WORLD_EDGE_FEATURE_NAMES),
        "target_feature_names": list(TARGET_FEATURE_NAMES),
        "loss_mask": "all nodes except actuator (node_type 1)",
        "contact_radius": contact_radius,
        "num_steps": num_steps,
        "time_stride": time_stride,
        "raw_dir": str(Path(raw_dir).resolve()),
        "source_tfrecord": source_tfrecord,
        "sample_counts": {name: r["count"] * samples_per_trajectory
                          for name, r in splits.items()},
        "splits": splits,
    }
    metadata_path = out / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {"db": str(db_path), "metadata": str(metadata_path),
            "normalizer": str(normalizer_path)}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _repo_path(path: str) -> Path:
    """Resolve a relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def main(argv=None) -> None:
    """Export a dataset; defaults come from ``EXPORT_DEFAULTS``."""
    d = EXPORT_DEFAULTS
    parser = argparse.ArgumentParser(
        description="Export DeepMind deforming_plate trajectories into a CrashDefNet "
                    "dataset.db (+ metadata.json, normalizer.json). Without arguments "
                    "the small test set is written.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", default=d["raw_dir"],
                        help="directory with the .tfrecord and meta.json")
    parser.add_argument("--source-tfrecord", default=d["source_tfrecord"],
                        help="TFRecord inside --raw-dir (test.tfrecord: 100 trajectories)")
    parser.add_argument("--out-dir", default=d["out_dir"],
                        help="output directory (== [data].data_dir in the config)")
    parser.add_argument("--n-train", type=int, default=d["n_train"], help="train trajectories")
    parser.add_argument("--n-val", type=int, default=d["n_val"], help="validation trajectories")
    parser.add_argument("--n-test", type=int, default=d["n_test"], help="test trajectories")
    parser.add_argument("--num-steps", type=int, default=d["num_steps"],
                        help="frames per trajectory (samples per trajectory = "
                             "num_steps - 1 - history)")
    parser.add_argument("--time-stride", type=int, default=d["time_stride"],
                        help="raw frames between two selected frames")
    parser.add_argument("--history", type=int, choices=(0, 1), default=d["history"],
                        help="1: append velocity_prev (+ stress_prev) to the node features "
                             "(enables residual_target / scheduled sampling); 0 = paper")
    parser.add_argument("--prev-stress", action=argparse.BooleanOptionalAction,
                        default=d["prev_stress"], help="with --history 1: include stress_prev")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction,
                        default=d["overwrite"], help="replace an existing dataset.db")
    args = parser.parse_args(argv)

    raw_dir, out_dir = _repo_path(args.raw_dir), _repo_path(args.out_dir)
    trajectory_length = read_raw_meta(raw_dir)["trajectory_length"]
    last_frame = args.time_stride * (args.num_steps - 1)
    if last_frame >= trajectory_length:
        parser.error(f"time_stride * (num_steps - 1) = {last_frame} must be < "
                     f"trajectory_length = {trajectory_length}")
    if args.num_steps < 2 + args.history:
        parser.error(f"num_steps must be >= {2 + args.history} with --history {args.history}")
    if min(args.n_train, args.n_val) <= 0 or args.n_test < 0:
        parser.error("n-train and n-val must be > 0, n-test >= 0")
    db_path = out_dir / "dataset.db"
    if db_path.exists() and not args.overwrite:
        parser.error(f"{db_path} exists — pass --overwrite to replace it")

    samples_per_trajectory = args.num_steps - 1 - args.history
    names = node_feature_names(args.history, args.prev_stress)
    print(f"Exporting {args.n_train}/{args.n_val}/{args.n_test} trajectories x "
          f"{samples_per_trajectory} samples (frames 0..{last_frame}, stride "
          f"{args.time_stride}, history {args.history}) -> {out_dir}")
    print(f"  node features ({len(names)}): {', '.join(names)}")
    start = time.time()
    paths = write_dataset(
        raw_dir=str(raw_dir), out_dir=str(out_dir), source_tfrecord=args.source_tfrecord,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        num_steps=args.num_steps, time_stride=args.time_stride,
        history=args.history, prev_stress=args.prev_stress,
    )
    counts = "/".join(str(n * samples_per_trajectory)
                      for n in (args.n_train, args.n_val, args.n_test))
    print(f"Done in {time.time() - start:.0f} s: {counts} samples, "
          f"dataset.db {db_path.stat().st_size / 1e6:.1f} MB")
    for name, path in paths.items():
        print(f"  {name:10s} {path}")


if __name__ == "__main__":
    main()
