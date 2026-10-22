"""
export.py — Export simulation runs into a compact dataset.

Reads the raw data once and writes into ``--out-dir``. Raw formats
(``--raw-format``): DeepMind's TFRecord (default), LS-DYNA runs (``d3plot``,
read by ``d3plot_reader.py``; nodal stress smoothed by the element
representation) or VTK time series (``vtk``, read by ``vtk_reader.py``).

``dataset.db``
    SQLite (schema: ``dataset.DB_SCHEMA``): every distinct mesh once (cells and
    mesh edges), the static data of every trajectory, world_pos / stress of
    every selected frame as its own row, and the running statistics of the
    train split.
``normalizer.json``
    mean/std per feature group, fitted on the train split.
``metadata.json``
    feature layout, dimensions, splits and contact radius.

Train/val/test are disjoint, contiguous trajectory ranges of the raw data
(e.g. ``test.tfrecord`` with 100 trajectories). Frames are subsampled with
``time_stride``: frames 0, stride, 2·stride, ... (paper: stride 1).

Large data
----------
* A run is read, written and folded into the statistics in blocks of frames
  (``--block-mb``), so memory stays bounded however large a run and the
  dataset are. Small runs are one block.
* Every run is committed together with the statistics so far. ``--resume``
  continues an interrupted export after the last complete run, with the same
  result as an uninterrupted one. ``dataset.db`` counts as finished only once
  ``info.format`` is written.
* Runs on the same mesh (same cells, e.g. the runs of a DoE) store it and its
  mesh edges once.
* ``--edge-stats-stride k`` fits the mesh and world edge statistics, which
  need a contact search per frame, on every k-th sample only.
* ``info.fingerprint`` identifies the content (SHA-256 over the export
  settings and per-trajectory hashes of all blobs, computed while writing), so
  that a rollout need not hash the whole file.

Usage (without arguments the small test set of ``EXPORT_DEFAULTS`` is written)::

    python -m dataset_preprocessor.export [--out-dir DIR] [--n-train N] ...
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):   # executed as a file, e.g. "Run Python File"
    sys.path.insert(0, str(_REPO_ROOT))

from dataset_preprocessor.dataset import DB_FORMAT, DB_SCHEMA  # noqa: E402
from dataset_preprocessor.graph_features import (  # noqa: E402
    MESH_EDGE_FEATURE_DIM,
    MESH_EDGE_FEATURE_NAMES,
    NODE_TYPE_ACTUATOR,
    NUM_NODE_TYPES,
    TARGET_DIM,
    TARGET_FEATURE_NAMES,
    WALL_DISTANCE_CUTOFF,
    WALL_FEATURE_DIM,
    WORLD_EDGE_FEATURE_DIM,
    WORLD_EDGE_FEATURE_NAMES,
    edge_features,
    mesh_edges_from_cells,
    node_feature_names,
    relative_features,
    trajectory_features,
    wall_features,
)
from utils.normalizer import Normalizer  # noqa: E402

# Settings used when the module is run without arguments. Every value can be
# overridden on the command line (see --help). Relative paths are resolved
# against the repository root, so the working directory does not matter.
#
# Small test set: 8/2/2 trajectories, stride 10 -> 312/78/78 samples, ~11 MB.
EXPORT_DEFAULTS = dict(
    raw_format="tfrecord",                         # or "d3plot", "vtk"
    raw_dir="datasets/deforming_plate/raw_dataset",
    raw_dir_d3plot="datasets/deforming_plate/raw_d3plot",   # tfrecord_to_d3plot output
    raw_dir_vtk="datasets/deforming_plate/raw_vtk",         # tfrecord_to_vtk output
    source_tfrecord="test.tfrecord",               # d3plot: runs converted from it
    out_dir="data/deforming_plate_s10_small",   # == [data].data_dir in the config
    n_train=8, n_val=2, n_test=2,                  # whole trajectories (= geometries)
    num_steps=40, time_stride=10,                  # frames 0, 10, ..., 390
    history=0, prev_stress=True,                   # 0 = paper; 1 adds *_prev inputs
    overwrite=True,                                # replace an existing dataset.db
    resume=False,                                  # continue an interrupted export
    edge_stats_stride=1,                           # edge statistics of every k-th sample
    block_mb=1024,                                 # frames of a run read per block
)
# Full set at stride 10 (~95 MB):
#   python -m dataset_preprocessor.export \
#       --out-dir data/deforming_plate_s10 --n-train 70 --n-val 15 --n-test 15
# All 100 trajectories at stride 4 (~225 MB, ~1 min):
#   python -m dataset_preprocessor.export --out-dir data/deforming_plate_s4 \
#       --n-train 80 --n-val 10 --n-test 10 --num-steps 100 --time-stride 4
# Every frame, as in the paper: --num-steps 400 --time-stride 1 (~0.85 GB, ~3.5 min)

RAW_FORMATS = ("tfrecord", "d3plot", "vtk")
# Guards against zero std (constant columns) when fitting the statistics.
_MIN_FEATURE_STD = 1e-6
_MIN_EDGE_STD = 1e-9
# SQLite page size of new DBs: 64 KiB once a frame takes 1 MiB (~65,000 nodes) —
# its blob spans fewer pages, writing is ~3x faster than with the default 4 KiB
# and reading as fast; smaller frames pack better into the default pages.
PAGE_SIZE_LARGE = 65536
LARGE_FRAME_BYTES = 1 << 20


@dataclass
class RawTrajectory:
    """One trajectory of the raw data, restricted to the selected frames.

    The frames are either in ``world_pos`` / ``stress`` or, for large runs, come
    in blocks from ``frame_blocks`` (an iterator, consumed once); ``blocks``
    yields them either way.
    """

    node_type: np.ndarray   # (N,)      raw node type {0, 1, 3}
    mesh_pos: np.ndarray    # (N, 3)    float32, rest configuration
    cells: np.ndarray       # (C, 4)    int32 tetrahedra, or (C, W) padded with -1
    world_pos: Optional[np.ndarray]   # (T, N, 3) float32; None: see frame_blocks
    stress: Optional[np.ndarray]      # (T, N)    float32 von-Mises stress
    cell_types: Optional[np.ndarray] = None       # (C,) VTK cell types; None: tetrahedra
    globals: Optional[Dict[str, float]] = None    # run parameters, e.g. impact velocity
    node_static: Optional[Dict[str, np.ndarray]] = None   # name -> (N,), e.g. thickness
    walls: Optional[np.ndarray] = None            # (W, 6) planar rigid walls: point, normal
    frame_blocks: Optional[Iterator[Tuple[np.ndarray, np.ndarray]]] = None  # (t,N,3), (t,N)

    def blocks(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """The frames in order, as (world_pos (t, N, 3), stress (t, N)) blocks."""
        if self.frame_blocks is None:
            yield self.world_pos, self.stress
        else:
            yield from self.frame_blocks


# ─────────────────────────────────────────────────────────────────────────────
# Reading the raw data
# ─────────────────────────────────────────────────────────────────────────────

def read_raw_meta(raw_dir) -> dict:
    """DeepMind ``meta.json`` (field shapes, dtypes, trajectory length)."""
    with open(os.path.join(raw_dir, "meta.json")) as f:
        return json.load(f)


def read_raw_trajectories(raw_dir, tfrecord_name: str, frames: Sequence[int],
                          count: Optional[int]) -> Iterator[RawTrajectory]:
    """Yield the first ``count`` trajectories of ``tfrecord_name`` (None: all).

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
        if count is not None and index >= count:
            break
        yield RawTrajectory(
            node_type=field(record, "node_type")[:, 0],
            mesh_pos=field(record, "mesh_pos"),
            cells=field(record, "cells"),
            world_pos=field(record, "world_pos"),
            stress=field(record, "stress")[..., 0],
        )


def raw_runs(raw_format: str, raw_dir, source_tfrecord: str, frames: Sequence[int],
             count: Optional[int], vtk_options: Optional[Dict] = None,
             block_bytes: Optional[int] = None
             ) -> Iterator[Tuple[str, Callable[[], RawTrajectory]]]:
    """``(source, read)`` of the first ``count`` runs (None: all), in order.

    ``read()`` returns the run as a ``RawTrajectory`` (LS-DYNA and VTK runs with
    their frames in blocks of about ``block_bytes``); a run that is skipped is
    not read. ``source`` names the run: the TFRecord record, or the run folder
    relative to ``raw_dir``.
    """
    if raw_format == "tfrecord":
        for index, traj in enumerate(read_raw_trajectories(raw_dir, source_tfrecord, frames,
                                                           count)):
            yield f"{source_tfrecord}[{index}]", (lambda traj=traj: traj)
        return
    if raw_format == "d3plot":
        from dataset_preprocessor.d3plot_reader import read_raw_run, run_dirs
        options: Dict = {}
    elif raw_format == "vtk":
        from dataset_preprocessor.vtk_reader import read_raw_run, run_dirs
        options = dict(vtk_options or {})
    else:
        raise ValueError(f"raw_format must be one of {RAW_FORMATS}, got {raw_format!r}")
    root = Path(raw_dir).resolve()
    for run_dir in run_dirs(raw_dir, source_tfrecord)[:count]:
        try:
            source = str(Path(run_dir).resolve().relative_to(root))
        except ValueError:
            source = str(run_dir)
        yield source, partial(read_raw_run, run_dir, frames, max_block_bytes=block_bytes,
                              **options)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

class _RunningMoments:
    """Column mean and std of many (M, D) tensors without keeping them.

    Each batch is folded in with the pairwise update of Chan et al. (float64),
    which stays accurate where the textbook E[x²] - E[x]² would cancel. The
    state (count, mean, m2) is exact in float64 and can be stored and restored.
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

    def mean_std(self, name: str, unbiased: bool, min_std: float, min_count: int = 2):
        """(mean, std) as float32 arrays; std below ``min_std`` becomes 1."""
        if self.count < min_count:
            raise ValueError(f"not enough {name} in the train split to fit statistics")
        std = torch.sqrt(self.m2 / (self.count - (1 if unbiased else 0)))
        std = torch.where(std < min_std, torch.ones_like(std), std)
        return self.mean.float().numpy(), std.float().numpy()

    def state(self) -> Tuple[int, bytes, bytes]:
        return self.count, self.mean.numpy().tobytes(), self.m2.numpy().tobytes()

    @classmethod
    def from_state(cls, count: int, mean: bytes, m2: bytes) -> "_RunningMoments":
        moments = cls(0)
        moments.count = count
        moments.mean = torch.from_numpy(np.frombuffer(mean, np.float64).copy())
        moments.m2 = torch.from_numpy(np.frombuffer(m2, np.float64).copy())
        return moments


class NormalizerFitter:
    """Fit all normalizer groups on the train trajectories, block by block.

    * ``node_features``: the one-hot columns get mean 0 / std 1 (pass through);
      the dynamic columns are standardized over all nodes and samples, static
      node features over all train nodes, run parameters over the train
      trajectories (one value each, population std).
    * ``target`` and ``target_delta`` (history only; change w.r.t. the
      ``*_prev`` value, for ``residual_target``): only nodes in the loss —
      the actuator's prescribed velocity would otherwise dominate.
    * ``mesh_edge_features`` / ``world_edge_features``: every edge of every
      sample (with ``edge_stats_stride`` k: of every k-th sample of a
      trajectory), since the world-space parts change with each time step.

    Per trajectory: ``start``, then ``add_frames`` for consecutive blocks of
    frames; a sample is counted with the block that holds its last frame. The
    running moments go into the DB with every train trajectory (``save``) and
    come back with ``restore``, so a resumed export continues them exactly.
    """

    # running moments -> attribute
    _GROUPS = {"dynamic": "_dynamic", "target": "_target", "delta": "_delta",
               "mesh": "_mesh", "world": "_world", "walls": "_walls",
               "static": "_static", "globals": "_globals"}

    def __init__(self, contact_radius: float, history: int = 0, prev_stress: bool = True,
                 edge_stats_stride: int = 1):
        if edge_stats_stride < 1:
            raise ValueError(f"edge_stats_stride must be >= 1, got {edge_stats_stride}")
        self.contact_radius = contact_radius
        self.history = history
        self.prev_stress = prev_stress
        self.edge_stats_stride = edge_stats_stride
        num_dynamic = len(node_feature_names(history, prev_stress)) - NUM_NODE_TYPES
        self._dynamic = _RunningMoments(num_dynamic)
        self._target = _RunningMoments(TARGET_DIM)
        self._delta = _RunningMoments(TARGET_DIM)
        self._mesh = _RunningMoments(MESH_EDGE_FEATURE_DIM)
        self._world = _RunningMoments(WORLD_EDGE_FEATURE_DIM)
        self._walls = _RunningMoments(WALL_FEATURE_DIM)
        self.global_names: List[str] = []   # set by the exporter from the first trajectory
        self._globals: Optional[_RunningMoments] = None
        self.static_names: List[str] = []   # likewise
        self._static: Optional[_RunningMoments] = None
        self._current: Optional[Dict] = None   # the trajectory being added

    def add(self, traj: RawTrajectory, mesh_edge_index: Optional[torch.Tensor] = None) -> None:
        """Fold all samples of one train trajectory into the statistics."""
        self.start(traj, mesh_edge_index)
        for world_pos, stress in traj.blocks():
            self.add_frames(world_pos, stress)

    def start(self, traj: RawTrajectory, mesh_edge_index: Optional[torch.Tensor] = None) -> None:
        """Begin a train trajectory: its run parameters, static node features and mesh."""
        if self.global_names:
            if self._globals is None:
                self._globals = _RunningMoments(len(self.global_names))
            self._globals.add(torch.tensor([[traj.globals[n] for n in self.global_names]],
                                           dtype=torch.float64))
        if self.static_names:
            if self._static is None:
                self._static = _RunningMoments(len(self.static_names))
            self._static.add(torch.tensor(
                np.stack([traj.node_static[n] for n in self.static_names], axis=1)))
        if mesh_edge_index is None:
            mesh_edge_index = mesh_edges_from_cells(traj.cells, traj.cell_types)
        self._current = {
            "actuator": torch.from_numpy(traj.node_type == NODE_TYPE_ACTUATOR),
            # kept only if the dataset has walls (normalizer())
            "walls": torch.zeros(1, 6) if traj.walls is None else torch.tensor(traj.walls),
            "mesh_edge_index": mesh_edge_index,
            "mesh_space": relative_features(torch.tensor(traj.mesh_pos, dtype=torch.float32),
                                            mesh_edge_index),
            "carry": None,      # last frames of the previous block
            "next_sample": 0,   # index of the next sample in the trajectory
        }

    def add_frames(self, world_pos: np.ndarray, stress: np.ndarray) -> None:
        """Fold in the samples whose last frame is in this block (t, N, 3) / (t, N)."""
        current = self._current
        world_pos = torch.tensor(world_pos, dtype=torch.float32)       # (T, N, 3)
        stress = torch.tensor(stress, dtype=torch.float32)[..., None]  # (T, N, 1)
        if current["carry"] is not None:   # samples reaching back into the last block
            world_pos = torch.cat((current["carry"][0], world_pos))
            stress = torch.cat((current["carry"][1], stress))
        keep = 1 + self.history            # frames of a sample before its last one
        current["carry"] = (world_pos[-keep:].clone(), stress[-keep:].clone())
        if world_pos.shape[0] <= keep:
            return
        actuator = current["actuator"]
        dynamic_inputs, targets, current_pos = trajectory_features(
            world_pos, stress, actuator, self.history, self.prev_stress)

        self._dynamic.add(dynamic_inputs)
        self._target.add(targets[:, ~actuator])
        if self.history:
            # velocity_t - velocity_{t-1} and stress_{t+1} - stress_t
            previous = torch.cat((world_pos[1:-1] - world_pos[:-2], stress[1:-1]), dim=-1)
            self._delta.add((targets - previous)[:, ~actuator])

        for pos in current_pos:
            self._walls.add(wall_features(pos, current["walls"],
                                          WALL_DISTANCE_CUTOFF * self.contact_radius))
        for k, pos in enumerate(current_pos):
            if (current["next_sample"] + k) % self.edge_stats_stride:
                continue
            mesh_edge_attr, _, world_edge_attr = edge_features(
                current["mesh_edge_index"], current["mesh_space"], pos, self.contact_radius)
            self._mesh.add(mesh_edge_attr)
            self._world.add(world_edge_attr)
        current["next_sample"] += current_pos.shape[0]

    def save(self, conn: sqlite3.Connection) -> None:
        """Store the running moments in the ``moments`` table."""
        rows = [(name, *getattr(self, attribute).state())
                for name, attribute in self._GROUPS.items()
                if getattr(self, attribute) is not None]
        conn.executemany("INSERT OR REPLACE INTO moments VALUES (?, ?, ?, ?)", rows)

    def restore(self, conn: sqlite3.Connection) -> None:
        """Continue from the moments stored by ``save``."""
        for name, count, mean, m2 in conn.execute("SELECT name, count, mean, m2 FROM moments"):
            setattr(self, self._GROUPS[name], _RunningMoments.from_state(count, mean, m2))

    def normalizer(self, with_walls: bool = False) -> Normalizer:
        """The fitted statistics; ``with_walls`` appends those of the wall columns.

        Node and target statistics use the unbiased std, edge statistics the
        population std (as the first exporter did; the difference is negligible).
        """
        normalizer = Normalizer()
        dynamic_mean, dynamic_std = self._dynamic.mean_std("samples", True, _MIN_FEATURE_STD)
        # Run parameters: one value per train trajectory (population std).
        global_mean, global_std = np.zeros(0), np.ones(0)
        if self.global_names:
            global_mean, global_std = self._globals.mean_std(
                "trajectories", False, _MIN_FEATURE_STD, min_count=1)
        means = [np.zeros(NUM_NODE_TYPES, np.float32), dynamic_mean]
        stds = [np.ones(NUM_NODE_TYPES, np.float32), dynamic_std]
        if self.static_names:
            static_mean, static_std = self._static.mean_std("nodes", True, _MIN_FEATURE_STD)
            means.append(static_mean)
            stds.append(static_std)
        means.append(global_mean.astype(np.float32))
        stds.append(global_std.astype(np.float32))
        if with_walls:
            wall_mean, wall_std = self._walls.mean_std("samples", True, _MIN_FEATURE_STD)
            means.append(wall_mean)
            stds.append(wall_std)
        normalizer.set_stats("node_features", np.concatenate(means), np.concatenate(stds))
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

def _hash(digest, blob: Optional[bytes]) -> None:
    """Fold ``blob`` into ``digest``, length-prefixed so that blobs cannot run together."""
    digest.update((-1 if blob is None else len(blob)).to_bytes(8, "little", signed=True))
    if blob is not None:
        digest.update(blob)


def _insert_mesh(conn: sqlite3.Connection, traj: RawTrajectory,
                 meshes: Dict[str, int]) -> Tuple[int, torch.Tensor]:
    """Id and mesh edges of the trajectory's mesh; a new mesh is stored with its edges.

    ``meshes`` maps the SHA-256 of every stored mesh to its id.
    """
    num_nodes = traj.mesh_pos.shape[0]
    cells = traj.cells.astype(np.int32)
    cell_types = None if traj.cell_types is None else traj.cell_types.astype(np.uint8).tobytes()
    digest = hashlib.sha256(np.array([num_nodes, *cells.shape], np.int64).tobytes())
    _hash(digest, cells.tobytes())
    _hash(digest, cell_types)
    key = digest.hexdigest()
    if key in meshes:
        (edges,) = conn.execute("SELECT mesh_edges FROM meshes WHERE id = ?",
                                (meshes[key],)).fetchone()
        return meshes[key], torch.from_numpy(
            np.frombuffer(edges, np.int32).reshape(2, -1).astype(np.int64))
    edge_index = mesh_edges_from_cells(cells, traj.cell_types)
    mesh_id = len(meshes)
    conn.execute("INSERT INTO meshes VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 (mesh_id, key, num_nodes, cells.shape[0], edge_index.shape[1], cells.tobytes(),
                  cell_types, edge_index.to(torch.int32).numpy().tobytes()))
    meshes[key] = mesh_id
    return mesh_id, edge_index


def _insert_trajectory(conn: sqlite3.Connection, index: int, split: str, source: str,
                       traj: RawTrajectory, num_frames: int, meshes: Dict[str, int],
                       global_names: Sequence[str] = (), static_names: Sequence[str] = (),
                       fitter: Optional[NormalizerFitter] = None) -> None:
    """One ``trajectories`` row plus one ``frames`` row per frame, block by block;
    with ``fitter`` (train split) the frames go into the statistics as well."""
    mesh_id, mesh_edge_index = _insert_mesh(conn, traj, meshes)
    static = (
        traj.node_type.astype(np.uint8).tobytes(),
        traj.mesh_pos.astype(np.float32).tobytes(),
        (np.array([traj.globals[n] for n in global_names], np.float32).tobytes()
         if global_names else None),
        (np.stack([traj.node_static[n] for n in static_names], axis=1)
         .astype(np.float32).tobytes() if static_names else None),
        None if traj.walls is None else traj.walls.astype(np.float32).tobytes(),
    )
    digest = hashlib.sha256()
    for blob in static:
        _hash(digest, blob)
    conn.execute(
        "INSERT INTO trajectories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (index, split, f"{split}_traj{index:04d}", source, mesh_id, traj.mesh_pos.shape[0],
         num_frames, *static, ""))
    if fitter is not None:
        fitter.start(traj, mesh_edge_index)
    frame = 0
    for world_pos, stress in traj.blocks():
        rows = []
        for k in range(world_pos.shape[0]):
            blobs = (world_pos[k].astype(np.float32).tobytes(),
                     stress[k].astype(np.float32).tobytes())
            for blob in blobs:
                _hash(digest, blob)
            rows.append((index, frame + k, *blobs))
        conn.executemany("INSERT INTO frames VALUES (?, ?, ?, ?)", rows)
        if fitter is not None:
            fitter.add_frames(world_pos, stress)
        frame += world_pos.shape[0]
    if frame != num_frames:
        raise ValueError(f"{source}: {frame} frames read, {num_frames} expected")
    conn.execute("UPDATE trajectories SET sha256 = ? WHERE id = ?", (digest.hexdigest(), index))


def _fingerprint(conn: sqlite3.Connection) -> str:
    """SHA-256 over the ``info`` settings and the hashes of every trajectory and mesh."""
    digest = hashlib.sha256()
    for row in conn.execute("SELECT key, value FROM info WHERE key != 'fingerprint' "
                            "ORDER BY key"):
        digest.update(json.dumps(row).encode())
    for row in conn.execute("SELECT t.id, t.split, t.sequence_id, m.sha256, t.sha256 "
                            "FROM trajectories t JOIN meshes m ON m.id = t.mesh_id "
                            "ORDER BY t.id"):
        digest.update(json.dumps(row).encode())
    return digest.hexdigest()


def _create_db(db_path: Path, settings: Dict[str, str], num_nodes: int) -> sqlite3.Connection:
    """New, empty ``dataset.db`` with the export settings in ``info``."""
    conn = sqlite3.connect(str(db_path))
    frame_bytes = 16 * num_nodes   # world_pos + stress, float32
    if frame_bytes >= LARGE_FRAME_BYTES:
        conn.execute(f"PRAGMA page_size = {PAGE_SIZE_LARGE}")   # before the first table
    conn.executescript(DB_SCHEMA)
    conn.executemany("INSERT INTO info (key, value) VALUES (?, ?)", settings.items())
    conn.commit()
    return conn


def _resume(db_path: Path, settings: Dict[str, str]) -> sqlite3.Connection:
    """Connection to an export that may continue: same format and settings."""
    conn = sqlite3.connect(str(db_path))
    resumable = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='moments'").fetchone()
    if not resumable:
        conn.close()
        raise ValueError(f"{db_path} was written by an older exporter and cannot be "
                         "continued — export it anew (--no-resume)")
    info = dict(conn.execute("SELECT key, value FROM info"))
    changed = [f"{key}: {info.get(key)} -> {value}" for key, value in settings.items()
               if info.get(key) != value]
    if changed:
        conn.close()
        raise ValueError(f"{db_path} was exported with other settings ({'; '.join(changed)}) "
                         "— --resume needs the same options; --no-resume starts anew")
    return conn


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
    raw_format: str = "tfrecord",
    contact_radius: Optional[float] = None,
    vtk_options: Optional[Dict] = None,
    resume: bool = False,
    edge_stats_stride: int = 1,
    block_mb: float = EXPORT_DEFAULTS["block_mb"],
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    """Write ``dataset.db``, ``normalizer.json`` and ``metadata.json`` to ``out_dir``.

    ``history`` / ``prev_stress`` fix the node feature layout; they are stored
    in the DB's ``info`` table and in ``metadata.json``. ``raw_format``
    ``"d3plot"`` / ``"vtk"`` read LS-DYNA runs / VTK time series (below
    ``raw_dir/<source_tfrecord name>`` or ``raw_dir``) instead of the TFRecord;
    ``vtk_options`` are passed to ``vtk_reader.read_raw_run`` (field names, parts).
    ``contact_radius`` overrides the ``collision_radius`` of ``meta.json``,
    which such raw data may lack. An existing ``dataset.db`` is replaced, or with
    ``resume`` continued (see the module docstring); ``progress`` receives one
    line per run. Returns the paths of the three files.
    """
    if raw_format not in RAW_FORMATS:
        raise ValueError(f"raw_format must be one of {RAW_FORMATS}, got {raw_format!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = read_raw_meta(raw_dir) if (Path(raw_dir) / "meta.json").exists() else {}
    if contact_radius is None:
        default = 0.03 if raw_format == "tfrecord" else None
        contact_radius = meta.get("collision_radius", default)
        if contact_radius is None:
            raise ValueError(f"{raw_dir}: no collision_radius in meta.json — pass the "
                             "contact radius (--contact-radius)")
    contact_radius = float(contact_radius)
    frames = [k * time_stride for k in range(num_steps)]
    samples_per_trajectory = num_steps - 1 - history
    splits = {"train": {"start": 0, "count": n_train},
              "val": {"start": n_train, "count": n_val},
              "test": {"start": n_train + n_val, "count": n_test}}
    num_trajectories = n_train + n_val + n_test
    # Everything that changes the content; --resume requires the same values.
    settings = {
        "contact_radius": repr(contact_radius),
        "time_stride": str(time_stride),
        "history": str(history),
        "prev_stress": "1" if prev_stress else "0",
        "num_steps": str(num_steps),
        "frames": json.dumps(frames),
        "source_tfrecord": source_tfrecord,
        "raw_format": raw_format,
        "splits": json.dumps(splits),
        "edge_stats_stride": str(edge_stats_stride),
    }
    if raw_format == "vtk":   # the options in effect, so that defaults compare equal
        from dataset_preprocessor.vtk_reader import read_raw_run
        vtk_options = {**{name: p.default for name, p in
                          inspect.signature(read_raw_run).parameters.items()
                          if p.default is not p.empty and name not in ("frames",
                                                                       "max_block_bytes")},
                       **(vtk_options or {})}
        settings["vtk_options"] = json.dumps(vtk_options, sort_keys=True, default=list)

    fitter = NormalizerFitter(contact_radius, history, prev_stress, edge_stats_stride)
    db_path = out / "dataset.db"
    done: Dict[int, str] = {}       # trajectories already in the DB -> source
    meshes: Dict[str, int] = {}     # SHA-256 of a stored mesh -> id
    global_names: List[str] = []
    static_names: List[str] = []
    if resume and db_path.exists():
        conn = _resume(db_path, settings)
        done = dict(conn.execute("SELECT id, source FROM trajectories"))
        meshes = dict(conn.execute("SELECT sha256, id FROM meshes"))
        info = dict(conn.execute("SELECT key, value FROM info"))
        global_names = fitter.global_names = json.loads(info.get("global_names", "[]"))
        static_names = fitter.static_names = json.loads(info.get("static_names", "[]"))
        fitter.restore(conn)
        if progress and done:
            progress(f"  continuing: {len(done)} of {num_trajectories} trajectories are "
                     "already in dataset.db")
    else:
        # Delete instead of reusing: SQLite keeps the old file size after DELETE.
        for path in (db_path, Path(f"{db_path}-journal")):
            path.unlink(missing_ok=True)
        conn = None   # created with the first run, whose size sets the page size

    block_bytes = int(block_mb * 2 ** 20) if block_mb else None
    num_read = 0
    try:
        for index, (source, read) in enumerate(raw_runs(
                raw_format, raw_dir, source_tfrecord, frames, num_trajectories,
                vtk_options, block_bytes)):
            num_read += 1
            if index in done:
                if done[index] != source:
                    raise ValueError(f"trajectory {index} in {db_path} comes from {done[index]}, "
                                     f"the raw data now hold {source} there — the raw data "
                                     "changed; export anew (--no-resume)")
                continue
            start = time.time()
            traj = read()
            if conn is None:
                conn = _create_db(db_path, settings, traj.mesh_pos.shape[0])
            names = (sorted(traj.globals or {}), sorted(traj.node_static or {}))
            if index == 0:
                global_names = fitter.global_names = names[0]
                static_names = fitter.static_names = names[1]
            elif names != (global_names, static_names):
                raise ValueError(f"trajectory {index}: run parameters / static node features "
                                 f"{names} differ from those of the first trajectory "
                                 f"{(global_names, static_names)}")
            split = next(name for name, r in splits.items()
                         if r["start"] <= index < r["start"] + r["count"])
            with conn:   # one transaction per run: the run and the statistics, or nothing
                _insert_trajectory(conn, index, split, source, traj, len(frames), meshes,
                                   global_names, static_names,
                                   fitter if split == "train" else None)
                if index == 0:
                    conn.executemany("INSERT OR REPLACE INTO info (key, value) VALUES (?, ?)",
                                     [("global_names", json.dumps(global_names)),
                                      ("static_names", json.dumps(static_names))])
                if split == "train":
                    fitter.save(conn)
            if progress:
                progress(f"  [{index + 1}/{num_trajectories}] {split:5s} {source}: "
                         f"{traj.mesh_pos.shape[0]:,} nodes, {len(frames)} frames, "
                         f"{time.time() - start:.1f} s")
        if num_read < num_trajectories:
            raise ValueError(f"{source_tfrecord} ({raw_format}) holds only {num_read} "
                             f"trajectories, {num_trajectories} requested")

        max_walls = (conn.execute("SELECT max(length(walls)) FROM trajectories").fetchone()[0]
                     or 0) // (6 * 4)
        finished = [("format", DB_FORMAT)] + ([("max_walls", str(max_walls))] if max_walls else [])
        with conn:
            conn.executemany("INSERT OR REPLACE INTO info (key, value) VALUES (?, ?)", finished)
            conn.execute("INSERT OR REPLACE INTO info (key, value) VALUES ('fingerprint', ?)",
                         (_fingerprint(conn),))
    finally:
        if conn is not None:
            conn.close()

    normalizer_path = out / "normalizer.json"
    fitter.normalizer(with_walls=max_walls > 0).save(str(normalizer_path))

    names = node_feature_names(history, prev_stress, global_names, static_names,
                               walls=max_walls > 0)
    metadata = {
        "dataset": "deforming_plate",
        "db_format": DB_FORMAT,
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
    # Only when they apply: the rollout compares metadata.json with the training
    # data, and TFRecord exports stay identical to those of earlier versions.
    if raw_format != "tfrecord":
        metadata["raw_format"] = raw_format
    if global_names:
        metadata["global_feature_names"] = global_names
    if static_names:
        metadata["static_feature_names"] = static_names
    if max_walls:
        metadata["max_walls"] = max_walls
        metadata["wall_distance_cutoff"] = WALL_DISTANCE_CUTOFF * contact_radius
    if edge_stats_stride != 1:
        metadata["edge_stats_stride"] = edge_stats_stride
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
    parser.add_argument("--raw-format", choices=RAW_FORMATS, default=d["raw_format"],
                        help="raw data: DeepMind TFRecord (standard), LS-DYNA runs (smoothed "
                             "stress, see docs/data.md) or VTK time series")
    parser.add_argument("--raw-dir", default=None,
                        help="directory with meta.json and the .tfrecord or the run folders "
                             f"(default: {d['raw_dir']}; d3plot: {d['raw_dir_d3plot']}; "
                             f"vtk: {d['raw_dir_vtk']})")
    parser.add_argument("--source-tfrecord", default=d["source_tfrecord"],
                        help="TFRecord inside --raw-dir (test.tfrecord: 100 trajectories); "
                             "d3plot / vtk: the runs converted from it (folder <name without "
                             ".tfrecord>), else all runs below --raw-dir")
    parser.add_argument("--contact-radius", type=float, default=None,
                        help="world edge radius; default: collision_radius of meta.json "
                             "(required for d3plot / vtk raw data without meta.json)")
    vtk = parser.add_argument_group("VTK raw data (--raw-format vtk): field names and parts")
    vtk.add_argument("--stress-field", default="stress",
                     help="point or cell field: scalar, or stress tensor (6 / 9 components)")
    vtk.add_argument("--node-type-field", default="node_type",
                     help="point field with the node types 0 / 1 / 3; absent: all normal")
    vtk.add_argument("--static-fields", nargs="*", default=[],
                     help="point or cell fields of the first state as static node features "
                          "(vectors: one per component)")
    vtk.add_argument("--displacement-field", default=None,
                     help="positions = points + this field (exports with reference geometry)")
    vtk.add_argument("--node-id-field", default=None,
                     help="point field with node IDs: copies of a node in several parts "
                          "become one node; None: the field marked as global IDs (e.g. "
                          "UserID of ParaView's LS-DYNA reader), '': ignore IDs")
    vtk.add_argument("--part-field", default=None,
                     help="cell field with part IDs for --actuator-parts / --clamped-parts; "
                          "None: parts named by their block or file")
    vtk.add_argument("--actuator-parts", nargs="*", default=[],
                     help="parts whose nodes are actuators (motion given, e.g. rigid impactor)")
    vtk.add_argument("--clamped-parts", nargs="*", default=[],
                     help="parts whose nodes are clamped")
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
    large = parser.add_argument_group("large data")
    large.add_argument("--resume", action=argparse.BooleanOptionalAction, default=d["resume"],
                       help="continue an interrupted export in --out-dir (same options) "
                            "after its last complete trajectory")
    large.add_argument("--edge-stats-stride", type=int, default=d["edge_stats_stride"],
                       help="fit the edge statistics (a contact search per frame) on every "
                            "k-th sample of a trajectory only; 1 = all")
    large.add_argument("--block-mb", type=float, default=d["block_mb"],
                       help="LS-DYNA / VTK: read the frames of a run in blocks of about "
                            "this size (MB), so a run need not fit into memory")
    args = parser.parse_args(argv)

    if args.raw_dir is None:
        args.raw_dir = d[{"d3plot": "raw_dir_d3plot", "vtk": "raw_dir_vtk"}.get(
            args.raw_format, "raw_dir")]
    raw_dir, out_dir = _repo_path(args.raw_dir), _repo_path(args.out_dir)
    num_trajectories = args.n_train + args.n_val + args.n_test
    if (raw_dir / "meta.json").exists():
        trajectory_length = read_raw_meta(raw_dir)["trajectory_length"]
    elif args.raw_format != "tfrecord":   # runs without meta.json: shortest run
        if args.raw_format == "d3plot":
            from dataset_preprocessor.d3plot_reader import num_states, run_dirs
        else:
            from dataset_preprocessor.vtk_reader import num_states, run_dirs
        runs = run_dirs(raw_dir, args.source_tfrecord)[:num_trajectories]
        trajectory_length = min(num_states(run) for run in runs)
    else:
        parser.error(f"{raw_dir / 'meta.json'} not found")
    if args.contact_radius is None and args.raw_format != "tfrecord" and \
            "collision_radius" not in (read_raw_meta(raw_dir) if (raw_dir / "meta.json").exists()
                                       else {}):
        parser.error(f"{raw_dir} has no meta.json with collision_radius — pass --contact-radius")
    last_frame = args.time_stride * (args.num_steps - 1)
    if last_frame >= trajectory_length:
        parser.error(f"time_stride * (num_steps - 1) = {last_frame} must be < "
                     f"trajectory_length = {trajectory_length}")
    if args.num_steps < 2 + args.history:
        parser.error(f"num_steps must be >= {2 + args.history} with --history {args.history}")
    if min(args.n_train, args.n_val) <= 0 or args.n_test < 0:
        parser.error("n-train and n-val must be > 0, n-test >= 0")
    if args.edge_stats_stride < 1:
        parser.error("--edge-stats-stride must be >= 1")
    db_path = out_dir / "dataset.db"
    if db_path.exists() and not args.overwrite and not args.resume:
        parser.error(f"{db_path} exists — pass --overwrite to replace it or --resume to "
                     "continue it")

    samples_per_trajectory = args.num_steps - 1 - args.history
    print(f"Exporting {args.n_train}/{args.n_val}/{args.n_test} trajectories x "
          f"{samples_per_trajectory} samples (frames 0..{last_frame}, stride "
          f"{args.time_stride}, history {args.history}, raw {args.raw_format}) -> {out_dir}")
    start = time.time()
    try:
        paths = write_dataset(
            raw_dir=str(raw_dir), out_dir=str(out_dir), source_tfrecord=args.source_tfrecord,
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            num_steps=args.num_steps, time_stride=args.time_stride,
            history=args.history, prev_stress=args.prev_stress, raw_format=args.raw_format,
            contact_radius=args.contact_radius,
            vtk_options=dict(stress_field=args.stress_field, node_type_field=args.node_type_field,
                             static_fields=args.static_fields,
                             displacement_field=args.displacement_field,
                             node_id_field=args.node_id_field, part_field=args.part_field,
                             actuator_parts=args.actuator_parts,
                             clamped_parts=args.clamped_parts),
            resume=args.resume, edge_stats_stride=args.edge_stats_stride,
            block_mb=args.block_mb, progress=print,
        )
    except ValueError as error:
        sys.exit(f"export.py: error: {error}")
    counts = "/".join(str(n * samples_per_trajectory)
                      for n in (args.n_train, args.n_val, args.n_test))
    print(f"Done in {time.time() - start:.0f} s: {counts} samples, "
          f"dataset.db {db_path.stat().st_size / 1e6:.1f} MB")
    with open(paths["metadata"]) as f:   # the layout includes global parameters, if any
        names = json.load(f)["node_feature_names"]
    print(f"  node features ({len(names)}): {', '.join(names)}")
    for name, path in paths.items():
        print(f"  {name:10s} {path}")


if __name__ == "__main__":
    main()
