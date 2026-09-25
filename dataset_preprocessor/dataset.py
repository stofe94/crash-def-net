"""
dataset.py — PyTorch datasets over a compact deforming_plate ``dataset.db``.

``dataset.db`` (written by ``dataset_preprocessor.export``) stores the static
mesh of every trajectory once (table ``trajectories``) and world_pos / stress
of every selected frame as one row each (table ``frames``).
``DeformingPlateDataset`` builds the graph of a sample only when it is
accessed, with the functions of ``graph_features``. The frames are either

* loaded into memory once (default; ~8 MB per trajectory at 400 frames). With
  ``spawn`` DataLoader workers PyTorch moves these tensors into shared memory,
  so every worker reads the same copy; or
* read from the DB on access (``lazy=True``): each process opens its own
  read-only connection and reads only the 2-3 frames of a sample, optionally
  keeping the last ``cache_frames`` frames. Memory then no longer grows with
  the dataset — for splits that do not fit into RAM / ``/dev/shm``.

Older DBs (format v1: all frames of a trajectory in its ``trajectories`` row)
can still be loaded into memory, but not lazily.
"""

from __future__ import annotations

import sqlite3
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from utils.normalizer import Normalizer

from .graph_features import (
    NODE_TYPE_ACTUATOR,
    NUM_NODE_TYPES,
    edge_features,
    mesh_edges_from_cells,
    node_feature_names,
    one_hot_node_type,
    relative_features,
    trajectory_features,
)

# Format tag in the ``info`` table; the DB holds raw frames only, so the tag
# does not depend on the feature layout (history is an ``info`` entry).
TRAJECTORY_DB_FORMAT = "deforming_plate_trajectories_v2"   # one row per frame
LEGACY_DB_FORMAT = "deforming_plate_trajectories_v1"       # frames inside the trajectory row

TRAJECTORY_DB_SCHEMA = """
CREATE TABLE info (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE trajectories (
    id          INTEGER PRIMARY KEY,
    split       TEXT NOT NULL,
    sequence_id TEXT NOT NULL,
    num_nodes   INTEGER NOT NULL,
    num_frames  INTEGER NOT NULL,
    num_cells   INTEGER NOT NULL,
    node_type   BLOB NOT NULL,   -- uint8   (N)    raw node_type {0, 1, 3}
    mesh_pos    BLOB NOT NULL,   -- float32 (N, 3)
    cells       BLOB NOT NULL    -- int32   (C, 4) tetrahedra
);
CREATE TABLE frames (
    trajectory_id INTEGER NOT NULL REFERENCES trajectories(id),
    frame         INTEGER NOT NULL,   -- 0 .. num_frames-1 (selected frames only)
    world_pos     BLOB NOT NULL,      -- float32 (N, 3)
    stress        BLOB NOT NULL,      -- float32 (N)
    PRIMARY KEY (trajectory_id, frame)
);
CREATE INDEX idx_trajectories_split ON trajectories(split);
"""

# Lazy mode: map up to this much of the DB file into memory. The pages live in
# the OS page cache, which all worker processes share and which can be evicted.
_MMAP_BYTES = 2 << 30


def required_stat_groups(history: int) -> Tuple[str, ...]:
    """Normalizer groups a dataset with this history needs."""
    groups = ("node_features", "target", "mesh_edge_features", "world_edge_features")
    return groups + (("target_delta",) if history else ())


def _tensor(blob: bytes, dtype, shape) -> torch.Tensor:
    # frombuffer is read-only; copy so torch gets a writable array.
    return torch.from_numpy(np.frombuffer(blob, dtype).reshape(shape).copy())


@dataclass
class _Trajectory:
    """Static data of one trajectory, ready for graph building."""

    db_id: int
    sequence_id: str
    num_frames: int
    mesh_pos: torch.Tensor               # (N, 3)    undeformed positions
    node_one_hot: torch.Tensor           # (N, 3)    first columns of x
    actuator_mask: torch.Tensor          # (N,) bool
    mesh_edge_index: torch.Tensor        # (2, E_M)
    mesh_space_features: torch.Tensor    # (E_M, 4)  u_ij, |u_ij|
    cells: torch.Tensor                  # (C, 4)    tetrahedra, for .vtu export
    world_pos: Optional[torch.Tensor] = None   # (T, N, 3); None when lazy
    stress: Optional[torch.Tensor] = None      # (T, N, 1); None when lazy

    @property
    def num_nodes(self) -> int:
        return self.node_one_hot.shape[0]


class DeformingPlateDataset(Dataset):
    """One-step samples of one split of a trajectory ``dataset.db``.

    Sample (trajectory k, step s) uses frames s .. s + 1 + history: an
    optional predecessor, the current and the next frame. Its ``Data`` holds
    ``x``, ``y``, ``edge_index``/``edge_attr``, ``world_edge_index``/
    ``world_edge_attr`` (all normalized), ``loss_mask``, the raw current
    positions ``world_pos`` and the undeformed ``mesh_pos``.

    Parameters
    ----------
    db_path : str
        Path to ``dataset.db``.
    split : str
        ``"train"``, ``"val"`` or ``"test"``.
    normalizer : Normalizer, optional
        Defaults to ``normalizer.json`` next to the DB.
    with_cells : bool
        Also attach the tetrahedra as ``data.cells`` (only the .vtu export needs them).
    lazy : bool
        Read the frames of a sample from the DB on access instead of loading
        the split into memory (needs a format-v2 DB).
    cache_frames : int
        Lazy mode: keep the last N frames per process (LRU); 0 = no cache.
    """

    def __init__(self, db_path, split: str, normalizer: Optional[Normalizer] = None,
                 with_cells: bool = False, lazy: bool = False, cache_frames: int = 0):
        super().__init__()
        self.db_path = str(db_path)
        self.split = split
        self.with_cells = with_cells
        self.lazy = lazy
        self.cache_frames = max(0, int(cache_frames))
        self.normalizer = normalizer or Normalizer.load(
            str(Path(self.db_path).parent / "normalizer.json"))

        info, self.trajectories = self._load(split)
        self.contact_radius = float(info["contact_radius"])
        # Layout fixed at export time (DBs from before --history: paper layout).
        self.history = int(info.get("history", 0))
        self.prev_stress = info.get("prev_stress", "1") == "1"
        self.node_feature_names = node_feature_names(self.history, self.prev_stress)
        # Sample index -> (trajectory index, first frame of the sample).
        self.samples: List[Tuple[int, int]] = [
            (k, step)
            for k, traj in enumerate(self.trajectories)
            for step in range(traj.num_frames - 1 - self.history)
        ]

        missing = [g for g in required_stat_groups(self.history) if g not in self.normalizer]
        if missing:
            raise ValueError(f"normalizer has no {missing} statistics — re-export the "
                             "dataset: python -m dataset_preprocessor.export")
        node_mean, node_std = self.normalizer.mean_std("node_features")
        # Only the dynamic columns are normalized here; the one-hot has mean 0 / std 1.
        self._dynamic_mean = node_mean[NUM_NODE_TYPES:]
        self._dynamic_std = node_std[NUM_NODE_TYPES:]
        self._target_mean, self._target_std = self.normalizer.mean_std("target")
        self._mesh_mean, self._mesh_std = self.normalizer.mean_std("mesh_edge_features")
        self._world_mean, self._world_std = self.normalizer.mean_std("world_edge_features")

        # Lazy mode: opened on first access in each process (see __getstate__).
        self._connection: Optional[sqlite3.Connection] = None
        self._frame_cache: OrderedDict = OrderedDict()

    # ── Dataset protocol ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        k, step = self.samples[idx]
        traj = self.trajectories[k]
        world_pos, stress = self._frames(traj, step, 2 + self.history)
        dynamic_inputs, targets, current_pos = trajectory_features(
            world_pos, stress, traj.actuator_mask, self.history, self.prev_stress)
        dynamic_inputs, targets, world_pos = dynamic_inputs[0], targets[0], current_pos[0]

        mesh_edge_attr, world_edge_index, world_edge_attr = edge_features(
            traj.mesh_edge_index, traj.mesh_space_features, world_pos, self.contact_radius)
        data = Data(
            x=torch.cat((traj.node_one_hot,
                         (dynamic_inputs - self._dynamic_mean) / self._dynamic_std), dim=-1),
            y=(targets - self._target_mean) / self._target_std,
            edge_index=traj.mesh_edge_index,
            edge_attr=(mesh_edge_attr - self._mesh_mean) / self._mesh_std,
            world_edge_index=world_edge_index,
            world_edge_attr=(world_edge_attr - self._world_mean) / self._world_std,
            loss_mask=~traj.actuator_mask,
            world_pos=world_pos,
            mesh_pos=traj.mesh_pos,       # undeformed; mgn_t's positional encoding
        )
        if self.with_cells:
            data.cells = traj.cells
        return data

    def __getstate__(self):
        """Pickling for spawned DataLoader workers: every worker opens its own
        connection and starts with an empty cache."""
        state = self.__dict__.copy()
        state["_connection"] = None
        state["_frame_cache"] = OrderedDict()
        return state

    # ── sequence structure ──────────────────────────────────────────────────

    def sequences(self) -> Dict[str, List[int]]:
        """Sample indices of every trajectory in time order, without building graphs."""
        result: Dict[str, List[int]] = {}
        for idx, (k, _) in enumerate(self.samples):
            result.setdefault(self.trajectories[k].sequence_id, []).append(idx)
        return result

    def predecessor(self, idx: int) -> Optional[int]:
        """Index of the previous time step of the same trajectory, or None."""
        if idx > 0 and self.samples[idx - 1][0] == self.samples[idx][0]:
            return idx - 1
        return None

    # ── frames ──────────────────────────────────────────────────────────────

    def _frames(self, traj: _Trajectory, first: int, count: int
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """world_pos (count, N, 3) and stress (count, N, 1) of frames first .. first+count-1."""
        if traj.world_pos is not None:
            return traj.world_pos[first:first + count], traj.stress[first:first + count]
        frames = [self._read_frame(traj, frame) for frame in range(first, first + count)]
        return (torch.stack([world_pos for world_pos, _ in frames]),
                torch.stack([stress for _, stress in frames]))

    def _read_frame(self, traj: _Trajectory, frame: int):
        """Lazy mode: one frame from the LRU cache or the DB."""
        key = (traj.db_id, frame)
        cached = self._frame_cache.get(key)
        if cached is not None:
            self._frame_cache.move_to_end(key)
            return cached
        world_pos, stress = self._db().execute(
            "SELECT world_pos, stress FROM frames WHERE trajectory_id = ? AND frame = ?",
            key).fetchone()
        value = (_tensor(world_pos, np.float32, (traj.num_nodes, 3)),
                 _tensor(stress, np.float32, (traj.num_nodes, 1)))
        if self.cache_frames:
            self._frame_cache[key] = value
            if len(self._frame_cache) > self.cache_frames:
                self._frame_cache.popitem(last=False)
        return value

    def _db(self) -> sqlite3.Connection:
        """Read-only connection of the current process, opened on first use."""
        if self._connection is None:
            self._connection = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True,
                                               check_same_thread=False)
            self._connection.execute(f"PRAGMA mmap_size = {_MMAP_BYTES}")
        return self._connection

    # ── loading ─────────────────────────────────────────────────────────────

    def _load(self, split: str) -> Tuple[Dict[str, str], List[_Trajectory]]:
        """``info`` table and the trajectories of ``split`` (with frames unless lazy)."""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trajectories'"
            ).fetchone()
            if not has_table:
                raise ValueError(
                    f"{self.db_path} is not a trajectory DB (no table 'trajectories') — "
                    "re-export it: python -m dataset_preprocessor.export --out-dir <dir>")
            info = dict(conn.execute("SELECT key, value FROM info").fetchall())
            db_format = info.get("format")
            if db_format not in (TRAJECTORY_DB_FORMAT, LEGACY_DB_FORMAT):
                raise ValueError(f"{self.db_path}: unsupported format {db_format!r}")
            if self.lazy and db_format != TRAJECTORY_DB_FORMAT:
                raise ValueError(f"{self.db_path} has the old format {db_format!r}; lazy "
                                 "loading needs one row per frame — re-export it: "
                                 "python -m dataset_preprocessor.export")

            rows = conn.execute(
                "SELECT id, sequence_id, num_nodes, num_frames, num_cells, node_type, "
                "mesh_pos, cells FROM trajectories WHERE split = ? ORDER BY id",
                (split,)).fetchall()
            trajectories = [self._trajectory_from_row(row) for row in rows]
            if not self.lazy:
                for traj in trajectories:
                    traj.world_pos, traj.stress = self._read_all_frames(conn, db_format, traj)
        finally:
            conn.close()
        return info, trajectories

    @staticmethod
    def _read_all_frames(conn, db_format: str, traj: _Trajectory):
        """All frames of ``traj``: world_pos (T, N, 3), stress (T, N, 1)."""
        if db_format == LEGACY_DB_FORMAT:
            world_pos, stress = conn.execute(
                "SELECT world_pos, stress FROM trajectories WHERE id = ?",
                (traj.db_id,)).fetchone()
        else:
            rows = conn.execute(
                "SELECT world_pos, stress FROM frames WHERE trajectory_id = ? ORDER BY frame",
                (traj.db_id,)).fetchall()
            if len(rows) != traj.num_frames:
                raise ValueError(f"trajectory {traj.sequence_id}: {len(rows)} frames in the "
                                 f"DB, {traj.num_frames} expected")
            # Frame blobs in order = the (T, N, ...) layout of the whole trajectory.
            world_pos = b"".join(row[0] for row in rows)
            stress = b"".join(row[1] for row in rows)
        shape = (traj.num_frames, traj.num_nodes)
        return (_tensor(world_pos, np.float32, shape + (3,)),
                _tensor(stress, np.float32, shape + (1,)))

    @staticmethod
    def _trajectory_from_row(row) -> _Trajectory:
        db_id, sequence_id, num_nodes, num_frames, num_cells, node_type, mesh_pos, cells = row
        raw_node_type = _tensor(node_type, np.uint8, (num_nodes,))
        mesh_pos = _tensor(mesh_pos, np.float32, (num_nodes, 3))
        cells = np.frombuffer(cells, np.int32).reshape(num_cells, 4)
        mesh_edge_index = mesh_edges_from_cells(cells)
        return _Trajectory(
            db_id=db_id,
            sequence_id=sequence_id,
            num_frames=num_frames,
            mesh_pos=mesh_pos,
            node_one_hot=one_hot_node_type(raw_node_type),
            actuator_mask=raw_node_type == NODE_TYPE_ACTUATOR,
            mesh_edge_index=mesh_edge_index,
            mesh_space_features=relative_features(mesh_pos, mesh_edge_index),
            cells=torch.from_numpy(cells.astype(np.int64)),
        )


class PredecessorPairDataset(Dataset):
    """``(sample_t, sample_{t-1}, has_predecessor)`` items for scheduled sampling.

    The trainer predicts step t from the predecessor sample and feeds that
    prediction into the ``*_prev`` inputs of sample t. The first step of a
    trajectory has no predecessor; it is paired with itself and flagged False.
    """

    def __init__(self, base: DeformingPlateDataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        current = self.base[idx]
        prev_idx = self.base.predecessor(idx)
        if prev_idx is None:
            return current, current, False
        return current, self.base[prev_idx], True


def collate_with_predecessor(items) -> Tuple[Batch, Batch]:
    """Collate ``PredecessorPairDataset`` items into ``(batch, prev_batch)``.

    ``batch.has_predecessor`` (N,) marks the nodes whose ``*_prev`` inputs may
    be replaced by a prediction. A module-level function, so it pickles into
    spawned DataLoader workers.
    """
    batch = Batch.from_data_list([current for current, _, _ in items])
    prev_batch = Batch.from_data_list([prev for _, prev, _ in items])
    batch.has_predecessor = torch.cat([
        torch.full((current.num_nodes,), valid, dtype=torch.bool)
        for current, _, valid in items])
    return batch, prev_batch
