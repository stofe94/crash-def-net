"""
dataset.py — PyTorch datasets over a compact deforming_plate ``dataset.db``.

``dataset.db`` (written by ``dataset_preprocessor.export``) stores every
distinct mesh once (table ``meshes``: cells and mesh edges), the static data of
every trajectory (table ``trajectories``) and world_pos / stress of every
selected frame as one row each (table ``frames``).
``DeformingPlateDataset`` builds the graph of a sample only when it is
accessed, with the functions of ``graph_features``. The data are either

* loaded into memory once (default; ~8 MB per trajectory at 400 frames). With
  ``spawn`` DataLoader workers PyTorch moves these tensors into shared memory,
  so every worker reads the same copy; or
* read from the DB on access (``lazy=True``): each process opens its own
  read-only connection and reads only the 2-3 frames of a sample, optionally
  keeping the last ``cache_frames`` frames. The static graph data of a
  trajectory (mesh edges, world edge selection, rest positions) are built when
  it is first needed and kept up to ``static_cache_mb`` per process. Memory
  then no longer grows with the dataset — for splits that do not fit into RAM
  / ``/dev/shm``.

Trajectories on the same mesh (same cells, e.g. the runs of a DoE) share its
mesh edges and world edge selection. ``build_world_edges=False`` leaves the
contact search out of the samples; ``graph_features.add_world_edges`` then adds
the world edges on the device (the trainer does this when position noise or a
rollout rebuilds them anyway).

Older DBs load as well: format v2/v3 (cells in every trajectory row, mesh edges
computed on loading), and format v1 (all frames of a trajectory in its
``trajectories`` row) only into memory.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Dict, Hashable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from utils.normalizer import Normalizer

from .graph_features import (
    DEFAULT_WORLD_EDGE_EXCLUDE_DISTANCE,
    NODE_TYPE_ACTUATOR,
    NUM_NODE_TYPES,
    WALL_DISTANCE_CUTOFF,
    WALL_FEATURE_DIM,
    WORLD_EDGE_MODES,
    edge_features,
    mesh_edge_features,
    mesh_edges_from_cells,
    node_feature_names,
    one_hot_node_type,
    relative_features,
    trajectory_features,
    wall_features,
    world_edge_selection,
)

# Format tag in the ``info`` table; the DB holds raw frames only, so the tag
# does not depend on the feature layout (history is an ``info`` entry).
DB_FORMAT = "deforming_plate_trajectories_v4"              # shared meshes, resumable export
GENERAL_DB_FORMAT = "deforming_plate_trajectories_v3"      # + cell types, global parameters
TRAJECTORY_DB_FORMAT = "deforming_plate_trajectories_v2"   # one row per frame
LEGACY_DB_FORMAT = "deforming_plate_trajectories_v1"       # frames inside the trajectory row
DB_FORMATS = (DB_FORMAT, GENERAL_DB_FORMAT, TRAJECTORY_DB_FORMAT, LEGACY_DB_FORMAT)

DB_SCHEMA = """
CREATE TABLE info (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE meshes (
    id          INTEGER PRIMARY KEY,
    sha256      TEXT NOT NULL UNIQUE,   -- of num_nodes, cells and cell_types
    num_nodes   INTEGER NOT NULL,
    num_cells   INTEGER NOT NULL,
    num_edges   INTEGER NOT NULL,
    cells       BLOB NOT NULL,   -- int32   (C, 4) tetrahedra, or (C, W) padded with -1
    cell_types  BLOB,            -- uint8   (C)    VTK cell types; NULL: tetrahedra
    mesh_edges  BLOB NOT NULL    -- int32   (2, E) both directions, sorted (mesh_edges_from_cells)
);
CREATE TABLE trajectories (
    id          INTEGER PRIMARY KEY,
    split       TEXT NOT NULL,
    sequence_id TEXT NOT NULL,
    source      TEXT NOT NULL,   -- raw run: TFRecord record or run folder below raw_dir
    mesh_id     INTEGER NOT NULL REFERENCES meshes(id),
    num_nodes   INTEGER NOT NULL,
    num_frames  INTEGER NOT NULL,
    node_type   BLOB NOT NULL,   -- uint8   (N)    raw node_type {0, 1, 3}
    mesh_pos    BLOB NOT NULL,   -- float32 (N, 3)
    globals     BLOB,            -- float32 (G)    run parameters (info: global_names)
    node_static BLOB,            -- float32 (N, S) static node features (info: static_names)
    walls       BLOB,            -- float32 (W, 6) rigid walls: point, unit normal
    sha256      TEXT NOT NULL    -- of the static data and all frames (info: fingerprint)
);
CREATE TABLE frames (
    trajectory_id INTEGER NOT NULL REFERENCES trajectories(id),
    frame         INTEGER NOT NULL,   -- 0 .. num_frames-1 (selected frames only)
    world_pos     BLOB NOT NULL,      -- float32 (N, 3)
    stress        BLOB NOT NULL,      -- float32 (N)
    PRIMARY KEY (trajectory_id, frame)
);
CREATE TABLE moments (                -- running statistics of the train split (export)
    name  TEXT PRIMARY KEY,           -- feature group
    count INTEGER NOT NULL,
    mean  BLOB NOT NULL,              -- float64 (D)
    m2    BLOB NOT NULL               -- float64 (D) sum of squared deviations
);
CREATE INDEX idx_trajectories_split ON trajectories(split);
"""

# Lazy mode: map up to this much of the DB file into memory. The pages live in
# the OS page cache, which all worker processes share and which can be evicted.
# (SQLite builds usually cap the map at 2 GB; beyond, pages are read normally.)
_MMAP_BYTES = 2 << 30
DEFAULT_STATIC_CACHE_MB = 1024


def connect_readonly(db_path) -> sqlite3.Connection:
    """Read-only connection. ``immutable=1``: SQLite neither locks the file nor
    checks it for changes — faster, and safe on network file systems, but the DB
    must not be written while it is open (no export into a DB in use)."""
    return sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True,
                           check_same_thread=False)


def dataset_fingerprint(db_path) -> Optional[str]:
    """Content fingerprint stored by the exporter (format v4), None for older DBs.

    It identifies the dataset without reading the whole file (SHA-256 over the
    export settings and the per-trajectory hashes, see ``export``).
    """
    conn = connect_readonly(db_path)
    try:
        row = conn.execute("SELECT value FROM info WHERE key = 'fingerprint'").fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def required_stat_groups(history: int) -> Tuple[str, ...]:
    """Normalizer groups a dataset with this history needs."""
    groups = ("node_features", "target", "mesh_edge_features", "world_edge_features")
    return groups + (("target_delta",) if history else ())


def _tensor(blob: bytes, dtype, shape) -> torch.Tensor:
    # frombuffer is read-only; copy so torch gets a writable array.
    return torch.from_numpy(np.frombuffer(blob, dtype).reshape(shape).copy())


@dataclass
class _TrajectoryInfo:
    """What the dataset keeps of every trajectory of its split at all times."""

    db_id: int
    sequence_id: str
    num_nodes: int
    num_frames: int
    mesh_key: Hashable   # ("mesh", id) in format v4; ("cells", trajectory id) before


@dataclass
class _Mesh:
    """Static graph data of one mesh, shared by the trajectories on it."""

    edge_index: torch.Tensor                   # (2, E_M) mesh edges
    world_selection: Dict[str, torch.Tensor]   # world edge variant (world_edge_selection)
    cells: Optional[torch.Tensor] = None       # (C, 4) or (C, W) padded; with_cells only
    cell_types: Optional[torch.Tensor] = None  # (C,) VTK types; None: tetrahedra


@dataclass
class _Trajectory:
    """Static data of one trajectory, ready for graph building."""

    mesh_pos: torch.Tensor               # (N, 3)    undeformed positions
    node_one_hot: torch.Tensor           # (N, 3)    first columns of x
    actuator_mask: torch.Tensor          # (N,) bool
    mesh_space_features: torch.Tensor    # (E_M, 4)  u_ij, |u_ij|
    globals: Optional[torch.Tensor] = None     # (G,) raw run parameters
    node_static: Optional[torch.Tensor] = None  # (N, S) raw static node features
    walls: Optional[torch.Tensor] = None       # (W_max, 6) walls, zero rows padded
    world_pos: Optional[torch.Tensor] = None   # (T, N, 3); None when lazy
    stress: Optional[torch.Tensor] = None      # (T, N, 1); None when lazy
    rest_key: Optional[Hashable] = None   # mesh + rest positions, if the world edges need them

    @property
    def num_nodes(self) -> int:
        return self.node_one_hot.shape[0]


def _nbytes(value) -> int:
    """Bytes of the tensors of a ``_Mesh`` / ``_Trajectory`` / world edge selection."""
    total = 0
    items = [value] if isinstance(value, dict) else [getattr(value, f.name)
                                                     for f in fields(value)]
    for item in items:
        for tensor in (item.values() if isinstance(item, dict) else [item]):
            if isinstance(tensor, torch.Tensor):
                total += tensor.numel() * tensor.element_size()
    return total


class _StaticCache:
    """Static graph data by key, least recently used first out above ``budget`` bytes
    (None: unbounded). The newest entry always stays."""

    def __init__(self, budget: Optional[int]):
        self.budget = budget
        self._entries: OrderedDict = OrderedDict()   # key -> (value, bytes)
        self._size = 0

    def get(self, key: Hashable, build: Callable):
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            return entry[0]
        value = build()
        size = _nbytes(value)
        self._entries[key] = (value, size)
        self._size += size
        while self.budget is not None and self._size > self.budget and len(self._entries) > 1:
            _, (_, evicted) = self._entries.popitem(last=False)
            self._size -= evicted
        return value


class DeformingPlateDataset(Dataset):
    """One-step samples of one split of a trajectory ``dataset.db``.

    Sample (trajectory k, step s) uses frames s .. s + 1 + history: an
    optional predecessor, the current and the next frame. Its ``Data`` holds
    ``x``, ``y``, ``edge_index``/``edge_attr``, ``world_edge_index``/
    ``world_edge_attr`` (all normalized), ``loss_mask``, the raw current
    positions ``world_pos``, the undeformed ``mesh_pos`` and the static
    attributes of the world edge variant (``body_id`` or ``world_exclude_index``).

    Parameters
    ----------
    db_path : str
        Path to ``dataset.db``.
    split : str
        ``"train"``, ``"val"`` or ``"test"``.
    normalizer : Normalizer, optional
        Defaults to ``normalizer.json`` next to the DB.
    with_cells : bool
        Also attach the cells as ``data.cells`` (+ ``data.cell_types`` for mixed meshes;
        only the .vtu export needs them).
    lazy : bool
        Read the frames of a sample from the DB on access instead of loading
        the split into memory (needs a DB of format v2 or later).
    cache_frames : int
        Lazy mode: keep the last N frames per process (LRU); 0 = no cache.
    static_cache_mb : float
        Lazy mode: keep the static graph data of recently used trajectories and
        meshes up to this size per process (LRU).
    world_edges : str
        World edge variant, one of ``WORLD_EDGE_MODES`` (``"all"``: paper).
    world_edge_exclude_distance : float
        ``self_contact_aware``: pairs with a path along the mesh edges (rest
        state) shorter than this many contact radii get no world edge.
    world_edge_exclude_hops : int, optional
        ``self_contact_aware`` as up to v0.3.0: pairs at most this many mesh
        edges apart get none (instead of ``world_edge_exclude_distance``).
    build_world_edges : bool
        False: samples come without ``world_edge_index`` / ``world_edge_attr``;
        ``graph_features.add_world_edges`` adds them later (e.g. on the GPU).
    """

    def __init__(self, db_path, split: str, normalizer: Optional[Normalizer] = None,
                 with_cells: bool = False, lazy: bool = False, cache_frames: int = 0,
                 world_edges: str = "all",
                 world_edge_exclude_distance: float = DEFAULT_WORLD_EDGE_EXCLUDE_DISTANCE,
                 world_edge_exclude_hops: Optional[int] = None,
                 static_cache_mb: float = DEFAULT_STATIC_CACHE_MB,
                 build_world_edges: bool = True):
        super().__init__()
        if world_edges not in WORLD_EDGE_MODES:
            raise ValueError(f"world_edges must be one of {WORLD_EDGE_MODES}, "
                             f"got {world_edges!r}")
        self.db_path = str(db_path)
        self.split = split
        self.with_cells = with_cells
        self.lazy = lazy
        self.cache_frames = max(0, int(cache_frames))
        self.world_edges = world_edges
        self.world_edge_exclude_distance = float(world_edge_exclude_distance)
        self.world_edge_exclude_hops = (None if world_edge_exclude_hops is None
                                        else int(world_edge_exclude_hops))
        # The selection then depends on the rest positions, not on the mesh alone.
        self._world_selection_by_rest = (world_edges == "self_contact_aware"
                                         and world_edge_exclude_hops is None)
        self.static_cache_bytes = int(float(static_cache_mb) * 2 ** 20) if lazy else None
        self.build_world_edges = build_world_edges
        self.normalizer = normalizer or Normalizer.load(
            str(Path(self.db_path).parent / "normalizer.json"))

        # Lazy mode: opened on first access in each process (see __getstate__).
        self._connection: Optional[sqlite3.Connection] = None
        self._frame_cache: OrderedDict = OrderedDict()
        self._static = _StaticCache(self.static_cache_bytes)

        info, self.trajectories = self._load(split)
        # Layout fixed at export time (DBs from before --history: paper layout).
        self.history = int(info.get("history", 0))
        self.prev_stress = info.get("prev_stress", "1") == "1"
        self.global_names: List[str] = json.loads(info.get("global_names", "[]"))
        self.static_names: List[str] = json.loads(info.get("static_names", "[]"))
        self.node_feature_names = node_feature_names(
            self.history, self.prev_stress, self.global_names, self.static_names,
            walls=self.max_walls > 0)
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
        # Only the dynamic and global columns are normalized here; the one-hot has
        # mean 0 / std 1.
        widths = [len(node_feature_names(self.history, self.prev_stress)) - NUM_NODE_TYPES,
                  len(self.static_names), len(self.global_names),
                  WALL_FEATURE_DIM if self.max_walls else 0]
        bounds = np.cumsum([NUM_NODE_TYPES] + widths).tolist()
        (self._dynamic_mean, self._static_mean, self._global_mean, self._wall_mean) = (
            node_mean[a:b] for a, b in zip(bounds[:-1], bounds[1:]))
        (self._dynamic_std, self._static_std, self._global_std, self._wall_std) = (
            node_std[a:b] for a, b in zip(bounds[:-1], bounds[1:]))
        self._target_mean, self._target_std = self.normalizer.mean_std("target")
        self._mesh_mean, self._mesh_std = self.normalizer.mean_std("mesh_edge_features")
        self._world_mean, self._world_std = self.normalizer.mean_std("world_edge_features")

    # ── Dataset protocol ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        return self.sample(idx, self.build_world_edges)

    def sample(self, idx: int, world_edges: bool = True) -> Data:
        """Sample ``idx``; ``world_edges=False`` leaves out the contact search
        (no ``world_edge_index`` / ``world_edge_attr``, see ``add_world_edges``)."""
        k, step = self.samples[idx]
        info = self.trajectories[k]
        mesh = self._mesh(info)
        traj = self._trajectory(info, mesh)
        selection = self._world_selection(mesh, traj)
        world_pos, stress = self._frames(info, traj, step, 2 + self.history)
        dynamic_inputs, targets, current_pos = trajectory_features(
            world_pos, stress, traj.actuator_mask, self.history, self.prev_stress)
        dynamic_inputs, targets, world_pos = dynamic_inputs[0], targets[0], current_pos[0]

        if world_edges:
            mesh_edge_attr, world_edge_index, world_edge_attr = edge_features(
                mesh.edge_index, traj.mesh_space_features, world_pos, self.contact_radius,
                **selection)
        else:
            mesh_edge_attr = mesh_edge_features(mesh.edge_index, traj.mesh_space_features,
                                                world_pos)
        columns = [traj.node_one_hot, (dynamic_inputs - self._dynamic_mean) / self._dynamic_std]
        if traj.node_static is not None:
            columns.append((traj.node_static - self._static_mean) / self._static_std)
        if traj.globals is not None:   # the same run parameters at every node
            columns.append(((traj.globals - self._global_mean) / self._global_std)
                           .expand(traj.num_nodes, -1))
        if self.max_walls:             # last columns; rebuilt with the edges
            walls = wall_features(world_pos, traj.walls,
                                  WALL_DISTANCE_CUTOFF * self.contact_radius)
            columns.append((walls - self._wall_mean) / self._wall_std)
        data = Data(
            x=torch.cat(columns, dim=-1),
            y=(targets - self._target_mean) / self._target_std,
            edge_index=mesh.edge_index,
            edge_attr=(mesh_edge_attr - self._mesh_mean) / self._mesh_std,
            loss_mask=~traj.actuator_mask,
            world_pos=world_pos,
            mesh_pos=traj.mesh_pos,       # undeformed; mgn_t's positional encoding
            **selection,                  # read back by rebuild_edges
        )
        if world_edges:
            data.world_edge_index = world_edge_index
            data.world_edge_attr = (world_edge_attr - self._world_mean) / self._world_std
        if self.max_walls:
            data.wall_planes = traj.walls   # (W_max, 6), read back by rebuild_edges
        if self.with_cells:
            data.cells = mesh.cells
            if mesh.cell_types is not None:
                data.cell_types = mesh.cell_types
        return data

    def __getstate__(self):
        """Pickling for spawned DataLoader workers: every worker opens its own
        connection and starts with empty caches (in memory: with all data)."""
        state = self.__dict__.copy()
        state["_connection"] = None
        state["_frame_cache"] = OrderedDict()
        if self.lazy:
            state["_static"] = _StaticCache(self.static_cache_bytes)
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

    def _frames(self, info: _TrajectoryInfo, traj: _Trajectory, first: int, count: int
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """world_pos (count, N, 3) and stress (count, N, 1) of frames first .. first+count-1."""
        if traj.world_pos is not None:
            return traj.world_pos[first:first + count], traj.stress[first:first + count]
        frames = [self._read_frame(info, frame) for frame in range(first, first + count)]
        return (torch.stack([world_pos for world_pos, _ in frames]),
                torch.stack([stress for _, stress in frames]))

    def _read_frame(self, info: _TrajectoryInfo, frame: int):
        """Lazy mode: one frame from the LRU cache or the DB."""
        key = (info.db_id, frame)
        cached = self._frame_cache.get(key)
        if cached is not None:
            self._frame_cache.move_to_end(key)
            return cached
        world_pos, stress = self._db().execute(
            "SELECT world_pos, stress FROM frames WHERE trajectory_id = ? AND frame = ?",
            key).fetchone()
        value = (_tensor(world_pos, np.float32, (info.num_nodes, 3)),
                 _tensor(stress, np.float32, (info.num_nodes, 1)))
        if self.cache_frames:
            self._frame_cache[key] = value
            if len(self._frame_cache) > self.cache_frames:
                self._frame_cache.popitem(last=False)
        return value

    def _db(self) -> sqlite3.Connection:
        """Read-only connection of the current process, opened on first use."""
        if self._connection is None:
            self._connection = connect_readonly(self.db_path)
            self._connection.execute(f"PRAGMA mmap_size = {_MMAP_BYTES}")
        return self._connection

    # ── static data ─────────────────────────────────────────────────────────

    def _mesh(self, info: _TrajectoryInfo) -> _Mesh:
        return self._static.get(info.mesh_key, lambda: self._read_mesh(self._db(), info))

    def _trajectory(self, info: _TrajectoryInfo, mesh: _Mesh) -> _Trajectory:
        return self._static.get(("trajectory", info.db_id),
                                lambda: self._read_trajectory(self._db(), info, mesh))

    def _world_selection(self, mesh: _Mesh, traj: _Trajectory) -> Dict[str, torch.Tensor]:
        """Attributes of the world edge variant: of the mesh, or with the path rule of
        ``self_contact_aware`` of the rest geometry (shared by runs with equal geometry)."""
        if traj.rest_key is None:
            return mesh.world_selection
        return self._static.get(traj.rest_key, lambda: world_edge_selection(
            self.world_edges, mesh.edge_index, traj.num_nodes, mesh_pos=traj.mesh_pos,
            exclude_length=self.world_edge_exclude_distance * self.contact_radius))

    def _read_trajectory(self, conn: sqlite3.Connection, info: _TrajectoryInfo,
                         mesh: _Mesh) -> _Trajectory:
        node_type, mesh_pos, global_values, node_static, walls = conn.execute(
            f"SELECT node_type, mesh_pos, {self._optional_columns} FROM trajectories "
            "WHERE id = ?", (info.db_id,)).fetchone()
        num_nodes = info.num_nodes
        raw_node_type = _tensor(node_type, np.uint8, (num_nodes,))
        mesh_pos = _tensor(mesh_pos, np.float32, (num_nodes, 3))
        return _Trajectory(
            mesh_pos=mesh_pos,
            node_one_hot=one_hot_node_type(raw_node_type),
            actuator_mask=raw_node_type == NODE_TYPE_ACTUATOR,
            mesh_space_features=relative_features(mesh_pos, mesh.edge_index),
            globals=None if global_values is None else _tensor(global_values, np.float32, (-1,)),
            node_static=(None if node_static is None
                         else _tensor(node_static, np.float32, (num_nodes, -1))),
            walls=self._padded_walls(walls),
            rest_key=(("world_selection", info.mesh_key,
                       hashlib.sha256(mesh_pos.numpy().tobytes()).hexdigest())
                      if self._world_selection_by_rest else None),
        )

    def _read_mesh(self, conn: sqlite3.Connection, info: _TrajectoryInfo) -> _Mesh:
        """Mesh edges stored by the exporter (v4), or computed from the cells (older)."""
        kind, key = info.mesh_key
        if kind == "mesh":
            cells = "cells, cell_types" if self.with_cells else "NULL, NULL"
            num_cells, cells, cell_types, edges = conn.execute(
                f"SELECT num_cells, {cells}, mesh_edges FROM meshes WHERE id = ?",
                (key,)).fetchone()
            edge_index = torch.from_numpy(
                np.frombuffer(edges, np.int32).reshape(2, -1).astype(np.int64))
        else:
            cell_types = "cell_types" if "cell_types" in self._trajectory_columns else "NULL"
            num_cells, cells, cell_types = conn.execute(
                f"SELECT num_cells, cells, {cell_types} FROM trajectories WHERE id = ?",
                (key,)).fetchone()
            edge_index = None
        if cells is not None:
            cells = np.frombuffer(cells, np.int32).reshape(num_cells, -1)
        if cell_types is not None:
            cell_types = np.frombuffer(cell_types, np.uint8)
        if edge_index is None:
            edge_index = mesh_edges_from_cells(cells, cell_types)
        return _Mesh(
            edge_index=edge_index,
            world_selection=({} if self._world_selection_by_rest else world_edge_selection(
                self.world_edges, edge_index, info.num_nodes, self.world_edge_exclude_hops)),
            cells=(torch.from_numpy(cells.astype(np.int64))
                   if self.with_cells and cells is not None else None),
            cell_types=(None if cell_types is None or not self.with_cells
                        else torch.from_numpy(cell_types.copy())),
        )

    # ── loading ─────────────────────────────────────────────────────────────

    def _load(self, split: str) -> Tuple[Dict[str, str], List[_TrajectoryInfo]]:
        """``info`` table and the trajectories of ``split``; in memory also their
        static data and frames."""
        conn = connect_readonly(self.db_path)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trajectories'"
            ).fetchone()
            if not has_table:
                raise ValueError(
                    f"{self.db_path} is not a trajectory DB (no table 'trajectories') — "
                    "re-export it: python -m dataset_preprocessor.export --out-dir <dir>")
            info = dict(conn.execute("SELECT key, value FROM info").fetchall())
            self.max_walls = int(info.get("max_walls", 0))   # needed by _padded_walls
            self.contact_radius = float(info["contact_radius"])   # and by _world_selection
            db_format = info.get("format")
            if db_format not in DB_FORMATS:
                hint = (" — the export did not finish; continue it with --resume"
                        if db_format is None else "")
                raise ValueError(f"{self.db_path}: unsupported format {db_format!r}{hint}")
            if self.lazy and db_format == LEGACY_DB_FORMAT:
                raise ValueError(f"{self.db_path} has the old format {db_format!r}; lazy "
                                 "loading needs one row per frame — re-export it: "
                                 "python -m dataset_preprocessor.export")

            self._trajectory_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(trajectories)")}
            self._optional_columns = ", ".join(
                name if name in self._trajectory_columns else f"NULL AS {name}"
                for name in ("globals", "node_static", "walls"))
            mesh = "mesh_id" if db_format == DB_FORMAT else "NULL"
            rows = conn.execute(
                f"SELECT id, sequence_id, num_nodes, num_frames, {mesh} FROM trajectories "
                "WHERE split = ? ORDER BY id", (split,)).fetchall()
            trajectories = [
                _TrajectoryInfo(db_id, sequence_id, num_nodes, num_frames,
                                ("cells", db_id) if mesh_id is None else ("mesh", mesh_id))
                for db_id, sequence_id, num_nodes, num_frames, mesh_id in rows]
            if not self.lazy:   # everything into memory now; no DB access afterwards
                for traj_info in trajectories:
                    mesh = self._static.get(traj_info.mesh_key,
                                            lambda: self._read_mesh(conn, traj_info))
                    traj = self._static.get(
                        ("trajectory", traj_info.db_id),
                        lambda: self._read_trajectory(conn, traj_info, mesh))
                    traj.world_pos, traj.stress = self._read_all_frames(conn, db_format,
                                                                        traj_info)
                    self._world_selection(mesh, traj)
        finally:
            conn.close()
        return info, trajectories

    @staticmethod
    def _read_all_frames(conn, db_format: str, info: _TrajectoryInfo):
        """All frames of a trajectory: world_pos (T, N, 3), stress (T, N, 1)."""
        if db_format == LEGACY_DB_FORMAT:
            world_pos, stress = conn.execute(
                "SELECT world_pos, stress FROM trajectories WHERE id = ?",
                (info.db_id,)).fetchone()
        else:
            rows = conn.execute(
                "SELECT world_pos, stress FROM frames WHERE trajectory_id = ? ORDER BY frame",
                (info.db_id,)).fetchall()
            if len(rows) != info.num_frames:
                raise ValueError(f"trajectory {info.sequence_id}: {len(rows)} frames in the "
                                 f"DB, {info.num_frames} expected")
            # Frame blobs in order = the (T, N, ...) layout of the whole trajectory.
            world_pos = b"".join(row[0] for row in rows)
            stress = b"".join(row[1] for row in rows)
        shape = (info.num_frames, info.num_nodes)
        return (_tensor(world_pos, np.float32, shape + (3,)),
                _tensor(stress, np.float32, shape + (1,)))

    def _padded_walls(self, walls) -> Optional[torch.Tensor]:
        """(W_max, 6) walls of a trajectory, zero rows (no wall) padded; None without walls."""
        if not self.max_walls:
            return None
        padded = torch.zeros(self.max_walls, 6)
        if walls is not None:
            values = _tensor(walls, np.float32, (-1, 6))
            padded[:len(values)] = values
        return padded


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
