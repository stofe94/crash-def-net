"""
export.py — Export simulation runs into a compact dataset.

Reads the raw data once and writes into ``--out-dir``. Raw formats
(``--raw-format``): DeepMind's TFRecord (default), LS-DYNA and OpenRadioss runs
(``d3plot``, read by ``d3plot_reader.py``, the OpenRadioss starter deck next to it by
``radioss_deck.py``; nodal stress smoothed by the element representation) or VTK time
series (``vtk``, read by ``vtk_reader.py``). ``--config`` defines the dataset in a TOML
file: its components with their roles and its features (``export_config.py``).

``dataset.db``
    SQLite (schema: ``dataset.DB_SCHEMA``): every distinct mesh once (cells and
    mesh edges), the static data of every trajectory, world_pos and the result
    fields (deforming_plate: the stress) of every selected frame as its own row,
    and the running statistics of the train split.
``normalizer.json``
    mean/std per feature group, fitted on the train split.
``metadata.json``
    feature layout, dimensions, splits and contact radius.

Train/val/test are disjoint. With one source (e.g. ``test.tfrecord`` with 100
trajectories) they are contiguous ranges of its runs, train first; with three
(e.g. DeepMind's ``train`` / ``valid`` / ``test.tfrecord``) each split comes
from its own. ``select_seed`` draws the runs of each split at random instead
(without replacement; read in ascending order); the chosen runs are stored in
``metadata.json``. Frames are subsampled with ``time_stride``: frames 0,
stride, 2·stride, ... (paper: stride 1).

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
    python -m dataset_preprocessor.export --config export.toml   # see export_config.py
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sqlite3
import struct
import sys
import time
import warnings
from dataclasses import dataclass, replace
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):   # executed as a file, e.g. "Run Python File"
    sys.path.insert(0, str(_REPO_ROOT))

from dataset_preprocessor.dataset import (  # noqa: E402
    DB_FORMAT,
    DB_SCHEMA,
    GLOBAL_INPUT_GRAPH,
    RENAMED_DB_FORMAT,
    RENAMED_KEYS,
    current_names,
)
from dataset_preprocessor.export_config import (  # noqa: E402
    ROLE_DEFORMABLE,
    ROLES,
    component_roles,
    expand_contact_pairs,
    parse_components,
    read_export_config,
)
from dataset_preprocessor.graph_features import (  # noqa: E402
    CELL_HEXAHEDRON,
    CELL_LINE,
    CELL_PYRAMID,
    CELL_QUAD,
    CELL_TETRA,
    CELL_TRIANGLE,
    CELL_WEDGE,
    DEFAULT_FIELD_NAMES,
    GIVEN_SUFFIX,
    MESH_EDGE_FEATURE_DIM,
    MESH_EDGE_FEATURE_NAMES,
    NODE_TYPE_ACTUATOR,
    NUM_NODE_TYPES,
    PLANAR_WALL_COLUMNS,
    WALL_DISTANCE_CUTOFF,
    WALL_FEATURE_DIM,
    WORLD_EDGE_FEATURE_DIM,
    WORLD_EDGE_FEATURE_NAMES,
    before_first_frame,
    edge_features,
    mesh_edges_from_cells,
    node_feature_names,
    relative_features,
    target_feature_names,
    trajectory_features,
    wall_features,
)
from utils.normalizer import Normalizer  # noqa: E402
from utils.units import LENGTH_UNITS, SI_LENGTH, SI_STRESS, STRESS_UNITS  # noqa: E402

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
    sources="test.tfrecord",                       # d3plot: runs converted from it
    out_dir="data/deforming_plate_s10_small",   # == [data].data_dir in the config
    n_train=8, n_val=2, n_test=2,                  # whole trajectories (= geometries)
    select_seed=None,                              # None: the first runs; int: random
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
# Every frame, as in the paper: --num-steps 400 --time-stride 1 (~0.85 GB, ~4 min)
# DeepMind's original splits, trajectories drawn at random, stride 4:
#   python -m dataset_preprocessor.export --out-dir data/deforming_plate_s4_t150 \
#       --source-tfrecord train.tfrecord valid.tfrecord test.tfrecord --select-seed 42 \
#       --n-train 100 --n-val 30 --n-test 20 --num-steps 100 --time-stride 4

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
    yields them either way. ``stress`` holds the result fields per node: the
    von-Mises stress (T, N) alone, or F fields (T, N, F) named by ``field_names``.
    """

    node_type: np.ndarray   # (N,)      raw node type {0, 1, 3}
    mesh_pos: np.ndarray    # (N, 3)    float32, rest configuration
    cells: np.ndarray       # (C, 4)    int32 tetrahedra, or (C, W) padded with -1
    world_pos: Optional[np.ndarray]   # (T, N, 3) float32; None: see frame_blocks
    stress: Optional[np.ndarray]      # (T, N) von-Mises stress, or (T, N, F) result fields
    cell_types: Optional[np.ndarray] = None       # (C,) VTK cell types; None: tetrahedra
    globals: Optional[Dict[str, float]] = None    # run parameters, e.g. impact velocity
    node_static: Optional[Dict[str, np.ndarray]] = None   # name -> (N,), e.g. thickness
    walls: Optional[np.ndarray] = None            # (W, 6) planes: point, normal; (W, 7) cylinders
    frame_blocks: Optional[Iterator[Tuple[np.ndarray, np.ndarray]]] = None  # as blocks()
    field_names: Optional[List[str]] = None       # of the F fields; None: ["stress"]
    component: Optional[np.ndarray] = None        # (N,) of every node, into component_names
    component_names: Optional[List[str]] = None   # e.g. the parts of a crash model
    rigid_bodies: Optional[List[str]] = None      # components that move as free rigid bodies
    component_roles: Optional[Dict[str, str]] = None   # with --components: name -> role
    left_out: Optional[List[str]] = None          # with --components: parts not in the data
    # (N, 3) motion over one frame interval before frame 0, at the initial velocity of the
    # decks: the velocity input of frame 0 with history; None: at rest before frame 0
    start_velocity: Optional[np.ndarray] = None

    def blocks(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """The frames in order, as (world_pos (t, N, 3), fields (t, N, F)) blocks."""
        blocks = [(self.world_pos, self.stress)] if self.frame_blocks is None \
            else self.frame_blocks
        for world_pos, fields in blocks:
            yield world_pos, fields if fields.ndim == 3 else fields[..., None]

    def result_field_names(self) -> List[str]:
        return list(DEFAULT_FIELD_NAMES if self.field_names is None else self.field_names)


# ─────────────────────────────────────────────────────────────────────────────
# Reading the raw data
# ─────────────────────────────────────────────────────────────────────────────

def read_raw_meta(raw_dir) -> dict:
    """DeepMind ``meta.json`` (field shapes, dtypes, trajectory length)."""
    with open(os.path.join(raw_dir, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def read_raw_trajectories(raw_dir, tfrecord_name: str, frames: Sequence[int],
                          count: Optional[int],
                          indices: Optional[Sequence[int]] = None) -> Iterator[RawTrajectory]:
    """Yield the first ``count`` trajectories of ``tfrecord_name`` (None: all), or
    those at the ascending positions ``indices``.

    Dynamic fields are restricted to ``frames``; static fields (mesh, node
    type, cells) are stored once per trajectory in the file.
    """
    from tfrecord.torch.dataset import TFRecordDataset

    meta = read_raw_meta(raw_dir)
    records = TFRecordDataset(os.path.join(raw_dir, tfrecord_name), None,
                              {name: "byte" for name in meta["field_names"]})
    frames = np.asarray(frames)
    if indices is not None:
        count, wanted = max(indices, default=-1) + 1, set(indices)

    def field(record, name):
        spec = meta["features"][name]
        values = np.frombuffer(record[name], dtype=getattr(np, spec["dtype"]))
        values = values.reshape(spec["shape"])
        return values[0] if spec["type"] == "static" else values[frames]

    for index, record in enumerate(records):
        if count is not None and index >= count:
            break
        if indices is not None and index not in wanted:
            continue
        yield RawTrajectory(
            node_type=field(record, "node_type")[:, 0],
            mesh_pos=field(record, "mesh_pos"),
            cells=field(record, "cells"),
            world_pos=field(record, "world_pos"),
            stress=field(record, "stress")[..., 0],
        )


def num_records(path) -> int:
    """Number of records in a TFRecord file, from the record headers alone."""
    count, size = 0, os.path.getsize(path)
    with open(path, "rb") as f:
        while f.tell() < size:   # length (uint64), its CRC, data, data CRC (uint32 each)
            (length,) = struct.unpack("<Q", f.read(8))
            f.seek(length + 8, os.SEEK_CUR)
            count += 1
    return count


def _run_dirs(raw_format: str):
    if raw_format == "d3plot":
        from dataset_preprocessor.d3plot_reader import run_dirs
    elif raw_format == "vtk":
        from dataset_preprocessor.vtk_reader import run_dirs
    else:
        raise ValueError(f"raw_format must be one of {RAW_FORMATS}, got {raw_format!r}")
    return run_dirs


def select_runs(raw_format: str, raw_dir, sources: Sequence[str], counts: Dict[str, int],
                seed: Optional[int] = None) -> Dict[str, Tuple[str, List[int]]]:
    """``split -> (source, positions)``: the runs of every split in ``counts``.

    One source: disjoint ranges of its runs, in the order of ``counts``; one source
    per split: the first runs of each. ``seed`` draws them at random instead,
    without replacement; the positions of a split are ascending. A position
    counts the records of a TFRecord or the run folders of ``source``.
    """
    names = dict(zip(counts, sources if len(sources) > 1 else list(sources) * len(counts)))
    rng = None if seed is None else np.random.default_rng(seed)
    selection: Dict[str, Tuple[str, List[int]]] = {}
    for name in dict.fromkeys(names.values()):   # every source once, in split order
        splits = [split for split in counts if names[split] == name]
        needed = sum(counts[split] for split in splits)
        available = (num_records(Path(raw_dir) / name) if raw_format == "tfrecord"
                     else len(_run_dirs(raw_format)(raw_dir, name)))
        if needed > available:
            raise ValueError(f"{name} ({raw_format}) holds only {available} trajectories, "
                             f"{needed} requested ({', '.join(splits)})")
        positions = np.arange(needed) if rng is None else rng.choice(available, needed,
                                                                     replace=False)
        start = 0
        for split in splits:
            selection[split] = (name, sorted(positions[start:start + counts[split]].tolist()))
            start += counts[split]
    return {split: selection[split] for split in counts}


def raw_runs(raw_format: str, raw_dir, runs: Sequence[Tuple[str, int]], frames: Sequence[int],
             reader_options: Optional[Dict] = None, block_bytes: Optional[int] = None
             ) -> Iterator[Tuple[str, Callable[[], RawTrajectory]]]:
    """``(source, read)`` of the ``runs`` (source, position), in their order.

    ``read()`` returns the run as a ``RawTrajectory`` (LS-DYNA and VTK runs with
    their frames in blocks of about ``block_bytes``, read with ``reader_options``:
    the keyword arguments of the reader's ``read_raw_run``); a run that is skipped is
    not read. ``source`` names the run: the TFRecord record, or the run folder
    relative to ``raw_dir``. A TFRecord is read in one pass per ascending stretch.
    """
    stretches: List[Tuple[str, List[int]]] = []
    for name, position in runs:
        if stretches and stretches[-1][0] == name and position > stretches[-1][1][-1]:
            stretches[-1][1].append(position)
        else:
            stretches.append((name, [position]))
    if raw_format == "tfrecord":
        for name, positions in stretches:
            for index, traj in zip(positions, read_raw_trajectories(raw_dir, name, frames,
                                                                    None, positions)):
                yield f"{name}[{index}]", (lambda traj=traj: traj)
        return
    if raw_format == "d3plot":
        from dataset_preprocessor.d3plot_reader import read_raw_run
    else:
        from dataset_preprocessor.vtk_reader import read_raw_run
    options = dict(reader_options or {})
    root = Path(raw_dir).resolve()
    for name, positions in stretches:
        run_dirs = _run_dirs(raw_format)(raw_dir, name)
        for run_dir in (run_dirs[p] for p in positions):
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
    NaN means no value and counts in nothing: once one appears, ``counts`` holds the
    values per column (stored behind the means).
    """

    def __init__(self, dim: int):
        self.count = 0
        self.counts: Optional[torch.Tensor] = None
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.m2 = torch.zeros(dim, dtype=torch.float64)   # sum of squared deviations

    def _add_with_gaps(self, rows: torch.Tensor, gaps: torch.Tensor) -> None:
        if self.counts is None:
            self.counts = torch.full_like(self.mean, float(self.count))
        values = (~gaps).sum(0).double()
        batch_mean = torch.nansum(rows, 0) / values.clamp_min(1)
        batch_m2 = torch.nansum((rows - batch_mean) ** 2, 0)
        total = self.counts + values
        weight = values / total.clamp_min(1)
        delta = batch_mean - self.mean
        self.mean += delta * weight
        self.m2 += batch_m2 + delta ** 2 * self.counts * weight
        self.counts = total
        self.count += rows.shape[0]

    def add(self, rows: torch.Tensor) -> None:
        rows = rows.reshape(-1, self.mean.shape[0]).double()
        n = rows.shape[0]
        if n == 0:
            return
        gaps = torch.isnan(rows)
        if self.counts is not None or gaps.any():
            self._add_with_gaps(rows, gaps)
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
        count = self.count if self.counts is None else self.counts.clamp_min(2)
        std = torch.sqrt(self.m2 / (count - (1 if unbiased else 0)))
        std = torch.where(std < min_std, torch.ones_like(std), std)
        return self.mean.float().numpy(), std.float().numpy()

    def state(self) -> Tuple[int, bytes, bytes]:
        means = self.mean if self.counts is None else torch.cat((self.mean, self.counts))
        return self.count, means.numpy().tobytes(), self.m2.numpy().tobytes()

    @classmethod
    def from_state(cls, count: int, mean: bytes, m2: bytes) -> "_RunningMoments":
        moments = cls(0)
        moments.count = count
        moments.m2 = torch.from_numpy(np.frombuffer(m2, np.float64).copy())
        means = torch.from_numpy(np.frombuffer(mean, np.float64).copy())
        moments.mean = means[:moments.m2.shape[0]]
        if means.shape[0] > moments.m2.shape[0]:
            moments.counts = means[moments.m2.shape[0]:]
        return moments


class NormalizerFitter:
    """Fit all normalizer groups on the train trajectories, block by block.

    * ``node_features``: the one-hot columns and the ``_given`` columns of static
      features get mean 0 / std 1 (pass through); the dynamic columns are standardized
      over all nodes and samples, static node features over the train nodes, run
      parameters over the train trajectories (one value each, population std).
    * ``target`` and ``target_delta`` (history only; change w.r.t. the
      ``*_prev`` value, for ``residual_target``): only nodes in the loss —
      the actuator's prescribed velocity would otherwise dominate.
    * A node without a value (NaN) counts in no statistics, see ``_RunningMoments``.
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
                 edge_stats_stride: int = 1, first_frame: Optional[int] = None):
        if edge_stats_stride < 1:
            raise ValueError(f"edge_stats_stride must be >= 1, got {edge_stats_stride}")
        self.contact_radius = contact_radius
        self.history = history
        # Current frame of the first sample (as up to v0.4.0: history); 0 with history
        # starts from the velocity at frame 0 (RawTrajectory.start_velocity).
        self.first_frame = history if first_frame is None else first_frame
        self.prev_stress = prev_stress
        self.edge_stats_stride = edge_stats_stride
        # Sized by the result fields: with the first train trajectory (or restore()).
        self._dynamic: Optional[_RunningMoments] = None
        self._target: Optional[_RunningMoments] = None
        self._delta: Optional[_RunningMoments] = None
        self._mesh = _RunningMoments(MESH_EDGE_FEATURE_DIM)
        self._world = _RunningMoments(WORLD_EDGE_FEATURE_DIM)
        self._walls = _RunningMoments(WALL_FEATURE_DIM)
        self.global_names: List[str] = []   # set by the exporter from the first trajectory
        self._globals: Optional[_RunningMoments] = None
        self.static_names: List[str] = []   # likewise
        self._static: Optional[_RunningMoments] = None
        self.field_names: List[str] = list(DEFAULT_FIELD_NAMES)   # likewise
        self._current: Optional[Dict] = None   # the trajectory being added

    def add(self, traj: RawTrajectory, mesh_edge_index: Optional[torch.Tensor] = None) -> None:
        """Fold all samples of one train trajectory into the statistics."""
        self.start(traj, mesh_edge_index)
        for world_pos, stress in traj.blocks():
            self.add_frames(world_pos, stress)

    def start(self, traj: RawTrajectory, mesh_edge_index: Optional[torch.Tensor] = None) -> None:
        """Begin a train trajectory: its run parameters, static node features and mesh."""
        if self._target is None:
            num_dynamic = len(node_feature_names(self.history, self.prev_stress,
                                                 field_names=self.field_names)) - NUM_NODE_TYPES
            num_targets = len(target_feature_names(self.field_names))
            self._dynamic = _RunningMoments(num_dynamic)
            self._target = _RunningMoments(num_targets)
            self._delta = _RunningMoments(num_targets)
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
            # the frames before frame 0 that the first sample needs: at the start velocity
            "before": (None if self.first_frame >= self.history
                       else _start_motion(traj)),
        }

    def add_frames(self, world_pos: np.ndarray, stress: np.ndarray) -> None:
        """Fold in the samples whose last frame is in this block (t, N, 3) / (t, N, F)."""
        current = self._current
        world_pos = torch.tensor(world_pos, dtype=torch.float32)       # (T, N, 3)
        stress = torch.tensor(stress, dtype=torch.float32)             # (T, N, F) fields
        if current["carry"] is not None:   # samples reaching back into the last block
            world_pos = torch.cat((current["carry"][0], world_pos))
            stress = torch.cat((current["carry"][1], stress))
        elif current["before"] is not None:   # the first block: frame 0 is a sample too
            world_pos, stress = before_first_frame(world_pos, stress, current["before"])
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
        # Run parameters: one value per train trajectory (population std), in their
        # own group for data.u (dataset.GLOBAL_INPUT_GRAPH).
        if self.global_names:
            normalizer.set_stats("global_features", *self._globals.mean_std(
                "trajectories", False, _MIN_FEATURE_STD, min_count=1))
        means = [np.zeros(NUM_NODE_TYPES, np.float32), dynamic_mean]
        stds = [np.ones(NUM_NODE_TYPES, np.float32), dynamic_std]
        if self.static_names:
            static_mean, static_std = self._static.mean_std("nodes", True, _MIN_FEATURE_STD)
            for column, name in enumerate(self.static_names):
                if name.endswith(GIVEN_SUFFIX):
                    static_mean[column], static_std[column] = 0.0, 1.0
            means.append(static_mean)
            stds.append(static_std)
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


def window_statistics(data_dir, start_frame: Optional[int] = None,
                      end_frame: Optional[int] = None,
                      block_mb: float = EXPORT_DEFAULTS["block_mb"]) -> Normalizer:
    """``normalizer.json`` of an export, with the statistics of the node inputs that change
    over a run (``bc_velocity``, the ``*_prev`` inputs) and of the targets fitted anew on the
    train samples from the raw frame ``start_frame`` to ``end_frame`` (``[data]`` of a
    training): a part of the run without its first frames then leaves e.g. the jump of the
    stress at the impact out of the scale. Static node features, run parameters, walls and
    edges keep those of the export (the edges would need a contact search per sample). One
    pass over the frames of the train split in this part, in blocks of about ``block_mb``."""
    from dataset_preprocessor.dataset import TrajectoryDataset

    normalizer = Normalizer.load(str(Path(data_dir) / "normalizer.json"))
    dataset = TrajectoryDataset(Path(data_dir) / "dataset.db", "train", normalizer, lazy=True,
                                start_frame=start_frame, end_frame=end_frame)
    history, num_targets = dataset.history, len(dataset.target_feature_names)
    num_dynamic = len(node_feature_names(history, dataset.prev_stress,
                                         field_names=dataset.field_names)) - NUM_NODE_TYPES
    dynamic, target, delta = (_RunningMoments(n) for n in (num_dynamic, num_targets, num_targets))
    # The first frame of every sample: -1 is the frame before frame 0.
    first, stop = dataset.initial_frame - history, dataset.last_frame - history
    for k, info in enumerate(dataset.trajectories):
        frame_bytes = 4 * (3 + len(dataset.field_names)) * info.num_nodes
        per_block = max(1, int(block_mb * 2 ** 20) // frame_bytes - 1 - history)
        for begin in range(first, stop, per_block):
            end = min(begin + per_block, stop)
            world_pos, fields, actuator = dataset.trajectory_frames(k, begin,
                                                                    end - begin + 1 + history)
            inputs, targets, _ = trajectory_features(world_pos, fields, actuator, history,
                                                     dataset.prev_stress)
            dynamic.add(inputs)
            target.add(targets[:, ~actuator])
            if history:   # as NormalizerFitter: the change against the *_prev values
                previous = torch.cat((world_pos[1:-1] - world_pos[:-2], fields[1:-1]), dim=-1)
                delta.add((targets - previous)[:, ~actuator])
    mean, std = (values.numpy().copy() for values in normalizer.mean_std("node_features"))
    columns = slice(NUM_NODE_TYPES, NUM_NODE_TYPES + num_dynamic)
    mean[columns], std[columns] = dynamic.mean_std("samples", True, _MIN_FEATURE_STD)
    normalizer.set_stats("node_features", mean, std)
    normalizer.set_stats("target", *target.mean_std("samples", True, _MIN_FEATURE_STD))
    if history:
        normalizer.set_stats("target_delta", *delta.mean_std("samples", True, _MIN_FEATURE_STD))
    return normalizer


# ─────────────────────────────────────────────────────────────────────────────
# Writing the dataset
# ─────────────────────────────────────────────────────────────────────────────

def _start_motion(traj: RawTrajectory) -> torch.Tensor:
    """(N, 3) start velocity of a run, per frame interval; zero: at rest before frame 0."""
    if traj.start_velocity is None:
        return torch.zeros(traj.mesh_pos.shape[0], 3)
    return torch.tensor(traj.start_velocity, dtype=torch.float32)


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
                       fitter: Optional[NormalizerFitter] = None,
                       component_names: Optional[List[str]] = None,
                       with_start: bool = False) -> None:
    """One ``trajectories`` row plus one ``frames`` row per frame, block by block;
    with ``fitter`` (train split) the frames go into the statistics as well. The
    components of the run are stored as indices into ``component_names``, the
    components of the dataset, to which new names are appended; ``with_start`` stores
    the start velocity of the run."""
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
    components = None
    if traj.component is not None:
        for name in traj.component_names:
            if name not in component_names:
                component_names.append(name)
        to_dataset = np.array([component_names.index(name) for name in traj.component_names],
                              dtype=np.uint32)
        components = to_dataset[np.asarray(traj.component)].tobytes()
    start = _start_motion(traj).numpy().tobytes() if with_start else None
    digest = hashlib.sha256()
    for blob in static + tuple(blob for blob in (components, start) if blob is not None):
        _hash(digest, blob)
    conn.execute(
        "INSERT INTO trajectories (id, split, sequence_id, source, mesh_id, num_nodes, "
        "num_frames, node_type, mesh_pos, globals, node_static, walls, sha256, components, "
        "start_velocity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (index, split, f"{split}_traj{index:04d}", source, mesh_id, traj.mesh_pos.shape[0],
         num_frames, *static, "", components, start))
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
    """SHA-256 over the ``info`` settings and the hashes of every trajectory and mesh. It
    identifies the content: the settings go in under their names up to v0.4.0, so that
    a new name leaves the fingerprint of the same export as it was."""
    digest = hashlib.sha256()
    earlier = {name: old for old, name in RENAMED_KEYS.items()}
    rows = [(earlier.get(key, key), RENAMED_DB_FORMAT if key == "format" else value)
            for key, value in conn.execute("SELECT key, value FROM info "
                                           "WHERE key != 'fingerprint'")]
    for row in sorted(rows):
        digest.update(json.dumps(row).encode())
    for row in conn.execute("SELECT t.id, t.split, t.sequence_id, m.sha256, t.sha256 "
                            "FROM trajectories t JOIN meshes m ON m.id = t.mesh_id "
                            "ORDER BY t.id"):
        digest.update(json.dumps(row).encode())
    return digest.hexdigest()


def _create_db(db_path: Path, settings: Dict[str, str], num_nodes: int,
               num_fields: int = 1) -> sqlite3.Connection:
    """New, empty ``dataset.db`` with the export settings in ``info``."""
    conn = sqlite3.connect(str(db_path))
    frame_bytes = 4 * (3 + num_fields) * num_nodes   # world_pos + result fields, float32
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
    info = current_names(dict(conn.execute("SELECT key, value FROM info")))
    changed = [f"{key}: {info.get(key)} -> {value}" for key, value in settings.items()
               if info.get(key) != value]
    if changed:
        conn.close()
        raise ValueError(f"{db_path} was exported with other settings ({'; '.join(changed)}) "
                         "— --resume needs the same options; --no-resume starts anew")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(trajectories)")}
    for column in ("components", "start_velocity"):   # older exporters
        if column not in columns:
            conn.execute(f"ALTER TABLE trajectories ADD COLUMN {column} BLOB")
    return conn


def _check_duplicates(conn: sqlite3.Connection, index: int, split: str, source: str) -> None:
    """A run with the content of another one: in another split that is data leakage, which
    stops the export; in the same split a warning. ``select_runs`` only keeps the paths apart."""
    twins = conn.execute("SELECT split, source FROM trajectories WHERE id != ? AND sha256 = "
                         "(SELECT sha256 FROM trajectories WHERE id = ?)", (index, index)).fetchall()
    across = [f"{other} ({other_split})" for other_split, other in twins if other_split != split]
    if across:
        raise ValueError(f"{source} ({split}) has the same content as {', '.join(across)}: the "
                         "same sample would be in two splits")
    if twins:
        warnings.warn(f"{source} has the same content as {', '.join(t[1] for t in twins)} in the "
                      f"{split} split")


def _run_parameters(values: Optional[Dict[str, float]], source: str,
                    table: Optional[Dict[str, Dict[str, float]]],
                    wanted: Optional[Sequence[str]]) -> Optional[Dict[str, float]]:
    """The run parameters of a run: those of the reader or the entry of its folder in
    ``table``, restricted to ``wanted`` (None: all)."""
    if table is not None:
        name = Path(source).name
        if name not in table:
            raise ValueError(f"{source}: no entry {name!r} in the file of run parameters")
        values = {str(key): float(value) for key, value in table[name].items()}
    if wanted is None:
        return values
    missing = sorted(set(wanted) - set(values or {}))
    if missing:
        raise ValueError(f"{source}: run parameters {missing} are missing; the run has "
                         f"{sorted(values or {})}")
    return {name: values[name] for name in wanted} or None


def _check_shell_thickness(traj: RawTrajectory, source: str) -> None:
    """Every shell of a deformable component needs a thickness, so that a missing one does
    not pass as 0; solids have none, and a rigid part needs none."""
    thickness = (traj.node_static or {}).get("thickness")
    if thickness is None or traj.cell_types is None:
        return
    nodes = np.unique(traj.cells[np.isin(traj.cell_types, (CELL_TRIANGLE, CELL_QUAD))])
    nodes = nodes[nodes >= 0]
    names = [traj.component_names[k] for k in np.unique(
        traj.component[nodes[~(thickness[nodes] > 0)]]).tolist()]
    names = [name for name in names if traj.component_roles.get(name) == ROLE_DEFORMABLE]
    if names:
        raise ValueError(f"{source}: the shells of the parts {names} have no thickness in the "
                         "d3plot and the deck")


def _fields_given(traj: RawTrajectory) -> np.ndarray:
    """(N, F) whether the nodes have a value of each result field, from the first frame;
    the frames of the run stay as they are."""
    if traj.frame_blocks is None:
        fields = traj.stress[0]
    else:
        blocks = iter(traj.frame_blocks)
        first = next(blocks)
        traj.frame_blocks = chain([first], blocks)
        fields = first[1][0]
    return ~np.isnan(fields if fields.ndim == 2 else fields[:, None])


def _mark_missing_features(traj: RawTrajectory, static_names: Sequence[str],
                           fields_given: np.ndarray, field_inputs: bool) -> None:
    """Give a static node feature that some nodes lack (NaN) its ``_given`` column: 1 where
    it has a value. With ``field_inputs`` (``<field>_prev`` in ``x``) the same for a result
    field. A run without gaps gets it too if the first run of the dataset has it."""
    columns = [(name, ~np.isnan(values)) for name, values in (traj.node_static or {}).items()]
    if field_inputs:
        columns += [(name, fields_given[:, k]) for k, name in enumerate(traj.result_field_names())]
    for name, given in columns:
        if not given.all() or name + GIVEN_SUFFIX in static_names:
            traj.node_static = {**(traj.node_static or {})}
            traj.node_static.setdefault(name + GIVEN_SUFFIX, given.astype(np.float32))


def _describe_components(traj: RawTrajectory) -> str:
    """The parts of a run in the dataset by role, and those left out."""
    lines = [f"  {role}: {', '.join(names)}" for role, names in
             ((role, [n for n, r in traj.component_roles.items() if r == role]) for role in ROLES)
             if names]
    return "\n".join(["Components of the dataset:"] + lines
                     + [f"  left out: {', '.join(traj.left_out) or 'none'}"])


def write_dataset(
    raw_dir: str,
    out_dir: str,
    sources: Union[str, Sequence[str]] = "test.tfrecord",
    n_train: int = 70,
    n_val: int = 15,
    n_test: int = 15,
    select_seed: Optional[int] = None,
    num_steps: int = 400,
    time_stride: int = 1,
    history: int = 0,
    prev_stress: bool = True,
    raw_format: str = "tfrecord",
    contact_radius: Optional[float] = None,
    vtk_options: Optional[Dict] = None,
    d3plot_options: Optional[Dict] = None,
    length_unit: str = SI_LENGTH,
    stress_unit: str = SI_STRESS,
    resume: bool = False,
    edge_stats_stride: int = 1,
    block_mb: float = EXPORT_DEFAULTS["block_mb"],
    progress: Optional[Callable[[str], None]] = None,
    contact_pairs: Optional[Sequence[Sequence[str]]] = None,
    global_features: Optional[Sequence[str]] = None,
    globals_file: Optional[str] = None,
    export_config: Optional[Dict] = None,
) -> Dict[str, str]:
    """Write ``dataset.db``, ``normalizer.json`` and ``metadata.json`` to ``out_dir``.

    ``history`` / ``prev_stress`` fix the node feature layout; they are stored
    in the DB's ``info`` table and in ``metadata.json``. ``raw_format``
    ``"d3plot"`` / ``"vtk"`` read LS-DYNA runs / VTK time series (below
    ``raw_dir/<source>`` or ``raw_dir``) instead of the TFRecord;
    ``vtk_options`` are passed to ``vtk_reader.read_raw_run`` (field names, parts),
    ``d3plot_options`` to ``d3plot_reader.read_raw_run`` (result fields, static node
    features). ``length_unit`` / ``stress_unit`` name the units of the raw data
    (``utils.units``); the values are kept as they are.
    ``sources`` names one source or one per split (train, val, test);
    ``select_seed`` draws the runs at random (see ``select_runs``).
    ``contact_radius`` overrides the ``collision_radius`` of ``meta.json``,
    which such raw data may lack. ``contact_pairs`` (pairs of component names, e.g.
    ``[("plate", "impactor")]``) are the contact rules of raw data with components:
    world edges then only join nodes of these pairs. ``global_features`` restricts the
    run parameters, ``globals_file`` (JSON: run folder -> parameters) holds them instead of
    a ``globals.json`` per run; ``export_config`` (the content of a ``--config`` file) is
    stored with the dataset. An existing ``dataset.db`` is
    replaced, or with
    ``resume`` continued (see the module docstring); ``progress`` receives one
    line per run. Returns the paths of the three files.
    """
    if raw_format not in RAW_FORMATS:
        raise ValueError(f"raw_format must be one of {RAW_FORMATS}, got {raw_format!r}")
    if length_unit not in LENGTH_UNITS or stress_unit not in STRESS_UNITS:
        raise ValueError(f"units must be one of {list(LENGTH_UNITS)} and {list(STRESS_UNITS)}, "
                         f"got {length_unit!r} and {stress_unit!r}")
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
    # With history, frame 0 is the current frame of a sample as well, from the start
    # velocity of the run: so a rollout starts from the undeformed state. (Up to v0.4.0
    # the first sample started at frame 1, which info then lacks.)
    samples_per_trajectory = num_steps - 1
    sources = [sources] if isinstance(sources, str) else list(sources)
    if len(sources) not in (1, 3):
        raise ValueError(f"one source or one per split (train, val, test), got {sources}")
    counts = {"train": n_train, "val": n_val, "test": n_test}
    selection = select_runs(raw_format, raw_dir, sources, counts, select_seed)
    splits, start = {}, 0
    for name, count in counts.items():
        splits[name] = {"start": start, "count": count}
        start += count
        if len(sources) > 1 or select_seed is not None:   # else: the ranges of one source
            splits[name].update(source=selection[name][0], indices=selection[name][1])
    num_trajectories = start
    # Everything that changes the content; --resume requires the same values.
    settings = {
        "contact_radius": repr(contact_radius),
        "time_stride": str(time_stride),
        "history": str(history),
        "prev_stress": "1" if prev_stress else "0",
        "num_steps": str(num_steps),
        "frames": json.dumps(frames),
        "sources": sources[0] if len(sources) == 1 else json.dumps(sources),
        "raw_format": raw_format,
        "splits": json.dumps(splits),
        "edge_stats_stride": str(edge_stats_stride),
    }
    if history:   # only then: exports without history stay as before
        settings["first_frame"] = "0"
    if select_seed is not None:
        settings["select_seed"] = str(select_seed)
    if contact_pairs:
        if any(len(pair) != 2 for pair in contact_pairs):
            raise ValueError(f"contact pairs need two components each, got {contact_pairs}")
        contact_pairs = [[str(a), str(b)] for a, b in contact_pairs]
        settings["contact_pairs"] = json.dumps(contact_pairs)
    # Units only when not SI, so that exports in SI stay as before.
    units = {key: unit for key, unit, si in (("length_unit", length_unit, SI_LENGTH),
                                             ("stress_unit", stress_unit, SI_STRESS))
             if unit != si}
    settings.update(units)
    reader_options = None
    if raw_format == "vtk":   # the options in effect, so that defaults compare equal
        from dataset_preprocessor.vtk_reader import read_raw_run
        vtk_options = {**{name: p.default for name, p in
                          inspect.signature(read_raw_run).parameters.items()
                          if p.default is not p.empty and name not in ("frames",
                                                                       "max_block_bytes")},
                       **(vtk_options or {})}
        for name in ("free_rigid_parts", "components"):   # exports without stay as before
            if not vtk_options.get(name):
                vtk_options.pop(name, None)
        settings["vtk_options"] = json.dumps(vtk_options, sort_keys=True, default=list)
        reader_options = vtk_options
    elif raw_format == "d3plot":   # only the options given, so that exports without stay
        reader_options = {name: value for name, value in (d3plot_options or {}).items()
                          if value is not None and (value or name == "static_fields")}
        if reader_options:
            settings["d3plot_options"] = json.dumps(reader_options, sort_keys=True,
                                                    default=list)
    # Only when given, so that other exports stay as before.
    run_table = None
    if globals_file is not None:
        with open(globals_file, encoding="utf-8") as f:
            run_table = json.load(f)
        settings["globals_file"] = Path(globals_file).name
    if global_features is not None:
        settings["global_features"] = json.dumps(list(global_features))
    if export_config is not None:
        settings["export_config"] = json.dumps(export_config, sort_keys=True)
    thickness_required = "thickness" in ((reader_options or {}).get("static_fields") or ())
    components_shown = False

    fitter = NormalizerFitter(contact_radius, history, prev_stress, edge_stats_stride,
                              first_frame=0)
    db_path = out / "dataset.db"
    done: Dict[int, str] = {}       # trajectories already in the DB -> source
    meshes: Dict[str, int] = {}     # SHA-256 of a stored mesh -> id
    global_names: List[str] = []
    static_names: List[str] = []
    field_names: List[str] = list(DEFAULT_FIELD_NAMES)
    component_names: List[str] = []   # of the whole dataset, in the order they appear
    rigid_bodies: List[str] = []      # components that are free rigid bodies
    wall_columns: Optional[int] = None   # of the walls of the runs so far
    if resume and db_path.exists():
        conn = _resume(db_path, settings)
        done = dict(conn.execute("SELECT id, source FROM trajectories"))
        meshes = dict(conn.execute("SELECT sha256, id FROM meshes"))
        info = dict(conn.execute("SELECT key, value FROM info"))
        global_names = fitter.global_names = json.loads(info.get("global_names", "[]"))
        static_names = fitter.static_names = json.loads(info.get("static_names", "[]"))
        field_names = fitter.field_names = json.loads(
            info.get("field_names", json.dumps(DEFAULT_FIELD_NAMES)))
        component_names = json.loads(info.get("component_names", "[]"))
        rigid_bodies = json.loads(info.get("rigid_bodies", "[]"))
        wall_columns = int(info["wall_columns"]) if "wall_columns" in info else None
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
    runs = [(name, position) for name, positions in selection.values() for position in positions]
    num_read, seen = 0, set()
    try:
        for index, (source, read) in enumerate(raw_runs(raw_format, raw_dir, runs, frames,
                                                        reader_options, block_bytes)):
            num_read += 1
            if source in seen:   # d3plot / vtk: sources without a folder share raw_dir
                raise ValueError(f"{source} would be in two splits — the sources "
                                 f"{sources} overlap")
            seen.add(source)
            if index in done:
                if done[index] != source:
                    raise ValueError(f"trajectory {index} in {db_path} comes from {done[index]}, "
                                     f"the raw data now hold {source} there — the raw data "
                                     "changed; export anew (--no-resume)")
                continue
            start = time.time()
            traj = replace(read())   # a copy: the additions below stay out of the reader's run
            traj.globals = _run_parameters(traj.globals, source, run_table, global_features)
            fields_given = _fields_given(traj)
            _mark_missing_features(traj, static_names, fields_given,
                                   bool(history and prev_stress))
            if traj.component_roles is not None:
                if thickness_required:
                    _check_shell_thickness(traj, source)
                if progress and not components_shown:
                    progress(_describe_components(traj))
                    components_shown = True
            names = (sorted(traj.globals or {}), sorted(traj.node_static or {}),
                     traj.result_field_names())
            if conn is None:
                conn = _create_db(db_path, settings, traj.mesh_pos.shape[0], len(names[2]))
            if index == 0:
                global_names = fitter.global_names = names[0]
                static_names = fitter.static_names = names[1]
                field_names = fitter.field_names = names[2]
            elif names != (global_names, static_names, field_names):
                raise ValueError(f"trajectory {index}: run parameters / static node features / "
                                 f"result fields {names} differ from those of the first "
                                 f"trajectory {(global_names, static_names, field_names)}")
            if index > 0 and (traj.component is not None) != bool(component_names):
                raise ValueError(f"trajectory {index}: components in some runs only — every "
                                 "run of a dataset has them, or none")
            if index == 0:
                rigid_bodies = list(traj.rigid_bodies or [])
            elif list(traj.rigid_bodies or []) != rigid_bodies:
                raise ValueError(f"trajectory {index}: free rigid bodies {traj.rigid_bodies} "
                                 f"differ from those of the first trajectory {rigid_bodies}")
            if traj.walls is not None:
                if wall_columns not in (None, traj.walls.shape[1]):
                    raise ValueError(f"trajectory {index}: walls of {traj.walls.shape[1]} columns "
                                     f"differ from the {wall_columns} of the runs before")
                wall_columns = traj.walls.shape[1]
            split = next(name for name, r in splits.items()
                         if r["start"] <= index < r["start"] + r["count"])
            with conn:   # one transaction per run: the run and the statistics, or nothing
                _insert_trajectory(conn, index, split, source, traj, len(frames), meshes,
                                   global_names, static_names,
                                   fitter if split == "train" else None, component_names,
                                   with_start=bool(history))
                _check_duplicates(conn, index, split, source)
                if not fields_given.all():   # only then: other exports stay as before
                    conn.execute("INSERT OR REPLACE INTO info (key, value) VALUES "
                                 "('field_gaps', '1')")
                if wall_columns not in (None, PLANAR_WALL_COLUMNS):   # others stay as before
                    conn.execute("INSERT OR REPLACE INTO info (key, value) VALUES "
                                 "('wall_columns', ?)", (str(wall_columns),))
                if component_names:
                    conn.execute("INSERT OR REPLACE INTO info (key, value) VALUES "
                                 "('component_names', ?)", (json.dumps(component_names),))
                if index == 0:   # field_names only if not the default, as before
                    conn.executemany("INSERT OR REPLACE INTO info (key, value) VALUES (?, ?)",
                                     [("global_names", json.dumps(global_names)),
                                      ("static_names", json.dumps(static_names))]
                                     + ([("field_names", json.dumps(field_names))]
                                        if field_names != DEFAULT_FIELD_NAMES else [])
                                     + ([("rigid_bodies", json.dumps(rigid_bodies))]
                                        if rigid_bodies else []))
                if split == "train":
                    fitter.save(conn)
            if progress:
                progress(f"  [{index + 1}/{num_trajectories}] {split:5s} {source}: "
                         f"{traj.mesh_pos.shape[0]:,} nodes, {len(frames)} frames, "
                         f"{time.time() - start:.1f} s")
        if num_read < num_trajectories:
            raise ValueError(f"{', '.join(sources)} ({raw_format}) held only {num_read} of "
                             f"the {num_trajectories} trajectories requested")

        unknown = sorted({name for pair in contact_pairs or () for name in pair}
                         - set(component_names))
        if unknown:
            raise ValueError(f"contact pairs name {unknown}, which are no components of the "
                             f"runs (components: {component_names[:20] or 'none'})")
        max_walls = (conn.execute("SELECT max(length(walls)) FROM trajectories").fetchone()[0]
                     or 0) // ((wall_columns or PLANAR_WALL_COLUMNS) * 4)
        finished = [("format", DB_FORMAT)] + ([("max_walls", str(max_walls))] if max_walls else [])
        if global_names:   # only then, so that exports without them stay as before
            finished.append(("global_input", GLOBAL_INPUT_GRAPH))
        with conn:
            conn.executemany("INSERT OR REPLACE INTO info (key, value) VALUES (?, ?)", finished)
            conn.execute("INSERT OR REPLACE INTO info (key, value) VALUES ('fingerprint', ?)",
                         (_fingerprint(conn),))
    finally:
        if conn is not None:
            conn.close()

    normalizer_path = out / "normalizer.json"
    fitter.normalizer(with_walls=max_walls > 0).save(str(normalizer_path))

    names = node_feature_names(history, prev_stress, static_names=static_names,
                               walls=max_walls > 0, field_names=field_names)
    targets = target_feature_names(field_names)
    metadata = {
        # TFRecord: DeepMind's benchmark; other raw data: the name of the export
        "dataset": "deforming_plate" if raw_format == "tfrecord" else out.name,
        "db_format": DB_FORMAT,
        "history": history,
        **({"first_frame": 0} if history else {}),
        "prev_stress": bool(prev_stress and history),
        "node_feature_dims": len(names),
        "edge_feature_dims": MESH_EDGE_FEATURE_DIM,
        "world_edge_feature_dims": WORLD_EDGE_FEATURE_DIM,
        "target_dim": len(targets),
        "node_feature_names": names,
        "edge_feature_names": list(MESH_EDGE_FEATURE_NAMES),
        "world_edge_feature_names": list(WORLD_EDGE_FEATURE_NAMES),
        "target_feature_names": targets,
        "loss_mask": "all nodes except actuator (node_type 1)",
        "contact_radius": contact_radius,
        "num_steps": num_steps,
        "time_stride": time_stride,
        "raw_dir": str(Path(raw_dir).resolve()),
        "sources": sources[0] if len(sources) == 1 else sources,
        "sample_counts": {name: r["count"] * samples_per_trajectory
                          for name, r in splits.items()},
        "splits": splits,
    }
    # Only when they apply: the rollout compares metadata.json with the training
    # data, and TFRecord exports stay identical to those of earlier versions.
    if raw_format != "tfrecord":
        metadata["raw_format"] = raw_format
    metadata.update(units)
    if global_names:
        metadata["global_feature_names"] = global_names
        metadata["global_feature_dims"] = len(global_names)
    if static_names:
        metadata["static_feature_names"] = static_names
    if component_names:
        metadata["component_names"] = component_names
    if contact_pairs:
        metadata["contact_pairs"] = contact_pairs
    if rigid_bodies:
        metadata["rigid_bodies"] = rigid_bodies
    if export_config is not None:
        metadata["export_config"] = export_config
    if max_walls:
        metadata["max_walls"] = max_walls
        metadata["wall_distance_cutoff"] = WALL_DISTANCE_CUTOFF * contact_radius
    if edge_stats_stride != 1:
        metadata["edge_stats_stride"] = edge_stats_stride
    if select_seed is not None:
        metadata["select_seed"] = select_seed
    metadata_path = out / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
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


def build_parser() -> argparse.ArgumentParser:
    """Command line options; the reference in docs/reference/cli.md is generated from them."""
    d = EXPORT_DEFAULTS
    parser = argparse.ArgumentParser(
        description="Export simulation runs (DeepMind TFRecord, LS-DYNA d3plot, VTK time "
                    "series) into a dataset folder: dataset.db, metadata.json, "
                    "normalizer.json. Without arguments the small test set of "
                    "deforming_plate is written.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None,
                        help="TOML file that defines the dataset: raw data, components with "
                             "their roles, features (docs/guides/export.md); options given "
                             "here override it")
    parser.add_argument("--list-parts", action="store_true",
                        help="list the parts of the first run with nodes, elements, node type "
                             "and thickness, and their roles with --components, then stop")
    parser.add_argument("--raw-format", choices=RAW_FORMATS, default=d["raw_format"],
                        help="raw data: DeepMind TFRecord (standard), d3plot of OpenRadioss or "
                             "LS-DYNA runs (docs/guides/openradioss.md, docs/guides/ls-dyna.md) "
                             "or VTK time series")
    parser.add_argument("--raw-dir", default=None,
                        help="directory with meta.json and the .tfrecord or the run folders "
                             f"(default: {d['raw_dir']}; d3plot: {d['raw_dir_d3plot']}; "
                             f"vtk: {d['raw_dir_vtk']})")
    parser.add_argument("--sources", "--source-tfrecord", nargs="+", default=[d["sources"]],
                        metavar="NAME",
                        help="where the runs come from, one name, or three, one per split: "
                             "train val test. TFRecord: the file inside --raw-dir "
                             "(test.tfrecord: 100 trajectories); d3plot / vtk: the folder below "
                             "--raw-dir (<name without .tfrecord> for converted runs), else all "
                             "runs below --raw-dir")
    parser.add_argument("--contact-radius", type=float, default=None,
                        help="world edge radius, in the length unit of the raw data; "
                             "default: collision_radius of meta.json "
                             "(required for d3plot / vtk raw data without meta.json)")
    parser.add_argument("--contact-pairs", nargs="*", default=[], metavar="A:B",
                        help="contact rules of raw data with components (LS-DYNA and VTK "
                             "parts): world edges only between these pairs, e.g. "
                             "plate:impactor plate:plate (self-contact), also with patterns "
                             "(box_*:beam_*); none given: every pair")
    parser.add_argument("--deck-walls", dest="use_deck_walls", default=True,
                        action=argparse.BooleanOptionalAction,
                        help="d3plot: use the rigid walls of the decks (*RIGIDWALL_PLANAR, "
                             "/RWALL/CYL) as node features")
    parser.add_argument("--length-unit", choices=list(LENGTH_UNITS), default=SI_LENGTH,
                        help="unit of the lengths in the raw data (positions, contact radius, "
                             "e.g. mm for LS-DYNA models); the values are kept, the reports "
                             "convert to SI")
    parser.add_argument("--stress-unit", choices=list(STRESS_UNITS), default=SI_STRESS,
                        help="unit of the stress in the raw data (e.g. MPa for LS-DYNA models "
                             "in mm, t, s); the values are kept, the reports convert to SI")
    fields = parser.add_argument_group("LS-DYNA and VTK raw data: components, result fields, "
                                       "static node features, run parameters and free rigid "
                                       "parts")
    fields.add_argument("--components", nargs="*", default=None, metavar="PATTERN=ROLE",
                        help="what goes into the dataset: name patterns of the parts with their "
                             f"role ({', '.join(ROLES)}); every other part is left out; "
                             "instead of --free-rigid-parts, --exclude-parts, --actuator-parts "
                             "and --clamped-parts; default: every part, roles from the raw data")
    fields.add_argument("--global-features", nargs="*", default=None, metavar="NAME",
                        help="run parameters in the dataset, one input per graph; default: all "
                             "that the runs have")
    fields.add_argument("--globals-file", default=None,
                        help="JSON file with the run parameters of all runs, by the name of the "
                             "run folder; default: a globals.json in every run folder")
    fields.add_argument("--fields", nargs="*", default=[],
                        help="further result fields per state as targets, each under its own "
                             "name; vtk: point or cell fields, e.g. plastic_strain (vectors: "
                             "one per component); d3plot: plastic_strain, thickness (shells)")
    fields.add_argument("--static-fields", nargs="*", default=None,
                        help="static node features; vtk: point or cell fields of the first "
                             "state (vectors: one per component); d3plot: thickness, "
                             "initial_velocity and the material properties of the deck "
                             "density, youngs_modulus, poisson_ratio, yield_stress (default: "
                             "thickness and initial_velocity where the runs have them)")
    fields.add_argument("--free-rigid-parts", nargs="*", default=[], metavar="PART",
                        help="rigid parts whose motion results from the crash (e.g. a barrier "
                             "with mass): the model predicts it and the rollout moves each "
                             "as a rigid body; default: rigid parts are actuators, their "
                             "motion given. d3plot: *MAT_RIGID and /RBODY parts without "
                             "prescribed motion, by title; vtk: any parts")
    fields.add_argument("--exclude-parts", nargs="*", default=[], metavar="PART",
                        help="d3plot only: parts left out of the export, named as the "
                             "components, e.g. display models written as shells: their "
                             "elements and the nodes only they hold; nodal fields are "
                             "averaged without them; default: every part stays")
    vtk = parser.add_argument_group("VTK raw data (--raw-format vtk): field names and parts")
    vtk.add_argument("--stress-field", default="stress",
                     help="point or cell field of the target stress: scalar, or stress tensor "
                          "(6 / 9 components, gives the von Mises stress); '': no stress")
    vtk.add_argument("--node-type-field", default="node_type",
                     help="point field with the node types 0 / 1 / 3; absent: all normal")
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
    parser.add_argument("--select-seed", type=int, default=d["select_seed"],
                        help="draw the trajectories of each split at random with this seed "
                             "(stored in metadata.json); default: the first ones")
    parser.add_argument("--num-steps", type=int, default=d["num_steps"],
                        help="frames per trajectory (samples per trajectory = "
                             "num_steps - 1)")
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
    return parser


CELL_NAMES = {CELL_LINE: "lines", CELL_TRIANGLE: "triangles", CELL_QUAD: "quads",
              CELL_TETRA: "tetrahedra", CELL_PYRAMID: "pyramids", CELL_WEDGE: "wedges",
              CELL_HEXAHEDRON: "hexahedra"}


def _reader_options(args: argparse.Namespace, components: Dict[str, str]) -> Tuple[Dict, Dict]:
    """The options of the VTK and the d3plot reader."""
    return (dict(stress_field=args.stress_field, node_type_field=args.node_type_field,
                 static_fields=args.static_fields or [], fields=args.fields,
                 displacement_field=args.displacement_field,
                 node_id_field=args.node_id_field, part_field=args.part_field,
                 actuator_parts=args.actuator_parts, clamped_parts=args.clamped_parts,
                 free_rigid_parts=args.free_rigid_parts, components=components),
            dict(fields=args.fields, static_fields=args.static_fields,
                 free_rigid_parts=args.free_rigid_parts, exclude_parts=args.exclude_parts,
                 components=components, no_deck_walls=not args.use_deck_walls))


def _first_run(args: argparse.Namespace, raw_dir: Path, components: Dict[str, str]
               ) -> Tuple[Path, RawTrajectory]:
    """The first run of the first source, read with its first state only."""
    if args.raw_format == "d3plot":
        from dataset_preprocessor.d3plot_reader import read_raw_run, run_dirs
    else:
        from dataset_preprocessor.vtk_reader import read_raw_run, run_dirs
    vtk_options, d3plot_options = _reader_options(args, components)
    run = run_dirs(raw_dir, args.sources[0])[0]
    return run, read_raw_run(run, [0], **(d3plot_options if args.raw_format == "d3plot"
                                          else vtk_options))


def _list_parts(args: argparse.Namespace, raw_dir: Path, components: Dict[str, str]) -> None:
    """Print what ``--components`` chooses from: the parts of the first run. A cell counts
    for the last part among its nodes, as a node shared by parts goes to the first."""
    run, traj = _first_run(args, raw_dir, {})
    names = traj.component_names or []
    parts = [name for name in names if name != "(no element)"]
    roles = component_roles(parts, components, str(run)) if components else {}
    thickness = (traj.node_static or {}).get("thickness")
    cell_types = (traj.cell_types if traj.cell_types is not None
                  else np.full(len(traj.cells), CELL_TETRA))
    part_of_cell = np.where(traj.cells >= 0, traj.component[traj.cells], -1).max(axis=1)
    print(f"Parts of {run} ({traj.mesh_pos.shape[0]:,} nodes):")
    for k, name in enumerate(names):
        nodes = traj.component == k
        kinds, counts = np.unique(cell_types[part_of_cell == k], return_counts=True)
        elements = ", ".join(f"{count} {CELL_NAMES[kind]}"
                             for kind, count in zip(kinds.tolist(), counts.tolist())) or "none"
        types = sorted(set(traj.node_type[nodes].tolist()))
        columns = [f"{name:32s}", f"{int(nodes.sum()):>9,} nodes", f"{elements:24s}",
                   f"node type {types}"]
        if thickness is not None:
            given = thickness[nodes][~np.isnan(thickness[nodes])]
            columns.append(f"thickness {given.min():g} to {given.max():g}" if len(given)
                           else "no thickness")
        if components:
            columns.append(f"role: {roles.get(name, 'left out')}")
        print("  " + "  ".join(columns))


def main(argv=None) -> None:
    """Export a dataset; defaults come from ``--config`` and ``EXPORT_DEFAULTS``."""
    d = EXPORT_DEFAULTS
    parser = build_parser()
    config_option = argparse.ArgumentParser(add_help=False)
    config_option.add_argument("--config")
    config_path = config_option.parse_known_args(argv)[0].config
    export_config = None
    if config_path:
        path = Path(config_path) if Path(config_path).exists() else _repo_path(config_path)
        try:
            export_config, defaults = read_export_config(path)
        except (OSError, ValueError) as error:
            parser.error(f"--config: {error}")
        for action in parser._actions:   # argparse checks only what is on the command line
            if action.choices and defaults.get(action.dest, action.default) not in action.choices:
                parser.error(f"--config: {action.dest} must be one of {list(action.choices)}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)

    if args.exclude_parts and args.raw_format != "d3plot":
        parser.error("--exclude-parts needs --raw-format d3plot")
    try:
        components = parse_components(args.components)
    except ValueError as error:
        parser.error(str(error))
    if components:
        if args.raw_format == "tfrecord":
            parser.error("components need --raw-format d3plot or vtk")
        both = [option for option, value in (("--free-rigid-parts", args.free_rigid_parts),
                                             ("--exclude-parts", args.exclude_parts),
                                             ("--actuator-parts", args.actuator_parts),
                                             ("--clamped-parts", args.clamped_parts)) if value]
        if both:
            parser.error(f"--components sets the parts and their roles; {', '.join(both)} "
                         "would contradict it")
    if args.raw_dir is None:
        args.raw_dir = d[{"d3plot": "raw_dir_d3plot", "vtk": "raw_dir_vtk"}.get(
            args.raw_format, "raw_dir")]
    raw_dir, out_dir = _repo_path(args.raw_dir), _repo_path(args.out_dir)
    if len(args.sources) not in (1, 3):
        parser.error("--sources takes one name, or three: train, val and test")
    if args.list_parts:
        if args.raw_format == "tfrecord":
            parser.error("--list-parts needs --raw-format d3plot or vtk")
        try:
            _list_parts(args, raw_dir, components)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return
    contact_pairs = [pair.split(":") for pair in args.contact_pairs]
    if any(set("*?[") & set(name) for pair in contact_pairs for name in pair):
        try:
            contact_pairs = expand_contact_pairs(
                contact_pairs, _first_run(args, raw_dir, components)[1].component_names or [])
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if (raw_dir / "meta.json").exists():
        trajectory_length = read_raw_meta(raw_dir)["trajectory_length"]
    elif args.raw_format != "tfrecord":   # runs without meta.json: shortest selected run
        if args.raw_format == "d3plot":
            from dataset_preprocessor.d3plot_reader import num_states, run_dirs
        else:
            from dataset_preprocessor.vtk_reader import num_states, run_dirs
        try:
            selection = select_runs(args.raw_format, raw_dir, args.sources,
                                    {"train": args.n_train, "val": args.n_val,
                                     "test": args.n_test}, args.select_seed)
        except ValueError as error:
            parser.error(str(error))
        lengths = []
        for name, positions in selection.values():
            runs = run_dirs(raw_dir, name)
            lengths += [num_states(runs[p]) for p in positions]
        trajectory_length = min(lengths)
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

    samples_per_trajectory = args.num_steps - 1
    drawn = "" if args.select_seed is None else f", drawn with seed {args.select_seed}"
    print(f"Exporting {args.n_train}/{args.n_val}/{args.n_test} trajectories x "
          f"{samples_per_trajectory} samples (frames 0..{last_frame}, stride "
          f"{args.time_stride}, history {args.history}, raw {args.raw_format}{drawn}) "
          f"-> {out_dir}")
    start = time.time()
    vtk_options, d3plot_options = _reader_options(args, components)
    try:
        paths = write_dataset(
            raw_dir=str(raw_dir), out_dir=str(out_dir), sources=args.sources,
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
            select_seed=args.select_seed, num_steps=args.num_steps, time_stride=args.time_stride,
            history=args.history, prev_stress=args.prev_stress, raw_format=args.raw_format,
            contact_radius=args.contact_radius,
            vtk_options=vtk_options, d3plot_options=d3plot_options,
            length_unit=args.length_unit, stress_unit=args.stress_unit,
            resume=args.resume, edge_stats_stride=args.edge_stats_stride,
            block_mb=args.block_mb, progress=print,
            contact_pairs=contact_pairs, global_features=args.global_features,
            globals_file=str(_repo_path(args.globals_file)) if args.globals_file else None,
            export_config=export_config,
        )
    except ValueError as error:
        sys.exit(f"export.py: error: {error}")
    counts = "/".join(str(n * samples_per_trajectory)
                      for n in (args.n_train, args.n_val, args.n_test))
    print(f"Done in {time.time() - start:.0f} s: {counts} samples, "
          f"dataset.db {db_path.stat().st_size / 1e6:.1f} MB")
    with open(paths["metadata"], encoding="utf-8") as f:
        metadata = json.load(f)
    names = metadata["node_feature_names"]
    print(f"  node features ({len(names)}): {', '.join(names)}")
    if metadata.get("global_feature_names"):
        print(f"  run parameters ({metadata['global_feature_dims']}, per graph): "
              f"{', '.join(metadata['global_feature_names'])}")
    for name, path in paths.items():
        print(f"  {name:10s} {path}")


if __name__ == "__main__":
    main()
