#!/usr/bin/env python3
"""
explore_data.py — Exploration of an exported dataset, split against split.

Describes every trajectory of ``data/<name>/`` (``dataset.db``, ``metadata.json``,
``normalizer.json``) by its mesh, its loading and its response, checks the data and
compares the splits: how far val and test lie from what the training sees. With
``--rollout`` it sets the error of every rolled-out trajectory against these
descriptors. Writes a new folder ``evaluations/<YYYYMMDD_HHMMSS>_<name>_data_exploration/``:

- ``01_descriptors.png``    every descriptor per split: box from the lower to the upper
                            quartile with the median, whiskers from the smallest to the
                            largest trajectory
- ``02_over_time.png``      displacement, stress, contact and actuator over the frames:
                            median and 10–90 % band of the trajectories of each split
- ``03_targets.png``        the targets as the model sees them, normalized with the train
                            statistics of ``normalizer.json``, per split
- ``04_coverage.png``       distance of every trajectory to its nearest train trajectory;
                            Kolmogorov–Smirnov distance train–val and train–test per
                            descriptor
- ``05_error_vs_data.png``  with ``--rollout``: the error of every trajectory against the
                            descriptors with the strongest rank correlation
- ``exploration.md``        dataset, findings, data checks, tables, figures
- ``trajectories.csv``      every trajectory with its descriptors (and errors)
- ``exploration.json``      the values of the report

Definitions. Structure = all nodes but the actuator, the nodes of the loss.
Displacement = position minus that of frame 0. Contact pairs = node pairs of
different bodies (connected parts of the mesh) closer than the contact radius r_W,
i.e. the world edges between bodies. Self-contact pairs = pairs of one body closer
than r_W that ``self_contact_aware`` keeps (farther apart through the material than
its exclusion distance); they can exist at rest, e.g. across a narrow gap of the
geometry. Element volume: tetrahedra only, relative to the rest state
``mesh_pos``; at or below 0 the element is inverted. Distance to train: the
descriptors standardized with mean and std of the train trajectories, then the RMS
difference to the nearest train trajectory (for a train trajectory: the nearest other
one). Kolmogorov–Smirnov: two-sample distance D with its asymptotic p-value. Rank
correlation: Spearman's ρ; for n trajectories, |ρ| above 1.96 / √(n − 1) is unlikely
by chance (p < 0.05).

The findings are measurements, not verdicts. The DB is read one trajectory and one
frame at a time, so memory does not grow with the dataset; the contact search runs on
every frame with ``graph_features`` (a cell list for large meshes).

Usage
-----
    python utils/explore_data.py data/<name> [--rollout rollouts/<rollout> ...]
    python utils/explore_data.py data/<name> --output-dir <new folder>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset_preprocessor.dataset import (  # noqa: E402
    DB_FORMAT,
    RENAMED_DB_FORMAT,
    _tensor,
    connect_readonly,
    dataset_fingerprint,
)
from dataset_preprocessor.graph_features import (  # noqa: E402
    DEFAULT_FIELD_NAMES,
    DEFAULT_WORLD_EDGE_EXCLUDE_DISTANCE,
    NODE_TYPE_ACTUATOR,
    NODE_TYPE_CLAMPED,
    mesh_components,
    target_feature_names,
    trajectory_features,
    world_edge_selection,
    world_edges,
)
from utils.compare_paper import (  # noqa: E402
    SERIES_COLORS,
    _figure_header,
    _significant,
    _style,
    load_rollout,
    markdown_table,
    tint,
)
from utils.evaluate_run import FAR_ABOVE_MEDIAN, TO_DISPLACEMENT, _legend, to_displacement  # noqa: E402
from utils.normalizer import Normalizer  # noqa: E402
from utils.plot_losses import SURFACE, TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY  # noqa: E402
from utils.timezone import now  # noqa: E402
from utils.units import SI_STRESS, si_factors, stress_display  # noqa: E402

EVALUATIONS_DIR = _REPO_ROOT / "evaluations"
SPLITS = ("train", "val", "test")
SPLIT_COLORS = dict(zip(SPLITS, SERIES_COLORS))

# Descriptors of a trajectory: key, title, factor and unit of the report (values in SI).
DESCRIPTORS = [
    # mesh and geometry of the rest state
    ("nodes", "Nodes", 1, ""),
    ("edge_length", "Mesh edge length, median", 1e3, "mm"),
    ("extent_largest", "Structure, largest extent", 1e3, "mm"),
    ("extent_smallest", "Structure, smallest extent", 1e3, "mm"),
    ("actuator_extent", "Actuator, largest extent", 1e3, "mm"),
    # loading
    ("actuator_travel", "Actuator travel, max", 1e3, "mm"),
    ("actuator_speed", "Actuator speed, max", 1e3, "mm/frame"),
    ("first_contact", "First contact", 1, "frame"),
    ("contact_share", "Frames with contact", 100, "%"),
    ("contact_pairs", "Contact pairs, max", 1, ""),
    ("self_contact_pairs", "Self-contact pairs, max", 1, ""),
    # response
    ("displacement_rms", "Displacement RMS, max", 1e3, "mm"),
    ("displacement_max", "Displacement of a node, max", 1e3, "mm"),
    ("displacement_end", "Displacement RMS at the end", 100, "% of max"),
    ("velocity_max", "Node velocity, max", 1e3, "mm/frame"),
    ("stress_max", "Stress, max", 1e-3, "kPa"),
    ("stress_rms", "Stress RMS", 1e-3, "kPa"),
    ("volume_min", "Element volume, min", 100, "% of rest"),
]
TITLES = {key: title for key, title, _, _ in DESCRIPTORS}
# Descriptors that are lengths or stresses, by their unit in the report.
LENGTHS = {key for key, _, _, unit in DESCRIPTORS if unit in ("mm", "mm/frame")}
STRESSES = {key for key, _, _, unit in DESCRIPTORS if unit == "kPa"}
DISTANCE = ("distance_to_train", "Distance to train", 1, "σ")
# Curves over the frames (02_over_time.png): key, title, factor, unit.
OVER_TIME = [("displacement_rms", "Displacement RMS of the structure", 1e3, "mm"),
             ("stress_max", "Largest stress", 1e-3, "kPa"),
             ("contact_pairs", "Contact pairs", 1, ""),
             ("actuator_travel", "Actuator travel", 1e3, "mm")]
# Targets in units of the train std: histogram range (values beyond in the outer bins)
# and the bound of a far value.
HISTOGRAM_LIMIT, HISTOGRAM_BINS = 10.0, 160
FAR_SIGMA = 5.0
# A trajectory farther from train than this share of the train trajectories are from
# each other lies outside the region the training covers densely.
COVERAGE_QUANTILE = 0.9
# A cell whose rest volume is below this share of the median cell is degenerate.
DEGENERATE_VOLUME = 1e-6
# Errors of a rollout per trajectory: key, title, factor, unit.
ERRORS = [("q_rmse_all", "Position RMSE-all", 1e3, "mm"),
          ("stress_rmse_all", "Stress RMSE-all", 1e-3, "kPa"),
          (TO_DISPLACEMENT, "Position error / displacement", 100, "%")]
# Descriptors per error in 05_error_vs_data.png: the strongest rank correlations.
SCATTER_COLUMNS = 3
# A descriptor whose largest value exceeds the smallest this many times: log scale.
WIDE_RANGE = 100.0


def text(value: Optional[float], factor: float = 1.0, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "–"
    if 0 < abs(value * factor) < 1e-3:
        return f"{value * factor:.{digits - 1}e}"
    return _significant(value * factor, digits)


def shown(factor: float, unit: str, su: str = SI_STRESS) -> Tuple[float, str]:
    """Factor (from SI) and unit of a table entry for data in the stress unit ``su``:
    the stress as ``utils.units.stress_display``, every other entry as in the table."""
    return stress_display(su) if unit == "kPa" else (factor, unit)


def with_unit(title: str, unit: str) -> str:
    return f"{title} [{unit}]" if unit else title


# ── one trajectory ───────────────────────────────────────────────────────────

@dataclass
class Trajectory:
    sequence_id: str
    split: str
    source: str
    mesh_id: int
    sha256: str
    values: Dict[str, Optional[float]]     # DESCRIPTORS, SI
    series: Dict[str, np.ndarray]          # OVER_TIME, one value per frame
    checks: Dict                           # data checks; largest normalized target


class TargetStats:
    """Normalized targets of one split, nodes of the structure: histogram per target,
    moments, extremes and the count beyond ``FAR_SIGMA``."""

    EDGES = np.linspace(-HISTOGRAM_LIMIT, HISTOGRAM_LIMIT, HISTOGRAM_BINS + 1)

    def __init__(self, names: List[str]):
        self.names = list(names)   # of the targets: velocity, then the result fields
        dims = len(self.names)
        self.counts = np.zeros((dims, HISTOGRAM_BINS), np.int64)
        self.n = np.zeros(dims, np.int64)   # values per target: a node may have none (NaN)
        self.sum, self.sum_sq = np.zeros(dims), np.zeros(dims)
        self.low, self.high = np.full(dims, np.inf), np.full(dims, -np.inf)
        self.far = np.zeros(dims, np.int64)

    def add(self, z: torch.Tensor) -> None:
        z = z.double().numpy()
        if not len(z):
            return
        given = ~np.isnan(z)
        self.n += given.sum(0)
        self.sum += np.nansum(z, 0)
        self.sum_sq += np.nansum(z * z, 0)
        self.low = np.fmin(self.low, np.nan_to_num(z, nan=np.inf).min(0))
        self.high = np.fmax(self.high, np.nan_to_num(z, nan=-np.inf).max(0))
        self.far += (np.abs(np.nan_to_num(z, nan=0.0)) > FAR_SIGMA).sum(0)
        for k in range(z.shape[1]):
            self.counts[k] += np.histogram(np.clip(z[given[:, k], k], -HISTOGRAM_LIMIT,
                                                   HISTOGRAM_LIMIT), self.EDGES)[0]

    def summary(self) -> Dict[str, List[float]]:
        mean = self.sum / self.n
        return {"values": int(self.n.max()), "mean": mean.tolist(),
                "std": np.sqrt(np.maximum(self.sum_sq / self.n - mean ** 2, 0)).tolist(),
                "min": self.low.tolist(), "max": self.high.tolist(),
                "far_share": (self.far / self.n).tolist()}


def tetra_volumes(pos: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Signed volume of every tetrahedron (C,), float64."""
    a, b, c, d = pos.double()[cells].unbind(1)
    return (torch.linalg.cross(b - a, c - a) * (d - a)).sum(-1) / 6


def scan(conn, row, radius: float, normalizer: Normalizer,
         targets: Dict[str, TargetStats], field_names: List[str],
         si: Tuple[float, float] = (1.0, 1.0)) -> Trajectory:
    """Descriptors, curves and checks of one trajectory, read frame by frame; its
    normalized targets go into ``targets[split]``. The stress descriptors stay empty
    for result fields without a ``stress``. ``si``: the factors from the units of the
    data to m and Pa (``utils.units``), applied to the descriptors at the end."""
    db_id, split, sequence_id, source, mesh_id, n, sha256 = row
    node_type, mesh_pos = conn.execute(
        "SELECT node_type, mesh_pos FROM trajectories WHERE id = ?", (db_id,)).fetchone()
    node_type = np.frombuffer(node_type, np.uint8)
    actuator = torch.from_numpy(node_type == NODE_TYPE_ACTUATOR)
    clamped = torch.from_numpy(node_type == NODE_TYPE_CLAMPED)
    structure = ~actuator
    mesh_pos = _tensor(mesh_pos, np.float32, (n, 3))
    num_cells, cells, cell_types, edges = conn.execute(
        "SELECT num_cells, cells, cell_types, mesh_edges FROM meshes WHERE id = ?",
        (mesh_id,)).fetchone()
    edge_index = torch.from_numpy(np.frombuffer(edges, np.int32).reshape(2, -1).astype(np.int64))

    body = mesh_components(edge_index, n)
    exclude = world_edge_selection(
        "self_contact_aware", edge_index, n, mesh_pos=mesh_pos,
        exclude_length=DEFAULT_WORLD_EDGE_EXCLUDE_DISTANCE * radius)["world_exclude_index"]
    inner = structure[edge_index[0]] & structure[edge_index[1]]
    lengths = (mesh_pos[edge_index[0, inner]] - mesh_pos[edge_index[1, inner]]).norm(dim=-1)
    extent = (mesh_pos[structure].amax(0) - mesh_pos[structure].amin(0)).sort().values
    has_actuator = bool(actuator.any())
    tetra, rest, degenerate = None, None, 0
    if cell_types is None:   # tetrahedra
        tetra = torch.from_numpy(np.frombuffer(cells, np.int32).reshape(num_cells, 4)
                                 .astype(np.int64))
        rest = tetra_volumes(mesh_pos, tetra)
        degenerate = rest.abs() < DEGENERATE_VOLUME * rest.abs().median()
        tetra, rest, degenerate = tetra[~degenerate], rest[~degenerate], int(degenerate.sum())

    series = {key: [] for key, *_ in OVER_TIME}
    first = previous = previous_fields = start = previous_centre = None
    largest = {"displacement_max": 0.0, "velocity_max": 0.0, "actuator_speed": 0.0,
               "self_contact_pairs": 0, "clamped_motion": 0.0}
    names = targets[split].names
    stress_column = field_names.index("stress") if "stress" in field_names else None
    target_sigma = torch.zeros(len(names), dtype=torch.float64)
    volume_min, first_contact, contact_frames, inverted_frames = math.inf, None, 0, 0
    stress_sq, non_finite, negative, frames = 0.0, 0, 0, 0
    for frame, (pos, fields) in enumerate(conn.execute(
            "SELECT world_pos, stress FROM frames WHERE trajectory_id = ? ORDER BY frame",
            (db_id,))):
        pos = _tensor(pos, np.float32, (n, 3))
        fields = _tensor(fields, np.float32, (n, len(field_names)))
        frames += 1
        if first is None:   # a node without a value in every frame (NaN) is no data error
            first = pos
            start = pos[actuator].mean(0) if has_actuator else None
            no_value = torch.isnan(fields)
            stress_nodes = (structure if stress_column is None
                            else structure & ~no_value[:, stress_column])
        non_finite += int((~torch.isfinite(pos)).sum() + (~torch.isfinite(fields) & ~no_value).sum())
        stress = None if stress_column is None else fields[:, stress_column]
        negative += 0 if stress is None else int((stress < 0).sum())
        moved = (pos - first).norm(dim=-1)
        series["displacement_rms"].append(float(moved[structure].pow(2).mean().sqrt()))
        largest["displacement_max"] = max(largest["displacement_max"],
                                          float(moved[structure].max()))
        if clamped.any():
            largest["clamped_motion"] = max(largest["clamped_motion"], float(moved[clamped].max()))
        if stress is not None:
            series["stress_max"].append(float(stress[stress_nodes].max()))
            stress_sq += float(stress[stress_nodes].double().pow(2).sum())
        centre = pos[actuator].mean(0) if has_actuator else None
        series["actuator_travel"].append(float((centre - start).norm()) if has_actuator else 0.0)

        pairs = world_edges(pos, edge_index, radius, world_exclude_index=exclude)
        between = body[pairs[0]] != body[pairs[1]]
        contact = int(between.sum()) // 2   # ordered pairs: both directions
        series["contact_pairs"].append(contact)
        largest["self_contact_pairs"] = max(largest["self_contact_pairs"],
                                            int((~between).sum()) // 2)
        if contact:
            contact_frames += 1
            first_contact = frame if first_contact is None else first_contact
        if tetra is not None:
            ratio = tetra_volumes(pos, tetra) / rest
            volume_min = min(volume_min, float(ratio.min()))
            inverted_frames += int(bool((ratio <= 0).any()))
        if previous is not None:
            velocity = (pos - previous).norm(dim=-1)
            largest["velocity_max"] = max(largest["velocity_max"], float(velocity[structure].max()))
            if has_actuator:
                largest["actuator_speed"] = max(largest["actuator_speed"],
                                                float((centre - previous_centre).norm()))
            _, target, _ = trajectory_features(torch.stack((previous, pos)),
                                               torch.stack((previous_fields, fields)), actuator)
            z = normalizer.normalize(target[0][structure], "target")
            targets[split].add(z)
            target_sigma = torch.maximum(target_sigma,
                                         torch.nan_to_num(z.double(), nan=0.0).abs().amax(0))
        previous, previous_fields, previous_centre = pos, fields, centre

    curves = {key: np.asarray(values, dtype=float) for key, values in series.items()}
    displacement = curves["displacement_rms"]
    values = {
        "nodes": float(n),
        "edge_length": float(lengths.median()) if len(lengths) else None,
        "extent_largest": float(extent[-1]),
        "extent_smallest": float(extent[0]),
        "actuator_extent": (float((mesh_pos[actuator].amax(0) - mesh_pos[actuator].amin(0)).max())
                            if has_actuator else None),
        "actuator_travel": float(curves["actuator_travel"].max()) if has_actuator else None,
        "actuator_speed": largest["actuator_speed"] if has_actuator else None,
        "first_contact": None if first_contact is None else float(first_contact),
        "contact_share": contact_frames / frames,
        "contact_pairs": float(curves["contact_pairs"].max()),
        "self_contact_pairs": float(largest["self_contact_pairs"]),
        "displacement_rms": float(displacement.max()),
        "displacement_max": largest["displacement_max"],
        "displacement_end": (float(displacement[-1] / displacement.max())
                             if displacement.max() > 0 else None),
        "velocity_max": largest["velocity_max"],
        "stress_max": float(curves["stress_max"].max()) if stress_column is not None else None,
        "stress_rms": (math.sqrt(stress_sq / (frames * int(stress_nodes.sum())))
                       if stress_column is not None else None),
        "volume_min": volume_min if tetra is not None else None,
    }
    for key in values.keys() & (LENGTHS | STRESSES):
        factor = si[0] if key in LENGTHS else si[1]
        values[key] = None if values[key] is None else values[key] * factor
        if key in curves:
            curves[key] = curves[key] * factor
    largest_target = int(target_sigma.argmax())
    checks = {"frames": frames, "non_finite": non_finite, "negative_stress": negative,
              "clamped_motion": largest["clamped_motion"] * si[0],
              "degenerate_cells": degenerate,
              "inverted_frames": inverted_frames,
              "target_sigma": float(target_sigma[largest_target]),
              "target_name": names[largest_target]}
    return Trajectory(sequence_id, split, source, mesh_id, sha256, values, curves, checks)


def scan_dataset(data_dir: Path) -> Tuple[Dict, List[Trajectory], Dict[str, TargetStats]]:
    """``info`` of the DB, every trajectory and the target statistics per split."""
    db_path = data_dir / "dataset.db"
    for name in ("dataset.db", "metadata.json", "normalizer.json"):
        if not (data_dir / name).exists():
            raise SystemExit(f"{data_dir}: no {name} — an export folder of "
                             "python -m dataset_preprocessor.export is needed")
    normalizer = Normalizer.load(str(data_dir / "normalizer.json"))
    conn = connect_readonly(db_path)
    try:
        info = dict(conn.execute("SELECT key, value FROM info"))
        if info.get("format") not in (DB_FORMAT, RENAMED_DB_FORMAT):
            raise SystemExit(f"{db_path}: format {info.get('format')!r}, this script reads "
                             f"{DB_FORMAT!r} — re-export it: python -m dataset_preprocessor.export")
        radius = float(info["contact_radius"])
        field_names = json.loads(info.get("field_names", json.dumps(DEFAULT_FIELD_NAMES)))
        rows = conn.execute("SELECT id, split, sequence_id, source, mesh_id, num_nodes, sha256 "
                            "FROM trajectories ORDER BY id").fetchall()
        targets = {split: TargetStats(target_feature_names(field_names))
                   for split in dict.fromkeys(row[1] for row in rows)}
        trajectories = []
        for split in targets:
            started = time.perf_counter()
            split_rows = [row for row in rows if row[1] == split]
            for k, row in enumerate(split_rows, start=1):
                trajectories.append(scan(conn, row, radius, normalizer, targets, field_names,
                                         si_factors(info)))
                if sys.stdout.isatty():
                    print(f"\r{split}: {k}/{len(split_rows)} trajectories", end="", flush=True)
            print(f"\r{split}: {len(split_rows)} trajectories in "
                  f"{time.perf_counter() - started:.0f} s")
    finally:
        conn.close()
    return info, trajectories, targets


# ── statistics ───────────────────────────────────────────────────────────────

def column(trajectories: List[Trajectory], key: str, split: Optional[str] = None) -> np.ndarray:
    """Finite values of ``key`` (descriptor or distance) over the trajectories of ``split``."""
    values = [t.values.get(key) for t in trajectories if split in (None, t.split)]
    return np.array([v for v in values if v is not None and math.isfinite(v)], dtype=float)


def ks_test(a: np.ndarray, b: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Two-sample Kolmogorov–Smirnov distance D and its asymptotic p-value
    (Numerical Recipes, ``probks``); None without values."""
    if not len(a) or not len(b):
        return None, None
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate((a, b))
    d = float(np.abs(np.searchsorted(a, grid, "right") / len(a)
                     - np.searchsorted(b, grid, "right") / len(b)).max())
    root = math.sqrt(len(a) * len(b) / (len(a) + len(b)))
    lam = (root + 0.12 + 0.11 / root) * d
    if lam < 0.2:
        return d, 1.0
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * (k * lam) ** 2) for k in range(1, 101))
    return d, min(max(p, 0.0), 1.0)


def ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1 .. n; equal values share their mean rank."""
    rank = np.empty(len(x))
    rank[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.bincount(inverse, weights=rank) / counts)[inverse]


def spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Spearman's rank correlation; None below 3 pairs or for a constant series."""
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def chance_bound(n: int) -> float:
    """|ρ| above which a rank correlation of n pairs is unlikely by chance (p < 0.05)."""
    return 1.96 / math.sqrt(n - 1) if n > 1 else math.inf


def add_distances(trajectories: List[Trajectory]) -> List[str]:
    """Distance of every trajectory to its nearest train trajectory, stored as
    ``values[DISTANCE]``; returns the descriptors used (finite and not constant in train)."""
    train = [t for t in trajectories if t.split == "train"]
    keys = [key for key, *_ in DESCRIPTORS
            if len(column(train, key)) == len(train) > 1 and column(train, key).std() > 0]
    if not keys:
        return keys

    def matrix(group: List[Trajectory]) -> np.ndarray:
        return np.array([[np.nan if t.values.get(k) is None else t.values[k] for k in keys]
                         for t in group], dtype=float)

    reference = matrix(train)
    mean, std = reference.mean(0), reference.std(0)
    reference = (reference - mean) / std
    for t, row in zip(trajectories, (matrix(trajectories) - mean) / std):
        distance = np.sqrt(np.nanmean((reference - row) ** 2, axis=1))
        if t.split == "train":
            distance[train.index(t)] = np.inf
        t.values[DISTANCE[0]] = float(np.nanmin(distance))
    return keys


def outside_train(trajectories: List[Trajectory]) -> Dict[str, List[Tuple[str, float, float, float]]]:
    """Per val/test trajectory the descriptors outside the range of the train
    trajectories: (key, value, train min, train max)."""
    bounds = {}
    for key, *_ in DESCRIPTORS:
        values = column(trajectories, key, "train")
        if len(values):
            bounds[key] = (values.min(), values.max())
    out = {}
    for t in trajectories:
        if t.split == "train":
            continue
        out[t.sequence_id] = [(key, t.values[key], low, high) for key, (low, high) in bounds.items()
                              if t.values.get(key) is not None
                              and not low <= t.values[key] <= high]
    return out


# ── rollouts ─────────────────────────────────────────────────────────────────

@dataclass
class RolloutErrors:
    label: str
    folder: Path
    split: str
    errors: Dict[str, Dict[str, Optional[float]]]   # trajectory -> ERRORS key -> value


def load_errors(folders: List[Path], db_path: Path) -> List[RolloutErrors]:
    """The errors per trajectory of every finished rollout made on this dataset."""
    ours = dataset_fingerprint(db_path)
    out = []
    for folder in folders:
        rollout = load_rollout(folder)
        if rollout is None or rollout.status != "finished":
            raise SystemExit(f"{folder}: no finished rollout")
        theirs = (rollout.info["data"].get("dataset_db") or {}).get("fingerprint")
        if ours and theirs and ours != theirs:
            raise SystemExit(f"{folder}: rolled out on another dataset (fingerprint "
                             f"{theirs[:12]}, here {ours[:12]})")
        errors = {sid: {"q_rmse_all": s["rollout_metrics"].get("q_rmse_all"),
                        "stress_rmse_all": s["rollout_metrics"].get("stress_rmse_all"),
                        TO_DISPLACEMENT: to_displacement(s.get("position_rmse_per_step"),
                                                         s.get("displacement_rms_per_step"))}
                  for sid, s in rollout.summary["sequences"].items()}
        out.append(RolloutErrors(rollout.name, folder, rollout.split, errors))
    labels = [r.label for r in out]
    for r in out:
        if labels.count(r.label) > 1:
            r.label = f"{r.label} ({r.folder.name[:15]})"
    return out


def correlations(trajectories: List[Trajectory], rollouts: List[RolloutErrors]
                 ) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """Spearman's ρ of every error with every descriptor and the distance to train:
    rollout label -> error key -> descriptor key -> ρ."""
    by_id = {t.sequence_id: t for t in trajectories}
    keys = [key for key, *_ in DESCRIPTORS] + [DISTANCE[0]]
    out = {}
    for r in rollouts:
        out[r.label] = {}
        for error, *_ in ERRORS:
            pairs = [(by_id[sid].values, e[error]) for sid, e in r.errors.items()
                     if sid in by_id and e.get(error) is not None]
            out[r.label][error] = {}
            for key in keys:
                points = [(v[key], e) for v, e in pairs if v.get(key) is not None]
                out[r.label][error][key] = (spearman(*map(np.array, zip(*points)))
                                            if points else None)
    return out


def strongest(rho: Dict[str, Dict[str, Dict[str, Optional[float]]]], error: str,
              count: int) -> List[str]:
    """The ``count`` descriptors with the largest mean |ρ| over the rollouts for ``error``."""
    keys = {key for per_error in rho.values() for key, value in per_error[error].items()
            if value is not None}
    mean = {key: np.mean([abs(per_error[error][key]) for per_error in rho.values()
                          if per_error[error].get(key) is not None]) for key in keys}
    return sorted(mean, key=lambda key: -mean[key])[:count]


# ── findings ─────────────────────────────────────────────────────────────────

def describe(key: str, su: str = SI_STRESS) -> Tuple[str, float, str]:
    """Title, factor and unit of a descriptor or of the distance to train (``su``: the
    stress unit of the data)."""
    if key == DISTANCE[0]:
        return DISTANCE[1], DISTANCE[2], DISTANCE[3]
    for k, title, factor, unit in DESCRIPTORS:
        if k == key:
            return (title, *shown(factor, unit, su))
    raise KeyError(key)


def value_text(key: str, value: Optional[float], su: str = SI_STRESS) -> str:
    _, factor, unit = describe(key, su)
    return f"{text(value, factor)} {unit}".strip()


def findings(trajectories, targets, ks, outside, bound, rollouts, rho,
             su: str = SI_STRESS) -> List[str]:
    lines = []
    splits = [s for s in SPLITS if any(t.split == s for t in trajectories)]
    meshes = {s: {t.mesh_id for t in trajectories if t.split == s} for s in splits}
    shared = {m for s in splits for m in meshes[s]
              if sum(m in meshes[other] for other in splits) > 1}
    by_hash: Dict[str, List[str]] = {}
    for t in trajectories:
        by_hash.setdefault(t.sha256, []).append(t.sequence_id)
    twins = [ids for ids in by_hash.values() if len(ids) > 1]
    lines.append(
        "/".join(str(sum(t.split == s for t in trajectories)) for s in splits)
        + f" trajectories ({'/'.join(splits)}) on {len({t.mesh_id for t in trajectories})} "
          "meshes; "
        + (f"{len(shared)} meshes appear in more than one split" if shared
           else "no mesh appears in two splits")
        + "; " + ("identical trajectories (SHA-256 of the export): "
                  + "; ".join(" = ".join(ids) for ids in twins) if twins
                  else "no two trajectories are identical (SHA-256 of the export)") + ".")

    total = {key: sum(t.checks[key] for t in trajectories)
             for key in ("non_finite", "negative_stress", "degenerate_cells", "inverted_frames")}
    clamped = max(t.checks["clamped_motion"] for t in trajectories)
    volumes = column(trajectories, "volume_min")
    lines.append(
        f"Data checks: {total['non_finite']} non-finite values, {total['negative_stress']} "
        f"negative von Mises stresses, clamped nodes move at most {text(clamped, 1e3)} mm, "
        f"{total['degenerate_cells']} degenerate cells, {total['inverted_frames']} frames with "
        "an inverted element"
        + (f" (smallest element volume {text(volumes.min(), 100)} % of rest)"
           if len(volumes) else " (no tetrahedra: volumes not checked)")
        + "".join(f"; {t.sequence_id} ({t.source}): {t.checks['inverted_frames']} frames, down "
                  f"to {text(t.values['volume_min'], 100)} %"
                  for t in trajectories if t.checks["inverted_frames"]) + ".")
    no_contact = [t.sequence_id for t in trajectories if t.values["first_contact"] is None]
    if no_contact:
        lines.append(f"Without contact in any frame: {', '.join(no_contact)}.")

    if "train" in targets and targets["train"].n.any():
        train = targets["train"].summary()
        lines.append(
            "Train targets normalized with normalizer.json: mean "
            f"{max(abs(m) for m in train['mean']):.3f} σ at most from 0, std "
            f"{min(train['std']):.3f} to {max(train['std']):.3f} (1 if the normalizer belongs "
            "to this export).")
        shares, extremes = [], []
        for split in splits:
            s = targets[split].summary()
            names = targets[split].names
            far = max(range(len(names)), key=lambda k: s["far_share"][k])
            shares.append(f"{split} {s['far_share'][far] * 100:.3f} % ({names[far]})")
            top = max((t for t in trajectories if t.split == split),
                      key=lambda t: t.checks["target_sigma"])
            extremes.append(f"{split} {top.checks['target_sigma']:.0f} σ ({top.checks['target_name']}"
                            f", {top.sequence_id})")
        lines.append(f"Targets beyond ±{FAR_SIGMA:g} σ of the train statistics, the target with "
                     f"the largest share per split: {'; '.join(shares)}. Largest |target|: "
                     f"{'; '.join(extremes)}.")

    for other in ("val", "test"):
        if other not in ks:
            continue
        tests = [(TITLES[key], d, p) for key, (d, p) in ks[other].items()
                 if p is not None and p < 0.05]
        expected = 0.05 * len([1 for d, _ in ks[other].values() if d is not None])
        lines.append(
            f"Train vs {other}, Kolmogorov–Smirnov p < 0.05: "
            + (", ".join(f"{title} (D = {d:.2f}, p = {p:.3f})" for title, d, p in tests)
               if tests else "no descriptor")
            + f"; about {expected:.1f} such results are expected by chance.")
        members = [t for t in trajectories if t.split == other]
        ids = [t.sequence_id for t in members if outside.get(t.sequence_id)]
        line = (f"{len(ids)} of {len(members)} {other} trajectories lie outside the range of "
                "the train trajectories in at least one descriptor")
        if ids and other == "test":
            line += ": " + "; ".join(
                f"{sid} " + ", ".join(f"{TITLES[key]} {value_text(key, value, su)} (train "
                                      f"{value_text(key, low, su)} – "
                                      f"{value_text(key, high, su)})"
                                      for key, value, low, high in outside[sid])
                for sid in ids)
        lines.append(line + ".")
        if bound is not None:
            far = sorted((t.values[DISTANCE[0]], t.sequence_id) for t in members
                         if t.values.get(DISTANCE[0], 0) > bound)[::-1]
            lines.append(
                f"Distance to train above {text(bound)} σ (farther than "
                f"{COVERAGE_QUANTILE * 100:.0f} % of the train trajectories from their nearest "
                f"neighbour): {len(far)} of {len(members)} {other} trajectories"
                + (": " + ", ".join(f"{sid} ({text(d)} σ)" for d, sid in far) if far else "")
                + ".")

    for r in rollouts:
        n = len(r.errors)
        for error, title, _, _ in ERRORS:
            values = {key: v for key, v in rho[r.label][error].items() if v is not None}
            if not values:
                continue
            top = sorted(values, key=lambda key: -abs(values[key]))[:3]
            distance = values.get(DISTANCE[0])
            lines.append(
                f"{r.label}, {title} ({r.split}, {n} trajectories): strongest rank correlation "
                "with " + ", ".join(f"{describe(key)[0]} (ρ = {values[key]:+.2f})" for key in top)
                + (f"; distance to train ρ = {distance:+.2f}" if distance is not None
                   and DISTANCE[0] not in top else "")
                + f". |ρ| above {chance_bound(n):.2f} is unlikely by chance (p < 0.05).")
    return lines


# ── figures ──────────────────────────────────────────────────────────────────

def _small_title(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=TEXT_PRIMARY, pad=6)


def _box(ax, data: List[np.ndarray], names: List[str], colors: List[str]) -> None:
    """Boxes from the lower to the upper quartile with the median, whiskers from the
    smallest to the largest value."""
    present = [i for i, values in enumerate(data) if len(values)]
    ax.set_xticks(range(len(names)), names, fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlim(-0.6, len(names) - 0.4)
    if not present:   # e.g. distances with a single train trajectory
        return
    box = ax.boxplot([data[i] for i in present], positions=present, whis=(0, 100),
                     widths=0.55, patch_artist=True, showfliers=False,
                     medianprops={"color": TEXT_PRIMARY, "linewidth": 1.5})
    for part in ("whiskers", "caps"):
        for line, i in zip(box[part], [i for i in present for _ in range(2)]):
            line.set_color(colors[i])
    for patch, i in zip(box["boxes"], present):
        patch.set_facecolor(tint(colors[i], 0.6))
        patch.set_edgecolor(colors[i])


def plot_descriptors(trajectories: List[Trajectory], splits: List[str], path: Path,
                     su: str = SI_STRESS) -> None:
    columns = 6
    rows = math.ceil(len(DESCRIPTORS) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(18, 2.9 * rows + 1.2), facecolor=SURFACE)
    for ax, (key, title, factor, unit) in zip(axes.flat, DESCRIPTORS):
        factor, unit = shown(factor, unit, su)
        _style(ax, "y")
        _small_title(ax, with_unit(title, unit))
        data = [column(trajectories, key, s) * factor for s in splits]
        every = np.concatenate(data)
        if not len(every) or every.min() == every.max():
            ax.text(0.5, 0.5, f"every trajectory: {text(every[0]) if len(every) else '–'} {unit}",
                    ha="center", va="center", transform=ax.transAxes, color=TEXT_MUTED)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        _box(ax, data, splits, [SPLIT_COLORS[s] for s in splits])
        if every.min() > 0 and every.max() / every.min() > WIDE_RANGE:
            ax.set_yscale("log")
    for ax in axes.flat[len(DESCRIPTORS):]:
        ax.axis("off")
    counts = ", ".join(f"{s} {sum(t.split == s for t in trajectories)}" for s in splits)
    top = _figure_header(
        fig, "Trajectories per split",
        f"{counts} trajectories. Box: lower to upper quartile with the median; whiskers: "
        "smallest to largest trajectory. Structure = all nodes but the actuator; displacement "
        "since frame 0; contact pairs: nodes of different bodies closer than the contact radius.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_over_time(trajectories: List[Trajectory], splits: List[str], path: Path,
                   su: str = SI_STRESS, raw_frames: Optional[List[int]] = None) -> None:
    """Descriptors over the frames, on the axis of the raw frames of the run
    (``raw_frames``: of every frame of the dataset; None: 0, 1, ...)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), facecolor=SURFACE)
    for ax, (key, title, factor, unit) in zip(axes.flat, OVER_TIME):
        factor, unit = shown(factor, unit, su)
        for split in splits:
            curves = [t.series[key] for t in trajectories if t.split == split]
            length = max(len(c) for c in curves)
            stacked = np.full((len(curves), length), np.nan)
            for row, curve in zip(stacked, curves):
                row[:len(curve)] = curve * factor
            if np.isnan(stacked).all():   # e.g. the stress of a dataset without it
                continue
            low, median, high = np.nanpercentile(stacked, [10, 50, 90], axis=0)
            frames = np.asarray(raw_frames[:length]) if raw_frames else np.arange(length)
            ax.fill_between(frames, low, high, color=SPLIT_COLORS[split], alpha=0.14, linewidth=0)
            ax.plot(frames, median, color=SPLIT_COLORS[split], linewidth=2,
                    label=f"{split} ({len(curves)})")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Frame of the run", fontsize=10, color=TEXT_SECONDARY)
        _small_title(ax, with_unit(title, unit))
        _style(ax, "both")
        _legend(ax)
    top = _figure_header(
        fig, "Over the frames",
        "Line: median of the trajectories of a split at each frame; band: 10 to 90 % of them. "
        "Displacement since frame 0 over the nodes of the structure; actuator travel: distance "
        "of the actuator's centre from its start.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_targets(targets: Dict[str, TargetStats], splits: List[str], path: Path) -> None:
    names = targets[splits[0]].names
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4.8), facecolor=SURFACE,
                             sharey=True, squeeze=False)
    axes = axes[0]
    width = TargetStats.EDGES[1] - TargetStats.EDGES[0]
    for k, (ax, name) in enumerate(zip(axes, names)):
        for split in splits:
            s = targets[split]
            density = s.counts[k] / s.n[k] / width
            ax.stairs(np.where(density > 0, density, np.nan), TargetStats.EDGES,
                      color=SPLIT_COLORS[split], linewidth=1.6, label=split)
        for x in (-FAR_SIGMA, FAR_SIGMA):
            ax.axvline(x, color=TEXT_MUTED, linewidth=0.75, linestyle=(0, (4, 3)))
        ax.set_yscale("log")
        ax.set_xlabel("normalized value [train std]", fontsize=10, color=TEXT_SECONDARY)
        _small_title(ax, name)
        _style(ax, "both")
    _legend(axes[0])
    top = _figure_header(
        fig, "Targets as the model sees them",
        "Velocity per frame and the result fields of the structure nodes, normalized with "
        f"normalizer.json (train). Density, log scale; values beyond ±{HISTOGRAM_LIMIT:g} in the "
        f"outer bins; dashed: ±{FAR_SIGMA:g}.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_coverage(trajectories: List[Trajectory], splits: List[str], ks, bound: Optional[float],
                  path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 7.4), facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1, 1.5]})
    _style(left, "y")
    _small_title(left, "Distance to the nearest train trajectory [σ]")
    _box(left, [column(trajectories, DISTANCE[0], s) for s in splits], splits,
         [SPLIT_COLORS[s] for s in splits])
    if bound is not None:
        left.axhline(bound, color=TEXT_MUTED, linewidth=0.75, linestyle=(0, (4, 3)))
        left.annotate(f"{COVERAGE_QUANTILE * 100:.0f} % of train", (-0.55, bound),
                      xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=TEXT_MUTED)
    left.set_ylim(bottom=0)

    others = [s for s in ("val", "test") if s in ks]
    positions = np.arange(len(DESCRIPTORS))[::-1]
    height = 0.8 / max(len(others), 1)
    for j, split in enumerate(others):
        for y, (key, *_) in zip(positions, DESCRIPTORS):
            d, p = ks[split].get(key, (None, None))
            if d is None:
                continue
            color = SPLIT_COLORS[split] if p < 0.05 else tint(SPLIT_COLORS[split], 0.6)
            right.barh(y + (len(others) / 2 - j - 0.5) * height, d, height=height * 0.9,
                       color=color, label=f"train vs {split}" if y == positions[0] else None)
    right.set_yticks(positions, [title for _, title, _, _ in DESCRIPTORS], fontsize=9,
                     color=TEXT_PRIMARY)
    right.set_xlim(0, 1)
    _style(right, "x")
    _small_title(right, "Kolmogorov–Smirnov distance D (dark: p < 0.05)")
    if others:
        _legend(right)
    top = _figure_header(
        fig, "How far val and test lie from train",
        "Left: descriptors standardized with the train mean and std, RMS difference to the "
        "nearest train trajectory (for train: the nearest other one); dashed: the distance below "
        f"which {COVERAGE_QUANTILE * 100:.0f} % of the train trajectories lie. Right: largest "
        "gap between the distributions of a descriptor (0 equal, 1 disjoint).")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_errors(trajectories: List[Trajectory], rollouts: List[RolloutErrors], rho,
                path: Path, su: str = SI_STRESS) -> bool:
    by_id = {t.sequence_id: t for t in trajectories}
    rows = [e for e in ERRORS if any(strongest(rho, e[0], 1))]
    if not rows:
        return False
    fig, axes = plt.subplots(len(rows), SCATTER_COLUMNS, figsize=(15, 4.3 * len(rows) + 1),
                             facecolor=SURFACE, squeeze=False)
    for row_axes, (error, title, error_factor, error_unit) in zip(axes, rows):
        error_factor, error_unit = shown(error_factor, error_unit, su)
        keys = strongest(rho, error, SCATTER_COLUMNS)
        for ax, key in zip(row_axes, keys):
            name, factor, unit = describe(key, su)
            for i, r in enumerate(rollouts):
                points = [(by_id[sid].values[key], e[error], sid) for sid, e in r.errors.items()
                          if sid in by_id and by_id[sid].values.get(key) is not None
                          and e.get(error) is not None]
                if not points:
                    continue
                x, y, ids = zip(*points)
                color = SERIES_COLORS[i % len(SERIES_COLORS)]
                ax.scatter(np.array(x) * factor, np.array(y) * error_factor, s=26, color=color,
                           edgecolor=SURFACE, linewidth=0.6, label=r.label, zorder=3)
                median = float(np.median(y))
                for xv, yv, sid in points:
                    if yv > FAR_ABOVE_MEDIAN * median:
                        ax.annotate(sid.rsplit("traj", 1)[-1], (xv * factor, yv * error_factor),
                                    xytext=(4, 2), textcoords="offset points", fontsize=8,
                                    color=TEXT_SECONDARY)
            values = " / ".join(f"{rho[r.label][error][key]:+.2f}" for r in rollouts
                                if rho[r.label][error].get(key) is not None)
            _small_title(ax, f"{name}: ρ = {values}")
            ax.set_xlabel(with_unit(name, unit), fontsize=9.5, color=TEXT_SECONDARY)
            ax.set_ylabel(with_unit(title, error_unit), fontsize=9.5, color=TEXT_SECONDARY)
            _style(ax, "both")
        for ax in row_axes[len(keys):]:
            ax.axis("off")
    _legend(axes[0][0])
    n = min(len(r.errors) for r in rollouts)
    top = _figure_header(
        fig, "Error against the data",
        "One point per rolled-out trajectory; per error the descriptors with the strongest rank "
        f"correlation (Spearman's ρ; for {n} trajectories |ρ| above {chance_bound(n):.2f} is "
        f"unlikely by chance). Labelled: error above {FAR_ABOVE_MEDIAN:g} × the median.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


# ── report ───────────────────────────────────────────────────────────────────

def dataset_rows(data_dir: Path, info: Dict, metadata: Dict, trajectories: List[Trajectory],
                 rollouts: List[RolloutErrors]) -> List[List[str]]:
    splits = metadata.get("splits") or {}
    sources = []
    for split in SPLITS:
        s = splits.get(split) or {}
        if s:
            sources.append(f"{split} {s.get('count')}"
                           + (f" from {s['source']}" if s.get("source") else ""))
    nodes = column(trajectories, "nodes")
    rows = [
        ["Dataset", f"`{data_dir}` ({metadata.get('dataset', '?')}, format {info.get('format')}"
                    + (f", fingerprint {info['fingerprint'][:12]}" if info.get("fingerprint")
                       else "") + ")"],
        ["Trajectories", ", ".join(sources)
         + (f"; drawn with seed {metadata['select_seed']}" if "select_seed" in metadata else "")],
        ["Frames", f"{metadata.get('num_steps', '?')} per trajectory at stride "
                   f"{metadata.get('time_stride', '?')}"],
        ["Nodes", f"{int(nodes.min())} to {int(nodes.max())} per trajectory, median "
                  f"{int(np.median(nodes))}"],
        ["Contact radius", f"{float(info['contact_radius']) * si_factors(info)[0] * 1e3:g} mm"],
    ]
    if rollouts:
        rows.append(["Rollouts", "; ".join(f"{r.label}: `{r.folder.name}` ({r.split})"
                                           for r in rollouts)])
    return rows


def split_summary(trajectories: List[Trajectory], split: str) -> Dict[str, Dict[str, float]]:
    """Smallest, median and largest value of every descriptor over ``split``."""
    out = {}
    for key, *_ in DESCRIPTORS:
        values = column(trajectories, key, split)
        if len(values):
            out[key] = {"min": float(values.min()), "median": float(np.median(values)),
                        "max": float(values.max())}
    return out


def spread_text(values: np.ndarray, factor: float) -> str:
    if not len(values):
        return "–"
    return (f"{text(float(np.median(values)), factor)} ({text(values.min(), factor)} – "
            f"{text(values.max(), factor)})")


def descriptor_rows(trajectories, splits, ks, su: str = SI_STRESS) -> List[List[str]]:
    rows = []
    for key, title, factor, unit in DESCRIPTORS:
        factor, unit = shown(factor, unit, su)
        row = [with_unit(title, unit)] + [spread_text(column(trajectories, key, s), factor)
                                          for s in splits]
        for other in ("val", "test"):
            if other in splits:
                d, p = ks[other].get(key, (None, None))
                row.append("–" if d is None else f"{d:.2f} ({p:.3f})")
        rows.append(row)
    return rows


def check_rows(trajectories, splits) -> List[List[str]]:
    checks = [("non_finite", "Non-finite values", sum),
              ("negative_stress", "Negative von Mises stresses", sum),
              ("clamped_motion", "Clamped nodes, largest motion [mm]", max),
              ("degenerate_cells", "Degenerate cells", sum),
              ("inverted_frames", "Frames with an inverted element", sum)]
    rows = []
    for key, title, reduce in checks:
        row = [title]
        for s in splits:
            value = reduce(t.checks[key] for t in trajectories if t.split == s)
            row.append(text(value, 1e3) if key == "clamped_motion" else str(int(value)))
        rows.append(row)
    rows.append(["Trajectories without contact"]
                + [str(sum(t.split == s and t.values["first_contact"] is None
                           for t in trajectories)) for s in splits])
    rows.append(["Largest normalized target [σ]"] + [
        "{0:.0f} ({1}, {2})".format(*max(((t.checks["target_sigma"], t.checks["target_name"],
                                           t.sequence_id) for t in trajectories if t.split == s)))
        for s in splits])
    rows.append(["Frames per trajectory"] + [
        "/".join(str(int(f)) for f in sorted({t.checks["frames"] for t in trajectories
                                              if t.split == s})) for s in splits])
    return rows


def target_rows(targets, splits) -> List[List[str]]:
    rows = []
    for k, name in enumerate(targets[splits[0]].names):
        for s in splits:
            v = targets[s].summary()
            rows.append([name, s, f"{v['mean'][k]:+.3f}", f"{v['std'][k]:.3f}",
                         f"{v['min'][k]:.1f} – {v['max'][k]:.1f}",
                         f"{v['far_share'][k] * 100:.3f}"])
    return rows


def split_rows(trajectories, split, outside, rollouts,
               su: str = SI_STRESS) -> Tuple[List[str], List[List[str]]]:
    """Every trajectory of ``split``, farthest from train first."""
    errors = [(key, title, *shown(factor, unit, su)) for key, title, factor, unit in ERRORS[:2]]
    members = sorted((t for t in trajectories if t.split == split),
                     key=lambda t: -(t.values.get(DISTANCE[0]) or 0))
    on_split = [r for r in rollouts if r.split == split]
    columns = (["Trajectory", "Source", "Distance to train [σ]", "Outside the train range"]
               + [f"{r.label}: {title} [{unit}]" for r in on_split
                  for key, title, _, unit in errors])
    rows = []
    for t in members:
        row = [t.sequence_id, t.source, text(t.values.get(DISTANCE[0])),
               ", ".join(TITLES[key] for key, *_ in outside.get(t.sequence_id, []))]
        for r in on_split:
            e = r.errors.get(t.sequence_id, {})
            row += [text(e.get(key), factor) for key, _, factor, _ in errors]
        rows.append(row)
    return columns, rows


def correlation_rows(rho, rollouts) -> Tuple[List[str], List[List[str]]]:
    columns = ["Descriptor"] + [f"{r.label}: {title}" for r in rollouts for _, title, _, _ in ERRORS]
    rows = []
    for key in [k for k, *_ in DESCRIPTORS] + [DISTANCE[0]]:
        rows.append([describe(key)[0]] + [
            "–" if rho[r.label][error].get(key) is None else f"{rho[r.label][error][key]:+.2f}"
            for r in rollouts for error, *_ in ERRORS])
    return columns, rows


def new_output_dir(data_dir: Path) -> Path:
    base = EVALUATIONS_DIR / f"{now():%Y%m%d_%H%M%S}_{data_dir.name}_data_exploration"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}_{n}")
    path.mkdir(parents=True)
    return path


def build_parser() -> argparse.ArgumentParser:
    """Command line options; the reference in docs/reference/cli.md is generated from them."""
    parser = argparse.ArgumentParser(
        description="Explore an exported dataset: descriptors, data checks and the splits "
                    "against each other.")
    parser.add_argument("data_dir", type=Path, help="export folder (data/<name>)")
    parser.add_argument("--rollout", type=Path, action="append", default=[],
                        help="rollout folder on this dataset; sets its errors per trajectory "
                             "against the descriptors (repeatable)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="new or empty folder (default: "
                             "evaluations/<time>_<name>_data_exploration)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Windows writes redirected output in its code page, which lacks characters such as
    # σ, ρ and −: replace them there instead of failing at the end (PYTHONUTF8=1 keeps them).
    sys.stdout.reconfigure(errors="replace")

    data_dir = args.data_dir
    if args.output_dir is not None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty — evaluations never overwrite")
    rollouts = load_errors(args.rollout, data_dir / "dataset.db")
    info, trajectories, targets = scan_dataset(data_dir)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    su = info.get("stress_unit", SI_STRESS)   # the reports show the stress in it
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir or new_output_dir(data_dir)

    splits = [s for s in SPLITS if any(t.split == s for t in trajectories)]
    ks = {other: {key: ks_test(column(trajectories, key, "train"),
                               column(trajectories, key, other)) for key, *_ in DESCRIPTORS}
          for other in ("val", "test") if other in splits and "train" in splits}
    distance_keys = add_distances(trajectories)
    train_distances = column(trajectories, DISTANCE[0], "train")
    bound = (float(np.quantile(train_distances, COVERAGE_QUANTILE))
             if len(train_distances) else None)
    outside = outside_train(trajectories)
    rho = correlations(trajectories, rollouts)
    lines = findings(trajectories, targets, ks, outside, bound, rollouts, rho, su)

    plot_descriptors(trajectories, splits, out / "01_descriptors.png", su)
    plot_over_time(trajectories, splits, out / "02_over_time.png", su,
                   json.loads(info.get("frames", "null")))
    plot_targets(targets, splits, out / "03_targets.png")
    plot_coverage(trajectories, splits, ks, bound, out / "04_coverage.png")
    with_errors = bool(rollouts) and plot_errors(trajectories, rollouts, rho,
                                                 out / "05_error_vs_data.png", su)

    ks_columns = [f"KS train–{other}: D (p)" for other in ("val", "test") if other in ks]
    report = [
        f"# Data exploration of {data_dir.name} ({now():%Y-%m-%d %H:%M})",
        "",
        markdown_table(["", ""], dataset_rows(data_dir, info, metadata, trajectories, rollouts)),
        "",
        "## Findings",
        "",
        *[f"- {line}" for line in lines],
        "",
        "## Data checks",
        "",
        markdown_table(["Check"] + splits, check_rows(trajectories, splits)),
        "",
        "## Trajectories per split",
        "",
        "Median (smallest – largest) over the trajectories of each split. Kolmogorov–Smirnov: "
        "largest gap D between the distributions of train and the other split (0 equal, 1 "
        f"disjoint) and its p-value; with {len(DESCRIPTORS)} descriptors about "
        f"{0.05 * len(DESCRIPTORS):.1f} of them fall below p = 0.05 by chance.",
        "",
        markdown_table(["Descriptor"] + splits + ks_columns,
                       descriptor_rows(trajectories, splits, ks, su)),
        "",
        "![descriptors](01_descriptors.png)",
        "",
        "![over time](02_over_time.png)",
        "",
        "## Targets",
        "",
        "The targets of the structure nodes over all frame pairs, normalized with "
        "`normalizer.json` (fitted on train): mean and std in units of the train std, range, "
        f"and the share beyond ±{FAR_SIGMA:g}.",
        "",
        markdown_table(["Target", "Split", "Mean", "Std", "Range", f"Beyond ±{FAR_SIGMA:g} [%]"],
                       target_rows(targets, splits)),
        "",
        "![targets](03_targets.png)",
        "",
        "## Coverage of val and test by train",
        "",
        f"Distance to train over {len(distance_keys)} descriptors (finite and not constant in "
        "train): " + ", ".join(TITLES[key] for key in distance_keys) + ".",
        "",
        "![coverage](04_coverage.png)",
        "",
    ]
    for split in ("test", "val"):
        if split in splits:
            columns, rows = split_rows(trajectories, split, outside, rollouts, su)
            report += [f"### {split.capitalize()} trajectories", "",
                       "Farthest from train first.", "", markdown_table(columns, rows), ""]
    if rollouts:
        columns, rows = correlation_rows(rho, rollouts)
        report += ["## Error against the data", "",
                   "Spearman's rank correlation of the error per trajectory with each "
                   "descriptor; for n trajectories |ρ| above 1.96 / √(n − 1) is unlikely by "
                   "chance (p < 0.05).", "", markdown_table(columns, rows), ""]
        if with_errors:
            report += ["![error against the data](05_error_vs_data.png)", ""]
    (out / "exploration.md").write_text("\n".join(report), encoding="utf-8")

    error_columns = [(r, key) for r in rollouts for key, *_ in ERRORS]
    with open(out / "trajectories.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trajectory", "split", "source", "mesh_id"]
                        + [key for key, *_ in DESCRIPTORS] + [DISTANCE[0], "outside_train_range"]
                        + [f"{r.label}: {key}" for r, key in error_columns])
        for t in trajectories:
            writer.writerow([t.sequence_id, t.split, t.source, t.mesh_id]
                            + [t.values.get(key) for key, *_ in DESCRIPTORS]
                            + [t.values.get(DISTANCE[0]),
                               ";".join(key for key, *_ in outside.get(t.sequence_id, []))]
                            + [r.errors.get(t.sequence_id, {}).get(key)
                               for r, key in error_columns])
    with open(out / "exploration.json", "w", encoding="utf-8") as f:
        json.dump({
            "created": now().isoformat(timespec="seconds"), "data_dir": str(data_dir),
            "fingerprint": info.get("fingerprint"),
            "units": "SI: lengths in m, stress in Pa, shares as fraction; targets in train std",
            "descriptors": {key: dict(zip(("title", "factor", "unit"), describe(key, su)))
                            for key, *_ in DESCRIPTORS},
            "findings": lines,
            "splits": {s: split_summary(trajectories, s) for s in splits},
            "ks": {other: {key: {"d": d, "p": p} for key, (d, p) in values.items()}
                   for other, values in ks.items()},
            "targets": {s: targets[s].summary() for s in splits},
            "coverage": {"descriptors": distance_keys, "bound": bound,
                         "quantile": COVERAGE_QUANTILE},
            "rollouts": [{"label": r.label, "folder": str(r.folder), "split": r.split}
                         for r in rollouts],
            "correlations": rho,
            "trajectories": {t.sequence_id: {
                "split": t.split, "source": t.source, "mesh_id": t.mesh_id, **t.values,
                "outside_train_range": [key for key, *_ in outside.get(t.sequence_id, [])],
                "checks": t.checks} for t in trajectories},
        }, f, indent=2, ensure_ascii=False)

    print(f"Data exploration: {out}")
    for line in lines:
        print(f"- {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
