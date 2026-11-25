"""
prepare.py — Prepare new FE runs for a prediction with a trained model, after checking
whether the model applies to them.

A trained model predicts a crash from the undeformed state. What it needs of a new run is
in the model definition, not in a simulation result: the mesh at t0 and the deck
(thickness, material, connections, initial velocity, rigid bodies, impactor). This tool
takes new runs that hold at least their initial state, a ``d3plot`` with the deck next to
it, reads them as the training data was read (the reader settings in the ``info`` of its
``dataset.db``), and

1. checks every run against the train runs of the model (``describe``,
   ``training_reference``, ``check``). A run the model cannot take stops the preparation
   (``error``); a quantity outside the range of the train runs is an extrapolation of the
   model (``outside``): it is prepared, and the result says so;
2. writes the runs as prediction data (``export.InitialState``): the initial state for
   every frame, the motion before it from the initial velocity of the deck, the layout and
   the normalizer of the training data, no ground truth. ``predict/rollout.py`` rolls them
   out like any dataset, without errors to measure.

The checks (lengths in the length unit of the data; structure: every node but the
actuator; the speed in length per time unit of the deck, e.g. mm/ms = m/s):

- *errors*: free rigid bodies, actuator, static node features and run parameters as in the
  training data; an initial velocity where the train runs have one; actuator and clamped
  nodes at rest in the training data, the actuator also in the run (their motion would
  have to be given);
- *ranges of the train runs*: extent of the structure, median length of its mesh edges,
  initial speed and its direction, the values of every static node feature (thickness,
  yield stress, ...), position and extent of the actuator relative to the structure and
  its clearance at t0, the number of steps;
- *deck*: the mass of each free rigid body and Young's modulus and density of the parts.
  The model does not see them, so the training data fixes them. Compared where a deck
  states them (OpenRadioss starter deck), else not checked;
- *parts*: parts the model has not seen and parts that no pattern of the training data
  matches (left out of the prediction) are to be reviewed (``review``): a display model may
  rightly be left out, a part of the structure not.

Lying inside every range does not make a prediction reliable: the train runs span a
region, and a combination of values inside the ranges may still be new to the model.

Usage::

    python predict/prepare.py --checkpoint checkpoints/<run>/best_model.pt \\
        --raw-dir <folder with the new runs> --out-dir data/<name>
    python predict/rollout.py --checkpoint checkpoints/<run>/best_model.pt \\
        --data-dir data/<name> --split test --export-vtu
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dataset_preprocessor.export import VELOCITY_FIELDS, InitialState, write_dataset  # noqa: E402
from dataset_preprocessor.graph_features import (  # noqa: E402
    GIVEN_SUFFIX,
    NODE_TYPE_ACTUATOR,
    NODE_TYPE_CLAMPED,
    NODE_TYPE_NORMAL,
    WALL_DISTANCE_CUTOFF,
    mesh_edges_from_cells,
    radius_pairs,
)
from utils.normalizer import Normalizer  # noqa: E402

OK, OUTSIDE, ERROR, INFO, UNCHECKED, REVIEW = ("ok", "outside", "error", "info",
                                               "not checked", "review")
# Relative tolerance of a range: values this close to a bound count as inside.
RANGE_TOLERANCE = 1e-3
# Largest angle between the impact direction of a run and the mean of the train runs that
# counts as the same direction, beyond the scatter of the train runs.
DIRECTION_TOLERANCE_DEG = 1.0


@dataclass
class Finding:
    """One check of a run: its value, that of the training data, the outcome."""

    check: str
    value: object
    reference: object
    status: str
    note: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# What a run looks like at t0
# ─────────────────────────────────────────────────────────────────────────────

def _base_name(component: str) -> str:
    """A part name without the ID the reader appends to names that occur twice."""
    return component.rsplit(" (", 1)[0] if component.endswith(")") else component


def _clearance(pos: np.ndarray, actuator: np.ndarray, structure: np.ndarray,
               reach: float) -> float:
    """Smallest distance between an actuator node and a structure node, ``reach`` if none
    is closer (a cell list over the structure near the actuator: scales with the mesh)."""
    act = pos[actuator]
    low, high = act.min(0) - reach, act.max(0) + reach
    near = structure & np.all((pos >= low) & (pos <= high), axis=1)
    if not near.any():
        return reach
    points = torch.from_numpy(np.concatenate([act, pos[near]]))
    pairs = radius_pairs(points, reach)
    cross = (pairs[0] < len(act)) & (pairs[1] >= len(act))
    if not bool(cross.any()):
        return reach
    i, j = pairs[0][cross], pairs[1][cross]
    return float((points[i] - points[j]).norm(dim=1).min())


def describe(mesh_pos: np.ndarray, node_type: np.ndarray, edge_index: np.ndarray,
             static: Dict[str, np.ndarray], start_velocity: Optional[np.ndarray],
             frame_interval: float, reach: float) -> Dict[str, object]:
    """The quantities of one run at t0 that the checks compare (module docstring);
    ``start_velocity``: motion over one frame interval before frame 0, None: at rest."""
    pos = np.asarray(mesh_pos, dtype=np.float64)
    structure = node_type != NODE_TYPE_ACTUATOR
    actuator = node_type == NODE_TYPE_ACTUATOR
    result: Dict[str, object] = {"structure nodes": int(structure.sum())}
    low, high = pos[structure].min(0), pos[structure].max(0)
    for axis, size in zip("xyz", high - low):
        result[f"structure extent {axis}"] = float(size)
    i, j = edge_index
    inside = structure[i] & structure[j]
    if inside.any():
        result["mesh edge length (median)"] = float(np.median(
            np.linalg.norm(pos[j[inside]] - pos[i[inside]], axis=1)))
    for name, values in sorted(static.items()):
        given = values[~np.isnan(values)]
        if not name.endswith(GIVEN_SUFFIX) and name not in VELOCITY_FIELDS and len(given):
            result[f"values of {name}"] = (float(given.min()), float(given.max()))
    if start_velocity is not None:
        free = node_type == NODE_TYPE_NORMAL
        velocity = np.asarray(start_velocity, dtype=np.float64)
        mean = velocity[free].mean(0) / frame_interval
        result["initial speed"] = float(np.linalg.norm(mean))
        if result["initial speed"] > 0:
            result["impact direction"] = (mean / np.linalg.norm(mean)).tolist()
        if actuator.any():
            result["initial speed of the actuator"] = float(
                np.linalg.norm(velocity[actuator], axis=1).max() / frame_interval)
    if actuator.any():
        offset = pos[actuator].mean(0) - pos[structure].mean(0)
        extent = pos[actuator].max(0) - pos[actuator].min(0)
        for axis, a, b in zip("xyz", offset, extent):
            result[f"actuator offset {axis}"] = float(a)
            result[f"actuator extent {axis}"] = float(b)
        result["clearance at t0"] = _clearance(pos, actuator, structure, reach)
    return result


def describe_raw(traj, frame_interval: float, reach: float) -> Dict[str, object]:
    """``describe`` of a run read with its first state (``RawTrajectory``)."""
    static = dict(traj.node_static or {})
    velocity = [static.get(name) for name in VELOCITY_FIELDS]
    start = (np.stack(velocity, axis=1) * frame_interval
             if all(v is not None for v in velocity) else None)
    edges = mesh_edges_from_cells(traj.cells, traj.cell_types).numpy()
    return describe(traj.mesh_pos, traj.node_type, edges, static, start, frame_interval, reach)


def deck_facts(run_dir) -> Optional[Dict[str, object]]:
    """What the deck states that the model does not see: the mass of every free rigid body
    and Young's modulus and density of the parts; None without an OpenRadioss starter deck."""
    from dataset_preprocessor.radioss_deck import read_deck, starter_deck

    path = starter_deck(run_dir)
    if path is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deck = read_deck(path)
    materials = list(deck.part_materials.values())
    return {"rigid body masses": dict(deck.rigid_body_masses),
            "youngs_modulus": sorted({m["youngs_modulus"] for m in materials}),
            "density": sorted({m["density"] for m in materials})}


# ─────────────────────────────────────────────────────────────────────────────
# What the model was trained on
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingData:
    """The train runs of a model: their settings and the quantities ``check`` compares."""

    info: Dict[str, str]
    runs: List[Dict[str, object]]        # ``describe`` of every train run
    decks: List[Dict[str, object]]       # ``deck_facts`` of the train runs that have one
    prescribed_move: bool                # actuator or clamped nodes move in the train runs
    raw_runs: List[Path]                 # the train runs in the raw data, if found


def _db(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def train_run_dirs(data_dir: Path, info: Dict[str, str]) -> List[Path]:
    """The folders of the train runs in the raw data the export read, [] if not there."""
    from dataset_preprocessor.d3plot_reader import run_dirs

    with open(data_dir / "metadata.json", encoding="utf-8") as f:
        raw_dir = Path(json.load(f).get("raw_dir", ""))
    if not raw_dir.is_dir():
        return []
    train = json.loads(info["splits"])["train"]
    sources = json.loads(info["sources"]) if info["sources"].startswith("[") \
        else [info["sources"]]
    source = train.get("source", sources[0])
    positions = train.get("indices", range(train["start"], train["start"] + train["count"]))
    try:
        runs = run_dirs(raw_dir, source)
        return [runs[p] for p in positions]
    except (OSError, IndexError, ValueError):
        return []


def training_reference(data_dir: Path, frame_interval: Optional[float] = None,
                       with_decks: bool = True) -> Tuple[TrainingData, float]:
    """The train runs of an export (``data_dir``, read-only) and the frame interval: given,
    or from the times of the first train run in the raw data."""
    from dataset_preprocessor.d3plot_reader import state_times

    conn = _db(data_dir / "dataset.db")
    info = dict(conn.execute("SELECT key, value FROM info"))
    stride = int(info.get("time_stride", 1))
    raw_runs = train_run_dirs(data_dir, info)
    if frame_interval is None:
        if not raw_runs:
            raise ValueError(f"the raw data of {data_dir} are not where the export read them: "
                             "pass the time between two frames of the training data "
                             "(--frame-interval)")
        times = state_times(raw_runs[0])
        frame_interval = float(times[stride] - times[0])
    static_names = json.loads(info.get("static_names", "[]"))
    contact_radius = float(info["contact_radius"])
    edges = {mesh_id: np.frombuffer(blob, np.int32).reshape(2, -1)
             for mesh_id, blob in conn.execute("SELECT id, mesh_edges FROM meshes")}
    runs = []
    rows = conn.execute("SELECT node_type, mesh_pos, node_static, start_velocity, mesh_id, "
                        "num_nodes FROM trajectories WHERE split = 'train' ORDER BY id")
    for node_type, mesh_pos, node_static, start, mesh_id, num_nodes in rows:
        values = (np.frombuffer(node_static, np.float32).reshape(num_nodes, -1)
                  if node_static else np.zeros((num_nodes, 0), np.float32))
        runs.append(describe(
            np.frombuffer(mesh_pos, np.float32).reshape(num_nodes, 3),
            np.frombuffer(node_type, np.uint8), edges[mesh_id],
            dict(zip(static_names, values.T)),
            None if start is None else np.frombuffer(start, np.float32).reshape(num_nodes, 3),
            frame_interval, WALL_DISTANCE_CUTOFF * contact_radius))
    first = conn.execute("SELECT id, node_type, num_nodes, num_frames FROM trajectories "
                         "WHERE split = 'train' ORDER BY id LIMIT 1").fetchone()
    moves = False
    if first is not None:
        prescribed = np.isin(np.frombuffer(first[1], np.uint8),
                             (NODE_TYPE_ACTUATOR, NODE_TYPE_CLAMPED))
        if prescribed.any():
            positions = [np.frombuffer(blob, np.float32).reshape(first[2], 3)[prescribed]
                         for (blob,) in conn.execute(
                             "SELECT world_pos FROM frames WHERE trajectory_id = ? AND frame IN "
                             "(0, ?) ORDER BY frame", (first[0], first[3] - 1))]
            moves = bool(np.abs(positions[-1] - positions[0]).max() > 1e-6 * max(
                1.0, float(np.abs(positions[0]).max())))
    conn.close()
    decks = [facts for facts in (deck_facts(run) for run in raw_runs) if facts] \
        if with_decks else []
    return TrainingData(info, runs, decks, moves, raw_runs), frame_interval


# ─────────────────────────────────────────────────────────────────────────────
# The checks
# ─────────────────────────────────────────────────────────────────────────────

def _inside(value: float, low: float, high: float) -> bool:
    tolerance = RANGE_TOLERANCE * max(abs(low), abs(high), 1e-12)
    return low - tolerance <= value <= high + tolerance


def _range(values: Sequence[float]) -> Tuple[float, float]:
    return float(min(values)), float(max(values))


def check(run: Dict[str, object], facts: Optional[Dict[str, object]], traj,
          training: TrainingData, num_steps: int) -> List[Finding]:
    """The findings for one run (``describe``, ``deck_facts``, its ``RawTrajectory``)."""
    info, runs = training.info, training.runs
    findings: List[Finding] = []

    def add(name, value, reference, status, note=""):
        findings.append(Finding(name, value, reference, status, note))

    # What the model cannot take at all
    rigid = json.loads(info.get("rigid_bodies", "[]"))
    add("free rigid bodies", list(traj.rigid_bodies or []), rigid,
        OK if list(traj.rigid_bodies or []) == rigid else ERROR,
        "" if list(traj.rigid_bodies or []) == rigid else "must be those of the training data")
    trained_actuator = any("clearance at t0" in r for r in runs)
    has_actuator = "clearance at t0" in run
    add("actuator", has_actuator, trained_actuator,
        OK if has_actuator == trained_actuator else ERROR,
        "" if has_actuator == trained_actuator else "the training data has one or has none")
    if training.prescribed_move:
        add("prescribed motion", "at rest", "moves in the training data", ERROR,
            "actuator or clamped nodes move in the train runs: their motion would have to "
            "be given; not supported")
    if run.get("initial speed of the actuator", 0.0) > 0:
        add("actuator motion", "initial velocity", "at rest", ERROR,
            "a moving actuator would have to be given its motion; not supported")
    needed = [name for name in json.loads(info.get("static_names", "[]"))
              if not name.endswith(GIVEN_SUFFIX)]
    missing = [name for name in needed if f"values of {name}" not in run]
    add("static node features", sorted(set(needed) - set(missing)), needed,
        ERROR if missing else OK, f"missing: {', '.join(missing)}" if missing else "")
    run_parameters = json.loads(info.get("global_names", "[]"))
    if run_parameters:
        absent = [name for name in run_parameters if name not in (traj.globals or {})]
        add("run parameters", sorted(traj.globals or {}), run_parameters,
            ERROR if absent else OK, f"missing: {', '.join(absent)}" if absent else "")
    trained_speed = [r["initial speed"] for r in runs if "initial speed" in r]
    if trained_speed and max(trained_speed) > 0 and not run.get("initial speed"):
        add("initial velocity", "none", "given in the train runs", ERROR,
            "without it nothing moves at t0")

    # Ranges of the train runs
    add("structure nodes", run["structure nodes"],
        _range([r["structure nodes"] for r in runs]), INFO)
    names = [key for key in runs[0] if isinstance(runs[0][key], (int, float))
             and key not in ("structure nodes", "initial speed of the actuator")]
    for name in names:
        if name not in run:
            continue
        reference = _range([r[name] for r in runs if name in r])
        add(name, run[name], reference, OK if _inside(run[name], *reference) else OUTSIDE)
    for name in (key for key in run if key.startswith("values of ")):
        trained = [r[name] for r in runs if name in r]
        if trained:
            reference = (min(t[0] for t in trained), max(t[1] for t in trained))
            inside = _inside(run[name][0], *reference) and _inside(run[name][1], *reference)
            add(name, run[name], reference, OK if inside else OUTSIDE)
    if "impact direction" in run:
        directions = np.array([r["impact direction"] for r in runs if "impact direction" in r])
        mean = directions.mean(0)
        mean /= np.linalg.norm(mean)

        def angle(d):
            return math.degrees(math.acos(float(np.clip(np.dot(d, mean), -1.0, 1.0))))
        spread = max(angle(d) for d in directions)
        value = angle(np.asarray(run["impact direction"]))
        add("impact direction (angle to the train runs, deg)", round(value, 2),
            (0.0, round(spread, 2)),
            OK if value <= spread + DIRECTION_TOLERANCE_DEG else OUTSIDE,
            f"mean direction {np.round(mean, 3).tolist()}")
    trained_steps = int(info["num_steps"])
    add("frames", num_steps, (2, trained_steps), OK if num_steps <= trained_steps else OUTSIDE,
        "" if num_steps <= trained_steps else "beyond the time span of the training data")

    # The deck: what the model does not see
    if facts is None or not training.decks:
        add("deck: rigid body masses, Young's modulus, density", "-", "-", UNCHECKED,
            "no OpenRadioss starter deck" if facts is None
            else "the decks of the train runs are not where the export read them")
    else:
        for body in rigid:
            trained = [d["rigid body masses"][body] for d in training.decks
                       if body in d["rigid body masses"]]
            mass = facts["rigid body masses"].get(body)
            if trained and mass is not None:
                reference = _range(trained)
                add(f"deck: mass of {body}", mass, reference,
                    OK if _inside(mass, *reference) else OUTSIDE,
                    "no input of the model: the training data fixes it")
            else:
                add(f"deck: mass of {body}", mass, trained and _range(trained), UNCHECKED,
                    "not stated in the deck")
        for name in ("youngs_modulus", "density"):
            trained = [v for d in training.decks for v in d[name]]
            if trained and facts[name]:
                reference = _range(trained)
                inside = all(_inside(v, *reference) for v in facts[name])
                add(f"deck: {name}", facts[name], reference, OK if inside else OUTSIDE,
                    "no input of the model: the training data fixes it")

    # Parts
    if traj.left_out:
        add("parts left out", traj.left_out, "-", REVIEW,
            "no part pattern of the training data matches them: they are not in the prediction")
    known = {_base_name(name) for name in json.loads(info.get("component_names", "[]"))}
    unseen = sorted({_base_name(name) for name in traj.component_names or []} - known)
    if unseen:
        add("parts the model has not seen", unseen, "-", REVIEW,
            "named differently from the parts of the training data")
    return findings


def verdict(findings: Sequence[Finding]) -> str:
    """``not applicable`` (an error), ``extrapolation`` (outside a range) or ``within the
    training data``, then with parts to review if a part is left out or new."""
    statuses = {f.status for f in findings}
    if ERROR in statuses:
        return "not applicable"
    result = "extrapolation" if OUTSIDE in statuses else "within the training data"
    return result + (", parts to review" if REVIEW in statuses else "")


def _show(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(
            isinstance(v, (int, float)) for v in value):
        return f"{_show(value[0])} .. {_show(value[1])}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_show(v) for v in value) or "none"
    return str(value)


def report(name: str, findings: Sequence[Finding]) -> List[str]:
    """Lines of a table of the findings of one run."""
    lines = [f"{name}: {verdict(findings)}",
             f"  {'check':48s} {'new run':>22s} {'train runs':>24s}  status"]
    for f in findings:
        lines.append(f"  {f.check:48s} {_show(f.value):>22s} {_show(f.reference):>24s}  "
                     f"{f.status}" + (f" — {f.note}" if f.note and f.status != OK else ""))
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def training_data_dir(trained, given: Optional[str] = None) -> Path:
    """The export a model was trained on: ``given``, else the ``data_dir`` of its training
    config, else ``data/<name of the export>`` (a training on another machine)."""
    candidates = [given] if given else [
        trained.config.data_dir,
        (trained.metadata.get("export_config") or {}).get("dataset", {}).get("out_dir"),
        str(Path("data") / trained.metadata.get("dataset", ""))]
    for candidate in filter(None, candidates):
        for path in (Path(candidate), _REPO_ROOT / candidate):
            if (path / "dataset.db").exists():
                return path
    return Path(candidates[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check new FE runs against the training data of a model and prepare "
                    "them for a prediction from their initial state.")
    parser.add_argument("--checkpoint", required=True,
                        help="trained model (best_model.pt or checkpoint_epoch_N.pt)")
    parser.add_argument("--raw-dir", required=True,
                        help="folder with the new runs: d3plot with at least the initial "
                             "state and the deck next to it, one folder per run")
    parser.add_argument("--source", default="",
                        help="subfolder of --raw-dir with the runs (default: every run below "
                             "--raw-dir)")
    parser.add_argument("--out-dir",
                        help="folder of the prediction data (dataset.db, metadata.json, "
                             "normalizer.json, applicability.json); needed unless --check-only")
    parser.add_argument("--train-data-dir",
                        help="export the model was trained on (default: data_dir of its "
                             "training config); its train runs are the reference of the checks")
    parser.add_argument("--frame-interval", type=float,
                        help="time between two frames of the training data in the time unit "
                             "of the decks (default: from the raw data of the training data)")
    parser.add_argument("--num-steps", type=int,
                        help="frames to predict, the initial state included (default: those "
                             "of the training data)")
    parser.add_argument("--check-only", action="store_true",
                        help="only check the runs, write nothing but applicability.json "
                             "(with --out-dir)")
    parser.add_argument("--force", action="store_true",
                        help="prepare runs whose check failed as well")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing dataset.db in --out-dir")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from dataset_preprocessor.d3plot_reader import read_raw_run, run_dirs
    from train.trainer import load_trained_model

    if not args.check_only and not args.out_dir:
        raise SystemExit("prepare.py: --out-dir is needed unless --check-only")
    trained = load_trained_model(args.checkpoint, device="cpu", data_dir=args.train_data_dir)
    train_dir = training_data_dir(trained, args.train_data_dir)
    if not (train_dir / "dataset.db").exists():
        raise SystemExit(f"prepare.py: {train_dir / 'dataset.db'} not found — the checks "
                         "compare with the training data (--train-data-dir)")
    try:
        training, interval = training_reference(train_dir, args.frame_interval)
    except ValueError as error:
        raise SystemExit(f"prepare.py: {error}")
    info = training.info
    if info.get("raw_format") != "d3plot":
        raise SystemExit(f"prepare.py: the training data were read from "
                         f"{info.get('raw_format', 'tfrecord')}; prediction data is "
                         "prepared from d3plot runs only")
    num_steps = args.num_steps or int(info["num_steps"])
    options = json.loads(info.get("d3plot_options", "{}"))
    if options.get("static_fields") is not None:   # the initial velocity, for frame 0
        options["static_fields"] = list(dict.fromkeys(options["static_fields"]
                                                      + ["initial_velocity"]))
    raw_dir = Path(args.raw_dir)
    runs = run_dirs(raw_dir, args.source)
    reach = WALL_DISTANCE_CUTOFF * float(info["contact_radius"])
    print(f"Model {args.checkpoint} (epoch {trained.epoch}), trained on {len(training.runs)} "
          f"train runs of {train_dir}; frame interval {interval:g}, {num_steps} frames")
    results = {}
    for run in runs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traj = read_raw_run(run, [0], **options)
        findings = check(describe_raw(traj, interval, reach), deck_facts(run), traj, training,
                         num_steps)
        name = str(run.relative_to(raw_dir)) if run != raw_dir else run.name
        results[name] = {"verdict": verdict(findings),
                         "findings": [asdict(f) for f in findings]}
        print("\n".join([""] + report(name, findings)))
    applicability = {"checkpoint": str(args.checkpoint), "epoch": trained.epoch,
                     "training_data": str(train_dir), "frame_interval": interval,
                     "runs": results}
    failed = [name for name, result in results.items() if result["verdict"] == "not applicable"]
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "applicability.json", "w", encoding="utf-8") as f:
            json.dump(applicability, f, indent=2, default=str)
    if args.check_only:
        return 1 if failed else 0
    if failed and not args.force:
        print(f"\nNot prepared: the model does not apply to {', '.join(failed)} "
              "(--force prepares them anyway)")
        return 1
    if (Path(args.out_dir) / "dataset.db").exists() and not args.overwrite:
        raise SystemExit(f"prepare.py: {Path(args.out_dir) / 'dataset.db'} exists — "
                         "pass --overwrite to replace it")

    static_names = json.loads(info.get("static_names", "[]"))
    paths = write_dataset(
        raw_dir=str(raw_dir), out_dir=str(args.out_dir), sources=args.source,
        n_train=0, n_val=0, n_test=len(runs), num_steps=num_steps,
        time_stride=int(info.get("time_stride", 1)), history=int(info.get("history", 0)),
        prev_stress=info.get("prev_stress", "1") == "1", raw_format="d3plot",
        contact_radius=float(info["contact_radius"]), d3plot_options=options,
        length_unit=info.get("length_unit", "m"), stress_unit=info.get("stress_unit", "Pa"),
        contact_pairs=json.loads(info.get("contact_pairs", "null")),
        global_features=(json.loads(info["global_features"]) if "global_features" in info
                         else None),
        initial_state=InitialState(interval, static_names,
                                   Normalizer.load(str(train_dir / "normalizer.json")),
                                   json.loads(info.get("edge_static_names", "[]"))),
        contact_gap=info.get("contact_gap") == "1", progress=print)
    with open(paths["metadata"], encoding="utf-8") as f:
        metadata = json.load(f)
    metadata["applicability"] = {name: result["verdict"] for name, result in results.items()}
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nPrediction data: {args.out_dir} ({len(runs)} run(s), split test)\n"
          f"  python predict/rollout.py --checkpoint {args.checkpoint} "
          f"--data-dir {args.out_dir} --split test --export-vtu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
