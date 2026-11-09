#!/usr/bin/env python3
"""
remesh_pole.py — Remesh the pole of BumperBeamCrashExample so that contact edges form.

The solver computes against the analytic cylinder ``/RWALL/CYL``; the shells ``RWALL`` in the
d3plot only display it: two rings at z = ±1012 mm and a node on the axis. No structural node comes
closer to one of these nodes than 127 mm, so at a contact radius of 30 mm not a single contact edge
forms. This script replaces the display by a finely meshed cylinder surface at the same place and
writes a second raw dataset, which ``export.py`` reads like the originals; export it with
``--no-deck-walls``, as the pole now acts through contact edges::

    <out-dir>/conversion.json
    <out-dir>/<source>/<run>/<name>.d3plot, <name>.d3plot01, <name>_0000.rad

The originals stay untouched, the target folder must be empty. The new data are derived, no longer
those of the solver; everything but the pole stays identical bit for bit.

* **Only the region around the structure** (``--half-height``): the surface does not reach to
  ±1012 mm. A long column up to there would bring mesh edges of over 800 mm in z, which then
  determine the normalizer of the edge features: the other edges have a median length of 10 mm.
* **Defaults:** ``--element-size`` like the structure (median 10 mm), ``--half-height`` from the
  reach measured over all 131 runs: structural nodes within the contact radius of the pole stay at
  \\|z\\| ≤ 105 mm, plus the contact radius.
* **Runs without a display:** a conversion that leaves the display models out writes no such
  part, only its nodes without elements. The pole then becomes a new part with this title, which
  must be the title of the ``/RWALL`` in the deck; the nodes without elements stay as they are.
* **Checks per run:** the part lies on the cylinder of the deck, stands still and shares no nodes;
  the structure stays within the meshed region. Then every run is read back with ``read_run`` and
  compared with the original, as in ``tfrecord_to_d3plot.py``.
* **Header version:** VTK's LS-DYNA reader (ParaView) detects word size and byte order from this
  word and rejects version 0.0, which the originals carry. A missing version becomes
  ``D3PLOT_VERSION``, as in ``tfrecord_to_d3plot.py``; a valid one stays.

Usage::

    python utils/remesh_pole.py --out-dir datasets/BumperBeamCrashExample_remeshed/RAW_DATA
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from dataset_preprocessor.d3plot_reader import d3plot_path, read_run, run_dirs  # noqa: E402
from dataset_preprocessor.graph_features import NODE_TYPE_ACTUATOR  # noqa: E402
from dataset_preprocessor.radioss_deck import read_deck, starter_deck  # noqa: E402
from dataset_preprocessor.tfrecord_to_d3plot import D3PLOT_VERSION  # noqa: E402

DEFAULTS = dict(
    raw_dir="datasets/BumperBeamCrashExample/RAW_DATA",
    sources=["TRAINING_DATA", "VALIDATION_DATA"],
    part="RWALL",
    element_size=10.0,    # mm
    half_height=140.0,    # mm
    contact_radius=30.0,  # mm, only for the check of the reach
)

_POSITION_TOLERANCE = 1.0   # mm, position of the old display relative to the cylinder of the deck
_STATIC_SHELL_ARRAYS = ("element_shell_ids", "element_shell_node_indexes",
                        "element_shell_part_indexes")
_NODE_ARRAYS = ("node_coordinates", "node_displacement", "node_ids")
_PART_TABLES = ("part_ids", "part_ids_unordered", "part_ids_cross_references", "part_titles_ids",
                "part_titles")
_PART_RESULTS = ("part_hourglass_energy", "part_internal_energy", "part_kinetic_energy",
                 "part_mass", "part_velocity")   # per state and part
# NaN, not 0: the reader takes it as "no value"; otherwise the pole nodes count in the statistics.
_NO_VALUE_ARRAYS = ("element_shell_stress", "element_shell_effective_plastic_strain")


def cylinder_mesh(wall: np.ndarray, element_size: float,
                  half_height: float) -> Tuple[np.ndarray, np.ndarray]:
    """Surface of the cylinder ``wall`` (point, axis in z, radius): nodes (R * C, 3) and
    outward oriented quadrilaterals (Q, 4), centred on the point on the axis."""
    if not np.allclose(wall[3:6], (0, 0, 1)):
        raise ValueError(f"cylinder axis {wall[3:6].tolist()}: only z is provided for")
    x, y, z, radius = (float(wall[i]) for i in (0, 1, 2, 6))
    columns = round(2 * np.pi * radius / element_size)
    rows = round(2 * half_height / element_size)
    height, angle = np.meshgrid(z + np.linspace(-half_height, half_height, rows + 1),
                                2 * np.pi * np.arange(columns) / columns, indexing="ij")
    points = np.stack((x + radius * np.cos(angle), y + radius * np.sin(angle), height), axis=-1)
    column = np.arange(columns)
    following = np.roll(column, -1)
    ring = np.arange(rows)[:, None] * columns
    quads = np.stack((ring + column, ring + following, ring + columns + following,
                      ring + columns + column), axis=-1)
    return points.reshape(-1, 3), quads.reshape(-1, 4)


def _part_index(arrays: Dict[str, np.ndarray], title: str) -> Optional[int]:
    """Index of the part with this title, None if the run has none. By title, as the order
    changes between the runs and several parts have the same ID."""
    titles = [bytes(name).decode(errors="replace").strip() for name in arrays["part_titles"]]
    if titles.count(title) > 1:
        raise ValueError(f"part {title!r} occurs {titles.count(title)} times: {titles}")
    return titles.index(title) if title in titles else None


def _add_part(arrays: Dict[str, np.ndarray], title: str) -> int:
    """Append the part ``title`` to the part tables of the lasso arrays, with IDs above all
    others and zero results per state; returns its index."""
    unknown = [key for key in arrays
               if key.startswith("part_") and key not in _PART_TABLES + _PART_RESULTS]
    if unknown:
        raise ValueError(f"part fields without a rule for the new part: {unknown}")
    new_id = max(arrays[key].max() for key in ("part_ids", "part_titles_ids") if key in arrays) + 1
    titles = arrays["part_titles"]
    values = {"part_ids": new_id, "part_ids_unordered": new_id, "part_titles_ids": new_id,
              "part_titles": title.encode().ljust(titles.dtype.itemsize)}
    if "part_ids_cross_references" in arrays:
        values["part_ids_cross_references"] = arrays["part_ids_cross_references"].max() + 1
    for key in [key for key in _PART_TABLES if key in arrays]:
        arrays[key] = np.append(arrays[key], np.array([values[key]], dtype=arrays[key].dtype))
    for key in [key for key in _PART_RESULTS if key in arrays]:
        zeros = np.zeros((arrays[key].shape[0], 1) + arrays[key].shape[2:], arrays[key].dtype)
        arrays[key] = np.concatenate((arrays[key], zeros), axis=1)
    return len(titles)


def _check_reach(world_pos: np.ndarray, wall: np.ndarray, half_height: float,
                 contact_radius: float) -> None:
    """Every structural node within the contact radius of the pole must find all its possible
    partners in the meshed region."""
    distance = np.hypot(world_pos[..., 0] - wall[0], world_pos[..., 1] - wall[1]) - wall[6]
    height = np.abs(world_pos[..., 2] - wall[2])[distance < contact_radius]
    if height.size and height.max() + contact_radius > half_height:
        raise ValueError(f"structure at |z| = {height.max():.1f} mm within the contact radius "
                         f"{contact_radius:g} mm: --half-height {half_height:g} is not enough")


def _replaced(array: np.ndarray, keep: np.ndarray, added: np.ndarray, axis: int) -> np.ndarray:
    """``array`` with the entries ``keep`` on ``axis``, ``added`` appended at the end."""
    return np.concatenate((np.compress(keep, array, axis=axis), added), axis=axis)


def replace_pole(arrays: Dict[str, np.ndarray], part: str, wall: np.ndarray, element_size: float,
                 half_height: float, contact_radius: float) -> Dict[str, np.ndarray]:
    """Replace the shells and nodes of ``part`` in the lasso arrays of a run (in place) by the
    surface from ``cylinder_mesh``; without such a part, add the surface as a new part. The
    other nodes keep their order and IDs; the new ones come last, with IDs above all existing
    ones, so the references of the deck stay valid. Returns ``kept_nodes``, the numbers of new
    nodes and shells and whether a display was replaced."""
    unknown = [key for key in arrays if key.startswith("node_") and key not in _NODE_ARRAYS]
    if unknown:
        raise ValueError(f"node fields without a rule for the new nodes: {unknown}")
    index = _part_index(arrays, part)
    shells = arrays["element_shell_node_indexes"]
    keep_shells = (np.ones(len(shells), dtype=bool) if index is None
                   else arrays["element_shell_part_indexes"] != index)
    if index is not None and keep_shells.all():
        raise ValueError(f"part {part!r} has no shells")
    nodes = np.unique(shells[~keep_shells])

    connectivity = [key for key in arrays
                    if key.startswith("element_") and key.endswith("_node_indexes")]
    others = [shells[keep_shells] if key == "element_shell_node_indexes" else arrays[key]
              for key in connectivity]
    if any(np.isin(other, nodes).any() for other in others):
        raise ValueError(f"part {part!r} shares nodes with other elements")

    coordinates, world_pos = arrays["node_coordinates"], arrays["node_displacement"]
    if index is not None:   # the display must show this pole
        if np.ptp(world_pos[:, nodes], axis=0).max() > _POSITION_TOLERANCE:
            raise ValueError(f"part {part!r} moves, but /RWALL/CYL stands still")
        axis_distance = np.hypot(coordinates[nodes, 0] - wall[0], coordinates[nodes, 1] - wall[1])
        if not np.all((axis_distance < _POSITION_TOLERANCE)
                      | (np.abs(axis_distance - wall[6]) < _POSITION_TOLERANCE)):
            raise ValueError(f"part {part!r} does not lie on the cylinder of the deck")

    keep = np.ones(coordinates.shape[0], dtype=bool)
    keep[nodes] = False
    held = np.zeros_like(keep)   # nodes of the other elements, of beams their two end nodes
    for key, other in zip(connectivity, others):
        held[other[:, :2] if key == "element_beam_node_indexes" else other] = True
    _check_reach(world_pos[:, held], wall, half_height, contact_radius)
    points, quads = cylinder_mesh(wall, element_size, half_height)
    num_states, num_kept = world_pos.shape[0], int(keep.sum())
    replaced = index is not None
    if not replaced:
        index = _add_part(arrays, part)

    old_to_new = np.cumsum(keep) - 1
    for key in connectivity:
        arrays[key] = old_to_new[arrays[key]]
    arrays["node_coordinates"] = _replaced(coordinates, keep, points, axis=0)
    arrays["node_ids"] = _replaced(arrays["node_ids"], keep,
                                   arrays["node_ids"].max() + 1 + np.arange(len(points)), axis=0)
    arrays["node_displacement"] = _replaced(
        world_pos, keep, np.broadcast_to(points, (num_states,) + points.shape), axis=1)

    added = {"element_shell_ids": arrays["element_shell_ids"].max() + 1 + np.arange(len(quads)),
             "element_shell_node_indexes": num_kept + quads,
             "element_shell_part_indexes": np.full(len(quads), index, dtype=np.int64)}
    for key in [key for key in arrays if key.startswith("element_shell_")]:
        if key in added:
            arrays[key] = _replaced(arrays[key], keep_shells, added[key], axis=0)
        else:   # results per state: the shells are intact, a rigid pole has no stress
            fill = (1.0 if key == "element_shell_is_alive"
                    else np.nan if key in _NO_VALUE_ARRAYS else 0.0)
            new = np.full((num_states, len(quads)) + arrays[key].shape[2:], fill)
            arrays[key] = _replaced(arrays[key], keep_shells, new, axis=1)
    return {"kept_nodes": keep, "pole_nodes": len(points), "pole_shells": len(quads),
            "display_replaced": replaced}


def verify_run(source: Path, converted: Path, kept_nodes: np.ndarray, wall: np.ndarray) -> None:
    """Read the written run back with the reader of the pipeline: all other nodes equal the
    original exactly, the new ones stand still on the cylinder."""
    original, written = read_run(source), read_run(converted)
    num_kept = int(kept_nodes.sum())
    for name in ("world_pos", "von_mises"):
        before, after = getattr(original, name), getattr(written, name)
        if not np.array_equal(before[:, kept_nodes], after[:, :num_kept], equal_nan=True):
            raise ValueError(f"{converted}: {name} differs from the original")   # NaN: no value
    for name in ("mesh_pos", "node_ids", "node_type"):
        if not np.array_equal(getattr(original, name)[kept_nodes],
                              getattr(written, name)[:num_kept]):
            raise ValueError(f"{converted}: {name} differs from the original")
    if not np.array_equal(original.times, written.times):
        raise ValueError(f"{converted}: the times differ from the original")
    pole = written.world_pos[:, num_kept:]
    radial = np.hypot(pole[..., 0] - wall[0], pole[..., 1] - wall[1])
    if np.abs(radial - wall[6]).max() > 1e-2 or np.ptp(pole, axis=0).max() > 0:
        raise ValueError(f"{converted}: the pole does not stand still on the cylinder")
    if not (written.node_type[num_kept:] == NODE_TYPE_ACTUATOR).all():
        raise ValueError(f"{converted}: the pole nodes are no actuators")
    if not np.isnan(written.von_mises[:, num_kept:]).all():
        raise ValueError(f"{converted}: the pole nodes have a stress")


def convert_run(run_dir: Path, out_dir: Path, part: str, element_size: float, half_height: float,
                contact_radius: float) -> Dict[str, int]:
    """Write a run to ``out_dir``: d3plot with the new pole, the starter deck unchanged."""
    from lasso.dyna import D3plot

    deck = read_deck(starter_deck(run_dir))
    walls = deck.walls
    if len(walls) != 1:
        raise ValueError(f"{run_dir}: {len(walls)} walls in the deck, expected one")
    if part not in deck.rigid_titles:   # else the reader would not take the pole as rigid
        raise ValueError(f"{run_dir}: {part!r} is not the title of a /RWALL or /RBODY in the "
                         f"deck ({sorted(deck.rigid_titles)})")
    source = d3plot_path(run_dir)
    d3plot = D3plot(str(source))
    numbers = replace_pole(d3plot.arrays, part, walls[0], element_size, half_height,
                           contact_radius)
    out_dir.mkdir(parents=True)
    d3plot.header.version = d3plot.header.version or D3PLOT_VERSION
    d3plot.write_d3plot(str(out_dir / source.name))
    shutil.copy2(starter_deck(run_dir), out_dir)
    verify_run(run_dir, out_dir, numbers.pop("kept_nodes"), walls[0])
    return numbers


def convert(raw_dir: str, out_dir: str, sources: Sequence[str], part: str = DEFAULTS["part"],
            element_size: float = DEFAULTS["element_size"],
            half_height: float = DEFAULTS["half_height"],
            contact_radius: float = DEFAULTS["contact_radius"],
            count: Optional[int] = None) -> dict:
    """Convert the first ``count`` runs of every source (None: all)."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: nothing is overwritten")
    runs = [(source, run) for source in sources for run in run_dirs(raw_dir, source)[:count]]
    numbers = {}
    for k, (source, run) in enumerate(runs, start=1):
        numbers = convert_run(run, out / source / run.name, part, element_size, half_height,
                              contact_radius)
        print(f"[{k}/{len(runs)}] {source}/{run.name}: {numbers['pole_nodes']:,} pole nodes, "
              f"{numbers['pole_shells']:,} shells"
              f"{'' if numbers['display_replaced'] else ', as a new part'}", flush=True)
    record = dict(source=str(raw_dir), sources=list(sources), runs=len(runs), part=part,
                  element_size=element_size, half_height=half_height,
                  contact_radius=contact_radius, lasso_python=version("lasso-python"), **numbers)
    (out / "conversion.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--raw-dir", default=DEFAULTS["raw_dir"],
                        help="folder of the originals, with the sources")
    parser.add_argument("--out-dir", required=True, help="target folder, empty or not there yet")
    parser.add_argument("--sources", nargs="+", default=DEFAULTS["sources"],
                        help="subfolders of --raw-dir")
    parser.add_argument("--part", default=DEFAULTS["part"],
                        help="title of the part that displays the pole, that of the /RWALL in "
                             "the deck; a run without it gets a new part of this title")
    parser.add_argument("--element-size", type=float, default=DEFAULTS["element_size"],
                        help="edge length of the new shells in the unit of the data (mm)")
    parser.add_argument("--half-height", type=float, default=DEFAULTS["half_height"],
                        help="meshed region: this far to both sides of the point on the axis")
    parser.add_argument("--contact-radius", type=float, default=DEFAULTS["contact_radius"],
                        help="contact radius of the export, only for the check of the reach")
    parser.add_argument("--count", type=int, default=None,
                        help="only the first N runs per source; default: all")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    record = convert(args.raw_dir, args.out_dir, args.sources, args.part, args.element_size,
                     args.half_height, args.contact_radius, args.count)
    print(f"Done: {record['runs']} runs -> {args.out_dir}")


if __name__ == "__main__":
    main()
