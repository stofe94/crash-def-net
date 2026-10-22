"""
tfrecord_to_d3plot.py — Convert DeepMind deforming_plate TFRecords into LS-DYNA runs.

Writes every trajectory of a TFRecord with all its frames as the files of an
LS-DYNA run: a second raw format of the same data, which ``export.py
--raw-format d3plot`` reads through ``d3plot_reader.py`` into the usual
``dataset.db``. Only the format changes: the content stays the deforming_plate
benchmark (quasi-static hyperelastic plate, tetrahedra). Layout::

    <out-dir>/meta.json                    collision_radius, trajectory_length, ...
    <out-dir>/<tfrecord name>/conversion.json
    <out-dir>/<tfrecord name>/traj<k>/     k = index in the TFRecord

Per run:

``d3plot``, ``d3plot01``
    Geometry and states, written with lasso-python.

    * Tetrahedra become 8-node solids in the LS-DYNA degenerate numbering
      n1 n2 n3 n4 n4 n4 n4 n4. Part 1 is the plate, part 2 the actuator.
    * The state coordinates are the world positions of all frames (d3plot
      stores current coordinates; lasso calls the array ``node_displacement``).
      The time of a state is its frame index, as the simulation is
      quasi-static and has no time step.
    * d3plot keeps stress tensors per element, the data a von Mises stress
      per node. Each element gets sigma = diag(s, 0, 0), s = mean nodal von
      Mises stress of its four nodes, so the von Mises stress of the tensor
      is exactly s. The 7 values per solid (stress + plastic strain) are
      always part of the format and make up ~90 % of a state (~130 kB per
      state, ~50 MB per run of 400 frames).
``model.k``
    Keyword deck with what a d3plot does not contain: the parts, ``*MAT_RIGID``
    for the actuator and ``*SET_NODE_LIST`` + ``*BOUNDARY_SPC_SET`` for the
    clamped nodes. Not a runnable simulation input: the material data are
    unknown and left as placeholders.

Every written run is read back with ``d3plot_reader.read_run`` and compared
with the TFRecord: geometry, positions and node types exactly, the stress
with the element representation.

DeepMind states no license for the datasets (see docs/data.md): keep the
converted files local.

Usage (without arguments all 100 trajectories of test.tfrecord, ~5 GB)::

    python -m dataset_preprocessor.tfrecord_to_d3plot [--source-tfrecord F] [--count N]

Needs ``lasso-python`` (``pip install lasso-python``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):   # executed as a file, e.g. "Run Python File"
    sys.path.insert(0, str(_REPO_ROOT))

from dataset_preprocessor.d3plot_reader import as_raw_trajectory, nodal_mean, read_run  # noqa: E402
from dataset_preprocessor.export import (  # noqa: E402
    EXPORT_DEFAULTS,
    RawTrajectory,
    _repo_path,
    read_raw_meta,
    read_raw_trajectories,
)
from dataset_preprocessor.graph_features import NODE_TYPE_ACTUATOR, NODE_TYPE_CLAMPED  # noqa: E402

# Settings used when the module is run without arguments.
D3PLOT_DEFAULTS = dict(
    raw_dir=EXPORT_DEFAULTS["raw_dir"],
    source_tfrecord=EXPORT_DEFAULTS["source_tfrecord"],
    out_dir=EXPORT_DEFAULTS["raw_dir_d3plot"],
    count=None,                                  # None: every trajectory of the TFRecord
    overwrite=True,
)

PART_PLATE, PART_ACTUATOR = 1, 2                 # LS-DYNA part IDs (part index + 1)
CLAMPED_NODE_SET = 1                             # *SET_NODE_LIST ID
SOLID_FORMULATION_TETRAHEDRON = 10               # *SECTION_SOLID ELFORM: 1-point tetrahedron
D3PLOT_VERSION = 960.0                           # header word 14 as LS-DYNA writes it
# Fields of DeepMind's meta.json that describe the simulation, not the TFRecord.
_SIMULATION_META = ("simulator", "dt", "collision_radius", "trajectory_length")


# ─────────────────────────────────────────────────────────────────────────────
# Mapping of the raw data
# ─────────────────────────────────────────────────────────────────────────────

def tetrahedra_to_solids(cells: np.ndarray) -> np.ndarray:
    """(C, 4) tetrahedra -> (C, 8) LS-DYNA solids n1 n2 n3 n4 n4 n4 n4 n4."""
    cells = np.asarray(cells, dtype=np.int64)
    return np.concatenate((cells, np.repeat(cells[:, 3:4], 4, axis=1)), axis=1)


def solid_part_indexes(cells: np.ndarray, node_type: np.ndarray) -> np.ndarray:
    """Part index per tetrahedron: 0 = plate, 1 = actuator (all four nodes actuator)."""
    actuator_nodes = (node_type == NODE_TYPE_ACTUATOR)[cells]
    mixed = actuator_nodes.any(1) & ~actuator_nodes.all(1)
    if mixed.any():
        raise ValueError(f"{int(mixed.sum())} tetrahedra join plate and actuator nodes")
    return actuator_nodes.all(1).astype(np.int64)


def element_mean(stress: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """(T, N) nodal von Mises stress -> (T, C) mean over the nodes of each tetrahedron."""
    return stress[:, cells].mean(axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Writing
# ─────────────────────────────────────────────────────────────────────────────

def write_d3plot(path: Path, traj: RawTrajectory) -> None:
    """Write geometry and states of ``traj`` as the d3plot family ``path``, ``path``01, ..."""
    from lasso.dyna import ArrayType, D3plot

    cells = np.asarray(traj.cells, dtype=np.int64)
    von_mises = element_mean(traj.stress, cells)                         # (T, C)
    stress = np.zeros(von_mises.shape + (1, 6))                          # (T, C, layer, 6)
    stress[..., 0, 0] = von_mises
    d3plot = D3plot()
    arrays = d3plot.arrays
    arrays[ArrayType.node_coordinates] = traj.mesh_pos.astype(np.float64)
    arrays[ArrayType.node_displacement] = traj.world_pos.astype(np.float64)
    arrays[ArrayType.element_solid_node_indexes] = tetrahedra_to_solids(cells)
    arrays[ArrayType.element_solid_part_indexes] = solid_part_indexes(cells, traj.node_type)
    arrays[ArrayType.global_timesteps] = np.arange(traj.world_pos.shape[0], dtype=np.float64)
    arrays[ArrayType.element_solid_stress] = stress
    arrays[ArrayType.element_solid_effective_plastic_strain] = np.zeros(von_mises.shape + (1,))
    # lasso writes version 0.0; VTK's LS-DYNA reader (ParaView) detects word size and
    # byte order from this word and rejects the file without a valid version.
    d3plot.header.version = D3PLOT_VERSION
    d3plot.write_d3plot(str(path))


def _card(*fields) -> str:
    """One keyword card: every field right-aligned in 10 characters."""
    return "".join(f"{field:>10}" for field in fields)


def write_keyword_file(path: Path, traj: RawTrajectory, source: str) -> None:
    """``model.k`` with parts, rigid actuator and clamped nodes (1-based node IDs)."""
    clamped = np.flatnonzero(traj.node_type == NODE_TYPE_CLAMPED) + 1
    lines = [
        "*KEYWORD",
        f"$ CrashDefNet: {source}, converted from DeepMind's MeshGraphNets data.",
        "$ Carries what the d3plot does not store: parts, rigid actuator, clamped nodes.",
        "$ Not a runnable simulation input: material data are unknown (placeholders).",
        "*PART",
        "plate",
        _card(PART_PLATE, 1, 1),
        "*PART",
        "actuator (rigid, prescribed motion)",
        _card(PART_ACTUATOR, 1, 2),
        "*SECTION_SOLID",
        _card(1, SOLID_FORMULATION_TETRAHEDRON),
        "*MAT_ELASTIC",
        "$ placeholder: the plate is hyperelastic, its parameters are not published",
        _card(1, 0.0, 0.0, 0.0),
        "*MAT_RIGID",
        _card(2, 0.0, 0.0, 0.0),
        _card(0.0, 0, 0),
        _card(0.0),
        "*SET_NODE_LIST",
        "$ clamped nodes",
        _card(CLAMPED_NODE_SET),
        *(_card(*clamped[i:i + 8]) for i in range(0, len(clamped), 8)),
        "*BOUNDARY_SPC_SET",
        _card(CLAMPED_NODE_SET, 0, 1, 1, 1, 1, 1, 1),
        "*END",
    ]
    path.write_text("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Check
# ─────────────────────────────────────────────────────────────────────────────

def verify_run(run_dir: Path, traj: RawTrajectory) -> None:
    """Read the run back with ``d3plot_reader``; raises ValueError on a mismatch."""
    read = as_raw_trajectory(read_run(run_dir), str(run_dir))
    cells = np.asarray(traj.cells, dtype=np.int64)
    expected_stress = nodal_mean(element_mean(traj.stress, cells), cells, traj.mesh_pos.shape[0])
    checks = {
        "rest coordinates": np.array_equal(read.mesh_pos, traj.mesh_pos),
        "state coordinates": np.array_equal(read.world_pos, traj.world_pos),
        "tetrahedra": np.array_equal(read.cells, traj.cells),
        "node types": np.array_equal(read.node_type, traj.node_type),
        "stress": np.allclose(read.stress, expected_stress, rtol=1e-5, atol=1e-3),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{run_dir}: read-back differs in {', '.join(failed)}")


# ─────────────────────────────────────────────────────────────────────────────
# Conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert(raw_dir: str, out_dir: str, source_tfrecord: str,
            count: Optional[int] = None) -> List[Path]:
    """Write the first ``count`` trajectories (None: all) with all frames as runs."""
    meta = read_raw_meta(raw_dir)
    frames: Sequence[int] = range(meta["trajectory_length"])
    out = Path(out_dir)
    source_dir = out / Path(source_tfrecord).stem
    source_dir.mkdir(parents=True, exist_ok=True)
    with open(out / "meta.json", "w") as f:
        json.dump({"raw_format": "d3plot",
                   **{key: meta[key] for key in _SIMULATION_META if key in meta}}, f, indent=2)

    runs = []
    start = time.time()
    for index, traj in enumerate(read_raw_trajectories(raw_dir, source_tfrecord, frames, count)):
        run_dir = source_dir / f"traj{index:04d}"
        run_dir.mkdir(exist_ok=True)
        for old in run_dir.glob("d3plot*"):   # a longer earlier run leaves more state files
            old.unlink()
        write_d3plot(run_dir / "d3plot", traj)
        write_keyword_file(run_dir / "model.k", traj, f"{source_tfrecord} trajectory {index}")
        verify_run(run_dir, traj)
        runs.append(run_dir)
        print(f"  {source_dir.name}/{run_dir.name}: {traj.mesh_pos.shape[0]} nodes, "
              f"{traj.cells.shape[0]} tetrahedra, {traj.world_pos.shape[0]} states — "
              f"read-back OK ({time.time() - start:.0f} s)")

    conversion = {
        "source": {"raw_dir": str(raw_dir), "tfrecord": source_tfrecord,
                   "trajectories": len(runs)},
        "states": "all frames; state time = frame index (quasi-static, no time step)",
        "parts": {str(PART_PLATE): "plate", str(PART_ACTUATOR): "actuator (rigid)"},
        "node_types": {"actuator": f"nodes of part {PART_ACTUATOR} (*MAT_RIGID)",
                       "clamped": f"model.k *SET_NODE_LIST {CLAMPED_NODE_SET} "
                                  "(*BOUNDARY_SPC_SET)",
                       "normal": "all other nodes"},
        "stress": ("per element sigma = diag(s, 0, 0), s = mean nodal von Mises stress "
                   "of the element's nodes"),
        "lasso_python": version("lasso-python"),
    }
    with open(source_dir / "conversion.json", "w") as f:
        json.dump(conversion, f, indent=2)
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    """Convert a TFRecord; defaults come from ``D3PLOT_DEFAULTS``."""
    d = D3PLOT_DEFAULTS
    parser = argparse.ArgumentParser(
        description="Convert DeepMind deforming_plate trajectories into LS-DYNA runs "
                    "(d3plot + model.k, all frames): the raw data for "
                    "export.py --raw-format d3plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", default=d["raw_dir"],
                        help="directory with the .tfrecord and meta.json")
    parser.add_argument("--source-tfrecord", default=d["source_tfrecord"],
                        help="TFRecord inside --raw-dir (test.tfrecord: 100 trajectories)")
    parser.add_argument("--out-dir", default=d["out_dir"],
                        help="raw d3plot directory; runs go to <out-dir>/<tfrecord name>/")
    parser.add_argument("--count", type=int, default=d["count"],
                        help="first N trajectories (default: all)")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction,
                        default=d["overwrite"], help="replace existing runs")
    args = parser.parse_args(argv)

    try:
        version("lasso-python")
    except Exception:
        parser.error("lasso-python is required: pip install lasso-python")
    if args.count is not None and args.count <= 0:
        parser.error("count must be > 0")
    raw_dir, out_dir = _repo_path(args.raw_dir), _repo_path(args.out_dir)
    source_dir = out_dir / Path(args.source_tfrecord).stem
    if (source_dir / "traj0000").exists() and not args.overwrite:
        parser.error(f"{source_dir} already holds runs — pass --overwrite to replace them")

    which = "all" if args.count is None else f"the first {args.count}"
    print(f"Converting {which} trajectories of {args.source_tfrecord} (all frames) "
          f"-> {source_dir}")
    start = time.time()
    runs = convert(str(raw_dir), str(out_dir), args.source_tfrecord, args.count)
    size = sum(f.stat().st_size for run in runs for f in run.iterdir())
    print(f"Done in {time.time() - start:.0f} s: {len(runs)} runs, {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
