"""
tfrecord_to_vtk.py — Convert DeepMind deforming_plate TFRecords into VTK time series.

Writes every trajectory of a TFRecord with all its frames as a VTK time
series, one file per state (the ParaView convention): a third raw format of
the same data, which ``export.py --raw-format vtk`` reads through
``vtk_reader.py`` into the usual ``dataset.db``. Unlike d3plot, VTK keeps the
stress per node, so the export is identical to the TFRecord export. Layout::

    <out-dir>/meta.json                           collision_radius, trajectory_length, ...
    <out-dir>/<tfrecord name>/conversion.json
    <out-dir>/<tfrecord name>/traj<k>/traj<k>.vtu.series   order and times (ParaView)
    <out-dir>/<tfrecord name>/traj<k>/t0000.vtu ...         one state per file

Per state: the points are the world positions of the frame, the cells the
tetrahedra; point data ``node_type`` (raw types 0 normal, 1 actuator,
3 clamped) and ``stress`` (von Mises); field data ``time`` = frame index
(quasi-static, no time step). ``--format vtk`` writes legacy ``.vtk`` files
instead of XML ``.vtu`` (zlib-compressed). A state takes ~60 kB as .vtu, ~190 kB as .vtk.

Every written run is read back with ``vtk_reader.read_run`` and compared with
the TFRecord. DeepMind states no license for the datasets (see docs/data.md):
keep the converted files local.

Usage (without arguments all 100 trajectories of test.tfrecord)::

    python -m dataset_preprocessor.tfrecord_to_vtk [--source-tfrecord F] [--count N] [--format vtu|vtk]

Needs ``pyvista``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):   # executed as a file, e.g. "Run Python File"
    sys.path.insert(0, str(_REPO_ROOT))

from dataset_preprocessor.export import (  # noqa: E402
    EXPORT_DEFAULTS,
    RawTrajectory,
    _repo_path,
    read_raw_meta,
    read_raw_trajectories,
)
from dataset_preprocessor.graph_features import CELL_TETRA  # noqa: E402
from dataset_preprocessor.tfrecord_to_d3plot import _SIMULATION_META  # noqa: E402
from dataset_preprocessor.vtk_reader import read_run  # noqa: E402

FORMATS = ("vtu", "vtk")
VTK_DEFAULTS = dict(
    raw_dir=EXPORT_DEFAULTS["raw_dir"],
    source_tfrecord=EXPORT_DEFAULTS["source_tfrecord"],
    out_dir=EXPORT_DEFAULTS["raw_dir_vtk"],
    count=None,                                  # None: every trajectory of the TFRecord
    file_format="vtu",
    overwrite=True,
)


def write_run(run_dir: Path, traj: RawTrajectory, file_format: str = "vtu") -> None:
    """One state file per frame plus ``<run>.<format>.series``."""
    import pyvista as pv

    cells = np.asarray(traj.cells, dtype=np.int64)
    connectivity = np.hstack([np.full((len(cells), 1), 4, dtype=np.int64), cells]).ravel()
    grid = pv.UnstructuredGrid(connectivity, np.full(len(cells), CELL_TETRA, dtype=np.uint8),
                               traj.world_pos[0].astype(np.float32))
    grid.point_data["node_type"] = traj.node_type.astype(np.uint8)
    files = []
    for frame in range(traj.world_pos.shape[0]):
        grid.points = traj.world_pos[frame].astype(np.float32)
        grid.point_data["stress"] = traj.stress[frame].astype(np.float32)
        grid.field_data["time"] = np.array([float(frame)])
        name = f"t{frame:04d}.{file_format}"
        grid.save(str(run_dir / name))
        files.append({"name": name, "time": float(frame)})
    with open(run_dir / f"{run_dir.name}.{file_format}.series", "w") as f:
        json.dump({"file-series-version": "1.0", "files": files}, f, indent=1)


def verify_run(run_dir: Path, traj: RawTrajectory) -> None:
    """Read the run back with ``vtk_reader``; raises ValueError on a mismatch."""
    read = read_run(run_dir)
    checks = {
        "rest coordinates": np.array_equal(read.mesh_pos, traj.mesh_pos),
        "state coordinates": np.array_equal(read.world_pos, traj.world_pos),
        "tetrahedra": read.cell_types is None and np.array_equal(read.cells, traj.cells),
        "node types": np.array_equal(read.node_type, traj.node_type),
        "stress": np.array_equal(read.stress, traj.stress),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"{run_dir}: read-back differs in {', '.join(failed)}")


def convert(raw_dir: str, out_dir: str, source_tfrecord: str, count: Optional[int] = None,
            file_format: str = "vtu") -> List[Path]:
    """Write the first ``count`` trajectories (None: all) with all frames as VTK time series."""
    meta = read_raw_meta(raw_dir)
    frames = range(meta["trajectory_length"])
    out = Path(out_dir)
    source_dir = out / Path(source_tfrecord).stem
    source_dir.mkdir(parents=True, exist_ok=True)
    with open(out / "meta.json", "w") as f:
        json.dump({"raw_format": "vtk",
                   **{key: meta[key] for key in _SIMULATION_META if key in meta}}, f, indent=2)

    runs = []
    start = time.time()
    for index, traj in enumerate(read_raw_trajectories(raw_dir, source_tfrecord, frames, count)):
        run_dir = source_dir / f"traj{index:04d}"
        run_dir.mkdir(exist_ok=True)
        for old in run_dir.iterdir():   # a longer earlier run leaves more state files
            old.unlink()
        write_run(run_dir, traj, file_format)
        verify_run(run_dir, traj)
        runs.append(run_dir)
        print(f"  {source_dir.name}/{run_dir.name}: {traj.mesh_pos.shape[0]} nodes, "
              f"{traj.cells.shape[0]} tetrahedra, {traj.world_pos.shape[0]} states — "
              f"read-back OK ({time.time() - start:.0f} s)")

    conversion = {
        "source": {"raw_dir": str(raw_dir), "tfrecord": source_tfrecord,
                   "trajectories": len(runs)},
        "format": file_format,
        "states": "one file per frame; points = world positions; time = frame index",
        "point_data": {"node_type": "raw node type: 0 normal, 1 actuator, 3 clamped",
                       "stress": "von Mises stress per node"},
    }
    with open(source_dir / "conversion.json", "w") as f:
        json.dump(conversion, f, indent=2)
    return runs


def main(argv=None) -> None:
    """Convert a TFRecord; defaults come from ``VTK_DEFAULTS``."""
    d = VTK_DEFAULTS
    parser = argparse.ArgumentParser(
        description="Convert DeepMind deforming_plate trajectories into VTK time series "
                    "(one file per state, all frames): the raw data for "
                    "export.py --raw-format vtk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", default=d["raw_dir"],
                        help="directory with the .tfrecord and meta.json")
    parser.add_argument("--source-tfrecord", default=d["source_tfrecord"],
                        help="TFRecord inside --raw-dir (test.tfrecord: 100 trajectories)")
    parser.add_argument("--out-dir", default=d["out_dir"],
                        help="raw VTK directory; runs go to <out-dir>/<tfrecord name>/")
    parser.add_argument("--count", type=int, default=d["count"],
                        help="first N trajectories (default: all)")
    parser.add_argument("--format", dest="file_format", choices=FORMATS,
                        default=d["file_format"], help="XML .vtu or legacy .vtk files")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction,
                        default=d["overwrite"], help="replace existing runs")
    args = parser.parse_args(argv)

    try:
        import pyvista  # noqa: F401
    except ImportError:
        parser.error("pyvista is required: pip install pyvista")
    if args.count is not None and args.count <= 0:
        parser.error("count must be > 0")
    raw_dir, out_dir = _repo_path(args.raw_dir), _repo_path(args.out_dir)
    source_dir = out_dir / Path(args.source_tfrecord).stem
    if (source_dir / "traj0000").exists() and not args.overwrite:
        parser.error(f"{source_dir} already holds runs — pass --overwrite to replace them")

    which = "all" if args.count is None else f"the first {args.count}"
    print(f"Converting {which} trajectories of {args.source_tfrecord} (all frames, "
          f".{args.file_format}) -> {source_dir}")
    start = time.time()
    runs = convert(str(raw_dir), str(out_dir), args.source_tfrecord, args.count, args.file_format)
    size = sum(f.stat().st_size for run in runs for f in run.iterdir())
    print(f"Done in {time.time() - start:.0f} s: {len(runs)} runs, {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
