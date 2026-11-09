#!/usr/bin/env python3
"""
animate_rollout.py — Animated GIF of one rollout trajectory: ground truth next to
the predictions of one or more models.

Reads rollout folders written by ``predict/rollout.py`` with the ``.vtu`` export
(``<split>/<trajectory>/tNNNN.vtu``) and renders one trajectory side by side: the
ground truth (FEM) of the first rollout, then the prediction of every rollout,
coloured by von Mises stress on one scale (datasets without a stress: by their
first result field). The actuator (node type 1, read from
the rollout's ``dataset.db``) is grey; the lines that tie a free rigid body
together (export ``--free-rigid-parts``) are not drawn. All rollouts must come from
the same dataset and cover the same steps. Writes ``rollout_<trajectory>.gif`` to a new
folder ``evaluations/<YYYYMMDD_HHMMSS>_animation/``, or to ``--output``.

Rendering runs off-screen with Mesa's software renderer, so the frames are the
same on every machine and need neither a GPU nor a display (set
``LIBGL_ALWAYS_SOFTWARE=0`` to use the GPU driver instead). Needs ``pyvista``.

The animation in docs/results/deforming-plate.md:

    python utils/animate_rollout.py rollouts/<mgn_t rollout> rollouts/<mgn rollout> \\
        --trajectory test_traj0136 --output docs/assets/rollout_test_traj0136.gif

Usage
-----
    python utils/animate_rollout.py rollouts/<a> [rollouts/<b> ...] \\
        [--trajectory test_traj0136] [--labels "MGN-T" "MGN"] \\
        [--stress-max 200] [--output <file.gif>]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Before VTK loads OpenGL: off-screen, the GPU driver of WSL2 (D3D12) returns a
# black first frame or crashes; in software the whole GIF takes ~20 s.
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from PIL import Image  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset_preprocessor.dataset import connect_readonly  # noqa: E402
from dataset_preprocessor.graph_features import NODE_TYPE_ACTUATOR  # noqa: E402
from utils.plot_losses import SURFACE, TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY  # noqa: E402
from utils.timezone import now  # noqa: E402
from utils.units import SI_STRESS, STRESS_UNITS, stress_display  # noqa: E402

EVALUATIONS_DIR = _REPO_ROOT / "evaluations"
# "meshgraphnet": misspelling of rollouts from v0.2.0 checkpoints (see train/config.py)
ARCHITECTURE_NAMES = {"mgn_t": "MGN-T", "meshgraphnets": "MeshGraphNets",
                      "meshgraphnet": "MeshGraphNets"}
# Sequential blue ramp of the reference palette (steps 100-700, see plot_losses.py).
STRESS_COLORS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
                 "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
                 "#0d366b"]
ACTUATOR_COLOR = "#c3c2bc"
PANEL_PX = (333, 250)        # width, height of one panel in the GIF
SUPERSAMPLING = 2            # rendered at twice the size, scaled down by matplotlib
HEADER_PX, LEGEND_PX, FOOTER_PX = 30, 46, 18
FRAME_MS, LAST_FRAME_MS = 120, 1500
VIEW_DIRECTION = (0.75, -1.0, 0.85)   # camera from the free corner, above the plate


@dataclass
class Rollout:
    folder: Path
    info: dict
    steps: List[Path]        # tNNNN.vtu of the trajectory, in step order

    @property
    def initial_frame(self) -> int:
        """Raw frame of the run the rollout starts from; it and the frames before are ground
        truth (rollouts up to v0.4.0 with history: the first two frames)."""
        recorded = (self.info.get("results") or {}).get("initial_frame")
        if recorded is not None:
            return int(recorded)
        metadata = self.info["data"].get("metadata") or {}
        return int(metadata.get("first_frame", metadata.get("history", 0)))

    @property
    def frames(self) -> List[int]:
        """Raw frame of the run of every file (tNNNN). Up to v0.4.0 the files counted the
        rollout steps from 1, without the given frames."""
        numbers = [int(path.stem[1:]) for path in self.steps]
        if "initial_frame" in (self.info.get("results") or {}) or 0 in numbers:
            return numbers
        return [n + self.initial_frame for n in numbers]

    @property
    def label(self) -> str:
        model = self.info["model"]
        name = ARCHITECTURE_NAMES.get(model["architecture"], model["architecture"])
        return f"{name} ({model['parameters'] / 1e6:.2f} M params)"

    @property
    def dataset_id(self) -> str:
        """Content fingerprint of dataset.db; SHA-256 of the file before format v4."""
        record = self.info["data"]["dataset_db"]
        return record.get("fingerprint") or record["sha256"]


def load_rollout(folder: Path, trajectory: Optional[str]) -> Rollout:
    info_path = folder / "rollout_info.json"
    if not info_path.exists():
        raise SystemExit(f"{folder}: no rollout_info.json (not a rollout folder)")
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    trajectory = trajectory or info["results"]["sequences"][0]
    steps = sorted((folder / info["data"]["split"] / trajectory).glob("t[0-9]*.vtu"))
    if not steps:
        raise SystemExit(f"{folder}: no .vtu files for {trajectory} "
                         "(rolled out with --no-export-vtu or without pyvista?)")
    return Rollout(folder, info, steps)


def read_nodes(rollouts: Sequence[Rollout], trajectory: str) -> Tuple[np.ndarray, np.ndarray]:
    """Raw node types of the trajectory and the mask of its nodes in free rigid bodies,
    from the first dataset.db that is found (recorded absolute path, else the path
    relative to the repository)."""
    candidates = [Path(p) for r in rollouts for p in (
        r.info["data"]["dataset_db"]["absolute_path"],
        _REPO_ROOT / r.info["data"]["dataset_db"]["path"])]
    db = next((p for p in candidates if p.exists()), None)
    if db is None:
        raise SystemExit("dataset.db of the rollouts not found; looked in:\n  "
                         + "\n  ".join(map(str, dict.fromkeys(candidates))))
    conn = connect_readonly(db)
    try:
        info = dict(conn.execute("SELECT key, value FROM info WHERE key IN "
                                 "('rigid_bodies', 'component_names')"))
        bodies = json.loads(info.get("rigid_bodies", "[]"))
        # Exports without components have no such column.
        components = "components" if bodies else "NULL"
        row = conn.execute(f"SELECT node_type, {components} FROM trajectories WHERE split = ? "
                           "AND sequence_id = ?", (rollouts[0].info["data"]["split"],
                                                   trajectory)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise SystemExit(f"{db}: no trajectory {trajectory}")
    node_type = np.frombuffer(row[0], dtype=np.uint8)
    if not bodies:
        return node_type, np.zeros(len(node_type), dtype=bool)
    names = json.loads(info["component_names"])
    return node_type, np.isin(np.frombuffer(row[1], dtype=np.uint32),
                              [names.index(name) for name in bodies])


def check_consistent(rollouts: Sequence[Rollout]) -> None:
    first = rollouts[0]
    for r in rollouts[1:]:
        if r.dataset_id != first.dataset_id:
            raise SystemExit(f"{r.folder.name}: other dataset than {first.folder.name}")
        if [p.name for p in r.steps] != [p.name for p in first.steps]:
            raise SystemExit(f"{r.folder.name}: other rollout steps than {first.folder.name}")


def colour_field(step: Path) -> Optional[str]:
    """The result field the panels are coloured by: the stress if the rollout has it,
    else its first other result field; None if it has none."""
    import pyvista as pv

    names = [name[:-len("_gt")] for name in pv.read(step).point_data.keys()
             if name.endswith("_gt")]
    fields = [name for name in names if name != "world_pos" and not name.startswith("velocity_")]
    return "stress" if "stress" in fields else (fields[0] if fields else None)


def field_scale(field: Optional[str], stress_unit: str = SI_STRESS) -> Tuple[float, str]:
    """Factor from the stored values to the colour scale, and its title; the stress of
    data in ``stress_unit`` as the reports show it (``utils.units.stress_display``)."""
    if field == "stress":
        unit = stress_display(stress_unit)[1]
        return STRESS_UNITS[stress_unit] / STRESS_UNITS[unit], f"von Mises stress [{unit}]"
    return 1.0, field or "no result field"


def read_frames(rollouts: Sequence[Rollout], field: Optional[str], factor: float = 1.0):
    """Per step: positions and the values of ``field`` (times ``factor``, see
    ``field_scale``; 0 without a field) of the ground truth and of each rollout."""
    import pyvista as pv

    def values(mesh, kind):
        if field is None:
            return np.zeros(mesh.n_points)
        return np.asarray(mesh.point_data[f"{field}_{kind}"]) * factor

    frames = []
    for step in range(len(rollouts[0].steps)):
        meshes = [pv.read(r.steps[step]) for r in rollouts]
        truth = np.asarray(meshes[0].point_data["world_pos_gt"])
        for r, mesh in zip(rollouts[1:], meshes[1:]):
            if not np.allclose(mesh.point_data["world_pos_gt"], truth, atol=1e-6):
                raise SystemExit(f"{r.folder.name}: other ground truth than "
                                 f"{rollouts[0].folder.name} (step {step + 1})")
        frames.append([(truth, values(meshes[0], "gt"))]
                      + [(np.asarray(m.points), values(m, "pred")) for m in meshes])
    return frames


def body_surfaces(first_step: Path, node_type: np.ndarray, rigid: np.ndarray):
    """Surfaces of the deformable bodies and of the actuator, each with the
    indices of its points in the full mesh (point data ``node``). Without the lines of
    the nodes in ``rigid``: they tie a free rigid body together and are no structure."""
    import pyvista as pv

    grid = pv.read(first_step)
    if grid.n_points != len(node_type):
        raise SystemExit(f"{first_step}: {grid.n_points} points, but the dataset has "
                         f"{len(node_type)} nodes in this trajectory")
    grid.clear_data()
    grid.point_data["node"] = np.arange(grid.n_points)
    # Bodies share no cells, so the first node of a cell tells its body. The cell
    # arrays come from VTK: pyvista renamed its accessor for the offsets in 0.49.
    cells = grid.GetCells()
    first_node = pv.convert_array(cells.GetConnectivityArray())[
        pv.convert_array(cells.GetOffsetsArray())[:-1]]
    actuator_cell = node_type[first_node] == NODE_TYPE_ACTUATOR
    tie = rigid[first_node] & (grid.celltypes == pv.CellType.LINE)
    with warnings.catch_warnings():   # pyvista >= 0.47 announces a new default algorithm
        warnings.filterwarnings("ignore", message=".*`algorithm`")
        return (grid.extract_cells(np.flatnonzero(~actuator_cell & ~tie)).extract_surface(),
                grid.extract_cells(np.flatnonzero(actuator_cell)).extract_surface())


def render(frames, structure, actuator, stress_max: float, cmap) -> List[np.ndarray]:
    """One image per step: all panels in a row, one camera for all panels and steps."""
    import pyvista as pv

    n_panels = len(frames[0])
    width, height = PANEL_PX[0] * SUPERSAMPLING, PANEL_PX[1] * SUPERSAMPLING
    plotter = pv.Plotter(off_screen=True, shape=(1, n_panels), border=False,
                         window_size=(width * n_panels, height))
    plotter.set_background(SURFACE)
    panels = []
    for i in range(n_panels):
        plotter.subplot(0, i)
        body, rigid = structure.copy(), actuator.copy()
        body.point_data["stress"] = np.zeros(body.n_points)
        plotter.add_mesh(body, scalars="stress", cmap=cmap, clim=(0.0, stress_max),
                         show_scalar_bar=False, show_edges=True, edge_color="#ffffff",
                         edge_opacity=0.25, line_width=0.5,
                         ambient=0.3, diffuse=0.75, specular=0.0)   # colours as on the bar
        if rigid.n_points:
            plotter.add_mesh(rigid, color=ACTUATOR_COLOR, smooth_shading=True)
        panels.append((body, rigid))
    plotter.link_views()

    # Camera over the bounding box of all panels and steps, fixed for the whole GIF.
    points = np.concatenate([pos for frame in frames for pos, _ in frame])
    lower, upper = points.min(0), points.max(0)
    center = (lower + upper) / 2
    camera = plotter.camera
    camera.focal_point = center
    camera.position = center + np.asarray(VIEW_DIRECTION)
    camera.up = (0.0, 0.0, 1.0)
    plotter.renderer.reset_camera(bounds=np.column_stack([lower, upper]).ravel())
    camera.zoom(1.3)

    images = []
    for frame in frames:
        for (body, rigid), (pos, stress) in zip(panels, frame):
            nodes = body.point_data["node"]
            body.points = pos[nodes]
            body.point_data["stress"][:] = np.clip(stress[nodes], 0.0, None)
            body.Modified()
            if rigid.n_points:
                rigid.points = pos[rigid.point_data["node"]]
        plotter.render()
        images.append(plotter.screenshot(return_img=True))
    plotter.close()
    return images


def compose(images, labels, stress_max: float, cmap, footer: str,
            scale_title: str = "von Mises stress [kPa]",
            counters: Optional[Sequence[str]] = None) -> List[Image.Image]:
    """Titles, colour bar, footer and counter (``counters``, one per image; default:
    step k / n) around the rendered row."""
    n_panels = len(labels)
    width = PANEL_PX[0] * n_panels
    height = HEADER_PX + PANEL_PX[1] + LEGEND_PX + FOOTER_PX
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=SURFACE)
    ax = fig.add_axes((0, (LEGEND_PX + FOOTER_PX) / height, 1, PANEL_PX[1] / height))
    ax.set_axis_off()
    shown = ax.imshow(images[0], interpolation="antialiased", aspect="auto")
    for i, label in enumerate(labels):
        fig.text((i + 0.5) / n_panels, 1 - HEADER_PX / 2 / height, label, ha="center",
                 va="center", fontsize=11, color=TEXT_PRIMARY)
    cax = fig.add_axes((0.33, (FOOTER_PX + 18) / height, 0.34, 12 / height))
    bar = fig.colorbar(ScalarMappable(Normalize(0.0, stress_max), cmap), cax=cax,
                       orientation="horizontal")
    bar.outline.set_visible(False)
    cax.tick_params(labelsize=7.5, colors=TEXT_SECONDARY, length=2)
    cax.set_title(scale_title, fontsize=7.5, color=TEXT_SECONDARY, loc="left", pad=3)
    fig.text(0.01, 5 / height, footer, fontsize=7.5, color=TEXT_MUTED, va="bottom")
    # Left of the colour bar, clear of the footer, which can span the whole width.
    counter = fig.text(0.01, (FOOTER_PX + 18) / height, "", fontsize=8.5,
                       color=TEXT_SECONDARY, ha="left", va="bottom")

    frames = []
    for step, image in enumerate(images, start=1):
        shown.set_data(image)
        counter.set_text(counters[step - 1] if counters else f"step {step} / {len(images)}")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
    plt.close(fig)
    return frames


def save_gif(frames: List[Image.Image], path: Path) -> None:
    """One palette for all frames (from every 6th one), so colours do not flicker."""
    sample = Image.fromarray(np.concatenate([np.asarray(f) for f in frames[::6]]))
    palette = sample.quantize(colors=255, method=Image.Quantize.MAXCOVERAGE)
    indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")   # stopped while saving: no GIF that looks finished
    indexed[0].save(partial, format="GIF", save_all=True, append_images=indexed[1:], loop=0,
                    duration=[FRAME_MS] * (len(frames) - 1) + [LAST_FRAME_MS])
    partial.replace(path)


def nice_ceiling(value: float) -> float:
    """The smallest 1, 2 or 5 times a power of ten that is at least ``value`` (> 0);
    1 for no positive value."""
    if not value > 0:
        return 1.0
    power = 10.0 ** math.floor(math.log10(value))
    return next(f * power for f in (1, 2, 5, 10) if f * power >= value)


def scale_end(field: Optional[str], peak: float) -> float:
    """Upper end of the colour scale for ``peak`` in the units of the scale: a stress of
    at least 50 up to the next 50, any other value by ``nice_ceiling``, e.g. a stress in
    GPa, which a step of 50 would leave without colours."""
    if field == "stress" and peak >= 50.0:
        return 50.0 * math.ceil(peak / 50.0)
    return nice_ceiling(peak)


def new_output_dir() -> Path:
    base = EVALUATIONS_DIR / f"{now():%Y%m%d_%H%M%S}_animation"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}_{n}")
    return path


def build_parser() -> argparse.ArgumentParser:
    """Command line options; the reference in docs/reference/cli.md is generated from them."""
    parser = argparse.ArgumentParser(
        description="Animated GIF of one rollout trajectory: ground truth and models.")
    parser.add_argument("rollouts", nargs="+", type=Path,
                        help="rollout folders with .vtu export; the first one also "
                             "provides the ground truth")
    parser.add_argument("--trajectory", default=None,
                        help="e.g. test_traj0136 (default: the first of the split)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="panel titles, one per rollout "
                             "(default: architecture and parameters)")
    parser.add_argument("--stress-max", type=float, default=None,
                        help="upper end of the colour scale: the stress in kPa, for data in "
                             "MPa or GPa (--stress-unit of the export) in that unit; for a "
                             "dataset without stress in the unit of its first result field "
                             "(default: 99.9th percentile of the ground truth, a stress of at "
                             "least 50 rounded up to the next 50, any other value to 1, 2 or 5 "
                             "times a power of ten)")
    parser.add_argument("--output", type=Path, default=None,
                        help="GIF file (default: evaluations/<time>_animation/"
                             "rollout_<trajectory>.gif)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.labels is not None and len(args.labels) != len(args.rollouts):
        parser.error(f"{len(args.labels)} labels for {len(args.rollouts)} rollouts")

    first = load_rollout(args.rollouts[0], args.trajectory)
    trajectory = first.steps[0].parent.name
    rollouts = [first] + [load_rollout(f, trajectory) for f in args.rollouts[1:]]
    check_consistent(rollouts)

    field = colour_field(first.steps[0])
    data = first.info["data"]
    factor, scale_title = field_scale(field, data["metadata"].get("stress_unit", SI_STRESS))
    frames = read_frames(rollouts, field, factor)
    structure, actuator = body_surfaces(first.steps[0], *read_nodes(rollouts, trajectory))
    # A percentile, not the maximum: single peaks would wash out the colours.
    peak = float(np.nanpercentile(np.concatenate([frame[0][1] for frame in frames]), 99.9))
    stress_max = args.stress_max or scale_end(field, peak)
    cmap = LinearSegmentedColormap.from_list("stress", STRESS_COLORS)

    stride = data["metadata"].get("time_stride", 1)
    split_note = " (not seen in training)" if data["split"] == "test" else ""
    initial = first.initial_frame
    footer = (f"{data['metadata'].get('dataset', '')} · {trajectory}{split_note} · "
              f"autoregressive rollout, {sum(f > initial for f in first.frames)} steps "
              f"at stride {stride}"
              + (" · grey: actuator" if actuator.n_points else ""))
    labels = ["Ground truth (FEM)"] + (args.labels or [r.label for r in rollouts])

    images = render(frames, structure, actuator, stress_max, cmap)
    counters = [f"frame {f} / {first.frames[-1]}" + (" · ground truth" if f <= initial else "")
                for f in first.frames]
    gif = compose(images, labels, stress_max, cmap, footer, scale_title, counters)
    output = args.output or new_output_dir() / f"rollout_{trajectory}.gif"
    save_gif(gif, output)
    print(f"{output} ({len(gif)} frames, {output.stat().st_size / 1e6:.1f} MB, "
          f"colour scale 0-{stress_max:g}: {scale_title})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
