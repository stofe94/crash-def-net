"""
vtk_reader.py — Read VTK time series (.vtu / .vtp / .vtk / .vtm) as raw trajectories.

The third raw format next to DeepMind's TFRecord and LS-DYNA d3plot
(``export.py --raw-format vtk``). A run is a folder with one file per state
— the ParaView convention, also written by ``tfrecord_to_vtk.py`` and by the
rollout's ``.vtu`` export::

    <raw-dir>/meta.json                     collision_radius, trajectory_length (optional)
    <raw-dir>/<source>/<run>/<run>.vtu.series   order and times (or a .pvd)
    <raw-dir>/<source>/<run>/t0000.vtu          state 0: points, cells, point data
    <raw-dir>/<source>/<run>/t0001.vtu          state 1 ...
    <raw-dir>/<source>/<run>/globals.json       run parameters (optional)

Order of the states: a ``.pvd`` (``<DataSet timestep part file>``; several
parts per time step allowed), else ``*.series`` files (one per part), else
the trailing number of the file names (``part_a_0001.vtp``: part ``part_a``).
The files may lie in subfolders, as ParaView saves a ``.pvd``
(``<run>/<run>.pvd`` + ``<run>/<run>/<run>_0_<t>.vtu``); the folders below an
index or a multiblock file (``.vtm``) belong to it and are no runs of their own.
A ``.vtm`` contributes each of its blocks as a part (ParaView's LS-DYNA reader:
one block per part). The parts of a state are joined into one mesh. A part is
named by its block (block name, else its index), else by its file (``part`` of
the ``.pvd``, name of the ``.series``, file name without the number), or per
cell by the cell field ``part_field`` (e.g. part IDs).

What a run provides (``read_run``)
----------------------------------
* nodes: exports give every part its own copies of the nodes it shares with
  other parts. With node IDs (point field ``node_id_field``; default: the field
  the files mark as global IDs, e.g. ``UserID`` of ParaView's LS-DYNA reader)
  the copies become one node again — they must lie at the same position in
  every state — and the states are matched by ID. Without IDs (or with
  ``node_id_field=""``) the points of the parts are the nodes (every state
  must list them in the same order). VTK 9.6.2 (ParaView's LS-DYNA reader)
  combines two user IDs of a 32-bit d3plot into one; the position check rejects
  such IDs.
* ``world_pos``: the node positions of every state, or the points plus a
  displacement field (``displacement_field``) for exports that keep the
  reference geometry. ``mesh_pos``: the positions of the first state. The cells
  of the first state hold for all states (fixed topology).
* ``cells``: VTK cells (lines, triangles, quadrilaterals, tetrahedra, pyramids,
  wedges, hexahedra); polygons are triangulated, quadratic cells linearized,
  vertices ignored. Degenerate cells as LS-DYNA writes them become the cells
  they are (quadrilateral n1 n2 n3 n3: triangle; hexahedra with repeated nodes:
  tetrahedron, pyramid, wedge); other cells with repeated nodes are rejected.
* ``node_type``: point field ``node_type_field`` with the raw types {0, 1, 3}
  (normal, actuator, clamped), then the nodes of the cells of ``actuator_parts``
  and ``clamped_parts``; without either every node is normal. A node takes the
  highest type of its copies and parts (clamped > actuator > normal).
* result fields (the targets besides the velocity), per state: ``stress`` from
  the point field ``stress_field`` — a scalar, or a stress tensor (6 Voigt
  components xx yy zz xy yz zx, or 9) from which the von Mises stress is
  computed; cell data: von Mises per cell, then averaged at the nodes (as in
  ``d3plot_reader``); ``stress_field=""``: none. Then the point or cell fields
  ``fields`` (e.g. a plastic strain), each under its own name; vectors give one
  field per component as in ``node_static``. Nodes of parts without a field have no
  value (NaN).
* ``node_static``: the point or cell fields ``static_fields`` of the first
  state, NaN (no value) at nodes of parts without them; vectors give one feature per
  component (``<name>_x`` / ``_y`` / ``_z``, or ``<name>_0`` ...).
* ``globals``: ``globals.json`` in the run folder.
* ``component``: the part of every node (``component_names``: the part names,
  by block, file or ``part_field`` as above). A node of several parts belongs to
  the one that appears first.
* ``rigid_bodies``: the parts named in ``free_rigid_parts``, rigid parts whose motion
  results from the crash: their nodes are normal (also where ``node_type_field``
  marks them as actuator), so the model predicts their motion, and the rollout moves
  each as a rigid body.
* ``components``: the parts of the run and their roles (``export_config.py``). Parts that
  no pattern matches are left out, with the points only they hold; ``node_type_field``
  still marks single nodes.

Needs ``pyvista``.
"""

from __future__ import annotations

import json
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dataset_preprocessor.d3plot_reader import (
    DEGENERATE_SOLIDS,
    GLOBALS_FILE,
    NodalAverage,
    degenerate_solids,
    distinct_corners,
    von_mises,
)
from dataset_preprocessor.export import RawTrajectory
from dataset_preprocessor.export_config import (
    ROLE_ACTUATOR,
    ROLE_FIXED,
    ROLE_FREE_RIGID,
    component_roles,
)
from dataset_preprocessor.graph_features import (
    CELL_EDGES,
    CELL_HEXAHEDRON,
    CELL_NUM_NODES,
    CELL_QUAD,
    CELL_TETRA,
    CELL_TRIANGLE,
    NODE_TYPE_ACTUATOR,
    NODE_TYPE_CLAMPED,
    NODE_TYPE_NORMAL,
)

VTK_SUFFIXES = (".vtu", ".vtp", ".vtk", ".vtm")
INDEX_SUFFIXES = (".pvd", ".series")
_VERTEX_TYPES = (1, 2)                     # VTK_VERTEX, VTK_POLY_VERTEX: no edges
# Axis-aligned VTK cells with their own node order -> (linear type, node order).
_REORDERED_TYPES = {8: (CELL_QUAD, [0, 1, 3, 2]),                    # VTK_PIXEL
                    11: (CELL_HEXAHEDRON, [0, 1, 3, 2, 4, 5, 7, 6])}  # VTK_VOXEL
_KNOWN_NODE_TYPES = (NODE_TYPE_NORMAL, NODE_TYPE_ACTUATOR, NODE_TYPE_CLAMPED)
_COPY_TOLERANCE = 1e-6                     # copies of a node: distance / mesh extent


# ─────────────────────────────────────────────────────────────────────────────
# States of a run
# ─────────────────────────────────────────────────────────────────────────────

def _vtk_files(run_dir: Path) -> List[Path]:
    return sorted(p for p in run_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in VTK_SUFFIXES)


def run_states(run_dir) -> List[Tuple[float, List[Tuple[str, Path]]]]:
    """(time, [(part name, file)]) of every state, in time order."""
    run_dir = Path(run_dir)
    pvd = sorted(run_dir.glob("*.pvd"))
    if len(pvd) > 1:
        raise ValueError(f"{run_dir}: several .pvd files ({', '.join(p.name for p in pvd)}) "
                         "— one run per folder")
    if pvd:
        by_time: Dict[float, List[Tuple[str, Path]]] = {}
        for dataset in ET.parse(pvd[0]).getroot().iter("DataSet"):
            time = float(dataset.get("timestep", len(by_time)))
            by_time.setdefault(time, []).append(
                (dataset.get("part", "0"), run_dir / dataset.get("file")))
        return [(time, sorted(parts)) for time, parts in sorted(by_time.items())]

    series = sorted(run_dir.glob("*.series"))
    if series:
        parts = []
        for path in series:   # one series file per part, named like it (part_a.vtu.series)
            with open(path, encoding="utf-8") as f:
                files = json.load(f)["files"]
            name = Path(path.stem).stem
            parts.append([(float(item.get("time", i)), (name, run_dir / item["name"]))
                          for i, item in enumerate(files)])
        if len({len(part) for part in parts}) != 1:
            raise ValueError(f"{run_dir}: the series files list different numbers of states")
        return [(part_states[0][0], [state[1] for state in part_states])
                for part_states in zip(*parts)]

    # no index file: trailing number of the file name, the rest names the part
    numbered: Dict[int, List[Tuple[str, Path]]] = {}
    for path in _vtk_files(run_dir):
        match = re.match(r"^(.*?)(\d+)$", path.stem)
        part, number = (match.group(1), int(match.group(2))) if match else (path.stem, 0)
        numbered.setdefault(number, []).append((part.rstrip("_-.") or part, path))
    return [(float(number), sorted(parts)) for number, parts in sorted(numbered.items())]


def num_states(run_dir) -> int:
    """Number of states of a run, without reading the meshes."""
    return len(run_states(run_dir))


def run_dirs(raw_dir, source: str) -> List[Path]:
    """Run folders below ``raw_dir/<source>`` (``source`` may name the TFRecord the runs
    were converted from), or below ``raw_dir`` if that does not exist: folders with an
    index or VTK files, except those inside a folder with an index or ``.vtm`` file."""
    root = Path(raw_dir) / Path(source).stem
    root = root if root.is_dir() else Path(raw_dir)
    files = [p for p in root.rglob("*")
             if p.is_file() and p.suffix.lower() in VTK_SUFFIXES + INDEX_SUFFIXES]
    owners = {p.parent for p in files if p.suffix.lower() in INDEX_SUFFIXES + (".vtm",)}
    runs = sorted({p.parent for p in files if owners.isdisjoint(p.parent.parents)})
    if not runs:
        raise FileNotFoundError(f"no runs (folders with .vtu / .vtp / .vtk / .vtm files) "
                                f"below {root}")
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Meshes and fields
# ─────────────────────────────────────────────────────────────────────────────

def _linear_grid(mesh, path: Path):
    """``mesh`` as ``pyvista.UnstructuredGrid`` with linear cells only."""
    import pyvista as pv

    grid = mesh if isinstance(mesh, pv.UnstructuredGrid) else mesh.cast_to_unstructured_grid()
    types = set(np.unique(grid.celltypes).tolist())
    if types & {6, 7} and isinstance(mesh, pv.PolyData):   # triangle strips, polygons
        warnings.warn(f"{path}: polygons / triangle strips are triangulated")
        grid = mesh.triangulate().cast_to_unstructured_grid()
        types = set(np.unique(grid.celltypes).tolist())
    if not types <= set(CELL_EDGES) | set(_VERTEX_TYPES) | set(_REORDERED_TYPES):
        grid = grid.linear_copy()           # quadratic -> linear cells
    return grid


def _read_file(path: Path) -> List[Tuple[Optional[str], object]]:
    """The meshes of a state file: one (name None), or the non-empty blocks of a
    multiblock file, named by the block name, else by the index of the block."""
    import pyvista as pv

    data = pv.read(str(path))
    if not isinstance(data, pv.MultiBlock):
        return [(None, _linear_grid(data, path))]
    blocks = list(data.recursive_iterator("items", skip_none=True))
    return [(name or str(index), _linear_grid(block, path))
            for index, (name, block) in enumerate(blocks) if block.n_points]


def _collapse(cells: np.ndarray, types: np.ndarray) -> None:
    """Degenerate cells as LS-DYNA writes them (also through ParaView's LS-DYNA
    reader) become the cells they are, in place: quadrilaterals n1 n2 n3 n3
    triangles, hexahedra with repeated nodes the shapes of ``DEGENERATE_SOLIDS``."""
    triangle = (types == CELL_QUAD) & (cells[:, 3] == cells[:, 2])
    cells[triangle, 3] = -1
    types[triangle] = CELL_TRIANGLE
    hexahedra = np.flatnonzero(types == CELL_HEXAHEDRON)
    nodes = cells[hexahedra]
    for (cell_type, corners), mask in zip(DEGENERATE_SOLIDS, degenerate_solids(nodes)):
        rows = hexahedra[mask]
        cells[rows] = -1
        cells[rows, :len(corners)] = nodes[mask][:, corners]
        types[rows] = cell_type


def _cells(mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(C, 8) node indexes padded with -1, (C,) VTK types and the original index
    of every cell (``cells_dict`` groups the cells by type); degenerate cells
    collapsed (``_collapse``)."""
    rows, types, order = [], [], []
    for cell_type, nodes in mesh.cells_dict.items():
        cell_type = int(cell_type)
        if cell_type in _VERTEX_TYPES:
            continue
        nodes = np.asarray(nodes, dtype=np.int64).reshape(len(nodes), -1)
        order.append(np.flatnonzero(mesh.celltypes == cell_type))
        if cell_type in _REORDERED_TYPES:
            cell_type, node_order = _REORDERED_TYPES[cell_type]
            nodes = nodes[:, node_order]
        if cell_type not in CELL_EDGES:
            raise ValueError(f"unsupported VTK cell type {cell_type}")
        rows.append(np.pad(nodes, ((0, 0), (0, 8 - nodes.shape[1])), constant_values=-1))
        types.append(np.full(len(nodes), cell_type, dtype=np.uint8))
    if not rows:
        return (np.empty((0, 8), dtype=np.int64), np.empty(0, dtype=np.uint8),
                np.empty(0, dtype=np.int64))
    cells, types = np.concatenate(rows), np.concatenate(types)
    _collapse(cells, types)
    return cells, types, np.concatenate(order)


@dataclass
class _Part:
    """One mesh of a state; ``nodes`` places its points in the joined mesh."""
    name: str                               # block / file name, see the module docstring
    mesh: object                            # pyvista.UnstructuredGrid, linear cells
    nodes: Optional[np.ndarray] = None      # (n_points,) node of every point
    _cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def cells(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``_cells`` of the mesh with the nodes of the joined mesh."""
        if self._cache is None:
            cells, types, order = _cells(self.mesh)
            self._cache = (np.where(cells >= 0, self.nodes[cells], -1), types, order)
        return self._cache


def _read_state(files: List[Tuple[str, Path]]) -> List[_Part]:
    return [_Part(name if block is None else block, mesh)
            for name, path in files for block, mesh in _read_file(path)]


def _node_ids(parts: List[_Part], node_id_field: Optional[str], where: str
              ) -> Optional[np.ndarray]:
    """IDs of the points of all parts, concatenated; None if the files have none or
    ``node_id_field`` is ``""``."""
    if node_id_field == "":
        return None
    ids = []
    for part in parts:
        name = node_id_field
        if name is None:   # the field the file marks as global point IDs
            marked = part.mesh.GetPointData().GetGlobalIds()
            name = None if marked is None else marked.GetName()
        ids.append(None if name is None or name not in part.mesh.point_data else
                   np.asarray(part.mesh.point_data[name]).astype(np.int64).reshape(-1))
    if all(i is None for i in ids):
        if node_id_field is not None:
            raise ValueError(f"{where}: no point field {node_id_field!r} (node IDs)")
        return None
    if any(i is None for i in ids):
        raise ValueError(f"{where}: parts {[p.name for p, i in zip(parts, ids) if i is None]} "
                         "have no node IDs, the others have")
    return np.concatenate(ids)


class _NodeIndex:
    """Node of every point: the copies of a node in several parts (same ID) are one
    node, in the order of their first copy in the first state. Without IDs every
    point is a node."""

    def __init__(self, ids: Optional[np.ndarray], num_points: int):
        self.ids = None if ids is None else ids[np.sort(np.unique(ids, return_index=True)[1])]
        self.num_nodes = num_points if ids is None else len(self.ids)
        self._order = None if ids is None else np.argsort(self.ids)

    def __call__(self, ids: Optional[np.ndarray], num_points: int, where: str) -> np.ndarray:
        if self.ids is None:
            if num_points != self.num_nodes:
                raise ValueError(f"{where} has {num_points} points, the first "
                                 f"{self.num_nodes} (the topology must stay fixed)")
            return np.arange(num_points)
        if ids is None:
            raise ValueError(f"{where}: no node IDs, the first state has them")
        found = np.searchsorted(self.ids, ids, sorter=self._order)
        index = self._order[np.minimum(found, self.num_nodes - 1)]
        if not np.array_equal(self.ids[index], ids):
            raise ValueError(f"{where}: node IDs that the first state does not have")
        if np.bincount(index, minlength=self.num_nodes).min() == 0:
            raise ValueError(f"{where}: nodes of the first state are missing")
        return index


def _points(mesh, displacement_field: Optional[str], where: str) -> np.ndarray:
    points = np.asarray(mesh.points, dtype=np.float64)
    if displacement_field:
        if displacement_field not in mesh.point_data:
            raise ValueError(f"{where}: no point field {displacement_field!r} (displacement)")
        points = points + np.asarray(mesh.point_data[displacement_field], np.float64)
    return points


def _field(parts: List[_Part], name: str, num_nodes: int, transform=None,
           part_field: Optional[str] = None, left_out: Sequence[str] = ()
           ) -> Tuple[Optional[np.ndarray], List[str]]:
    """(k, N) point or cell field ``name`` at the nodes, and the parts without it.

    Point values are averaged over the copies of a node, cell values over the
    cells around it; ``transform`` maps (M, k) values to (M,) — e.g. a tensor
    to its von Mises stress — before, as post-processors do. Nodes of parts
    without the field have no value (NaN); None if no part has it. The parts ``left_out``
    do not count (the cells named so by ``part_field``, else the meshes of that name).
    """
    average, missing = None, []
    for part in parts:
        mesh = part.mesh
        if part_field is None and part.name in left_out:
            continue
        if name in mesh.point_data:
            values = np.asarray(mesh.point_data[name], dtype=np.float64).reshape(mesh.n_points, -1)
            corners = part.nodes[:, None]
        elif name in mesh.cell_data:
            values = np.asarray(mesh.cell_data[name], dtype=np.float64).reshape(mesh.n_cells, -1)
            cells, _, order = part.cells()
            values = values[order]
            corners = np.where(cells >= 0, cells, cells[:, :1])   # padding: counted once
            if part_field is not None and len(left_out):
                in_run = ~np.isin(_cell_parts(part, part_field, name), list(left_out))
                values, corners = values[in_run], corners[in_run]
        else:
            missing.append(part.name)
            continue
        if transform is not None:
            values = transform(values)[:, None]
        if average is None:
            average = NodalAverage(values.shape[1], num_nodes)
        elif values.shape[1] != average.total.shape[0]:
            raise ValueError(f"field {name!r}: {values.shape[1]} components in part "
                             f"{part.name}, {average.total.shape[0]} in others")
        average.add(values.T, corners)
    return (None if average is None else average.result()), missing


def _scalar_stress(values: np.ndarray) -> np.ndarray:
    """(N, 1) scalar, (N, 6) Voigt or (N, 9) tensor -> (N,) von Mises / scalar."""
    if values.shape[1] == 1:
        return values[:, 0]
    if values.shape[1] == 9:
        t = values.reshape(-1, 3, 3)
        values = np.stack([t[:, 0, 0], t[:, 1, 1], t[:, 2, 2],
                           t[:, 0, 1], t[:, 1, 2], t[:, 2, 0]], axis=1)
    if values.shape[1] != 6:
        raise ValueError(f"stress field with {values.shape[1]} components; expected 1, 6 or 9")
    return von_mises(values)


def _node_types(parts: List[_Part], num_nodes: int, node_type_field: str,
                part_field: Optional[str], actuator_parts: Sequence, clamped_parts: Sequence,
                where: str, with_roles: bool = False) -> np.ndarray:
    """Node types from ``node_type_field`` and the parts; the highest type of a node wins."""
    node_type = np.full(num_nodes, NODE_TYPE_NORMAL, dtype=np.int32)
    with_field = [p for p in parts if node_type_field in p.mesh.point_data]
    for part in with_field:
        values = np.asarray(part.mesh.point_data[node_type_field]).reshape(-1).astype(np.int32)
        unknown = set(np.unique(values).tolist()) - set(_KNOWN_NODE_TYPES)
        if unknown:
            raise ValueError(f"{where}: node types {sorted(unknown)}; expected "
                             f"{list(_KNOWN_NODE_TYPES)}")
        np.maximum.at(node_type, part.nodes, values)
    if not with_field and not (actuator_parts or clamped_parts or with_roles):
        warnings.warn(f"{where}: no point field {node_type_field!r} — all nodes normal")

    found = set()
    for kind, wanted in ((NODE_TYPE_ACTUATOR, actuator_parts), (NODE_TYPE_CLAMPED, clamped_parts)):
        wanted = [str(name) for name in wanted]
        for part in parts if wanted else ():
            cells = part.cells()[0]
            names = _cell_parts(part, part_field, where)
            found.update(np.unique(names).tolist())
            nodes = cells[np.isin(names, wanted)]
            nodes = nodes[nodes >= 0]
            node_type[nodes] = np.maximum(node_type[nodes], kind)
    unknown = sorted({str(n) for n in (*actuator_parts, *clamped_parts)} - found)
    if unknown:
        raise ValueError(f"{where}: parts {unknown} not found; the run has "
                         f"{sorted(found)[:20]}")
    return node_type


def _cell_parts(part: _Part, part_field: Optional[str], where: str) -> np.ndarray:
    """The part name of every cell of ``part`` (its own name, or ``part_field``)."""
    cells, _, order = part.cells()
    if part_field is None:
        return np.full(len(cells), part.name)
    if part_field not in part.mesh.cell_data:
        raise ValueError(f"{where}: part {part.name} has no cell field {part_field!r}")
    values = np.asarray(part.mesh.cell_data[part_field]).reshape(-1)[order]
    return values.astype(np.int64).astype(str)


def _node_components(parts: List[_Part], num_nodes: int, part_field: Optional[str],
                     where: str, only: Optional[Sequence[str]] = None,
                     leading: Sequence[str] = ()) -> Tuple[np.ndarray, List[str]]:
    """The part of every node (N,) as index into the part names, in the order the
    parts appear; a node of several parts belongs to the first. ``only``: the parts that
    count, without points outside their cells; ``leading``: parts that go first."""
    names: Dict[str, int] = {name: k for k, name in enumerate(leading)}
    unset = np.iinfo(np.int64).max
    component = np.full(num_nodes, unset, dtype=np.int64)
    for part in parts:
        cells = part.cells()[0]
        cell_names = _cell_parts(part, part_field, where)
        if only is not None:
            in_run = np.isin(cell_names, list(only))
            cells, cell_names = cells[in_run], cell_names[in_run]
        unique, first, inverse = np.unique(cell_names, return_index=True, return_inverse=True)
        for k in np.argsort(first):
            names.setdefault(str(unique[k]), len(names))
        index = np.array([names[str(name)] for name in unique], np.int64)[inverse.reshape(-1)]
        valid = cells >= 0
        np.minimum.at(component, cells[valid], np.broadcast_to(index[:, None], cells.shape)[valid])
        loose = part.nodes[component[part.nodes] == unset]   # points without cells
        if len(loose) and only is None:
            component[loose] = names.setdefault(part.name, len(names))
    return component, list(names)


# ─────────────────────────────────────────────────────────────────────────────
# Runs
# ─────────────────────────────────────────────────────────────────────────────

# Memory per node and state while a run is read (positions in float64 and float32,
# stress, the meshes of pyvista); sets the frames per block of read_raw_run.
_BYTES_PER_NODE_STATE = 128


def _component_names(name: str, count: int) -> List[str]:
    """``name`` for a scalar, else ``name_x/_y/_z`` (3 components) or ``name_0`` ..."""
    if count == 1:
        return [name]
    return [f"{name}_{s}" for s in (("x", "y", "z") if count == 3 else range(count))]


class _StateReader:
    """Positions and result fields of the states of one run; the first state fixes
    the nodes and the names of the fields (``field_names``)."""

    def __init__(self, run_dir: Path, node_id_field: Optional[str],
                 displacement_field: Optional[str], stress_field: str,
                 fields: Sequence[str] = (), components: Optional[Dict[str, str]] = None,
                 part_field: Optional[str] = None):
        self.run_dir = run_dir
        self.node_id_field = node_id_field
        self.displacement_field = displacement_field
        self.stress_field = stress_field
        self.fields = list(fields)
        self.components = components
        self.part_field = part_field
        self.roles: Optional[Dict[str, str]] = None      # with components, from the first state
        self.left_out: List[str] = []
        self.field_names: List[str] = []
        self.node_index: Optional[_NodeIndex] = None
        self.num_nodes = 0
        self.tolerance = 0.0

    def read(self, files: List[Tuple[str, Path]], where: str
             ) -> Tuple[List[_Part], np.ndarray, np.ndarray]:
        """Parts, positions (N, 3) float32 and result fields (N, F) of one state."""
        parts = _read_state(files)
        ids = _node_ids(parts, self.node_id_field, where)
        num_points = sum(part.mesh.n_points for part in parts)
        first = self.node_index is None
        if first:
            self.node_index = _NodeIndex(ids, num_points)
            self.num_nodes = self.node_index.num_nodes
            if ids is None and len(parts) > 1:
                warnings.warn(f"{self.run_dir}: {len(parts)} parts and no node IDs — nodes "
                              "shared by parts stay separate copies (node_id_field)")
        nodes = self.node_index(ids, num_points, where)
        offsets = np.cumsum([0] + [part.mesh.n_points for part in parts])
        for part, start, end in zip(parts, offsets[:-1], offsets[1:]):
            part.nodes = nodes[start:end]
        if first and self.components:
            names = list(dict.fromkeys(name for part in parts for name in
                                       np.unique(_cell_parts(part, self.part_field, where)).tolist()))
            self.roles = component_roles(names, self.components, where)
            self.left_out = [name for name in names if name not in self.roles]

        points = np.concatenate([_points(part.mesh, self.displacement_field, where)
                                 for part in parts])
        positions = np.zeros((self.num_nodes, 3))
        positions[nodes] = points
        if first:
            self.tolerance = _COPY_TOLERANCE * max(np.ptp(points, axis=0).max(), 1e-30)
        apart = np.abs(points - positions[nodes]).max(axis=1)
        if apart.max(initial=0.0) > self.tolerance:
            raise ValueError(f"{where}: the copies of node "
                             f"{self.node_index.ids[nodes[apart.argmax()]]} lie "
                             f"{apart.max():.3g} apart — the node IDs do not identify the "
                             "nodes (node_id_field '' ignores them)")

        columns, names = [], []
        requested = ([(self.stress_field, _scalar_stress, "stress")] if self.stress_field
                     else []) + [(name, None, name) for name in self.fields]
        for field, transform, name in requested:
            values, missing = _field(parts, field, self.num_nodes, transform=transform,
                                     part_field=self.part_field, left_out=self.left_out)
            if values is None:
                raise ValueError(f"{where}: field {field!r} missing")
            if first and missing:
                warnings.warn(f"{self.run_dir}: parts without field {field!r} "
                              f"(no value at their nodes): {missing}")
            columns.append(values)
            names += _component_names(name, len(values))
        if first:
            self.field_names = names
        elif names != self.field_names:
            raise ValueError(f"{where}: fields {names}, the first state has {self.field_names}")
        values = np.concatenate(columns).T if columns else np.zeros((self.num_nodes, 0))
        return parts, positions.astype(np.float32), values


def read_run(run_dir, frames: Optional[Sequence[int]] = None, stress_field: str = "stress",
             node_type_field: str = "node_type", static_fields: Sequence[str] = (),
             displacement_field: Optional[str] = None, node_id_field: Optional[str] = None,
             part_field: Optional[str] = None, actuator_parts: Sequence = (),
             clamped_parts: Sequence = (), fields: Sequence[str] = (),
             free_rigid_parts: Sequence[str] = (),
             components: Optional[Dict[str, str]] = None) -> RawTrajectory:
    """One VTK time series as a ``RawTrajectory`` with all frames in memory; ``frames``
    selects states (default: all)."""
    traj = read_raw_run(run_dir, frames, stress_field=stress_field,
                        node_type_field=node_type_field, static_fields=static_fields,
                        displacement_field=displacement_field, node_id_field=node_id_field,
                        part_field=part_field, actuator_parts=actuator_parts,
                        clamped_parts=clamped_parts, fields=fields,
                        free_rigid_parts=free_rigid_parts, components=components)
    blocks = list(traj.frame_blocks)
    traj.world_pos = blocks[0][0] if len(blocks) == 1 else np.concatenate([b[0] for b in blocks])
    traj.stress = blocks[0][1] if len(blocks) == 1 else np.concatenate([b[1] for b in blocks])
    traj.frame_blocks = None
    return traj


def read_raw_run(run_dir, frames: Optional[Sequence[int]] = None, stress_field: str = "stress",
                 node_type_field: str = "node_type", static_fields: Sequence[str] = (),
                 displacement_field: Optional[str] = None, node_id_field: Optional[str] = None,
                 part_field: Optional[str] = None, actuator_parts: Sequence = (),
                 clamped_parts: Sequence = (), max_block_bytes: Optional[int] = None,
                 fields: Sequence[str] = (), free_rigid_parts: Sequence[str] = (),
                 components: Optional[Dict[str, str]] = None) -> RawTrajectory:
    """One VTK time series as a ``RawTrajectory`` whose frames are read in blocks of
    about ``max_block_bytes`` (None: one block) while ``blocks()`` is consumed;
    ``frames`` selects states (default: all). Topology, node types and static
    fields come from the first selected state. ``components``: the parts of the run and
    their roles (``export_config.py``), instead of ``actuator_parts``, ``clamped_parts``
    and ``free_rigid_parts``; the others are left out, with the points only they hold."""
    run_dir = Path(run_dir)
    states = run_states(run_dir)
    if not states:
        raise FileNotFoundError(f"{run_dir}: no VTK files")
    selected = list(range(len(states))) if frames is None else list(frames)
    if max(selected) >= len(states):
        raise ValueError(f"{run_dir}: state {max(selected)} requested, {len(states)} found")

    reader = _StateReader(run_dir, node_id_field, displacement_field, stress_field, fields,
                          components, part_field)
    where = f"{run_dir}: state {selected[0]}"
    parts, positions, stress = reader.read(states[selected[0]][1], where)
    num_nodes = reader.num_nodes
    cells = np.concatenate([part.cells()[0] for part in parts])
    cell_types = np.concatenate([part.cells()[1] for part in parts])
    roles, kept = reader.roles, None
    if roles is not None:
        names = np.concatenate([_cell_parts(part, part_field, where) for part in parts])
        in_run = np.isin(names, list(roles))
        cells, cell_types = cells[in_run], cell_types[in_run]
        kept = np.unique(cells[cells >= 0])
        actuator_parts = [name for name, role in roles.items() if role == ROLE_ACTUATOR]
        clamped_parts = [name for name, role in roles.items() if role == ROLE_FIXED]
        free_rigid_parts = [name for name, role in roles.items() if role == ROLE_FREE_RIGID]
    for cell_type in np.unique(cell_types).tolist():
        if not distinct_corners(cells[cell_types == cell_type, :CELL_NUM_NODES[cell_type]]).all():
            raise ValueError(f"{where}: cells of VTK type {cell_type} with repeated nodes "
                             "(unsupported degenerate cell)")
    node_type = _node_types(parts, num_nodes, node_type_field, part_field, actuator_parts,
                            clamped_parts, where, with_roles=roles is not None)
    free_rigid_parts = [str(name) for name in free_rigid_parts]
    component, component_names = _node_components(
        parts, num_nodes, part_field, where, only=roles,
        # A rigid body moves all its nodes, also those it shares with other parts.
        leading=free_rigid_parts if roles is not None else ())
    if kept is not None:
        node_type, component = node_type[kept], component[kept]
    if free_rigid_parts:
        unknown = sorted(set(free_rigid_parts) - set(component_names))
        both = sorted(set(free_rigid_parts) & {str(name) for name in actuator_parts})
        if unknown or both:
            raise ValueError(f"{where}: free rigid parts {unknown or both} "
                             + ("not found" if unknown else "are actuator parts as well")
                             + f"; the run has {component_names[:20]}")
        free = np.isin(np.asarray(component_names)[component], free_rigid_parts)
        node_type[free & (node_type == NODE_TYPE_ACTUATOR)] = NODE_TYPE_NORMAL
    node_static: Dict[str, np.ndarray] = {}
    for name in static_fields:
        values, _ = _field(parts, name, num_nodes, part_field=part_field,
                           left_out=reader.left_out)
        if values is None:
            raise ValueError(f"{where}: static field {name!r} missing")
        node_static.update(zip(_component_names(name, len(values)),
                               values if kept is None else values[:, kept]))
    del parts

    if kept is not None:
        new_index = np.full(num_nodes, -1, dtype=np.int64)
        new_index[kept] = np.arange(len(kept))
        cells = np.where(cells >= 0, new_index[cells], -1)
    width = max(CELL_NUM_NODES[int(t)] for t in np.unique(cell_types))
    cells = cells[:, :width]
    if (cell_types == CELL_TETRA).all():
        cells, cell_types = cells[:, :4], None   # plain tetrahedra: the (C, 4) layout
    globals_path = run_dir / GLOBALS_FILE
    run_globals = None
    if globals_path.exists():
        with open(globals_path, encoding="utf-8") as f:
            run_globals = {str(k): float(v) for k, v in json.load(f).items()}

    size = (max(1, int(max_block_bytes // (_BYTES_PER_NODE_STATE * max(num_nodes, 1))))
            if max_block_bytes else len(selected))
    held = [(positions, stress)]   # the first state, handed out once
    field_names = reader.field_names
    stress_only = field_names == ["stress"]   # as (t, N), the layout of the other readers

    def frame_blocks():
        world_pos, stresses = [], []
        for step, index in enumerate(selected):
            if step == 0:
                state_pos, state_stress = held.pop()
            else:
                _, state_pos, state_stress = reader.read(states[index][1],
                                                         f"{run_dir}: state {index}")
            world_pos.append(state_pos if kept is None else state_pos[kept])
            stresses.append(state_stress if kept is None else state_stress[kept])
            if len(world_pos) == size or step == len(selected) - 1:
                fields_block = np.stack(stresses).astype(np.float32)
                yield np.stack(world_pos), fields_block[..., 0] if stress_only else fields_block
                world_pos, stresses = [], []

    return RawTrajectory(node_type=node_type, mesh_pos=positions if kept is None else positions[kept],
                         cells=cells.astype(np.int32),
                         world_pos=None, stress=None, cell_types=cell_types,
                         globals=run_globals, node_static=node_static or None,
                         frame_blocks=frame_blocks(),
                         field_names=None if stress_only else field_names,
                         component=component, component_names=component_names,
                         rigid_bodies=[name for name in component_names
                                       if name in free_rigid_parts] or None,
                         component_roles=roles, left_out=reader.left_out if roles else None)
