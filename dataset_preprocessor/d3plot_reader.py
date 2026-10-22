"""
d3plot_reader.py — Read LS-DYNA runs (d3plot + keyword deck).

``read_run`` reads one run into a ``D3plotRun``: geometry, states, element
blocks, node types and nodal result fields. It is meant for general LS-DYNA
output (shells, solids, thick shells, beams, user IDs, rigid parts).
``read_run_blocks`` yields the states in blocks of bounded size instead, so a
large run (a full vehicle) need not fit into memory. ``read_raw_run`` turns a
run into the ``RawTrajectory`` of ``export.py`` (``export.py --raw-format
d3plot``; ``as_raw_trajectory`` for a run read with ``read_run``): all elements
as VTK cells (lines, triangles, quadrilaterals, tetrahedra, pyramids, wedges,
hexahedra), the von Mises stress as target and the run parameters. Raw
directory layout::

    <raw-dir>/meta.json                       collision_radius, trajectory_length (optional)
    <raw-dir>/<source>/<run>/d3plot           geometry + states (d3plot01, ... follow)
    <raw-dir>/<source>/<run>/*.k|*.key|*.dyn  keyword deck (optional)
    <raw-dir>/<source>/<run>/globals.json     run parameters, e.g. {"impact_velocity": 15.6}

What a run provides
-------------------
* ``mesh_pos`` (rest configuration) and ``world_pos`` (state coordinates; lasso
  calls them ``node_displacement``, d3plot stores current coordinates).
* ``elements``: node indexes and part IDs per element type. Degenerate
  elements repeat nodes (triangle n1 n2 n3 n3, tetrahedron n1 n2 n3 n4 n4 n4 n4
  n4); beams keep their two end nodes.
* ``node_type``: actuator = nodes of rigid parts (``*MAT_RIGID`` in the deck or
  material type 20 in the d3plot) and nodes with prescribed motion
  (``*BOUNDARY_PRESCRIBED_MOTION_NODE/_SET/_RIGID``), whose motion is taken as
  given; clamped = nodes fixed in x, y and z by ``*BOUNDARY_SPC_SET`` /
  ``*BOUNDARY_SPC_NODE``; all others normal.
* ``node_static``: ``thickness`` (shells, first state) and
  ``initial_velocity_x/y/z`` (``*INITIAL_VELOCITY*``).
* ``walls``: stationary planar rigid walls (``*RIGIDWALL_PLANAR``).
* ``von_mises``, ``plastic_strain``, ``thickness``: element results reduced
  over the integration points (``layer_reduction``) and averaged at each node
  over the elements around it. Rigid shells carry no state data and do not
  contribute.

Keyword deck: fixed (10 characters), long (20, ``*KEYWORD LONG=Y`` or a
trailing ``+``) and free (comma) format; ``*INCLUDE`` is followed; node and
part sets from ``*SET_NODE`` / ``*SET_PART`` (``_LIST``, ``_GENERATE``, ``_ADD``);
``_TITLE`` / ``_ID`` / ``_BIRTH_DEATH`` options. Not supported yet:
``*INCLUDE_TRANSFORM`` (ID offsets), ``*PART_INERTIA`` / ``*PART_COMPOSITE``,
eroded elements, moving or non-planar rigid walls, the optional second card
of ``*BOUNDARY_PRESCRIBED_MOTION`` and rotational initial velocities.
Needs ``lasso-python``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from dataset_preprocessor.export import RawTrajectory
from dataset_preprocessor.graph_features import (
    CELL_HEXAHEDRON,
    CELL_LINE,
    CELL_NUM_NODES,
    CELL_PYRAMID,
    CELL_QUAD,
    CELL_TETRA,
    CELL_TRIANGLE,
    CELL_WEDGE,
    NODE_TYPE_ACTUATOR,
    NODE_TYPE_CLAMPED,
    NODE_TYPE_NORMAL,
)

KEYWORD_SUFFIXES = (".k", ".key", ".dyn")
GLOBALS_FILE = "globals.json"                  # run parameters next to the d3plot
RIGID_MATERIAL_TYPE = 20                       # LS-DYNA material type of *MAT_RIGID
LAYER_REDUCTIONS = ("mean", "max")
# 8-node solids with repeated nodes, as LS-DYNA writes them: VTK type and the corners
# in VTK order — tetrahedron n1 n2 n3 n4 n4 n4 n4 n4, pyramid n1 .. n5 n5 n5 n5, wedge
# n1 n2 n3 n4 n5 n5 n6 n6 (triangles a b e, d c f)
DEGENERATE_SOLIDS = ((CELL_TETRA, [0, 1, 2, 3]), (CELL_PYRAMID, [0, 1, 2, 3, 4]),
                     (CELL_WEDGE, [0, 1, 4, 3, 2, 6]))
# lasso array names per element type: node indexes, part indexes, stress,
# effective plastic strain, thickness
_ELEMENT_ARRAYS = {
    "shell": ("element_shell_node_indexes", "element_shell_part_indexes", "element_shell_stress",
              "element_shell_effective_plastic_strain", "element_shell_thickness"),
    "solid": ("element_solid_node_indexes", "element_solid_part_indexes", "element_solid_stress",
              "element_solid_effective_plastic_strain", None),
    "tshell": ("element_tshell_node_indexes", "element_tshell_part_indexes",
               "element_tshell_stress", "element_tshell_effective_plastic_strain", None),
    "beam": ("element_beam_node_indexes", "element_beam_part_indexes", None, None, None),
}
# State arrays a run needs; lasso skips all others (memory of large models).
_STATE_ARRAYS = ["node_displacement", "timesteps"] + [
    name for arrays in _ELEMENT_ARRAYS.values() for name in arrays[2:] if name]


# ─────────────────────────────────────────────────────────────────────────────
# Keyword deck
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KeywordBlock:
    keyword: str            # upper case, without the long-format flag
    lines: List[str]        # data cards, comments removed
    width: int              # field width: 10 (standard) or 20 (long format)


def read_keyword_file(path, _visited: Optional[set] = None) -> List[KeywordBlock]:
    """Keyword blocks of ``path`` and of every file it includes (``*INCLUDE``)."""
    path = Path(path).resolve()
    visited = set() if _visited is None else _visited
    if path in visited:
        return []
    visited.add(path)
    blocks: List[KeywordBlock] = []
    long_format = False
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("$") or not line.strip():
            continue
        if line.startswith("*"):
            words = line.split()
            keyword = words[0].upper()
            if keyword == "*KEYWORD":
                long_format = any(w.upper() in ("LONG=Y", "LONG=S") for w in words[1:])
            width = 20 if long_format else 10
            if keyword.endswith("+"):
                keyword, width = keyword[:-1], 20
            elif keyword.endswith("-"):
                keyword, width = keyword[:-1], 10
            blocks.append(KeywordBlock(keyword, [], width))
        elif blocks:
            blocks[-1].lines.append(line)

    result: List[KeywordBlock] = []
    for block in blocks:
        if block.keyword == "*INCLUDE":
            for name in block.lines:
                result += read_keyword_file(path.parent / name.strip(), visited)
        elif block.keyword.startswith("*INCLUDE"):
            warnings.warn(f"{path}: {block.keyword} is not followed (not supported)")
        else:
            result.append(block)
    return result


def _fields(line: str, width: int) -> List[str]:
    """Fields of a card: comma separated (free format) or ``width`` characters each."""
    if "," in line:
        return [value.strip() for value in line.split(",")]
    return [line[i:i + width].strip() for i in range(0, len(line), width)]


def _ints(line: str, width: int, count: Optional[int] = None) -> List[int]:
    """Fields as integers (blank = 0); ``count`` pads or cuts to that many."""
    values = [int(float(value)) if value else 0 for value in _fields(line, width)]
    if count is None:
        return values
    return (values + [0] * count)[:count]


def _has_option(keyword: str, option: str) -> bool:
    return f"_{option}" in keyword


def _id_sets(blocks: List[KeywordBlock], entity: str = "NODE") -> Dict[int, set]:
    """IDs of every ``*SET_<entity>`` (NODE or PART): list, generate and add sets."""
    sets: Dict[int, set] = {}
    additions: Dict[int, List[int]] = {}
    prefix = f"*SET_{entity}"
    for block in blocks:
        keyword = block.keyword
        if keyword != prefix and not keyword.startswith(prefix + "_"):
            continue
        options = set(keyword[len(prefix):].strip("_").split("_")) - {""}
        if not options <= {"LIST", "GENERATE", "ADD", "TITLE"}:
            warnings.warn(f"{keyword} is not supported; its set is ignored")
            continue
        lines = block.lines[1:] if _has_option(keyword, "TITLE") else block.lines
        if not lines:
            continue
        set_id = _ints(lines[0], block.width, 1)[0]
        values = [v for line in lines[1:] for v in _ints(line, block.width)]
        if _has_option(keyword, "GENERATE"):
            ids = set()
            for begin, end in zip(values[0::2], values[1::2]):
                if begin > 0 and end >= begin:
                    ids.update(range(begin, end + 1))
            sets[set_id] = ids
        elif _has_option(keyword, "ADD"):
            additions[set_id] = [v for v in values if v > 0]
        else:
            sets[set_id] = {v for v in values if v > 0}

    def resolve(set_id: int, seen: set) -> set:
        if set_id in sets or set_id in seen:
            return sets.get(set_id, set())
        seen.add(set_id)
        return set().union(*(resolve(member, seen) for member in additions.get(set_id, [])))

    for set_id in additions:
        sets[set_id] = resolve(set_id, set())
    return sets


def rigid_part_ids(blocks: List[KeywordBlock]) -> set:
    """Part IDs whose material is ``*MAT_RIGID`` (``*MAT_020``)."""
    rigid_materials = set()
    for block in blocks:
        if block.keyword.startswith(("*MAT_RIGID", "*MAT_020")) \
                and "DISCRETE" not in block.keyword:
            title = _has_option(block.keyword, "TITLE")
            step = 4 if title else 3                  # (title card +) 3 cards per material
            for first in range(1 if title else 0, len(block.lines), step):
                rigid_materials.add(_ints(block.lines[first], block.width, 1)[0])

    parts = set()
    for block in blocks:
        keyword = block.keyword
        if keyword != "*PART" and not keyword.startswith("*PART_"):
            continue
        options = keyword[len("*PART"):].strip("_").split("_") if keyword != "*PART" else []
        extra = {"CONTACT": 1, "PRINT": 1, "ATTACHMENT": 1, "NODES": 0}
        if any(option not in extra for option in options):
            warnings.warn(f"{keyword} is not supported; its parts are not checked for rigidity")
            continue
        cards_per_part = 2 + sum(extra[option] for option in options)   # heading + card 1
        for heading in range(0, len(block.lines) - 1, cards_per_part):
            part_id, _, material_id = _ints(block.lines[heading + 1], block.width, 3)
            if material_id in rigid_materials:
                parts.add(part_id)
    return parts


def clamped_node_ids(blocks: List[KeywordBlock]) -> set:
    """Node IDs fixed in x, y and z by single-point constraints."""
    node_sets = _id_sets(blocks, "NODE")
    clamped = set()
    for block in blocks:
        keyword = block.keyword
        on_set = keyword.startswith("*BOUNDARY_SPC_SET")
        if not (on_set or keyword.startswith("*BOUNDARY_SPC_NODE")):
            continue
        with_id, with_birth = _has_option(keyword, "ID"), _has_option(keyword, "BIRTH")
        step = 1 + with_id + with_birth
        for first in range(int(with_id), len(block.lines), step):
            target, _, dof_x, dof_y, dof_z = _ints(block.lines[first], block.width, 5)
            if (dof_x, dof_y, dof_z) != (1, 1, 1):
                continue
            clamped |= node_sets.get(target, set()) if on_set else {target}
    return clamped


def _floats(line: str, width: int, count: int) -> List[float]:
    """First ``count`` fields as floats (blank = 0)."""
    values = [float(value) if value else 0.0 for value in _fields(line, width)[:count]]
    return values + [0.0] * (count - len(values))


def prescribed_motion(blocks: List[KeywordBlock]) -> Tuple[set, set]:
    """Node IDs and part IDs with ``*BOUNDARY_PRESCRIBED_MOTION_NODE/_SET/_RIGID``."""
    node_sets = _id_sets(blocks, "NODE")
    nodes, parts = set(), set()
    for block in blocks:
        keyword = block.keyword
        if not keyword.startswith("*BOUNDARY_PRESCRIBED_MOTION_"):
            continue
        options = keyword[len("*BOUNDARY_PRESCRIBED_MOTION_"):].split("_")
        if options[0] not in ("NODE", "SET", "RIGID") or not set(options[1:]) <= {"ID", "LOCAL"}:
            warnings.warn(f"{keyword} is not supported; its nodes are not marked as prescribed")
            continue
        with_id = "ID" in options
        for first in range(int(with_id), len(block.lines), 1 + with_id):
            target = _ints(block.lines[first], block.width, 1)[0]
            if options[0] == "NODE":
                nodes.add(target)
            elif options[0] == "SET":
                nodes |= node_sets.get(target, set())
            else:
                parts.add(target)
    return nodes, parts


def initial_velocities(blocks: List[KeywordBlock]) -> List[Tuple[str, int, np.ndarray]]:
    """Initial velocity definitions as (target kind, target ID, (vx, vy, vz)).

    Target kinds: ``all`` (every node), ``node``, ``node_set``, ``part``,
    ``part_set``. Applied in this order, so the later, more specific ones win:
    ``*INITIAL_VELOCITY``, ``*INITIAL_VELOCITY_GENERATION``,
    ``*INITIAL_VELOCITY_RIGID_BODY``, ``*INITIAL_VELOCITY_NODE``. Rotations and
    local coordinate systems are not supported (warning).
    """
    found: Dict[str, List[Tuple[str, int, np.ndarray]]] = {
        "base": [], "generation": [], "rigid": [], "node": []}
    for block in blocks:
        keyword, lines, width = block.keyword, block.lines, block.width
        if keyword == "*INITIAL_VELOCITY" and len(lines) >= 2:   # one definition
            set_id, exempt, box = _ints(lines[0], width, 3)
            if box or exempt:
                warnings.warn(f"{keyword}: box / exempted nodes are not supported")
            velocity = np.array(_floats(lines[1], width, 3))
            found["base"].append(("node_set" if set_id else "all", set_id, velocity))
        elif keyword == "*INITIAL_VELOCITY_GENERATION":
            for first in range(0, len(lines) - 1, 2):   # 2 cards per definition
                target, kind = _ints(lines[first], width, 2)
                omega, vx, vy, vz = _floats(lines[first], width, 6)[2:6]
                if omega:
                    warnings.warn(f"{keyword}: rotation (OMEGA) is ignored")
                target_kind = {1: "part_set", 2: "part", 3: "node_set"}.get(kind)
                if target_kind is None:
                    warnings.warn(f"{keyword}: STYP {kind} is not supported")
                    continue
                found["generation"].append((target_kind, target, np.array([vx, vy, vz])))
        elif keyword in ("*INITIAL_VELOCITY_NODE", "*INITIAL_VELOCITY_RIGID_BODY"):
            group = "node" if keyword.endswith("NODE") else "rigid"
            for line in lines:
                target = _ints(line, width, 1)[0]
                velocity = np.array(_floats(line, width, 4)[1:4])
                found[group].append((group if group == "node" else "part", target, velocity))
        elif keyword.startswith("*INITIAL_VELOCITY"):
            warnings.warn(f"{keyword} is not supported; it is ignored")
    return found["base"] + found["generation"] + found["rigid"] + found["node"]


def rigid_walls(blocks: List[KeywordBlock]) -> np.ndarray:
    """Stationary planar rigid walls as (W, 6): point on the wall and unit normal.

    The normal points from the tail to the head point (XT, YT, ZT -> XH, YH, ZH),
    i.e. to the side where the structure is. Finite walls are treated as
    infinite planes; moving walls and other wall types are not supported.
    """
    walls = []
    for block in blocks:
        keyword = block.keyword
        if keyword.startswith("*RIGIDWALL") and not keyword.startswith("*RIGIDWALL_PLANAR"):
            warnings.warn(f"{keyword} is not supported; the wall is ignored")
            continue
        if not keyword.startswith("*RIGIDWALL_PLANAR"):
            continue
        options = set(keyword[len("*RIGIDWALL_PLANAR"):].strip("_").split("_")) - {""}
        if "MOVING" in options or not options <= {"ID", "FINITE", "ORTHO", "FORCES"}:
            warnings.warn(f"{keyword} is not supported; the wall is ignored")
            continue
        extra = ("FINITE" in options) + 2 * ("ORTHO" in options) + ("FORCES" in options)
        cards = 2 + ("ID" in options) + extra
        for first in range(int("ID" in options), len(block.lines) - 1, cards):
            tail_head = np.array(_floats(block.lines[first + 1], block.width, 6))
            normal = tail_head[3:] - tail_head[:3]
            length = np.linalg.norm(normal)
            if length == 0:
                warnings.warn(f"{keyword}: tail and head coincide; the wall is ignored")
                continue
            walls.append(np.concatenate([tail_head[:3], normal / length]))
    return np.array(walls, dtype=np.float32).reshape(-1, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Fields
# ─────────────────────────────────────────────────────────────────────────────

def von_mises(stress: np.ndarray) -> np.ndarray:
    """(..., 6) Voigt stress [xx, yy, zz, xy, yz, zx] -> (...) von Mises stress."""
    xx, yy, zz, xy, yz, zx = np.moveaxis(stress, -1, 0)
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2)
                   + 3.0 * (xy ** 2 + yz ** 2 + zx ** 2))


def distinct_corners(nodes: np.ndarray) -> np.ndarray:
    """(E, k) bool: corner holds a node not already listed before it in the element."""
    distinct = np.ones(nodes.shape, dtype=bool)
    for corner in range(1, nodes.shape[1]):
        distinct[:, corner] = (nodes[:, :corner] != nodes[:, corner:corner + 1]).all(1)
    return distinct


class NodalAverage:
    """Mean of element values at the nodes, over every element block that has them."""

    def __init__(self, num_states: int, num_nodes: int):
        self.total = np.zeros((num_states, num_nodes), dtype=np.float64)
        self.count = np.zeros(num_nodes, dtype=np.int64)

    def add(self, values: np.ndarray, nodes: np.ndarray) -> None:
        """``values`` (T, E) of elements with node indexes ``nodes`` (E, k)."""
        distinct = distinct_corners(nodes)
        element = np.nonzero(distinct)[0]
        node = nodes[distinct]
        order = np.argsort(node, kind="stable")
        node, element = node[order], element[order]
        starts = np.flatnonzero(np.r_[True, node[1:] != node[:-1]])
        self.total[:, node[starts]] += np.add.reduceat(values[:, element], starts, axis=1)
        self.count += np.bincount(node, minlength=self.count.shape[0])

    def result(self) -> Optional[np.ndarray]:
        """(T, N) float32 mean; 0 at nodes without values; None if nothing was added."""
        if not self.count.any():
            return None
        return (self.total / np.maximum(self.count, 1)).astype(np.float32)


def nodal_mean(values: np.ndarray, nodes: np.ndarray, num_nodes: int) -> np.ndarray:
    """(T, E) element values -> (T, N) mean over the elements at each node."""
    average = NodalAverage(values.shape[0], num_nodes)
    average.add(values, nodes)
    return average.result()


# ─────────────────────────────────────────────────────────────────────────────
# Runs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ElementBlock:
    nodes: np.ndarray       # (E, k) node indexes; degenerate elements repeat nodes
    part_ids: np.ndarray    # (E,)   user part IDs


@dataclass
class D3plotRun:
    node_ids: np.ndarray                    # (N,)       user node IDs
    mesh_pos: np.ndarray                    # (N, 3)     rest configuration
    world_pos: Optional[np.ndarray]         # (T, N, 3)  state coordinates; None: in blocks
    times: Optional[np.ndarray]             # (T,)
    node_type: np.ndarray                   # (N,)       NODE_TYPE_*
    elements: Dict[str, ElementBlock] = field(default_factory=dict)
    von_mises: Optional[np.ndarray] = None       # (T, N)
    plastic_strain: Optional[np.ndarray] = None  # (T, N)
    thickness: Optional[np.ndarray] = None       # (T, N) shells only
    globals: Optional[Dict[str, float]] = None   # from globals.json
    node_static: Dict[str, np.ndarray] = field(default_factory=dict)   # name -> (N,)
    walls: Optional[np.ndarray] = None           # (W, 6) point + unit normal


def keyword_files(run_dir: Path) -> List[Path]:
    """Keyword decks in ``run_dir`` (``.k``, ``.key``, ``.dyn``)."""
    return sorted(p for p in Path(run_dir).iterdir()
                  if p.is_file() and p.suffix.lower() in KEYWORD_SUFFIXES)


def _state_rows(arrays, element_type: str, part_index: np.ndarray, state_array) -> np.ndarray:
    """Elements that have rows in ``state_array``; rigid shells are left out by LS-DYNA."""
    rows = np.ones(part_index.shape[0], dtype=bool)
    if state_array.shape[1] != part_index.shape[0]:
        material_type = arrays.get("part_material_type")
        if element_type != "shell" or material_type is None:
            raise ValueError(f"{element_type} state data hold {state_array.shape[1]} of "
                             f"{part_index.shape[0]} elements")
        rows = material_type[part_index] != RIGID_MATERIAL_TYPE
        if rows.sum() != state_array.shape[1]:
            raise ValueError("shell state data do not match the deformable shells")
    return rows


@dataclass
class StateBlock:
    """Consecutive selected states of a run."""
    times: np.ndarray                            # (t,)
    world_pos: np.ndarray                        # (t, N, 3)  state coordinates
    von_mises: Optional[np.ndarray] = None       # (t, N)
    plastic_strain: Optional[np.ndarray] = None  # (t, N)
    thickness: Optional[np.ndarray] = None       # (t, N) shells only


def read_run(run_dir, frames: Optional[Sequence[int]] = None,
             layer_reduction: str = "mean") -> D3plotRun:
    """One LS-DYNA run with all its states in memory; ``frames`` selects states
    (default: all).

    ``layer_reduction`` combines the integration points of an element
    (through the thickness of shells): ``mean`` or ``max``.
    """
    run, blocks = read_run_blocks(run_dir, frames, layer_reduction)
    states = list(blocks)

    def joined(name):
        parts = [getattr(block, name) for block in states]
        if parts[0] is None:
            return None
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    run.times, run.world_pos = joined("times"), joined("world_pos")
    run.von_mises, run.plastic_strain = joined("von_mises"), joined("plastic_strain")
    run.thickness = joined("thickness")
    return run


def read_run_blocks(run_dir, frames: Optional[Sequence[int]] = None,
                    layer_reduction: str = "mean", max_block_bytes: Optional[int] = None
                    ) -> Tuple[D3plotRun, Iterator[StateBlock]]:
    """The static data of a run plus an iterator over its selected states in blocks.

    The ``D3plotRun`` holds no states (``world_pos`` and the nodal fields are
    None); the iterator (consumed once) yields ``StateBlock``s of at most
    ``max_block_bytes`` while being read (estimated from the d3plot header;
    None: one block), so a large run need not fit into memory. The first
    block is read here: the node types need the geometry, the static
    thickness the first state.
    """
    if layer_reduction not in LAYER_REDUCTIONS:
        raise ValueError(f"layer_reduction must be one of {LAYER_REDUCTIONS}")
    reduce = np.mean if layer_reduction == "mean" else np.max
    run_dir = Path(run_dir)
    if frames is None and not max_block_bytes:
        chunks: List[Optional[List[int]]] = [None]   # all states, number unknown
    else:
        selected = (list(range(num_states(run_dir))) if frames is None
                    else [int(f) for f in frames])
        size = (max(1, int(max_block_bytes // _state_bytes(run_dir))) if max_block_bytes
                else len(selected))
        chunks = [selected[i:i + size] for i in range(0, len(selected), size)]
    arrays = _read_states(run_dir, chunks[0])
    run = _static_run(run_dir, arrays)
    held = [_state_block(arrays, run, reduce)]   # handed out once, then released
    del arrays
    if held[0].thickness is not None:   # shell thickness at the first state
        run.node_static["thickness"] = held[0].thickness[0]

    def blocks() -> Iterator[StateBlock]:
        yield held.pop()
        for chunk in chunks[1:]:
            yield _state_block(_read_states(run_dir, chunk), run, reduce)

    return run, blocks()


def _state_bytes(run_dir: Path) -> int:
    """Rough memory of one state while it is read: lasso arrays plus derived fields."""
    from lasso.dyna import D3plotHeader

    header = D3plotHeader(str(run_dir / "d3plot"))
    words = (3 * header.n_nodes + header.n_global_vars
             + header.n_solids * header.n_solid_vars
             + header.n_thick_shells * header.n_thick_shell_vars
             + header.n_beams * header.n_beam_vars + header.n_shells * header.n_shell_vars)
    return 3 * words * header.wordsize


def _read_states(run_dir: Path, states: Optional[List[int]]):
    """lasso arrays of the geometry and of ``states`` (None: all states)."""
    from lasso.dyna import D3plot

    state_filter = None if states is None else set(states)
    # lasso applies a state filter only with buffered reading, which it switches on
    # by any(state_filter) — False for {0} alone; so it is switched on here.
    arrays = D3plot(str(run_dir / "d3plot"), state_filter=state_filter,
                    state_array_filter=_STATE_ARRAYS,
                    buffered_reading=state_filter is not None).arrays
    if "node_displacement" not in arrays:
        raise ValueError(f"{run_dir}: the d3plot holds no state coordinates (IU = 0)")
    found = arrays["node_displacement"].shape[0]
    if state_filter is not None and found != len(state_filter):
        raise ValueError(f"{run_dir}: {found} of {len(state_filter)} requested states found")
    return arrays


def _state_block(arrays, run: D3plotRun, reduce) -> StateBlock:
    """Positions and nodal fields of the states in ``arrays``."""
    world_pos = arrays["node_displacement"]
    num_states, num_nodes = world_pos.shape[:2]
    fields = {name: NodalAverage(num_states, num_nodes)
              for name in ("von_mises", "plastic_strain", "thickness")}
    for element_type, (_, parts_name, stress_name, strain_name, thickness_name) \
            in _ELEMENT_ARRAYS.items():
        if element_type not in run.elements:
            continue
        nodes = run.elements[element_type].nodes
        part_index = arrays[parts_name]
        for name, array_name, per_layer in (("von_mises", stress_name, True),
                                            ("plastic_strain", strain_name, True),
                                            ("thickness", thickness_name, False)):
            values = arrays.get(array_name) if array_name else None
            if values is None:
                continue
            if name == "von_mises":
                values = von_mises(values)
            if per_layer and values.ndim == 3:
                values = reduce(values, axis=2)                      # (T, E_state)
            rows = _state_rows(arrays, element_type, part_index, values)
            fields[name].add(values, nodes[rows])
    return StateBlock(times=np.asarray(arrays.get("timesteps", np.arange(num_states))),
                      world_pos=world_pos.astype(np.float32),
                      von_mises=fields["von_mises"].result(),
                      plastic_strain=fields["plastic_strain"].result(),
                      thickness=fields["thickness"].result())


def _static_run(run_dir: Path, arrays) -> D3plotRun:
    """Geometry, elements, node types, static node features, walls and run
    parameters of a run (deck and d3plot geometry), without states."""
    mesh_pos = arrays["node_coordinates"]
    num_nodes = mesh_pos.shape[0]
    node_ids = arrays.get("node_ids", np.arange(1, num_nodes + 1))
    part_ids = arrays.get("part_ids", arrays.get("part_titles_ids"))

    blocks = [b for deck in keyword_files(run_dir) for b in read_keyword_file(deck)]
    if not blocks:
        warnings.warn(f"{run_dir}: no keyword deck — clamped nodes unknown, rigid parts "
                      "only from the d3plot material types")
    rigid_parts = rigid_part_ids(blocks)
    material_type = arrays.get("part_material_type")
    if material_type is not None and part_ids is not None:
        rigid_parts |= set(part_ids[material_type == RIGID_MATERIAL_TYPE].tolist())

    run = D3plotRun(node_ids=node_ids, mesh_pos=mesh_pos.astype(np.float32),
                    world_pos=None, times=None,
                    node_type=np.full(num_nodes, NODE_TYPE_NORMAL, dtype=np.int32))
    for element_type, (nodes_name, parts_name, *_) in _ELEMENT_ARRAYS.items():
        nodes = arrays.get(nodes_name)
        if nodes is None or nodes.shape[0] == 0:
            continue
        nodes = nodes[:, :2] if element_type == "beam" else nodes   # end nodes, no orientation
        part_index = arrays[parts_name]
        block_part_ids = part_ids[part_index] if part_ids is not None else part_index + 1
        run.elements[element_type] = ElementBlock(nodes=nodes, part_ids=block_part_ids)
        run.node_type[np.unique(nodes[np.isin(block_part_ids, list(rigid_parts))])] = \
            NODE_TYPE_ACTUATOR

    index_of_id = {node_id: index for index, node_id in enumerate(node_ids.tolist())}

    def node_indexes(ids) -> List[int]:
        return [index_of_id[i] for i in ids if i in index_of_id]

    def nodes_of_parts(ids) -> np.ndarray:
        ids = list(ids)
        return np.unique(np.concatenate(
            [block.nodes[np.isin(block.part_ids, ids)].ravel()
             for block in run.elements.values()] + [np.empty(0, dtype=np.int64)])).astype(np.int64)

    # Prescribed motion: the motion is taken as given, like that of rigid parts.
    prescribed_nodes, prescribed_parts = prescribed_motion(blocks)
    run.node_type[node_indexes(prescribed_nodes)] = NODE_TYPE_ACTUATOR
    run.node_type[nodes_of_parts(prescribed_parts)] = NODE_TYPE_ACTUATOR
    run.node_type[node_indexes(clamped_node_ids(blocks))] = NODE_TYPE_CLAMPED

    definitions = initial_velocities(blocks)
    if definitions:
        node_sets, part_sets = _id_sets(blocks, "NODE"), _id_sets(blocks, "PART")
        velocity = np.zeros((num_nodes, 3), dtype=np.float32)
        for kind, target, value in definitions:
            nodes = {"all": lambda: slice(None),
                     "node": lambda: node_indexes([target]),
                     "node_set": lambda: node_indexes(node_sets.get(target, set())),
                     "part": lambda: nodes_of_parts([target]),
                     "part_set": lambda: nodes_of_parts(part_sets.get(target, set()))}[kind]()
            velocity[nodes] = value
        for axis, name in enumerate("xyz"):
            run.node_static[f"initial_velocity_{name}"] = velocity[:, axis]
    walls = rigid_walls(blocks)
    run.walls = walls if len(walls) else None

    if (run_dir / GLOBALS_FILE).exists():
        with open(run_dir / GLOBALS_FILE) as f:
            run.globals = {str(k): float(v) for k, v in json.load(f).items()}
    return run


def num_states(run_dir) -> int:
    """Number of states of a run, without reading the state data."""
    from lasso.dyna import D3plot

    arrays = D3plot(str(Path(run_dir) / "d3plot"), state_array_filter=["timesteps"]).arrays
    return int(arrays["timesteps"].shape[0])


def degenerate_solids(nodes: np.ndarray) -> List[np.ndarray]:
    """Masks of the 8-node solids (E, 8) that are the shapes of ``DEGENERATE_SOLIDS``."""
    tetra = (nodes[:, 3:] == nodes[:, 3:4]).all(1)
    pyramid = ~tetra & (nodes[:, 4:] == nodes[:, 4:5]).all(1)
    wedge = ~tetra & ~pyramid & (nodes[:, 4] == nodes[:, 5]) & (nodes[:, 6] == nodes[:, 7])
    return [tetra, pyramid, wedge]


def vtk_cells(elements: Dict[str, ElementBlock]) -> Tuple[np.ndarray, np.ndarray]:
    """All elements as VTK cells: (C, W) node indexes padded with -1, (C,) VTK types.

    LS-DYNA writes triangles as n1 n2 n3 n3 and solids as 8-node elements;
    tetrahedra (n1 n2 n3 n4 n4 n4 n4 n4), pyramids (n1 .. n5 n5 n5 n5) and wedges
    (n1 n2 n3 n4 n5 n5 n6 n6) are recognized from the repeated nodes.
    """
    rows: List[np.ndarray] = []
    types: List[np.ndarray] = []

    def add(nodes: np.ndarray, cell_type: int) -> None:
        if len(nodes):
            rows.append(np.pad(nodes, ((0, 0), (0, 8 - nodes.shape[1])), constant_values=-1))
            types.append(np.full(len(nodes), cell_type, dtype=np.uint8))

    for element_type, block in elements.items():
        nodes = block.nodes
        if element_type == "beam":
            add(nodes[:, :2], CELL_LINE)
        elif element_type == "shell":
            triangle = nodes[:, 3] == nodes[:, 2]
            add(nodes[triangle, :3], CELL_TRIANGLE)
            add(nodes[~triangle], CELL_QUAD)
        else:                                   # solids and thick shells: 8 nodes
            shapes = degenerate_solids(nodes)
            for (cell_type, corners), mask in zip(DEGENERATE_SOLIDS, shapes):
                add(nodes[mask][:, corners], cell_type)
            add(nodes[~np.any(shapes, axis=0)], CELL_HEXAHEDRON)
    cells, cell_types = np.concatenate(rows), np.concatenate(types)
    for cell_type in np.unique(cell_types).tolist():
        corners = cells[cell_types == cell_type, :CELL_NUM_NODES[cell_type]]
        if not distinct_corners(corners).all():
            raise ValueError(f"cells of VTK type {cell_type} with repeated nodes "
                             "(unsupported degenerate element)")
    width = max(CELL_NUM_NODES[t] for t in np.unique(cell_types).tolist())
    return cells[:, :width], cell_types


def _raw_cells(run: D3plotRun, name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Cells of a run for ``RawTrajectory``; pure tetrahedral meshes keep the (C, 4) layout."""
    if not run.elements:
        raise ValueError(f"{name}: no elements")
    cells, cell_types = vtk_cells(run.elements)
    if (cell_types == CELL_TETRA).all():
        cells, cell_types = cells[:, :4], None
    return cells.astype(np.int32), cell_types


def as_raw_trajectory(run: D3plotRun, name: str = "run") -> RawTrajectory:
    """``RawTrajectory`` of a run read with ``read_run`` (all states in memory)."""
    cells, cell_types = _raw_cells(run, name)
    if run.von_mises is None:
        raise ValueError(f"{name}: no stress in the d3plot")
    return RawTrajectory(node_type=run.node_type, mesh_pos=run.mesh_pos,
                         cells=cells, world_pos=run.world_pos,
                         stress=run.von_mises, cell_types=cell_types, globals=run.globals,
                         node_static=run.node_static or None, walls=run.walls)


def read_raw_run(run_dir, frames: Optional[Sequence[int]] = None,
                 max_block_bytes: Optional[int] = None) -> RawTrajectory:
    """A run as ``RawTrajectory`` for ``export.py``, its states restricted to ``frames``
    and read in blocks of about ``max_block_bytes`` (``read_run_blocks``)."""
    run, blocks = read_run_blocks(run_dir, frames, max_block_bytes=max_block_bytes)
    cells, cell_types = _raw_cells(run, str(run_dir))

    def frame_blocks():
        for block in blocks:
            if block.von_mises is None:
                raise ValueError(f"{run_dir}: no stress in the d3plot")
            yield block.world_pos, block.von_mises

    return RawTrajectory(node_type=run.node_type, mesh_pos=run.mesh_pos, cells=cells,
                         world_pos=None, stress=None, cell_types=cell_types,
                         globals=run.globals, node_static=run.node_static or None,
                         walls=run.walls, frame_blocks=frame_blocks())


def run_dirs(raw_dir, source: str) -> List[Path]:
    """Folders with a ``d3plot`` below ``raw_dir/<source>`` (``source`` may name the
    TFRecord the runs were converted from), or below ``raw_dir`` if that does not exist."""
    root = Path(raw_dir) / Path(source).stem
    root = root if root.is_dir() else Path(raw_dir)
    runs = sorted({path.parent for path in root.rglob("d3plot")})
    if not runs:
        raise FileNotFoundError(f"no runs (folders with a d3plot) below {root}")
    return runs
