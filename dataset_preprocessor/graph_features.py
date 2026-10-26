"""
graph_features.py — Feature layout and graph construction for deforming_plate.

Shared by the exporter (statistics), the dataset (one graph per time step),
the trainer (position noise, scheduled sampling) and the rollout (graph
rebuilt from predicted positions), so every stage builds identical graphs.

Features per node and time step t (MeshGraphNets, Pfaff et al. 2021, App. A.1)
------------------------------------------------------------------------------
``x`` (6)
    one-hot node type [normal, actuator, clamped] + ``bc_velocity`` (3).
    ``bc_velocity`` = x_{t+1} - x_t on the scripted actuator nodes, 0 elsewhere.
``y`` (4)
    velocity x_{t+1} - x_t (3) + von-Mises stress at t+1 (1).
``edge_attr`` (8), mesh edges from the cells (tetrahedra, or mixed VTK cell
types: lines, triangles, quadrilaterals, tetrahedra, pyramids, wedges, hexahedra)
    u_ij, |u_ij| (mesh space) + x_ij, |x_ij| (world space).
``world_edge_attr`` (4), world edges
    x_ij, |x_ij| for node pairs closer than the contact radius r_W. Which pairs
    qualify is selected by ``world_edges`` (see ``WORLD_EDGE_MODES``); the
    paper takes all pairs that are not already mesh neighbours.
``loss_mask`` (N)
    every node except the actuator, whose motion is prescribed.

Relative vectors point from sender to receiver: v_ij = v[sender] - v[receiver].

Optional history (``history = 1``, not in the paper)
----------------------------------------------------
Appends ``velocity_{x,y,z}_prev`` = x_t - x_{t-1} and, with ``prev_stress``,
``stress_prev`` = σ_t. Each ``<name>_prev`` column pairs with the target
``<name>``; that pairing drives ``residual_target``, scheduled sampling and the
rollout's feedback of predictions. The first frame has no predecessor, so
samples start at t = 1.

Optional columns of general raw data (not in the paper), in this order
-------------------------------------------------------------------
* static node features of a run (e.g. ``thickness``, ``initial_velocity_x``),
  standardized over the train nodes;
* run parameters (e.g. impact velocity) as constant columns ``global_<name>``,
  standardized over the train trajectories;
* rigid walls, always the last columns: signed distance to the nearest planar
  wall, clipped to ±``WALL_DISTANCE_CUTOFF`` x contact radius, and its unit
  normal. They depend on the positions and are rebuilt with the edges.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ── raw node types (DeepMind ``node_type``) ─────────────────────────────────
NODE_TYPE_NORMAL = 0     # free plate nodes: motion is predicted
NODE_TYPE_ACTUATOR = 1   # scripted obstacle that pushes the plate
NODE_TYPE_CLAMPED = 3    # fixed boundary nodes
_ONE_HOT_COLUMN = {NODE_TYPE_NORMAL: 0, NODE_TYPE_ACTUATOR: 1, NODE_TYPE_CLAMPED: 2}
NUM_NODE_TYPES = len(_ONE_HOT_COLUMN)
# Lookup table raw node type -> one-hot column; -1 marks types that must not occur.
_COLUMN_OF_RAW_TYPE = torch.full((max(_ONE_HOT_COLUMN) + 1,), -1, dtype=torch.long)
_COLUMN_OF_RAW_TYPE[list(_ONE_HOT_COLUMN)] = torch.tensor(list(_ONE_HOT_COLUMN.values()))

# ── feature names (stored in metadata.json; order = column order) ───────────
NODE_TYPE_NAMES = ["node_type_0", "node_type_1", "node_type_2"]
BC_VELOCITY_NAMES = ["bc_velocity_x", "bc_velocity_y", "bc_velocity_z"]
PREV_VELOCITY_NAMES = ["velocity_x_prev", "velocity_y_prev", "velocity_z_prev"]
PREV_STRESS_NAMES = ["stress_prev"]
TARGET_FEATURE_NAMES = ["velocity_x", "velocity_y", "velocity_z", "stress"]
GLOBAL_FEATURE_PREFIX = "global_"   # node feature columns of run parameters
WALL_FEATURE_NAMES = ["wall_distance", "wall_normal_x", "wall_normal_y", "wall_normal_z"]
WALL_FEATURE_DIM = len(WALL_FEATURE_NAMES)
WALL_DISTANCE_CUTOFF = 5.0          # x contact radius; also the distance without a wall
MESH_EDGE_FEATURE_NAMES = [
    "mesh_dx", "mesh_dy", "mesh_dz", "mesh_dist",
    "world_dx", "world_dy", "world_dz", "world_dist",
]
WORLD_EDGE_FEATURE_NAMES = ["world_dx", "world_dy", "world_dz", "world_dist"]

# ── cell types (VTK numbering, also used for the .vtu export) ───────────────
CELL_LINE, CELL_TRIANGLE, CELL_QUAD = 3, 5, 9
CELL_TETRA, CELL_HEXAHEDRON, CELL_WEDGE, CELL_PYRAMID = 10, 12, 13, 14
CELL_NUM_NODES = {CELL_LINE: 2, CELL_TRIANGLE: 3, CELL_QUAD: 4, CELL_TETRA: 4,
                  CELL_PYRAMID: 5, CELL_WEDGE: 6, CELL_HEXAHEDRON: 8}
# Edges of each cell type as pairs of local node indexes (VTK node order).
CELL_EDGES = {
    CELL_LINE: [(0, 1)],
    CELL_TRIANGLE: [(0, 1), (1, 2), (2, 0)],
    CELL_QUAD: [(0, 1), (1, 2), (2, 3), (3, 0)],
    CELL_TETRA: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    CELL_PYRAMID: [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)],
    CELL_WEDGE: [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)],
    CELL_HEXAHEDRON: [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                      (0, 4), (1, 5), (2, 6), (3, 7)],
}

# ── world edge variants (config ``[model] world_edges``) ────────────────────
# all                 every pair within r_W that is not a mesh edge (paper, Sec. 3.1)
# contact             only pairs of different bodies (connected parts of the mesh)
# self_contact_aware  every pair within r_W that is not connected through the
#                     material: contact between bodies plus self-contact,
#                     without the static near neighbours a radius larger than the
#                     edge length adds inside a body. Connected: the shortest
#                     path along the mesh edges in the rest state is shorter
#                     than ``exclude_distance`` x r_W. Counting mesh edges
#                     instead (``exclude_hops``, configs up to v0.3.0) fails
#                     where the mesh is locally refined: there, neighbours in the
#                     material lie many edges apart.
WORLD_EDGE_MODES = ("all", "contact", "self_contact_aware")
# x r_W; on a mesh with edges of r_W / 2 the same pairs as 4 mesh edges
DEFAULT_WORLD_EDGE_EXCLUDE_DISTANCE = 2.0
# Static per-sample attributes that select the variant (see world_edge_selection).
WORLD_SELECTION_KEYS = ("body_id", "world_exclude_index")

TARGET_DIM = len(TARGET_FEATURE_NAMES)                    # 4
MESH_EDGE_FEATURE_DIM = len(MESH_EDGE_FEATURE_NAMES)      # 8
WORLD_EDGE_FEATURE_DIM = len(WORLD_EDGE_FEATURE_NAMES)    # 4
VELOCITY_TARGET_COLUMNS = [0, 1, 2]                       # velocity_x/y/z in ``y``


# ─────────────────────────────────────────────────────────────────────────────
# Feature layout
# ─────────────────────────────────────────────────────────────────────────────

def node_feature_names(history: int = 0, prev_stress: bool = True,
                       global_names: Sequence[str] = (), static_names: Sequence[str] = (),
                       walls: bool = False) -> List[str]:
    """Column names of ``x`` for the given layout (see module docstring)."""
    if history not in (0, 1):
        raise ValueError(f"history must be 0 or 1, got {history!r}")
    names = NODE_TYPE_NAMES + BC_VELOCITY_NAMES
    if history:
        names += PREV_VELOCITY_NAMES + (PREV_STRESS_NAMES if prev_stress else [])
    names += list(static_names) + [f"{GLOBAL_FEATURE_PREFIX}{name}" for name in global_names]
    return names + (WALL_FEATURE_NAMES if walls else [])


def wall_features(pos: torch.Tensor, wall_planes: torch.Tensor, cutoff: float,
                  batch: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Signed distance to the nearest wall and its unit normal per node -> (N, 4).

    ``wall_planes`` (G * W, 6) holds W walls (point, unit normal) per graph;
    rows with a zero normal are no walls. The distance is clipped to
    ±``cutoff``; nodes of a graph without walls get (cutoff, 0, 0, 0).
    """
    graph = (torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device) if batch is None
             else batch.to(pos.device))
    num_graphs = int(graph.max()) + 1 if graph.numel() else 1
    planes = wall_planes.to(pos.device).view(num_graphs, -1, 6)[graph]      # (N, W, 6)
    normal = planes[..., 3:]
    distance = ((pos[:, None, :] - planes[..., :3]) * normal).sum(-1)       # (N, W)
    valid = normal.abs().sum(-1) > 0
    nearest = torch.where(valid, distance.abs(), torch.inf).argmin(1)
    rows = torch.arange(pos.shape[0], device=pos.device)
    has_wall = valid.any(1)
    nearest_distance = torch.where(has_wall, distance[rows, nearest].clamp(-cutoff, cutoff),
                                   torch.full_like(distance[:, 0], cutoff))
    return torch.cat((nearest_distance[:, None], normal[rows, nearest] * has_wall[:, None]), -1)


def prev_to_target_columns(node_names: Sequence[str],
                           target_names: Sequence[str]) -> Dict[int, int]:
    """Map each ``<name>_prev`` column of ``x`` to the column of target ``<name>``.

    Empty for the paper layout (no history). Used wherever a prediction is fed
    back as the next step's ``*_prev`` input.
    """
    mapping = {}
    for column, name in enumerate(node_names):
        base = name[:-len("_prev")]
        if name.endswith("_prev") and base in target_names:
            mapping[column] = list(target_names).index(base)
    return mapping


def one_hot_node_type(raw_node_type: torch.Tensor) -> torch.Tensor:
    """(N,) raw node types {0, 1, 3} -> (N, 3) float one-hot [normal, actuator, clamped]."""
    raw = raw_node_type.long().cpu()
    known = (raw >= 0) & (raw < len(_COLUMN_OF_RAW_TYPE))
    columns = torch.full_like(raw, -1)
    columns[known] = _COLUMN_OF_RAW_TYPE[raw[known]]
    if (columns < 0).any():
        unknown = sorted(set(raw[columns < 0].tolist()))
        raise ValueError(f"unknown node types {unknown}; expected {sorted(_ONE_HOT_COLUMN)}")
    return torch.eye(NUM_NODE_TYPES)[columns].to(raw_node_type.device)


def free_node_mask(x: torch.Tensor) -> torch.Tensor:
    """Normal nodes, read from the one-hot column of ``x``.

    Only these nodes move by the predicted velocity; actuator and clamped nodes
    follow their prescribed (kinematic) motion. The one-hot columns are
    normalized with mean 0 / std 1, so they are still exactly 0 or 1.
    """
    return x[:, _ONE_HOT_COLUMN[NODE_TYPE_NORMAL]] > 0.5


def structure_node_mask(x: torch.Tensor) -> torch.Tensor:
    """Nodes of the simulated structure (normal and clamped), i.e. all but the
    actuator — the nodes of ``loss_mask``, read from the one-hot column of ``x``."""
    return x[:, _ONE_HOT_COLUMN[NODE_TYPE_ACTUATOR]] < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Edges
# ─────────────────────────────────────────────────────────────────────────────

def relative_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """pos[sender] - pos[receiver] and its length -> (E, 4)."""
    offset = pos[edge_index[0]] - pos[edge_index[1]]
    return torch.cat((offset, torch.linalg.norm(offset, dim=-1, keepdim=True)), dim=-1)


def mesh_edges_from_cells(cells, cell_types=None) -> torch.Tensor:
    """Mesh edges (2, E) of the cells.

    Without ``cell_types`` the cells are tetrahedra (C, 4) and each of the six
    vertex pairs is an edge. With ``cell_types`` (C,) (VTK types, see
    ``CELL_EDGES``) every row of ``cells`` (C, W) holds the nodes of one cell
    in VTK order, padded with -1. Edges run in both directions; edges shared
    by neighbouring cells appear once. Sorted by (sender, receiver).
    """
    cells = torch.from_numpy(np.asarray(cells, dtype=np.int64))
    if cell_types is None:
        first, second = torch.triu_indices(4, 4, offset=1)   # the 6 vertex pairs
        a, b = cells[:, first].reshape(-1), cells[:, second].reshape(-1)
    else:
        cell_types = np.asarray(cell_types)
        senders, receivers = [], []
        for cell_type in np.unique(cell_types).tolist():
            if cell_type not in CELL_EDGES:
                raise ValueError(f"unsupported cell type {cell_type}")
            local = torch.tensor(CELL_EDGES[cell_type])
            of_type = cells[torch.from_numpy(cell_types == cell_type)]
            senders.append(of_type[:, local[:, 0]].reshape(-1))
            receivers.append(of_type[:, local[:, 1]].reshape(-1))
        a, b = torch.cat(senders), torch.cat(receivers)
    num_nodes = int(cells.max()) + 1 if cells.numel() else 0
    if num_nodes == 0:
        return torch.empty(2, 0, dtype=torch.long)
    # One int64 key per edge: the unique keys come sorted by (sender, receiver),
    # like unique(dim=1), but ~30x faster and with a sixth of the memory.
    keys = torch.unique(torch.cat((a * num_nodes + b, b * num_nodes + a)))
    return torch.stack((keys // num_nodes, keys % num_nodes))


def mesh_components(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Body label (N,) of every node: the smallest node index of its connected
    part of the mesh. ``edge_index`` must hold both directions of every edge."""
    labels = torch.arange(num_nodes, device=edge_index.device)
    senders, receivers = edge_index
    while True:
        # Smallest label among self and neighbours, then jump to that label's label.
        new = labels.scatter_reduce(0, receivers, labels[senders], reduce="amin")
        new = new[new]
        if torch.equal(new, labels):
            return labels
        labels = new


def hop_neighbour_pairs(edge_index: torch.Tensor, num_nodes: int, hops: int) -> torch.Tensor:
    """All ordered pairs (i, j), i != j, at most ``hops`` mesh edges apart -> (2, P).

    Breadth-first over pairs: the pairs first reached in the last step are
    extended by every edge of their end node. ``hops = 1`` gives the edges.
    """
    order = torch.argsort(edge_index[0])
    senders, receivers = edge_index[0, order], edge_index[1, order]
    degree = torch.bincount(senders, minlength=num_nodes)
    first_edge = torch.cumsum(degree, 0) - degree      # start of each node's edges
    found = torch.unique(senders * num_nodes + receivers)
    frontier = found
    for _ in range(hops - 1):
        start, end = frontier // num_nodes, frontier % num_nodes
        count = degree[end]
        offset = (torch.arange(int(count.sum()), device=edge_index.device)
                  - (torch.cumsum(count, 0) - count).repeat_interleave(count))
        reached = receivers[first_edge[end].repeat_interleave(count) + offset]
        keys = torch.unique(start.repeat_interleave(count) * num_nodes + reached)
        frontier = keys[~torch.isin(keys, found)]
        found = torch.cat((found, frontier))
    pairs = torch.stack((found // num_nodes, found % num_nodes))
    return pairs[:, pairs[0] != pairs[1]]


def mesh_path_pairs(edge_index: torch.Tensor, mesh_pos: torch.Tensor,
                    max_length: float) -> torch.Tensor:
    """All ordered pairs (i, j), i != j, whose shortest path along the mesh edges
    (edge lengths from ``mesh_pos``) is shorter than ``max_length`` -> (2, P),
    sorted by (i, j).

    Shortest paths from every node at once, bounded by ``max_length`` (label
    correcting, float64): the pairs whose distance fell in the last round are
    extended by every edge of their end node, until no distance falls any more.
    """
    num_nodes = mesh_pos.shape[0]
    device = edge_index.device
    order = torch.argsort(edge_index[0])
    senders, receivers = edge_index[0, order], edge_index[1, order]
    pos = mesh_pos.to(device, torch.float64)
    lengths = torch.linalg.norm(pos[senders] - pos[receivers], dim=-1)
    degree = torch.bincount(senders, minlength=num_nodes)
    first_edge = torch.cumsum(degree, 0) - degree      # start of each node's edges
    nodes = torch.arange(num_nodes, device=device)
    keys = nodes * num_nodes + nodes                   # (i, i) at distance 0, sorted
    distance = torch.zeros(num_nodes, dtype=torch.float64, device=device)
    frontier, frontier_distance = keys, distance
    while frontier.numel():
        start, end = frontier // num_nodes, frontier % num_nodes
        count = degree[end]
        offset = (torch.arange(int(count.sum()), device=device)
                  - (torch.cumsum(count, 0) - count).repeat_interleave(count))
        edge = first_edge[end].repeat_interleave(count) + offset
        reached = start.repeat_interleave(count) * num_nodes + receivers[edge]
        length = frontier_distance.repeat_interleave(count) + lengths[edge]
        short = length < max_length
        reached, inverse = torch.unique(reached[short], return_inverse=True)
        length = torch.full(reached.shape, torch.inf, dtype=torch.float64,
                            device=device).scatter_reduce(0, inverse, length[short], "amin")
        at = torch.searchsorted(keys, reached).clamp(max=keys.numel() - 1)
        known = keys[at] == reached
        shorter = known & (length < distance[at])
        distance[at[shorter]] = length[shorter]
        frontier, frontier_distance = reached[shorter | ~known], length[shorter | ~known]
        keys, order = torch.sort(torch.cat((keys, reached[~known])))
        distance = torch.cat((distance, length[~known]))[order]
    pairs = torch.stack((keys // num_nodes, keys % num_nodes))
    return pairs[:, pairs[0] != pairs[1]]


def world_edge_selection(mode: str, mesh_edge_index: torch.Tensor, num_nodes: int,
                         exclude_hops: Optional[int] = None,
                         mesh_pos: Optional[torch.Tensor] = None,
                         exclude_length: Optional[float] = None) -> Dict[str, torch.Tensor]:
    """Static sample attributes that select the world edge variant ``mode``.

    ``all``: none, the mesh edges are excluded. ``contact``: ``body_id`` (N,),
    the connected part of the mesh of every node. ``self_contact_aware``:
    ``world_exclude_index`` (2, P), the pairs connected through the material,
    i.e. with a path along the mesh edges of the rest state ``mesh_pos`` shorter
    than ``exclude_length``; with ``exclude_hops`` (configs up to v0.3.0) the
    pairs at most that many mesh edges apart. The dataset stores them in every
    sample and ``rebuild_edges`` reads them back, so noise and rollout rebuild
    the same variant.
    """
    if mode == "all":
        return {}
    if mode == "contact":
        return {"body_id": mesh_components(mesh_edge_index, num_nodes)}
    if mode == "self_contact_aware":
        if exclude_hops is not None:
            if exclude_hops < 1:
                raise ValueError(f"exclude_hops must be >= 1, got {exclude_hops}")
            return {"world_exclude_index":
                    hop_neighbour_pairs(mesh_edge_index, num_nodes, exclude_hops)}
        if mesh_pos is None or exclude_length is None or exclude_length <= 0:
            raise ValueError("self_contact_aware needs the rest positions and a positive "
                             "exclude_length (or exclude_hops)")
        return {"world_exclude_index": mesh_path_pairs(mesh_edge_index, mesh_pos,
                                                       exclude_length)}
    raise ValueError(f"world_edges must be one of {WORLD_EDGE_MODES}, got {mode!r}")


# Nodes per graph up to which the dense distance matrix is faster than the cell list.
DENSE_RADIUS_SEARCH_LIMIT = 2500


def radius_pairs(pos: torch.Tensor, radius: float,
                 batch: Optional[torch.Tensor] = None) -> torch.Tensor:
    """All ordered pairs (i, j), i != j, of one graph with |pos_i - pos_j| < ``radius``.

    Graphs up to ``DENSE_RADIUS_SEARCH_LIMIT`` nodes use the dense distance
    matrix, larger ones a cell list (``_cell_list_pairs``); both compute the
    distances directly, so the result does not depend on the path. With
    ``batch`` (node -> graph index, the nodes of a graph in one block as in a
    PyG batch) only pairs inside a graph are formed. Returns (2, P), sorted by (i, j).
    """
    sizes = [pos.shape[0]] if batch is None else graph_sizes(batch)
    pairs = []
    start = 0
    for size in sizes:
        graph_pos = pos[start:start + size]
        if size <= DENSE_RADIUS_SEARCH_LIMIT:
            close = torch.cdist(graph_pos, graph_pos,
                                compute_mode="donot_use_mm_for_euclid_dist") < radius
            close.fill_diagonal_(False)
            senders, receivers = torch.nonzero(close, as_tuple=True)
        else:
            senders, receivers = _cell_list_pairs(graph_pos, radius)
        pairs.append(torch.stack((senders, receivers)) + start)
        start += size
    return (torch.cat(pairs, dim=1) if pairs
            else torch.empty(2, 0, dtype=torch.long, device=pos.device))


def graph_sizes(batch: torch.Tensor) -> List[int]:
    """Nodes per graph of a batch whose graphs are blocks of consecutive nodes (PyG);
    one read from the device instead of one per graph."""
    return torch.bincount(batch).tolist() if batch.numel() else []


def _cell_list_pairs(pos: torch.Tensor, radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pairs closer than ``radius`` of one graph via a cell list, sorted by (i, j).

    The points are binned into cubes of edge ``radius``; only the points of the
    27 cubes around a point can be closer than ``radius``, so time and memory
    grow with the number of candidate pairs instead of N².
    """
    device = pos.device
    num_nodes = pos.shape[0]
    cell = torch.floor((pos - pos.min(0).values) / radius).long() + 1   # >= 1: neighbours >= 0
    extent = cell.max(0).values + 2
    if float(extent.prod()) >= 2.0 ** 62:
        raise ValueError(f"radius {radius} is too small for the extent of the positions")

    def key(cells: torch.Tensor) -> torch.Tensor:
        return (cells[:, 0] * extent[1] + cells[:, 1]) * extent[2] + cells[:, 2]

    order = torch.argsort(key(cell))
    sorted_keys = key(cell)[order]
    point = torch.arange(num_nodes, device=device)
    senders, receivers = [], []
    steps = torch.tensor([-1, 0, 1], device=device)
    for offset in torch.cartesian_prod(steps, steps, steps):
        neighbour = key(cell + offset)
        start = torch.searchsorted(sorted_keys, neighbour)
        count = torch.searchsorted(sorted_keys, neighbour, right=True) - start
        i = point.repeat_interleave(count)
        rank = (torch.arange(int(count.sum()), device=device)
                - (torch.cumsum(count, 0) - count).repeat_interleave(count))
        j = order[start.repeat_interleave(count) + rank]
        close = (i != j) & (torch.linalg.norm(pos[i] - pos[j], dim=-1) < radius)
        senders.append(i[close])
        receivers.append(j[close])
    i, j = torch.cat(senders), torch.cat(receivers)
    pair_order = torch.argsort(i * num_nodes + j)
    return i[pair_order], j[pair_order]


def world_edges(world_pos: torch.Tensor, mesh_edge_index: torch.Tensor, radius: float,
                batch: Optional[torch.Tensor] = None, body_id: Optional[torch.Tensor] = None,
                world_exclude_index: Optional[torch.Tensor] = None) -> torch.Tensor:
    """World edges (2, E_W) of the current positions (paper Sec. 3.1).

    All ordered pairs i != j with |x_i - x_j| < ``radius`` that are not
    connected by a mesh edge, or — with ``world_exclude_index`` — not in that
    pair set. With ``body_id`` only pairs of different bodies remain. With
    ``batch`` (node -> graph index) pairs are only formed inside the same graph.
    """
    device = world_pos.device
    num_nodes = world_pos.shape[0]
    edge_index = radius_pairs(world_pos, radius, batch)

    if body_id is not None and edge_index.shape[1]:
        body_id = body_id.to(device)
        edge_index = edge_index[:, body_id[edge_index[0]] != body_id[edge_index[1]]]

    excluded = mesh_edge_index if world_exclude_index is None else world_exclude_index
    if edge_index.shape[1] and excluded.shape[1]:
        # Drop the excluded pairs: compare (sender, receiver) as one key.
        keys = edge_index[0] * num_nodes + edge_index[1]
        excluded = excluded.to(device)
        excluded_keys = excluded[0] * num_nodes + excluded[1]
        edge_index = edge_index[:, ~torch.isin(keys, excluded_keys)]
    return edge_index


def edge_features(mesh_edge_index: torch.Tensor, mesh_space_features: torch.Tensor,
                  world_pos: torch.Tensor, radius: float,
                  batch: Optional[torch.Tensor] = None, **selection: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw (unnormalized) edge features for the positions ``world_pos``.

    ``mesh_space_features`` (E_M, 4) is the static mesh-space part u_ij, |u_ij|
    of the mesh edges; ``selection`` the output of ``world_edge_selection``
    (empty: paper variant). Returns ``mesh_edge_attr`` (E_M, 8),
    ``world_edge_index`` (2, E_W) and ``world_edge_attr`` (E_W, 4).
    """
    mesh_edge_attr = mesh_edge_features(mesh_edge_index, mesh_space_features, world_pos)
    world_edge_index = world_edges(world_pos, mesh_edge_index, radius, batch, **selection)
    world_edge_attr = relative_features(world_pos, world_edge_index).float()
    return mesh_edge_attr, world_edge_index, world_edge_attr


def mesh_edge_features(mesh_edge_index: torch.Tensor, mesh_space_features: torch.Tensor,
                       world_pos: torch.Tensor) -> torch.Tensor:
    """Raw mesh edge features (E_M, 8): the static mesh-space part plus x_ij, |x_ij|."""
    return torch.cat((mesh_space_features.to(world_pos.dtype),
                      relative_features(world_pos, mesh_edge_index)), dim=-1).float()


def add_world_edges(data, normalizer, radius: float) -> None:
    """World edges of the current positions ``data.world_pos`` (in place).

    For samples built without them (``DeformingPlateDataset(build_world_edges=False)``):
    the contact search then runs on the device of ``data`` (e.g. the GPU) instead of
    the DataLoader. Mesh edge features and wall columns stay as built; the result
    equals that of the dataset. Works on single and batched graphs.
    """
    world_mean, world_std = normalizer.mean_std("world_edge_features", data.world_pos.device)
    selection = {key: getattr(data, key) for key in WORLD_SELECTION_KEYS
                 if getattr(data, key, None) is not None}
    index = world_edges(data.world_pos, data.edge_index, radius,
                        getattr(data, "batch", None), **selection)
    data.world_edge_index = index
    data.world_edge_attr = (relative_features(data.world_pos, index).float()
                            - world_mean) / world_std


# ─────────────────────────────────────────────────────────────────────────────
# Node features and targets
# ─────────────────────────────────────────────────────────────────────────────

def trajectory_features(world_pos: torch.Tensor, stress: torch.Tensor,
                        actuator_mask: torch.Tensor, history: int = 0,
                        prev_stress: bool = True
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw per-sample features of consecutive frames.

    ``world_pos`` (T, N, 3) and ``stress`` (T, N, 1) are T consecutive selected
    frames. Every step t = history .. T-2 becomes one sample:

        bc_velocity[t]    = world_pos[t+1] - world_pos[t] on actuator nodes, else 0
        velocity_prev[t]  = world_pos[t] - world_pos[t-1]        (history = 1)
        stress_prev[t]    = stress[t]                            (history = 1, prev_stress)
        target[t]         = world_pos[t+1] - world_pos[t], stress[t+1]
        current_pos[t]    = world_pos[t]

    Returns the *unnormalized* ``dynamic_inputs`` (S, N, 3 | 6 | 7) — the ``x``
    columns after the one-hot node type — ``targets`` (S, N, 4) and
    ``current_pos`` (S, N, 3), with S = T - 1 - history. The exporter passes a
    whole trajectory, the dataset only the 2 + history frames of one sample.
    """
    velocity = world_pos[1:] - world_pos[:-1]          # velocity[k] = x_{k+1} - x_k
    target_velocity = velocity[history:]
    bc_velocity = torch.zeros_like(target_velocity)
    bc_velocity[:, actuator_mask] = target_velocity[:, actuator_mask]

    dynamic_inputs = [bc_velocity]
    if history:
        dynamic_inputs.append(velocity[:-1])           # x_t - x_{t-1}
        if prev_stress:
            dynamic_inputs.append(stress[1:-1])        # σ_t
    targets = torch.cat((target_velocity, stress[1 + history:]), dim=-1)
    return torch.cat(dynamic_inputs, dim=-1), targets, world_pos[history:-1]


# ─────────────────────────────────────────────────────────────────────────────
# Moving the current positions of a (batched) sample
# ─────────────────────────────────────────────────────────────────────────────

def rebuild_edges(data, world_pos: torch.Tensor, normalizer, radius: float) -> None:
    """Recompute ``data``'s edge features for new positions ``world_pos`` (in place).

    The mesh edges and their mesh-space part u_ij stay; the world-space part of
    the mesh edges and the whole world edge set are rebuilt, in the variant
    stored in ``data`` (``WORLD_SELECTION_KEYS``). With rigid walls
    (``data.wall_planes``) the wall columns of ``x`` follow as well. Works on
    single and batched graphs (uses ``data.batch`` if present).
    """
    mesh_mean, mesh_std = normalizer.mean_std("mesh_edge_features", world_pos.device)
    world_mean, world_std = normalizer.mean_std("world_edge_features", world_pos.device)
    # Recover the raw mesh-space columns from the normalized edge_attr.
    mesh_space_features = data.edge_attr[:, :4] * mesh_std[:4] + mesh_mean[:4]
    selection = {key: getattr(data, key) for key in WORLD_SELECTION_KEYS
                 if getattr(data, key, None) is not None}
    mesh_edge_attr, world_edge_index, world_edge_attr = edge_features(
        data.edge_index, mesh_space_features, world_pos, radius,
        getattr(data, "batch", None), **selection)
    data.edge_attr = (mesh_edge_attr - mesh_mean) / mesh_std
    data.world_edge_index = world_edge_index
    data.world_edge_attr = (world_edge_attr - world_mean) / world_std
    wall_planes = getattr(data, "wall_planes", None)
    if wall_planes is not None:
        node_mean, node_std = normalizer.mean_std("node_features", world_pos.device)
        features = wall_features(world_pos, wall_planes, WALL_DISTANCE_CUTOFF * radius,
                                 getattr(data, "batch", None))
        x = data.x.clone()
        x[:, -WALL_FEATURE_DIM:] = ((features - node_mean[-WALL_FEATURE_DIM:])
                                    / node_std[-WALL_FEATURE_DIM:])
        data.x = x


def displace_nodes(data, delta: torch.Tensor, normalizer, radius: float,
                   velocity_prev_columns: Optional[List[int]] = None) -> None:
    """Move the current positions by ``delta`` (N, 3, raw units) consistently.

    With x_t' = x_t + delta the next state x_{t+1} stays unchanged (paper
    App. A.2.2), so the velocity target becomes x_{t+1} - x_t' = target - delta,
    and all edges are rebuilt from x_t'. If ``velocity_prev_columns`` is given,
    ``velocity_prev`` = x_t' - x_{t-1} grows by ``delta`` as well (training
    noise); scheduled sampling passes None because it has already written the
    new ``velocity_prev``. ``delta`` must be 0 on kinematic nodes.
    """
    device = data.x.device
    if velocity_prev_columns:
        _, node_std = normalizer.mean_std("node_features", device)
        x = data.x.clone()
        x[:, velocity_prev_columns] += delta / node_std[velocity_prev_columns]
        data.x = x
    _, target_std = normalizer.mean_std("target", device)
    y = data.y.clone()
    y[:, VELOCITY_TARGET_COLUMNS] -= delta / target_std[VELOCITY_TARGET_COLUMNS]
    data.y = y
    data.world_pos = data.world_pos + delta
    rebuild_edges(data, data.world_pos, normalizer, radius)
