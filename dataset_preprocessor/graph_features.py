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
``edge_attr`` (8), mesh edges from the tetrahedra
    u_ij, |u_ij| (mesh space) + x_ij, |x_ij| (world space).
``world_edge_attr`` (4), world edges
    x_ij, |x_ij| for all node pairs closer than the contact radius r_W that are
    not already mesh neighbours.
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
MESH_EDGE_FEATURE_NAMES = [
    "mesh_dx", "mesh_dy", "mesh_dz", "mesh_dist",
    "world_dx", "world_dy", "world_dz", "world_dist",
]
WORLD_EDGE_FEATURE_NAMES = ["world_dx", "world_dy", "world_dz", "world_dist"]

TARGET_DIM = len(TARGET_FEATURE_NAMES)                    # 4
MESH_EDGE_FEATURE_DIM = len(MESH_EDGE_FEATURE_NAMES)      # 8
WORLD_EDGE_FEATURE_DIM = len(WORLD_EDGE_FEATURE_NAMES)    # 4
VELOCITY_TARGET_COLUMNS = [0, 1, 2]                       # velocity_x/y/z in ``y``


# ─────────────────────────────────────────────────────────────────────────────
# Feature layout
# ─────────────────────────────────────────────────────────────────────────────

def node_feature_names(history: int = 0, prev_stress: bool = True) -> List[str]:
    """Column names of ``x`` for the given layout (see module docstring)."""
    if history not in (0, 1):
        raise ValueError(f"history must be 0 or 1, got {history!r}")
    names = NODE_TYPE_NAMES + BC_VELOCITY_NAMES
    if history:
        names += PREV_VELOCITY_NAMES + (PREV_STRESS_NAMES if prev_stress else [])
    return names


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


# ─────────────────────────────────────────────────────────────────────────────
# Edges
# ─────────────────────────────────────────────────────────────────────────────

def relative_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """pos[sender] - pos[receiver] and its length -> (E, 4)."""
    offset = pos[edge_index[0]] - pos[edge_index[1]]
    return torch.cat((offset, torch.linalg.norm(offset, dim=-1, keepdim=True)), dim=-1)


def mesh_edges_from_cells(cells) -> torch.Tensor:
    """Mesh edges (2, E) of tetrahedra (C, 4).

    Each of the six vertex pairs of every cell yields an edge in both
    directions; edges shared by neighbouring cells appear once. Sorted by
    (sender, receiver).
    """
    cells = torch.from_numpy(np.asarray(cells, dtype=np.int64))
    first, second = torch.triu_indices(4, 4, offset=1)   # the 6 vertex pairs
    a, b = cells[:, first].reshape(-1), cells[:, second].reshape(-1)
    both_directions = torch.stack((torch.cat((a, b)), torch.cat((b, a))))
    return torch.unique(both_directions, dim=1)


def world_edges(world_pos: torch.Tensor, mesh_edge_index: torch.Tensor, radius: float,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
    """World edges (2, E_W) of the current positions (paper Sec. 3.1).

    All ordered pairs i != j with |x_i - x_j| < ``radius`` that are not
    connected by a mesh edge. With ``batch`` (node -> graph index) pairs are
    only formed inside the same graph.
    """
    device = world_pos.device
    num_nodes = world_pos.shape[0]
    if batch is None:
        graphs = [torch.arange(num_nodes, device=device)]
    else:
        graphs = [torch.nonzero(batch == b, as_tuple=True)[0] for b in torch.unique(batch)]

    pairs = []
    for nodes in graphs:
        close = torch.cdist(world_pos[nodes], world_pos[nodes]) < radius
        close.fill_diagonal_(False)
        senders, receivers = torch.nonzero(close, as_tuple=True)
        pairs.append(torch.stack((nodes[senders], nodes[receivers])))
    edge_index = (torch.cat(pairs, dim=1) if pairs
                  else torch.empty(2, 0, dtype=torch.long, device=device))

    if edge_index.shape[1] and mesh_edge_index.shape[1]:
        # Drop pairs that are mesh edges: compare (sender, receiver) as one key.
        keys = edge_index[0] * num_nodes + edge_index[1]
        mesh_edge_index = mesh_edge_index.to(device)
        mesh_keys = mesh_edge_index[0] * num_nodes + mesh_edge_index[1]
        edge_index = edge_index[:, ~torch.isin(keys, mesh_keys)]
    return edge_index


def edge_features(mesh_edge_index: torch.Tensor, mesh_space_features: torch.Tensor,
                  world_pos: torch.Tensor, radius: float,
                  batch: Optional[torch.Tensor] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw (unnormalized) edge features for the positions ``world_pos``.

    ``mesh_space_features`` (E_M, 4) is the static mesh-space part u_ij, |u_ij|
    of the mesh edges. Returns ``mesh_edge_attr`` (E_M, 8),
    ``world_edge_index`` (2, E_W) and ``world_edge_attr`` (E_W, 4).
    """
    mesh_edge_attr = torch.cat(
        (mesh_space_features.to(world_pos.dtype),
         relative_features(world_pos, mesh_edge_index)), dim=-1).float()
    world_edge_index = world_edges(world_pos, mesh_edge_index, radius, batch)
    world_edge_attr = relative_features(world_pos, world_edge_index).float()
    return mesh_edge_attr, world_edge_index, world_edge_attr


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
    the mesh edges and the whole world edge set are rebuilt. Works on single
    and batched graphs (uses ``data.batch`` if present).
    """
    mesh_mean, mesh_std = normalizer.mean_std("mesh_edge_features", world_pos.device)
    world_mean, world_std = normalizer.mean_std("world_edge_features", world_pos.device)
    # Recover the raw mesh-space columns from the normalized edge_attr.
    mesh_space_features = data.edge_attr[:, :4] * mesh_std[:4] + mesh_mean[:4]
    mesh_edge_attr, world_edge_index, world_edge_attr = edge_features(
        data.edge_index, mesh_space_features, world_pos, radius,
        getattr(data, "batch", None))
    data.edge_attr = (mesh_edge_attr - mesh_mean) / mesh_std
    data.world_edge_index = world_edge_index
    data.world_edge_attr = (world_edge_attr - world_mean) / world_std


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
