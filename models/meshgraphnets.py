#!/usr/bin/env python3
"""
meshgraphnets.py — MeshGraphNets (Pfaff et al., ICLR 2021) with mesh + world edges.

PyTorch implementation of the Encode-Process-Decode model described in the
paper for Lagrangian systems with two edge sets, as used for DEFORMING PLATE
(Sec. 3, Eq. 1, App. A.2.1); the processor blocks are GraphNet blocks in the
sense of Battaglia et al. (2018), generalized to several edge sets:

Encoder
    ε^V, ε^M, ε^W: MLPs for nodes, mesh edges and world edges, each followed by
    LayerNorm.
Processor
    L blocks with separate parameters. Per block
        e'^M_ij = f^M(e^M_ij, v_i, v_j)
        e'^W_ij = f^W(e^W_ij, v_i, v_j)
        v'_i    = f^V(v_i, Σ_j e'^M_ij, Σ_j e'^W_ij)
    every f is an MLP + LayerNorm; the aggregation uses the *updated* edge
    latents, and residual connections are added afterwards:
        v <- v + v',  e <- e + e'.
Decoder
    δ^V: MLP without LayerNorm on the final node latents only.

All MLPs are ReLU-activated with ``num_mlp_layers`` hidden layers of width
``latent_size`` (paper: 2 x 128, L = 15).

Global block (only with run parameters, ``global_in_dim > 0``; not in the paper)
    The full GraphNet block of Battaglia et al. (2018, Sec. 3.2): the run
    parameters u of each graph get their own encoder ε^U, and every processor
    block reads and updates them:
        e'_ij = f^E(e_ij, v_i, v_j, u)          for each edge set
        v'_i  = f^V(v_i, Σ_j e'^M_ij, Σ_j e'^W_ij, u)
        u'    = f^U(u, mean_i v'_i, mean e'^M, mean e'^W)
    with the residual u <- u + u'; the last block only reads u, as nothing
    reads its update (the decoder sees the nodes). The means run over the whole graph: a
    non-physical path that carries information across the structure within one
    block, regardless of the distance through the material. Without run
    parameters there is no global block and the model is that of the paper.

Input (PyG ``Data``)
    x                  (N, node_in_dim)
    edge_index         (2, E_M)   mesh edges,  edge_attr       (E_M, mesh_edge_in_dim)
    world_edge_index   (2, E_W)   world edges, world_edge_attr (E_W, world_edge_in_dim)
    u                  (B, global_in_dim)  run parameters, one row per graph (global block)
    batch              (N,)       graph of every node; missing for a single graph
Edges point sender = ``index[0]`` -> receiver = ``index[1]``; messages are
aggregated at the receiver.

The attribute names (``node_encoder``, ``processor``, ...) are the keys of the
saved ``state_dict``; renaming them breaks existing checkpoints.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

AGGREGATIONS = ("sum", "mean")

# (edge latents, edge_index) of one edge set
EdgeSet = Tuple[torch.Tensor, torch.Tensor]


def make_mlp(in_dim: int, out_dim: int, latent_size: int, num_layers: int,
             layer_norm: bool = True, activation=nn.ReLU) -> nn.Sequential:
    """``num_layers`` hidden layers of width ``latent_size`` -> ``out_dim``.

    The output layer has no activation; ``layer_norm`` appends a LayerNorm.
    ``activation`` is ReLU for MeshGraphNets and LeakyReLU for MGN-T; both are
    parameter-free, so the ``state_dict`` layout does not depend on it.
    """
    layers: List[nn.Module] = []
    width = in_dim
    for _ in range(num_layers):
        layers += [nn.Linear(width, latent_size), activation()]
        width = latent_size
    layers.append(nn.Linear(width, out_dim))
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


def aggregate_at_receivers(edge_latents: torch.Tensor, receivers: torch.Tensor,
                           num_nodes: int, aggregation: str) -> torch.Tensor:
    """Sum (or mean) of the edge latents arriving at each node -> (N, D).

    Nodes without incoming edges get zeros.
    """
    out = edge_latents.new_zeros(num_nodes, edge_latents.shape[-1])
    out.index_add_(0, receivers, edge_latents)
    if aggregation == "mean":
        count = edge_latents.new_zeros(num_nodes)
        count.index_add_(0, receivers, edge_latents.new_ones(receivers.shape[0]))
        out = out / count.clamp(min=1).unsqueeze(-1)
    return out


def graph_index(data, num_nodes: int) -> torch.Tensor:
    """Node -> graph index; a single (unbatched) graph carries no ``batch``."""
    batch = getattr(data, "batch", None)
    if batch is None:
        return torch.zeros(num_nodes, dtype=torch.long, device=data.x.device)
    return batch


class GraphNetBlock(nn.Module):
    """One message passing step over several edge sets (DeepMind ``GraphNetBlock``);
    with ``use_globals`` the full block with global latents (Battaglia et al. 2018)."""

    def __init__(self, latent_size: int, num_layers: int, num_edge_sets: int,
                 aggregation: str = "sum", activation=nn.ReLU, use_globals: bool = False,
                 update_globals: bool = True):
        super().__init__()
        self.aggregation = aggregation
        self.use_globals = use_globals
        g = int(use_globals)
        # One edge MLP per edge set: input [sender, receiver, edge (, global)] latents.
        self.edge_fns = nn.ModuleList([
            make_mlp((3 + g) * latent_size, latent_size, latent_size, num_layers,
                     activation=activation)
            for _ in range(num_edge_sets)
        ])
        # Node MLP: input [node, aggregated messages of every edge set (, global)].
        self.node_fn = make_mlp((1 + num_edge_sets + g) * latent_size, latent_size,
                                latent_size, num_layers, activation=activation)
        # Global MLP: input [global, mean of the node updates and of each edge set's].
        # The last block of a model only reads the globals (``update_globals=False``):
        # an update there would feed nothing.
        self.global_fn = (make_mlp((2 + num_edge_sets) * latent_size, latent_size,
                                   latent_size, num_layers, activation=activation)
                          if use_globals and update_globals else None)

    def forward(self, nodes: torch.Tensor, edge_sets: List[EdgeSet],
                globals_: Optional[torch.Tensor] = None, graph: Optional[torch.Tensor] = None):
        """Update edges, then nodes; return both with residual connections added.

        With ``use_globals`` also pass the global latents ``globals_`` (B, D) of the B
        graphs and the graph of every node ``graph`` (N,); then the updated global
        latents are returned as a third value.
        """
        if (globals_ is not None) != self.use_globals:
            raise ValueError("global latents are needed exactly when the block has use_globals")
        edge_updates = []
        for edge_fn, (edge_latents, index) in zip(self.edge_fns, edge_sets):
            senders, receivers = index[0], index[1]
            inputs = [nodes[senders], nodes[receivers], edge_latents]
            if globals_ is not None:
                inputs.append(globals_[graph[receivers]])
            edge_updates.append(edge_fn(torch.cat(inputs, dim=-1)))

        messages = [aggregate_at_receivers(update, index[1], nodes.shape[0], self.aggregation)
                    for update, (_, index) in zip(edge_updates, edge_sets)]
        node_inputs = [nodes, *messages] + ([globals_[graph]] if globals_ is not None else [])
        node_update = self.node_fn(torch.cat(node_inputs, dim=-1))

        if self.global_fn is not None:
            # Means over the whole graph: the non-physical path of the global block.
            num_graphs = globals_.shape[0]
            pooled = [aggregate_at_receivers(node_update, graph, num_graphs, "mean")]
            pooled += [aggregate_at_receivers(update, graph[index[1]], num_graphs, "mean")
                       for update, (_, index) in zip(edge_updates, edge_sets)]
            globals_ = globals_ + self.global_fn(torch.cat([globals_, *pooled], dim=-1))

        nodes = nodes + node_update
        edge_sets = [(edge_latents + update, index)
                     for update, (edge_latents, index) in zip(edge_updates, edge_sets)]
        if globals_ is None:
            return nodes, edge_sets
        return nodes, edge_sets, globals_


def run_blocks(blocks, nodes: torch.Tensor, edge_sets: List[EdgeSet],
               globals_: Optional[torch.Tensor] = None, graph: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, List[EdgeSet], Optional[torch.Tensor]]:
    """Apply the ``GraphNetBlock``s in turn, with global latents if given."""
    for block in blocks:
        if globals_ is None:
            nodes, edge_sets = block(nodes, edge_sets)
        else:
            nodes, edge_sets, globals_ = block(nodes, edge_sets, globals_, graph)
    return nodes, edge_sets, globals_


class MeshGraphNets(nn.Module):
    """Encode-Process-Decode MeshGraphNets with mesh and (optional) world edges.

    ``world_edge_in_dim = 0`` builds a mesh-only model with one edge set,
    ``global_in_dim > 0`` one with the global block (run parameters ``data.u``).
    ``forward`` returns the normalized node predictions (N, ``output_dim``).
    """

    def __init__(
        self,
        node_in_dim: int,
        mesh_edge_in_dim: int,
        world_edge_in_dim: int,
        output_dim: int,
        latent_size: int = 128,
        num_mlp_layers: int = 2,
        num_message_passing_steps: int = 15,
        aggregation: str = "sum",
        global_in_dim: int = 0,
    ):
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        self.output_dim = output_dim
        self.world_edge_in_dim = world_edge_in_dim
        self.use_world_edges = world_edge_in_dim > 0
        self.use_globals = global_in_dim > 0
        num_edge_sets = 2 if self.use_world_edges else 1

        self.node_encoder = make_mlp(node_in_dim, latent_size, latent_size, num_mlp_layers)
        self.mesh_edge_encoder = make_mlp(mesh_edge_in_dim, latent_size, latent_size,
                                          num_mlp_layers)
        self.world_edge_encoder = (
            make_mlp(world_edge_in_dim, latent_size, latent_size, num_mlp_layers)
            if self.use_world_edges else None)
        self.global_encoder = (
            make_mlp(global_in_dim, latent_size, latent_size, num_mlp_layers)
            if self.use_globals else None)
        self.processor = nn.ModuleList([
            GraphNetBlock(latent_size, num_mlp_layers, num_edge_sets, aggregation,
                          use_globals=self.use_globals,
                          update_globals=step < num_message_passing_steps - 1)
            for step in range(num_message_passing_steps)
        ])
        self.decoder = make_mlp(latent_size, output_dim, latent_size, num_mlp_layers,
                                layer_norm=False)

    def forward(self, data) -> torch.Tensor:
        nodes = self.node_encoder(data.x)
        edge_sets = [(self.mesh_edge_encoder(data.edge_attr), data.edge_index)]
        if self.use_world_edges:
            world_index = getattr(data, "world_edge_index", None)
            world_attr = getattr(data, "world_edge_attr", None)
            if world_index is None:   # sample without world edges: empty edge set
                world_index = data.edge_index.new_zeros(2, 0)
                world_attr = data.edge_attr.new_zeros(0, self.world_edge_in_dim)
            edge_sets.append((self.world_edge_encoder(world_attr), world_index))
        globals_, graph = None, None
        if self.use_globals:
            globals_, graph = self.global_encoder(data.u), graph_index(data, nodes.shape[0])

        nodes, _, _ = run_blocks(self.processor, nodes, edge_sets, globals_, graph)
        return self.decoder(nodes)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
