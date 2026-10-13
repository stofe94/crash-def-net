#!/usr/bin/env python3
"""
meshgraphnet.py — MeshGraphNet (Pfaff et al., ICLR 2021) with mesh + world edges.

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

Input (PyG ``Data``)
    x                  (N, node_in_dim)
    edge_index         (2, E_M)   mesh edges,  edge_attr       (E_M, mesh_edge_in_dim)
    world_edge_index   (2, E_W)   world edges, world_edge_attr (E_W, world_edge_in_dim)
Edges point sender = ``index[0]`` -> receiver = ``index[1]``; messages are
aggregated at the receiver.

The attribute names (``node_encoder``, ``processor``, ...) are the keys of the
saved ``state_dict``; renaming them breaks existing checkpoints.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

AGGREGATIONS = ("sum", "mean")

# (edge latents, edge_index) of one edge set
EdgeSet = Tuple[torch.Tensor, torch.Tensor]


def make_mlp(in_dim: int, out_dim: int, latent_size: int, num_layers: int,
             layer_norm: bool = True, activation=nn.ReLU) -> nn.Sequential:
    """``num_layers`` hidden layers of width ``latent_size`` -> ``out_dim``.

    The output layer has no activation; ``layer_norm`` appends a LayerNorm.
    ``activation`` is ReLU for MeshGraphNet and LeakyReLU for MGN-T; both are
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


class GraphNetBlock(nn.Module):
    """One message passing step over several edge sets (DeepMind ``GraphNetBlock``)."""

    def __init__(self, latent_size: int, num_layers: int, num_edge_sets: int,
                 aggregation: str = "sum", activation=nn.ReLU):
        super().__init__()
        self.aggregation = aggregation
        # One edge MLP per edge set: input [sender, receiver, edge] latents.
        self.edge_fns = nn.ModuleList([
            make_mlp(3 * latent_size, latent_size, latent_size, num_layers,
                     activation=activation)
            for _ in range(num_edge_sets)
        ])
        # Node MLP: input [node, aggregated messages of every edge set].
        self.node_fn = make_mlp((1 + num_edge_sets) * latent_size, latent_size,
                                latent_size, num_layers, activation=activation)

    def forward(self, nodes: torch.Tensor,
                edge_sets: List[EdgeSet]) -> Tuple[torch.Tensor, List[EdgeSet]]:
        """Update edges, then nodes; return both with residual connections added."""
        edge_updates = []
        for edge_fn, (edge_latents, index) in zip(self.edge_fns, edge_sets):
            senders, receivers = index[0], index[1]
            edge_updates.append(edge_fn(torch.cat(
                (nodes[senders], nodes[receivers], edge_latents), dim=-1)))

        messages = [aggregate_at_receivers(update, index[1], nodes.shape[0], self.aggregation)
                    for update, (_, index) in zip(edge_updates, edge_sets)]
        node_update = self.node_fn(torch.cat([nodes, *messages], dim=-1))

        nodes = nodes + node_update
        edge_sets = [(edge_latents + update, index)
                     for update, (edge_latents, index) in zip(edge_updates, edge_sets)]
        return nodes, edge_sets


class MeshGraphNet(nn.Module):
    """Encode-Process-Decode MeshGraphNet with mesh and (optional) world edges.

    ``world_edge_in_dim = 0`` builds a mesh-only model with one edge set.
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
    ):
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        self.output_dim = output_dim
        self.world_edge_in_dim = world_edge_in_dim
        self.use_world_edges = world_edge_in_dim > 0
        num_edge_sets = 2 if self.use_world_edges else 1

        self.node_encoder = make_mlp(node_in_dim, latent_size, latent_size, num_mlp_layers)
        self.mesh_edge_encoder = make_mlp(mesh_edge_in_dim, latent_size, latent_size,
                                          num_mlp_layers)
        self.world_edge_encoder = (
            make_mlp(world_edge_in_dim, latent_size, latent_size, num_mlp_layers)
            if self.use_world_edges else None)
        self.processor = nn.ModuleList([
            GraphNetBlock(latent_size, num_mlp_layers, num_edge_sets, aggregation)
            for _ in range(num_message_passing_steps)
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

        for block in self.processor:
            nodes, edge_sets = block(nodes, edge_sets)
        return self.decoder(nodes)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
