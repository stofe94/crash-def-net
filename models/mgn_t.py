#!/usr/bin/env python3
"""
mgn_t.py — MeshGraphNet-Transformer (MGN-T), Iparraguirre et al., arXiv:2601.23177v4.

Independent implementation based on the paper; not affiliated with or endorsed
by the authors.

Instead of a deep stack of message passing steps (MeshGraphNets: 15) MGN-T uses a
Transformer as the global processor (paper Sec. 3.2):

    Encoder (3 MLPs)
      -> pre-processing MPNN   2 message passing steps (local, uses edges)
      -> Transformer           2 blocks with physics attention (global, over the
                               simulated structure only)
      -> refinement MPNN       2 message passing steps (local, uses edges)
    -> Decoder

The transformer and its positional encoding see only the nodes of the simulated
structure (paper Sec. 3.1-3.3, 5.1): obstacles such as the actuator act through the
contact edges of the pre-processor and take no tokens. ``transformer_nodes = "all"``
restores the behaviour up to v0.2.0, where every node took part.

Physics attention (Transolver, arXiv:2402.02366, with the eidetic states of
Transolver++, arXiv:2502.02414) maps the N nodes onto P << N physical tokens,
attends over the tokens (O(P²) instead of O(N²)) and scatters the result back:

    tau  = tau_0 + Linear(x)                            learned temperature
    w    = Softmax((Linear(x) - log(-log(eps))) / tau)   slice weights (N, P)
    z_j  = sum_i w_ij x_i / sum_i w_ij                   slicing
    z'   = Softmax(Q K^T / sqrt(c)) V                    attention over tokens
    x'_i = sum_j w_ij z'_j                               de-slicing

Paper settings for deforming_plate (Sec. 5.1, Table 2): 2 + 2 message passing
steps, 2 transformer blocks, 4 heads, P = 16 tokens, ~0.5 M parameters; tau_0 = 1.0
(Sec. 3.3). The defaults here (P = 128, widths 64-32-64) come from Table 2 of v3,
which belongs to its Pi-beam example. Every MLP has 2 hidden layers, LeakyReLU and
LayerNorm (Sec. 3.2).

Assumptions where the paper leaves a gap (chosen here, flagged in the code):

1. De-slicing ("projects P back onto N" in the paper): with the same slice
   weights as slicing, as in Transolver.
2. Gumbel noise -log(-log(eps)) with eps ~ U(0, 1), the standard form of the
   Gumbel trick, and only while training (the paper states eps ~ N(0, 1)).
3. tau kept positive with softplus, otherwise the temperature could reach zero.
   The paper writes tau = tau_0 + Linear(x) with tau_0 = 1.0; with softplus the
   initial tau is ``tau_init`` + ln 2 (about 1.19 for the default 0.5).
4. Positional encoding: 3 axes x sin/cos x ``pe_frequencies`` frequencies
   (2^0 .. 2^(F-1)), normalized per graph to [0, 1]. The paper only cites
   Vaswani et al. and gives no dimension.
5. Widths "64-32-64" (v3 only) read as MPNN 64, Transformer 32, refinement 64.
6. Residual connection around the transformer stage, pre-norm blocks (standard,
   not described in the paper).
7. Decoder without LayerNorm (as in MeshGraphNets); normalizing the output would
   destroy the magnitude of the prediction.
8. A batch may mix samples of different trajectories — slicing and attention run
   per graph. The paper batches the snapshots of one trajectory.

Input as in ``models/meshgraphnets.py``, plus ``data.mesh_pos`` (undeformed node
positions) for the positional encoding.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_preprocessor.graph_features import structure_node_mask
from models.meshgraphnets import GraphNetBlock, count_parameters, make_mlp  # noqa: F401

_ACTIVATION = nn.LeakyReLU   # paper: LeakyReLU in every MLP
# Nodes of the global stage: the simulated structure (paper) or every node (up to v0.2.0).
TRANSFORMER_NODES = ("structure", "all")


def _graph_batch(data, num_nodes: int) -> torch.Tensor:
    """Node -> graph index; a single (unbatched) graph carries no ``batch``."""
    batch = getattr(data, "batch", None)
    if batch is None:
        return torch.zeros(num_nodes, dtype=torch.long, device=data.x.device)
    return batch


class SinusoidalPositionEncoding(nn.Module):
    """Stationary waves over the undeformed geometry (paper Sec. 3.1).

    One sine and one cosine per axis and frequency, i.e. ``6 * num_frequencies``
    features. Coordinates are normalized per graph to [0, 1], so the encoding
    does not depend on absolute positions or on the size of the component.
    """

    def __init__(self, num_frequencies: int = 8):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_frequencies).float())

    @property
    def dim(self) -> int:
        return 6 * self.num_frequencies

    def forward(self, mesh_pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        normalized = torch.empty_like(mesh_pos)
        for graph in range(int(batch.max().item()) + 1):
            nodes = batch == graph
            pos = mesh_pos[nodes]
            low = pos.min(dim=0).values
            span = (pos.max(dim=0).values - low).clamp(min=1e-9)
            normalized[nodes] = (pos - low) / span

        angles = 2 * torch.pi * normalized.unsqueeze(-1) * self.frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(1)


class PhysicsAttention(nn.Module):
    """Slice onto P tokens, attend over the tokens, scatter back (paper Sec. 3.3)."""

    def __init__(self, dim: int, num_heads: int, num_tokens: int, tau_init: float = 0.5):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.tau_init = tau_init
        self.slice_projection = nn.Linear(dim, num_tokens)
        self.temperature_projection = nn.Linear(dim, 1)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output_projection = nn.Linear(dim, dim)

    def slice_weights(self, x: torch.Tensor) -> torch.Tensor:
        """(N, P) assignment of nodes to tokens; each row sums to 1."""
        temperature = self.tau_init + F.softplus(self.temperature_projection(x))
        logits = self.slice_projection(x)
        if self.training:
            uniform = torch.rand_like(logits).clamp(min=1e-9, max=1 - 1e-9)
            logits = logits - torch.log(-torch.log(uniform))   # Gumbel noise
        return F.softmax(logits / temperature, dim=-1)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        weights = self.slice_weights(x)
        num_graphs = int(batch.max().item()) + 1

        # Slicing per graph: z_j = sum_i w_ij x_i / sum_i w_ij
        tokens = x.new_empty(num_graphs, self.num_tokens, x.shape[-1])
        for graph in range(num_graphs):
            nodes = batch == graph
            graph_weights = weights[nodes]
            tokens[graph] = (graph_weights.t() @ x[nodes]) \
                / graph_weights.sum(dim=0).clamp(min=1e-9).unsqueeze(-1)

        # Attention over the P tokens of one graph
        qkv = self.qkv(tokens).view(num_graphs, self.num_tokens, 3, self.num_heads,
                                    self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(query, key, value)
        tokens = self.output_projection(
            attended.transpose(1, 2).reshape(num_graphs, self.num_tokens, -1))

        # De-slicing: x'_i = sum_j w_ij z'_j (the weights of a node sum to 1)
        out = x.new_empty(x.shape)
        for graph in range(num_graphs):
            nodes = batch == graph
            out[nodes] = weights[nodes] @ tokens[graph]
        return out


class TransformerBlock(nn.Module):
    """Pre-norm block: physics attention + feed forward, both with a residual."""

    def __init__(self, dim: int, num_heads: int, num_tokens: int, ffn_ratio: int = 4,
                 tau_init: float = 0.5):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = PhysicsAttention(dim, num_heads, num_tokens, tau_init)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_ratio * dim), _ACTIVATION(),
            nn.Linear(ffn_ratio * dim, dim))

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), batch)
        return x + self.ffn(self.ffn_norm(x))


class MgnTransformer(nn.Module):
    """Local message passing stages around a global transformer.

    ``forward`` returns the normalized node predictions (N, ``output_dim``),
    the same interface as ``MeshGraphNets``.
    """

    def __init__(
        self,
        node_in_dim: int,
        mesh_edge_in_dim: int,
        world_edge_in_dim: int,
        output_dim: int,
        latent_size: int = 64,          # paper: widths 64-32-64
        token_dim: int = 32,
        num_tokens: int = 128,
        num_heads: int = 4,
        num_transformer_blocks: int = 2,
        num_pre_mp_steps: int = 2,
        num_post_mp_steps: int = 2,
        num_mlp_layers: int = 2,
        pe_frequencies: int = 8,
        ffn_ratio: int = 4,
        tau_init: float = 0.5,
        aggregation: str = "sum",
        transformer_nodes: str = "structure",
    ):
        super().__init__()
        if transformer_nodes not in TRANSFORMER_NODES:
            raise ValueError(f"transformer_nodes must be one of {TRANSFORMER_NODES}, "
                             f"got {transformer_nodes!r}")
        self.transformer_nodes = transformer_nodes
        self.output_dim = output_dim
        self.world_edge_in_dim = world_edge_in_dim
        self.use_world_edges = world_edge_in_dim > 0
        num_edge_sets = 2 if self.use_world_edges else 1
        mlp = dict(num_layers=num_mlp_layers, activation=_ACTIVATION)

        self.node_encoder = make_mlp(node_in_dim, latent_size, latent_size, **mlp)
        self.mesh_edge_encoder = make_mlp(mesh_edge_in_dim, latent_size, latent_size, **mlp)
        self.world_edge_encoder = (
            make_mlp(world_edge_in_dim, latent_size, latent_size, **mlp)
            if self.use_world_edges else None)

        def mp_stack(num_steps):
            return nn.ModuleList([
                GraphNetBlock(latent_size, num_mlp_layers, num_edge_sets, aggregation,
                              activation=_ACTIVATION)
                for _ in range(num_steps)])

        self.pre_processor = mp_stack(num_pre_mp_steps)
        self.refinement_processor = mp_stack(num_post_mp_steps)

        # Node latents + positional encoding -> transformer width (paper Sec. 3.2)
        self.position_encoding = SinusoidalPositionEncoding(pe_frequencies)
        self.to_token_dim = nn.Linear(latent_size + self.position_encoding.dim, token_dim)
        self.transformer = nn.ModuleList([
            TransformerBlock(token_dim, num_heads, num_tokens, ffn_ratio, tau_init)
            for _ in range(num_transformer_blocks)])
        self.from_token_dim = nn.Linear(token_dim, latent_size)

        self.decoder = make_mlp(latent_size, output_dim, latent_size, layer_norm=False, **mlp)

    def forward(self, data) -> torch.Tensor:
        nodes = self.node_encoder(data.x)
        edge_sets = [(self.mesh_edge_encoder(data.edge_attr), data.edge_index)]
        if self.use_world_edges:
            world_index = getattr(data, "world_edge_index", None)
            world_attr = getattr(data, "world_edge_attr", None)
            if world_index is None:   # sample without contact
                world_index = data.edge_index.new_zeros(2, 0)
                world_attr = data.edge_attr.new_zeros(0, self.world_edge_in_dim)
            edge_sets.append((self.world_edge_encoder(world_attr), world_index))

        for block in self.pre_processor:
            nodes, edge_sets = block(nodes, edge_sets)

        batch = _graph_batch(data, nodes.shape[0])
        latents, mesh_pos = nodes, self._mesh_positions(data)
        if self.transformer_nodes == "structure":   # without the actuator (paper Sec. 3.3)
            index = structure_node_mask(data.x).nonzero().squeeze(1)
            latents, batch, mesh_pos = nodes[index], batch[index], mesh_pos[index]
        tokens = self.to_token_dim(
            torch.cat((latents, self.position_encoding(mesh_pos, batch)), dim=-1))
        for block in self.transformer:
            tokens = block(tokens, batch)
        update = self.from_token_dim(tokens)   # residual around the global stage
        nodes = nodes + update if self.transformer_nodes == "all" else \
            nodes.index_add(0, index, update)

        for block in self.refinement_processor:
            nodes, edge_sets = block(nodes, edge_sets)
        return self.decoder(nodes)

    @staticmethod
    def _mesh_positions(data) -> torch.Tensor:
        mesh_pos: Optional[torch.Tensor] = getattr(data, "mesh_pos", None)
        if mesh_pos is None:
            raise ValueError(
                "mgn_t needs data.mesh_pos (undeformed node positions) for the "
                "positional encoding — this dataset does not provide it")
        return mesh_pos
