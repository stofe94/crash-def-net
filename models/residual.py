"""
residual.py — residual ("delta") target wrapper for datasets with ``*_prev`` inputs.

The wrapped network predicts the *normalized change* of every target that has
a matching ``<name>_prev`` input column (needs an export with ``--history 1``):

    delta_norm      = net(data)                                  (N, C)
    prediction_raw  = prev_raw + delta_norm * delta_std + delta_mean
    output          = (prediction_raw - target_mean) / target_std

The wrapper therefore returns predictions in the usual normalized *target*
space: loss, metrics, rollout, scheduled sampling and noise work unchanged,
and a freshly initialised network (delta ~ 0) starts at the "copy previous
value" baseline instead of having to learn it. Targets without a ``_prev``
column are passed through as absolute normalized predictions.

All conversion statistics are registered buffers: they are set once from the
normalizer before training (``set_stats``) and travel with the checkpoint.
The buffer names are ``state_dict`` keys; renaming them breaks checkpoints.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ResidualTargetModel(nn.Module):
    """Turn a delta-predicting network into an absolute-target model."""

    def __init__(self, model: nn.Module, node_feature_names: List[str],
                 target_feature_names: List[str]):
        super().__init__()
        self.model = model
        self.output_dim = model.output_dim
        self.target_feature_names = list(target_feature_names)

        prev_idx, has_prev = [], []
        for name in self.target_feature_names:
            column = f"{name}_prev"
            has_prev.append(column in node_feature_names)
            # Targets without a _prev column read column 0; torch.where ignores it.
            prev_idx.append(node_feature_names.index(column) if has_prev[-1] else 0)
        if not any(has_prev):
            raise ValueError("residual_target: no target has a matching '<name>_prev' "
                             "input column — export the dataset with --history 1")

        num_targets = len(self.target_feature_names)
        self.register_buffer("prev_idx", torch.tensor(prev_idx, dtype=torch.long))
        self.register_buffer("has_prev", torch.tensor(has_prev, dtype=torch.bool))
        for name in ("prev_mean", "target_mean", "delta_mean"):
            self.register_buffer(name, torch.zeros(num_targets))
        for name in ("prev_std", "target_std", "delta_std"):
            self.register_buffer(name, torch.ones(num_targets))

    def set_stats(self, normalizer) -> None:
        """Copy the conversion statistics from a fitted ``Normalizer``."""
        if "target_delta" not in normalizer:
            raise KeyError("normalizer has no 'target_delta' statistics — re-export the "
                           "dataset with --history 1 to use residual_target = true")
        device = self.prev_idx.device
        node_mean, node_std = normalizer.mean_std("node_features", device)
        self.prev_mean.copy_(node_mean[self.prev_idx])
        self.prev_std.copy_(node_std[self.prev_idx])
        for prefix, group in (("target", "target"), ("delta", "target_delta")):
            mean, std = normalizer.mean_std(group, device)
            getattr(self, f"{prefix}_mean").copy_(mean)
            getattr(self, f"{prefix}_std").copy_(std)

    def forward(self, data) -> torch.Tensor:
        delta_norm = self.model(data)
        prev_raw = data.x[:, self.prev_idx] * self.prev_std + self.prev_mean
        prediction_raw = prev_raw + delta_norm * self.delta_std + self.delta_mean
        absolute = (prediction_raw - self.target_mean) / self.target_std
        return torch.where(self.has_prev, absolute, delta_norm)
