"""
normalizer.py — Per-group z-score statistics with JSON persistence.

Every feature group (``node_features``, ``target``, ``mesh_edge_features``,
``world_edge_features`` and, with history, ``target_delta``) stores one mean
and one standard deviation per column. The exporter fits them once on the
train split and writes ``normalizer.json`` next to the dataset; training reads
that file and stores a copy in every checkpoint, which rollout and resumed
training use — so a model keeps its scale even if the dataset is exported anew.

File format (unchanged since the first export)::

    {"method": "standardize",
     "stats": {"<group>": {"mean": [...], "std": [...]}, ...}}
"""

from __future__ import annotations

import json
from typing import Dict, Tuple, Union

import numpy as np
import torch

Array = Union[np.ndarray, torch.Tensor]

_METHOD = "standardize"   # the only method ever written to normalizer.json


class Normalizer:
    """Mean/std per feature group; ``normalize`` = (x - mean) / std.

    The standard deviations are guarded against zero when they are fitted
    (see ``dataset_preprocessor.export``), so no epsilon is added here and
    ``denormalize(normalize(x)) == x`` holds exactly up to float rounding.
    """

    def __init__(self) -> None:
        self.stats: Dict[str, Dict[str, np.ndarray]] = {}

    def __contains__(self, group: str) -> bool:
        return group in self.stats

    def set_stats(self, group: str, mean: np.ndarray, std: np.ndarray) -> None:
        """Store the statistics of ``group`` (cast to float32)."""
        self.stats[group] = {
            "mean": np.asarray(mean, dtype=np.float32),
            "std": np.asarray(std, dtype=np.float32),
        }

    def mean_std(self, group: str, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean and std of ``group`` as float32 tensors on ``device``."""
        stats = self._group(group)
        return (torch.as_tensor(stats["mean"], dtype=torch.float32, device=device),
                torch.as_tensor(stats["std"], dtype=torch.float32, device=device))

    def normalize(self, values: Array, group: str) -> Array:
        """Raw -> normalized; works on numpy arrays and torch tensors."""
        mean, std = self._stats_like(values, group)
        return (values - mean) / std

    def denormalize(self, values: Array, group: str) -> Array:
        """Normalized -> raw; works on numpy arrays and torch tensors."""
        mean, std = self._stats_like(values, group)
        return values * std + mean

    def to_dict(self) -> Dict:
        """All groups in the file format (plain lists; stored in checkpoints too)."""
        state = {group: {key: value.tolist() for key, value in stats.items()}
                 for group, stats in self.stats.items()}
        return {"method": _METHOD, "stats": state}

    @classmethod
    def from_dict(cls, data: Dict, source: str = "normalizer") -> "Normalizer":
        """Inverse of ``to_dict``."""
        if data.get("method") != _METHOD:
            raise ValueError(f"{source}: unsupported normalization method "
                             f"{data.get('method')!r} (expected {_METHOD!r})")
        normalizer = cls()
        for group, stats in data["stats"].items():
            normalizer.set_stats(group, np.array(stats["mean"]), np.array(stats["std"]))
        return normalizer

    def save(self, path: str) -> None:
        """Write all groups to ``path`` (JSON)."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        """Read a file written by ``save``."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f), path)

    def __eq__(self, other) -> bool:
        """Same groups with exactly the same statistics."""
        if not isinstance(other, Normalizer) or self.stats.keys() != other.stats.keys():
            return False
        return all(np.array_equal(stats[key], other.stats[group][key])
                   for group, stats in self.stats.items() for key in ("mean", "std"))

    # ── internals ───────────────────────────────────────────────────────────

    def _group(self, group: str) -> Dict[str, np.ndarray]:
        if group not in self.stats:
            raise KeyError(f"normalizer has no statistics for {group!r}")
        return self.stats[group]

    def _stats_like(self, values: Array, group: str):
        """Mean/std in the container type (and device) of ``values``."""
        self._check_width(values, group)
        if isinstance(values, torch.Tensor):
            return self.mean_std(group, values.device)
        stats = self._group(group)
        return stats["mean"], stats["std"]

    def _check_width(self, values: Array, group: str) -> None:
        expected = self._group(group)["mean"].shape[0]
        if values.shape[-1] != expected:
            raise ValueError(f"normalizer {group!r}: expected {expected} columns, "
                             f"got {values.shape[-1]}")
