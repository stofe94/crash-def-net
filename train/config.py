"""
config.py — Training configuration: TOML file -> flat ``TrainingConfig``.

The TOML sections (see ``config.example.toml``) map onto the
fields of one flat dataclass via ``TOML_KEYS``. Unknown keys are reported, so
typos and settings that no longer exist do not go unnoticed. The config is
stored in every checkpoint (``to_dict``) and restored with ``from_dict``,
which also understands the field names of older checkpoints.
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import toml
import torch

SCHEDULERS = ("exponential", "plateau", "cosine", "none")
LOSS_TYPES = ("mse", "mae", "huber", "smooth_l1")
ARCHITECTURES = ("meshgraphnet", "mgn_t")
# Rollout metric that drives model selection and early stopping.
SELECTION_METRICS = ("mse", "pos_rmse", "pos_rmse_final")


@dataclass
class TrainingConfig:
    """All training settings. Defaults follow the paper where it gives a value."""

    # [data]
    data_dir: str = "data/deforming_plate_s10_small"
    num_dataloader_workers: int = 0          # > 0: build graphs in spawned workers
    lazy_loading: bool = False               # read frames from the DB on access (v2 DB)
    cache_frames: int = 0                    # lazy: LRU cache per worker, in frames

    # [model]
    architecture: str = "meshgraphnet"
    residual_target: bool = False            # predict the change vs. *_prev (needs history)
    latent_size: int = 128
    num_mlp_layers: int = 2
    num_message_passing_steps: int = 15
    aggregation: str = "sum"

    # [model.mgn_t] — MeshGraphNet-Transformer (arXiv:2601.23177), paper values
    mgn_t_latent_size: int = 64              # width of the message passing stages
    mgn_t_token_dim: int = 32                # width of the transformer
    mgn_t_num_tokens: int = 128              # physical tokens P
    mgn_t_num_heads: int = 4
    mgn_t_num_transformer_blocks: int = 2
    mgn_t_num_pre_mp_steps: int = 2
    mgn_t_num_post_mp_steps: int = 2
    mgn_t_pe_frequencies: int = 8            # not given in the paper
    mgn_t_ffn_ratio: int = 4                 # not given in the paper
    mgn_t_tau_init: float = 0.5              # not given in the paper

    # [training]
    epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0               # gradient clipping (not in the paper)
    seed: int = 42
    device: str = "auto"                     # auto -> cuda if available, else cpu

    # [training.scheduler]
    scheduler: str = "exponential"           # exponential | plateau | cosine | none
    min_learning_rate: float = 1e-6          # end LR (exponential), floor (plateau, cosine)
    lr_decay_steps: Union[int, str] = 5_000_000   # exponential: optimizer steps to reach the
                                                  # end LR; "auto" = epochs x steps per epoch
    scheduler_patience: int = 8              # plateau: epochs without improvement
    scheduler_factor: float = 0.5            # plateau: LR multiplier

    # [training.rollout_validation]
    rollout_validation: bool = False         # select the model by the val rollout error
    rollout_validation_every: int = 1
    rollout_validation_max_sequences: Optional[int] = None
    # mse: normalized target MSE over all steps; pos_rmse: position RMSE (m) over
    # all steps, catches drift that barely shows in the MSE; pos_rmse_final: last step
    rollout_selection_metric: str = "mse"

    # [training.early_stopping]
    early_stopping: bool = True
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 0.0
    early_stopping_min_epochs: int = 30
    early_stopping_smoothing_window: int = 3
    early_stopping_efficiency_window: int = 30
    early_stopping_efficiency_threshold: float = 0.1
    early_stopping_gap_weight: float = 0.5   # weight of the train/val gap in the score

    # [loss]
    loss_type: str = "mse"

    # [noise]
    world_noise_std: float = 0.0             # position noise on free nodes (paper: 3e-3)
    stress_noise_std: float = 0.0            # noise on stress_prev in Pa (history only)

    # [scheduled_sampling]
    scheduled_sampling: bool = False         # needs history; paper: off
    scheduled_sampling_start_ratio: float = 1.0   # teacher-forcing ratio at epoch 0
    scheduled_sampling_end_ratio: float = 0.3     # ... after the warmup
    scheduled_sampling_warmup_epochs: int = 50

    # [checkpointing]
    checkpoint_dir: str = "checkpoints"
    save_every: int = 10                     # periodic checkpoint every N epochs
    log_every: int = 1                       # epoch summary every N epochs
    resume: Optional[str] = None             # checkpoint to continue from

    def __post_init__(self) -> None:
        _check_choice("architecture", self.architecture, ARCHITECTURES)
        _check_choice("scheduler", self.scheduler, SCHEDULERS)
        _check_choice("loss_type", self.loss_type, LOSS_TYPES)
        _check_choice("selection_metric", self.rollout_selection_metric, SELECTION_METRICS)
        if isinstance(self.lr_decay_steps, str) and self.lr_decay_steps != "auto":
            raise ValueError(f"decay_steps must be a number or \"auto\", "
                             f"got {self.lr_decay_steps!r}")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.resume:
            self.resume = None

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_toml(cls, path: str, overrides: Optional[Dict[str, Any]] = None
                  ) -> "TrainingConfig":
        """Read ``path``; ``overrides`` (field -> value, e.g. from the CLI) win."""
        with open(path, "r") as f:
            raw = toml.load(f)
        values: Dict[str, Any] = {}
        unknown = []
        for key, value in _flatten(raw):
            if key in TOML_KEYS:
                values[TOML_KEYS[key]] = value
            else:
                unknown.append(key)
        if unknown:
            print(f"Warning: {path}: ignoring unknown config keys: {', '.join(unknown)}",
                  file=sys.stderr)
        values.update(overrides or {})
        return cls(**values)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TrainingConfig":
        """Restore from ``to_dict`` output, including older checkpoints."""
        field_names = {f.name for f in dataclasses.fields(cls)}
        restored = {}
        for key, value in values.items():
            key = LEGACY_FIELD_NAMES.get(key, key)
            if key in field_names:
                restored[key] = value
        return cls(**restored)

    def resolve_lr_decay_steps(self, steps_per_epoch: int) -> None:
        """``decay_steps = "auto"``: the LR reaches ``min_lr`` at the end of the last
        epoch. ``epochs`` counts from the start of the run, also when resuming."""
        if self.lr_decay_steps == "auto":
            self.lr_decay_steps = self.epochs * steps_per_epoch

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# TOML "section.key" -> TrainingConfig field.
TOML_KEYS = {
    "data.data_dir": "data_dir",
    "data.num_dataloader_workers": "num_dataloader_workers",
    "data.lazy": "lazy_loading",
    "data.cache_frames": "cache_frames",

    "model.architecture": "architecture",
    "model.residual_target": "residual_target",
    "model.meshgraphnet.latent_size": "latent_size",
    "model.meshgraphnet.num_mlp_layers": "num_mlp_layers",
    "model.meshgraphnet.num_message_passing_steps": "num_message_passing_steps",
    "model.meshgraphnet.aggregation": "aggregation",

    "model.mgn_t.latent_size": "mgn_t_latent_size",
    "model.mgn_t.token_dim": "mgn_t_token_dim",
    "model.mgn_t.num_tokens": "mgn_t_num_tokens",
    "model.mgn_t.num_heads": "mgn_t_num_heads",
    "model.mgn_t.num_transformer_blocks": "mgn_t_num_transformer_blocks",
    "model.mgn_t.num_pre_mp_steps": "mgn_t_num_pre_mp_steps",
    "model.mgn_t.num_post_mp_steps": "mgn_t_num_post_mp_steps",
    "model.mgn_t.num_mlp_layers": "num_mlp_layers",
    "model.mgn_t.aggregation": "aggregation",
    "model.mgn_t.pe_frequencies": "mgn_t_pe_frequencies",
    "model.mgn_t.ffn_ratio": "mgn_t_ffn_ratio",
    "model.mgn_t.tau_init": "mgn_t_tau_init",

    "training.epochs": "epochs",
    "training.batch_size": "batch_size",
    "training.learning_rate": "learning_rate",
    "training.weight_decay": "weight_decay",
    "training.max_grad_norm": "max_grad_norm",
    "training.seed": "seed",
    "training.device": "device",

    "training.scheduler.type": "scheduler",
    "training.scheduler.min_lr": "min_learning_rate",
    "training.scheduler.lr": "min_learning_rate",            # former name
    "training.scheduler.decay_steps": "lr_decay_steps",
    "training.scheduler.patience": "scheduler_patience",
    "training.scheduler.factor": "scheduler_factor",

    "training.rollout_validation.enabled": "rollout_validation",
    "training.rollout_validation.every_n_epochs": "rollout_validation_every",
    "training.rollout_validation.max_sequences": "rollout_validation_max_sequences",
    "training.rollout_validation.selection_metric": "rollout_selection_metric",

    "training.early_stopping.enabled": "early_stopping",
    "training.early_stopping.patience": "early_stopping_patience",
    "training.early_stopping.min_delta": "early_stopping_min_delta",
    "training.early_stopping.min_epochs": "early_stopping_min_epochs",
    "training.early_stopping.smoothing_window": "early_stopping_smoothing_window",
    "training.early_stopping.efficiency_window": "early_stopping_efficiency_window",
    "training.early_stopping.efficiency_threshold": "early_stopping_efficiency_threshold",
    "training.early_stopping.gap_weight": "early_stopping_gap_weight",
    "training.early_stopping.lambda": "early_stopping_gap_weight",   # former name

    "loss.type": "loss_type",

    "noise.world_noise_std": "world_noise_std",
    "noise.stress_noise_std": "stress_noise_std",

    "scheduled_sampling.enabled": "scheduled_sampling",
    "scheduled_sampling.start_ratio": "scheduled_sampling_start_ratio",
    "scheduled_sampling.end_ratio": "scheduled_sampling_end_ratio",
    "scheduled_sampling.warmup_epochs": "scheduled_sampling_warmup_epochs",
    # former section [rollout]
    "rollout.scheduled_sampling": "scheduled_sampling",
    "rollout.scheduled_sampling_start_ratio": "scheduled_sampling_start_ratio",
    "rollout.scheduled_sampling_end_ratio": "scheduled_sampling_end_ratio",
    "rollout.scheduled_sampling_warmup_epochs": "scheduled_sampling_warmup_epochs",

    "checkpointing.checkpoint_dir": "checkpoint_dir",
    "checkpointing.save_every": "save_every",
    "checkpointing.log_every": "log_every",
    "checkpointing.resume": "resume",
}

# Field names used by checkpoints written before the refactoring.
LEGACY_FIELD_NAMES = {
    "node_hidden_dim": "latent_size",
    "sda_lr": "min_learning_rate",
    "scheduler_decay_steps": "lr_decay_steps",
    "early_stopping_lambdas": "early_stopping_gap_weight",
}


def _flatten(table: Dict[str, Any], prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """Nested TOML tables -> ("section.sub.key", value) pairs."""
    for key, value in table.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _flatten(value, f"{dotted}.")
        else:
            yield dotted, value


def _check_choice(name: str, value: str, choices: Tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
