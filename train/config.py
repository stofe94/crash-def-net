"""
config.py — Training configuration: TOML file -> flat ``TrainingConfig``.

The TOML sections (see ``config.example.toml``) map onto the
fields of one flat dataclass via ``TOML_KEYS``. Unknown keys are reported, so
typos and settings that no longer exist do not go unnoticed. The config is
stored in every checkpoint (``to_dict``) and restored with ``from_dict``,
which also understands the field names of older checkpoints.

The stored config names the TOML file it was read from (path and SHA-256) and
the command-line options that overrode it, and lists only the settings that
take effect in the run (``unused_fields``): no MGN-T settings for a
MeshGraphNets run, no plateau settings for the exponential scheduler, and so on.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import toml
import torch

from dataset_preprocessor.graph_features import (
    DEFAULT_WORLD_EDGE_EXCLUDE_HOPS,
    WORLD_EDGE_MODES,
)
from models.mgn_t import TRANSFORMER_NODES
from train.early_stopping import CRITERIA

SCHEDULERS = ("exponential", "plateau", "cosine", "none")
LOSS_TYPES = ("mse", "mae", "huber", "smooth_l1")
ARCHITECTURES = ("meshgraphnets", "mgn_t")
# Up to v0.2.0 the architecture was misspelled "meshgraphnet"; it was corrected to
# "meshgraphnets" to match the name of the method, MeshGraphNets (Pfaff et al., 2021).
# Configs and checkpoints of v0.2.0 still carry the old spelling, so it is translated
# when they are read (never written), and they keep working.
LEGACY_ARCHITECTURE_NAMES = {"meshgraphnet": "meshgraphnets"}
# Configs and checkpoints before the selection by criteria chose one metric
# ([training.rollout_validation] selection_metric, default "mse"); it is
# translated into the criteria list when they are read.
LEGACY_SELECTION_METRICS = {"mse": ["rollout_mse"], "pos_rmse": ["rollout_pos_rmse"],
                            "pos_rmse_final": ["rollout_pos_rmse_final"],
                            "consistent": list(CRITERIA)}
# The generalization gap was a selection criterion in development versions
# before v0.3.0; it favoured early, undertrained epochs. Configs and checkpoints
# that list it are read without it.
LEGACY_CRITERIA = {"gap"}

_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainingConfig:
    """All training settings. Defaults follow the paper where it gives a value."""

    # Origin of the settings (set by ``from_toml``, not TOML keys)
    config_file: Optional[str] = None        # TOML file, relative to the repo if inside it
    config_file_sha256: Optional[str] = None  # its content at the start of the run
    config_overrides: Dict[str, Any] = field(default_factory=dict)  # command-line options

    # [data]
    data_dir: str = "data/deforming_plate_s10_small"
    num_dataloader_workers: int = 0          # > 0: build graphs in spawned workers
    lazy_loading: bool = False               # read frames from the DB on access (v2 DB)
    cache_frames: int = 0                    # lazy: LRU cache per worker, in frames
    static_cache_mb: float = 1024            # lazy: static graph data per worker (LRU), MB

    # [model]
    architecture: str = "meshgraphnets"
    residual_target: bool = False            # predict the change vs. *_prev (needs history)
    world_edges: str = "all"                 # all (paper) | contact | self_contact_aware
    world_edge_exclude_hops: int = DEFAULT_WORLD_EDGE_EXCLUDE_HOPS   # self_contact_aware
    latent_size: int = 128
    num_mlp_layers: int = 2
    num_message_passing_steps: int = 15
    aggregation: str = "sum"

    # [model.mgn_t] — MeshGraphNet-Transformer (arXiv:2601.23177v4; see models/mgn_t.py)
    mgn_t_latent_size: int = 64              # width of the message passing stages
    mgn_t_token_dim: int = 32                # width of the transformer
    mgn_t_num_tokens: int = 128              # physical tokens P
    mgn_t_num_heads: int = 4
    mgn_t_num_transformer_blocks: int = 2
    mgn_t_num_pre_mp_steps: int = 2
    mgn_t_num_post_mp_steps: int = 2
    mgn_t_pe_frequencies: int = 8            # not given in the paper
    mgn_t_ffn_ratio: int = 4                 # not given in the paper
    mgn_t_tau_init: float = 0.5              # see models/mgn_t.py (softplus)
    mgn_t_transformer_nodes: str = "structure"   # structure (paper) | all (up to v0.2.0)

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

    # [training.early_stopping] — also the model selection (train/early_stopping.py)
    early_stopping: bool = True              # False: all epochs, the selection still applies
    early_stopping_patience: int = 30        # evaluations the selected model may be old
    early_stopping_min_epochs: int = 30
    # selected: the evaluation closest to the best value in all criteria at once
    early_stopping_criteria: List[str] = field(default_factory=lambda: list(CRITERIA))

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
        self.architecture = LEGACY_ARCHITECTURE_NAMES.get(self.architecture, self.architecture)
        _check_choice("architecture", self.architecture, ARCHITECTURES)
        _check_choice("world_edges", self.world_edges, WORLD_EDGE_MODES)
        _check_choice("transformer_nodes", self.mgn_t_transformer_nodes, TRANSFORMER_NODES)
        if self.world_edge_exclude_hops < 1:
            raise ValueError(f"world_edge_exclude_hops must be >= 1, "
                             f"got {self.world_edge_exclude_hops}")
        _check_choice("scheduler", self.scheduler, SCHEDULERS)
        _check_choice("loss_type", self.loss_type, LOSS_TYPES)
        self.early_stopping_criteria = [c for c in self.early_stopping_criteria
                                        if c not in LEGACY_CRITERIA]
        unknown = [c for c in self.early_stopping_criteria if c not in CRITERIA]
        if unknown or not self.early_stopping_criteria:
            raise ValueError(f"early_stopping criteria must be a non-empty subset of "
                             f"{CRITERIA}, got {self.early_stopping_criteria}")
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
        content = Path(path).read_bytes()
        raw = toml.loads(content.decode("utf-8"))
        values: Dict[str, Any] = {}
        unknown, obsolete = [], []
        for key, value in _flatten(raw):
            if key in TOML_KEYS:
                values[TOML_KEYS[key]] = value
            elif key in OBSOLETE_TOML_KEYS:
                obsolete.append(key)
            else:
                unknown.append(key)
        if unknown:
            print(f"Warning: {path}: ignoring unknown config keys: {', '.join(unknown)}",
                  file=sys.stderr)
        if obsolete:
            print(f"Note: {path}: no longer used (early stopping by criteria, see "
                  f"docs/training.md): {', '.join(obsolete)}", file=sys.stderr)
        dropped = LEGACY_CRITERIA.intersection(values.get("early_stopping_criteria", ()))
        if dropped:
            print(f"Note: {path}: no longer a selection criterion (it favours early, "
                  f"undertrained epochs, see docs/training.md): {', '.join(sorted(dropped))}",
                  file=sys.stderr)
        if "selection_metric" in values:   # older config: one selection metric
            metric = values.pop("selection_metric")
            values.setdefault("early_stopping_criteria",
                              _legacy_criteria(metric, values.get("rollout_validation", False)))
        values.update(overrides or {})
        values.update(config_file=_display_path(Path(path)),
                      config_file_sha256=hashlib.sha256(content).hexdigest(),
                      config_overrides=dict(overrides or {}))
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
        # Stored before the option: the transformer covered every node.
        restored.setdefault("mgn_t_transformer_nodes", "all")
        if "early_stopping_criteria" not in restored:
            # Stored before the selection by criteria: it used one metric.
            restored["early_stopping_criteria"] = _legacy_criteria(
                values.get("rollout_selection_metric", "mse"),
                values.get("rollout_validation", False))
        return cls(**restored)

    def resolve_lr_decay_steps(self, steps_per_epoch: int) -> None:
        """``decay_steps = "auto"``: the LR reaches ``min_lr`` at the end of the last
        epoch. ``epochs`` counts from the start of the run, also when resuming."""
        if self.lr_decay_steps == "auto":
            self.lr_decay_steps = self.epochs * steps_per_epoch

    def unused_fields(self) -> Set[str]:
        """Settings without effect given the others: those of the other
        architecture and of options that are switched off or not selected."""
        names = [f.name for f in dataclasses.fields(self)]
        unused = set()
        if self.architecture != "meshgraphnets":
            unused |= {"latent_size", "num_message_passing_steps"}
        if self.architecture != "mgn_t":
            unused |= {name for name in names if name.startswith("mgn_t_")}
        if self.world_edges != "self_contact_aware":
            unused.add("world_edge_exclude_hops")
        if not self.lazy_loading:
            unused |= {"cache_frames", "static_cache_mb"}
        if self.scheduler != "exponential":
            unused.add("lr_decay_steps")
        if self.scheduler != "plateau":
            unused |= {"scheduler_patience", "scheduler_factor"}
        if self.scheduler == "none":
            unused.add("min_learning_rate")
        if not self.rollout_validation:
            unused |= {"rollout_validation_every", "rollout_validation_max_sequences"}
        if not self.early_stopping:   # the criteria still select the model
            unused |= {"early_stopping_patience", "early_stopping_min_epochs"}
        if not self.scheduled_sampling:
            unused |= {name for name in names if name.startswith("scheduled_sampling_")}
        return unused

    def to_dict(self) -> Dict[str, Any]:
        """The settings in effect (see ``unused_fields``); ``from_dict`` restores
        the others with their defaults."""
        unused = self.unused_fields()
        return {key: value for key, value in dataclasses.asdict(self).items()
                if key not in unused}


# TOML "section.key" -> TrainingConfig field.
TOML_KEYS = {
    "data.data_dir": "data_dir",
    "data.num_dataloader_workers": "num_dataloader_workers",
    "data.lazy": "lazy_loading",
    "data.cache_frames": "cache_frames",
    "data.static_cache_mb": "static_cache_mb",

    "model.architecture": "architecture",
    "model.residual_target": "residual_target",
    "model.world_edges": "world_edges",
    "model.world_edge_exclude_hops": "world_edge_exclude_hops",
    "model.meshgraphnets.latent_size": "latent_size",
    "model.meshgraphnets.num_mlp_layers": "num_mlp_layers",
    "model.meshgraphnets.num_message_passing_steps": "num_message_passing_steps",
    "model.meshgraphnets.aggregation": "aggregation",
    # former, misspelled section [model.meshgraphnet] (up to v0.2.0, see ARCHITECTURES)
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
    "model.mgn_t.transformer_nodes": "mgn_t_transformer_nodes",

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
    # former single selection metric, translated into the criteria by from_toml
    "training.rollout_validation.selection_metric": "selection_metric",

    "training.early_stopping.enabled": "early_stopping",
    "training.early_stopping.patience": "early_stopping_patience",
    "training.early_stopping.min_epochs": "early_stopping_min_epochs",
    "training.early_stopping.criteria": "early_stopping_criteria",

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

# Keys of the former early stopping (one metric with min_delta, smoothing and the
# gap / efficiency terms; in development versions before v0.3.0 also the stop on
# a growing gap, overfitting_gap); older configs still load, the keys have no effect.
OBSOLETE_TOML_KEYS = {f"training.early_stopping.{key}" for key in (
    "min_delta", "smoothing_window", "efficiency_window", "efficiency_threshold",
    "gap_weight", "lambda", "overfitting_gap")}

# Field names used by checkpoints written before the refactoring.
LEGACY_FIELD_NAMES = {
    "node_hidden_dim": "latent_size",
    "sda_lr": "min_learning_rate",
    "scheduler_decay_steps": "lr_decay_steps",
}


def _legacy_criteria(selection_metric: str, rollout_validation: bool) -> List[str]:
    """Criteria equivalent to the former single selection metric: the rollout
    metric with rollout validation, else the single-step val loss."""
    if not rollout_validation:
        return ["val_loss"]
    if selection_metric not in LEGACY_SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of "
                         f"{tuple(LEGACY_SELECTION_METRICS)}, got {selection_metric!r}")
    return list(LEGACY_SELECTION_METRICS[selection_metric])


def _flatten(table: Dict[str, Any], prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """Nested TOML tables -> ("section.sub.key", value) pairs."""
    for key, value in table.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _flatten(value, f"{dotted}.")
        else:
            yield dotted, value


def _display_path(path: Path) -> str:
    """Path relative to the repo if the file lies inside it, else absolute."""
    path = path.resolve()
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _check_choice(name: str, value: str, choices: Tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
