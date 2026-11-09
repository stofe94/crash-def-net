"""
config.py — Training configuration: TOML file -> flat ``TrainingConfig``.

The TOML sections (see ``examples/quick_start/train.toml``) map onto the
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
    DEFAULT_WORLD_EDGE_EXCLUDE_DISTANCE,
    WORLD_EDGE_MODES,
    FeatureSelection,
)
from models.mgn_t import TRANSFORMER_NODES
from train.early_stopping import CRITERIA

SCHEDULERS = ("exponential", "plateau", "cosine", "none")
LOSS_TYPES = ("mse", "mae", "huber", "smooth_l1")
ARCHITECTURES = ("meshgraphnets", "mgn_t")
# Fields with a fixed set of values: checked on construction, listed in the reference.
CHOICES = {"architecture": ARCHITECTURES, "world_edges": WORLD_EDGE_MODES,
           "mgn_t_transformer_nodes": TRANSFORMER_NODES, "scheduler": SCHEDULERS,
           "loss_type": LOSS_TYPES}
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


def setting(default: Any = dataclasses.MISSING, doc: str = "", *,
            default_factory: Any = dataclasses.MISSING) -> Any:
    """A config field with its description, the text of the configuration
    reference in the documentation (``utils/docs_reference.py``)."""
    return field(default=default, default_factory=default_factory, metadata={"doc": doc})


@dataclass
class TrainingConfig:
    """All training settings. Defaults follow the paper where it gives a value."""

    # Origin of the settings (set by ``from_toml``, not TOML keys)
    config_file: Optional[str] = None        # TOML file, relative to the repo if inside it
    config_file_sha256: Optional[str] = None  # its content at the start of the run
    config_overrides: Dict[str, Any] = field(default_factory=dict)  # command-line options

    # [data]
    data_dir: str = setting("data/deforming_plate_s10_small",
                            "dataset folder written by the exporter (`dataset.db`, "
                            "`normalizer.json`, `metadata.json`)")
    num_dataloader_workers: int = setting(0, "DataLoader processes that build the graphs; "
                                             "0: in the main process (CPU), 2–4 on a GPU machine")
    lazy_loading: bool = setting(False, "read the frames from `dataset.db` on access instead of "
                                        "loading the splits into memory, for splits that do not "
                                        "fit into RAM")
    cache_frames: int = setting(0, "lazy: frames kept per process, least recently used out; "
                                   "0: no cache")
    static_cache_mb: float = setting(1024, "lazy: static graph data (mesh edges, world edge "
                                           "selection, rest positions) kept per process, in MB")
    local_features: Optional[List[str]] = setting(
        None, "local input features the model sees: static node features of the export by "
              "name (`static_feature_names` in `metadata.json`); `[]`: none; unset: all")
    global_features: Optional[List[str]] = setting(
        None, "global input features the model sees: run parameters of the export by name "
              "(`global_feature_names` in `metadata.json`); `[]`: none; unset: all")
    target_fields: Optional[List[str]] = setting(
        None, "result fields the model predicts besides the velocity, by name "
              "(`target_feature_names` in `metadata.json`), with their `<name>_prev` inputs; "
              "`[]`: the velocity only; unset: all")
    start_frame: Optional[int] = setting(
        None, "raw frame of the run the training starts with: the first current frame of the "
              "samples and the initial state of the rollouts (validation, test, and by default "
              "`predict/rollout.py`), one of the dataset; unset: its first frame")
    end_frame: Optional[int] = setting(
        None, "raw frame of the run the training ends with: the last target of the samples and "
              "of the rollouts, one of the dataset; unset: its last frame")

    # [model]
    architecture: str = setting("meshgraphnets", "model architecture")
    residual_target: bool = setting(False, "predict the change of every target that has a "
                                           "`*_prev` input instead of its value; needs an "
                                           "export with `--history 1`")
    world_edges: str = setting("all", "node pairs within the contact radius that get a world "
                                      "edge: `all` (paper), `contact` (different bodies only), "
                                      "`self_contact_aware` (pairs not connected through the "
                                      "material)")
    world_edge_exclude_distance: float = setting(
        DEFAULT_WORLD_EDGE_EXCLUDE_DISTANCE,
        "`self_contact_aware`: pairs with a path along the mesh edges of the rest state "
        "shorter than this many contact radii get no world edge")
    world_edge_exclude_hops: Optional[int] = setting(
        None, "`self_contact_aware`: pairs at most this many mesh edges apart get no world "
              "edge; instead of `world_edge_exclude_distance`")
    latent_size: int = setting(128, "width of every MLP and latent (paper 128)")
    num_mlp_layers: int = setting(2, "hidden layers of every MLP")
    num_message_passing_steps: int = setting(15, "message passing steps of the processor "
                                                 "(paper 15)")
    aggregation: str = setting("sum", "aggregation of the messages at a node: `sum` (paper) "
                                      "or `mean`")

    # [model.mgn_t] — MeshGraphNet-Transformer (arXiv:2601.23177v4; see models/mgn_t.py)
    mgn_t_latent_size: int = setting(64, "width of the message passing stages")
    mgn_t_token_dim: int = setting(32, "width of the transformer")
    mgn_t_num_tokens: int = setting(128, "physical tokens P of the physics attention "
                                         "(paper, deforming_plate: 16)")
    mgn_t_num_heads: int = setting(4, "attention heads")
    mgn_t_num_transformer_blocks: int = setting(2, "transformer blocks")
    mgn_t_num_pre_mp_steps: int = setting(2, "local message passing steps before the "
                                             "transformer")
    mgn_t_num_post_mp_steps: int = setting(2, "local message passing steps after the "
                                              "transformer")
    mgn_t_pe_frequencies: int = setting(8, "frequencies of the positional encoding (not given "
                                           "in the paper)")
    mgn_t_ffn_ratio: int = setting(4, "hidden width of the feed-forward block over "
                                      "`token_dim` (not given in the paper)")
    mgn_t_tau_init: float = setting(0.5, "initial temperature of the slice weights under "
                                         "softplus: τ starts at `tau_init` + ln 2 (paper: "
                                         "τ₀ = 1.0; see models/mgn_t.py)")
    mgn_t_transformer_nodes: str = setting("structure", "nodes of the global stage: "
                                                        "`structure` (paper: without the "
                                                        "actuator) or `all` (every node)")

    # [training]
    epochs: int = setting(100, "training epochs")
    batch_size: int = setting(2, "graphs per optimizer step (MeshGraphNets paper 2, MGN-T "
                                 "paper 32)")
    micro_batch_size: Optional[int] = setting(
        None, "machine setting: graphs per forward pass, a divisor of `batch_size`; "
              "`batch_size / micro_batch_size` passes make one optimizer step with the same "
              "result. Unset: `batch_size`")
    learning_rate: float = setting(1e-4, "initial learning rate of Adam (paper 1e-4)")
    weight_decay: float = setting(0.0, "weight decay of Adam")
    max_grad_norm: float = setting(1.0, "gradient clipping to this norm (not in the paper)")
    seed: int = setting(42, "seed of the weight initialization, the sample order and the noise")
    device: str = setting("auto", "`auto` (GPU if available), `cpu` or `cuda`")

    # [training.scheduler]
    scheduler: str = setting("exponential", "learning rate schedule: per optimizer step "
                                            "(`exponential`), per epoch (`cosine`), at "
                                            "evaluations (`plateau`) or none")
    min_learning_rate: float = setting(1e-6, "end learning rate (`exponential`), lower bound "
                                             "(`plateau`, `cosine`)")
    lr_decay_steps: Union[int, str] = setting(
        5_000_000, "`exponential`: optimizer steps to reach `min_lr` (paper 5 M); `\"auto\"` "
                   "= epochs × optimizer steps per epoch, so it is reached at the end of the run")
    scheduler_patience: int = setting(8, "`plateau`: evaluations without improvement of the "
                                         "first criterion before the learning rate drops")
    scheduler_factor: float = setting(0.5, "`plateau`: factor of the drop")

    # [training.rollout_validation]
    rollout_validation: bool = setting(False, "roll out the val split at evaluations; its "
                                              "errors become selection criteria")
    rollout_validation_every: int = setting(1, "epochs between evaluations (also in the first "
                                               "and last epoch)")
    rollout_validation_max_sequences: Optional[int] = setting(
        None, "roll out only the first N val trajectories. Unset: all")

    # [training.early_stopping] — also the model selection (train/early_stopping.py)
    early_stopping: bool = setting(True, "stop when no criterion reaches a new low; `false`: "
                                         "all epochs, the model selection still applies")
    early_stopping_patience: int = setting(30, "evaluations without a new low in any "
                                               "criterion before the run stops")
    early_stopping_min_epochs: int = setting(30, "epochs before a run may stop")
    early_stopping_criteria: List[str] = setting(
        doc="selection criteria; the evaluation closest to the best value in all of them at "
            "once is selected",
        default_factory=lambda: list(CRITERIA))

    # [loss]
    loss_type: str = setting("mse", "loss on the normalized targets")

    # [noise]
    world_noise_std: float = setting(0.0, "position noise on the free nodes, standard "
                                          "deviation in the length unit of the data "
                                          "(`length_unit` of the export; paper 3e-3 m)")
    stress_noise_std: float = setting(0.0, "noise on `stress_prev`, standard deviation in "
                                           "the stress unit of the data (`stress_unit` of the "
                                           "export, Pa by default; history layout only)")

    # [scheduled_sampling]
    scheduled_sampling: bool = setting(False, "replace the `*_prev` inputs of a share of the "
                                              "nodes by the model's prediction (history layout "
                                              "only; not in the paper)")
    scheduled_sampling_start_ratio: float = setting(1.0, "share of the nodes that keep the "
                                                         "ground truth at epoch 0")
    scheduled_sampling_end_ratio: float = setting(0.3, "share of the nodes that keep the "
                                                       "ground truth after the warmup")
    scheduled_sampling_warmup_epochs: int = setting(50, "epochs over which the share goes "
                                                        "linearly from the start to the end "
                                                        "ratio")

    # [checkpointing]
    checkpoint_dir: str = setting("checkpoints", "output folder of the run")
    save_every: int = setting(10, "write `checkpoint_epoch_N.pt` every N epochs "
                                  "(`best_model.pt` always)")
    log_every: int = setting(1, "epoch summary in the log every N epochs")
    resume: Optional[str] = setting(None, "checkpoint to continue from")

    def __post_init__(self) -> None:
        self.architecture = LEGACY_ARCHITECTURE_NAMES.get(self.architecture, self.architecture)
        for name, choices in CHOICES.items():
            _check_choice(name, getattr(self, name), choices)
        if self.world_edge_exclude_hops is not None and self.world_edge_exclude_hops < 1:
            raise ValueError(f"world_edge_exclude_hops must be >= 1, "
                             f"got {self.world_edge_exclude_hops}")
        if self.micro_batch_size is not None and not (
                1 <= self.micro_batch_size <= self.batch_size
                and self.batch_size % self.micro_batch_size == 0):
            raise ValueError(f"micro_batch_size must divide batch_size {self.batch_size}, "
                             f"got {self.micro_batch_size}")
        if None not in (self.start_frame, self.end_frame) and self.end_frame <= self.start_frame:
            raise ValueError(f"end_frame {self.end_frame} must come after start_frame "
                             f"{self.start_frame}")
        if self.world_edge_exclude_distance <= 0:
            raise ValueError(f"world_edge_exclude_distance must be > 0, "
                             f"got {self.world_edge_exclude_distance}")
        for name in ("local_features", "global_features", "target_fields"):
            names = getattr(self, name)
            if names is not None and not (isinstance(names, list)
                                          and all(isinstance(n, str) for n in names)):
                raise ValueError(f"{name} must be a list of names, got {names!r}")
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
            if key in TOML_KEYS or key in LEGACY_TOML_KEYS:
                values[TOML_KEYS.get(key) or LEGACY_TOML_KEYS[key]] = value
            elif key in OBSOLETE_TOML_KEYS:
                obsolete.append(key)
            else:
                unknown.append(key)
        if unknown:
            print(f"Warning: {path}: ignoring unknown config keys: {', '.join(unknown)}",
                  file=sys.stderr)
        if obsolete:
            print(f"Note: {path}: no longer used (early stopping by criteria, see "
                  f"docs/concepts/training.md): {', '.join(obsolete)}", file=sys.stderr)
        dropped = LEGACY_CRITERIA.intersection(values.get("early_stopping_criteria", ()))
        if dropped:
            print(f"Note: {path}: no longer a selection criterion (it favours early, "
                  f"undertrained epochs, see docs/concepts/training.md): "
                  f"{', '.join(sorted(dropped))}",
                  file=sys.stderr)
        if {"world_edge_exclude_distance", "world_edge_exclude_hops"} <= values.keys():
            raise ValueError(f"{path}: set world_edge_exclude_distance or (as up to v0.3.0) "
                             "world_edge_exclude_hops, not both")
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

    @property
    def features(self) -> FeatureSelection:
        """The features of the export this run uses."""
        return FeatureSelection(self.local_features, self.global_features, self.target_fields)

    @property
    def accumulation_steps(self) -> int:
        """Micro-batches per optimizer step."""
        return self.batch_size // (self.micro_batch_size or self.batch_size)

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
            unused |= {"world_edge_exclude_distance", "world_edge_exclude_hops"}
        elif self.world_edge_exclude_hops is None:
            unused.add("world_edge_exclude_hops")
        else:
            unused.add("world_edge_exclude_distance")
        if not self.lazy_loading:
            unused |= {"cache_frames", "static_cache_mb"}
        if self.micro_batch_size is None:
            unused.add("micro_batch_size")
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
    "data.local_features": "local_features",
    "data.global_features": "global_features",
    "data.target_fields": "target_fields",
    "data.start_frame": "start_frame",
    "data.end_frame": "end_frame",

    "model.architecture": "architecture",
    "model.residual_target": "residual_target",
    "model.world_edges": "world_edges",
    "model.world_edge_exclude_distance": "world_edge_exclude_distance",
    "model.world_edge_exclude_hops": "world_edge_exclude_hops",
    "model.meshgraphnets.latent_size": "latent_size",
    "model.meshgraphnets.num_mlp_layers": "num_mlp_layers",
    "model.meshgraphnets.num_message_passing_steps": "num_message_passing_steps",
    "model.meshgraphnets.aggregation": "aggregation",

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
    "training.micro_batch_size": "micro_batch_size",
    "training.learning_rate": "learning_rate",
    "training.weight_decay": "weight_decay",
    "training.max_grad_norm": "max_grad_norm",
    "training.seed": "seed",
    "training.device": "device",

    "training.scheduler.type": "scheduler",
    "training.scheduler.min_lr": "min_learning_rate",
    "training.scheduler.decay_steps": "lr_decay_steps",
    "training.scheduler.patience": "scheduler_patience",
    "training.scheduler.factor": "scheduler_factor",

    "training.rollout_validation.enabled": "rollout_validation",
    "training.rollout_validation.every_n_epochs": "rollout_validation_every",
    "training.rollout_validation.max_sequences": "rollout_validation_max_sequences",

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

    "checkpointing.checkpoint_dir": "checkpoint_dir",
    "checkpointing.save_every": "save_every",
    "checkpointing.log_every": "log_every",
    "checkpointing.resume": "resume",
}

# Former TOML keys, translated when a config is read (never written).
LEGACY_TOML_KEYS = {
    # misspelled section [model.meshgraphnet] (up to v0.2.0, see ARCHITECTURES)
    "model.meshgraphnet.latent_size": "latent_size",
    "model.meshgraphnet.num_mlp_layers": "num_mlp_layers",
    "model.meshgraphnet.num_message_passing_steps": "num_message_passing_steps",
    "model.meshgraphnet.aggregation": "aggregation",
    "training.scheduler.lr": "min_learning_rate",
    # single selection metric, translated into the criteria by from_toml
    "training.rollout_validation.selection_metric": "selection_metric",
    # section [rollout]
    "rollout.scheduled_sampling": "scheduled_sampling",
    "rollout.scheduled_sampling_start_ratio": "scheduled_sampling_start_ratio",
    "rollout.scheduled_sampling_end_ratio": "scheduled_sampling_end_ratio",
    "rollout.scheduled_sampling_warmup_epochs": "scheduled_sampling_warmup_epochs",
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
