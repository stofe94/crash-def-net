"""
trainer.py — Training of the models: one step from the ground truth, or rollout training.

Per batch of the one-step training (in this order):

1. **Scheduled sampling** (optional, needs history): the model predicts step t
   from its predecessor sample (with ``scheduled_sampling_steps`` > 1: rolls out from
   that many steps back); on a random subset of nodes that prediction replaces the
   ``*_prev`` inputs, and the current positions move with it.
2. **Position noise** (``world_noise_std``, paper App. A.2.2): Gaussian noise on
   the current positions of the free nodes; velocity target, ``velocity_prev``
   and all edges are adjusted consistently.
3. **Stress noise** (optional, needs history) on ``stress_prev``.
4. Loss on the normalized targets of the nodes in ``loss_mask``, without targets a
   node does not have (``target_given``), plus the optional terms on the length change
   of the mesh edges and on the mean velocity of the free rigid bodies
   (``loss_*_weight``); gradient clipping, optimizer step; the exponential LR schedule
   steps per batch.

**Rollout training** (``rollout_training``, needs history) replaces the one-step
batches by windows of up to ``rollout_training_steps`` consecutive samples of a
trajectory (``RolloutWindowSampler``): the first step of a window starts from the ground
truth (with the noise above), every further one from the model's prediction of the step
before, written into the ``*_prev`` inputs with the positions and all edges moved along,
as in the rollout. The loss covers every step; its velocity target, corrected by the
displacement, makes it the position error after the step. The gradients flow through the
whole window; every step but the last is recomputed in the backward pass (activation
checkpointing), so that memory stays near that of one step.

After every epoch the single-step validation loss is computed; with rollout
validation every N epochs also the autoregressive rollout error on the val
split, and at every such evaluation the single-step loss on fixed train samples
without noise (with the val loss: the generalization gap). ``EarlyStopping``
(train/early_stopping.py) selects the model by the configured criteria and
decides when to stop. Since it may return to an earlier evaluation, the weights
of the evaluations it can still select are kept in ``candidates/`` until the
run has ended.

The last history record notes how the run ended (``end``: ``early_stopping`` or
``epochs``); an aborted run has none. Every record holds the peak memory of the run
so far and the machine (``peak_memory``), so that the figures of a GPU and of a CPU
machine can be told apart.
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from tqdm import tqdm

from dataset_preprocessor import (
    PredecessorPairDataset,
    RolloutWindowSampler,
    TrajectoryDataset,
    collate_with_predecessor,
)
from dataset_preprocessor.dataset import load_statistics
from dataset_preprocessor.graph_features import (
    ALL_FEATURES,
    PREV_STRESS_NAMES,
    PREV_VELOCITY_NAMES,
    VELOCITY_TARGET_COLUMNS,
    VELOCITY_TARGET_NAMES,
    FeatureSelection,
    add_world_edges,
    displace_nodes,
    free_node_mask,
    prev_to_target_columns,
)
from models.meshgraphnets import MeshGraphNets, count_parameters
from train.config import TrainingConfig
from train.early_stopping import ROLLOUT_CRITERIA, EarlyStopping
from utils.normalizer import Normalizer
from utils.timezone import Formatter, now
from utils.units import SI_LENGTH, SI_STRESS, STRESS_FIELDS, si_factors

logger = logging.getLogger("train")

# Optimizer steps between two progress bar updates of the loss.
PROGRESS_EVERY = 100
# Train samples for the loss without noise at every evaluation (generalization gap):
# this share of the train split, evenly spread, at least TRAIN_EVAL_MIN_SAMPLES (all
# of a smaller split). A few hundred samples give a stable mean, and the cost stays a
# fixed share of an epoch instead of growing with the val split.
TRAIN_EVAL_SHARE = 0.05
TRAIN_EVAL_MIN_SAMPLES = 200

LOSS_FUNCTIONS = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.HuberLoss,
    "smooth_l1": nn.SmoothL1Loss,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data and model
# ─────────────────────────────────────────────────────────────────────────────

def train_eval_indices(num_samples: int) -> List[int]:
    """Train samples for the loss without noise: evenly spread over the split."""
    count = min(num_samples, max(TRAIN_EVAL_MIN_SAMPLES, round(TRAIN_EVAL_SHARE * num_samples)))
    return sorted({round(i * (num_samples - 1) / max(count - 1, 1)) for i in range(count)})


def create_dataloaders(config: TrainingConfig, normalizer: Normalizer
                       ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Train/val/test loaders over ``<data_dir>/dataset.db``.

    With scheduled sampling every train item also carries its predecessor
    sample, with rollout training the samples before it in its window. The test loader is None for an empty test split. With
    ``lazy_loading`` the frames stay in the DB and every worker reads the
    frames of its samples through its own connection.
    """
    db_path = Path(config.data_dir) / "dataset.db"
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found — export it first: "
                                "python -m dataset_preprocessor.export")
    loading = dict(lazy=config.lazy_loading, cache_frames=config.cache_frames,
                   static_cache_mb=config.static_cache_mb, world_edges=config.world_edges,
                   world_edge_exclude_distance=config.world_edge_exclude_distance,
                   world_edge_exclude_hops=config.world_edge_exclude_hops,
                   features=config.features, start_frame=config.start_frame,
                   end_frame=config.end_frame)
    # With position noise every train batch gets its world edges anew on the
    # device (displace_nodes), with rollout training every step after the first of a
    # window: the contact search in the loader would be wasted.
    world_edges_in_loader = config.world_noise_std <= 0 and not config.rollout_training
    train_set = TrajectoryDataset(db_path, "train", normalizer,
                                      build_world_edges=world_edges_in_loader, **loading)
    val_set = TrajectoryDataset(db_path, "val", normalizer, **loading)
    test_set = TrajectoryDataset(db_path, "test", normalizer, **loading)
    mode = (f"lazy, cache {config.cache_frames} frames and {config.static_cache_mb:g} MB "
            "of static data per worker" if config.lazy_loading else "in memory")
    world = config.world_edges
    if world == "self_contact_aware" and config.world_edge_exclude_hops is not None:
        world += f", > {config.world_edge_exclude_hops} hops"
    elif world == "self_contact_aware":
        world += f", path through the mesh >= {config.world_edge_exclude_distance:g} r_W"
    if not world_edges_in_loader:
        world += f"; train: built on {config.device}"
    logger.info(f"Data: {db_path} — train {len(train_set)}, val {len(val_set)}, "
                f"test {len(test_set)} samples (history {train_set.history}, {mode}, "
                f"{config.num_dataloader_workers} workers, world edges: {world})")
    if config.features != ALL_FEATURES:
        logger.info(f"Features of the export in use: local {train_set.static_names}, "
                    f"global {train_set.global_names}, result fields {train_set.field_names}")
        if train_set.unused_features:
            logger.warning(f"Features of the export that [data] of the config leaves out: "
                           f"{train_set.unused_features}")

    train_collate = Batch.from_data_list
    sampler = None
    if config.rollout_training:
        sampler = RolloutWindowSampler(train_set, config.rollout_training_steps)
        train_set = PredecessorPairDataset(train_set, config.rollout_training_steps - 1)
        train_collate = collate_with_predecessor
    elif config.scheduled_sampling:
        train_set = PredecessorPairDataset(train_set, config.scheduled_sampling_steps)
        train_collate = collate_with_predecessor

    workers = config.num_dataloader_workers
    if workers > 0:
        # Workers receive the dataset's tensors through shared memory. By default
        # every shared tensor keeps a file descriptor open in the main process, and
        # splits of a few hundred trajectories exceed the usual limit of 1,024.
        torch.multiprocessing.set_sharing_strategy("file_system")
    common = dict(batch_size=config.batch_size, pin_memory=config.device == "cuda")
    # Train: micro-batches, accumulated to batch_size per optimizer step.
    train_batch_size = config.micro_batch_size or config.batch_size
    parallel = dict(num_workers=workers, persistent_workers=workers > 0,
                    multiprocessing_context="spawn" if workers > 0 else None)
    train_loader = DataLoader(train_set, shuffle=sampler is None, sampler=sampler,
                              collate_fn=train_collate,
                              **{**common, "batch_size": train_batch_size}, **parallel)
    val_loader = DataLoader(val_set, shuffle=False, collate_fn=Batch.from_data_list,
                            **common, **parallel)
    test_loader = (DataLoader(test_set, shuffle=False, collate_fn=Batch.from_data_list,
                              **common)
                   if len(test_set) else None)
    return train_loader, val_loader, test_loader


def create_model(config: TrainingConfig, metadata: Dict,
                 normalizer: Optional[Normalizer] = None) -> nn.Module:
    """Model of ``config.architecture``, sized from ``metadata``: the dataset's
    ``metadata.json`` with the selected features (``select_metadata``).

    With ``residual_target`` the network is wrapped in ``ResidualTargetModel``;
    its statistics come from ``normalizer`` when training and from the
    checkpoint's buffers when a trained model is reloaded.
    """
    dims = dict(
        node_in_dim=metadata["node_feature_dims"],
        mesh_edge_in_dim=metadata["edge_feature_dims"],
        world_edge_in_dim=metadata.get("world_edge_feature_dims", 0),
        output_dim=metadata["target_dim"],
    )
    if metadata.get("global_feature_dims"):   # run parameters as their own input
        dims["global_in_dim"] = metadata["global_feature_dims"]
    if config.architecture == "mgn_t":
        from models.mgn_t import MgnTransformer
        model = MgnTransformer(
            **dims,
            latent_size=config.mgn_t_latent_size,
            token_dim=config.mgn_t_token_dim,
            num_tokens=config.mgn_t_num_tokens,
            num_heads=config.mgn_t_num_heads,
            num_transformer_blocks=config.mgn_t_num_transformer_blocks,
            num_pre_mp_steps=config.mgn_t_num_pre_mp_steps,
            num_post_mp_steps=config.mgn_t_num_post_mp_steps,
            num_mlp_layers=config.num_mlp_layers,
            pe_frequencies=config.mgn_t_pe_frequencies,
            ffn_ratio=config.mgn_t_ffn_ratio,
            tau_init=config.mgn_t_tau_init,
            aggregation=config.aggregation,
            transformer_nodes=config.mgn_t_transformer_nodes,
        )
    else:
        model = MeshGraphNets(
            **dims,
            latent_size=config.latent_size,
            num_mlp_layers=config.num_mlp_layers,
            num_message_passing_steps=config.num_message_passing_steps,
            aggregation=config.aggregation,
        )
    if config.residual_target:
        from models.residual import ResidualTargetModel
        model = ResidualTargetModel(model, metadata["node_feature_names"],
                                    metadata["target_feature_names"])
        if normalizer is not None:
            model.set_stats(normalizer)
    return model


@dataclass
class TrainedModel:
    """A model restored from a checkpoint, ready for inference."""

    model: nn.Module
    config: TrainingConfig
    metadata: Dict
    normalizer: Normalizer
    epoch: int
    device: torch.device
    normalizer_source: str = "checkpoint"   # or the normalizer.json it was read from


def checkpoint_normalizer(checkpoint: Dict, data_dir, source: str = "checkpoint",
                          features: FeatureSelection = ALL_FEATURES, windowed: bool = False
                          ) -> Tuple[Normalizer, str]:
    """Statistics a model was trained with, and where they come from.

    Checkpoints store them, for the selected ``features``; older ones fall back to
    ``<data_dir>/normalizer.json``. If that file differs from the stored statistics
    (dataset exported anew or grown), the stored ones win and a warning is logged; not
    for a run on part of the frames (``windowed``), whose statistics differ anyway.
    """
    path = Path(data_dir) / "normalizer.json"
    if checkpoint.get("normalizer") is None:
        return Normalizer.load(str(path)), str(path)
    normalizer = Normalizer.from_dict(checkpoint["normalizer"], source)
    if not windowed and path.exists() and load_statistics(data_dir, features) != normalizer:
        logger.warning(f"{path} differs from the statistics the model was trained with — "
                       f"using those of {source}")
    return normalizer, "checkpoint"


def load_trained_model(checkpoint_path: str, device: str = "auto",
                       data_dir: Optional[str] = None) -> TrainedModel:
    """Restore model, config, metadata and normalizer of a ``Trainer`` checkpoint
    (eval mode).

    The normalizer comes from the checkpoint; checkpoints without one read it
    from ``data_dir`` (default: the training data_dir), which must then be the
    dataset the model was trained on.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TrainingConfig.from_dict(checkpoint["config"])
    metadata = checkpoint["metadata"]
    normalizer, source = checkpoint_normalizer(
        checkpoint, data_dir or config.data_dir, str(checkpoint_path), config.features,
        windowed=config.start_frame is not None or config.end_frame is not None)

    model = create_model(config, metadata)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return TrainedModel(model, config, metadata, normalizer, checkpoint["epoch"], device,
                        source)


def describe_config_source(config: TrainingConfig) -> str:
    """TOML file of the settings plus the command-line options that overrode it."""
    source = config.config_file or "unknown (checkpoint without config_file)"
    if config.config_overrides:
        source += " with " + ", ".join(f"{key}={value}"
                                       for key, value in config.config_overrides.items())
    return source


def describe_machine(device: torch.device) -> str:
    """The GPU of a run with its memory, or the CPU, and the RAM of the machine (Linux)."""
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        machine = f"{properties.name}, {properties.total_memory / 2 ** 30:.1f} GiB"
    else:
        cpu = re.search(r"^model name\s*:\s*(.+)$", _read("/proc/cpuinfo"), re.M)
        machine = f"{cpu.group(1).strip() if cpu else 'CPU'}, {os.cpu_count()} threads"
    ram = re.search(r"^MemTotal:\s+(\d+) kB", _read("/proc/meminfo"), re.M)
    return machine + (f"; RAM {int(ram.group(1)) / 2 ** 20:.1f} GiB" if ram else "")


def peak_memory(device: torch.device) -> Dict[str, float]:
    """Peak memory of the run so far in GiB: resident memory (RSS) of the main process
    and of the largest child process (the DataLoader workers; pages of the dataset in
    shared memory count in every process that reads them), on a GPU the memory
    allocated by tensors and reserved by PyTorch's allocator (nvidia-smi shows the
    reserved memory plus the CUDA context). No read from the GPU."""
    gib = 2 ** 30
    peak = {"ram_main": _peak_rss() / gib}
    workers = []
    for stat in Path("/proc").glob("[0-9]*/stat"):   # Linux; elsewhere no worker figure
        try:   # parent pid: the 2nd field after the command name in parentheses
            if int(stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()[1]) == os.getpid():
                status = (stat.parent / "status").read_text(encoding="utf-8")
                hwm = re.search(r"^VmHWM:\s+(\d+) kB", status, re.M)
                workers.append(int(hwm.group(1)) * 1024 / gib)
        except (OSError, IndexError, ValueError, AttributeError):
            continue   # ended meanwhile, or not readable
    if workers:
        peak["ram_worker"] = max(workers)
    if device.type == "cuda":
        peak["gpu_allocated"] = torch.cuda.max_memory_allocated(device) / gib
        peak["gpu_reserved"] = torch.cuda.max_memory_reserved(device) / gib
    return {key: round(value, 2) for key, value in peak.items()}


def _peak_rss() -> int:
    """Peak resident memory of this process in bytes; on Windows the peak working set
    (the module ``resource`` exists on Unix only)."""
    if sys.platform != "win32":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024   # KiB on Linux
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):   # PROCESS_MEMORY_COUNTERS
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage")]

    kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                           wintypes.DWORD]
    counters = Counters(cb=ctypes.sizeof(Counters))
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                                      counters.cb):
        return 0
    return counters.PeakWorkingSetSize


def format_memory(peak: Dict[str, float]) -> str:
    text = f"RAM {peak['ram_main']:.2f} GiB"
    if "ram_worker" in peak:
        text += f" (largest worker {peak['ram_worker']:.2f} GiB)"
    if "gpu_reserved" in peak:
        text += (f", GPU {peak['gpu_allocated']:.2f} GiB allocated, "
                 f"{peak['gpu_reserved']:.2f} GiB reserved")
    return text


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def format_duration(seconds: float) -> str:
    """12.3 -> '12s', 754 -> '12m 34s', 4000 -> '1h 06m 40s'."""
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def gap_ratio(val_loss: float, train_loss_clean: float) -> float:
    """Generalization gap as ``max(1, val / train)``: 1 = no gap. Recorded, but no
    selection criterion (see train/early_stopping.py).

    ``train_loss_clean`` is the single-step loss on train samples without
    noise, measured like the val loss, so both are directly comparable.
    """
    if train_loss_clean <= 0:
        return math.inf if val_loss > 0 else 1.0
    return max(1.0, val_loss / train_loss_clean)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Runs training, validation, checkpointing and the final test evaluation."""

    def __init__(self, model: nn.Module, config: TrainingConfig, normalizer: Normalizer,
                 metadata: Dict, run_name: Optional[str] = None):
        self.config = config
        self.normalizer = normalizer
        self.metadata = metadata
        self.device = torch.device(config.device)
        self.model = model.to(self.device)

        # Feature layout of the dataset (the export and the feature selection of the config).
        self.node_names: List[str] = metadata["node_feature_names"]
        self.target_names: List[str] = metadata["target_feature_names"]
        self.prev_to_target = prev_to_target_columns(self.node_names, self.target_names)
        self.velocity_prev_columns = ([self.node_names.index(n) for n in PREV_VELOCITY_NAMES]
                                      if PREV_VELOCITY_NAMES[0] in self.node_names else None)
        self.stress_prev_column = (self.node_names.index(PREV_STRESS_NAMES[0])
                                   if PREV_STRESS_NAMES[0] in self.node_names else None)
        self.contact_radius = float(metadata["contact_radius"])
        self._check_history_options()

        self.loss_fn = LOSS_FUNCTIONS[config.loss_type](reduction="none")
        # Additional loss terms (name -> weight) and their scale: the velocity change per
        # step of the targets (the delta statistics; without them the target statistics).
        self.num_bodies = len(metadata.get("rigid_bodies", []))
        self.extra_loss_weights = {name: weight for name, weight in (
            ("edge_stretch", config.loss_edge_stretch_weight),
            ("rigid_body", config.loss_rigid_body_weight if self.num_bodies else 0.0))
            if weight > 0}
        if config.loss_rigid_body_weight > 0 and not self.num_bodies:
            logger.warning("loss rigid_body_weight > 0 but the dataset has no free rigid "
                           "bodies — the term is left out")
        self.velocity_change_std = None
        if self.extra_loss_weights:
            group = "target_delta" if "target_delta" in normalizer else "target"
            self.velocity_change_std = normalizer.mean_std(group, self.device)[1][
                VELOCITY_TARGET_COLUMNS]
        self.last_extra_losses: Dict[str, float] = {}
        self.last_rollout_loss: Optional[float] = None   # rollout training: all steps
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate,
                                          weight_decay=config.weight_decay)
        self.scheduler = self._create_scheduler()
        # Model selection and early stopping by criteria; those of the rollout need
        # rollout validation and are left out without it.
        criteria = [name for name in config.early_stopping_criteria
                    if config.rollout_validation or name not in ROLLOUT_CRITERIA]
        if not criteria:
            raise ValueError(f"criteria {config.early_stopping_criteria} need rollout "
                             "validation — enable it or add val_loss")
        self.selection = EarlyStopping(criteria, patience=config.early_stopping_patience,
                                       min_epochs=config.early_stopping_min_epochs,
                                       stop=config.early_stopping)

        self.rollout_runner = self._new_rollout_runner() if config.rollout_validation else None
        self._last_rollout_metrics: Optional[Dict[str, float]] = None

        self.history: List[Dict] = []
        self.best_epoch: Optional[int] = None          # 1-based, of this run
        self._best_model_state: Optional[Dict] = None

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Weights of the evaluations the selection may still return to.
        self.candidates_dir = self.checkpoint_dir / "candidates"
        self.run_name = run_name or new_run_name()

    def _new_rollout_runner(self):
        from predict.rollout import RolloutRunner
        return RolloutRunner(self.model, self.normalizer,
                             Path(self.config.data_dir) / "dataset.db", self.device,
                             lazy=self.config.lazy_loading,
                             world_edges=self.config.world_edges,
                             world_edge_exclude_distance=self.config.world_edge_exclude_distance,
                             world_edge_exclude_hops=self.config.world_edge_exclude_hops,
                             static_cache_mb=self.config.static_cache_mb,
                             features=self.config.features)

    def _check_history_options(self) -> None:
        """Options that need the history layout (``*_prev`` inputs)."""
        for option in ("scheduled_sampling", "rollout_training"):
            if getattr(self.config, option) and not self.velocity_prev_columns:
                raise ValueError(f"{option} needs *_prev inputs — export the "
                                 "dataset with --history 1")
        if self.config.stress_noise_std > 0 and self.stress_prev_column is None:
            logger.warning("stress_noise_std > 0 but the dataset has no stress_prev "
                           "column — stress noise is disabled")

    def _create_scheduler(self):
        cfg = self.config
        if cfg.scheduler == "exponential":
            # MeshGraphNets: learning_rate -> min_learning_rate over lr_decay_steps
            # optimizer steps, constant afterwards. Stepped per batch.
            end_factor = min(cfg.min_learning_rate, cfg.learning_rate) / cfg.learning_rate
            decay_steps = max(1, int(cfg.lr_decay_steps))
            return LambdaLR(self.optimizer,
                            lambda step: max(end_factor ** (step / decay_steps), end_factor))
        if cfg.scheduler == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode="min", factor=cfg.scheduler_factor,
                                     patience=cfg.scheduler_patience,
                                     min_lr=cfg.min_learning_rate)
        if cfg.scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=cfg.epochs,
                                     eta_min=cfg.min_learning_rate)
        return None

    # ── batch augmentation ──────────────────────────────────────────────────

    def teacher_forcing_ratio(self, epoch: int) -> float:
        """Share of nodes that keep their ground-truth ``*_prev`` inputs.

        Falls linearly from the start to the end ratio over the warmup epochs.
        """
        cfg = self.config
        if not cfg.scheduled_sampling:
            return 1.0
        if epoch < cfg.scheduled_sampling_warmup_epochs:
            progress = epoch / max(cfg.scheduled_sampling_warmup_epochs, 1)
            return (cfg.scheduled_sampling_start_ratio
                    - (cfg.scheduled_sampling_start_ratio
                       - cfg.scheduled_sampling_end_ratio) * progress)
        return cfg.scheduled_sampling_end_ratio

    def _apply_scheduled_sampling(self, batch, prev_batches, epoch: int):
        """Replace ``*_prev`` inputs by the model's prediction from the predecessors.

        This hands the model the state a rollout would produce. With one predecessor the
        prediction replaces the inputs of a share of the nodes. With several
        (``scheduled_sampling_steps``) the model rolls out from the oldest one: every
        predecessor gets the prediction of the one before it on all its nodes, and a
        share of the samples takes the rolled-out state on all nodes, so that its drift
        stays coherent as in a rollout (mixed per node, the drift of several steps would
        tear the mesh between neighbours as no rollout does); a batch in which no sample
        does skips the rollout. Velocities of kinematic nodes stay ground truth, and the
        actuator keeps all its inputs.
        """
        ratio = self.teacher_forcing_ratio(epoch)
        if ratio >= 1.0:
            return batch
        chosen = None
        if len(prev_batches) > 1:
            # Whole samples. Drawn on the CPU, so that deciding needs no wait for the GPU;
            # if no sample takes the rolled-out state, the rollout is not needed.
            chosen = torch.rand(batch.num_graphs) > ratio
            if not chosen.any():
                return batch

        was_training = self.model.training
        self.model.eval()
        prediction = offset = None
        with torch.no_grad():
            for prev_batch in reversed(prev_batches):   # the oldest first
                if prediction is not None:   # moving the nodes rebuilds all edges
                    rows = prev_batch.has_predecessor & prev_batch.loss_mask
                    offset = self._feed_prediction(prev_batch, prediction, rows, offset)
                prev_batch = self._with_world_edges(prev_batch)   # only where still missing
                prediction = self.normalizer.denormalize(self.model(prev_batch), "target")
        self.model.train(was_training)

        if chosen is not None:   # whole samples, so that their drift stays coherent
            graph = getattr(batch, "batch", None)
            graph = torch.zeros_like(batch.loss_mask, dtype=torch.long) if graph is None else graph
            use_prediction = chosen.to(batch.x.device)[graph]
        else:                    # a share of the nodes
            use_prediction = torch.rand(batch.x.shape[0], device=batch.x.device) > ratio
        use_prediction &= batch.has_predecessor & batch.loss_mask
        self._feed_prediction(batch, prediction, use_prediction, offset)
        return batch

    def _feed_prediction(self, batch, prediction: torch.Tensor, rows: torch.Tensor,
                         offset: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Write ``prediction`` (raw targets of the step before) into the ``*_prev``
        inputs of the nodes ``rows`` and move their positions consistently
        (``_write_prediction``). Returns the displacement applied here (None without
        velocity inputs)."""
        delta = self._write_prediction(batch, prediction, rows, offset)
        if delta is not None:
            displace_nodes(batch, delta, self.normalizer, self.contact_radius)
        return delta

    def _write_prediction(self, batch, prediction: torch.Tensor, rows: torch.Tensor,
                          offset: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Write ``prediction`` into the ``*_prev`` inputs of the nodes ``rows`` and return
        the displacement of their current positions that goes with it, not yet applied:
        the change of ``velocity_prev`` plus ``offset``, the displacement already applied
        to the step before (x_t = x_{t-1} + velocity_prev); None without velocity inputs.
        Masks select instead of indexing: no wait for the GPU, and the gradients flow
        (rollout training)."""
        x_before = batch.x
        node_mean, node_std = self.normalizer.mean_std("node_features", x_before.device)
        free = free_node_mask(x_before)
        velocity_columns = self.velocity_prev_columns or []

        x = x_before.clone()
        given = getattr(batch, "target_given", None)
        for column, target_column in self.prev_to_target.items():
            selected = rows & free if column in velocity_columns else rows
            if given is not None:
                selected = selected & given[:, target_column]
            value = (prediction[:, target_column] - node_mean[column]) / node_std[column]
            x[:, column] = torch.where(selected, value, x[:, column])
        batch.x = x

        if not velocity_columns:
            return None
        delta = (x[:, velocity_columns] - x_before[:, velocity_columns]) \
            * node_std[velocity_columns]
        if offset is not None:
            delta = delta + offset * (rows & free)[:, None]
        return delta

    def _apply_position_noise(self, batch):
        """Gaussian noise on the current positions of the free nodes, resampled
        every batch; targets, ``velocity_prev`` and edges follow (App. A.2.2)."""
        free = free_node_mask(batch.x)
        delta = torch.randn(batch.x.shape[0], 3, device=batch.x.device) \
            * self.config.world_noise_std
        delta = delta * free[:, None]
        displace_nodes(batch, delta, self.normalizer, self.contact_radius,
                       self.velocity_prev_columns)
        return batch

    def _window_start_noise(self, batch, nodes: torch.Tensor) -> torch.Tensor:
        """Rollout training: position noise on the free ``nodes`` (those of the samples a
        window starts with), written into ``velocity_prev`` as ``displace_nodes`` does;
        returns the displacement, which the caller applies with that of the fed nodes in
        one contact search."""
        noise = torch.randn(batch.x.shape[0], 3, device=batch.x.device) \
            * self.config.world_noise_std * (free_node_mask(batch.x) & nodes)[:, None]
        _, node_std = self.normalizer.mean_std("node_features", batch.x.device)
        columns = self.velocity_prev_columns
        x = batch.x.clone()
        x[:, columns] += noise / node_std[columns]
        batch.x = x
        return noise

    def _apply_stress_noise(self, batch, nodes: Optional[torch.Tensor] = None):
        """Gaussian noise (Pa) on ``stress_prev`` of the loss nodes (and of ``nodes``);
        the target stays, so the model learns to correct a drifting previous value."""
        column = self.stress_prev_column
        column_std = float(self.normalizer.stats["node_features"]["std"][column])
        noise = torch.randn(batch.x.shape[0], device=batch.x.device) \
            * (self.config.stress_noise_std / column_std)
        given = getattr(batch, "target_given", None)
        if given is not None:
            noise = noise * given[:, self.target_names.index("stress")]
        if nodes is not None:
            noise = noise * nodes
        x = batch.x.clone()
        x[:, column] = x[:, column] + noise * batch.loss_mask
        batch.x = x
        return batch

    # ── epochs ──────────────────────────────────────────────────────────────

    def _loss_sum(self, batch, prediction: torch.Tensor,
                  nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Loss summed over the targets of the nodes in ``loss_mask`` (and in ``nodes``),
        without targets a node does not have (``target_given``); the masks multiply
        instead of indexing, which would wait for the GPU."""
        mask = batch.loss_mask if nodes is None else batch.loss_mask & nodes
        terms = self.loss_fn(prediction, batch.y) * mask[:, None]
        given = getattr(batch, "target_given", None)
        return (terms if given is None else terms * given).sum()

    def _num_loss_terms(self, batch, nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask = batch.loss_mask if nodes is None else batch.loss_mask & nodes
        given = getattr(batch, "target_given", None)
        if given is None:
            return mask.sum() * batch.y.shape[-1]
        return (mask[:, None] & given).sum()

    def _extra_losses(self, batch, prediction: torch.Tensor,
                      nodes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """The additional loss terms in use (``extra_loss_weights``), each a mean, in
        units of the velocity change per step of the targets:

        - ``edge_stretch``: length change of every mesh edge between two nodes of
          ``loss_mask`` in this step, prediction against ground truth. Rotations of the
          edge (folding) leave it unchanged, stretching does not.
        - ``rigid_body``: mean velocity of every free rigid body (the translation the
          rollout gives it), prediction against ground truth, per component.

        The target means cancel in both differences, so the normalized values times the
        target std suffice. Masks multiply instead of indexing (no wait for the GPU).
        """
        terms: Dict[str, torch.Tensor] = {}
        if not self.extra_loss_weights:
            return terms
        loss_mask = batch.loss_mask if nodes is None else batch.loss_mask & nodes
        columns = VELOCITY_TARGET_COLUMNS
        _, target_std = self.normalizer.mean_std("target", prediction.device)
        std = target_std[columns]
        if "edge_stretch" in self.extra_loss_weights:
            i, j = batch.edge_index
            keep = ((i < j) & loss_mask[i] & loss_mask[j]).to(prediction.dtype)
            edge = batch.world_pos[j] - batch.world_pos[i]
            velocity, true_velocity = prediction[:, columns] * std, batch.y[:, columns] * std
            predicted = edge + velocity[j] - velocity[i]
            true = edge + true_velocity[j] - true_velocity[i]
            error = (predicted.norm(dim=1) - true.norm(dim=1)) / self.velocity_change_std.mean()
            terms["edge_stretch"] = (error.pow(2) * keep).sum() / keep.sum().clamp(min=1)
        if "rigid_body" in self.extra_loss_weights and hasattr(batch, "rigid_body"):
            member = (batch.rigid_body >= 0) & loss_mask
            graph = getattr(batch, "batch", None)
            graph = torch.zeros_like(batch.rigid_body) if graph is None else graph
            index = torch.where(member, batch.rigid_body + graph * self.num_bodies, 0)
            groups = int(batch.num_graphs) * self.num_bodies
            weight = member.to(prediction.dtype)
            error = (prediction[:, columns] - batch.y[:, columns]) * std * weight[:, None]
            sums = torch.zeros((groups, len(columns)), dtype=prediction.dtype,
                               device=prediction.device).index_add_(0, index, error)
            counts = torch.zeros(groups, dtype=prediction.dtype,
                                 device=prediction.device).index_add_(0, index, weight)
            mean_error = sums / counts.clamp(min=1)[:, None] / self.velocity_change_std
            present = (counts > 0).to(prediction.dtype)
            terms["rigid_body"] = ((mean_error.pow(2).sum(1) * present).sum()
                                   / (present.sum() * len(columns)).clamp(min=1))
        return terms

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """One pass over the train split; returns the mean loss per graph (with rollout
        training: of the steps from the ground truth, per window).

        ``accumulation_steps`` micro-batches of the loader make one optimizer
        step. Their loss is normalized by the nodes of all of them, so the
        gradient is that of one batch of ``batch_size`` (gradient accumulation).
        The loss stays on the device; it is read once per epoch.
        """
        self.model.train()
        total_loss = torch.zeros((), device=self.device)
        rollout_total = torch.zeros((), device=self.device)
        extra_totals = {name: torch.zeros((), device=self.device)
                        for name in self.extra_loss_weights}
        num_graphs = 0
        steps = self.config.accumulation_steps
        micro_batches = iter(loader)
        progress = tqdm(total=math.ceil(len(loader) / steps), desc=f"Epoch {epoch + 1}",
                        leave=False)
        while group := list(itertools.islice(micro_batches, steps)):
            self.optimizer.zero_grad()
            if self.config.rollout_training:
                step_loss, step_extra, graphs, step_rollout = self._rollout_training_step(group)
                rollout_total += step_rollout * graphs
            else:
                step_loss, step_extra, graphs = self._one_step_training_step(group, epoch)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=self.config.max_grad_norm)
            self.optimizer.step()
            if self.config.scheduler == "exponential":
                self.scheduler.step()

            total_loss += step_loss * graphs
            for name, value in step_extra.items():
                extra_totals[name] += value * graphs
            num_graphs += graphs
            progress.update()
            if progress.n % PROGRESS_EVERY == 0:   # reading the loss waits for the GPU
                progress.set_postfix(loss=f"{step_loss.item():.4f}")
        progress.close()
        # The additional terms unweighted, per graph; the returned loss is that of the
        # targets alone, comparable to the val loss.
        self.last_extra_losses = {name: (value / max(num_graphs, 1)).item()
                                  for name, value in extra_totals.items()}
        self.last_rollout_loss = ((rollout_total / max(num_graphs, 1)).item()
                                  if self.config.rollout_training else None)
        return (total_loss / max(num_graphs, 1)).item()

    def _one_step_training_step(self, group: List, epoch: int
                                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], int]:
        """Forward and backward passes over the micro-batches of one optimizer step:
        the loss, the additional terms and the number of graphs."""
        batches = [self._train_batch(item, epoch) for item in group]
        count = sum(self._num_loss_terms(batch) for batch in batches)
        step_loss = torch.zeros((), device=self.device)
        step_extra = {name: torch.zeros((), device=self.device)
                      for name in self.extra_loss_weights}
        for batch in batches:
            prediction = self.model(batch)
            node_loss = self._loss_sum(batch, prediction) / count
            loss = node_loss
            for name, value in self._extra_losses(batch, prediction).items():
                loss = loss + self.extra_loss_weights[name] * value / len(batches)
                step_extra[name] += value.detach() / len(batches)
            loss.backward()
            step_loss += node_loss.detach()
        return step_loss, step_extra, sum(batch.num_graphs for batch in batches)

    def _rollout_training_step(self, group: List) -> Tuple[torch.Tensor,
                                                           Dict[str, torch.Tensor], int,
                                                           torch.Tensor]:
        """Forward and backward passes over the windows of one optimizer step (rollout
        training): the loss of the steps from the ground truth (comparable to the val
        loss), the additional terms, the number of windows and the loss over all steps of
        the windows, the one optimized. Every term counts once, over all windows."""
        windows = [self._window(item) for item in group]
        count = sum(self._num_loss_terms(step, nodes)
                    for chain, valid in windows for step, nodes in zip(chain, valid))
        first_sum = torch.zeros((), device=self.device)
        first_terms = torch.zeros((), device=self.device)
        step_rollout = torch.zeros((), device=self.device)
        step_extra = {name: torch.zeros((), device=self.device)
                      for name in self.extra_loss_weights}
        for chain, valid in windows:
            loss_sum, first, terms, extra = self._window_loss(chain, valid)
            loss = loss_sum / count
            for name, value in extra.items():
                loss = loss + self.extra_loss_weights[name] * value / len(windows)
                step_extra[name] += value.detach() / len(windows)
            loss.backward()
            step_rollout += loss_sum.detach() / count
            first_sum += first
            first_terms += terms
        windows_count = sum(chain[-1].num_graphs for chain, _ in windows)
        return first_sum / first_terms.clamp(min=1), step_extra, windows_count, step_rollout

    def _window(self, item) -> Tuple[List, List[Optional[torch.Tensor]]]:
        """The batches of the windows of a micro-batch on the device, oldest first, and the
        nodes of each that count in the loss (None: all): a window shorter than the chain
        is padded with repeated samples, flagged by ``has_predecessor`` of the step after.
        Depths without a sample of any window are left out, decided on the CPU before the
        transfer."""
        batch, prev_batches = item
        chain = [batch]
        for prev_batch in prev_batches:
            if not bool(chain[-1].has_predecessor.any()):
                break
            chain.append(prev_batch)
        chain = [step.to(self.device) for step in reversed(chain)]
        return chain, [newer.has_predecessor for newer in chain[1:]] + [None]

    def _window_loss(self, chain: List, valid: List[Optional[torch.Tensor]]
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                Dict[str, torch.Tensor]]:
        """Roll the model out over a window (``_window``). The first step of every window
        starts from the ground truth, with the noise; every further step from the
        prediction of the step before, written into its ``*_prev`` inputs with the
        positions and all edges moved along, as in the rollout, without cutting the
        gradient. Returns the loss summed over all steps, the loss and the number of terms
        of the steps from the ground truth, and the additional terms, means over the
        steps."""
        cfg = self.config
        loss_sum = torch.zeros((), device=self.device)
        first_sum = torch.zeros((), device=self.device)
        first_terms = torch.zeros((), device=self.device)
        extra = {name: torch.zeros((), device=self.device) for name in self.extra_loss_weights}
        prediction = offset = None
        for k, (step, nodes) in enumerate(zip(chain, valid)):
            # The samples whose window starts at this step: inputs from the ground truth.
            starts = ~step.has_predecessor if nodes is None else nodes & ~step.has_predecessor
            delta = None
            if prediction is not None:
                rows = step.has_predecessor & step.loss_mask
                delta = self._write_prediction(
                    step, self.normalizer.denormalize(prediction, "target"), rows, offset)
            if cfg.world_noise_std > 0:
                noise = self._window_start_noise(step, starts)
                delta = noise if delta is None else delta + noise
            if delta is not None:   # one contact search for fed and noisy nodes
                displace_nodes(step, delta, self.normalizer, self.contact_radius)
            else:
                self._with_world_edges(step)
            if cfg.stress_noise_std > 0 and self.stress_prev_column is not None:
                self._apply_stress_noise(step, starts)
            offset = delta
            if k < len(chain) - 1:   # activations recomputed in the backward pass
                prediction = checkpoint(self.model, step, use_reentrant=False)
            else:
                prediction = self.model(step)
            loss_sum = loss_sum + self._loss_sum(step, prediction, nodes)
            first_sum = first_sum + self._loss_sum(step, prediction.detach(), starts)
            first_terms = first_terms + self._num_loss_terms(step, starts)
            for name, value in self._extra_losses(step, prediction, nodes).items():
                extra[name] = extra[name] + value / len(chain)
        return loss_sum, first_sum, first_terms, extra

    def _train_batch(self, item, epoch: int):
        """A micro-batch of the train loader on the device, with scheduled sampling,
        noise and world edges."""
        prev_batches = None
        if isinstance(item, (tuple, list)):          # (sample, [predecessors])
            item, prev_batches = item
            prev_batches = [prev_batch.to(self.device) for prev_batch in prev_batches]
        batch = item.to(self.device)
        if prev_batches and epoch > 0:
            batch = self._apply_scheduled_sampling(batch, prev_batches, epoch)
        if self.config.world_noise_std > 0:   # also builds the world edges
            batch = self._apply_position_noise(batch)
        else:
            batch = self._with_world_edges(batch)
        if self.config.stress_noise_std > 0 and self.stress_prev_column is not None:
            batch = self._apply_stress_noise(batch)
        return batch

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        """Mean single-step loss per graph (no noise, no scheduled sampling)."""
        self.model.eval()
        total_loss = torch.zeros((), device=self.device)
        num_graphs = 0
        for batch in self._prepared(loader):
            mean = self._loss_sum(batch, self.model(batch)) / self._num_loss_terms(batch)
            total_loss += mean * batch.num_graphs
            num_graphs += batch.num_graphs
        return (total_loss / max(num_graphs, 1)).item()

    def _with_world_edges(self, batch):
        """``batch`` with world edges; samples built without them
        (``build_world_edges=False``) get them here, on the device."""
        if getattr(batch, "world_edge_index", None) is None:
            add_world_edges(batch, self.normalizer, self.contact_radius)
        return batch

    def _prepared(self, loader: DataLoader):
        """The batches of ``loader`` on the device, with world edges."""
        for batch in loader:
            yield self._with_world_edges(batch.to(self.device))

    def validate_rollout(self) -> Dict[str, float]:
        """Autoregressive rollout over the val split with the current weights."""
        was_training = self.model.training
        try:
            summary = self.rollout_runner.run(
                split="val", max_sequences=self.config.rollout_validation_max_sequences,
                export_vtu=False, verbose=False, start_frame=self.config.start_frame,
                end_frame=self.config.end_frame)
        finally:
            self.model.train(was_training)
        position_rmse = summary["position_rmse_per_step"]
        metrics = {
            "rollout_mse": summary["mean_mse_normalized"],
            "rollout_pos_rmse": float(np.mean(position_rmse)),
            "rollout_pos_rmse_final": float(position_rmse[-1]),
        }
        # Recorded, no selection criteria: stretching of the predicted mesh, and the end
        # state of the free rigid bodies.
        if summary.get("edge_length_rmse_per_step"):
            metrics["rollout_edge_rmse"] = float(np.mean(summary["edge_length_rmse_per_step"]))
        for name, key in (("rigid_body_position_rmse", "rollout_body_pos_rmse_final"),
                          ("rigid_body_velocity_rmse", "rollout_body_vel_rmse_final")):
            if summary.get(f"{name}_per_step"):
                metrics[key] = float(summary[f"{name}_per_step"][-1])
        return metrics

    def _rollout_metrics(self, epoch: int) -> Optional[Dict[str, float]]:
        """Val rollout metrics if a rollout validation is due after ``epoch``:
        every ``rollout_validation_every`` epochs, in the first and in the last
        epoch. None in between and without rollout validation."""
        if self.rollout_runner is None:
            return None
        every = max(1, int(self.config.rollout_validation_every))
        if ((epoch + 1) % every == 0 or epoch + 1 == self.config.epochs
                or self._last_rollout_metrics is None):
            self._last_rollout_metrics = self.validate_rollout()
            return self._last_rollout_metrics
        return None

    def _train_eval_loader(self, train_loader: DataLoader, val_loader: DataLoader
                           ) -> DataLoader:
        """Fixed train samples (``train_eval_indices``) for the single-step loss
        without noise (comparable to the val loss). Loaded in the main process:
        starting workers anew at every evaluation takes far longer than loading a
        few hundred samples."""
        train_set = train_loader.dataset
        if isinstance(train_set, PredecessorPairDataset):
            train_set = train_set.base
        indices = train_eval_indices(len(train_set))
        return DataLoader(Subset(train_set, indices), batch_size=val_loader.batch_size,
                          shuffle=False, collate_fn=Batch.from_data_list,
                          pin_memory=val_loader.pin_memory)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              start_epoch: int = 0) -> None:
        """Main loop: train, validate, select, checkpoint, early-stop."""
        cfg = self.config
        machine = describe_machine(self.device)
        logger.info(f"Run {self.run_name} on {self.device} ({machine}), "
                    f"output: {self.checkpoint_dir}")
        logger.info(f"Config: {describe_config_source(cfg)}")
        if cfg.architecture == "meshgraphnets":
            architecture = (f"latent {cfg.latent_size}, {cfg.num_message_passing_steps} "
                            "message passing steps")
        elif cfg.architecture == "mgn_t":
            architecture = (f"mgn_t: {cfg.mgn_t_num_pre_mp_steps}+{cfg.mgn_t_num_post_mp_steps} "
                            f"message passing steps, {cfg.mgn_t_num_transformer_blocks} "
                            f"transformer blocks over {cfg.mgn_t_transformer_nodes} nodes, "
                            f"{cfg.mgn_t_num_tokens} tokens, "
                            f"widths {cfg.mgn_t_latent_size}-{cfg.mgn_t_token_dim}")
        else:
            architecture = cfg.architecture
        logger.info(f"Model: {count_parameters(self.model):,} parameters "
                    f"({architecture}, residual_target={cfg.residual_target})")
        self._log_training_options()
        config_path = self.checkpoint_dir / f"{self.run_name}_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        history_path = self.checkpoint_dir / "history.json"

        # Single-step loss without noise on fixed train samples at every
        # evaluation: with the val loss it gives the generalization gap (recorded,
        # no selection criterion).
        train_eval_loader = self._train_eval_loader(train_loader, val_loader)
        if self.rollout_runner is not None:   # no second copy of the val split
            self.rollout_runner.share_dataset("val", val_loader.dataset)
        for record in self.history:   # resumed: the run goes on
            record.pop("end", None)
        # With one criterion the selection never returns to an earlier evaluation.
        keep_candidates = len(self.selection.criteria) > 1

        start_time = time.time()
        epoch_durations: List[float] = []
        epoch = start_epoch - 1
        for epoch in range(start_epoch, cfg.epochs):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            rollout_metrics = self._rollout_metrics(epoch)
            learning_rate = self.optimizer.param_groups[0]["lr"]
            # An evaluation: every epoch without rollout validation, else every
            # rollout validation. Only evaluations select the model and stop.
            evaluated = self.rollout_runner is None or rollout_metrics is not None
            values = {"val_loss": val_loss, **(rollout_metrics or {})}
            train_loss_clean = self.validate(train_eval_loader) if evaluated else None

            if isinstance(self.scheduler, ReduceLROnPlateau):
                if evaluated:   # follows the first criterion
                    self.scheduler.step(values[self.selection.criteria[0]])
            elif isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()

            is_best, selected = False, None   # selected: 1-based epoch for best_model.pt
            if evaluated:
                if keep_candidates:
                    self._save_candidate(epoch + 1)
                best = self.selection.step(epoch, values) + 1
                is_best = best == epoch + 1
                if best != self.best_epoch:   # new, or back to an earlier evaluation
                    selected = self.best_epoch = best
                    self._best_model_state = (copy.deepcopy(self.model.state_dict())
                                              if is_best else self._load_candidate(best))
                if keep_candidates:
                    for dominated in self.selection.dominated_by_latest():
                        (self.candidates_dir / f"epoch_{dominated + 1:04d}.pt").unlink(
                            missing_ok=True)

            epoch_durations.append(time.time() - epoch_start)
            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "epoch_time_seconds": round(epoch_durations[-1], 2),
                "is_best": is_best,
                "peak_memory": {**peak_memory(self.device), "machine": machine},
            }
            record.update({f"train_{name}_loss": value
                           for name, value in self.last_extra_losses.items()})
            if self.last_rollout_loss is not None:
                record["train_rollout_loss"] = self.last_rollout_loss
            if rollout_metrics is not None:
                record.update({f"val_{key}": value for key, value in rollout_metrics.items()})
            if train_loss_clean is not None:
                record["train_loss_clean"] = train_loss_clean
                record["gap_ratio"] = gap_ratio(val_loss, train_loss_clean)
            self.history.append(record)
            if evaluated:
                self._mark_selection()
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

            if selected is not None:
                self._save_best(selected, current=is_best)
            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                                     epoch + 1)
            if (epoch + 1) % cfg.log_every == 0:
                self._log_epoch(epoch, record, rollout_metrics, epoch_durations, start_time)

            if self.selection.should_stop:
                logger.info(f"Early stopping after epoch {epoch + 1}: "
                            f"{self.selection.stop_reason()}")
                break

        last_epoch = epoch + 1
        if last_epoch > start_epoch:
            # How the run ended; an aborted run has no such record.
            self.history[-1]["end"] = ("early_stopping" if self.selection.should_stop
                                       else "epochs")
            if self.selection.should_stop:
                self.history[-1]["stop_reason"] = self.selection.stop_reason()
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
        if last_epoch > start_epoch and last_epoch % cfg.save_every != 0:
            # Final state, so the run can be resumed.
            self.save_checkpoint(self.checkpoint_dir / f"checkpoint_epoch_{last_epoch}.pt",
                                 last_epoch)
        # Test evaluation and later use start from the selected model.
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
            for line in self.selection.report():
                logger.info(line)
            logger.info(f"Restored the selected model (epoch {self.best_epoch})")
            if self.candidates_dir.exists():
                shutil.rmtree(self.candidates_dir)
        elif (self.checkpoint_dir / "best_model.pt").exists():   # resumed, no new best
            best = torch.load(self.checkpoint_dir / "best_model.pt",
                              map_location=self.device, weights_only=False)
            self.model.load_state_dict(best["model_state_dict"])
            logger.info(f"Restored best model from {self.checkpoint_dir / 'best_model.pt'} "
                        f"(epoch {best['epoch']})")
        logger.info(f"Training time: {format_duration(time.time() - start_time)}; "
                    f"history: {history_path}")
        if self.history and "peak_memory" in self.history[-1]:   # not in older histories
            logger.info(f"Peak memory: {format_memory(self.history[-1]['peak_memory'])}")

    def _log_training_options(self) -> None:
        cfg = self.config
        scheduler = cfg.scheduler + (f" to {cfg.min_learning_rate} over {cfg.lr_decay_steps:,} "
                                     "steps" if cfg.scheduler == "exponential" else "")
        logger.info(f"Optimizer: Adam lr={cfg.learning_rate}, weight_decay={cfg.weight_decay}, "
                    f"scheduler={scheduler}, grad clip {cfg.max_grad_norm}")
        accumulation = (f" ({cfg.accumulation_steps} micro-batches of {cfg.micro_batch_size}"
                        ", gradients accumulated)" if cfg.accumulation_steps > 1 else "")
        logger.info(f"Batch: {cfg.batch_size} graphs per optimizer step{accumulation}")
        stress_noise = cfg.stress_noise_std if self.stress_prev_column is not None else 0.0
        logger.info(f"Noise: position {cfg.world_noise_std} "
                    f"{self.metadata.get('length_unit', SI_LENGTH)}, stress_prev {stress_noise}")
        if self.extra_loss_weights:
            logger.info("Additional loss terms: " + ", ".join(
                f"{name} x {weight}" for name, weight in self.extra_loss_weights.items()))
        if cfg.scheduled_sampling:
            steps = (f", rolled out over {cfg.scheduled_sampling_steps} steps"
                     if cfg.scheduled_sampling_steps > 1 else "")
            logger.info(f"Scheduled sampling: teacher forcing {cfg.scheduled_sampling_start_ratio}"
                        f" -> {cfg.scheduled_sampling_end_ratio} over "
                        f"{cfg.scheduled_sampling_warmup_epochs} epochs{steps}")
        if cfg.rollout_training:
            logger.info(f"Rollout training: windows of up to {cfg.rollout_training_steps} "
                        "steps on the model's own predictions, the loss on every step, the "
                        "gradients through the window (activations recomputed per step)")
        selection = self.selection
        when = (f"every {cfg.rollout_validation_every} epoch(s) with a val rollout"
                if self.rollout_runner else "every epoch")
        logger.info(f"Model selection {when} by {', '.join(selection.criteria)}"
                    + (" — the evaluation closest to the best in all of them"
                       if len(selection.criteria) > 1 else ""))
        left_out = [c for c in cfg.early_stopping_criteria if c not in selection.criteria]
        if left_out:
            logger.info(f"    without rollout validation not used: {', '.join(left_out)}")
        if cfg.early_stopping:
            logger.info(f"Early stopping after at least {selection.min_epochs} epochs: "
                        f"{selection.patience} evaluations without a new low in any "
                        f"criterion")

    def _log_epoch(self, epoch: int, record: Dict, rollout_metrics: Optional[Dict],
                   epoch_durations: List[float], start_time: float) -> None:
        message = (f"Epoch {epoch + 1:4d}/{self.config.epochs} | "
                   f"train {record['train_loss']:.4f} | val {record['val_loss']:.4f}")
        for name in self.last_extra_losses:
            message += f" | {name} {record[f'train_{name}_loss']:.4f}"
        if "train_rollout_loss" in record:
            message += f" | train over the windows {record['train_rollout_loss']:.4f}"
        if record.get("train_loss_clean") is not None:
            message += (f" | train (no noise) {record['train_loss_clean']:.4f}, "
                        f"gap x{record['gap_ratio']:.2f}")
        if rollout_metrics is not None:
            message += (f" | rollout mse {rollout_metrics['rollout_mse']:.4f}, "
                        f"pos rmse {rollout_metrics['rollout_pos_rmse']:.4f} "
                        f"(final {rollout_metrics['rollout_pos_rmse_final']:.4f})")
            if "rollout_edge_rmse" in rollout_metrics:
                message += f", edge rmse {rollout_metrics['rollout_edge_rmse']:.2e}"
            if "rollout_body_pos_rmse_final" in rollout_metrics:
                message += (f", rigid body final {rollout_metrics['rollout_body_pos_rmse_final']:.4f}"
                            f" / {rollout_metrics['rollout_body_vel_rmse_final']:.2e} per step")
        message += f" | lr {record['learning_rate']:.2e}"
        if record["is_best"]:
            message += " | * best"
        logger.info(message)

        remaining = (self.config.epochs - epoch - 1) * float(np.mean(epoch_durations[-10:]))
        finish = now() + timedelta(seconds=remaining)
        timing = (f"    {format_duration(epoch_durations[-1])} per epoch, "
                  f"elapsed {format_duration(time.time() - start_time)}, "
                  f"remaining ~{format_duration(remaining)} "
                  f"(finish ~{finish:%d.%m. %H:%M})")
        timing += f" | {self.selection.status()}"
        logger.info(timing)

    # ── checkpoints ─────────────────────────────────────────────────────────

    def save_checkpoint(self, path: Path, epoch: int,
                        model_state: Optional[Dict] = None) -> None:
        """Save the current state; ``epoch`` = number of completed epochs.

        With ``model_state`` (weights of an earlier evaluation) the optimizer and
        scheduler state are left out: they belong to the current weights.
        """
        current = model_state is None
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict() if current else model_state,
            "optimizer_state_dict": self.optimizer.state_dict() if current else None,
            "scheduler_state_dict": (self.scheduler.state_dict()
                                     if current and self.scheduler else None),
            "best_val_loss": self.selection.best_score,   # selection score (0 = best in all)
            "config": self.config.to_dict(),
            "metadata": self.metadata,
            "normalizer": self.normalizer.to_dict(),   # the scale the weights belong to
            "history": self.history,
        }, path)
        logger.debug(f"Saved {path}")

    def _save_best(self, epoch: int, current: bool) -> None:
        """``best_model.pt`` with the selected weights (``self._best_model_state``)."""
        self.save_checkpoint(self.checkpoint_dir / "best_model.pt", epoch,
                             model_state=None if current else self._best_model_state)

    def _save_candidate(self, epoch: int) -> None:
        """Weights after ``epoch``; the consistent selection may return to them."""
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.candidates_dir / f"epoch_{epoch:04d}.pt")

    def _load_candidate(self, epoch: int) -> Dict:
        return torch.load(self.candidates_dir / f"epoch_{epoch:04d}.pt",
                          map_location=self.device, weights_only=True)

    def _mark_selection(self) -> None:
        """Flag the selected epoch in the history (``selected``) and store the
        current score of every evaluation, which changes as the lowest values of
        the criteria move."""
        scores = {epoch + 1: score for epoch, score
                  in zip(self.selection.epochs, self.selection.scores())}
        for record in self.history:
            record["selected"] = record["epoch"] == self.best_epoch
            if record["epoch"] in scores:
                record["selection_score"] = scores[record["epoch"]]
            else:
                record.pop("selection_score", None)

    def load_checkpoint(self, path: str) -> int:
        """Continue from ``path``; returns the number of completed epochs.

        The selection state is not stored and starts fresh: only the evaluations
        after the resume are candidates, and the first one is selected at first.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:   # best_model.pt of an earlier evaluation
            logger.warning(f"{path} holds no optimizer state — the optimizer starts fresh")
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = list(checkpoint.get("history", []))
        logger.info(f"Resumed from {path} after epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]

    # ── test evaluation ─────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        """Single-step test metrics on the nodes in ``loss_mask``.

        Normalized MSE/MAE over all targets, plus RMSE/MAE per target field in SI
        (velocity in m per selected frame, the stresses in Pa; other result fields in
        the units of the data, ``utils.units``). The errors are summed per batch
        (float64), so memory does not grow with the split.
        """
        self.model.eval()
        sums = torch.zeros(4, len(self.target_names), dtype=torch.float64)
        rows = torch.zeros(len(self.target_names), dtype=torch.float64)   # values per target
        for batch in self._prepared(loader):
            mask = batch.loss_mask
            prediction, target = self.model(batch)[mask], batch.y[mask]
            given = getattr(batch, "target_given", None)
            given = torch.ones_like(target, dtype=torch.bool) if given is None else given[mask]
            raw_error = (self.normalizer.denormalize(prediction, "target")
                         - self.normalizer.denormalize(target, "target")) * given
            error = (prediction - target) * given
            sums += torch.stack((error.pow(2).sum(0), error.abs().sum(0),
                                 raw_error.pow(2).sum(0), raw_error.abs().sum(0))).double().cpu()
            rows += given.sum(0).double().cpu()
        squared, absolute, raw_squared, raw_absolute = sums / rows.clamp_min(1)
        length, stress = si_factors(self.metadata)
        si = [length if name in VELOCITY_TARGET_NAMES else stress if name in STRESS_FIELDS
              else 1.0 for name in self.target_names]
        return {
            "mse_normalized": squared.mean().item(),
            "mae_normalized": absolute.mean().item(),
            "rmse_per_field": {name: math.sqrt(raw_squared[i].item()) * si[i]
                               for i, name in enumerate(self.target_names)},
            "mae_per_field": {name: raw_absolute[i].item() * si[i]
                              for i, name in enumerate(self.target_names)},
        }

    def evaluate_rollout(self, split: str = "test", dataset=None) -> Dict:
        """Full rollout of every trajectory of ``split`` with the rollout metrics
        (RMSE-1, RMSE-all, R-RMSE of position and stress, see predict/rollout.py);
        ``dataset`` (that split, e.g. of the test loader) is used instead of a new one."""
        runner = self.rollout_runner or self._new_rollout_runner()
        if dataset is not None:
            runner.share_dataset(split, dataset)
        was_training = self.model.training
        try:
            summary = runner.run(split=split, export_vtu=False, verbose=False, one_step=True,
                                 start_frame=self.config.start_frame,
                                 end_frame=self.config.end_frame)
        finally:
            self.model.train(was_training)
        position_rmse = summary["position_rmse_per_step"]
        return {
            "rollout_metrics": summary["rollout_metrics"],
            "mean_mse_normalized": summary["mean_mse_normalized"],
            "position_rmse": float(np.mean(position_rmse)) if position_rmse else None,
            "position_rmse_final": position_rmse[-1] if position_rmse else None,
            "rmse_per_field": summary["rmse_per_field"],
            "sequences": {sequence_id: result["rollout_metrics"]
                          for sequence_id, result in summary["sequences"].items()},
        }

    def evaluate_and_save(self, loader: Optional[DataLoader]) -> Optional[Dict]:
        """Evaluate on the test split (single step and full rollout), log and write
        ``test_metrics.json``."""
        if loader is None:
            logger.info("No test split — skipping test evaluation")
            return None
        metrics = self.evaluate(loader)
        logger.info(f"Test (single step): MSE {metrics['mse_normalized']:.4f}, "
                    f"MAE {metrics['mae_normalized']:.4f} (normalized)")
        for name in self.target_names:
            logger.info(f"    {name:12s} RMSE {metrics['rmse_per_field'][name]:.4e}  "
                        f"MAE {metrics['mae_per_field'][name]:.4e}")
        from predict.rollout import format_rollout_metrics
        metrics["rollout"] = self.evaluate_rollout("test", loader.dataset)
        logger.info(f"Test (full rollout, {len(metrics['rollout']['sequences'])} trajectories), "
                    "rollout metrics (± = standard error over trajectories):")
        for line in format_rollout_metrics(metrics["rollout"]["rollout_metrics"],
                                         self.metadata.get("stress_unit", SI_STRESS)):
            logger.info(line)
        path = self.checkpoint_dir / "test_metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Test metrics: {path}")
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def new_run_name() -> str:
    """Name of a training run; prefixes its log and config file."""
    return now().strftime("train_%Y%m%d_%H%M%S")


def setup_logging(log_file: Optional[Path] = None) -> None:
    """Send the ``train`` logger to stdout and, optionally, to ``log_file``."""
    formatter = Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]  # tqdm uses stderr
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logger.handlers.clear()
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
