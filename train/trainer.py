"""
trainer.py — One-step training of the MeshGraphNet on deforming_plate.

Per batch (in this order):

1. **Scheduled sampling** (optional, needs history): the model predicts step t
   from its predecessor sample; on a random subset of nodes that prediction
   replaces the ``*_prev`` inputs, and the current positions move with it.
2. **Position noise** (``world_noise_std``, paper App. A.2.2): Gaussian noise on
   the current positions of the free nodes; velocity target, ``velocity_prev``
   and all edges are adjusted consistently.
3. **Stress noise** (optional, needs history) on ``stress_prev``.
4. Loss on the normalized targets of the nodes in ``loss_mask``, gradient
   clipping, optimizer step; the exponential LR schedule steps per batch.

After every epoch the single-step validation loss and, if enabled, the
autoregressive rollout error on the val split are computed. The latter then
drives model selection, early stopping and the plateau scheduler.
"""

from __future__ import annotations

import copy
import json
import logging
import math
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
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from dataset_preprocessor import (
    DeformingPlateDataset,
    PredecessorPairDataset,
    collate_with_predecessor,
)
from dataset_preprocessor.graph_features import (
    PREV_STRESS_NAMES,
    PREV_VELOCITY_NAMES,
    displace_nodes,
    free_node_mask,
    prev_to_target_columns,
)
from models.meshgraphnet import MeshGraphNet, count_parameters
from train.config import TrainingConfig
from train.early_stopping import EarlyStopping
from utils.normalizer import Normalizer
from utils.timezone import Formatter, now

logger = logging.getLogger("train")

LOSS_FUNCTIONS = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "huber": nn.HuberLoss,
    "smooth_l1": nn.SmoothL1Loss,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data and model
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloaders(config: TrainingConfig, normalizer: Normalizer
                       ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Train/val/test loaders over ``<data_dir>/dataset.db``.

    With scheduled sampling every train item also carries its predecessor
    sample. The test loader is None for an empty test split. With
    ``lazy_loading`` the frames stay in the DB and every worker reads the
    frames of its samples through its own connection.
    """
    db_path = Path(config.data_dir) / "dataset.db"
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found — export it first: "
                                "python -m dataset_preprocessor.export")
    loading = dict(lazy=config.lazy_loading, cache_frames=config.cache_frames)
    train_set = DeformingPlateDataset(db_path, "train", normalizer, **loading)
    val_set = DeformingPlateDataset(db_path, "val", normalizer, **loading)
    test_set = DeformingPlateDataset(db_path, "test", normalizer, **loading)
    mode = (f"lazy, cache {config.cache_frames} frames per worker" if config.lazy_loading
            else "in memory")
    logger.info(f"Data: {db_path} — train {len(train_set)}, val {len(val_set)}, "
                f"test {len(test_set)} samples (history {train_set.history}, {mode}, "
                f"{config.num_dataloader_workers} workers)")

    train_collate = Batch.from_data_list
    if config.scheduled_sampling:
        train_set = PredecessorPairDataset(train_set)
        train_collate = collate_with_predecessor

    workers = config.num_dataloader_workers
    common = dict(batch_size=config.batch_size, pin_memory=config.device == "cuda")
    parallel = dict(num_workers=workers, persistent_workers=workers > 0,
                    multiprocessing_context="spawn" if workers > 0 else None)
    train_loader = DataLoader(train_set, shuffle=True, collate_fn=train_collate,
                              **common, **parallel)
    val_loader = DataLoader(val_set, shuffle=False, collate_fn=Batch.from_data_list,
                            **common, **parallel)
    test_loader = (DataLoader(test_set, shuffle=False, collate_fn=Batch.from_data_list,
                              **common)
                   if len(test_set) else None)
    return train_loader, val_loader, test_loader


def create_model(config: TrainingConfig, metadata: Dict,
                 normalizer: Optional[Normalizer] = None) -> nn.Module:
    """Model of ``config.architecture``, sized from the dataset's ``metadata.json``.

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
        )
    else:
        model = MeshGraphNet(
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


def load_trained_model(checkpoint_path: str, device: str = "auto",
                       data_dir: Optional[str] = None) -> TrainedModel:
    """Restore model, config and metadata of a ``Trainer`` checkpoint (eval mode).

    The normalizer is read from ``data_dir`` (default: the training data_dir),
    which must be the dataset the model was trained on.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = TrainingConfig.from_dict(checkpoint["config"])
    metadata = checkpoint["metadata"]
    normalizer = Normalizer.load(str(Path(data_dir or config.data_dir) / "normalizer.json"))

    model = create_model(config, metadata)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return TrainedModel(model, config, metadata, normalizer, checkpoint["epoch"], device)


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

        # Feature layout of the dataset (fixed at export time).
        self.node_names: List[str] = metadata["node_feature_names"]
        self.target_names: List[str] = metadata["target_feature_names"]
        self.prev_to_target = prev_to_target_columns(self.node_names, self.target_names)
        self.velocity_prev_columns = ([self.node_names.index(n) for n in PREV_VELOCITY_NAMES]
                                      if PREV_VELOCITY_NAMES[0] in self.node_names else None)
        self.stress_prev_column = (self.node_names.index(PREV_STRESS_NAMES[0])
                                   if PREV_STRESS_NAMES[0] in self.node_names else None)
        self.contact_radius = float(metadata["contact_radius"])
        self._check_history_options()

        self.loss_fn = LOSS_FUNCTIONS[config.loss_type]()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate,
                                          weight_decay=config.weight_decay)
        self.scheduler = self._create_scheduler()
        self.early_stopper = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            min_epochs=config.early_stopping_min_epochs,
            gap_weight=config.early_stopping_gap_weight,
            smoothing_window=config.early_stopping_smoothing_window,
            efficiency_window=config.early_stopping_efficiency_window,
            efficiency_threshold=config.early_stopping_efficiency_threshold,
        ) if config.early_stopping else None

        self.rollout_runner = self._new_rollout_runner() if config.rollout_validation else None
        self._last_rollout_metrics: Optional[Dict[str, float]] = None

        self.history: List[Dict] = []
        self.best_selection_loss = float("inf")
        self.best_epoch: Optional[int] = None          # 1-based, of this run
        self._best_model_state: Optional[Dict] = None

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or new_run_name()

    def _new_rollout_runner(self):
        from predict.rollout import RolloutRunner
        return RolloutRunner(self.model, self.normalizer,
                             Path(self.config.data_dir) / "dataset.db", self.device,
                             lazy=self.config.lazy_loading)

    def _check_history_options(self) -> None:
        """Options that need the history layout (``*_prev`` inputs)."""
        if self.config.scheduled_sampling and not self.prev_to_target:
            raise ValueError("scheduled_sampling needs *_prev inputs — export the "
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

    def _apply_scheduled_sampling(self, batch, prev_batch, epoch: int):
        """Replace ``*_prev`` inputs by the model's prediction from the predecessor.

        This hands the model the state a rollout would produce. Velocities of
        kinematic nodes stay ground truth, and the actuator keeps all its
        inputs. Where ``velocity_prev`` changes by d, the current positions move
        by d as well (x_t = x_{t-1} + velocity_prev).
        """
        ratio = self.teacher_forcing_ratio(epoch)
        if ratio >= 1.0:
            return batch

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            prediction = self.normalizer.denormalize(self.model(prev_batch), "target")
        self.model.train(was_training)

        x_before = batch.x
        node_mean, node_std = self.normalizer.mean_std("node_features", x_before.device)
        use_prediction = torch.rand(x_before.shape[0], device=x_before.device) > ratio
        use_prediction &= batch.has_predecessor & batch.loss_mask
        free = free_node_mask(x_before)
        velocity_columns = self.velocity_prev_columns or []

        x = x_before.clone()
        for column, target_column in self.prev_to_target.items():
            rows = use_prediction & free if column in velocity_columns else use_prediction
            x[rows, column] = ((prediction[rows, target_column] - node_mean[column])
                               / node_std[column])
        batch.x = x

        if velocity_columns:
            delta = (x[:, velocity_columns] - x_before[:, velocity_columns]) \
                * node_std[velocity_columns]
            displace_nodes(batch, delta, self.normalizer, self.contact_radius)
        return batch

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

    def _apply_stress_noise(self, batch):
        """Gaussian noise (Pa) on ``stress_prev`` of the loss nodes; the target
        stays, so the model learns to correct a drifting previous value."""
        column = self.stress_prev_column
        column_std = float(self.normalizer.stats["node_features"]["std"][column])
        noise = torch.randn(batch.x.shape[0], device=batch.x.device) \
            * (self.config.stress_noise_std / column_std)
        x = batch.x.clone()
        x[:, column] = x[:, column] + noise * batch.loss_mask
        batch.x = x
        return batch

    # ── epochs ──────────────────────────────────────────────────────────────

    def _loss(self, batch, prediction: torch.Tensor) -> torch.Tensor:
        mask = batch.loss_mask
        return self.loss_fn(prediction[mask], batch.y[mask])

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """One pass over the train split; returns the mean loss per graph."""
        self.model.train()
        total_loss, num_graphs = 0.0, 0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)
        for batch in progress:
            prev_batch = None
            if isinstance(batch, (tuple, list)):          # (sample, predecessor) pairs
                batch, prev_batch = batch
                prev_batch = prev_batch.to(self.device)
            batch = batch.to(self.device)

            if prev_batch is not None and epoch > 0:
                batch = self._apply_scheduled_sampling(batch, prev_batch, epoch)
            if self.config.world_noise_std > 0:
                batch = self._apply_position_noise(batch)
            if self.config.stress_noise_std > 0 and self.stress_prev_column is not None:
                batch = self._apply_stress_noise(batch)

            self.optimizer.zero_grad()
            loss = self._loss(batch, self.model(batch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=self.config.max_grad_norm)
            self.optimizer.step()
            if self.config.scheduler == "exponential":
                self.scheduler.step()

            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
            progress.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / max(num_graphs, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> float:
        """Mean single-step loss per graph (no noise, no scheduled sampling)."""
        self.model.eval()
        total_loss, num_graphs = 0.0, 0
        for batch in loader:
            batch = batch.to(self.device)
            total_loss += self._loss(batch, self.model(batch)).item() * batch.num_graphs
            num_graphs += batch.num_graphs
        return total_loss / max(num_graphs, 1)

    def validate_rollout(self) -> Dict[str, float]:
        """Autoregressive rollout over the val split with the current weights."""
        was_training = self.model.training
        try:
            summary = self.rollout_runner.run(
                split="val", max_sequences=self.config.rollout_validation_max_sequences,
                export_vtu=False, verbose=False)
        finally:
            self.model.train(was_training)
        position_rmse = summary["position_rmse_per_step"]
        return {
            "rollout_mse": summary["mean_mse_normalized"],
            "rollout_pos_rmse": float(np.mean(position_rmse)),
            "rollout_pos_rmse_final": float(position_rmse[-1]),
        }

    def _selection_loss(self, epoch: int, val_loss: float
                        ) -> Tuple[float, Optional[Dict[str, float]]]:
        """Loss used for model selection, plus fresh rollout metrics if computed.

        Without rollout validation this is the single-step val loss. With it,
        the rollout runs every ``rollout_validation_every`` epochs and in the
        last epoch; in between the last rollout result is reused. Which rollout
        metric counts is set by ``rollout_selection_metric``.
        """
        if self.rollout_runner is None:
            return val_loss, None
        every = max(1, int(self.config.rollout_validation_every))
        fresh = None
        if ((epoch + 1) % every == 0 or epoch + 1 == self.config.epochs
                or self._last_rollout_metrics is None):
            fresh = self._last_rollout_metrics = self.validate_rollout()
        metric = f"rollout_{self.config.rollout_selection_metric}"
        return self._last_rollout_metrics[metric], fresh

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              start_epoch: int = 0) -> None:
        """Main loop: train, validate, select, checkpoint, early-stop."""
        cfg = self.config
        logger.info(f"Run {self.run_name} on {self.device}, output: {self.checkpoint_dir}")
        if cfg.architecture == "meshgraphnet":
            architecture = (f"latent {cfg.latent_size}, {cfg.num_message_passing_steps} "
                            "message passing steps")
        elif cfg.architecture == "mgn_t":
            architecture = (f"mgn_t: {cfg.mgn_t_num_pre_mp_steps}+{cfg.mgn_t_num_post_mp_steps} "
                            f"message passing steps, {cfg.mgn_t_num_transformer_blocks} "
                            f"transformer blocks, {cfg.mgn_t_num_tokens} tokens, "
                            f"widths {cfg.mgn_t_latent_size}-{cfg.mgn_t_token_dim}")
        else:
            architecture = cfg.architecture
        logger.info(f"Model: {count_parameters(self.model):,} parameters "
                    f"({architecture}, residual_target={cfg.residual_target})")
        self._log_training_options()
        config_path = self.checkpoint_dir / f"{self.run_name}_config.json"
        with open(config_path, "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        history_path = self.checkpoint_dir / "history.json"

        start_time = time.time()
        epoch_durations: List[float] = []
        epoch = start_epoch - 1
        for epoch in range(start_epoch, cfg.epochs):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)
            selection_loss, rollout_metrics = self._selection_loss(epoch, val_loss)
            learning_rate = self.optimizer.param_groups[0]["lr"]
            # With rollout validation every N epochs the epochs in between reuse
            # the last value; only fresh values may count as (no) improvement.
            fresh = self.rollout_runner is None or rollout_metrics is not None

            if isinstance(self.scheduler, ReduceLROnPlateau):
                if fresh:
                    self.scheduler.step(selection_loss)
            elif isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()

            if not fresh:
                is_best = False
            elif self.early_stopper is not None:
                # With rollout validation the train loss (with noise and scheduled
                # sampling) is not comparable to the rollout error: no gap term.
                reference = selection_loss if self.rollout_runner else train_loss
                is_best = self.early_stopper.step(epoch, selection_loss, reference)
            else:
                is_best = selection_loss < self.best_selection_loss
            if is_best:
                self.best_selection_loss = selection_loss
                self.best_epoch = epoch + 1
                self._best_model_state = copy.deepcopy(self.model.state_dict())

            epoch_durations.append(time.time() - epoch_start)
            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "epoch_time_seconds": round(epoch_durations[-1], 2),
                "is_best": is_best,
            }
            if rollout_metrics is not None:
                record.update({f"val_{key}": value for key, value in rollout_metrics.items()})
            self.history.append(record)
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=2)

            if is_best:
                self.save_checkpoint(self.checkpoint_dir / "best_model.pt", epoch + 1)
            if (epoch + 1) % cfg.save_every == 0:
                self.save_checkpoint(self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                                     epoch + 1)
            if (epoch + 1) % cfg.log_every == 0:
                self._log_epoch(epoch, record, rollout_metrics, epoch_durations, start_time)

            if self.early_stopper is not None and self.early_stopper.should_stop:
                logger.info(f"Early stopping after epoch {epoch + 1}: "
                            f"{self.early_stopper.stop_reason()}")
                break

        last_epoch = epoch + 1
        if last_epoch > start_epoch and last_epoch % cfg.save_every != 0:
            # Final state, so the run can be resumed.
            self.save_checkpoint(self.checkpoint_dir / f"checkpoint_epoch_{last_epoch}.pt",
                                 last_epoch)
        # Test evaluation and later use start from the selected model.
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
            logger.info(f"Restored best model from epoch {self.best_epoch} "
                        f"(selection loss {self.best_selection_loss:.4f})")
        elif (self.checkpoint_dir / "best_model.pt").exists():   # resumed, no new best
            best = torch.load(self.checkpoint_dir / "best_model.pt",
                              map_location=self.device, weights_only=False)
            self.model.load_state_dict(best["model_state_dict"])
            logger.info(f"Restored best model from {self.checkpoint_dir / 'best_model.pt'} "
                        f"(epoch {best['epoch']})")
        logger.info(f"Training time: {format_duration(time.time() - start_time)}; "
                    f"history: {history_path}")

    def _log_training_options(self) -> None:
        cfg = self.config
        scheduler = cfg.scheduler + (f" to {cfg.min_learning_rate} over {cfg.lr_decay_steps:,} "
                                     "steps" if cfg.scheduler == "exponential" else "")
        logger.info(f"Optimizer: Adam lr={cfg.learning_rate}, weight_decay={cfg.weight_decay}, "
                    f"scheduler={scheduler}, grad clip {cfg.max_grad_norm}")
        stress_noise = cfg.stress_noise_std if self.stress_prev_column is not None else 0.0
        logger.info(f"Noise: position {cfg.world_noise_std}, stress_prev {stress_noise}")
        if cfg.scheduled_sampling:
            logger.info(f"Scheduled sampling: teacher forcing {cfg.scheduled_sampling_start_ratio}"
                        f" -> {cfg.scheduled_sampling_end_ratio} over "
                        f"{cfg.scheduled_sampling_warmup_epochs} epochs")
        if self.rollout_runner is not None:
            metric = {"mse": "MSE", "pos_rmse": "position RMSE",
                      "pos_rmse_final": "final position RMSE"}[cfg.rollout_selection_metric]
            logger.info(f"Model selection: val rollout {metric} every "
                        f"{cfg.rollout_validation_every} epoch(s)")
        if self.early_stopper is not None:
            es = self.early_stopper
            logger.info(f"Early stopping: patience {es.patience}, min_epochs {es.min_epochs}, "
                        f"min_delta {es.min_delta}, gap_weight {es.gap_weight}, "
                        f"smoothing {es.smoothing_window}, efficiency window "
                        f"{es.efficiency_window} / threshold {es.efficiency_threshold}")

    def _log_epoch(self, epoch: int, record: Dict, rollout_metrics: Optional[Dict],
                   epoch_durations: List[float], start_time: float) -> None:
        message = (f"Epoch {epoch + 1:4d}/{self.config.epochs} | "
                   f"train {record['train_loss']:.4f} | val {record['val_loss']:.4f}")
        if rollout_metrics is not None:
            message += (f" | rollout mse {rollout_metrics['rollout_mse']:.4f}, "
                        f"pos rmse {rollout_metrics['rollout_pos_rmse']:.4f} "
                        f"(final {rollout_metrics['rollout_pos_rmse_final']:.4f})")
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
        if self.early_stopper is not None:
            timing += f" | early stopping: {self.early_stopper.status()}"
        logger.info(timing)

    # ── checkpoints ─────────────────────────────────────────────────────────

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Save the current state; ``epoch`` = number of completed epochs."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_selection_loss,
            "config": self.config.to_dict(),
            "metadata": self.metadata,
            "history": self.history,
        }, path)
        logger.debug(f"Saved {path}")

    def load_checkpoint(self, path: str) -> int:
        """Continue from ``path``; returns the number of completed epochs.

        The early-stopping state is not stored and starts fresh, so the first
        resumed epoch counts as a new best for the early stopper.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_selection_loss = checkpoint["best_val_loss"]
        self.history = list(checkpoint.get("history", []))
        logger.info(f"Resumed from {path} after epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]

    # ── test evaluation ─────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        """Single-step test metrics on the nodes in ``loss_mask``.

        Normalized MSE/MAE over all targets, plus RMSE/MAE per target field in
        raw units (velocity in m per selected frame, stress in Pa).
        """
        self.model.eval()
        predictions, targets = [], []
        for batch in loader:
            batch = batch.to(self.device)
            mask = batch.loss_mask
            predictions.append(self.model(batch)[mask].cpu())
            targets.append(batch.y[mask].cpu())
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        error = prediction - target
        raw_error = (self.normalizer.denormalize(prediction, "target")
                     - self.normalizer.denormalize(target, "target"))
        return {
            "mse_normalized": error.pow(2).mean().item(),
            "mae_normalized": error.abs().mean().item(),
            "rmse_per_field": {name: math.sqrt(raw_error[:, i].pow(2).mean().item())
                               for i, name in enumerate(self.target_names)},
            "mae_per_field": {name: raw_error[:, i].abs().mean().item()
                              for i, name in enumerate(self.target_names)},
        }

    def evaluate_rollout(self, split: str = "test") -> Dict:
        """Full rollout of every trajectory of ``split`` with the paper metrics
        (RMSE-1, RMSE-all, R-RMSE of position and stress, see predict/rollout.py)."""
        runner = self.rollout_runner or self._new_rollout_runner()
        was_training = self.model.training
        try:
            summary = runner.run(split=split, export_vtu=False, verbose=False, one_step=True)
        finally:
            self.model.train(was_training)
        position_rmse = summary["position_rmse_per_step"]
        return {
            "paper_metrics": summary["paper_metrics"],
            "mean_mse_normalized": summary["mean_mse_normalized"],
            "position_rmse": float(np.mean(position_rmse)) if position_rmse else None,
            "position_rmse_final": position_rmse[-1] if position_rmse else None,
            "rmse_per_field": summary["rmse_per_field"],
            "sequences": {sequence_id: result["paper_metrics"]
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
        from predict.rollout import format_paper_metrics
        metrics["rollout"] = self.evaluate_rollout("test")
        logger.info(f"Test (full rollout, {len(metrics['rollout']['sequences'])} trajectories), "
                    "paper metrics (± = standard error over trajectories):")
        for line in format_paper_metrics(metrics["rollout"]["paper_metrics"]):
            logger.info(line)
        path = self.checkpoint_dir / "test_metrics.json"
        with open(path, "w") as f:
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
