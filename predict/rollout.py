#!/usr/bin/env python3
"""
rollout.py — Autoregressive rollout of a trained model (MeshGraphNet or MGN-T) on deforming_plate.

For every trajectory of a split (``dataset.db``) the model is applied step by
step to its own predictions:

1. Step 0 takes the ground truth as the initial state (no model call).
2. Positions: x_{t+1} = x_t + predicted velocity for free nodes; actuator and
   clamped nodes follow their prescribed motion.
3. The world-space part of the mesh edges and the world edges are rebuilt from
   the predicted positions (``graph_features.rebuild_edges``).
4. With history (``--history 1`` export) the ``*_prev`` inputs are replaced by
   the previous prediction. Otherwise stress is predicted directly and not
   fed back.

Errors are measured on the nodes in ``loss_mask`` from step 1 on.

Every call writes into a folder of its own and never overwrites an earlier
rollout: ``rollouts/<YYYYMMDD_HHMMSS>_<training run>_<checkpoint>_<split>/``
(or ``--output-dir``, which must not exist yet or be empty). It holds

- ``rollout_info.json``: what was rolled out and how — checkpoint (path,
  SHA-256, epoch, full training config), dataset (path, SHA-256 of dataset.db
  and normalizer.json, split, trajectories, check against the dataset the model
  was trained on), options, command line, git commit, versions, results;
- ``<split>_rollout_summary.json``: the metrics of the MGN-T paper
  (``paper_metrics``: RMSE-1, RMSE-all and R-RMSE of position and stress, see
  ``_PaperMetrics``) plus normalized MSE, RMSE/MAE per target field and position
  RMSE, each per step and per trajectory;
- ``rollout.log`` and optionally one ``.vtu`` per step under
  ``<split>/<trajectory>/`` (prediction, ground truth, error; needs pyvista)
  plus ``<trajectory>.vtu.series``, which ParaView opens as one time series.

Usage
-----
    python predict/rollout.py --checkpoint checkpoints/<run>/best_model.pt \\
        --data-dir data/deforming_plate_s10_small --split test [--no-export-vtu]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dataset_preprocessor import DeformingPlateDataset  # noqa: E402
from dataset_preprocessor.graph_features import (  # noqa: E402
    TARGET_FEATURE_NAMES,
    VELOCITY_TARGET_COLUMNS,
    free_node_mask,
    prev_to_target_columns,
    rebuild_edges,
)
from utils.timezone import Formatter, from_timestamp, now  # noqa: E402

logger = logging.getLogger("rollout")

ROLLOUTS_DIR = _REPO_ROOT / "rollouts"
STRESS_COLUMN = TARGET_FEATURE_NAMES.index("stress")
# metadata.json keys that may differ between machines without changing the data
_MACHINE_SPECIFIC_METADATA = {"raw_dir"}


class RolloutRunner:
    """Roll a model out over the trajectories of a ``dataset.db``.

    Parameters
    ----------
    model : nn.Module
        Returns normalized targets for a ``Data`` sample.
    normalizer : Normalizer
        Statistics of the dataset the model was trained on.
    db_path : str or Path
        The ``dataset.db`` to roll out on.
    device : torch.device or str
    output_dir : Path, optional
        Where the summary and the .vtu files go; None writes nothing
        (rollout validation during training).
    node_feature_names : list[str], optional
        Input layout the model was trained with; checked against the dataset.
    lazy : bool
        Read the frames from the DB on access instead of loading the split.
    """

    def __init__(self, model, normalizer, db_path, device, output_dir=None,
                 node_feature_names: Optional[List[str]] = None, lazy: bool = False):
        self.model = model
        self.normalizer = normalizer
        self.db_path = Path(db_path)
        self.device = torch.device(device)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.expected_node_names = node_feature_names
        self.lazy = lazy
        self._datasets: Dict[tuple, DeformingPlateDataset] = {}

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, data_dir: str, device: str = "auto",
                        output_dir: Optional[str] = None) -> "RolloutRunner":
        """Runner around a ``Trainer`` checkpoint; ``data_dir`` must hold the
        dataset (and normalizer) the model was trained on."""
        from train.trainer import load_trained_model

        trained = load_trained_model(checkpoint_path, device=device, data_dir=data_dir)
        logger.info(f"Loaded {checkpoint_path} (epoch {trained.epoch}) on {trained.device}")
        return cls.from_trained(trained, data_dir, output_dir)

    @classmethod
    def from_trained(cls, trained, data_dir: str, output_dir: Optional[str] = None
                     ) -> "RolloutRunner":
        """Runner around a model restored with ``load_trained_model``."""
        return cls(trained.model, trained.normalizer, Path(data_dir) / "dataset.db",
                   trained.device, output_dir, trained.metadata["node_feature_names"])

    # ── main loop ───────────────────────────────────────────────────────────

    def run(self, split: str = "test", max_sequences: Optional[int] = None,
            max_steps: Optional[int] = None, export_vtu: bool = False,
            verbose: bool = True, one_step: bool = False) -> Dict:
        """Roll out every sequence of ``split``; returns the summary dict.

        ``max_sequences`` / ``max_steps`` limit the number of trajectories and
        the steps per trajectory. ``export_vtu`` needs ``output_dir``.
        ``one_step`` adds the one-step (teacher-forced) paper metrics RMSE-1,
        at the cost of a second model call per step.
        """
        if export_vtu and self.output_dir is None:
            raise ValueError("export_vtu needs an output_dir")
        dataset = self._dataset(split, with_cells=export_vtu)
        if self.expected_node_names and dataset.node_feature_names != self.expected_node_names:
            raise ValueError(
                f"{self.db_path}: node features {dataset.node_feature_names} differ from "
                f"the model's {self.expected_node_names} — roll out on the dataset the "
                "model was trained on")
        sequences = list(dataset.sequences().items())[:max_sequences]
        export_vtu = export_vtu and self._pyvista_available()

        log = logger.info if verbose else logger.debug
        log(f"Rollout of {len(sequences)} sequence(s) of split {split!r} from {self.db_path}")

        self.model.eval()
        prev_to_target = prev_to_target_columns(dataset.node_feature_names, TARGET_FEATURE_NAMES)
        all_metrics = _StepMetrics()
        paper_metrics: List[_PaperMetrics] = []
        sequence_results = {}
        for sequence_id, sample_indices in sequences:
            metrics, paper = self._rollout_sequence(dataset, split, sequence_id,
                                                    sample_indices[:max_steps], prev_to_target,
                                                    export_vtu, one_step)
            all_metrics.merge(metrics)
            paper_metrics.append(paper)
            sequence_results[sequence_id] = {**metrics.sequence_summary(),
                                             "paper_metrics": paper.result()}

        summary = {"split": split, "num_sequences": len(sequence_results),
                   "target_feature_names": list(TARGET_FEATURE_NAMES),
                   "paper_metrics": _PaperMetrics.aggregate(paper_metrics),
                   **all_metrics.summary(), "sequences": sequence_results}
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = self.output_dir / f"{split}_rollout_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            log(f"Summary: {summary_path}")
        if verbose:
            _log_summary(summary, self.output_dir if export_vtu else None)
        return summary

    @torch.no_grad()
    def _rollout_sequence(self, dataset, split, sequence_id, sample_indices,
                          prev_to_target, export_vtu, one_step=False
                          ) -> Tuple["_StepMetrics", "_PaperMetrics"]:
        """Roll out one trajectory; returns its per-step errors and paper metrics."""
        node_mean, node_std = self.normalizer.mean_std("node_features", self.device)
        _, target_std = self.normalizer.mean_std("target", self.device)
        is_velocity = torch.zeros(len(TARGET_FEATURE_NAMES), dtype=torch.bool,
                                  device=self.device)
        is_velocity[VELOCITY_TARGET_COLUMNS] = True
        metrics = _StepMetrics()
        paper = _PaperMetrics()
        prev_prediction = prev_pos = None      # raw prediction / positions of step t-1
        exported_steps: List[int] = []

        for step, idx in enumerate(sample_indices):
            data = dataset[idx].to(self.device)
            target = self.normalizer.denormalize(data.y, "target")
            true_pos = data.world_pos
            free = free_node_mask(data.x)
            # Prescribed values are not predicted: the actuator keeps all its
            # ground-truth targets, clamped nodes their (zero) velocity.
            prescribed = (~data.loss_mask)[:, None] | ((~free)[:, None] & is_velocity)
            paper.observe(true_pos, target[:, STRESS_COLUMN])

            if one_step:   # from the ground-truth state: x_{t+1} error = velocity error
                single = torch.where(prescribed, target, self.normalizer.denormalize(
                    self.model(data), "target"))
                paper.add("q_1", (single - target)[:, VELOCITY_TARGET_COLUMNS])
                paper.add("stress_1", (single - target)[:, STRESS_COLUMN])

            if step == 0:
                pos = true_pos
                prediction = target.clone()   # initial condition = ground truth
            else:
                pos = torch.where(free[:, None],
                                  prev_pos + prev_prediction[:, VELOCITY_TARGET_COLUMNS],
                                  true_pos)
                rebuild_edges(data, pos, self.normalizer, dataset.contact_radius)
                data.world_pos = pos
                if prev_to_target:
                    x = data.x.clone()
                    for column, target_column in prev_to_target.items():
                        x[:, column] = ((prev_prediction[:, target_column] - node_mean[column])
                                        / node_std[column])
                    data.x = x
                prediction = self.normalizer.denormalize(self.model(data), "target")

            prediction = torch.where(prescribed, target, prediction)

            if step > 0:   # step 0 is the ground-truth seed
                loss_mask = data.loss_mask
                metrics.add(step, prediction[loss_mask] - target[loss_mask], target_std,
                            (pos - true_pos)[loss_mask])
                paper.add("q_all", pos - true_pos)
                paper.add("stress_all", (prediction - target)[:, STRESS_COLUMN])
                if export_vtu:
                    exported_steps.append(step)
                    self._export_vtu(split, sequence_id, step, data, true_pos,
                                     prediction, target)
            prev_prediction, prev_pos = prediction, pos
        if exported_steps:
            self._write_vtu_series(split, sequence_id, exported_steps)
        return metrics, paper

    # ── helpers ─────────────────────────────────────────────────────────────

    def _dataset(self, split: str, with_cells: bool) -> DeformingPlateDataset:
        key = (split, with_cells)
        if key not in self._datasets:
            self._datasets[key] = DeformingPlateDataset(
                self.db_path, split, self.normalizer, with_cells=with_cells, lazy=self.lazy)
        return self._datasets[key]

    @staticmethod
    def _pyvista_available() -> bool:
        try:
            import pyvista  # noqa: F401
        except ImportError:
            logger.warning("pyvista not installed — skipping .vtu export")
            return False
        return True

    def _export_vtu(self, split, sequence_id, step, data, true_pos, prediction, target):
        """Write the predicted mesh of one step with prediction, ground truth and error."""
        import pyvista as pv

        points = data.world_pos.cpu().numpy().astype("float64")
        cells = data.cells.cpu().numpy().astype("int64")                  # (C, 4)
        cell_array = np.hstack([np.full((cells.shape[0], 1), 4, dtype="int64"), cells]).ravel()
        cell_types = np.full(cells.shape[0], pv.CellType.TETRA, dtype=np.uint8)
        mesh = pv.UnstructuredGrid(cell_array, cell_types, points)

        true_points = true_pos.cpu().numpy().astype("float32")
        mesh.point_data["world_pos_gt"] = true_points
        mesh.point_data["position_error"] = np.linalg.norm(
            points.astype("float32") - true_points, axis=1)
        prediction, target = prediction.cpu().numpy(), target.cpu().numpy()
        for i, name in enumerate(TARGET_FEATURE_NAMES):
            mesh.point_data[f"{name}_pred"] = prediction[:, i].astype("float32")
            mesh.point_data[f"{name}_gt"] = target[:, i].astype("float32")
            mesh.point_data[f"{name}_error"] = np.abs(
                prediction[:, i] - target[:, i]).astype("float32")

        out = self.output_dir / split / str(sequence_id)
        out.mkdir(parents=True, exist_ok=True)
        mesh.save(str(out / f"t{step:04d}.vtu"))

    def _write_vtu_series(self, split, sequence_id, steps: List[int]) -> None:
        """``<trajectory>.vtu.series`` next to the .vtu files: ParaView opens the
        whole trajectory as one time series; time = rollout step (file tNNNN)."""
        series = {"file-series-version": "1.0",
                  "files": [{"name": f"t{step:04d}.vtu", "time": step} for step in steps]}
        path = self.output_dir / split / str(sequence_id) / f"{sequence_id}.vtu.series"
        with open(path, "w") as f:
            json.dump(series, f, indent=1)


class _StepMetrics:
    """Errors per rollout step, collected over one or more sequences."""

    def __init__(self):
        self.mse_normalized = defaultdict(list)                # step -> [value per sequence]
        self.position_rmse = defaultdict(list)
        self.rmse = defaultdict(lambda: defaultdict(list))     # field -> step -> [...]
        self.mae = defaultdict(lambda: defaultdict(list))

    def add(self, step: int, error: torch.Tensor, target_std: torch.Tensor,
            position_error: torch.Tensor) -> None:
        """``error`` (M, 4) raw target error and ``position_error`` (M, 3) of the loss nodes."""
        self.mse_normalized[step].append((error / target_std).pow(2).mean().item())
        self.position_rmse[step].append(position_error.pow(2).sum(1).mean().sqrt().item())
        for i, name in enumerate(TARGET_FEATURE_NAMES):
            self.rmse[name][step].append(error[:, i].pow(2).mean().sqrt().item())
            self.mae[name][step].append(error[:, i].abs().mean().item())

    def merge(self, other: "_StepMetrics") -> None:
        for mine, theirs in ((self.mse_normalized, other.mse_normalized),
                             (self.position_rmse, other.position_rmse)):
            for step, values in theirs.items():
                mine[step].extend(values)
        for mine, theirs in ((self.rmse, other.rmse), (self.mae, other.mae)):
            for name, steps in theirs.items():
                for step, values in steps.items():
                    mine[name][step].extend(values)

    @staticmethod
    def _per_step(values: Dict[int, List[float]]) -> List[float]:
        return [float(np.mean(values[step])) for step in sorted(values)]

    @staticmethod
    def _overall(values: Dict[int, List[float]]) -> float:
        flat = [v for step_values in values.values() for v in step_values]
        return float(np.mean(flat)) if flat else float("nan")

    def summary(self) -> Dict:
        """Means over sequences per step, and over all steps and sequences."""
        return {
            # mean over steps and sequences, in normalized target units (= val loss units)
            "mean_mse_normalized": self._overall(self.mse_normalized),
            "mse_normalized_per_step": self._per_step(self.mse_normalized),
            # step 1 starts from exact positions, so its position error is 0
            "position_rmse_per_step": self._per_step(self.position_rmse),
            "rmse_per_field": {n: self._overall(s) for n, s in self.rmse.items()},
            "mae_per_field": {n: self._overall(s) for n, s in self.mae.items()},
            "rmse_per_field_per_step": {n: self._per_step(s) for n, s in self.rmse.items()},
        }

    def sequence_summary(self) -> Dict:
        """Summary of a single sequence."""
        last = max(self.position_rmse, default=None)
        return {
            "num_predicted_steps": len(self.mse_normalized),
            "mean_mse_normalized": self._overall(self.mse_normalized),
            "final_position_rmse": self.position_rmse[last][0] if last else float("nan"),
            "final_rmse_per_field": ({n: s[last][0] for n, s in self.rmse.items()}
                                     if last else {}),
        }


class _PaperMetrics:
    """Error metrics of the MGN-T paper (Iparraguirre et al., arXiv:2601.23177,
    Tables 4/5) for one trajectory, for the position q (m) and the von Mises
    stress (Pa):

    - RMSE-1: one step from the ground-truth state (``one_step``),
    - RMSE-all: over the whole rollout, from step 1 on,
    - R-RMSE: the same with every error divided by the infinity norm of the
      trajectory (largest |q| component / |stress| of the ground truth).

    As in the paper the squared errors are averaged over all nodes (the
    prescribed ones with error 0), the components of q, the steps and, in
    ``aggregate``, the trajectories before taking the root; the ± value there is
    the standard error over the trajectories. MeshGraphNet (Pfaff et al.) reports
    the position RMSE the same way.
    """

    VARIABLES = ("q", "stress")
    UNITS = {"q": "m", "stress": "Pa"}

    def __init__(self):
        self.squared = defaultdict(float)     # "q_1", "q_all", "stress_1", "stress_all"
        self.count = defaultdict(int)
        self.max_abs = {name: 0.0 for name in self.VARIABLES}

    def observe(self, position: torch.Tensor, stress: torch.Tensor) -> None:
        """Ground truth of one frame, for the infinity norm of the trajectory."""
        self.max_abs["q"] = max(self.max_abs["q"], position.abs().max().item())
        self.max_abs["stress"] = max(self.max_abs["stress"], stress.abs().max().item())

    def add(self, key: str, error: torch.Tensor) -> None:
        self.squared[key] += error.double().pow(2).sum().item()
        self.count[key] += error.numel()

    def result(self) -> Dict[str, float]:
        """``<variable>_rmse_<1|all>`` and ``<variable>_r_rmse_<1|all>`` of this trajectory."""
        out = {}
        for key in self.squared:
            name, horizon = key.split("_")
            rmse = math.sqrt(self.squared[key] / self.count[key])
            out[f"{name}_rmse_{horizon}"] = rmse
            out[f"{name}_r_rmse_{horizon}"] = rmse / self.max_abs[name] if self.max_abs[name] else None
        return out

    @classmethod
    def aggregate(cls, trajectories: List["_PaperMetrics"]) -> Dict:
        """Pooled over all trajectories, plus the standard error among them."""
        out: Dict = {"units": {**cls.UNITS, "r_rmse": "fraction (x100 = %)"},
                     "num_trajectories": len(trajectories)}
        keys = sorted({key for t in trajectories for key in t.squared})
        for key in keys:
            name, horizon = key.split("_")
            parts = [t for t in trajectories if t.count[key]]
            count = sum(t.count[key] for t in parts)
            relative = [t for t in parts if t.max_abs[name]]
            per_trajectory = {
                "rmse": [math.sqrt(t.squared[key] / t.count[key]) for t in parts],
                "r_rmse": [math.sqrt(t.squared[key] / t.count[key]) / t.max_abs[name]
                           for t in relative]}
            pooled = {
                "rmse": math.sqrt(sum(t.squared[key] for t in parts) / count),
                "r_rmse": (math.sqrt(sum(t.squared[key] / t.max_abs[name] ** 2 for t in relative)
                                     / sum(t.count[key] for t in relative)) if relative else None)}
            for metric in ("rmse", "r_rmse"):
                values = per_trajectory[metric]
                out[f"{name}_{metric}_{horizon}"] = pooled[metric]
                out[f"{name}_{metric}_{horizon}_se"] = (
                    float(np.std(values, ddof=1) / math.sqrt(len(values)))
                    if len(values) > 1 else None)
        return out


def format_paper_metrics(paper: Dict) -> List[str]:
    """Log lines in the units of the paper tables: q x1e-3 m, stress x1e3 Pa, R-RMSE %."""
    scale = {"q": (1e3, "e-3 m"), "stress": (1e-3, "e3 Pa")}
    lines = []
    for name in _PaperMetrics.VARIABLES:
        parts = []
        for horizon in ("1", "all"):
            key = f"{name}_rmse_{horizon}"
            if paper.get(key) is None:
                continue
            factor, unit = scale[name]
            se = paper.get(f"{key}_se")
            r_rmse = paper.get(f"{name}_r_rmse_{horizon}")
            text = f"RMSE-{horizon} {paper[key] * factor:.3g}"
            text += f" ± {se * factor:.2g}" if se is not None else ""
            text += f" ({unit})"
            text += f", R-RMSE-{horizon} {r_rmse * 100:.3g} %" if r_rmse is not None else ""
            parts.append(text)
        if parts:
            lines.append(f"    {name:7s} " + " | ".join(parts))
    return lines


def _log_summary(summary: Dict, vtu_dir: Optional[Path]) -> None:
    logger.info(f"Rollout of {summary['num_sequences']} sequence(s):")
    logger.info("  paper metrics (MGN-T, Tab. 4/5; ± = standard error over trajectories):")
    for line in format_paper_metrics(summary["paper_metrics"]):
        logger.info(line)
    logger.info(f"  normalized MSE (mean over steps): {summary['mean_mse_normalized']:.6f}")
    position = summary["position_rmse_per_step"]
    if position:
        logger.info(f"  position RMSE step 1 / mid / last: {position[0]:.6f} / "
                    f"{position[len(position) // 2]:.6f} / {position[-1]:.6f}")
    logger.info(f"  {'per field (mean over steps)':28s} {'RMSE':>12s} {'MAE':>12s}")
    for name in summary["rmse_per_field"]:
        logger.info(f"    {name:26s} {summary['rmse_per_field'][name]:12.6g} "
                    f"{summary['mae_per_field'][name]:12.6g}")
    if vtu_dir is not None:
        logger.info(f"  .vtu files: {vtu_dir}")


def _setup_logging(output_dir: Path) -> None:
    """Log to the console and to ``<output_dir>/rollout.log``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    log_file = output_dir / "rollout.log"
    logger.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(log_file, mode="w", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info(f"Log file: {log_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Output folder and provenance
# ─────────────────────────────────────────────────────────────────────────────

def _new_output_dir(checkpoint: Path, epoch: int, split: str) -> Path:
    """``rollouts/<time>_<training run>_<checkpoint>_<split>``, never an existing one."""
    name = checkpoint.stem                       # best_model -> best_model_ep35
    if not name.endswith(f"_{epoch}"):           # checkpoint_epoch_75 stays as it is
        name += f"_ep{epoch}"
    run = checkpoint.resolve().parent.name
    base = ROLLOUTS_DIR / f"{now():%Y%m%d_%H%M%S}_{run}_{name}_{split}"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}_{n}")
    return path


def _check_output_dir(path: Path) -> Path:
    """An explicit ``--output-dir`` must be new or empty, so nothing gets overwritten."""
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"{path} already contains files — rollouts never overwrite each "
                         "other; choose a new --output-dir or leave it out")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> Dict:
    """Path, size, modification time and SHA-256: identifies the exact file even if
    a later training overwrites it (best_model.pt)."""
    stat = path.stat()
    return {"path": str(path), "absolute_path": str(path.resolve()),
            "size_bytes": stat.st_size,
            "modified": from_timestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "sha256": _sha256(path)}


def _git_state() -> Dict:
    """Commit and uncommitted changes of the code that ran (None outside a git checkout)."""
    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(_REPO_ROOT), *args], capture_output=True,
                                    text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.rstrip() if result.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    if commit is None:            # e.g. in the Docker image, which has no .git
        return {"commit": None}
    changed = git("status", "--porcelain", "--untracked-files=no") or ""
    return {"commit": commit, "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_changes": changed.splitlines()}


def _rollout_info(args, trained, checkpoint: Path, data_dir: Path) -> Dict:
    """Everything needed to trace a rollout back to model, data and code."""
    import torch_geometric

    from models.meshgraphnet import count_parameters

    with open(data_dir / "metadata.json") as f:
        data_metadata = json.load(f)
    differences = {
        key: {"training": trained.metadata.get(key), "rollout": data_metadata.get(key)}
        for key in sorted(set(trained.metadata) | set(data_metadata))
        if key not in _MACHINE_SPECIFIC_METADATA
        and trained.metadata.get(key) != data_metadata.get(key)}
    return {
        "status": "running",
        "started": now().isoformat(timespec="seconds"),
        "command": shlex.join([sys.executable, *sys.argv]),
        "model": {
            "checkpoint": _file_record(checkpoint),
            "training_run": checkpoint.resolve().parent.name,
            "epoch": trained.epoch,
            "architecture": trained.config.architecture,
            "parameters": count_parameters(trained.model),
            "training_config": trained.config.to_dict(),
        },
        "data": {
            "data_dir": str(data_dir),
            "dataset_db": _file_record(data_dir / "dataset.db"),
            "normalizer": _file_record(data_dir / "normalizer.json"),
            "split": args.split,
            "max_sequences": args.max_sequences,
            "max_steps": args.max_steps,
            "metadata": {key: value for key, value in data_metadata.items()
                         if not isinstance(value, (list, dict))},
            # Same export settings as the dataset the model was trained on?
            "matches_training_dataset": not differences,
            "differences_to_training_dataset": differences,
        },
        "settings": {"device": str(trained.device), "export_vtu": args.export_vtu,
                     "one_step": args.one_step},
        "environment": {"git": _git_state(), "host": platform.node(),
                        "python": platform.python_version(), "torch": torch.__version__,
                        "torch_geometric": torch_geometric.__version__},
    }


def _write_json(path: Path, content: Dict) -> None:
    with open(path, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Autoregressive rollout of a trained model (MeshGraphNet or MGN-T) "
                    "on deforming_plate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", default=str(
        _REPO_ROOT / "checkpoints" / "example_run" / "best_model.pt"),
        help="Trainer checkpoint (.pt)")
    parser.add_argument("--data-dir", default=str(_REPO_ROOT / "data" / "deforming_plate_s10_small"),
                        help="dataset.db + normalizer.json — the dataset the checkpoint "
                             "was trained on")
    parser.add_argument("--output-dir", default=None,
                        help="new or empty folder (default: rollouts/<time>_<training run>_"
                             "<checkpoint>_<split>)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="roll out only the first N trajectories (default: all)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="steps per trajectory (default: all)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--export-vtu", action=argparse.BooleanOptionalAction, default=True,
                        help="write one .vtu per step (prediction / ground truth / error); "
                             "needs pyvista")
    parser.add_argument("--one-step", action=argparse.BooleanOptionalAction, default=True,
                        help="also compute the one-step paper metrics RMSE-1 (a second "
                             "model call per step)")
    args = parser.parse_args(argv)

    from train.trainer import load_trained_model

    checkpoint, data_dir = Path(args.checkpoint), Path(args.data_dir)
    trained = load_trained_model(str(checkpoint), device=args.device, data_dir=str(data_dir))
    output_dir = (_check_output_dir(Path(args.output_dir)) if args.output_dir
                  else _new_output_dir(checkpoint, trained.epoch, args.split))
    _setup_logging(output_dir)
    logger.info(f"Rollout folder: {output_dir}")
    logger.info(f"Loaded {checkpoint} (epoch {trained.epoch}) on {trained.device}")

    info_path = output_dir / "rollout_info.json"
    info = _rollout_info(args, trained, checkpoint, data_dir)
    if not info["data"]["matches_training_dataset"]:
        logger.warning("dataset differs from the one the model was trained on in: "
                       f"{', '.join(info['data']['differences_to_training_dataset'])} "
                       f"(details in {info_path.name})")
    _write_json(info_path, info)

    start = time.time()
    try:
        runner = RolloutRunner.from_trained(trained, str(data_dir), output_dir)
        summary = runner.run(split=args.split, max_sequences=args.max_sequences,
                             max_steps=args.max_steps, export_vtu=args.export_vtu,
                             one_step=args.one_step)
    except BaseException as error:
        info.update(status="failed", error=repr(error))
        raise
    else:
        position_rmse = summary["position_rmse_per_step"]
        info.update(status="finished", results={
            "summary_file": f"{args.split}_rollout_summary.json",
            "sequences": list(summary["sequences"]),
            "paper_metrics": summary["paper_metrics"],
            "mean_mse_normalized": summary["mean_mse_normalized"],
            "mean_position_rmse": float(np.mean(position_rmse)) if position_rmse else None,
            "final_position_rmse": position_rmse[-1] if position_rmse else None,
            "rmse_per_field": summary["rmse_per_field"],
        })
    finally:
        info.update(finished=now().isoformat(timespec="seconds"),
                    duration_seconds=round(time.time() - start, 1))
        _write_json(info_path, info)
        logger.info(f"Info: {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
