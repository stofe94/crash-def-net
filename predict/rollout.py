#!/usr/bin/env python3
"""
rollout.py — Autoregressive rollout of a trained model (MeshGraphNets or MGN-T) on a ``dataset.db``.

For every trajectory of a split (``dataset.db``) the model is applied step by
step to its own predictions:

1. The initial state is the ground truth of the first frame of the training
   (``[data] start_frame``), without one of the first frame of a trajectory, the
   undeformed state (with ``--history 1`` from its start velocity: the initial velocity
   of the decks, else at rest; exports up to v0.4.0 start from frames 0 and 1);
   ``--start-frame`` chooses another raw frame of the dataset. Every later frame is
   predicted, up to the last frame of the training (``end_frame``), else of the
   trajectory, or ``--end-frame``. Rollout step t is the frame t steps after the
   initial state, with the fields predicted for it.
2. Positions: x_{t+1} = x_t + predicted velocity for free nodes; actuator and
   clamped nodes follow their prescribed motion. Free rigid bodies (export
   ``--free-rigid-parts``) move by the rigid motion closest to their predicted
   velocities (``rigid_body_step``), so they keep their shape.
3. The world-space part of the mesh edges and the world edges are rebuilt from
   the predicted positions (``graph_features.rebuild_edges``).
4. With history (``--history 1`` export) the ``*_prev`` inputs are replaced by
   the previous prediction. Otherwise the result fields (e.g. the stress) are
   predicted directly and not fed back.

Errors are measured on the nodes in ``loss_mask`` at every predicted frame, without
targets a node does not have (``target_given``).

Every call writes into a folder of its own and never overwrites an earlier
rollout: ``rollouts/<YYYYMMDD_HHMMSS>_<training run>_<checkpoint>_<split>/``
(or ``--output-dir``, which must not exist yet or be empty). It holds

- ``rollout_info.json``: what was rolled out and how — checkpoint (path,
  SHA-256, epoch, full training config), dataset (path, the content fingerprint
  of dataset.db — the SHA-256 of the file for exports before format v4 —, the
  SHA-256 of normalizer.json and where the normalizer came from, split,
  trajectories, check against the dataset the model was trained on), options,
  command line, git commit, versions, results;
- ``<split>_rollout_summary.json``: the metrics of the MGN-T paper
  (``rollout_metrics``: RMSE-1, RMSE-all and R-RMSE of position and, if the
  dataset predicts it, the stress; see ``_RolloutMetrics``) plus normalized MSE, RMSE/MAE per target field, position
  RMSE, the RMS displacement of the ground truth (its scale) and the relative L2
  position error (``position_rel_l2``: ||x_pred - x_true|| / ||x_true|| over the nodes
  and their coordinates, as in Nabian et al., arXiv:2510.15201), each per step,
  for every trajectory and as the mean over them. The metrics are in SI, whatever
  the units of the data (``utils.units``): positions and velocities in m, the
  stress in Pa; other result fields in the units of the data;
- ``rollout.log`` and optionally one ``.vtu`` per frame of the run under
  ``<split>/<trajectory>/`` (prediction, ground truth, error; needs pyvista)
  plus ``<trajectory>.vtu.series``, which ParaView opens as one time series. Each file
  is named by its raw frame of the run (``tNNNN``, the time of the series), from the
  first frame of the dataset: the given frames up to the initial state with prediction
  = ground truth, then one file per rollout step.

``--config FILE`` takes the settings from a TOML file (``ROLLOUT_CONFIG_KEYS``, example:
``examples/quick_start/rollout.toml``); options given on the command line override it. With
``--evaluate`` (``[evaluation] enabled``) ``utils/evaluate_run.py`` evaluates the run on
the new rollout folder right after the rollout; with ``--animate`` (``[animation]
enabled``) ``utils/animate_rollout.py`` renders one of its trajectories.

Usage
-----
    python predict/rollout.py --checkpoint checkpoints/<run>/best_model.pt \\
        --data-dir data/deforming_plate_s10_small --split test [--no-export-vtu]
    python predict/rollout.py --config examples/quick_start/rollout.toml
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
import toml  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from dataset_preprocessor import TrajectoryDataset  # noqa: E402
from dataset_preprocessor.dataset import DEFAULT_STATIC_CACHE_MB, current_names, dataset_fingerprint  # noqa: E402
from dataset_preprocessor.export_config import _flatten  # noqa: E402
from dataset_preprocessor.graph_features import (  # noqa: E402
    ALL_FEATURES,
    CELL_NUM_NODES,
    CELL_TETRA,
    DEFAULT_WORLD_EDGE_EXCLUDE_DISTANCE,
    VELOCITY_TARGET_COLUMNS,
    FeatureSelection,
    add_world_edges,
    free_node_mask,
    prev_to_target_columns,
    rebuild_edges,
    select_metadata,
)
from utils.timezone import Formatter, from_timestamp, now  # noqa: E402
from utils.units import SI_STRESS, stress_display  # noqa: E402

logger = logging.getLogger("rollout")

ROLLOUTS_DIR = _REPO_ROOT / "rollouts"
# TOML key of ``--config`` -> option of this script (dest of its parser), in the order of
# the reference.
ROLLOUT_CONFIG_KEYS = {
    "rollout.checkpoint": "checkpoint",
    "rollout.data_dir": "data_dir",
    "rollout.split": "split",
    "rollout.output_dir": "output_dir",
    "rollout.max_sequences": "max_sequences",
    "rollout.start_frame": "start_frame",
    "rollout.end_frame": "end_frame",
    "rollout.max_steps": "max_steps",
    "rollout.device": "device",
    "rollout.export_vtu": "export_vtu",
    "rollout.one_step": "one_step",
    "rollout.lazy": "lazy",
    "evaluation.enabled": "evaluate",
    "evaluation.output_dir": "evaluation_output_dir",
    "animation.enabled": "animate",
    "animation.trajectory": "animation_trajectory",
    "animation.output_dir": "animation_output_dir",
}
# Trajectories of equal length are rolled out together, as one batch of graphs with
# one model call per step, up to this many nodes in all.
BATCH_NODES = 100_000
# Nodes per VTK cell type, indexed by the type number.
_NODES_OF_CELL_TYPE = np.zeros(max(CELL_NUM_NODES) + 1, dtype=np.int64)
_NODES_OF_CELL_TYPE[list(CELL_NUM_NODES)] = list(CELL_NUM_NODES.values())
# metadata.json keys that may differ without changing the samples: the machine
# (raw_dir) and the storage format of dataset.db (db_format)
_MACHINE_SPECIFIC_METADATA = {"raw_dir", "db_format"}


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
    world_edges, world_edge_exclude_distance, world_edge_exclude_hops
        World edge variant the model was trained with (``TrainingConfig``).
    static_cache_mb : float
        Lazy: static graph data kept per split (``TrajectoryDataset``).
    features : FeatureSelection
        Features of the export the model was trained with; ``normalizer`` holds these.

    The world edges are built on ``device`` (``add_world_edges`` /
    ``rebuild_edges``), not by the dataset.
    """

    def __init__(self, model, normalizer, db_path, device, output_dir=None,
                 node_feature_names: Optional[List[str]] = None, lazy: bool = False,
                 world_edges: str = "all",
                 world_edge_exclude_distance: float = DEFAULT_WORLD_EDGE_EXCLUDE_DISTANCE,
                 world_edge_exclude_hops: Optional[int] = None,
                 static_cache_mb: float = DEFAULT_STATIC_CACHE_MB,
                 features: FeatureSelection = ALL_FEATURES):
        self.model = model
        self.normalizer = normalizer
        self.db_path = Path(db_path)
        self.device = torch.device(device)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.expected_node_names = node_feature_names
        self.lazy = lazy
        self.world_edges = world_edges
        self.world_edge_exclude_distance = world_edge_exclude_distance
        self.world_edge_exclude_hops = world_edge_exclude_hops
        self.static_cache_mb = static_cache_mb
        self.features = features
        self._datasets: Dict[tuple, TrajectoryDataset] = {}

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
    def from_trained(cls, trained, data_dir: str, output_dir: Optional[str] = None,
                     lazy: Optional[bool] = None) -> "RolloutRunner":
        """Runner around a model restored with ``load_trained_model``; ``lazy``
        defaults to the loading mode of the training."""
        config = trained.config
        return cls(trained.model, trained.normalizer, Path(data_dir) / "dataset.db",
                   trained.device, output_dir, trained.metadata["node_feature_names"],
                   lazy=config.lazy_loading if lazy is None else lazy,
                   world_edges=config.world_edges,
                   world_edge_exclude_distance=config.world_edge_exclude_distance,
                   world_edge_exclude_hops=config.world_edge_exclude_hops,
                   static_cache_mb=config.static_cache_mb, features=config.features)

    # ── main loop ───────────────────────────────────────────────────────────

    def run(self, split: str = "test", max_sequences: Optional[int] = None,
            max_steps: Optional[int] = None, export_vtu: bool = False,
            verbose: bool = True, one_step: bool = False,
            start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> Dict:
        """Roll out every sequence of ``split``; returns the summary dict.

        ``max_sequences`` / ``max_steps`` limit the number of trajectories and
        the steps per trajectory; ``start_frame`` / ``end_frame`` (raw frames of the run,
        of the dataset) start from the ground truth of a later frame and end earlier
        than the dataset's samples. ``export_vtu`` needs ``output_dir``.
        ``one_step`` adds the one-step (teacher-forced) rollout metrics RMSE-1,
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
        initial, last = self._window(dataset, start_frame, end_frame)
        offset = initial - dataset.initial_frame   # samples before the initial state
        sequences = [(sequence_id, indices[offset:offset + last - initial])
                     for sequence_id, indices in list(dataset.sequences().items())[:max_sequences]]
        export_vtu = export_vtu and self._pyvista_available()

        log = logger.info if verbose else logger.debug
        log(f"Rollout of {len(sequences)} sequence(s) of split {split!r} from {self.db_path}")

        self.model.eval()
        self._target_names = dataset.target_feature_names
        prev_to_target = prev_to_target_columns(dataset.node_feature_names, self._target_names)
        all_metrics = _StepMetrics(self._target_names)
        rollout_metrics: List[_RolloutMetrics] = []
        sequence_results = {}
        num_nodes = {traj.sequence_id: traj.num_nodes for traj in dataset.trajectories}
        for group in _sequence_batches(sequences, num_nodes, max_steps):
            results = self._rollout_batch(dataset, split, group, prev_to_target, export_vtu,
                                          one_step, initial)
            for (sequence_id, _), (metrics, final) in zip(group, results):
                all_metrics.merge(metrics)
                rollout_metrics.append(final)
                sequence_results[sequence_id] = {**metrics.sequence_summary(),
                                                 "rollout_metrics": final.result()}

        num_steps = len(all_metrics.summary()["position_rmse_per_step"])
        summary = {"split": split, "num_sequences": len(sequence_results),
                   "target_feature_names": list(self._target_names),
                   # raw frames of the run: the one the rollout starts from, and from the
                   # first frame of the dataset to the last predicted one
                   "initial_frame": dataset.raw_frames[initial],
                   "frames": dataset.raw_frames[:initial + 1 + num_steps],
                   "rollout_metrics": _RolloutMetrics.aggregate(rollout_metrics),
                   **all_metrics.summary(), "sequences": sequence_results}
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = self.output_dir / f"{split}_rollout_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            log(f"Summary: {summary_path}")
        if verbose:
            _log_summary(summary, self.output_dir if export_vtu else None, dataset.stress_unit)
        return summary

    @staticmethod
    def _window(dataset, start_frame: Optional[int], end_frame: Optional[int]
                ) -> Tuple[int, int]:
        """Frames of the dataset the rollout starts from and ends with: those of the raw
        frames ``start_frame`` / ``end_frame``, by default the bounds of its samples."""
        frames = dataset.raw_frames
        allowed = frames[dataset.initial_frame:dataset.last_frame + 1]
        for name, frame in (("start", start_frame), ("end", end_frame)):
            if frame is not None and frame not in allowed:
                raise ValueError(f"{name} frame {frame}: a raw frame of the dataset from "
                                 f"{allowed[0]} to {allowed[-1]} ({frames[0]}, {frames[1]}, "
                                 f"... {frames[-1]})")
        initial = dataset.initial_frame if start_frame is None else frames.index(start_frame)
        last = dataset.last_frame if end_frame is None else frames.index(end_frame)
        if last <= initial:
            raise ValueError(f"end frame {frames[last]} must come after the start frame "
                             f"{frames[initial]}")
        return initial, last

    @torch.no_grad()
    def _rollout_batch(self, dataset, split, group, prev_to_target, export_vtu,
                       one_step=False, initial: Optional[int] = None
                       ) -> List[Tuple["_StepMetrics", "_RolloutMetrics"]]:
        """Roll out the trajectories of ``group`` ((sequence id, sample indices), all
        of the same length) together, as one batch of graphs; returns the per-step
        errors and rollout metrics of each. The errors are summed per graph on the
        device and read once at the end, so the steps do not wait for the GPU."""
        initial = dataset.first_frame if initial is None else initial
        node_mean, node_std = self.normalizer.mean_std("node_features", self.device)
        _, target_std = self.normalizer.mean_std("target", self.device)
        names = self._target_names
        is_velocity = torch.zeros(len(names), dtype=torch.bool, device=self.device)
        is_velocity[VELOCITY_TARGET_COLUMNS] = True
        # The rollout metrics of the stress, if the dataset predicts it.
        stress = names.index("stress") if "stress" in names else None
        num_bodies = len(dataset.rigid_bodies)   # free rigid bodies per graph
        # Errors in SI (utils.units): positions and velocities in m, the stress in Pa.
        length, stress_si = dataset.si_factors
        si = torch.ones(len(names), device=self.device)
        si[VELOCITY_TARGET_COLUMNS] = length
        if stress is not None:
            si[stress] = stress_si
        num_graphs, num_steps = len(group), len(group[0][1])
        step_errors: List[torch.Tensor] = []   # (len(_step_rows(names)), graphs) per step
        totals = _RolloutSums(num_graphs, self.device)
        prev_prediction = prev_next = None     # raw prediction of the step before, its positions
        stress_sizes = None                    # nodes per graph that have a stress

        for step in range(num_steps):
            samples = [dataset.sample(indices[step], world_edges=False) for _, indices in group]
            sizes = [sample.num_nodes for sample in samples]
            # World edges on the device: from the ground truth only where the model
            # sees it (one_step, the initial state), else rebuilt from the prediction below.
            data = Batch.from_data_list(samples).to(self.device)
            target = self.normalizer.denormalize(data.y, "target")
            true_pos = data.world_pos
            free = free_node_mask(data.x)
            # Prescribed values are not predicted: the actuator keeps all its
            # ground-truth targets, clamped nodes their (zero) velocity.
            prescribed = (~data.loss_mask)[:, None] | ((~free)[:, None] & is_velocity)
            # A target a node does not have stays neutral (its "ground truth" is the mean in
            # y): the prediction equals it, and the metrics do not count these nodes.
            given = getattr(data, "target_given", None)
            if given is not None:
                prescribed = prescribed | ~given
            if stress is not None and stress_sizes is None:   # once: the same over a run
                stress_sizes = (sizes if given is None else _per_graph(
                    given[:, stress], data.batch, num_graphs).long().cpu().numpy())
            true_stress = None
            if stress is not None:
                true_stress = target[:, stress] * stress_si
                true_stress = true_stress if given is None else true_stress * given[:, stress]
            totals.observe(true_pos * length, true_stress, data.batch)

            single = None
            if one_step:   # from the ground-truth state: x_{t+1} error = velocity error
                add_world_edges(data, self.normalizer, dataset.contact_radius)
                single = torch.where(prescribed, target, self.normalizer.denormalize(
                    self.model(data), "target"))
                totals.add("q_1", (single - target)[:, VELOCITY_TARGET_COLUMNS] * length,
                          data.batch, sizes)
                if stress is not None:
                    totals.add("stress_1", (single - target)[:, stress] * stress_si, data.batch,
                              stress_sizes)

            if step == 0:   # the initial condition: the ground truth of the first sample
                pos = start_pos = true_pos
                if export_vtu:   # the given frames, so that the results start with the run
                    for (sequence_id, indices), sample in zip(group, samples):
                        given_pos, given_target = dataset.initial_state(indices[0], initial)
                        for frame in range(len(given_pos)):
                            self._export_vtu(split, sequence_id, dataset.raw_frames[frame],
                                             sample, given_pos[frame], given_pos[frame],
                                             given_target[frame], given_target[frame])
                if single is None:
                    add_world_edges(data, self.normalizer, dataset.contact_radius)
                    prediction = self.normalizer.denormalize(self.model(data), "target")
                else:       # the same input: the one-step prediction
                    prediction = single
            else:
                pos = torch.where(free[:, None], prev_next, true_pos)
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
            if num_bodies:   # free rigid bodies keep their shape
                body = torch.where(data.rigid_body >= 0,
                                   data.rigid_body + data.batch * num_bodies, -1)
                prediction = prediction.clone()
                prediction[:, VELOCITY_TARGET_COLUMNS] = rigid_body_step(
                    pos, prediction[:, VELOCITY_TARGET_COLUMNS], body, num_graphs * num_bodies)

            # Step t + 1 is the frame this prediction leads to: its positions against the
            # ground truth of that frame, with the fields predicted for it. Prescribed nodes
            # are at their ground truth (pos) and move by it, so their error is 0.
            next_pos = pos + prediction[:, VELOCITY_TARGET_COLUMNS]
            true_next = true_pos + target[:, VELOCITY_TARGET_COLUMNS]
            totals.observe(true_next * length, None, data.batch)
            step_errors.append(_step_errors((prediction - target) * si,
                                            (next_pos - true_next) * length,
                                            (true_next - start_pos) * length, true_next * length,
                                            data.loss_mask, data.batch, num_graphs,
                                            target_std * si, given))
            totals.add("q_all", (next_pos - true_next) * length, data.batch, sizes)
            if stress is not None:
                totals.add("stress_all", (prediction - target)[:, stress] * stress_si,
                          data.batch, stress_sizes)
            if export_vtu:
                bounds = np.cumsum([0] + sizes)
                for (sequence_id, _), sample, a, b in zip(group, samples, bounds[:-1],
                                                          bounds[1:]):
                    self._export_vtu(split, sequence_id, dataset.raw_frames[initial + step + 1],
                                     sample, next_pos[a:b], true_next[a:b], prediction[a:b],
                                     target[a:b])
            prev_prediction, prev_next = prediction, next_pos
        if export_vtu:
            for sequence_id, _ in group:
                self._write_vtu_series(split, sequence_id,
                                       dataset.raw_frames[:initial + num_steps + 1])

        errors = (torch.stack(step_errors).cpu().numpy() if step_errors
                  else np.zeros((0, len(_step_rows(names)), num_graphs)))
        metrics = []
        for graph in range(num_graphs):
            graph_metrics = _StepMetrics(names)
            for step, values in enumerate(errors[:, :, graph], start=1):
                graph_metrics.add(step, values)
            metrics.append(graph_metrics)
        return list(zip(metrics, totals.trajectories()))

    # ── helpers ─────────────────────────────────────────────────────────────

    def share_dataset(self, split: str, dataset: TrajectoryDataset) -> None:
        """Roll out ``split`` on an existing dataset (e.g. of the trainer's val loader)
        instead of loading it a second time; it must use the same settings."""
        self._datasets[(split, dataset.with_cells)] = dataset

    def _dataset(self, split: str, with_cells: bool) -> TrajectoryDataset:
        key = (split, with_cells)
        if key not in self._datasets:
            self._datasets[key] = TrajectoryDataset(
                self.db_path, split, self.normalizer, with_cells=with_cells, lazy=self.lazy,
                world_edges=self.world_edges,
                world_edge_exclude_distance=self.world_edge_exclude_distance,
                world_edge_exclude_hops=self.world_edge_exclude_hops,
                static_cache_mb=self.static_cache_mb, build_world_edges=False,
                features=self.features)
        return self._datasets[key]

    @staticmethod
    def _pyvista_available() -> bool:
        try:
            import pyvista  # noqa: F401
        except ImportError:
            logger.warning("pyvista not installed — skipping .vtu export")
            return False
        return True

    def _export_vtu(self, split, sequence_id, frame, sample, pos, true_pos, prediction, target):
        """Write the predicted mesh ``pos`` of one frame of the run with prediction, ground
        truth and error; ``sample`` (of the dataset) holds the cells."""
        import pyvista as pv

        points = pos.cpu().numpy().astype("float64")
        cells = sample.cells.cpu().numpy().astype("int64")     # (C, 4) or (C, W) padded
        cell_types = getattr(sample, "cell_types", None)
        if cell_types is None:                                  # tetrahedra
            cell_types = np.full(cells.shape[0], CELL_TETRA, dtype=np.uint8)
        else:
            cell_types = cell_types.cpu().numpy().astype(np.uint8)
        nodes_per_cell = _NODES_OF_CELL_TYPE[cell_types]
        # VTK layout: [n, id_1 .. id_n] per cell, padding dropped
        keep = np.hstack([np.ones((cells.shape[0], 1), dtype=bool),
                          np.arange(cells.shape[1]) < nodes_per_cell[:, None]])
        cell_array = np.hstack([nodes_per_cell[:, None], cells])[keep]
        mesh = pv.UnstructuredGrid(cell_array, cell_types, points)

        true_points = true_pos.cpu().numpy().astype("float32")
        mesh.point_data["world_pos_gt"] = true_points
        mesh.point_data["position_error"] = np.linalg.norm(
            points.astype("float32") - true_points, axis=1)
        prediction, target = prediction.cpu().numpy(), target.cpu().numpy()
        given = getattr(sample, "target_given", None)
        if given is not None:   # no value: empty in the file, not a number that looks real
            prediction = np.where(given.numpy(), prediction, np.nan)
            target = np.where(given.numpy(), target, np.nan)
        for i, name in enumerate(self._target_names):
            mesh.point_data[f"{name}_pred"] = prediction[:, i].astype("float32")
            mesh.point_data[f"{name}_gt"] = target[:, i].astype("float32")
            mesh.point_data[f"{name}_error"] = np.abs(
                prediction[:, i] - target[:, i]).astype("float32")

        out = self.output_dir / split / str(sequence_id)
        out.mkdir(parents=True, exist_ok=True)
        mesh.save(str(out / f"t{frame:04d}.vtu"))

    def _write_vtu_series(self, split, sequence_id, frames: List[int]) -> None:
        """``<trajectory>.vtu.series`` next to the .vtu files: ParaView opens the
        whole trajectory as one time series; time = frame of the run (file tNNNN)."""
        series = {"file-series-version": "1.0",
                  "files": [{"name": f"t{frame:04d}.vtu", "time": frame} for frame in frames]}
        path = self.output_dir / split / str(sequence_id) / f"{sequence_id}.vtu.series"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(series, f, indent=1)


def _sequence_batches(sequences, num_nodes: Dict[str, int], max_steps: Optional[int]):
    """Consecutive sequences of equal length, up to ``BATCH_NODES`` nodes per batch."""
    group, size = [], 0
    for sequence_id, indices in sequences:
        indices = indices[:max_steps]
        if group and (size + num_nodes[sequence_id] > BATCH_NODES
                      or len(indices) != len(group[0][1])):
            yield group
            group, size = [], 0
        group.append((sequence_id, indices))
        size += num_nodes[sequence_id]
    if group:
        yield group


def rigid_body_step(pos: torch.Tensor, displacement: torch.Tensor, body: torch.Tensor,
                    num_bodies: int) -> torch.Tensor:
    """``displacement`` (N, 3) with that of the nodes of rigid bodies (``body``: index of
    the body, -1 for other nodes) replaced by the rigid motion closest to it: per body
    the mean translation and the rotation about the centre that fits it best (least
    squares), applied as an exact rotation, so that a body keeps its shape. On the
    device, without synchronization."""
    member = (body >= 0).double()[:, None]
    index = body.clamp(min=0)
    x, v = pos.double(), displacement.double()

    def per_body(values):   # sum over the nodes of every body
        return torch.zeros((num_bodies,) + values.shape[1:], dtype=torch.float64,
                           device=pos.device).index_add_(0, index, values)

    count = per_body(member).clamp(min=1)
    centre, shift = per_body(x * member) / count, per_body(v * member) / count
    r = x - centre[index]
    eye = torch.eye(3, dtype=torch.float64, device=pos.device)
    inertia = per_body(((r * r).sum(1)[:, None, None] * eye - r[:, :, None] * r[:, None, :])
                       * member[:, :, None])
    moment = per_body(torch.linalg.cross(r, v - shift[index]) * member)
    # v = shift + omega x r in the least squares sense; a body on a line has no
    # rotation about it, an empty one none at all.
    scale = inertia.diagonal(dim1=1, dim2=2).sum(1)[:, None, None]
    omega = torch.linalg.solve_ex(inertia + (1e-9 * scale + (scale == 0)) * eye,
                                  moment)[0]
    angle = omega.norm(dim=1)[:, None, None]
    a, b, c = (omega / angle[:, 0].clamp(min=1e-300)).unbind(1)
    zero = torch.zeros_like(a)
    cross = torch.stack([zero, -c, b, c, zero, -a, -b, a, zero], 1).view(-1, 3, 3)
    rotation = eye + angle.sin() * cross + (1 - angle.cos()) * (cross @ cross)   # Rodrigues
    moved = centre[index] + shift[index] + (rotation[index] @ r[:, :, None])[:, :, 0]
    return torch.where(member > 0, (moved - x).to(displacement.dtype), displacement)


def _per_graph(values: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Sum of ``values`` (N, ...) over the nodes of each graph -> (num_graphs, ...), float64."""
    sums = torch.zeros((num_graphs,) + values.shape[1:], dtype=torch.float64,
                       device=values.device)
    return sums.index_add_(0, batch, values.double())


def _step_errors(error: torch.Tensor, position_error: torch.Tensor, displacement: torch.Tensor,
                 position: torch.Tensor, loss_mask: torch.Tensor, batch: torch.Tensor,
                 num_graphs: int, target_std: torch.Tensor,
                 given: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Errors of one step per graph over the nodes in ``loss_mask`` -> (rows, graphs),
    rows as in ``_step_rows``, without targets a node does not have (``given``, N, F).
    ``error`` (N, F): raw target error; ``displacement`` (N, 3): ground-truth position
    minus that of the initial state; ``position`` (N, 3): the ground-truth position."""
    mask = loss_mask.double()[:, None]
    error = error.double()
    given = torch.ones_like(error) if given is None else given.double()
    normalized = ((error / target_std).pow(2) * given).sum(1, keepdim=True) \
        / given.sum(1, keepdim=True)
    per_node = torch.cat((normalized,
                          position_error.double().pow(2).sum(1, keepdim=True),
                          displacement.double().pow(2).sum(1, keepdim=True),
                          position.double().pow(2).sum(1, keepdim=True),
                          error.pow(2), error.abs()), dim=1)
    weights = torch.cat((mask, mask, mask, mask, mask * given, mask * given), dim=1)
    mean = (_per_graph(per_node * weights, batch, num_graphs)
            / _per_graph(weights, batch, num_graphs)).t()
    fields = error.shape[1]
    return torch.cat((mean[:1], mean[1:4].sqrt(), mean[4:4 + fields].sqrt(), mean[4 + fields:]))


def _step_rows(target_names: List[str]) -> List[str]:
    """Rows of one step of one sequence (``_step_errors``): normalized MSE, position
    RMSE, RMS displacement of the ground truth since the initial state, RMS of the
    ground-truth positions, then RMSE and MAE per target field."""
    return (["mse_normalized", "position_rmse", "displacement_rms", "position_rms"]
            + [f"rmse_{name}" for name in target_names]
            + [f"mae_{name}" for name in target_names])


class _StepMetrics:
    """Errors per rollout step, collected over one or more sequences."""

    def __init__(self, target_names: List[str]):
        self.target_names = list(target_names)
        self.mse_normalized = defaultdict(list)                # step -> [value per sequence]
        self.position_rmse = defaultdict(list)
        self.displacement_rms = defaultdict(list)
        self.position_rel_l2 = defaultdict(list)
        self.rmse = defaultdict(lambda: defaultdict(list))     # field -> step -> [...]
        self.mae = defaultdict(lambda: defaultdict(list))

    def add(self, step: int, values) -> None:
        """The errors of one sequence at ``step``, in the order of ``_step_rows``."""
        fields = len(self.target_names)
        self.mse_normalized[step].append(float(values[0]))
        self.position_rmse[step].append(float(values[1]))
        self.displacement_rms[step].append(float(values[2]))
        # ||x_pred - x_true|| / ||x_true|| over the nodes and their coordinates
        self.position_rel_l2[step].append(float(values[1] / values[3]) if values[3]
                                          else float("nan"))
        for i, name in enumerate(self.target_names):
            self.rmse[name][step].append(float(values[4 + i]))
            self.mae[name][step].append(float(values[4 + fields + i]))

    def merge(self, other: "_StepMetrics") -> None:
        for mine, theirs in ((self.mse_normalized, other.mse_normalized),
                             (self.position_rmse, other.position_rmse),
                             (self.displacement_rms, other.displacement_rms),
                             (self.position_rel_l2, other.position_rel_l2)):
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
            "position_rmse_per_step": self._per_step(self.position_rmse),
            # since the initial state, the last exact positions; same nodes and norm: the
            # position error of a prediction in which nothing moves, the scale of position_rmse
            "displacement_rms_per_step": self._per_step(self.displacement_rms),
            # relative L2 position error per step, mean over the sequences, and its mean
            # over the steps, as in Nabian et al. (arXiv:2510.15201); depends on the origin
            "position_rel_l2_per_step": self._per_step(self.position_rel_l2),
            "position_rel_l2": (float(np.mean(self._per_step(self.position_rel_l2)))
                                if self.position_rel_l2 else float("nan")),
            "rmse_per_field": {n: self._overall(s) for n, s in self.rmse.items()},
            "mae_per_field": {n: self._overall(s) for n, s in self.mae.items()},
            "rmse_per_field_per_step": {n: self._per_step(s) for n, s in self.rmse.items()},
        }

    def sequence_summary(self) -> Dict:
        """Summary of a single sequence: as ``summary``, plus the last step."""
        last = max(self.position_rmse, default=None)
        return {
            "num_predicted_steps": len(self.mse_normalized),
            **self.summary(),
            "final_position_rmse": self.position_rmse[last][0] if last else float("nan"),
            "final_rmse_per_field": ({n: s[last][0] for n, s in self.rmse.items()}
                                     if last else {}),
        }


class _RolloutMetrics:
    """Error metrics of the MGN-T paper (Iparraguirre et al., arXiv:2601.23177v4,
    App. A and Table 3) for one trajectory, for the position q (m) and the von Mises
    stress (Pa):

    - RMSE-1: one step from the ground-truth state (``one_step``),
    - RMSE-all: over the whole rollout, every frame after the initial state,
    - R-RMSE: the same with every error divided by the infinity norm of the
      trajectory (largest |q| component / |stress| of the ground truth).

    As in the paper the squared errors are averaged over all nodes (the
    prescribed ones with error 0), the components of q, the steps and, in
    ``aggregate``, the trajectories before taking the root; the ± value there is
    the standard error over the trajectories. MeshGraphNets (Pfaff et al.) reports
    the position RMSE the same way.
    """

    VARIABLES = ("q", "stress")
    UNITS = {"q": "m", "stress": "Pa"}

    def __init__(self):
        self.squared = defaultdict(float)     # "q_1", "q_all", "stress_1", "stress_all"
        self.count = defaultdict(int)
        self.max_abs = {name: 0.0 for name in self.VARIABLES}

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
    def aggregate(cls, trajectories: List["_RolloutMetrics"]) -> Dict:
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


class _RolloutSums:
    """The sums of ``_RolloutMetrics`` for the trajectories of a batch, on the device."""

    def __init__(self, num_graphs: int, device):
        self.num_graphs = num_graphs
        self.squared: Dict[str, torch.Tensor] = {}       # key -> (graphs,) float64
        self.count: Dict[str, np.ndarray] = {}
        self.max_abs = {name: torch.zeros(num_graphs, dtype=torch.float64, device=device)
                        for name in _RolloutMetrics.VARIABLES}

    def observe(self, position: torch.Tensor, stress: Optional[torch.Tensor],
                batch: torch.Tensor) -> None:
        """Ground truth of one frame, for the infinity norm of each trajectory (``stress``
        None: a dataset without it)."""
        observed = [("q", position.abs().amax(1))]
        if stress is not None:
            observed.append(("stress", stress.abs()))
        for name, values in observed:
            self.max_abs[name] = self.max_abs[name].scatter_reduce(
                0, batch, values.double(), "amax")

    def add(self, key: str, error: torch.Tensor, batch: torch.Tensor, sizes: List[int]) -> None:
        """``error`` (N,) or (N, C) of every node of the batch; ``sizes``: nodes per graph."""
        squared = error.double().pow(2)
        squared = _per_graph(squared.sum(1) if squared.dim() > 1 else squared, batch,
                             self.num_graphs)
        components = error.shape[1] if error.dim() > 1 else 1
        self.squared[key] = self.squared[key] + squared if key in self.squared else squared
        self.count[key] = self.count.get(key, 0) + np.asarray(sizes) * components

    def trajectories(self) -> List[_RolloutMetrics]:
        squared = {key: value.cpu().numpy() for key, value in self.squared.items()}
        max_abs = {name: value.cpu().numpy() for name, value in self.max_abs.items()}
        result = []
        for graph in range(self.num_graphs):
            metrics = _RolloutMetrics()
            for key, values in squared.items():
                metrics.squared[key] = float(values[graph])
                metrics.count[key] = int(self.count[key][graph])
            metrics.max_abs = {name: float(values[graph]) for name, values in max_abs.items()}
            result.append(metrics)
        return result


def format_rollout_metrics(metrics: Dict, stress_unit: str = SI_STRESS) -> List[str]:
    """Log lines in the units of the paper tables: q x1e-3 m, stress x1e3 Pa, R-RMSE %;
    the stress of data in MPa or GPa in that unit (``utils.units.stress_display``)."""
    factor, unit = stress_display(stress_unit)
    scale = {"q": (1e3, "e-3 m"), "stress": (factor, "e3 Pa" if unit == "kPa" else unit)}
    lines = []
    for name in _RolloutMetrics.VARIABLES:
        parts = []
        for horizon in ("1", "all"):
            key = f"{name}_rmse_{horizon}"
            if metrics.get(key) is None:
                continue
            factor, unit = scale[name]
            se = metrics.get(f"{key}_se")
            r_rmse = metrics.get(f"{name}_r_rmse_{horizon}")
            text = f"RMSE-{horizon} {metrics[key] * factor:.3g}"
            text += f" ± {se * factor:.2g}" if se is not None else ""
            text += f" ({unit})"
            text += f", R-RMSE-{horizon} {r_rmse * 100:.3g} %" if r_rmse is not None else ""
            parts.append(text)
        if parts:
            lines.append(f"    {name:7s} " + " | ".join(parts))
    return lines


def _log_summary(summary: Dict, vtu_dir: Optional[Path], stress_unit: str = SI_STRESS) -> None:
    logger.info(f"Rollout of {summary['num_sequences']} sequence(s):")
    logger.info("  rollout metrics (as defined in MGN-T v4, Tab. 3; ± = standard error over trajectories):")
    for line in format_rollout_metrics(summary["rollout_metrics"], stress_unit):
        logger.info(line)
    logger.info(f"  normalized MSE (mean over steps): {summary['mean_mse_normalized']:.6f}")
    position = summary["position_rmse_per_step"]
    if position:
        logger.info(f"  position RMSE step 1 / mid / last: {position[0]:.6f} / "
                    f"{position[len(position) // 2]:.6f} / {position[-1]:.6f} m")
        logger.info(f"  position relative L2 (mean over steps / last step): "
                    f"{summary['position_rel_l2']:.4g} / "
                    f"{summary['position_rel_l2_per_step'][-1]:.4g}")
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


def _check_output_dir(path: Path, option: str = "--output-dir") -> Path:
    """An explicit output folder must be new or empty, so nothing gets overwritten."""
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"{path} already contains files — nothing gets overwritten; "
                         f"choose a new {option} or leave it out")
    return path


def median_trajectory(summary: Dict) -> str:
    """The trajectory at the median of the position error, typical of the split; the
    first one of a split can be any, also the best. A trajectory without a predicted
    step has no error (and no .vtu files, which the animation then reports)."""
    sequences = summary["sequences"]
    ranked = sorted(sequences,
                    key=lambda name: sequences[name]["rollout_metrics"].get("q_rmse_all", 0.0))
    return ranked[(len(ranked) - 1) // 2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, sha256: bool = True) -> Dict:
    """Path, size, modification time and SHA-256: identifies the exact file even if
    a later training overwrites it (best_model.pt)."""
    stat = path.stat()
    record = {"path": str(path), "absolute_path": str(path.resolve()),
              "size_bytes": stat.st_size,
              "modified": from_timestamp(stat.st_mtime).isoformat(timespec="seconds")}
    if sha256:
        record["sha256"] = _sha256(path)
    return record


def _dataset_record(path: Path) -> Dict:
    """``_file_record`` of a dataset.db with its content fingerprint (export format
    v4) instead of the SHA-256 of the whole file, which would take minutes for a
    large dataset; older exports have no fingerprint and keep the SHA-256."""
    fingerprint = dataset_fingerprint(path)
    if fingerprint is None:
        return _file_record(path)
    return {**_file_record(path, sha256=False), "fingerprint": fingerprint}


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


def _frame_setting(given: Optional[int], training: Optional[int]) -> Optional[int]:
    """A frame bound of the rollout: the given one, else that of the training."""
    return training if given is None else given


def _rollout_info(args, trained, checkpoint: Path, data_dir: Path) -> Dict:
    """Everything needed to trace a rollout back to model, data and code."""
    import torch_geometric

    from models.meshgraphnets import count_parameters

    with open(data_dir / "metadata.json", encoding="utf-8") as f:
        data_metadata = json.load(f)
    selected = current_names(select_metadata(data_metadata, trained.config.features))
    training = current_names(trained.metadata)                           # as in the training
    differences = {
        key: {"training": training.get(key), "rollout": selected.get(key)}
        for key in sorted(set(training) | set(selected))
        if key not in _MACHINE_SPECIFIC_METADATA
        and training.get(key) != selected.get(key)}
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
            "dataset_db": _dataset_record(data_dir / "dataset.db"),
            "normalizer": _file_record(data_dir / "normalizer.json"),
            # "checkpoint", or for older checkpoints the normalizer.json used
            "normalizer_source": trained.normalizer_source,
            "split": args.split,
            "max_sequences": args.max_sequences,
            # the frames of the rollout: given, else those of the training (None: all)
            "start_frame": _frame_setting(args.start_frame, trained.config.start_frame),
            "end_frame": _frame_setting(args.end_frame, trained.config.end_frame),
            "max_steps": args.max_steps,
            "metadata": {key: value for key, value in data_metadata.items()
                         if not isinstance(value, (list, dict))},
            # Same export settings as the dataset the model was trained on?
            "matches_training_dataset": not differences,
            "differences_to_training_dataset": differences,
        },
        "settings": {"device": str(trained.device), "export_vtu": args.export_vtu,
                     "one_step": args.one_step, "config_file": args.config,
                     "evaluate": args.evaluate, "animate": args.animate},
        "environment": {"git": _git_state(), "host": platform.node(),
                        "python": platform.python_version(), "torch": torch.__version__,
                        "torch_geometric": torch_geometric.__version__},
    }


def _write_json(path: Path, content: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    """Command line options; the reference in docs/reference/cli.md is generated from them."""
    parser = argparse.ArgumentParser(
        description="Autoregressive rollout of a trained model over every trajectory of a "
                    "split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="TOML file with the settings of the rollout and of the evaluation "
                             "after it (docs/guides/rollout.md); options given here override it")
    parser.add_argument("--checkpoint", default=str(
        _REPO_ROOT / "checkpoints" / "example_run" / "best_model.pt"),
        help="Trainer checkpoint (.pt)")
    parser.add_argument("--data-dir", default=str(_REPO_ROOT / "data" / "deforming_plate_s10_small"),
                        help="dataset.db + normalizer.json — the dataset the checkpoint "
                             "was trained on")
    parser.add_argument("--output-dir", default=None,
                        help="new or empty folder (default: rollouts/<time>_<training run>_"
                             "<checkpoint>_<split>)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="split of the dataset to roll out")
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="roll out only the first N trajectories (default: all)")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="raw frame of the run the rollout starts from, its ground truth "
                             "(default: that of the training, else the first frame of the "
                             "dataset)")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="raw frame of the run the rollout ends with (default: that of the "
                             "training, else the last frame of the dataset)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="steps per trajectory (default: all)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="auto: the GPU if available")
    parser.add_argument("--export-vtu", action=argparse.BooleanOptionalAction, default=True,
                        help="write one .vtu per step (prediction / ground truth / error); "
                             "needs pyvista")
    parser.add_argument("--one-step", action=argparse.BooleanOptionalAction, default=True,
                        help="also compute the one-step rollout metrics RMSE-1 (a second "
                             "model call per step)")
    parser.add_argument("--lazy", action=argparse.BooleanOptionalAction, default=None,
                        help="read the frames from dataset.db on access instead of loading "
                             "the split (default: as in the training)")
    parser.add_argument("--evaluate", action=argparse.BooleanOptionalAction, default=False,
                        help="evaluate the run on this rollout right after it "
                             "(utils/evaluate_run.py)")
    parser.add_argument("--evaluation-output-dir", default=None,
                        help="with --evaluate: new or empty folder for the evaluation "
                             "(default: evaluations/<time>_<run>_evaluation)")
    parser.add_argument("--animate", action=argparse.BooleanOptionalAction, default=False,
                        help="render one trajectory of this rollout as an animated GIF right "
                             "after it (utils/animate_rollout.py); needs the .vtu export")
    parser.add_argument("--animation-trajectory", default=None,
                        help="with --animate: e.g. test_traj0136 (default: the trajectory at "
                             "the median of the position error)")
    parser.add_argument("--animation-output-dir", default=None,
                        help="with --animate: new or empty folder for the GIF "
                             "(default: evaluations/<time>_animation)")
    return parser


def read_rollout_config(path) -> Dict:
    """The parser defaults a rollout config sets; unknown keys are rejected."""
    flat = _flatten(toml.loads(Path(path).read_text(encoding="utf-8")))
    unknown = sorted(set(flat) - set(ROLLOUT_CONFIG_KEYS))
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}; these are known: "
                         f"{list(ROLLOUT_CONFIG_KEYS)}")
    return {ROLLOUT_CONFIG_KEYS[key]: value for key, value in flat.items()}


def parse_args(argv=None) -> argparse.Namespace:
    """The options; their defaults come from ``--config`` and the parser."""
    parser = build_parser()
    config_option = argparse.ArgumentParser(add_help=False)
    config_option.add_argument("--config")
    config_path = config_option.parse_known_args(argv)[0].config
    if config_path:
        try:
            defaults = read_rollout_config(config_path)
        except (OSError, ValueError) as error:
            parser.error(f"--config: {error}")
        for action in parser._actions:   # argparse checks only what is on the command line
            if action.choices and defaults.get(action.dest, action.default) not in action.choices:
                parser.error(f"--config: {action.dest} must be one of {list(action.choices)}")
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from train.trainer import describe_config_source, load_trained_model

    evaluate_run = None
    if args.evaluate:   # before hours of rollout: what the evaluation needs is there
        from utils.evaluate_run import main as evaluate_run
        if args.evaluation_output_dir:
            _check_output_dir(Path(args.evaluation_output_dir), "--evaluation-output-dir")
    animate_rollout = None
    if args.animate:   # likewise for the animation
        if not args.export_vtu:
            raise SystemExit("--animate renders the .vtu files: it needs --export-vtu")
        import pyvista  # noqa: F401

        from utils.animate_rollout import main as animate_rollout
        if args.animation_output_dir:
            _check_output_dir(Path(args.animation_output_dir), "--animation-output-dir")

    checkpoint, data_dir = Path(args.checkpoint), Path(args.data_dir)
    trained = load_trained_model(str(checkpoint), device=args.device, data_dir=str(data_dir))
    output_dir = (_check_output_dir(Path(args.output_dir)) if args.output_dir
                  else _new_output_dir(checkpoint, trained.epoch, args.split))
    _setup_logging(output_dir)
    logger.info(f"Rollout folder: {output_dir}")
    logger.info(f"Loaded {checkpoint} (epoch {trained.epoch}) on {trained.device}")
    logger.info(f"Training config: {describe_config_source(trained.config)}")

    info_path = output_dir / "rollout_info.json"
    info = _rollout_info(args, trained, checkpoint, data_dir)
    if not info["data"]["matches_training_dataset"]:
        logger.warning("dataset differs from the one the model was trained on in: "
                       f"{', '.join(info['data']['differences_to_training_dataset'])} "
                       f"(details in {info_path.name})")
    _write_json(info_path, info)

    start = time.time()
    try:
        runner = RolloutRunner.from_trained(trained, str(data_dir), output_dir, lazy=args.lazy)
        summary = runner.run(split=args.split, max_sequences=args.max_sequences,
                             max_steps=args.max_steps, export_vtu=args.export_vtu,
                             one_step=args.one_step, start_frame=info["data"]["start_frame"],
                             end_frame=info["data"]["end_frame"])
    except BaseException as error:
        info.update(status="failed", error=repr(error))
        raise
    else:
        position_rmse = summary["position_rmse_per_step"]
        info.update(status="finished", results={
            "summary_file": f"{args.split}_rollout_summary.json",
            "initial_frame": summary["initial_frame"],
            "sequences": list(summary["sequences"]),
            "rollout_metrics": summary["rollout_metrics"],
            "mean_mse_normalized": summary["mean_mse_normalized"],
            "mean_position_rmse": float(np.mean(position_rmse)) if position_rmse else None,
            "final_position_rmse": position_rmse[-1] if position_rmse else None,
            "position_rel_l2": summary["position_rel_l2"],
            "rmse_per_field": summary["rmse_per_field"],
        })
    finally:
        info.update(finished=now().isoformat(timespec="seconds"),
                    duration_seconds=round(time.time() - start, 1))
        _write_json(info_path, info)
        logger.info(f"Info: {info_path}")
    status = 0
    if evaluate_run is not None:
        status = evaluate_run([str(output_dir)] + (["--output-dir", args.evaluation_output_dir]
                                                   if args.evaluation_output_dir else []))
    if animate_rollout is not None:
        trajectory = args.animation_trajectory or median_trajectory(summary)
        gif = (["--output", str(Path(args.animation_output_dir) / f"rollout_{trajectory}.gif")]
               if args.animation_output_dir else [])
        status = animate_rollout([str(output_dir), "--trajectory", trajectory] + gif) or status
    return status


if __name__ == "__main__":
    raise SystemExit(main())
