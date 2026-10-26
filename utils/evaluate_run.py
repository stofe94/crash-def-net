#!/usr/bin/env python3
"""
evaluate_run.py — One training run evaluated on its own, without the paper.

Reads a checkpoint folder (``checkpoints/<run>/``) or a rollout folder written by
``predict/rollout.py`` (``rollouts/<...>/``) together with the checkpoint folder it
names, and writes a new folder ``evaluations/<YYYYMMDD_HHMMSS>_<run>_evaluation/``:

- ``01_training.png``        losses, val rollout and generalization gap over the
                             epochs, the selected epoch marked
- ``02_error_per_step.png``  position and stress error over the rollout: mean over
                             the trajectories, best and worst trajectory, and the RMS
                             displacement of the ground truth as the scale of the
                             position error (rollout folder only)
- ``03_trajectories.png``    every metric over the trajectories as all (one RMSE over
                             all trajectories together, as the paper metrics), median,
                             best and worst
- ``evaluation.md``          run, findings, tables, figures
- ``trajectories.csv``       every trajectory with its metrics
- ``evaluation.json``        the values of the report

The figures summarize the trajectories instead of showing each one; the tables
and the CSV list all of them. Without a rollout folder the test results come from
``test_metrics.json``, the rollout at the end of the training (paper metrics per
trajectory, no values per step). The displacement and the curves of single
trajectories need a rollout summary that stores them (``displacement_rms_per_step``
and the per-step values of every trajectory, written since this script exists).

The findings are measurements, not verdicts: how the training ended, the range of
the selection criterion over the last evaluations, the generalization gap, the
single-step test error against the val loss, the pooled error against the median
trajectory, trajectories far above the median, the error growth over the rollout
and the position error as a share of the displacement.

Usage
-----
    python utils/evaluate_run.py checkpoints/<run>
    python utils/evaluate_run.py rollouts/<rollout> [--output-dir <new folder>]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.config import TOML_KEYS  # noqa: E402
from utils.compare_paper import (  # noqa: E402
    ARCHITECTURE_NAMES,
    SERIES_COLORS,
    Rollout,
    _figure_header,
    _selection_description,
    _significant,
    _style,
    display,
    load_rollout,
    markdown_table,
    tint,
)
from utils.plot_losses import SURFACE, TEXT_MUTED, TEXT_PRIMARY, TEXT_SECONDARY  # noqa: E402
from utils.timezone import now  # noqa: E402

EVALUATIONS_DIR = _REPO_ROOT / "evaluations"

# Per-trajectory metrics (key, title): the paper metrics of the rollout and the
# position error as a share of the ground-truth displacement.
TO_DISPLACEMENT = "q_to_displacement"
METRICS = [
    ("q_rmse_1", "Position RMSE-1"),
    ("q_rmse_all", "Position RMSE-all"),
    (TO_DISPLACEMENT, "Position error / displacement"),
    ("stress_rmse_1", "Stress RMSE-1"),
    ("stress_rmse_all", "Stress RMSE-all"),
    ("stress_r_rmse_all", "Stress R-RMSE-all"),
]
# A trajectory this many times above the median is listed among the findings.
FAR_ABOVE_MEDIAN = 3.0
# Range of the selection criterion over this many last evaluations.
LAST_EVALUATIONS = 4

# History series: key, label, color (the first four as in plot_losses.py).
TRAIN, VAL, ROLLOUT_MSE, TRAIN_CLEAN = (
    ("train_loss", "Train loss (with noise)", SERIES_COLORS[0]),
    ("val_loss", "Val loss (single step)", SERIES_COLORS[1]),
    ("val_rollout_mse", "Val rollout MSE", SERIES_COLORS[2]),
    ("train_loss_clean", "Train loss without noise", SERIES_COLORS[3]))
POS_MEAN, POS_FINAL = (("val_rollout_pos_rmse", "mean over the steps", "#e87ba4"),
                       ("val_rollout_pos_rmse_final", "last step", "#008300"))
GAP_COLOR = "#6250d6"
# Error over the rollout: mean over the trajectories, best and worst trajectory.
MEAN_COLOR, BEST_COLOR, WORST_COLOR = SERIES_COLORS[0], SERIES_COLORS[2], SERIES_COLORS[1]


def unit(key: str) -> Tuple[float, str]:
    """(factor, unit) of a metric in the report."""
    return (100.0, "%") if key == TO_DISPLACEMENT else display(key)


def text(value: Optional[float], key: str, digits: int = 3) -> str:
    if value is None:
        return "–"
    return _significant(value * unit(key)[0], digits)


# ── the run ──────────────────────────────────────────────────────────────────

@dataclass
class Run:
    name: str
    run_dir: Path
    checkpoint_name: str
    checkpoint: Optional[Dict]        # epoch, config, normalizer, metadata (no weights)
    history: List[Dict]
    test: Optional[Dict]              # test_metrics.json of the training
    rollout: Optional[Rollout]
    model_line: Optional[str]         # "N parameters (...)" from the training log

    @property
    def config(self) -> Dict:
        if self.rollout is not None:
            return self.rollout.model["training_config"]
        return (self.checkpoint or {}).get("config") or {}

    @property
    def metadata(self) -> Dict:
        if self.rollout is not None:
            return self.rollout.info["data"]["metadata"]
        return (self.checkpoint or {}).get("metadata") or {}

    @property
    def split(self) -> str:
        return self.rollout.split if self.rollout is not None else "test"

    @property
    def source(self) -> str:
        if self.rollout is not None:
            return f"rollout `{self.rollout.folder.name}`"
        return "`test_metrics.json` (rollout at the end of the training)"

    @property
    def results(self) -> Optional[Dict]:
        """Paper metrics and trajectories: the rollout summary or the test rollout."""
        if self.rollout is not None:
            return self.rollout.summary
        return (self.test or {}).get("rollout")


def load_checkpoint(path: Path) -> Optional[Dict]:
    """Epoch, config, normalizer and dataset metadata of a checkpoint."""
    if not path.exists():
        return None
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {key: checkpoint.get(key) for key in ("epoch", "config", "normalizer", "metadata")}


def model_line(run_dir: Path) -> Optional[str]:
    """The model line of the newest training log: parameters and layout."""
    for log in sorted(run_dir.glob("train_*.log"), reverse=True):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.search(r"\| Model: (.*)$", line)
            if match:
                return match.group(1)
    return None


def load_run(folder: Path) -> Run:
    rollout = load_rollout(folder)
    if rollout is not None:
        if rollout.status != "finished":
            raise SystemExit(f"{folder}: rollout not finished ({rollout.status})")
        checkpoint_path = Path(rollout.model["checkpoint"]["absolute_path"])
    elif (folder / "history.json").exists():
        checkpoint_path = folder / "best_model.pt"
    else:
        raise SystemExit(f"{folder}: neither a rollout folder (rollout_info.json) nor a "
                         "checkpoint folder (history.json)")
    run_dir = checkpoint_path.parent

    def read(name: str):
        path = run_dir / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    # test_metrics.json belongs to the selected model, best_model.pt
    test = read("test_metrics.json") if checkpoint_path.name == "best_model.pt" else None
    run = Run(name=run_dir.name, run_dir=run_dir, checkpoint_name=checkpoint_path.name,
              checkpoint=load_checkpoint(checkpoint_path), history=read("history.json") or [],
              test=test, rollout=rollout, model_line=model_line(run_dir))
    if run.results is None:
        raise SystemExit(f"{folder}: no test results — neither a rollout nor a "
                         "test_metrics.json with a rollout")
    return run


# ── metrics over the trajectories ────────────────────────────────────────────

def to_displacement(errors: Optional[List[float]], scale: Optional[List[float]]
                    ) -> Optional[float]:
    """RMS of the position error over the RMS displacement, both over the steps."""
    if not errors or not scale:
        return None
    total = sum(s * s for s in scale)
    return math.sqrt(sum(e * e for e in errors) / total) if total else None


def trajectory_values(run: Run) -> Dict[str, Dict[str, Optional[float]]]:
    """The metrics of every trajectory, keys of ``METRICS``."""
    out = {}
    for sid, sequence in run.results["sequences"].items():
        # rollout summary: dict with paper_metrics; test_metrics.json: the paper metrics
        values = dict(sequence.get("paper_metrics", sequence))
        values[TO_DISPLACEMENT] = to_displacement(sequence.get("position_rmse_per_step"),
                                                  sequence.get("displacement_rms_per_step"))
        out[sid] = values
    return out


@dataclass
class Spread:
    pooled: Optional[float]
    se: Optional[float]
    median: Optional[float]
    best: Optional[Tuple[str, float]]
    worst: Optional[Tuple[str, float]]


def spreads(run: Run, values: Dict[str, Dict]) -> Dict[str, Spread]:
    """Every metric pooled over all trajectories (the paper metrics of the summary; the
    displacement share over all steps of all trajectories) with median, best, worst."""
    paper = run.results["paper_metrics"]
    sequences = run.results["sequences"]
    out = {}
    for key, _ in METRICS:
        per = {sid: v[key] for sid, v in values.items() if v.get(key) is not None}
        if key == TO_DISPLACEMENT:
            pooled = to_displacement(
                [e for s in sequences.values() for e in s.get("position_rmse_per_step", [])],
                [d for s in sequences.values() for d in s.get("displacement_rms_per_step", [])]
            ) if len(per) == len(sequences) else None
            se = None
        else:
            pooled, se = paper.get(key), paper.get(f"{key}_se")
        if not per:
            out[key] = Spread(pooled, se, None, None, None)
            continue
        ranked = sorted(per.items(), key=lambda item: item[1])
        out[key] = Spread(pooled, se, statistics.median(per.values()), ranked[0], ranked[-1])
    return out


# ── findings ─────────────────────────────────────────────────────────────────

def selected_record(history: List[Dict]) -> Optional[Dict]:
    """The epoch flagged ``selected``, in older histories the last ``is_best``."""
    for flag in ("selected", "is_best"):
        flagged = [r for r in history if r.get(flag)]
        if flagged:
            return flagged[-1]
    return None


def _rate(value: Optional[float]) -> str:
    return f"{value:.1e}".replace("e-0", "e-") if value is not None else "?"


def training_findings(run: Run) -> List[str]:
    history, config = run.history, run.config
    if not history:
        return ["No `history.json` next to the checkpoint: training not evaluated."]
    last = history[-1]
    epochs = config.get("epochs")
    end = last.get("end")
    lines = [{"epochs": f"The training ran to its last epoch ({last['epoch']}).",
              "early_stopping": f"Early stopping after epoch {last['epoch']} of {epochs}"
                                f" ({last.get('stop_reason', 'no reason recorded')})."
              }.get(end, f"No end recorded: the training is still running or was aborted "
                         f"(last epoch {last['epoch']} of {epochs}).")]
    selected = selected_record(history)
    evaluations = [r for r in history if r.get("val_rollout_mse") is not None]
    if selected is not None:
        lines.append(f"Selected: epoch {selected['epoch']}, among {len(evaluations) or len(history)}"
                     f" evaluations; criterion: {_selection_description(config)}.")
    if len(evaluations) >= 2 and selected is not None:
        tail = evaluations[-LAST_EVALUATIONS:]
        low, high = min(r["val_rollout_mse"] for r in tail), max(r["val_rollout_mse"] for r in tail)
        lines.append(
            f"Val rollout MSE of the last {len(tail)} evaluations (epochs {tail[0]['epoch']}–"
            f"{tail[-1]['epoch']}, learning rate {_rate(tail[0].get('learning_rate'))} to "
            f"{_rate(tail[-1].get('learning_rate'))}): {low:.3f} to {high:.3f}, the highest "
            f"{(high - low) / low * 100:.0f} % above the lowest"
            + (f"; selected epoch {selected['val_rollout_mse']:.3f}."
               if selected.get("val_rollout_mse") is not None else "."))
    gaps = [r for r in history if r.get("train_loss_clean") is not None]
    if gaps and selected is not None and selected.get("train_loss_clean") is not None:
        ratio = selected["val_loss"] / selected["train_loss_clean"]
        first, final = gaps[0], gaps[-1]
        lines.append(
            f"Generalization gap (val loss / train loss without noise) at the selected epoch: "
            f"× {ratio:.2f}; × {first['val_loss'] / first['train_loss_clean']:.2f} at epoch "
            f"{first['epoch']}, × {final['val_loss'] / final['train_loss_clean']:.2f} at "
            f"epoch {final['epoch']}.")
    if run.test is not None and selected is not None:
        test = run.test["mse_normalized"]
        lines.append(f"Single-step test MSE (normalized) {test:.3f} = × "
                     f"{test / selected['val_loss']:.2f} the val loss of the selected epoch "
                     f"({selected['val_loss']:.3f}).")
    return lines


def rollout_findings(run: Run, values: Dict[str, Dict], spread: Dict[str, Spread]) -> List[str]:
    lines = []
    for key, title in (("q_rmse_all", "Position RMSE-all"), ("stress_rmse_all", "Stress RMSE-all")):
        s = spread[key]
        if s.median is None:
            continue
        _, u = unit(key)
        far = sorted(((v[key], sid) for sid, v in values.items()
                      if v.get(key) is not None and v[key] > FAR_ABOVE_MEDIAN * s.median),
                     reverse=True)
        line = (f"{title}: {text(s.pooled, key)} {u} over all trajectories, median "
                f"{text(s.median, key)} {u} (× {s.pooled / s.median:.1f}).")
        if far:
            line += (f" Above {FAR_ABOVE_MEDIAN:g} × the median: "
                     + ", ".join(f"{sid} ({text(v, key)} {u})" for v, sid in far) + ".")
        lines.append(line)
    q1, qa = spread["q_rmse_1"].pooled, spread["q_rmse_all"].pooled
    steps = len(run.results.get("position_rmse_per_step", [])) if run.rollout else None
    if q1 and qa:
        lines.append(f"Error growth: position RMSE-all is × {qa / q1:.1f} the one-step error "
                     f"RMSE-1" + (f" over {steps} steps." if steps else "."))
    share = spread[TO_DISPLACEMENT]
    if share.pooled is not None:
        above = sorted(sid for sid, v in values.items()
                       if v.get(TO_DISPLACEMENT) is not None and v[TO_DISPLACEMENT] > 1)
        lines.append(
            f"Position error = {text(share.pooled, TO_DISPLACEMENT, 2)} % of the RMS "
            f"displacement of the ground truth (median trajectory "
            f"{text(share.median, TO_DISPLACEMENT, 2)} %, worst {share.worst[0]} "
            f"{text(share.worst[1], TO_DISPLACEMENT, 2)} %). "
            + (f"Above 100 %, i.e. larger than the error of a prediction in which nothing "
               f"moves: {', '.join(above)}." if above else
               "Every trajectory stays below 100 % (a prediction in which nothing moves)."))
    elif run.rollout is None:
        lines.append("Position error relative to the displacement and curves per step: "
                     "roll out the checkpoint with predict/rollout.py and evaluate the rollout.")
    return lines


# ── figures ──────────────────────────────────────────────────────────────────

def _legend(ax) -> None:
    legend = ax.legend(fontsize=9, frameon=False)
    for item in legend.get_texts():
        item.set_color(TEXT_PRIMARY)


def _title(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11.5, color=TEXT_PRIMARY, pad=8)


def _epoch_axis(ax, selected: Optional[Dict]) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch", fontsize=10, color=TEXT_SECONDARY)
    if selected is not None:
        ax.axvline(selected["epoch"], color=TEXT_MUTED, linewidth=0.75, zorder=1)
    _style(ax, "both")


def _log_axis(ax) -> None:
    ax.set_yscale("log")
    decimal = FuncFormatter(lambda value, _: f"{value:g}")
    ax.yaxis.set_major_formatter(decimal)
    low, high = ax.get_ylim()
    ax.yaxis.set_minor_formatter(decimal if high / low < 30 else NullFormatter())


def _line(ax, history: List[Dict], key: str, label: str, color: str, factor: float = 1.0,
          markers: bool = False) -> None:
    points = [(r["epoch"], r[key] * factor) for r in history if r.get(key) is not None]
    if points:
        ax.plot(*zip(*points), label=label, color=color, linewidth=2,
                marker="o" if markers else None, markersize=4, solid_joinstyle="round",
                solid_capstyle="round")


def plot_training(run: Run, path: Path) -> None:
    history = run.history
    selected = selected_record(history)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.4), facecolor=SURFACE)
    (losses, rollout), (position, gap) = axes

    for key, label, color in (TRAIN, VAL, TRAIN_CLEAN):
        _line(losses, history, key, label, color, markers=key == TRAIN_CLEAN[0])
    _log_axis(losses)
    _title(losses, "Loss (normalized MSE, log scale)")
    _legend(losses)

    evaluations = [r for r in history if r.get("val_rollout_mse") is not None]
    for ax in (rollout, position):
        if not evaluations:
            ax.text(0.5, 0.5, "no rollout validation", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color=TEXT_MUTED)
    _line(rollout, evaluations, ROLLOUT_MSE[0], ROLLOUT_MSE[1], ROLLOUT_MSE[2], markers=True)
    if selected is not None and selected.get("val_rollout_mse") is not None:
        value = selected["val_rollout_mse"]
        rollout.plot(selected["epoch"], value, marker="o", markersize=9, color=ROLLOUT_MSE[2],
                     markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", zorder=5)
        rollout.annotate(f"selected: epoch {selected['epoch']}, {value:.3f}",
                         (selected["epoch"], value), xytext=(-10, 12), ha="right",
                         textcoords="offset points", fontsize=9.5, color=TEXT_PRIMARY)
    if evaluations:
        _log_axis(rollout)
    _title(rollout, "Val rollout MSE (normalized, log scale)")

    for key, label, color in (POS_MEAN, POS_FINAL):
        _line(position, evaluations, key, label, color, factor=1e3, markers=True)
    if evaluations:
        position.set_ylim(bottom=0)
        _legend(position)
    _title(position, "Val rollout position error [mm]")

    gaps = [(r["epoch"], r["val_loss"] / r["train_loss_clean"]) for r in history
            if r.get("train_loss_clean")]
    if gaps:
        gap.plot(*zip(*gaps), color=GAP_COLOR, linewidth=2, marker="o", markersize=4)
        gap.axhline(1.0, color=TEXT_MUTED, linewidth=0.75)
    else:
        gap.text(0.5, 0.5, "no train loss without noise recorded", ha="center",
                 va="center", transform=gap.transAxes, fontsize=10, color=TEXT_MUTED)
    _title(gap, "Generalization gap: val loss / train loss without noise")

    for ax in axes.flat:
        _epoch_axis(ax, selected)
    subtitle = (f"{run.name}. Vertical line: selected epoch"
                + (f" {selected['epoch']}" if selected else "") + ". Train loss with the "
                "training noise; without noise and the val rollout at every evaluation.")
    top = _figure_header(fig, "Training", subtitle)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot_steps(run: Run, spread: Dict[str, Spread], path: Path) -> None:
    """Error over the rollout: mean, best and worst trajectory (by RMSE-all)."""
    summary = run.rollout.summary
    sequences = summary["sequences"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURFACE)
    panels = [(axes[0], "q_rmse_all", 1e3, "Position error per step [mm]",
               lambda s: s.get("position_rmse_per_step")),
              (axes[1], "stress_rmse_all", 1e-3, "Stress RMSE per step [kPa]",
               lambda s: (s.get("rmse_per_field_per_step") or {}).get("stress"))]
    single = False
    for ax, key, factor, title, series in panels:
        curves = [("mean over the trajectories", series(summary), MEAN_COLOR, 2.4)]
        for name, pick, color in (("best", spread[key].best, BEST_COLOR),
                                  ("worst", spread[key].worst, WORST_COLOR)):
            values = series(sequences[pick[0]]) if pick else None
            if values:
                curves.append((f"{name}: {pick[0]}", values, color, 1.6))
                single = True
        if key == "q_rmse_all" and summary.get("displacement_rms_per_step"):
            curves.append(("RMS displacement of the ground truth (mean)",
                           summary["displacement_rms_per_step"], TEXT_MUTED, 1.4))
        for label, values, color, width in curves:
            steps = range(1, len(values) + 1)
            ax.plot(steps, [v * factor for v in values], color=color, linewidth=width,
                    label=label, linestyle=(0, (4, 3)) if color == TEXT_MUTED else "-",
                    solid_joinstyle="round", solid_capstyle="round")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Rollout step", fontsize=10, color=TEXT_SECONDARY)
        _title(ax, title)
        _style(ax, "both")
        _legend(ax)
    note = ("Best and worst trajectory by RMSE-all of the quantity." if single else
            "This rollout stores no curves per trajectory: only the mean.")
    top = _figure_header(fig, "Error over the rollout",
                         f"{run.split} split, {len(sequences)} trajectories. Position error = "
                         "length of the error vector per node (nodes of the loss), RMS over "
                         "the nodes. " + note)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


ROWS = ["All (pooled)", "Median", "Best", "Worst"]


def plot_trajectories(run: Run, spread: Dict[str, Spread], path: Path) -> None:
    """Per metric: all trajectories pooled, median, best and worst."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.6), facecolor=SURFACE, sharey=True)
    positions = list(range(len(ROWS)))[::-1]
    for ax, (key, title) in zip(axes.flat, METRICS):
        s = spread[key]
        factor, u = unit(key)
        _title(ax, f"{title} [{u}]")
        _style(ax)
        if s.median is None:
            ax.text(0.02, 0.5, "needs a rollout that stores the displacement" if
                    key == TO_DISPLACEMENT else "not computed", transform=ax.transAxes,
                    fontsize=9.5, color=TEXT_MUTED, va="center")
            ax.set_xticks([])
            ax.spines["left"].set_visible(False)
            continue
        bars = [(s.pooled, s.se, ""), (s.median, None, ""),
                (s.best[1], None, s.best[0]), (s.worst[1], None, s.worst[0])]
        limit = max((v + (e or 0)) * factor for v, e, _ in bars if v is not None) * 1.6
        for y, (value, se, sid), color in zip(positions, bars,
                                              [MEAN_COLOR] + [tint(MEAN_COLOR, 0.45)] * 3):
            if value is None:
                ax.text(limit * 0.01, y, "–", va="center", fontsize=9, color=TEXT_MUTED)
                continue
            ax.barh(y, value * factor, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
            if se is not None:
                ax.errorbar(value * factor, y, xerr=se * factor, fmt="none",
                            ecolor=TEXT_SECONDARY, elinewidth=1, capsize=3)
            label = text(value, key) + (f" ± {text(se, key, 2)}" if se is not None else "")
            ax.text((value + (se or 0)) * factor + limit * 0.015, y,
                    label + (f"  {sid}" if sid else ""), va="center", fontsize=9,
                    color=TEXT_PRIMARY)
        ax.set_xlim(0, limit)
    for ax in axes[:, 0]:
        ax.set_yticks(positions, ROWS, fontsize=9.5, color=TEXT_PRIMARY)
    top = _figure_header(
        fig, "Metrics over the trajectories",
        f"{run.split} split, {len(run.results['sequences'])} trajectories, from "
        f"{run.source.replace('`', '')}. All (pooled) = one RMSE over all trajectories "
        "together: the squared errors of all nodes, steps and trajectories averaged, then the "
        "root, as in the papers; ± standard error over the trajectories. Median, best and "
        "worst: the RMSE of single trajectories. RMSE-1: one step from the "
        "ground truth; RMSE-all: whole rollout; R-RMSE: divided by the largest |σ| of the "
        "trajectory; displacement: RMS of the ground-truth motion since step 1 (the last exact positions of the rollout).")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ── report ───────────────────────────────────────────────────────────────────

def run_rows(run: Run) -> List[List[str]]:
    config, metadata, history = run.config, run.metadata, run.history
    architecture = config.get("architecture", "?")
    name = ARCHITECTURE_NAMES.get(architecture, architecture)
    parameters = (f"{run.rollout.model['parameters']:,} parameters" if run.rollout is not None
                  else (run.model_line or "").split(" (")[0])
    settings = [f"{key.rsplit('.', 1)[1]} {config[field]}" for key, field in TOML_KEYS.items()
                if key.startswith(f"model.{architecture}.") and field in config]
    world = config.get("world_edges", "?")
    if world == "self_contact_aware":
        world += f" (≥ {config.get('world_edge_exclude_distance')} r_W through the mesh)"
    splits = metadata.get("splits") or {}
    minutes = round(sum(r.get("epoch_time_seconds", 0) for r in history) / 60)
    duration = f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"
    selected = selected_record(history)
    epoch = (run.rollout.model["epoch"] if run.rollout is not None
             else (run.checkpoint or {}).get("epoch"))
    steps = (f", {len(run.results['position_rmse_per_step'])} steps each"
             if run.rollout is not None else "")
    rows = [
        ["Run", f"`{run.run_dir}`"],
        ["Model", f"{name}, {parameters}" if parameters else name],
        ["Settings", ", ".join(settings) or "–"],
        ["World edges", world],
        ["Dataset", f"{metadata.get('dataset', '?')}: "
                    + "/".join(str((splits.get(s) or {}).get("count", "?"))
                               for s in ("train", "val", "test"))
                    + f" trajectories (train/val/test), {metadata.get('num_steps', '?')} frames"
                    f" at stride {metadata.get('time_stride', '?')}"],
        ["Training", (f"{len(history)} of {config.get('epochs', '?')} epochs, {duration}; "
                      if history else "")
         + f"batch {config.get('batch_size', '?')}, learning rate "
           f"{_rate(config.get('learning_rate'))}"
           + (f" → {_rate(config.get('min_learning_rate'))} ({config['scheduler']})"
              if config.get("scheduler") else "")
           + f", position noise {config.get('world_noise_std', '?')}"],
        *([["Peak memory", memory_text(history[-1]["peak_memory"])]]
          if history and history[-1].get("peak_memory") else []),
        ["Model selection", _selection_description(config)
         + (f", every {config.get('rollout_validation_every')} epochs"
            if config.get("rollout_validation") else "")
         + (f"; selected epoch {selected['epoch']}" if selected else "")],
        ["Evaluated", f"{run.checkpoint_name} (epoch {epoch}); {run.split} split, "
                      f"{len(run.results['sequences'])} trajectories{steps}; {run.source}"],
    ]
    return rows


def memory_text(peak: Dict) -> str:
    """Peak memory recorded by the trainer (GiB), with the machine."""
    text = f"RAM {peak['ram_main']} GiB main process"
    if "ram_worker" in peak:
        text += f", {peak['ram_worker']} GiB largest DataLoader worker"
    if "gpu_reserved" in peak:
        text += f"; GPU {peak['gpu_allocated']} GiB allocated, {peak['gpu_reserved']} GiB reserved"
    return text + (f" ({peak['machine']})" if peak.get("machine") else "")


def training_rows(history: List[Dict]) -> List[List[str]]:
    selected = selected_record(history)
    rows = []
    for r in (r for r in history if r.get("val_rollout_mse") is not None
              or r.get("train_loss_clean") is not None):
        def number(key, factor=1.0, digits=4):
            return f"{r[key] * factor:.{digits}f}" if r.get(key) is not None else "–"
        gap = (f"× {r['val_loss'] / r['train_loss_clean']:.2f}" if r.get("train_loss_clean")
               else "–")
        rows.append([str(r["epoch"]) + (" ◀ selected" if r is selected else ""),
                     number("train_loss"), number("train_loss_clean"), number("val_loss"), gap,
                     number("val_rollout_mse"), number("val_rollout_pos_rmse", 1e3, 2),
                     number("val_rollout_pos_rmse_final", 1e3, 2)])
    return rows


TRAINING_COLUMNS = ["Epoch", "Train (noise)", "Train (no noise)", "Val", "Gap",
                    "Val rollout MSE", "Val pos. error [mm]", "… last step [mm]"]


def single_step_rows(run: Run) -> List[List[str]]:
    """Single-step test errors per target, also over the spread (std) of the target."""
    std = None
    normalizer = (run.checkpoint or {}).get("normalizer")
    if normalizer:
        std = normalizer["stats"]["target"]["std"]
    rows = []
    for i, (name, rmse) in enumerate(run.test["rmse_per_field"].items()):
        factor, u = (1e-3, "kPa") if name == "stress" else (1e3, "mm per frame")
        rows.append([name, f"{_significant(rmse * factor, 3)} {u}",
                     f"{_significant(run.test['mae_per_field'][name] * factor, 3)} {u}",
                     f"{rmse / std[i]:.2f}" if std else "–"])
    return rows


def spread_rows(spread: Dict[str, Spread]) -> List[List[str]]:
    rows = []
    for key, title in METRICS:
        s = spread[key]
        _, u = unit(key)
        rows.append([f"{title} [{u}]",
                     text(s.pooled, key) + (f" ± {text(s.se, key, 2)}" if s.se else ""),
                     text(s.median, key),
                     f"{text(s.best[1], key)} ({s.best[0]})" if s.best else "–",
                     f"{text(s.worst[1], key)} ({s.worst[0]})" if s.worst else "–"])
    return rows


def trajectory_rows(values: Dict[str, Dict], spread: Dict[str, Spread]) -> List[List[str]]:
    """Every trajectory, worst position RMSE-all first."""
    order = sorted(values, key=lambda sid: -(values[sid].get("q_rmse_all") or 0))
    rows = []
    for sid in order:
        far = [title for key, title in METRICS if spread[key].median
               and (values[sid].get(key) or 0) > FAR_ABOVE_MEDIAN * spread[key].median]
        rows.append([sid] + [text(values[sid].get(key), key) for key, _ in METRICS]
                    + [", ".join(far) or ""])
    return rows


def new_output_dir(run: Run) -> Path:
    base = EVALUATIONS_DIR / f"{now():%Y%m%d_%H%M%S}_{run.name}_evaluation"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}_{n}")
    path.mkdir(parents=True)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one training run on its own: training, single step, rollout.")
    parser.add_argument("folder", type=Path,
                        help="checkpoint folder (checkpoints/<run>) or rollout folder "
                             "(rollouts/<rollout>)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="new or empty folder (default: evaluations/<time>_<run>_evaluation)")
    args = parser.parse_args(argv)

    run = load_run(args.folder)
    if args.output_dir is not None:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise SystemExit(f"{args.output_dir} is not empty — evaluations never overwrite")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir or new_output_dir(run)

    values = trajectory_values(run)
    spread = spreads(run, values)
    findings = training_findings(run) + rollout_findings(run, values, spread)

    figures = []
    if run.history:
        plot_training(run, out / "01_training.png")
        figures.append("01_training.png")
    if run.rollout is not None:
        plot_steps(run, spread, out / "02_error_per_step.png")
        figures.append("02_error_per_step.png")
    plot_trajectories(run, spread, out / "03_trajectories.png")
    figures.append("03_trajectories.png")

    trajectory_columns = (["Trajectory"] + [f"{title} [{unit(key)[1]}]" for key, title in METRICS]
                          + [f"> {FAR_ABOVE_MEDIAN:g} × median"])
    report = [
        f"# Evaluation of {run.name} ({now():%Y-%m-%d %H:%M})",
        "",
        markdown_table(["", ""], run_rows(run)),
        "",
        "## Findings",
        "",
        *[f"- {line}" for line in findings],
        "",
    ]
    if run.history:
        report += ["## Training", "", "![training](01_training.png)", "",
                   "Evaluations (normalized MSE; gap = val / train without noise):", "",
                   markdown_table(TRAINING_COLUMNS, training_rows(run.history)), ""]
    if run.test is not None:
        report += ["## Single step (test split)", "",
                   "From the ground-truth state, nodes of the loss; RMSE / std = RMSE over the "
                   "standard deviation of the target in the train split (1 = as large as the "
                   "spread of the target itself).", "",
                   markdown_table(["Target", "RMSE", "MAE", "RMSE / std"], single_step_rows(run)),
                   ""]
    report += [f"## Rollout ({run.split} split)", "",
               f"Source: {run.source}. All (pooled) = one RMSE over all trajectories "
               "together: the squared errors of all nodes, steps and trajectories averaged, "
               "then the root, as in the papers (± standard error over the trajectories); a "
               "trajectory with a large error weighs more than in the median. "
               "Median, best and worst: the RMSE of single trajectories.", "",
               markdown_table(["Metric", "All (pooled)", "Median", "Best", "Worst"],
                              spread_rows(spread)), ""]
    if run.rollout is not None:
        report += ["![error per step](02_error_per_step.png)", ""]
    report += ["![trajectories](03_trajectories.png)", "",
               "### Every trajectory", "",
               "Sorted by position RMSE-all, worst first.", "",
               markdown_table(trajectory_columns, trajectory_rows(values, spread)), ""]
    (out / "evaluation.md").write_text("\n".join(report), encoding="utf-8")

    with open(out / "trajectories.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trajectory"] + [key for key, _ in METRICS])
        for sid, v in values.items():
            writer.writerow([sid] + [v.get(key) for key, _ in METRICS])
    with open(out / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump({"created": now().isoformat(timespec="seconds"), "run": str(run.run_dir),
                   "checkpoint": run.checkpoint_name,
                   "rollout": str(run.rollout.folder) if run.rollout else None,
                   "split": run.split, "units": "SI: q in m, stress in Pa, shares as fraction",
                   "findings": findings,
                   "metrics": {key: {"pooled": s.pooled, "se": s.se, "median": s.median,
                                     "best": s.best, "worst": s.worst}
                               for key, s in spread.items()},
                   "trajectories": values}, f, indent=2, ensure_ascii=False)

    print(f"Evaluation: {out}")
    for line in findings:
        print(f"- {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
