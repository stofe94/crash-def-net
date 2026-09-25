#!/usr/bin/env python3
"""
compare_paper.py — Own rollout results next to the MGN-T paper.

Reads rollout folders written by ``predict/rollout.py`` (``rollouts/<...>/``)
and compares their paper metrics with Tables 4 and 5 of Iparraguirre et al.,
"MeshGraphNet-Transformer", arXiv:2601.23177 (deforming_plate). Writes a new
folder ``evaluations/<YYYYMMDD_HHMMSS>_paper_comparison/`` with

- ``01_absolute_errors.png``       RMSE-1 and RMSE-all of position q and stress
- ``02_relative_errors.png``       R-RMSE (%) of position q and stress
- ``03_error_per_step.png``        error over the rollout steps
- ``04_error_per_trajectory.png``  RMSE-all per trajectory, paper value as reference
- ``05_table_metrics.png``, ``05b_table_compact.png`` (only RMSE-all, with the
  multiple of the paper), ``06_table_conditions.png``
- ``comparison.md`` (tables, figures, notes), ``metrics.csv``, ``metrics.json``

Rollouts that are still running or failed are listed in the report but not
plotted. ``--exclude`` adds rows with the metrics recomputed without the given
trajectories (e.g. an outlier), pooled like the paper (weighted by node count)
from the per-trajectory values of the summary.

Usage
-----
    python utils/compare_paper.py                      # every rollout in rollouts/
    python utils/compare_paper.py rollouts/<a> rollouts/<b> --exclude test_traj0038
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.plot_losses import (  # noqa: E402
    AXIS,
    GRIDLINE,
    SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from utils.timezone import now  # noqa: E402

ROLLOUTS_DIR = _REPO_ROOT / "rollouts"
EVALUATIONS_DIR = _REPO_ROOT / "evaluations"

# Categorical slots of the reference palette, fixed order (see plot_losses.py);
# the paper rows are reference values and stay neutral.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
PAPER_COLOR = TEXT_MUTED
ARCHITECTURE_NAMES = {"mgn_t": "MGN-T", "meshgraphnet": "MGN"}
SELECTION_NAMES = {"mse": "rollout MSE (normalized)", "pos_rmse": "position error",
                   "pos_rmse_final": "position error at the last step"}

# ── paper values ─────────────────────────────────────────────────────────────

PAPER_SOURCE = ("Iparraguirre et al., MeshGraphNet-Transformer, arXiv:2601.23177, "
                "Tab. 4/5 (deforming_plate)")
# Table 5, MGN-T: (value, standard error over the test trajectories) in SI units
# (q in m, stress in Pa, R-RMSE as a fraction).
PAPER_MGN_T = {
    "q_rmse_1": (0.10e-3, 0.005e-3), "q_rmse_all": (3.16e-3, 0.25e-3),
    "stress_rmse_1": (3.87e3, 0.17e3), "stress_rmse_all": (5.83e3, 0.34e3),
    "q_r_rmse_1": (0.02e-2, 0.001e-2), "q_r_rmse_all": (0.63e-2, 0.05e-2),
    "stress_r_rmse_1": (2.5e-2, 0.13e-2), "stress_r_rmse_all": (3.61e-2, 0.21e-2),
}
# Table 4: position RMSE (x1e-3 m) of MGN-T and the baselines, no standard errors.
PAPER_TABLE_4 = [  # model, RMSE-1, RMSE-all, parameters
    ("MGN", 0.25, 15.1, "2.0M"), ("BSMS-GNN", 0.29, 16.0, "2.8M"),
    ("EvoMesh", 0.28, 12.9, "3.2M"), ("HCMT", None, 7.3, "2.5M"),
    ("M4GN", 0.26, 2.6, "2.0M (estimated)"), ("MGN-T", 0.10, 3.2, "0.5M")]
PAPER_CONDITIONS = {
    "Dataset": "DeepMind deforming_plate; number of trajectories not stated in the paper",
    "Rollout": "400 steps, stride 1",
    "Nodes per mesh": "~1,250 (Tab. 1)",
    "Parameters": "MGN-T 0.5 M, MGN 2.0 M (Tab. 2/4)",
    "Training": "MGN-T ~8 h on an RTX 4090",
    "Spread (±)": "standard error over the test trajectories",
}

# ── metrics and display units ────────────────────────────────────────────────

# (key, variable, horizon, relative)
ABSOLUTE = [("q_rmse_1", "q", "1"), ("q_rmse_all", "q", "all"),
            ("stress_rmse_1", "stress", "1"), ("stress_rmse_all", "stress", "all")]
RELATIVE = [("q_r_rmse_1", "q", "1"), ("q_r_rmse_all", "q", "all"),
            ("stress_r_rmse_1", "stress", "1"), ("stress_r_rmse_all", "stress", "all")]
ALL_KEYS = [key for key, _, _ in ABSOLUTE + RELATIVE]
VARIABLE_NAMES = {"q": "Position q", "stress": "Von Mises stress σ"}
HORIZON_NAMES = {"1": "RMSE-1 (one step)", "all": "RMSE-all (whole rollout)"}


def display(key: str) -> Tuple[float, str]:
    """(factor, unit) that turns an SI value of ``key`` into the table units."""
    if "_r_rmse" in key:
        return 100.0, "%"
    return (1e3, "mm") if key.startswith("q_") else (1e-3, "kPa")


def _significant(value: float, digits: int) -> str:
    """``digits`` significant digits, but no exponent and no lost integer digits."""
    if value == 0:
        return "0"
    decimals = max(digits - 1 - int(math.floor(math.log10(abs(value)))), 0)
    return f"{value:.{decimals}f}"


def fmt(value: Optional[float], se: Optional[float], key: str) -> str:
    """Value with 3 significant digits, standard error with 2."""
    if value is None:
        return "–"
    factor, _ = display(key)
    text = _significant(value * factor, 3)
    return text + (f" ± {_significant(se * factor, 2)}" if se is not None else "")


# ── rollouts ─────────────────────────────────────────────────────────────────

@dataclass
class Rollout:
    folder: Path
    info: Dict
    summary: Optional[Dict]

    @property
    def status(self) -> str:
        """"finished" only with a summary; else the status of rollout_info.json."""
        return "finished" if self.summary is not None else self.info.get("status", "unknown")

    @property
    def model(self) -> Dict:
        return self.info["model"]

    @property
    def name(self) -> str:
        return f"{ARCHITECTURE_NAMES.get(self.model['architecture'], self.model['architecture'])}" \
               f" Ep. {self.model['epoch']}"

    @property
    def split(self) -> str:
        return self.info["data"]["split"]


def load_rollout(folder: Path) -> Optional[Rollout]:
    info_path = folder / "rollout_info.json"
    if not info_path.exists():
        return None
    with open(info_path) as f:
        info = json.load(f)
    summary_name = (info.get("results") or {}).get(
        "summary_file", f"{info['data']['split']}_rollout_summary.json")
    summary = None
    if info.get("status") == "finished" and (folder / summary_name).exists():
        with open(folder / summary_name) as f:
            summary = json.load(f)
    return Rollout(folder, info, summary)


def node_counts(rollout: Rollout) -> Dict[str, int]:
    """Nodes per trajectory, the weights for pooling like the paper."""
    db = Path(rollout.info["data"]["dataset_db"]["absolute_path"])
    if not db.exists():
        return {}
    with sqlite3.connect(db) as connection:
        return dict(connection.execute("SELECT sequence_id, num_nodes FROM trajectories"))


def pooled_metrics(rollout: Rollout, exclude: Sequence[str]
                   ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Paper metrics of the rollout; without ``exclude`` taken from the summary,
    otherwise pooled from the per-trajectory values (weights: node count)."""
    paper = rollout.summary["paper_metrics"]
    sequences = rollout.summary["sequences"]
    kept = [sid for sid in sequences if sid not in exclude]
    if len(kept) == len(sequences):
        return {key: (paper.get(key), paper.get(f"{key}_se")) for key in ALL_KEYS}
    nodes = node_counts(rollout)
    out = {}
    for key in ALL_KEYS:
        values = [(sequences[sid]["paper_metrics"].get(key), nodes.get(sid, 1)) for sid in kept]
        values = [(v, w) for v, w in values if v is not None]
        if not values:
            out[key] = (None, None)
            continue
        v = np.array([v for v, _ in values])
        w = np.array([w for _, w in values], dtype=float)
        se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None
        out[key] = (float(math.sqrt((w * v ** 2).sum() / w.sum())), se)
    return out


# ── rows of the comparison ───────────────────────────────────────────────────

@dataclass
class Entry:
    label: str
    color: str
    values: Dict[str, Tuple[Optional[float], Optional[float]]]
    source: str                                    # "paper" | "own"
    rollout: Optional[Rollout] = None
    subset: Optional[str] = None
    reference: bool = False                        # the paper's MGN-T, the yardstick


def tint(color: str, amount: float = 0.5) -> str:
    """``color`` mixed with the chart surface: the same hue, lighter."""
    base, surface = np.array(to_rgb(color)), np.array(to_rgb(SURFACE))
    return to_hex(base + (surface - base) * amount)


def build_entries(rollouts: List[Rollout], exclude: Sequence[str]) -> List[Entry]:
    entries = [
        Entry("MGN-T · paper", PAPER_COLOR, dict(PAPER_MGN_T), "paper", reference=True),
        Entry("MGN · paper (Tab. 4)", PAPER_COLOR,
              {"q_rmse_1": (0.25e-3, None), "q_rmse_all": (15.1e-3, None)}, "paper"),
    ]
    names = [r.name for r in rollouts]
    for index, rollout in enumerate(rollouts):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        name = rollout.name if names.count(rollout.name) == 1 else \
            f"{rollout.name} ({rollout.folder.name[:15]})"
        n = len(rollout.summary["sequences"])
        entries.append(Entry(f"{name} · own ({n} traj.)", color,
                             pooled_metrics(rollout, ()), "own", rollout))
        dropped = [sid for sid in exclude if sid in rollout.summary["sequences"]]
        if dropped:
            entries.append(Entry(f"{name} · own, without {', '.join(dropped)}", tint(color),
                                 pooled_metrics(rollout, dropped), "own", rollout,
                                 subset=f"without {', '.join(dropped)}"))
    return entries


# ── figures ──────────────────────────────────────────────────────────────────

def _style(ax, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=grid_axis, color=GRIDLINE, linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=TEXT_MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def _figure_header(fig, title: str, subtitle: str) -> float:
    """Title and wrapped subtitle at the top; returns the top of the plot area."""
    width_inches, height_inches = fig.get_size_inches()
    lines = textwrap.wrap(subtitle, width=int(width_inches * 13.5))
    fig.text(0.01, 1 - 0.12 / height_inches, title, fontsize=15, color=TEXT_PRIMARY,
             va="top")
    fig.text(0.01, 1 - 0.48 / height_inches, "\n".join(lines), fontsize=9.5,
             color=TEXT_SECONDARY, va="top", linespacing=1.4)
    return 1 - (0.75 + 0.2 * len(lines)) / height_inches


def plot_bar_panels(entries: List[Entry], metrics, title: str, subtitle: str,
                    path: Path) -> None:
    """2 x 2 horizontal bars, one panel per metric; ± = standard error."""
    rows = len(entries)
    fig, axes = plt.subplots(2, 2, figsize=(14, 2.6 + 0.95 * rows), sharey=True,
                             facecolor=SURFACE)
    positions = np.arange(rows)[::-1]
    for ax, (key, variable, horizon) in zip(axes.flat, metrics):
        factor, unit = display(key)
        extents = [(v + (s or 0)) * factor for v, s in (e.values.get(key, (None, None))
                                                        for e in entries) if v is not None]
        limit = max(extents) * 1.45 if extents else 1.0
        for y, entry in zip(positions, entries):
            value, se = entry.values.get(key, (None, None))
            if value is None:
                ax.text(limit * 0.01, y, "not reported", va="center", fontsize=9,
                        color=TEXT_MUTED)
                continue
            ax.barh(y, value * factor, height=0.62, color=entry.color, edgecolor=SURFACE,
                    linewidth=2)
            if se is not None:
                ax.errorbar(value * factor, y, xerr=se * factor, fmt="none",
                            ecolor=TEXT_SECONDARY, elinewidth=1, capsize=3)
            ax.text((value + (se or 0)) * factor + limit * 0.015, y, fmt(value, se, key),
                    va="center", fontsize=9, color=TEXT_PRIMARY)
        ax.set_xlim(0, limit)
        prefix = "R-RMSE" if "_r_rmse" in key else "RMSE"
        ax.set_title(f"{VARIABLE_NAMES[variable]} — {prefix}-{horizon} [{unit}]",
                     loc="left", fontsize=11.5, color=TEXT_PRIMARY, pad=8)
        _style(ax)
    for ax in axes[:, 0]:
        ax.set_yticks(positions, [e.label for e in entries], fontsize=9.5, color=TEXT_PRIMARY)
    for ax in axes[:, 1]:
        ax.tick_params(axis="y", length=0)
    top = _figure_header(fig, title, subtitle)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot_steps(own: List[Entry], path: Path) -> None:
    """Error over the rollout steps: position (length of the error vector, loss
    nodes) and stress RMSE, each averaged over the trajectories of the rollout."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURFACE)
    panels = [("position_rmse_per_step", None, 1e3,
               "Position error per step [mm]"),
              ("rmse_per_field_per_step", "stress", 1e-3, "Stress, RMSE per step [kPa]")]
    for ax, (key, field_name, factor, label) in zip(axes, panels):
        for entry in own:
            series = entry.rollout.summary[key]
            series = series[field_name] if field_name else series
            steps = np.arange(1, len(series) + 1)
            values = np.array(series) * factor
            ax.plot(steps, values, color=entry.color, linewidth=2, label=entry.label,
                    solid_joinstyle="round", solid_capstyle="round")
            ax.text(steps[-1] + 0.6, values[-1], entry.label.split(" · ")[0], va="center",
                    fontsize=9, color=TEXT_PRIMARY)
        ax.set_xlim(0, len(steps) * 1.14)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Rollout step", fontsize=10, color=TEXT_SECONDARY)
        ax.set_title(label, loc="left", fontsize=11.5, color=TEXT_PRIMARY, pad=8)
        _style(ax, "both")
    legend = axes[0].legend(fontsize=9, frameon=False, loc="upper left")
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)
    top = _figure_header(fig, "Error over the rollout",
                         "Mean over all trajectories of the rollout. Position error = length "
                         "of the error vector per node (internal quantity, not the paper's "
                         "definition); the paper shows no curve per step.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot_trajectories(own: List[Entry], exclude: Sequence[str], path: Path) -> None:
    """RMSE-all per trajectory, grouped by trajectory; paper mean as reference line."""
    trajectories = sorted({sid for e in own for sid in e.rollout.summary["sequences"]})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=SURFACE)
    width = 0.8 / max(len(own), 1)
    x = np.arange(len(trajectories))
    for ax, key in zip(axes, ["q_rmse_all", "stress_rmse_all"]):
        factor, unit = display(key)
        for index, entry in enumerate(own):
            sequences = entry.rollout.summary["sequences"]
            values = [sequences[sid]["paper_metrics"][key] * factor if sid in sequences
                      else np.nan for sid in trajectories]
            offset = (index - (len(own) - 1) / 2) * width
            ax.bar(x + offset, values, width=width, color=entry.color, edgecolor=SURFACE,
                   linewidth=2, label=entry.label)
            for xi, value in zip(x + offset, values):
                if not np.isnan(value):
                    ax.text(xi, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8,
                            color=TEXT_PRIMARY)
        reference = PAPER_MGN_T[key][0] * factor
        ax.axhline(reference, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)),
                   label=f"MGN-T · paper, mean over its test set: "
                         f"{_significant(reference, 3)} {unit}")
        ax.set_xticks(x, [sid + (" *" if sid in exclude else "") for sid in trajectories],
                      fontsize=9, color=TEXT_PRIMARY)
        # Headroom for the legend above the tallest bar.
        tallest = max(e.rollout.summary["sequences"][sid]["paper_metrics"][key] * factor
                      for e in own for sid in e.rollout.summary["sequences"])
        ax.set_ylim(0, max(tallest, reference) * (1.12 + 0.07 * (len(own) + 1)))
        variable = "q" if key.startswith("q_") else "stress"
        ax.set_title(f"{VARIABLE_NAMES[variable]} — RMSE-all per trajectory [{unit}]",
                     loc="left", fontsize=11.5, color=TEXT_PRIMARY, pad=8)
        _style(ax, "y")
    for ax in axes:
        legend = ax.legend(fontsize=9, frameon=False, loc="upper left")
        for text in legend.get_texts():
            text.set_color(TEXT_PRIMARY)
    note = " * = left out of the additional rows with --exclude." if exclude else ""
    top = _figure_header(fig, "Error per trajectory",
                         "Whole rollout per test trajectory; dashed: the paper's mean for "
                         "MGN-T over its test set." + note)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot_table(columns: List[str], rows: List[List[str]], title: str, subtitle: str,
               path: Path, first_width: float = 0.3, numeric: bool = True) -> None:
    """A table as an image: header in bold, hairlines between rows, long cells
    wrapped; columns after the first right-aligned for numbers (``numeric``),
    else left-aligned."""
    width_inches = 15
    other = (1 - first_width) / max(len(columns) - 1, 1)
    widths = [first_width] + [other] * (len(columns) - 1)
    chars = [max(int(w * width_inches * 12.5), 8) for w in widths]   # ~9.5 pt text
    wrapped = [["\n".join(textwrap.wrap(str(text), width=c)) or "" for text, c in zip(row, chars)]
               for row in [columns] + rows]
    lines = [max(cell.count("\n") + 1 for cell in row) for row in wrapped]
    line_inches = 0.2
    table_height = sum(lines) * line_inches + 0.22 * len(lines)
    fig_height = table_height + 1.0
    fig, ax = plt.subplots(figsize=(width_inches, fig_height), facecolor=SURFACE)
    ax.axis("off")
    align = "right" if numeric else "left"
    table = ax.table(cellText=wrapped[1:], colLabels=wrapped[0], loc="upper left",
                     cellLoc=align, colWidths=widths, bbox=(0, 0, 1, 1))
    table.auto_set_font_size(False)
    for (row, column), cell in table.get_celld().items():
        cell.set_height(lines[row] * line_inches + 0.22)       # relative, scaled to bbox
        cell.set_facecolor(SURFACE)
        cell.set_edgecolor(GRIDLINE)
        cell.visible_edges = "B"
        cell.set_text_props(color=TEXT_PRIMARY, fontsize=9.5, linespacing=1.35,
                            ha="left" if column == 0 else align)
        cell.PAD = 0.02
        if row == 0:
            cell.set_text_props(weight="bold", color=TEXT_SECONDARY)
            cell.set_edgecolor(AXIS)
    fig.text(0.01, 1 - 0.1 / fig_height, title, fontsize=14, color=TEXT_PRIMARY, va="top")
    fig.text(0.01, 1 - 0.45 / fig_height, subtitle, fontsize=9, color=TEXT_SECONDARY,
             va="top")
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.8 / fig_height,
                        bottom=0.15 / fig_height)
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ── tables ───────────────────────────────────────────────────────────────────

METRIC_COLUMNS = [("q_rmse_1", "q RMSE-1 [mm]"), ("q_rmse_all", "q RMSE-all [mm]"),
                  ("stress_rmse_1", "σ RMSE-1 [kPa]"), ("stress_rmse_all", "σ RMSE-all [kPa]"),
                  ("q_r_rmse_all", "q R-RMSE-all [%]"),
                  ("stress_r_rmse_all", "σ R-RMSE-all [%]")]


def metric_rows(entries: List[Entry]) -> List[List[str]]:
    return [[e.label] + [fmt(*e.values.get(key, (None, None)), key) for key, _ in METRIC_COLUMNS]
            for e in entries]


COMPACT_COLUMNS = ["Model", "Rollout (steps × stride)", "q RMSE-all [mm]",
                   "× paper", "σ RMSE-all [kPa]", "× paper"]


def compact_rows(entries: List[Entry]) -> List[List[str]]:
    """Only the metrics that compare with the paper across step sizes: RMSE-all of
    position and stress, each with its multiple of the paper's MGN-T. RMSE-1 depends
    on the step size; R-RMSE of q is RMSE-all / ~0.5 m and adds nothing."""
    rows = []
    for e in entries:
        if e.rollout is None:
            rollout = "400 × 1"
        else:
            steps = len(e.rollout.summary["position_rmse_per_step"])
            rollout = f"{steps} × {e.rollout.info['data']['metadata'].get('time_stride', '?')}"
        row = [e.label, rollout]
        for key in ("q_rmse_all", "stress_rmse_all"):
            value, se = e.values.get(key, (None, None))
            row.append(fmt(value, se, key))
            row.append("reference" if e.reference else
                       f"× {value / PAPER_MGN_T[key][0]:.1f}" if value is not None else "–")
        rows.append(row)
    return rows


def ratio_rows(entries: List[Entry]) -> List[List[str]]:
    """Own results as a multiple of the paper's MGN-T (1.0 = on par)."""
    rows = []
    for e in (e for e in entries if e.source == "own"):
        row = [e.label]
        for key, _ in METRIC_COLUMNS:
            value, reference = e.values.get(key, (None, None))[0], PAPER_MGN_T[key][0]
            row.append(f"× {value / reference:.1f}" if value is not None else "–")
        rows.append(row)
    return rows


def training_summary(rollout: Rollout) -> Dict[str, str]:
    """Conditions of an own result: data, rollout, model, training."""
    info, model = rollout.info, rollout.model
    config = model["training_config"]
    metadata_path = Path(info["data"]["data_dir"]) / "metadata.json"
    if not metadata_path.is_absolute():
        metadata_path = _REPO_ROOT / metadata_path
    splits = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            splits = {k: v["count"] for k, v in json.load(f).get("splits", {}).items()}
    history_path = Path(model["checkpoint"]["absolute_path"]).parent / "history.json"
    trained = "unknown"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        minutes = round(sum(r.get("epoch_time_seconds", 0) for r in history) / 60)
        duration = f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else \
            f"{minutes} min"
        trained = f"{len(history)} of {config['epochs']} epochs, {duration}"
    meta = info["data"]["metadata"]
    steps = len(rollout.summary["position_rmse_per_step"])
    nodes = [n for sid, n in node_counts(rollout).items() if sid in rollout.summary["sequences"]]
    return {
        "Dataset": (f"{Path(info['data']['data_dir']).name} ({meta.get('source_tfrecord')})"
                    + ("; trajectories train/val/test "
                       + "/".join(str(splits.get(s, '?')) for s in ("train", "val", "test"))
                       if splits else "")),
        "Rollout": f"{steps} steps, stride {meta.get('time_stride')}, "
                   f"{len(rollout.summary['sequences'])} {rollout.split} trajectories",
        "Parameters": f"{model['parameters'] / 1e6:.2f} M",
        "Evaluated epoch": f"{model['epoch']} ({Path(model['checkpoint']['path']).name})",
        "Training": f"{trained}, batch {config['batch_size']}",
        "Model selection": SELECTION_NAMES.get(config.get("rollout_selection_metric", "mse"),
                                               config.get("rollout_selection_metric")),
        "Nodes per mesh": f"{min(nodes):,}–{max(nodes):,}" if nodes else "–",
    }


def condition_rows(own: List[Entry]) -> Tuple[List[str], List[List[str]]]:
    columns = ["", "MGN-T · paper"] + [e.label.split(" · ")[0] + " · own" for e in own]
    summaries = [training_summary(e.rollout) for e in own]
    keys = ["Dataset", "Rollout", "Parameters", "Evaluated epoch", "Training",
            "Model selection", "Nodes per mesh", "Spread (±)"]
    rows = []
    for key in keys:
        paper = PAPER_CONDITIONS.get(key, "–")
        own_values = [s.get(key, "as in the paper" if key == "Spread (±)" else "–")
                      for s in summaries]
        rows.append([key, paper] + own_values)
    return columns, rows


def markdown_table(columns: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def new_output_dir() -> Path:
    base = EVALUATIONS_DIR / f"{now():%Y%m%d_%H%M%S}_paper_comparison"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}_{n}")
    path.mkdir(parents=True)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare rollout results with the MGN-T paper (deforming_plate).")
    parser.add_argument("rollouts", nargs="*", type=Path,
                        help="rollout folders (default: every folder in rollouts/)")
    parser.add_argument("--split", default="test",
                        help="only rollouts of this split (default: test)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="trajectories for additional rows without them, "
                             "e.g. test_traj0038")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: evaluations/<time>_paper_comparison")
    args = parser.parse_args(argv)

    folders = args.rollouts or sorted(p for p in ROLLOUTS_DIR.iterdir() if p.is_dir())
    loaded = [r for r in (load_rollout(f) for f in folders) if r is not None]
    loaded = [r for r in loaded if r.split == args.split]
    finished = [r for r in loaded if r.status == "finished"]
    pending = [r for r in loaded if r.status != "finished"]
    for r in pending:
        print(f"Skipped ({r.status}): {r.folder.name}")
    if not finished:
        print(f"No finished rollouts found for split {args.split!r}.")
        return 1
    finished.sort(key=lambda r: (r.model["architecture"] != "mgn_t", r.model["epoch"]))
    datasets = {r.info["data"]["dataset_db"]["sha256"] for r in finished}
    if len(datasets) > 1:
        print("Warning: the rollouts come from different datasets — "
              "the values are not directly comparable.")

    out = args.output_dir or new_output_dir()
    out.mkdir(parents=True, exist_ok=True)
    entries = build_entries(finished, args.exclude)
    own = [e for e in entries if e.source == "own" and e.subset is None]
    split_note = f"own values: {args.split} split; ± = standard error over the trajectories"

    plot_bar_panels(entries, ABSOLUTE, "Absolute errors compared with the MGN-T paper",
                    f"{PAPER_SOURCE}; {split_note}. Paper: 400 steps, stride 1 — "
                    "own rollouts: see the conditions table.",
                    out / "01_absolute_errors.png")
    plot_bar_panels(entries, RELATIVE, "Relative errors (R-RMSE) compared with the MGN-T paper",
                    "RMSE divided by the infinity norm of the trajectory (largest |q| "
                    f"component or |σ|); {split_note}. MGN from Tab. 4 with absolute values "
                    "only.", out / "02_relative_errors.png")
    plot_steps(own, out / "03_error_per_step.png")
    plot_trajectories(own, args.exclude, out / "04_error_per_trajectory.png")

    metric_columns = ["Model"] + [label for _, label in METRIC_COLUMNS]
    plot_table(metric_columns, metric_rows(entries), "Metrics at a glance",
               f"{PAPER_SOURCE}; {split_note}.", out / "05_table_metrics.png")
    plot_table(COMPACT_COLUMNS, compact_rows(entries), "Summary",
               "RMSE-all over the whole rollout: same physical time span as in the paper, "
               "but fewer steps, so only partly comparable; × paper = multiple of the "
               f"MGN-T paper; {split_note}.",
               out / "05b_table_compact.png", first_width=0.3)
    condition_columns, conditions = condition_rows(own)
    plot_table(condition_columns, conditions, "Conditions",
               "Differences between the paper and the own runs that matter when comparing "
               "the numbers.", out / "06_table_conditions.png", first_width=0.14,
               numeric=False)

    table_4 = [[m, f"{r1:.2f}" if r1 is not None else "–", f"{ra:.1f}", p]
               for m, r1, ra, p in PAPER_TABLE_4]
    report = [
        f"# Comparison with the MGN-T paper ({now():%Y-%m-%d %H:%M})",
        "",
        f"Source: {PAPER_SOURCE}. Own rollouts ({args.split} split):",
        "",
        *[f"- `{r.folder.name}` — {r.name}" for r in finished],
        *([""] + [f"- not evaluated ({r.status}): `{r.folder.name}`" for r in pending]
          if pending else []),
        "",
        "## Summary",
        "",
        "RMSE-all of position q and stress σ over the whole rollout, with the multiple of "
        "the paper's MGN-T. RMSE-1 is left out here because it depends on the stride.",
        "",
        markdown_table(COMPACT_COLUMNS, compact_rows(entries)),
        "",
        "## Metrics",
        "",
        "q = node position, σ = von Mises stress. RMSE-1: one step from the ground-truth "
        "state; RMSE-all: whole rollout; R-RMSE: RMSE divided by the infinity norm of the "
        "trajectory. Averaged over all nodes, components, steps and trajectories; "
        "± = standard error over the trajectories.",
        "",
        markdown_table(metric_columns, metric_rows(entries)),
        "",
        "### Multiple of the paper's MGN-T (× 1.0 = on par)",
        "",
        markdown_table(["Model"] + [label.split(" [")[0] for _, label in METRIC_COLUMNS],
                       ratio_rows(entries)),
        "",
        "RMSE-1 is not comparable at a different stride: at stride 8 the nodes move about "
        "8 times as far per step as in the paper. RMSE-all instead covers the same physical "
        "time span, but with fewer rollout steps (less opportunity for error accumulation).",
        "",
        "### Table 4 of the paper (q, ×10⁻³ m)",
        "",
        markdown_table(["Model", "RMSE-1", "RMSE-all", "Parameters"], table_4),
        "",
        "## Conditions",
        "",
        markdown_table(condition_columns, conditions),
        "",
        "The numbers are only partly comparable: different stride and rollout length, far "
        "less training data and training time, a very small test set.",
        "",
        "## Figures",
        "",
        *[f"![{name}]({name})" for name in (
            "01_absolute_errors.png", "02_relative_errors.png", "03_error_per_step.png",
            "04_error_per_trajectory.png", "05_table_metrics.png", "05b_table_compact.png",
            "06_table_conditions.png")],
        "",
    ]
    (out / "comparison.md").write_text("\n".join(report), encoding="utf-8")

    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entry", "source", "rollout"] + [c for key in ALL_KEYS
                                                            for c in (key, f"{key}_se")])
        for e in entries:
            writer.writerow([e.label, e.source, e.rollout.folder.name if e.rollout else ""]
                            + [c for key in ALL_KEYS for c in e.values.get(key, (None, None))])
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"created": now().isoformat(timespec="seconds"), "paper": PAPER_SOURCE,
                   "split": args.split, "exclude": args.exclude,
                   "rollouts": [str(r.folder) for r in finished],
                   "not_evaluated": {str(r.folder): r.status for r in pending},
                   "units": "SI: q in m, stress in Pa, R-RMSE as fraction",
                   "entries": [{"label": e.label, "source": e.source, "subset": e.subset,
                                "rollout": str(e.rollout.folder) if e.rollout else None,
                                "values": {k: {"value": v, "se": s}
                                           for k, (v, s) in e.values.items()}}
                               for e in entries]}, f, indent=2, ensure_ascii=False)

    print(f"Comparison: {out}")
    for line in markdown_table(metric_columns, metric_rows(entries)).splitlines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
