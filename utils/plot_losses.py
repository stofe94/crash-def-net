"""
plot_losses.py — Loss curves of a training run.

Reads ``history.json`` from a checkpoint directory (one dict per epoch, written
by ``train/train.py``) and writes ``loss_curves.png`` next to it:

* train loss (with position noise / scheduled sampling, so it can sit above
  the validation curves),
* single-step validation loss,
* validation rollout MSE (if rollout validation was on),

all in normalized target units on one log axis. The selected model (last
epoch flagged ``is_best``) is marked on the curve it was selected by. The
values are also printed as a table.

Usage
-----
    python utils/plot_losses.py [--run-dir checkpoints/example_run]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = _REPO_ROOT / "checkpoints" / "example_run"

# Reference data-viz palette (light chart surface): categorical slots 1-3 and ink.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# (history key, legend label, color) in fixed categorical order.
SERIES = [
    ("train_loss", "Train loss", "#2a78d6"),
    ("val_loss", "Validation loss (single step)", "#eb6834"),
    ("val_rollout_mse", "Validation rollout MSE", "#1baf7a"),
]


def load_history(run_dir: Path) -> list:
    with open(run_dir / "history.json", "r") as f:
        return json.load(f)


def plot_loss_curves(run_dir: Path) -> Path:
    """Write ``<run_dir>/loss_curves.png``; returns its path."""
    history = load_history(run_dir)
    present = [s for s in SERIES if any(s[0] in record for record in history)]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for key, label, color in present:
        points = [(r["epoch"], r[key]) for r in history if r.get(key) is not None]
        ax.plot(*zip(*points), label=label, color=color, linewidth=1.5,
                solid_joinstyle="round", solid_capstyle="round")

    best = _selected_epoch(history, present)
    if best is not None:
        label, epoch, value, color = best
        ax.plot(epoch, value, marker="o", markersize=7, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=5)
        ax.annotate(f"{label}: epoch {epoch}, {value:.4f}", (epoch, value),
                    xytext=(8, 10), textcoords="offset points", fontsize=10,
                    color=TEXT_PRIMARY)

    ax.set_yscale("log")
    # Plain decimals instead of 3x10^0; label minor ticks only over short ranges.
    decimal = FuncFormatter(lambda value, _: f"{value:g}")
    ax.yaxis.set_major_formatter(decimal)
    low, high = ax.get_ylim()
    ax.yaxis.set_minor_formatter(decimal if high / low < 30 else NullFormatter())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Epoch", fontsize=11, color=TEXT_SECONDARY)
    ax.set_ylabel("Loss (normalized MSE, log scale)", fontsize=11, color=TEXT_SECONDARY)
    ax.set_title("Training and validation loss", fontsize=14, color=TEXT_PRIMARY,
                 loc="left", pad=22)
    ax.text(0, 1.02, _display_path(run_dir), transform=ax.transAxes, fontsize=9,
            color=TEXT_MUTED)
    ax.grid(True, which="major", color=GRIDLINE, linewidth=0.75)
    ax.tick_params(which="both", colors=TEXT_MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    if len(present) > 1:
        legend = ax.legend(fontsize=10, frameon=False)
        for text in legend.get_texts():
            text.set_color(TEXT_PRIMARY)
    fig.tight_layout()

    output_path = run_dir / "loss_curves.png"
    fig.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    _print_table(history, present)
    print(f"Plot saved to {output_path}")
    return output_path


def _selected_epoch(history: list, present: list):
    """(label, epoch, value, color) of the selected model on its selection curve.

    With rollout validation the model is selected by the rollout MSE, else by
    the single-step validation loss. The selected model is the last epoch
    flagged ``is_best``; histories written before that flag existed get the
    minimum of the selection curve instead.
    """
    keys = [key for key, _, _ in present]
    key = "val_rollout_mse" if "val_rollout_mse" in keys else "val_loss"
    if key not in keys:
        return None
    color = next(c for k, _, c in present if k == key)
    candidates = [r for r in history if r.get(key) is not None]
    if any("is_best" in r for r in history):
        best = [r for r in candidates if r.get("is_best")]
        if not best:
            return None
        return "selected model", best[-1]["epoch"], best[-1][key], color
    lowest = min(candidates, key=lambda r: r[key])
    return "lowest", lowest["epoch"], lowest[key], color


def _display_path(run_dir: Path) -> str:
    """``run_dir`` relative to the repository if it lies inside, else absolute."""
    run_dir = run_dir.resolve()
    try:
        return str(run_dir.relative_to(_REPO_ROOT))
    except ValueError:
        return str(run_dir)


def _print_table(history: list, present: list) -> None:
    keys = [key for key, _, _ in present]
    print("epoch  " + "  ".join(f"{key:>16s}" for key in keys))
    for record in history:
        values = "  ".join(f"{record[key]:16.6f}" if record.get(key) is not None
                           else f"{'-':>16s}" for key in keys)
        print(f"{record['epoch']:5d}  {values}{'  *' if record.get('is_best') else ''}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the loss curves of a training run.")
    parser.add_argument("--run-dir", "--base_path", dest="run_dir", type=Path,
                        default=DEFAULT_RUN_DIR,
                        help="checkpoint directory with history.json")
    plot_loss_curves(parser.parse_args().run_dir)
