"""
early_stopping.py — Model selection and early stopping by several criteria.

At every evaluation the trainer records the selection criteria, each to be as
low as possible (``CRITERIA``):

* ``val_loss``                single-step loss on the val split,
* ``rollout_mse``             normalized MSE of the val rollout,
* ``rollout_pos_rmse``        position error of the val rollout, mean over the steps,
* ``rollout_pos_rmse_final``  position error at the last rollout step.

The generalization gap (val loss / train loss without noise), which the trainer
records at every evaluation, is deliberately no criterion: its lowest value
belongs to early, undertrained evaluations, and a model that keeps learning
almost always widens it. As a criterion it pulls the selection back to early
evaluations and stops runs that still improve.

Every evaluation is scored by its worst criterion: the relative excess
``value / lowest value - 1`` over the lowest value of that criterion among all
evaluations so far. The selected model is the evaluation with the lowest score,
i.e. close to the best value in every criterion at the same time; a lucky
minimum of a single criterion does not win. With one criterion this is the
plain minimum of that criterion.

When a criterion reaches a new low, the earlier scores change and an earlier
evaluation can be selected again; the trainer therefore keeps the weights of
every evaluation until the run has ended.

Early stopping, only after at least ``min_epochs`` epochs: no criterion reached
a new low in the last ``patience`` evaluations (with rollout validation every N
epochs, N epochs each). The age of the selected evaluation is no trigger: on a
small val split the criteria scatter by several % per evaluation, so the
selected evaluation is the one that was lucky in every criterion at once, and a
model that still improves often needs longer than ``patience`` evaluations to
beat that luck with its worst criterion.

The gap is no trigger either. A stop on a growing gap while the val loss
reaches no new low fires earlier than this rule only while another criterion
still reaches new lows, i.e. while the model still improves on the val split.
The selection already guards against overfitting; training on only costs time.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

ROLLOUT_CRITERIA = ("rollout_mse", "rollout_pos_rmse", "rollout_pos_rmse_final")
CRITERIA = ("val_loss",) + ROLLOUT_CRITERIA


class EarlyStopping:
    """Select the model by ``criteria`` and decide when to stop (see module docstring)."""

    def __init__(self, criteria: Sequence[str] = CRITERIA, patience: int = 8,
                 min_epochs: int = 30, stop: bool = True):
        unknown = [name for name in criteria if name not in CRITERIA]
        if unknown or not criteria:
            raise ValueError(f"criteria must be a non-empty subset of {CRITERIA}, "
                             f"got {list(criteria)}")
        self.criteria = list(criteria)
        self.patience = patience
        self.min_epochs = min_epochs
        self.stop = stop
        self.epochs: List[int] = []                  # 0-based, one per evaluation
        self.values: List[Dict[str, float]] = []
        self.best_index = -1
        self.best_score = math.inf
        self.progress_index = -1                     # last evaluation with a new low
        self.should_stop = False
        self._stop_reason = ""

    @property
    def best_epoch(self) -> int:
        """0-based epoch of the selected evaluation (-1 before the first)."""
        return self.epochs[self.best_index] if self.values else -1

    def _lowest(self) -> Dict[str, float]:
        lowest = {}
        for name in self.criteria:
            finite = [v[name] for v in self.values if math.isfinite(v[name])]
            lowest[name] = min(finite) if finite else math.inf
        return lowest

    def _excess(self, value: float, lowest: float) -> float:
        """Relative excess over the lowest value; inf for a non-finite value."""
        if not math.isfinite(value) or not math.isfinite(lowest):
            return math.inf
        return value / max(lowest, 1e-12) - 1.0

    def scores(self) -> List[float]:
        """Current score of every evaluation (worst relative excess, 0 = best in all)."""
        lowest = self._lowest()
        return [max(self._excess(v[name], lowest[name]) for name in self.criteria)
                for v in self.values]

    def step(self, epoch: int, values: Dict[str, float]) -> int:
        """Record an evaluation after epoch ``epoch`` (0-based); returns the
        0-based epoch of the selected evaluation."""
        missing = [name for name in self.criteria if name not in values]
        if missing:
            raise ValueError(f"model selection needs {', '.join(missing)}")
        previous = self._lowest()
        self.epochs.append(epoch)
        self.values.append({name: float(values[name]) for name in self.criteria})
        if any(self.values[-1][name] < previous[name] for name in self.criteria):
            self.progress_index = len(self.values) - 1
        scores = self.scores()
        # Ties go to the later evaluation (trained longer at the same quality).
        self.best_index = min(range(len(scores)), key=lambda i: (scores[i], -i))
        self.best_score = scores[self.best_index]
        self.should_stop = (self.stop and epoch + 1 >= self.min_epochs
                            and self.evaluations_without_progress >= self.patience)
        self._stop_reason = (f"no criterion reached a new low in the last "
                             f"{self.evaluations_without_progress} evaluations"
                             if self.should_stop else "")
        return self.best_epoch

    @property
    def evaluations_without_progress(self) -> int:
        """Evaluations since the last one that set a new low in any criterion."""
        return len(self.values) - 1 - self.progress_index

    def dominated_by_latest(self) -> List[int]:
        """0-based epochs of earlier evaluations that the latest one matches or
        beats in every criterion. Their score can never be lower (ties go to the
        later one), so they cannot be selected again and need no weights."""
        latest = self.values[-1]
        return [self.epochs[i] for i, v in enumerate(self.values[:-1])
                if all(latest[name] <= v[name] for name in self.criteria)]

    def status(self) -> str:
        """One-line state for the epoch log."""
        if not self.values:
            return "no evaluation yet"
        text = f"selected epoch {self.best_epoch + 1}"
        if len(self.criteria) > 1:
            text += f" (score {100 * self.best_score:.1f} %)"
        if self.stop:
            text += (f", {self.evaluations_without_progress}/{self.patience} "
                     f"evaluations without a new low")
        return text

    def stop_reason(self) -> str:
        """Why ``should_stop`` became True."""
        return self._stop_reason

    def report(self) -> List[str]:
        """Lines for the end of training: every criterion of the selected
        evaluation next to its lowest value, so the choice can be checked."""
        if not self.values:
            return []
        lowest = self._lowest()
        chosen = self.values[self.best_index]
        lines = [f"Selected epoch {self.best_epoch + 1} of {len(self.values)} evaluations"
                 + (f", score {100 * self.best_score:.1f} %:" if len(self.criteria) > 1
                    else ":")]
        for name in self.criteria:
            best = min(range(len(self.values)),
                       key=lambda i: (self.values[i][name], -i))
            lines.append(f"    {name:24s} {chosen[name]:.4g} (lowest {lowest[name]:.4g} "
                         f"at epoch {self.epochs[best] + 1}, "
                         f"+{100 * self._excess(chosen[name], lowest[name]):.1f} %)")
        return lines
