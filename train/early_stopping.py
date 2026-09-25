"""
early_stopping.py — Overfitting-aware early stopping and best-model selection.

Two signals are tracked per evaluation:

* **Stagnation**: the validation loss has not improved by ``min_delta`` for
  ``patience`` evaluations.
* **Inefficiency**: the generalization gap ``max(0, val - train)`` grows while
  the validation loss barely falls. Over ``efficiency_window`` evaluations

      efficiency = -Δ(smoothed val loss) / Δ(smoothed gap)

  is the validation gain per unit of added gap. It only counts as inefficient
  (< ``efficiency_threshold``) when the gap actually grew; ``patience``
  consecutive inefficient evaluations also stop training.

Neither signal stops before ``min_epochs``. The best epoch is the one with the
lowest composite score ``smoothed_val + gap_weight * smoothed_gap``.

``step`` is called once per *evaluation*. With rollout validation every N
epochs, ``patience``, ``smoothing_window`` and ``efficiency_window`` therefore
count evaluations (N epochs each); ``min_epochs`` always counts epochs.
"""

from __future__ import annotations

from typing import List


def _mean_of_last(values: List[float], window: int) -> float:
    recent = values[-window:]
    return sum(recent) / len(recent)


class EarlyStopping:
    """Decide per evaluation whether it is the best so far and whether to stop."""

    _SCORE_EPS = 1e-8   # a new best must beat the old score by more than this

    def __init__(
        self,
        patience: int = 30,
        min_delta: float = 0.0,
        min_epochs: int = 30,
        gap_weight: float = 0.5,
        smoothing_window: int = 3,
        efficiency_window: int = 30,
        efficiency_threshold: float = 0.1,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.gap_weight = gap_weight
        self.smoothing_window = max(1, smoothing_window)
        self.efficiency_window = max(2, efficiency_window)
        self.efficiency_threshold = efficiency_threshold

        self.val_losses: List[float] = []
        self.gaps: List[float] = []

        self.best_score = float("inf")
        self.best_epoch = -1                  # 0-based
        self.best_val_loss = float("inf")     # raw val loss of the best-score epoch

        self._lowest_val_loss = float("inf")  # drives the stagnation counter
        self.evaluations_without_improvement = 0
        self.inefficient_evaluations = 0
        self.last_efficiency = float("inf")
        self.should_stop = False

    def step(self, epoch: int, val_loss: float, train_loss: float) -> bool:
        """Record an evaluation after epoch ``epoch`` (0-based); True if it is the new best."""
        gap = max(0.0, val_loss - train_loss)
        self.val_losses.append(val_loss)
        self.gaps.append(gap)

        score = (_mean_of_last(self.val_losses, self.smoothing_window)
                 + self.gap_weight * _mean_of_last(self.gaps, self.smoothing_window))
        is_best = score < self.best_score - self._SCORE_EPS
        if is_best:
            self.best_score = score
            self.best_epoch = epoch
            self.best_val_loss = val_loss

        if val_loss < self._lowest_val_loss - self.min_delta:
            self._lowest_val_loss = val_loss
            self.evaluations_without_improvement = 0
        else:
            self.evaluations_without_improvement += 1

        self.last_efficiency = self._efficiency()
        if self.last_efficiency < self.efficiency_threshold:
            self.inefficient_evaluations += 1
        else:
            self.inefficient_evaluations = 0

        self.should_stop = epoch + 1 >= self.min_epochs and (
            self.evaluations_without_improvement >= self.patience
            or self.inefficient_evaluations >= self.patience)
        return is_best

    def _efficiency(self) -> float:
        """Validation gain per unit of gap growth over ``efficiency_window`` evaluations.

        ``inf`` (= efficient) while there is not enough history or the gap did
        not grow.
        """
        n = len(self.val_losses)
        past_end = n - self.efficiency_window
        if past_end <= 0:
            return float("inf")
        past_start = max(0, past_end - self.smoothing_window)
        past_val = self.val_losses[past_start:past_end]
        past_gap = self.gaps[past_start:past_end]

        val_change = (_mean_of_last(self.val_losses, self.smoothing_window)
                      - sum(past_val) / len(past_val))
        gap_change = (_mean_of_last(self.gaps, self.smoothing_window)
                      - sum(past_gap) / len(past_gap))
        if gap_change <= 0:
            return float("inf")
        return -val_change / (gap_change + self._SCORE_EPS)

    def status(self) -> str:
        """One-line state for the epoch log."""
        return (f"no improvement {self.evaluations_without_improvement}/{self.patience}, "
                f"inefficient {self.inefficient_evaluations}/{self.patience}, "
                f"best epoch {self.best_epoch + 1}")

    def stop_reason(self) -> str:
        """Why ``should_stop`` became True."""
        if self.evaluations_without_improvement >= self.patience:
            return (f"val loss did not improve by {self.min_delta} for "
                    f"{self.evaluations_without_improvement} evaluations")
        return (f"efficiency {self.last_efficiency:.3f} below "
                f"{self.efficiency_threshold} for {self.inefficient_evaluations} evaluations")
