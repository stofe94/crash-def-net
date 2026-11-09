"""
results.py — Read the JSON files of rollouts and trainings of any version.

Files written up to v0.4.0 name the rollout metrics ``paper_metrics``. Every reader of
these files passes ``current_metric_names`` to ``json.load``, so that it only knows the
current name. Standard library only: the app imports it as well.
"""

from __future__ import annotations

from typing import Dict


def current_metric_names(entries: Dict) -> Dict:
    """``object_hook`` of ``json.load``: ``paper_metrics`` as ``rollout_metrics``, at any depth."""
    if "paper_metrics" in entries:
        entries["rollout_metrics"] = entries.pop("paper_metrics")
    return entries
