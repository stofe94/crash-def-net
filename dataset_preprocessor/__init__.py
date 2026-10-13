"""
dataset_preprocessor — DeepMind ``deforming_plate`` data for CrashDefNet.

Modules
-------
graph_features.py
    Feature layout (names, one-hot node types) and graph construction (mesh +
    world edges), shared by export, dataset, training and rollout.
dataset.py
    ``DeformingPlateDataset`` over a compact ``dataset.db`` and the
    predecessor pairs used for scheduled sampling.
export.py
    CLI: TFRecord -> ``dataset.db`` + ``normalizer.json`` + ``metadata.json``
    (``python -m dataset_preprocessor.export``).
"""

from .dataset import DeformingPlateDataset, PredecessorPairDataset, collate_with_predecessor

__all__ = ["DeformingPlateDataset", "PredecessorPairDataset", "collate_with_predecessor"]
