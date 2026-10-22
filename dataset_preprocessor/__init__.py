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
tfrecord_to_d3plot.py
    CLI: TFRecord -> LS-DYNA runs (``d3plot`` + ``model.k``), the second raw
    format (``python -m dataset_preprocessor.tfrecord_to_d3plot``).
d3plot_reader.py
    LS-DYNA runs -> raw trajectories for ``export.py --raw-format d3plot``.
tfrecord_to_vtk.py
    CLI: TFRecord -> VTK time series, the third raw format
    (``python -m dataset_preprocessor.tfrecord_to_vtk``).
vtk_reader.py
    VTK time series (.vtu / .vtp / .vtk) -> raw trajectories for
    ``export.py --raw-format vtk``.
"""

from .dataset import DeformingPlateDataset, PredecessorPairDataset, collate_with_predecessor

__all__ = ["DeformingPlateDataset", "PredecessorPairDataset", "collate_with_predecessor"]
