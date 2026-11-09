# Scaling

The data layout targets crash data far beyond `deforming_plate`, such as large crash runs
and design studies with many runs, in a `dataset.db` of up to a few hundred GB without
extreme RAM. What a stage needs depends on one run or one sample, not on the size of the
dataset. More runs cost disk and training time, not memory.

## Export

Runs are read, written and folded into the statistics in blocks of frames, so a run need
not fit into memory, and a content fingerprint identifies the dataset without hashing the
whole file. The options for large exports:
[export a dataset](../guides/export.md#large-and-interrupted-exports).

## Loading during training

[`TrajectoryDataset`](https://github.com/stofe94/crash-def-net/blob/main/dataset_preprocessor/dataset.py)
builds one graph per time step when a sample is requested.

- **In memory (default).** The frames and the static graph data of a split are loaded
  into RAM once (~8 MB per trajectory of `deforming_plate` at 400 frames). DataLoader
  workers (`num_dataloader_workers`) share them through `/dev/shm`; with `deforming_plate`
  each worker costs only ~0.3 GB.
- **Lazy (`[data] lazy = true`).** The frames stay in the database. Each process opens
  its own read-only connection and reads only the 2–3 frames of a sample (optionally
  cached with `cache_frames`). The static graph data of a trajectory (mesh edges, world
  edge selection, rest positions) are built when it is first needed and kept up to
  `static_cache_mb` (default 1024) per process, least recently used first out. The speed
  is the same, and memory does not depend on the dataset size. Use this for splits that
  do not fit into RAM or `/dev/shm`.

Trajectories on the same mesh share its mesh edges and, while these keep their lengths,
its world edge selection in memory. With position noise the training batches get their
world edges rebuilt on the training device; the rollout rebuilds them on its own device.

## Memory of one sample

Micro-batches (`[training] micro_batch_size`) split a batch into several forward passes
with the same optimizer step, so only one micro-batch is in memory
([training](training.md)). Beyond that, the limit for large meshes is the GPU memory of
one sample: training keeps the activations of every message passing step, and their size
grows with the number of nodes and edges.

The contact edges matter here. With `world_edges = "all"` the number of world edges grows
with the mesh density, since near neighbours inside a body are linked as well (up to
250,000 edges and ~13.5 GiB of activations for one graph of the `deforming_plate`
protocol dataset with MeshGraphNets in paper size). `self_contact_aware` keeps only contact between bodies and
self-contact ([world edge variants](graphs.md#world-edge-variants-model-world_edges)).

## Outlook: full-vehicle meshes

The goal is to scale to full-vehicle meshes as soon as the data require it
([roadmap](../about/roadmap.md)).
