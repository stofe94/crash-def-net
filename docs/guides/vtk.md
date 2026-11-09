# VTK time series

The exporter reads VTK time series as a third raw format: one file per state, the
ParaView convention, which the rollout writes as well. Unlike d3plot, VTK keeps the
stress per node, so a dataset exported from files converted from the benchmark is
**identical** to the TFRecord export. Needs `pyvista`.

!!! warning "Early stage of development"
    The reader was developed and checked on runs converted from `deforming_plate` and on
    a few small example files; it has not been used on real crash simulations yet ([verification status](../concepts/graphs.md#general-meshes-and-run-parameters)).

## Convert the benchmark

[`tfrecord_to_vtk.py`](https://github.com/stofe94/crash-def-net/blob/main/dataset_preprocessor/tfrecord_to_vtk.py)
writes every trajectory of a TFRecord with all frames (all 100 of `test.tfrecord`: 2.3 GB,
~4 min; `--format vtk` writes uncompressed legacy files, ~3× larger):

```bash
python -m dataset_preprocessor.tfrecord_to_vtk [--source-tfrecord test.tfrecord] [--count N] [--format vtu|vtk] [--no-overwrite]
```

```
datasets/deforming_plate/raw_vtk/
  meta.json                         collision_radius, trajectory_length (as DeepMind's)
  test/conversion.json              source and representation
  test/traj0000/traj0000.vtu.series order and times (ParaView opens the series)
  test/traj0000/t0000.vtu …         one state per file: points = world_pos, tetrahedra,
                                    point data node_type (0 / 1 / 3) and stress
```

## Export

As for the TFRecord, with `--raw-format vtk` (raw directory default:
`datasets/deforming_plate/raw_vtk`):

```bash
python -m dataset_preprocessor.export --raw-format vtk --out-dir data/deforming_plate_vtk_s10_small
```

The options of the VTK reader (field names, parts) are listed in the
[command line reference](../reference/cli.md#export).

## The VTK reader

[`vtk_reader.py`](https://github.com/stofe94/crash-def-net/blob/main/dataset_preprocessor/vtk_reader.py)
reads every folder with VTK files as one run:

| | Supported |
|---|---|
| Files | `.vtu`, `.vtp`, legacy `.vtk` and multiblock `.vtm`; order and times from a `.pvd`, else from `*.series` files, else from the trailing number of the file names. The files may lie in subfolders, as ParaView saves a `.pvd` (`run/run.pvd` + `run/run/run_0_<t>.vtu` or `.vtm`); folders below a `.pvd`, `.series` or `.vtm` belong to it and are no runs of their own. One `.pvd` per run folder |
| Parts | several files per state (e.g. `part_a_0001.vtp`, `part_b_0001.vtp`, or `part` in the `.pvd`) and the blocks of a `.vtm` are joined into one mesh. Every part becomes a [component](../concepts/graphs.md#components-and-contact-rules) under its name; `--contact-pairs` (e.g. `plate:impactor plate:plate`) limits the world edges to the pairs of components that can touch |
| Shared nodes | exports give every part its own copies of the nodes it shares with other parts (ParaView's LS-DYNA reader: one block per part). With node IDs, `--node-id-field` (default: the field the files mark as global IDs, e.g. `UserID` of ParaView's LS-DYNA reader), the copies become one node again; they must lie at the same position in every state, and the states are matched by ID. Without IDs (or `--node-id-field ''`) the parts stay separate bodies (warning) |
| Cells | lines, triangles, quadrilaterals, tetrahedra, pyramids, wedges, hexahedra; voxels / pixels reordered, polygons and triangle strips triangulated, quadratic cells linearized (their mid-side nodes stay unconnected), vertices ignored. Degenerate cells as LS-DYNA writes them, also through ParaView, become the cells they are: quadrilateral n1 n2 n3 n3 → triangle, hexahedra with repeated nodes → tetrahedron, pyramid, wedge; other cells with repeated nodes are rejected. The cells of the first state hold for all states |
| Positions | the points of every state, or with `--displacement-field U` the points plus that field (exports that keep the reference geometry); rest positions = first state |
| Node types | `--node-type-field` (default `node_type`): 0 normal, 1 actuator, 3 clamped; then the nodes of `--actuator-parts` (e.g. a rigid impactor whose motion is given) and `--clamped-parts`. Parts are named by their block in a `.vtm` (block name, else its index), else by their file (`part` of the `.pvd`, name of the `.series`, file name without the number), or per cell by `--part-field` (e.g. a part ID field). A node takes the highest type of its copies and parts (clamped > actuator > normal); without field and parts all nodes are normal. `--free-rigid-parts`: rigid parts whose motion the model predicts, normal nodes also where the field marks them as actuator ([free rigid parts](../concepts/graphs.md#free-rigid-parts)). `--components` sets the parts and their roles instead ([config file](export.md#define-the-dataset-in-a-config-file)) |
| Fields | `--stress-field` (default `stress`; `''`: none): scalar, or stress tensor with 6 (xx yy zz xy yz zx) or 9 components → von Mises; cell data: von Mises per cell, averaged at the nodes over the cells of all parts. Nodes of parts without the field have [no value](../concepts/graphs.md#general-meshes-and-run-parameters) (warning). `--fields`: further point or cell fields of every state as [result fields](../concepts/graphs.md#general-meshes-and-run-parameters) (targets), each under its own name, e.g. `--fields plastic_strain`; vectors give one field per component. `--static-fields`: point or cell fields of the first state as static node features, with [no value](../concepts/graphs.md#general-meshes-and-run-parameters) at nodes of parts without them (e.g. shell thickness at solid nodes); vectors give one feature per component (`<name>_x` / `_y` / `_z`). `globals.json` in the run folder (next to the `.pvd`): run parameters |

A training can use a part of these fields ([choose the features](training.md#choose-the-features)).

Checked: TFRecord → VTU → `dataset.db` gives the same dataset as the TFRecord export, and
a crash-like LS-DYNA model saved by ParaView (multiblock, one file per part, merged grid)
gives the nodes, cells, positions and stress of the d3plot.

Not supported yet: changing topology (eroded cells), partitioned collections (`.vtpc`),
layered shell stress (one field), rigid walls and other boundary conditions beyond node
types.

## LS-DYNA runs via ParaView

Open the `d3plot` with ParaView's LS-DYNA reader and save it as `<run>/<run>.pvd` with
all time steps, one run folder each; ParaView writes one `.vtm` per state with one block
per part. Then, e.g. with the impactor as block 3:

```bash
python -m dataset_preprocessor.export --raw-format vtk --raw-dir runs_vtk --sources none \
    --stress-field Stress --static-fields Thickness --actuator-parts 3 --contact-radius 5
```

Keep the blocks rather than applying *Merge Blocks*: the merged grid keeps only the fields
that every block has (no `Thickness` next to solids) and loses the parts. ParaView's
`Stress` of a shell is the mid surface; the inner and outer surfaces are separate fields
(the [d3plot reader](ls-dyna.md#the-d3plot-reader) reduces over all integration points).
VTK 9.6.2 combines two user IDs of a 32-bit d3plot into one (newer VTK versions convert
them one by one), so the copies of a node no longer match and the reader stops with an
error; check `UserID` in ParaView, or ignore the IDs with `--node-id-field ''`.
