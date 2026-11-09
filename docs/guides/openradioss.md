# OpenRadioss runs

OpenRadioss is an open explicit finite element solver for crash. The exporter reads its runs as
d3plot files converted from the animation files, together with the starter deck `*_0000.rad`,
which holds the model definition that a d3plot does not: one folder per run. Needs
`lasso-python` (`pip install lasso-python`). The whole chain on a real crash example, from the
download to the trained model: [tutorial: bumper beam](../getting-started/bumper-beam.md).

!!! warning "Early stage of development"
    The processing of OpenRadioss runs is at an early stage of development. It is verified on a
    real crash example, the runs of a bumper beam against a pole, exported, trained and rolled
    out end to end ([results](../results/bumper-beam.md),
    [verification status](../concepts/graphs.md#general-meshes-and-run-parameters)).

## Export

The starter deck next to the d3plot is read without any option, and files named
`<name>.d3plot`, `<name>.d3plot01`, ... are found as they are:

```bash
python -m dataset_preprocessor.export --raw-format d3plot --raw-dir my_runs --sources none \
    --contact-radius 30 --n-train 40 --n-val 5 --n-test 5 --num-steps 11 --time-stride 1 \
    --history 1 --length-unit mm --stress-unit GPa --free-rigid-parts Body_in_White \
    --out-dir data/my_runs
```

The options are those of [LS-DYNA runs](ls-dyna.md#export), as both are read as d3plot.
The parts and what each of them is are best defined in an
[export config](export.md#define-the-dataset-in-a-config-file), as in
[`examples/bumper_beam/export.toml`](https://github.com/stofe94/crash-def-net/blob/main/examples/bumper_beam/export.toml).
Units follow the deck: kg, mm, ms give a stress in GPa.

## The starter deck

| In the starter deck | Becomes |
|---|---|
| `/PROP/SHELL` of every `/PART` | the static node feature `thickness`, where the d3plot holds only zeros; averaged at the nodes over the shells of these parts |
| `/RWALL/CYL`, stationary | a rigid wall: the distance to the surface of the cylinder and the radial normal as [node features](../concepts/graphs.md#general-meshes-and-run-parameters); treated as infinite |
| Titles of `/RWALL` and `/RBODY` | parts of the d3plot with these titles are rigid parts: actuators, or predicted with `--free-rigid-parts` |
| `/BCS` with all three translations fixed | clamped nodes |
| `/INTER/TYPE2` on `/SURF/PART` | a mesh edge from every tied node to the nearest node of its surface |
| `/INIVEL/TRA` | the initial velocity, also on `/GRNOD/PART`; a tied node takes that of its surface. With `--history 1` the velocity input of the first frame, so that a rollout starts from the undeformed state ([optional history](../concepts/graphs.md#optional-history-history-1-not-in-the-paper)) |

The conversion may give several parts the same ID; they are told apart by their titles. It may
also write the display models of walls and sections as shells, which stay in the mesh: shells
without area are left out with a warning, and the shells of a section have no stress, which
lowers the nodal mean of the stress at their nodes. `--exclude-parts` leaves such parts out,
e.g. `--exclude-parts "New SECT 1"`; a wall of the deck stays when its display model goes.

Not read: other wall types, moving walls, other kinds of `/INIVEL` and velocities in a skew
frame, node groups other than `/GRNOD/NODE` and `/GRNOD/GRNOD` (for `/INIVEL` also
`/GRNOD/PART`), surfaces other than `/SURF/PART` and `#include`. The deformation, the results,
the elements and the parts come from the d3plot, read as for LS-DYNA runs
([the d3plot reader](ls-dyna.md#the-d3plot-reader)).

## A pole as contact partner

A wall of the deck reaches the model through the node features. For a pole that acts through
[contact edges](../concepts/graphs.md#world-edge-variants-model-world_edges) instead,
`utils/remesh_pole.py` writes the runs of
[BumperBeamCrashExample](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample)
as a second raw dataset: a meshed cylinder surface takes the place of the display model of the
pole. A run whose conversion left the display models out gets the surface as a new part, titled
as the `/RWALL` of the deck. All else stays identical, and every run is read back and compared
with the original; only a missing version in the header is set, as ParaView's LS-DYNA reader
needs it. The data are then derived, no longer those of the solver. Export them with `--no-deck-walls`; options:
[remesh_pole](../reference/cli.md#remesh_pole), the whole chain:
[tutorial](../getting-started/bumper-beam.md#2-mesh-the-pole).
