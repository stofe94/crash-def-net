# Project structure

Every module starts with a docstring that describes its purpose and design; it is the
best entry point into the code.

```
dataset_preprocessor/
  export.py                  CLI: raw data (TFRecord, d3plot, VTK) -> dataset.db + normalizer.json + metadata.json
  export_config.py           export config (--config): components with their roles, features, contact
  dataset.py                 TrajectoryDataset: one graph per time step
  graph_features.py          feature layout, mesh and world edges, consistent position noise
  tfrecord_to_d3plot.py      CLI: DeepMind TFRecord -> LS-DYNA runs (d3plot + model.k)
  d3plot_reader.py           d3plot reader (LS-DYNA keyword deck), raw data for --raw-format d3plot
  radioss_deck.py            OpenRadioss starter deck next to a converted d3plot
  tfrecord_to_vtk.py         CLI: DeepMind TFRecord -> VTK time series (.vtu / .vtk)
  vtk_reader.py              VTK time series reader (.vtu / .vtp / .vtk), raw data for --raw-format vtk
models/
  meshgraphnets.py           MeshGraphNets (mesh + world edges)
  mgn_t.py                   MeshGraphNet-Transformer (physics attention)
  residual.py                optional residual target wrapper
train/                       training CLI, config, trainer, early stopping
predict/rollout.py           autoregressive rollout, rollout metrics, .vtu export
protocols/                   training config of the comparison protocol
examples/                    export, training and rollout configs: quick_start/ (the quick start,
                             train.toml also the default of train.py, and the Colab notebook
                             crashdefnet-quickstart.ipynb), deforming_plate/ (first training),
                             bumper_beam/; copy and adapt them for own runs
utils/                       normalizer, loss curves, evaluation of a run, data exploration,
                             paper comparison, protocol check, rollout animation, time zone,
                             units of a dataset,
                             reference pages and links of the documentation,
                             meshing the pole of the bumper beam runs,
                             the CrashDefNet app (app.py + .html)
docker/, Dockerfile,         GPU training in a container
docker-compose.yml
docs/                        documentation website: pages, figures and styles
mkdocs.yml                   website configuration and navigation
.github/workflows/docs.yml   builds and publishes the website
```

Not versioned: `datasets/` (raw data), `data/` (exports), `checkpoints/`, `rollouts/`,
`evaluations/`.
