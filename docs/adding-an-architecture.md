# Adding an architecture

CrashDefNet is built so that a new model architecture can be tested quickly: the model
plugs into the existing pipeline, and data, training, rollout, metrics and the
comparison with the paper work unchanged. MGN-T was added this way. It needs four small
steps.

## 1. The model (`models/<name>.py`)

A `torch.nn.Module` with this interface:

```python
class MyModel(nn.Module):
    def __init__(self, node_in_dim: int, mesh_edge_in_dim: int, world_edge_in_dim: int,
                 output_dim: int, **hyperparameters):
        ...

    def forward(self, data) -> torch.Tensor:
        """Normalized predictions per node, shape (N, output_dim)."""
        ...
```

The input dimensions come from the dataset's `metadata.json`; the model does not need to
know the feature layout. `data` is a (batched) PyTorch Geometric graph with

| Attribute | Shape | Content |
|---|---|---|
| `x` | (N, node_in_dim) | normalized node features |
| `edge_index`, `edge_attr` | (2, E_M), (E_M, mesh_edge_in_dim) | mesh edges and their normalized features |
| `world_edge_index`, `world_edge_attr` | (2, E_W), (E_W, world_edge_in_dim) | contact edges; may be empty |
| `mesh_pos` | (N, 3) | undeformed positions (e.g. for positional encodings) |
| `world_pos` | (N, 3) | current positions, raw units |
| `batch` | (N,) | graph index of each node; missing for a single graph (e.g. in the rollout) |

Details of the features: [data.md](data.md). The building blocks of the existing models
can be reused, e.g. `make_mlp` and `GraphNetBlock` from `models/meshgraphnet.py`.

## 2. The config (`train/config.py`)

- Add the name to `ARCHITECTURES`.
- Add a field per hyperparameter to `TrainingConfig`, prefixed with the name, with a
  default (the paper value where there is one), e.g. `my_model_latent_size: int = 64`.
- Map the TOML keys onto the fields in `TOML_KEYS`, e.g.
  `"model.my_model.latent_size": "my_model_latent_size"`.

The config file then gets its own section:

```toml
[model]
architecture = "my_model"

[model.my_model]
latent_size = 64
```

## 3. Building the model (`train/trainer.py`)

Add a branch to `create_model` that passes the dimensions and the hyperparameters:

```python
if config.architecture == "my_model":
    from models.my_model import MyModel
    model = MyModel(**dims, latent_size=config.my_model_latent_size)
```

`create_model` is also used when a checkpoint is loaded, so the rollout finds the new
model without further changes. Optionally add a one-line description of the model to the
model summary at the start of `Trainer.train`.

## 4. Display name (optional)

`ARCHITECTURE_NAMES` in `utils/compare_paper.py` sets the name used in the figures and
tables of the comparison report (default: the architecture key).

## Testing the new architecture

```bash
# quick check on the CPU with the small dataset (architecture = "my_model" in the config)
python train/train.py --config configs/my_model.toml --epochs 1

# full comparison: train, roll out and compare with MeshGraphNet and the paper
python predict/rollout.py --checkpoint checkpoints/<run>/best_model.pt \
    --data-dir data/<dataset> --split test
python utils/compare_paper.py rollouts/<my_model rollout> rollouts/<mgn rollout>
```

Everything else applies to the new model automatically: position noise, rollout
validation and model selection, early stopping, the optional residual target, the
paper metrics, the `.vtu` export and the traceability records of every rollout.
