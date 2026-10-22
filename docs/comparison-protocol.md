# Comparison protocol (deforming_plate, stride 8)

This protocol fixes how a new architecture is compared with the MeshGraphNets baseline
on a small subset of DeepMind's `deforming_plate` data: same dataset, same training
budget and recipe, same evaluation. Only the model changes, plus a few declared
variations that are reported with the result. Every comparison is checked with
[`utils/check_protocol.py`](../utils/check_protocol.py).

The subset is small on purpose: a run takes hours, not days, on one GPU. The numbers are
therefore not comparable with the papers (see [results](results.md#why-the-numbers-are-not-comparable-with-the-paper)),
but they are comparable with each other.

## Steps

```bash
# 1) Dataset (once, ~10 s, ~48 MB)
python -m dataset_preprocessor.export --out-dir data/deforming_plate_s8_t40 \
    --n-train 28 --n-val 8 --n-test 4 --num-steps 50 --time-stride 8

# 2) Training: copy the protocol config, change only [model] (and checkpoint_dir)
cp protocols/deforming_plate_s8_t40.toml configs/my_model_s8_t40.toml
python train/train.py --config configs/my_model_s8_t40.toml

# 3) Rollout of the selected model on the whole test split
python predict/rollout.py --checkpoint checkpoints/<run>/best_model.pt \
    --data-dir data/deforming_plate_s8_t40 --split test

# 4) Protocol check and paired comparison (first folder: the baseline)
python utils/check_protocol.py rollouts/<baseline rollout> rollouts/<new rollout> \
    --exclude test_traj0038

# 5) Comparison with the paper, unchanged
python utils/compare_paper.py rollouts/<new rollout> rollouts/<baseline rollout> \
    --exclude test_traj0038
```

The baseline itself is trained with the protocol config as it stands. `check_protocol.py`
checks against `protocols/deforming_plate_s8_t40.toml` unless `--protocol` names another
protocol config.

## Dataset

| | |
|---|---|
| Source | `test.tfrecord` of DeepMind's `deforming_plate` (100 trajectories) |
| Split | trajectories 0–27 train, 28–35 val, 36–39 test |
| Frames | 50 per trajectory at stride 8 (400 simulation steps) → 49 samples per trajectory |
| Samples | 1,372 train / 392 val / 196 test |

The dataset is not distributed: everyone exports it with step 1. All rollouts of one
comparison must come from the same export; the check compares the dataset identity
recorded in each rollout (the content fingerprint, for exports before format v4 the
SHA-256 of the file, see [data.md](data.md#large-datasets)). Train the baseline on your
own export as well and compare with that run, not with rollouts made on another export.

## Training

The protocol config [`protocols/deforming_plate_s8_t40.toml`](../protocols/deforming_plate_s8_t40.toml)
holds the fixed settings:

| Setting | Value |
|---|---|
| Budget | **batch 2, 120 epochs** = 82,320 optimizer steps for every model |
| Optimizer | Adam, learning rate 1e-4 → 1e-6 exponentially over the whole run (`decay_steps = "auto"`), no weight decay |
| Gradient clipping | norm 1.0 |
| Loss | MSE on the normalized targets of all nodes except the actuator |
| Noise | position noise σ = 3·10⁻³ m on the normal nodes, targets corrected; no stress noise |
| Inputs | history 0, no residual target, no scheduled sampling |
| Validation | rollout of all 8 val trajectories every 5 epochs |
| Model selection | `criteria = ["rollout_mse"]`: the evaluation with the lowest rollout MSE on the val split |
| Early stopping | off (`enabled = false`): every model trains all 120 epochs |
| Seed | `seed = 42` for every model: same weight initialization, sample order and noise draws |
| Evaluated model | `best_model.pt`, the selected model, from a run that trained all epochs (the check reports aborted runs) |

Equal epochs at equal batch size give every architecture the same number of optimizer
steps and the same learning-rate course. Wall-clock time and parameter count are reported,
not equalized.

Machine settings are free: `data_dir`, data loading (`num_dataloader_workers`, `lazy`),
`device`, `checkpoint_dir`, `save_every`, `log_every`.

## Declared variations

These may differ from the protocol. The check lists them, and they are stated with every
result:

| Variation | Settings | Why it is open |
|---|---|---|
| Architecture | `[model] architecture` and the model's own section | the subject of the comparison |
| World edges | `world_edges`, `world_edge_exclude_hops` | edge construction that scales to larger meshes |

The dataset (the same export, see above), the seed, the model selection and early
stopping are not variations: every model is trained on exactly the same data with the
same seed and selected by the same rule.

A variation of the world edges changes more than the architecture. To attribute a
difference to the architecture alone, train the baseline with the same variation as well
(e.g. MeshGraphNets with `world_edges = "contact"`) and compare with that run.

## Reading the result

The test split holds 4 trajectories. The ± of the paper metrics is the standard error
over these trajectories: it shows how different the four geometries are, not how
precisely a model is measured. Both models are evaluated on the same trajectories, so
they are compared **per trajectory** (paired), which is what `check_protocol.py` prints:

- **A difference counts as established only if one model has the lower error on every
  trajectory (4 of 4).** With two equally good models one of them is lower on all four by
  chance in 12.5 % of cases, on three of four in 50 %. A 3:1 therefore says almost
  nothing, whatever the mean.
- **Size matters too.** A 4:0 with one difference close to zero is weaker than one with
  clear differences everywhere. Relative differences up to 0.1 % count as equal
  (numerical noise between machines).
- **With and without trajectory 38.** Its largest displacement exceeds every training
  trajectory, so both models extrapolate there and it dominates the means
  (see [results](results.md#what-the-numbers-show)). Read the result for all four
  trajectories and for the other three (`--exclude test_traj0038`).

## Reference run

MeshGraphNets in paper size (latent 128, 2 hidden layers, 15 message passing steps,
3.85 M parameters), trained for v0.2.0 on this dataset with the protocol's recipe (the
protocol was derived from this run), for 93 of 120 epochs (3 h 41 min on an RTX A4000;
learning rate at 2.8·10⁻⁶). Selected model: epoch 80 by rollout MSE. Rolled out again
with the code of October 2026, every paper metric matched to four digits.

Its numbers are orientation values. A baseline trained on another machine or export
will not match them exactly; a comparison always uses your own baseline run.

Test trajectories (48 rollout steps):

| Trajectory | q RMSE-all [mm] | σ RMSE-all [kPa] | q RMSE-1 [mm] | σ RMSE-1 [kPa] |
|---|---|---|---|---|
| test_traj0036 | 4.11 | 8.05 | 0.356 | 5.90 |
| test_traj0037 | 2.50 | 11.2 | 0.462 | 8.00 |
| test_traj0038 | 88.5 | 26.7 | 8.42 | 60.2 |
| test_traj0039 | 2.11 | 13.8 | 0.497 | 11.5 |
| pooled (paper metric) | 44.1 ± 21 | 16.5 ± 4.1 | 4.21 ± 2.0 | 30.9 ± 13 |

## Relation to the MeshGraphNets paper

Model, inputs and outputs follow Pfaff et al. (ICLR 2021) for `deforming_plate`: latent
size 128, two-hidden-layer ReLU MLPs with LayerNorm except in the decoder, 15 message
passing steps with residual connections and sum aggregation; node type and the
actuator's next-step velocity as node inputs; mesh edges with u_ij, |u_ij|, x_ij, |x_ij|
and world edges with x_ij, |x_ij| within r_W = 0.03; velocity and von Mises stress as
outputs; history 0; batch 2; position noise 3·10⁻³; learning rate 1e-4 → 1e-6.

Differences besides the dataset subset:

| | Paper | Protocol |
|---|---|---|
| Training length | 10 M steps, learning rate decayed over 5 M | 82,320 steps (the reference run ~64,000), decayed over the whole run |
| End of training | fixed number of steps | fixed number of steps (120 epochs), model selection on val rollouts |
| Gradient clipping | not mentioned | norm 1.0 |
| Loss nodes | not specified for `deforming_plate` | all except the actuator, so the clamped nodes' stress is learned |
| Noise | 3·10⁻³ m at stride 1 | the same absolute value at stride 8, where a step moves the nodes about eight times as far: relatively weaker |

The baseline is therefore "MeshGraphNets in paper configuration, trained briefly", not
a reproduction of the paper's model.
