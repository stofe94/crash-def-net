# Comparison protocol (deforming_plate, original splits, stride 4)

This protocol fixes how a new architecture is compared with MeshGraphNets on a subset
of DeepMind's original `deforming_plate` splits: same dataset, budget, recipe and
evaluation; only the model and a few declared variations change, checked with
[`utils/check_protocol.py`](https://github.com/stofe94/crash-def-net/blob/main/utils/check_protocol.py). The subset is small on purpose,
so that a run takes hours on one GPU; the numbers compare with each other, not with the
papers ([results](../results/deforming-plate.md#why-the-numbers-are-not-comparable-with-the-paper)). A
walk-through with both architectures: [compare architectures](compare-architectures.md).

## Baselines

Results are related to MeshGraphNets in two ways:

| Reference | What it is | Role |
|---|---|---|
| Original | the published results of the MeshGraphNets paper, with world edges between all node pairs closer than r_W that are not mesh neighbours (`world_edges = "all"`; Pfaff et al. 2021, Sec. 3.1 on p. 4); taken from the paper, not reproduced here | relates results to the literature ([`compare_paper.py`](../guides/evaluation.md#compare-with-the-papers)) |
| Comparable baseline | MeshGraphNets in paper size, trained under this protocol with `world_edges = "self_contact_aware"` (declared variation) | the reference for new architectures, with the contact edges they use (`check_protocol.py`) |

New architectures are compared with the comparable baseline. The reason is scalability:
`"all"` also links near neighbours inside a body, their number grows with the mesh
density ([memory](../concepts/scaling.md#memory-of-one-sample)), which
does not carry over to crash meshes; `self_contact_aware` keeps only contact between
bodies and self-contact ([world edge variants](../concepts/graphs.md#world-edge-variants-model-world_edges)). With
the same edges on both sides a difference comes from the architecture. Every result
states which reference it is compared with. The protocol config itself keeps the
paper's setting `"all"`.

## Steps

```bash
# 1) Dataset (once, ~110 s, ~330 MB; needs train, valid and test.tfrecord, see Installation)
python -m dataset_preprocessor.export --out-dir data/deforming_plate_s4_t150 \
    --sources train.tfrecord valid.tfrecord test.tfrecord --select-seed 42 \
    --n-train 100 --n-val 30 --n-test 20 --num-steps 100 --time-stride 4

# 2) Training: copy the protocol config, change only [model] (and checkpoint_dir);
#    world_edges = "self_contact_aware", as the comparable baseline
mkdir -p configs && cp protocols/deforming_plate_s4_t150.toml configs/my_model_s4_t150.toml
python train/train.py --config configs/my_model_s4_t150.toml

# 3) Rollout of the selected model on the whole test split
python predict/rollout.py --checkpoint checkpoints/<run>/best_model.pt \
    --data-dir data/deforming_plate_s4_t150 --split test

# 4) Protocol check and paired comparison (first folder: the comparable baseline)
python utils/check_protocol.py rollouts/<baseline rollout> rollouts/<new rollout>

# 5) Evaluation of the new run on its own, and comparison with the paper
python utils/evaluate_run.py rollouts/<new rollout>
python utils/compare_paper.py rollouts/<new rollout> rollouts/<baseline rollout>
```

`check_protocol.py` checks against `protocols/deforming_plate_s4_t150.toml` unless
`--protocol` names another protocol config.

## Dataset

| | |
|---|---|
| Source | 100 of the 1,200 trajectories of `train.tfrecord`, 30 of the 100 of `valid.tfrecord`, 20 of the 100 of `test.tfrecord`, drawn with seed 42 (the drawn ones are listed in `metadata.json`) |
| Frames | 100 per trajectory at stride 4 (400 simulation steps) → 99 samples per trajectory |
| Samples | 9,900 train / 2,970 val / 1,980 test |

The dataset is not distributed: everyone exports it with step 1 above. All rollouts of one
comparison must come from the same export, which the check verifies by the content
fingerprint: train the baseline on your own export.

## Training

The protocol config [`protocols/deforming_plate_s4_t150.toml`](https://github.com/stofe94/crash-def-net/blob/main/protocols/deforming_plate_s4_t150.toml)
holds the fixed settings:

| Setting | Value |
|---|---|
| Budget | **batch 8, 60 epochs** = 74,280 optimizer steps (about 594,000 samples) for every model |
| Optimizer | Adam, learning rate 2e-4 → 1e-6 along a cosine over the epochs, no weight decay; 2e-4 = 1e-4 × √(8 / 2), the gradient noise of batch 2 at 1e-4 |
| Gradient clipping | norm 1.0 |
| Loss | MSE on the normalized targets of all nodes except the actuator |
| Noise | position noise σ = 3·10⁻³ m on the normal nodes, targets corrected; no stress noise |
| Inputs | history 0, no residual target, no scheduled sampling |
| Validation | rollout of all 30 val trajectories every 4 epochs, as one batch of graphs |
| Model selection | `criteria = ["rollout_mse"]`: the evaluation with the lowest rollout MSE on the val split |
| Early stopping | off (`enabled = false`): every model trains all 60 epochs |
| Seed | `seed = 42` for every model: same weight initialization, sample order and noise draws |
| Evaluated model | `best_model.pt`, the selected model, from a run that trained all epochs (the check reports aborted runs) |

Equal epochs at equal batch size give every architecture the same number of optimizer
steps and the same learning-rate course. Wall-clock time and parameter count are reported,
not equalized.

Machine settings are free: `data_dir`, data loading (`num_dataloader_workers`, `lazy`),
`micro_batch_size` (gradient accumulation, the same optimizer step, see
[training](../concepts/training.md#loss-and-optimizer-step)), `device`, `checkpoint_dir`,
`save_every`, `log_every`.

Memory: with `world_edges = "all"` MeshGraphNets in paper size keeps up to ~13.5 GiB of
activations for the largest graph of this dataset, so the protocol config trains it in
micro-batches of one graph (`micro_batch_size = 1`, 16 GB GPU). With
`self_contact_aware` both models of the [results](../results/deforming-plate.md) train at batch 8 without
micro-batches.

## Declared variations

These may differ from the protocol. The check lists them, and they are stated with every
result:

| Variation | Settings | Why it is open |
|---|---|---|
| Architecture | `[model] architecture` and the model's own section | the subject of the comparison |
| World edges | `world_edges`, `world_edge_exclude_distance` | edge construction that scales to larger meshes |

Dataset, seed, model selection and early stopping are not variations: every model
trains on the same data with the same seed and is selected by the same rule.

## Reading the result

The test split holds 20 trajectories. The ± of the rollout metrics is the standard error
over these trajectories: it shows how different the geometries are, not how precisely a
model is measured. Both models are evaluated on the same trajectories, so
`check_protocol.py` compares them **per trajectory** (paired): the difference on each
trajectory and how often each model has the lower error, for RMSE-all and RMSE-1 of
position and stress. Relative differences up to 0.1 % count as equal (numerical noise
between machines). `--exclude` adds the counts without the given trajectories.

A few trajectories with large deformations can dominate the pooled rollout metrics; the
median trajectory is in the report of
[`utils/evaluate_run.py`](../guides/evaluation.md#evaluate-a-run).

## Reference run

The comparable baseline of the [deforming plate results](../results/deforming-plate.md): MeshGraphNets in paper
size with `self_contact_aware`, trained with the protocol's recipe
([setup](../results/deforming-plate.md#setup), [values](../results/deforming-plate.md#results)). Its numbers are
orientation values; a comparison always uses your own baseline run.

## Relation to the MeshGraphNets paper

Model, inputs and outputs follow Pfaff et al. (ICLR 2021) for `deforming_plate`: latent
size 128, two-hidden-layer ReLU MLPs with LayerNorm except in the decoder, 15 message
passing steps with residual connections and sum aggregation; node type and the
actuator's next-step velocity as node inputs; mesh edges with u_ij, |u_ij|, x_ij, |x_ij|
and world edges with x_ij, |x_ij| within r_W = 0.03; velocity and von Mises stress as
outputs; history 0; position noise 3·10⁻³.

Differences besides the dataset subset:

| | Paper | Protocol |
|---|---|---|
| Training length | 10 M steps at batch 2, learning rate 1e-4 → 1e-6 decayed over 5 M | 74,280 steps at batch 8, learning rate 2e-4 → 1e-6 along a cosine over the whole run |
| End of training | fixed number of steps | fixed number of steps (60 epochs), model selection on val rollouts |
| Gradient clipping | not mentioned | norm 1.0 |
| Loss nodes | not specified for `deforming_plate` | all except the actuator, so the clamped nodes' stress is learned |
| Noise | 3·10⁻³ m at stride 1 | the same absolute value at stride 4, where a step moves the nodes about four times as far: relatively weaker |
| Contact edges | all pairs within r_W = 0.03 that are not mesh neighbours (Pfaff et al. 2021, Sec. 3.1 on p. 4; r_W: Tab. A.1 on p. 13) | protocol config: the same; comparable baseline: `self_contact_aware` |

A run of the protocol config is therefore "MeshGraphNets in paper configuration, trained
briefly", not a reproduction of the paper's model; the comparable baseline differs from it
in the contact edges as well. The paper's own results are the original reference.
