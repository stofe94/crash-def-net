# Compare architectures

Two architectures under the same [comparison protocol](comparison-protocol.md), step
by step: MGN-T and MeshGraphNets on `deforming_plate`, rolled out on 20 unseen geometries and
compared per trajectory, side by side and next to the values of the papers. This is how the
[deforming plate results](../results/deforming-plate.md) were made.

Needs the [installation](../getting-started/installation.md) and a GPU: on an RTX A4000
(16 GB) the training takes 2 h 54 min for MGN-T and 7 h 34 min for MeshGraphNets.

## 1. Raw data

The comparison draws each split from DeepMind's original one, so it needs all three files
`train.tfrecord`, `valid.tfrecord` and `test.tfrecord` next to `meta.json`
([raw data](../getting-started/installation.md#raw-data)). This command downloads what is
missing, ~11.5 GB if nothing is there yet. Complete files are skipped, an interrupted download
continues:

```bash
wget -c -P datasets/deforming_plate/raw_dataset \
    https://storage.googleapis.com/dm-meshgraphnets/deforming_plate/{meta.json,train.tfrecord,valid.tfrecord,test.tfrecord}
```

## 2. Dataset

100 training, 30 validation and 20 test trajectories, drawn from DeepMind's original splits
(~110 s, ~330 MB):

```bash
python -m dataset_preprocessor.export --out-dir data/deforming_plate_s4_t150 \
    --sources train.tfrecord valid.tfrecord test.tfrecord --select-seed 42 \
    --n-train 100 --n-val 30 --n-test 20 --num-steps 100 --time-stride 4
```

## 3. Configs

One copy of the protocol config per model:

```bash
mkdir -p configs
cp protocols/deforming_plate_s4_t150.toml configs/mgn.toml
cp protocols/deforming_plate_s4_t150.toml configs/mgn_t.toml
```

In each copy, set its own `[checkpointing] checkpoint_dir`, here `checkpoints/mgn` and
`checkpoints/mgn_t`, and the settings below; data, seed, budget and training recipe stay.
Both copies set the contact edges

```toml
[model]
world_edges                 = "self_contact_aware"
world_edge_exclude_distance = 2.0
```

and train without micro-batches (remove `micro_batch_size`), which these edges allow on a
16 GB GPU. MeshGraphNets is then the
[comparable baseline](comparison-protocol.md#baselines): the same contact edges as
MGN-T, so a difference comes from the architecture. The MGN-T copy also sets the architecture,
in the [configuration of its paper](../results/deforming-plate.md#mgn-t-configuration):

```toml
[model]
architecture = "mgn_t"

[model.mgn_t]
latent_size            = 80
token_dim              = 48
num_tokens             = 16
num_heads              = 4
num_transformer_blocks = 2
num_pre_mp_steps       = 2
num_post_mp_steps      = 2
num_mlp_layers         = 2
aggregation            = "sum"
transformer_nodes      = "structure"
tau_init               = 0.31
pe_frequencies         = 4
ffn_ratio              = 4
```

The keys go into the tables the copy already has: `[model]` of the protocol config, and
`[model.mgn_t]` in place of its `[model.meshgraphnets]`.

## 4. Train both

```bash
python train/train.py --config configs/mgn.toml
python train/train.py --config configs/mgn_t.toml
```

## 5. Roll out both

```bash
python predict/rollout.py --checkpoint checkpoints/mgn/best_model.pt \
    --data-dir data/deforming_plate_s4_t150 --split test
python predict/rollout.py --checkpoint checkpoints/mgn_t/best_model.pt \
    --data-dir data/deforming_plate_s4_t150 --split test
```

Each call writes a new folder `rollouts/<time>_<run>_best_model_ep<epoch>_test/` and names it
in the first line of its log. Below, `rollouts/<mgn>` and `rollouts/<mgn_t>` stand for these
two folders.

## 6. Compare per trajectory

```bash
python utils/check_protocol.py rollouts/<mgn> rollouts/<mgn_t>
python utils/evaluate_run.py rollouts/<mgn_t>
python utils/evaluate_run.py rollouts/<mgn>
```

The protocol check takes the first folder as the reference and checks that every run was
trained under the protocol config: the same dataset (by its fingerprint), seed, budget and
recipe, with only the declared variations. Then it compares the runs on every test trajectory
and counts how often each one has the lower error
([reading the result](comparison-protocol.md#reading-the-result)). `evaluate_run.py` reports each
run on its own: training, error over the rollout, spread over the test geometries.

## 7. Side by side

```bash
python utils/animate_rollout.py rollouts/<mgn_t> rollouts/<mgn> --trajectory test_traj0136
```

The GIF in `evaluations/<time>_animation/` shows the simulation and both predictions of the
same geometry on one stress scale: the animation at the top of the
[deforming plate results](../results/deforming-plate.md).

## 8. Next to the papers

```bash
python utils/compare_paper.py rollouts/<mgn_t> rollouts/<mgn>
```

The report `evaluations/<time>_paper_comparison/comparison.md` sets both runs against the
published values of MGN-T and MeshGraphNets, with figures of the error over the rollout and
over the trajectories. What the numbers show:
[deforming plate results](../results/deforming-plate.md#what-the-numbers-show); why they are not directly comparable
with the papers:
[conditions](../results/deforming-plate.md#why-the-numbers-are-not-comparable-with-the-paper).

## Next steps

- [Add an architecture](adding-an-architecture.md): a third model in the same
  comparison.
- [Compare with the papers](evaluation.md#compare-with-the-papers): all options and
  files of the report.
