# Deforming plate

The two architectures in CrashDefNet, **MeshGraphNets (MGN)** and
**MeshGraphNet-Transformer (MGN-T)**, on DeepMind's `deforming_plate`, next to the values
of the papers. The runs are deliberately small: a subset of the data, one GPU, a few
hours. Both models share data, seed, budget and recipe of the
[comparison protocol](../guides/comparison-protocol.md) and the contact edges
(`self_contact_aware`): MGN-T as in arXiv v4 ([configuration](#mgn-t-configuration)),
MeshGraphNets in paper size as the [comparable baseline](../guides/comparison-protocol.md#baselines).
The runs show that the pipeline works end to end and how the architectures behave; they
are **not** a reproduction of the papers' numbers.

![Rollout of an unseen test geometry: ground truth, MGN-T and MeshGraphNets](../assets/rollout_test_traj0136.gif)

*Autoregressive rollout of test trajectory 136, a geometry the models never saw during
training. Colour: von Mises stress, same scale in all three panels. Grey: the actuator, whose
motion is prescribed.*

## Setup

| | MGN-T (paper configuration) | MeshGraphNets (comparable baseline) |
|---|---|---|
| Architecture | 2 + 2 message passing steps, 2 transformer blocks over the plate nodes, 16 physics tokens, widths 80 / 48 ([configuration](#mgn-t-configuration)) | 15 message passing steps, latent 128 |
| Contact edges | between bodies and for self-contact (`self_contact_aware`, paper v4) | the same (paper: all pairs within the contact radius, `all`; Pfaff et al. 2021, Sec. 3.1 on p. 4) |
| Parameters | **0.51 M** | 3.85 M |
| Data | DeepMind's original splits: 100 / 30 / 20 trajectories drawn from `train` / `valid` / `test.tfrecord` (seed 42), 100 frames per trajectory at stride 4 | same |
| Training | batch 8, all 60 epochs, **2 h 54 min** (~2 min 45 s per epoch) | batch 8, all 60 epochs, 7 h 34 min (~7 min 15 s per epoch) |
| Hardware | 1× NVIDIA RTX A4000 (16 GB) | same |
| Memory in training | GPU ~2 GB; RAM ~2 GB in the main process, < 1 GB per DataLoader worker (3 workers) | GPU ~8 GB; RAM as MGN-T |
| Evaluated checkpoint | epoch 52 (selected model) | epoch 56 (selected model) |
| Model selection | rollout MSE on the val split | same |
| Rollout | 98 autoregressive steps over the whole load case (loading and unloading), 20 unseen test geometries | same |

Both models see the same graph, so a difference comes from the architecture
([baselines](../guides/comparison-protocol.md#baselines)). The MeshGraphNets paper's own results,
with its edges, appear as reference rows below.

## MGN-T configuration

MGN-T is trained in the configuration of its paper (Iparraguirre et al., arXiv v4,
Sec. 3 and 5.1; every key in [compare architectures](../guides/compare-architectures.md#3-configs)),
with a global stage over the simulated structure only. The actuator thus acts on the plate
through contact, and the plate passes the load on through its material.

Values the paper leaves open are set as follows; the implementation's general
assumptions are listed in [models](../concepts/models.md#assumptions-where-the-paper-leaves-gaps):

| Setting | Value | Why |
|---|---|---|
| Latent / token width | 80 / 48, 0.51 M parameters | v4 gives ~0.5 M parameters (Tab. 2) but no widths |
| Initial temperature | τ₀ = 1.0 (`tau_init = 0.31`) | the paper's τ₀; softplus keeps τ positive |
| Positional encoding | 4 frequencies (`pe_frequencies`) | not given in the paper |
| Feed-forward width | 4 × token width (`ffn_ratio`) | standard value; not given in the paper |

## Results

Metrics as in the MGN-T paper (arXiv v4, App. A on p. 12;
[metrics](../concepts/metrics.md#rollout-metrics)): **q** node position (m), **σ** von Mises
stress (Pa); **RMSE-all** over the whole rollout, **RMSE-1** one step from the ground-truth
state; **20 test traj. (pooled)** one RMSE over all test trajectories together, as in the
papers, **median trajectory** the median of the RMSEs of the single trajectories; **±**
standard error over the test trajectories.

| Model | Params | q RMSE-all [mm] | σ RMSE-all [kPa] | q RMSE-1 [mm] | σ RMSE-1 [kPa] |
|---|---|---|---|---|---|
| MGN-T · own (20 test traj.) | 0.51 M | **12.2 ± 1.9** | 17.8 ± 1.7 | **0.895 ± 0.15** | 11.1 ± 0.82 |
| MeshGraphNets (`self_contact_aware`) · own (20 test traj.) | 3.85 M | 26.4 ± 4.9 | **16.1 ± 1.9** | 1.36 ± 0.26 | **7.15 ± 0.77** |
| MGN-T · own, median trajectory | 0.51 M | 5.62 | 15.3 | 0.339 | 10.2 |
| MeshGraphNets (`self_contact_aware`) · own, median trajectory | 3.85 M | **4.64** | **11.0** | **0.277** | **5.22** |
| *MGN-T · paper (Tab. 3)* | *0.5 M* | *3.16 ± 0.25* | *5.83 ± 0.34* | *0.100 ± 0.005* | *3.87 ± 0.17* |
| *MGN · Pfaff et al. 2021 (Tab. 1)* | *2.0 M* | *15.1 ± 4.0* | *–* | *0.25 ± 0.05* | *–* |

Reference rows: MGN-T from Iparraguirre et al. (arXiv v4, Tab. 3), MGN from Pfaff et al.
(ICLR 2021, Tab. 1; parameter count from the MGN-T paper, Tab. 2).

![Error over the test trajectories](../assets/error_per_trajectory.png)

All metrics next to the paper's values (whiskers: standard error):

![Absolute errors compared with the MGN-T paper](../assets/absolute_errors.png)

Error over the rollout, averaged over the 20 test trajectories. The drop at the end is
the unloading phase, where the plate springs back:

![Error over the rollout](../assets/error_per_step.png)

### What the numbers show

- **Pooled and median.** The pooled values lie well above the median because a few test
  geometries deform more than any training trajectory: trajectory 149 by up to 597 mm
  (training: maximum 445 mm, median 77 mm). Squared before averaging, their errors dominate
  the pooled values; for the typical geometry the median is the better guide, only the pooled
  values compare with the paper.
- **Efficiency.** MGN-T has **7.5× fewer parameters** than MeshGraphNets, needs
  **2.6× less training time** (2 h 54 min instead of 7 h 34 min) and about a quarter of
  the GPU memory (~2 instead of ~8 GB).
- **Position.** MGN-T has the lower position error over all 20 test geometries
  (12.2 vs 26.4 mm). Paired per geometry it is lower on 10 of the 20: on the five with
  the largest errors of MeshGraphNets (trajectories 149, 146, 147, 137 and 145) by a
  factor of 1.9 to 4. On the other ten MeshGraphNets is lower; on the median geometry it
  reaches 4.6 mm, MGN-T 5.6 mm.
- **Stress.** MeshGraphNets has the lower stress error on 14 of the 20 geometries
  (16.1 vs 17.8 kPa over all of them).
- **One step.** From the ground-truth state MeshGraphNets has the lower error on most
  geometries: position on 16 of 20, stress on 19 of 20.
- **Large deformations.** Trajectories 149, 146 and 147 have the largest errors of both
  models and deform most (597, 380 and 276 mm).

### Why the numbers are not comparable with the paper

| | Paper | Own runs |
|---|---|---|
| Training data | number of trajectories not stated (DeepMind's training split holds 1,200) | 100 trajectories |
| Time step | stride 1, 400 rollout steps | stride 4, 98 rollout steps |
| MGN-T configuration | arXiv v4 | arXiv v4; open values as [above](#mgn-t-configuration) |
| Contact edges of MeshGraphNets | all pairs within the contact radius (`all`) | `self_contact_aware`, as MGN-T |
| Training | MGN-T: 25 epochs, batch 32 (arXiv v4, Sec. 5.1); ~8 h on an RTX 4090 (v3) | MGN-T: 60 epochs, batch 8, 2 h 54 min on an RTX A4000 |
| Test set | size not stated | 20 trajectories |

- **RMSE-1** is not comparable: at stride 4 the nodes move about four times as far per
  step as in the paper.
- **RMSE-all** covers the same physical time span, but with fewer steps, which means
  fewer chances for errors to accumulate.

## Discussion

In this setup neither architecture is more accurate on every test geometry: MGN-T keeps
the position error lower on the geometries that deform most, MeshGraphNets is more
accurate on the others and in stress. Hybrid architectures such as MGN-T aim at very
large meshes, on which pure message passing needs many steps to carry information
through the structure; meshes of 700 to 1,900 nodes cannot show this. The public plate
dataset nevertheless suits the development of such methods: its geometries vary
strongly, and at stride 4 it has few time steps, as real crash data often do.

A scientific evaluation of the results is planned for later releases of CrashDefNet and
will be published separately, as a paper or in another form.

Made with release [v0.4.0](https://github.com/stofe94/crash-def-net/tree/v0.4.0). The rollout
of the current version predicts every frame after the initial state, one step more than
v0.4.0, so RMSE-all differs by a few percent (MGN-T: 12.1 instead of 12.2 mm over all 20 test
trajectories, 5.76 instead of 5.62 mm on the median trajectory); RMSE-1 is the same. How to
run it: [compare architectures](../guides/compare-architectures.md).
