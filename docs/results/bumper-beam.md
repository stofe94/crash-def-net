# Bumper beam against a pole

MGN-T on a real crash case: OpenRadioss runs of a bumper beam that hits a rigid pole. The model
predicts the impact autoregressively, each state from its own previous prediction, with the
contact to the pole and the self-contact of the structure rebuilt from the predicted geometry at
every step. The runs are read as d3plot files with their OpenRadioss starter decks, only the
display of the pole replaced by a mesh, and pass through every stage of CrashDefNet: reader and
starter deck, export config with the roles of the parts, a free rigid body, contact edges to
the pole, units in mm and GPa and features that some nodes do not have.

![Rollout of an unseen test run: simulation and MGN-T](../assets/bumper_beam_test_traj0128.gif)

*Autoregressive rollout of test run 128, the test run at the median of the position error, from
10 to 100 ms. Its beam thickness and pole position did not occur in training. Colour: von Mises
stress, same scale in both panels. Grey: the pole.*

## Data

[BumperBeamCrashExample](https://huggingface.co/datasets/AIRBORNEPANDA/BumperBeamCrashExample)
on Hugging Face, generated with
[OpenRadioss2PhysicsNemo](https://github.com/HoussemMouradi/OpenRadioss2PhysicsNemo); the decks
of the runs carry the notice "Copyright Altair Engineering, CC BY-NC 4.0".

| | |
|---|---|
| Model | bumper beam with two crash boxes, shell parts joined by spot welds, on a rigid body of 500 kg; rigid cylindrical pole of 254 mm diameter |
| Load case | impact at 5 m/s; 11 states at 10 ms, 0 to 100 ms |
| Varied | thickness of the beam, 1.21 to 2.0 mm (crash boxes 2.2 mm), and position of the pole across the beam, −480 to 492 mm |
| Mesh | 13,832 nodes of the structure and 2,320 of the meshed pole, the same in every run |
| Splits | 110 / 14 / 7 runs: training and validation from `TRAINING_DATA`, test = `VALIDATION_DATA` of the dataset |

The solver computes the impact against an analytic cylinder.
[`utils/remesh_pole.py`](../guides/openradioss.md#a-pole-as-contact-partner) meshes it at the same place, so
that the pole acts on the structure through contact edges; all other nodes and results stay
identical.

## Setup

| | MGN-T |
|---|---|
| Architecture | as arXiv v4, 0.51 M parameters ([configuration](deforming-plate.md#mgn-t-configuration)); transformer over the structure, without the pole (`transformer_nodes = "structure"`) |
| Contact edges | between bodies and for self-contact (`self_contact_aware`), radius 30 mm; each with the distance from the shell surface of the sender and the clearance between the shell surfaces ([contact gap](../concepts/graphs.md#general-meshes-and-run-parameters)) |
| Inputs per node | node type, prescribed velocity, previous velocity and result fields, shell thickness and yield stress from the starter deck, distance to the [cylinder of the deck](../guides/openradioss.md#a-pole-as-contact-partner) and its normal, with markers for nodes without a result field, thickness or yield stress |
| Inputs per mesh edge | besides the geometry, thickness and yield stress of the elements next to the edge |
| Targets | change of velocity, von Mises stress, plastic strain, membrane mean stress and bending stress from one state to the next (`residual_target`) |
| Loss | MSE of the targets, plus the length change of the mesh edges in a step and the velocity of the rigid body, each against the simulation, with weight 1 ([loss](../concepts/training.md#loss-and-optimizer-step)) |
| Frames | training and rollout from 10 ms (`[data] start_frame = 1`) |
| Training | 80 epochs, batch 4, learning rate 2.8 · 10⁻⁴ to 10⁻⁶ (cosine), noise 0.3 mm and 10 MPa, scheduled sampling, seed 42; **3 h 20 min** |
| Hardware | 1× NVIDIA RTX A4000 (16 GB); GPU 7.3 GB, RAM 5.5 GB in the main process |
| Model selection | rollouts of the 14 validation runs every 4 epochs, by the position error over the rollout and at its last step; epoch 80, the last (early stopping from epoch 41 on, not triggered) |
| Rollout | 9 autoregressive steps from 10 to 100 ms, 7 test runs |

The training data leave out the first frame on purpose: the step from the undeformed state to
the first deformed state, in which the beam hits the pole, is too large to be resolved with the
current physical modelling.

With scheduled sampling a growing share of the nodes takes its previous velocity and result
fields from the model's own prediction of the step before, as in the rollout, instead of from
the simulation: from none of the nodes at the start to 70 % after 40 of the 80 epochs
([training](../concepts/training.md#one-step-training)).

## Results

Test split, 7 runs ([metrics](../concepts/metrics.md#rollout-metrics)):

| Metric | All (pooled) | Median | Best | Worst |
|---|---|---|---|---|
| Position RMSE-1 [mm] | 0.784 ± 0.056 | 0.830 | 0.535 | 0.945 |
| Position RMSE-all [mm] | **4.80 ± 0.74** | 3.92 | 2.71 | 8.38 |
| Position error / displacement [%] | 7.76 | 6.43 | 4.16 | 14.8 |
| Position rel. L2 [×10⁻³] | 11.9 | 11.0 | 7.67 | 20.8 |
| Stress RMSE-1 [MPa] | 29.8 ± 0.7 | 29.3 | 27.5 | 33.0 |
| Stress RMSE-all [MPa] | 63.2 ± 4.6 | 59.3 | 51.7 | 89.0 |
| Stress R-RMSE-all [%] | **7.60 ± 0.53** | 7.16 | 6.28 | 10.5 |

Position error / displacement: the position error as a share of the displacement of the
ground truth since the initial state
([internal metrics](../concepts/metrics.md#internal-metrics)). Position rel. L2: as in the
crash studies of Nabian et al. and Akhare et al., on the coordinates of the dataset
([relative L2 position error](../concepts/metrics.md#relative-l2-position-error)).

![Error over the rollout](../assets/bumper_beam_error_per_step.png)

*Position error and stress RMSE per frame over the 7 test runs; dashed: the displacement of the
ground truth; grey: before the first prediction.*

## Discussion

Over the whole impact the model carries the deformation on its own predictions: the position
error stays at 4 to 15 % of the displacement in the test runs, 6 % in the median run, and the
stress error at 6 to 11 % of the largest stress of the run. The surface of the structure stays
smooth over the whole impact, close to the simulation. The results show
that the chain from OpenRadioss runs to the trained model works on a real crash example; the
processing of such data is at an early stage of development
([verification status](../concepts/graphs.md#general-meshes-and-run-parameters)).

A scientific evaluation of the results is planned for later releases of CrashDefNet and will be
published separately, as a paper or in another form.

Trained and rolled out with the development state of v1.1.0. How to run it:
[tutorial](../getting-started/bumper-beam.md).
