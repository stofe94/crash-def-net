# About CrashDefNet

## Background

CrashDefNet is developed by **Felix Stocker**, a mechanical engineer who came to data science
as a career changer over 4.5 years in CAE and crash simulation, one of them entirely focused
on research and development in deep learning for 3D geometries.
Today Felix works where CAE and data science meet and keeps training intensively in the field.

CrashDefNet began quite some time ago with a practical question: what does it take to change and
try out such a pipeline yourself? Research on learned crash simulation is still young, so what
counts is a framework in which changes can be made flexibly and quickly. Step by step, the
parts such a pipeline needs were collected and examined for their methods and feasibility, with
the aim of the simplest framework that serves this application. In mid-2026 the decision
followed to finalize and optimize it and to make it available to the community.

CrashDefNet builds on this experience as a **self-developed, independent tool:
complete in function, lean in code** ([what it does](../purpose.md#highlights)).

- **Fast integration.** A new architecture goes from the paper into a trained,
  evaluated model within one pipeline. MGN-T, published in 2026, is the first example.
- **Full control.** Every part, from data preparation to model and evaluation, can be
  changed as needed; the code base is compact and readable.
- **Focused on crash.** The application is crash deformation; CrashDefNet is not meant to
  become a general physics AI framework ([purpose](../purpose.md)).

Where it goes next: [roadmap](roadmap.md).

## Versioning

The version number stands for the completeness of the workflow: from 1.0.0 on, every stage
from the raw data to the evaluation is in place. What the predictions achieve shows on the
results pages ([deforming plate](../results/deforming-plate.md),
[bumper beam](../results/bumper-beam.md)). The processing of LS-DYNA, OpenRadioss and VTK
data and the general data structures are at an early stage; OpenRadioss runs are verified
on a first real crash example
([verification status](../concepts/graphs.md#general-meshes-and-run-parameters)). The
version number follows [semantic versioning](https://semver.org/) as a guideline:
compatibility is aimed for, not guaranteed, and the [changelog](changelog.md) names what
changes.

## References

- T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, P. W. Battaglia. *Learning Mesh-Based
  Simulation with Graph Networks.* ICLR 2021. [arXiv:2010.03409](https://arxiv.org/abs/2010.03409)
- M. M. Iparraguirre, I. Alfaro, D. González, E. Cueto. *MeshGraphNet-Transformer:
  Scalable Mesh-based Learned Simulation for Solid Mechanics.* 2026.
  [arXiv:2601.23177v4](https://arxiv.org/abs/2601.23177v4)
- H. Wu, H. Luo, H. Wang, J. Wang, M. Long. *Transolver: A Fast Transformer Solver for
  PDEs on General Geometries.* ICML 2024. [arXiv:2402.02366](https://arxiv.org/abs/2402.02366)
  (physics attention, used in the global processor of MGN-T)
- H. Luo et al. *Transolver++: An Accurate Neural Solver for PDEs on Million-Scale
  Geometries.* 2025. [arXiv:2502.02414](https://arxiv.org/abs/2502.02414)
  (eidetic states of the physics attention, used in MGN-T)
- M. A. Nabian, S. Chavare, D. Akhare, R. Ranade, R. Cherukuri, S. Tadepalli. *Automotive
  Crash Dynamics Modeling Accelerated with Machine Learning.* 2025.
  [arXiv:2510.15201](https://arxiv.org/abs/2510.15201)
  (relative L2 position error of the rollout)
- P. W. Battaglia et al. *Relational inductive biases, deep learning, and graph networks.* 2018.
  [arXiv:1806.01261](https://arxiv.org/abs/1806.01261)
  (full GraphNet block, used in the global block)
- A. Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
  (positional encoding, cited by the MGN-T paper)
- Dataset: `deforming_plate` from DeepMind's
  [MeshGraphNets release](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
  (downloaded separately, not redistributed)

## Acknowledgements

Thanks to DeepMind for publishing MeshGraphNets together with its datasets, and to the
authors of MGN-T for the detailed description of their architecture.

--8<-- "README.md:citation"

--8<-- "README.md:license"
