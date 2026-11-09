# Roadmap

Further refinements of the data structures, and of the architectures built on them, aim
at modelling the physical process of crash deformation better and in more detail
([design principles](../concepts/overview.md#design-principles)).

This roadmap names directions only: technical details of planned developments are
deliberately not published in advance, so that the lead from this development work stays
with the project.

- Further recent architectures, integrated and tested as they are published
- More complex crash scenarios, with all architectures side by side
- Training on the full `deforming_plate` training split, as in the papers
- Less memory and computation per node
- Scaling to full-vehicle meshes as soon as the data require it
- A published test suite for every stage of the pipeline
- Verification of the general data structures and of the readers on further real crash
  datasets ([verification status](../concepts/graphs.md#general-meshes-and-run-parameters))
- Further development of the CrashDefNet app ([the app](../guides/app.md))
- A finer contact search ([world edges today](../concepts/graphs.md#graph-per-time-step-t))
- Further data structures in the graph, as far as new studies and more detailed crash
  scenarios require them
