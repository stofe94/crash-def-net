# Contributing

Suggestions for the further development of CrashDefNet are welcome.

## Issues

Open an issue for bugs, questions and ideas, for example a new model architecture that
should be integrated. For a larger change, please open an issue before writing code, so
that the approach can be agreed first.

A proposal for a new input or data structure names the physical quantity it represents
and where it acts in the crash. CrashDefNet is refined to model the physics of crash
deformation in more detail, not to take any kind of data
([design principles](https://stofe94.github.io/crash-def-net/concepts/overview/#design-principles)).

## Pull requests

- Open pull requests against `main`.
- Every change is reviewed by the maintainer, and only changes approved by the
  maintainer are accepted.
- Accepted changes are taken over into the next release rather than merged directly,
  so the pull request is closed with a reference to the release that contains it.
- Contributions are licensed under the GNU AGPL-3.0, like the project
  ([license](https://github.com/stofe94/crash-def-net/blob/main/LICENSE)).

Before opening a pull request, please:

- keep the style of the surrounding code; `ruff check .` must pass (configuration in
  `pyproject.toml`)
- keep documentation and comments in English
- check that training with the example config runs, e.g.
  `python train/train.py --config examples/quick_start/train.toml --epochs 1`
- describe what the change does and how it was tested
