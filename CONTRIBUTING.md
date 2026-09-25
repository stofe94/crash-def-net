# Contributing

Suggestions for the further development of CrashDefNet are welcome.

## Issues

Open an issue for bugs, questions and ideas, for example a new model architecture that
should be integrated. For a larger change, please open an issue before writing code, so
that the approach can be agreed first.

## Pull requests

- Open pull requests against `main`.
- Every change is reviewed by the maintainer, and only changes approved by the
  maintainer are accepted.
- Accepted changes are taken over into the next release rather than merged directly,
  so the pull request is closed with a reference to the release that contains it.
- Contributions are licensed under the Apache License 2.0, like the project
  (section 5 of the [license](LICENSE)).

Before opening a pull request, please:

- keep the style of the surrounding code; `ruff check .` must pass (configuration in
  `pyproject.toml`)
- keep documentation and comments in English
- check that training with the example config runs, e.g.
  `python train/train.py --config config.example.toml --epochs 1`
- describe what the change does and how it was tested
