# The CrashDefNet app

A graphical interface for CrashDefNet: a page in the browser, on your own machine, that leads
through the workflow without typing commands.

!!! warning "Early stage"
    The app is made mainly for the [quick start](../getting-started/quick-start.md), and later
    versions will develop it further. It can already be used beyond it, to train with any
    configuration, predict with any trained model and look at the results; what it does not
    cover stays on the [command line](../reference/cli.md).

```bash
python utils/app.py
```

The page opens at `http://127.0.0.1:8765` (another port: `--port`; stop the app with
Ctrl+C). Needs the [installation](../getting-started/installation.md). It serves the page
only on this computer, so data and results stay there, and it needs no package beyond the
installation.

Every step runs the [command line tool](../reference/cli.md) you would run yourself as a
child process and shows its log, so the outputs are the same files in the same folders. Runs
started on the command line, or on another machine, appear in the app, and the other way
round.

## A first run

Without any data of your own, the five steps with their defaults are the
[quick start](../getting-started/quick-start.md): they download `deforming_plate`, create the
small dataset, train MGN-T in a small size on it, and predict plates the model has never
seen, in minutes on a CPU.

## Beyond the first run

The app can already work with your own datasets, configurations and runs:

| Step | What it does with your own work |
|---|---|
| Raw data | only for `deforming_plate`; skip it with datasets of your own |
| Dataset | lists every export in `data/`, also of your own simulations: [OpenRadioss runs](openradioss.md), [LS-DYNA runs](ls-dyna.md) and [VTK time series](vtk.md) are exported on the command line. New datasets from `deforming_plate`: any number of simulations per split and time step |
| Training | any configuration: `examples/quick_start/train.toml` (it selects the [features](training.md#choose-the-features) of `deforming_plate`; for other data use a copy adapted to them), your own in `configs/` or `configs/train/`, or those in `protocols/`, with their sizes, learning rate, noise and all other settings; architecture and number of epochs can be changed on the page. Progress, remaining time and the loss of every epoch while it trains. Every run goes into its own folder `checkpoints/<architecture>_<time>/` |
| Prediction | lists every run in `checkpoints/` with its dataset and test error, rolls the chosen one out on the test split and renders the animation of its first simulation |
| Result | lists every rollout in `rollouts/`: the errors per simulation, the animation of any simulation next to its prediction, and the folder of the `.vtu` files for ParaView. The stress in kPa, for data in MPa or GPa in that unit ([units](export.md#units)) |

One step runs at a time; a running step can be cancelled. Closing the page does not stop it:
open the page again and it shows the step as it runs. Stopping the app stops the step.
