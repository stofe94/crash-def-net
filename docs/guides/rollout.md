# Roll out a model

A rollout applies a trained model to its own predictions, step after step over every
trajectory of a split, and measures how far it drifts from the simulation. How a step is
computed and which metrics it records: [rollout and metrics](../concepts/metrics.md). The
[CrashDefNet app](app.md) runs it on the test split and shows the result.

```bash
python predict/rollout.py --checkpoint checkpoints/<run>/best_model.pt \
    --data-dir data/<dataset> --split test \
    [--max-sequences N] [--start-frame F] [--end-frame F] [--max-steps N] [--no-export-vtu]
    [--no-one-step] [--device cuda] [--lazy | --no-lazy]
```

The rollout starts from the ground truth of the first frame the model was trained on and
ends with its last frame ([choose the frames](training.md#choose-the-frames)); a model
trained on all frames runs from the first to the last frame of the dataset. `--start-frame`
and `--end-frame` choose other raw frames that the dataset holds, with the motion into the
start frame taken from the ground truth; `--max-steps` ends the rollout after that many
steps. So one model can be rolled out on the same dataset from different points in time.

`--lazy` reads the frames from `dataset.db` on access instead of loading the split
(default: as in the training). The normalizer and the
[selected features](training.md#choose-the-features) come from the checkpoint. The rollout
checks that the dataset has the features the model was trained with, and warns if its
export settings (e.g. the stride) differ from those of the training dataset. Every
option: [command line reference](../reference/cli.md#rollout).

## Define a rollout in a config file

`--config FILE` takes the settings from a TOML file, so that a rollout can be repeated with
one command. `[evaluation]` chooses whether the [evaluation of the run](evaluation.md#evaluate-a-run)
follows on the new rollout folder in the same call, `[animation]` whether an
[animation](#animate-a-trajectory) of one trajectory does:

```toml
[rollout]
checkpoint = "checkpoints/<run>/best_model.pt"
data_dir   = "data/<dataset>"
split      = "test"

[evaluation]
enabled = true

[animation]
enabled = true
```

```bash
python predict/rollout.py --config rollout.toml
```

The rollout config of the quick start,
[`examples/quick_start/rollout.toml`](https://github.com/stofe94/crash-def-net/blob/main/examples/quick_start/rollout.toml), is a runnable
example with all settings. Options on the command line
override the file, e.g. `--split val` or `--no-evaluate`, and unknown keys stop the rollout.
Without a config, `--evaluate` and `--animate` do the same. The rollout does not start if the
folder for the evaluation or the animation is not empty, or if an animation is asked for without
the `.vtu` export. If the evaluation or the animation fails, the finished rollout stays, and the
failed step can be run on its own. The animation shows the trajectory at the median of the
position error, unless `trajectory` names another. Every key:
[configuration reference](../reference/configuration.md#rollout-config).

## Output

Every call writes into a folder of its own and never overwrites an earlier rollout:
`rollouts/<YYYYMMDD_HHMMSS>_<training run>_<checkpoint>_<split>/`
(`best_model.pt` appears as `best_model_ep<epoch>`). A folder given with
`--output-dir` must be new or empty.

| File | Content |
|---|---|
| `rollout_info.json` | checkpoint, dataset, options, command line, git commit, package versions, status, duration, metrics |
| `<split>_rollout_summary.json` | the errors per step, for every trajectory and as their mean ([metrics](../concepts/metrics.md)) |
| `rollout.log` | log of the run |
| `<split>/<trajectory>/tNNNN.vtu` | per frame, named by its raw frame of the run: predicted mesh with prediction, ground truth and error as point data; the frames up to the initial state are ground truth, error 0 (needs pyvista; skip with `--no-export-vtu`) |
| `<split>/<trajectory>/<trajectory>.vtu.series` | opens the whole trajectory in ParaView as a time series (time = raw frame of the run) |

Details of the JSON files: [file formats](../reference/file-formats.md#rollout-folder).

## Visualize in ParaView

Open `<split>/<trajectory>/<trajectory>.vtu.series`. The mesh points are the predicted
positions; the point data holds ground truth, prediction and error of every field
([list](../reference/file-formats.md#vtu-time-series)). Colouring by `position_error` or
`stress_error` shows where the prediction deviates.

## Animate a trajectory

```bash
python utils/animate_rollout.py rollouts/<a> [rollouts/<b> ...] --trajectory test_traj0136
```

Renders one trajectory as an animated GIF: the ground truth (FEM) next to the prediction
of every given rollout, coloured by von Mises stress on one scale (data without a stress
field: by its first result field), the actuator in grey
(node types from the rollout's `dataset.db`), without the lines that tie a free rigid body
together. The rollouts need the `.vtu` export and
must come from the same dataset with the same steps; the first one also provides the
ground truth. Writes `rollout_<trajectory>.gif` into a new folder
`evaluations/<YYYYMMDD_HHMMSS>_animation/`, or to the file given with `--output`.

| Option | Default | |
|---|---|---|
| `--trajectory` | first trajectory of the split | trajectory to render |
| `--labels` | architecture and parameter count | panel titles, one per rollout |
| `--stress-max` | 99.9th percentile of the ground truth, rounded up: from 50 on to the next multiple of 50, below (e.g. a stress in GPa) to 1, 2 or 5 times a power of ten | upper end of the colour scale in kPa, for data in MPa or GPa in that unit (other fields: their unit) |
| `--output` | new folder in `evaluations/` | GIF file |

Rendering runs off-screen in software (Mesa), so it needs neither a GPU nor a display
and gives the same frames on every machine; a `deforming_plate` trajectory of 98 steps with
three panels takes about 20 s.
Needs `pyvista`.

## Next steps

- [Evaluate the run](evaluation.md#evaluate-a-run) from the rollout folder.
- [Compare rollouts with the papers](evaluation.md#compare-with-the-papers).
