# Explore a dataset

Before or after a training: how do the trajectories of an export differ, are the data
sound, and how far do val and test lie from what the training sees?

```bash
python utils/explore_data.py data/<dataset>
python utils/explore_data.py data/<dataset> --rollout rollouts/<a> --rollout rollouts/<b>
```

Describes every trajectory of an export by its mesh, loading and response, checks the
data and compares the splits. `--rollout` (repeatable; rollouts on this dataset) relates the
error of every trajectory to these descriptors by rank correlation. The DB is read one trajectory and
one frame at a time, so memory does not grow with the dataset; the 150 trajectories of
the `deforming_plate` protocol export take about 2 minutes. Needs an export in the current format.

## Output

A new folder `evaluations/<YYYYMMDD_HHMMSS>_<dataset>_data_exploration/`, or the new or
empty folder given with `--output-dir`:

| File | Content |
|---|---|
| `01_descriptors.png` | every descriptor per split: quartiles and median, whiskers from the smallest to the largest trajectory |
| `02_over_time.png` | displacement, largest stress, contact pairs and actuator travel over the frames: median and 10–90 % of each split |
| `03_targets.png` | the targets as the model sees them, normalized with `normalizer.json`, per split |
| `04_coverage.png` | distance of every trajectory to its nearest train trajectory; Kolmogorov–Smirnov distance train–val and train–test per descriptor |
| `05_error_vs_data.png` | with `--rollout`: error per trajectory against the descriptors with the strongest rank correlation |
| `exploration.md` | dataset, findings, data checks, descriptors per split, targets, val and test trajectories, rank correlations |
| `trajectories.csv`, `exploration.json` | the values, in SI units |

## Descriptors and checks

Descriptors: nodes, mesh edge length, extent of structure and actuator; actuator travel
and speed, first contact, frames with contact, contact and self-contact pairs;
displacement (RMS, of a node, at the end), node velocity, stress (largest, RMS), smallest
element volume.

Data checks: non-finite values, negative von Mises stress, motion of clamped nodes,
degenerate and inverted elements, trajectories without contact, identical trajectories
and meshes shared between splits, the largest normalized target.

The definitions are in the module docstring of
[`explore_data.py`](https://github.com/stofe94/crash-def-net/blob/main/utils/explore_data.py);
the findings are measurements, not verdicts.
