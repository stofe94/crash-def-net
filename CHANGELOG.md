# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [0.1.0] — 2026-09-24

First public release, developed and validated on DeepMind's `deforming_plate` benchmark.

### Added

- **Data pipeline:** streaming export of the DeepMind TFRecord data into a compact
  SQLite dataset (`dataset.db`, `normalizer.json`, `metadata.json`) with train/val/test
  split, selectable stride and number of frames; in-memory or lazy loading; one graph
  per time step with mesh and world (contact) edges; optional history features.
- **Models:** MeshGraphNet (Pfaff et al., ICLR 2021) with mesh and world edges, and
  MeshGraphNet-Transformer (Iparraguirre et al., 2026) with physics attention,
  implemented from the paper with all assumptions documented; optional residual target.
- **Training:** single-step training with consistent position noise, exponential /
  plateau / cosine LR schedules (`decay_steps = "auto"`), gradient clipping, rollout
  validation with selectable selection metric (`mse`, `pos_rmse`, `pos_rmse_final`),
  early stopping, resumable checkpoints, test metrics including a full test rollout.
- **Rollout:** autoregressive rollout with contact edges rebuilt each step; metrics of
  the MGN-T paper (RMSE-1, RMSE-all, R-RMSE); `.vtu` export with `.vtu.series` for
  ParaView; one folder per call with `rollout_info.json` for traceability.
- **Evaluation:** `utils/compare_paper.py` compares rollouts with the tables of the
  MGN-T paper (figures, tables, Markdown report, CSV/JSON); `utils/plot_losses.py`.
- **Example config:** `config.example.toml`, a commented CPU quick test that covers
  both architectures.
- **Docker:** GPU training image (CUDA 12.4) with docker compose.
- **Documentation:** README, `docs/` (installation, data, models, adding an
  architecture, training, rollout and evaluation, Docker, first results).

[0.1.0]: https://github.com/stofe94/crash-def-net/releases/tag/v0.1.0
