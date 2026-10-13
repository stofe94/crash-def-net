#!/usr/bin/env python3
"""
train.py — Train a MeshGraphNet on a deforming_plate ``dataset.db``.

All settings come from the TOML config; command-line options override it only
when they are given explicitly.

Usage
-----
    python train/train.py --config config.example.toml
    python train/train.py --epochs 50 --device cuda --data-dir data/deforming_plate_s10

Output in ``[checkpointing].checkpoint_dir``: ``best_model.pt``, periodic
``checkpoint_epoch_N.pt``, ``history.json`` (per-epoch losses), the run's log
and config, and ``test_metrics.json``.
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))   # packages: dataset_preprocessor, models, utils, ...

import torch  # noqa: E402

from train.config import TrainingConfig  # noqa: E402
from train.trainer import (  # noqa: E402
    Trainer,
    create_dataloaders,
    create_model,
    new_run_name,
    setup_logging,
)
from utils.normalizer import Normalizer  # noqa: E402

DEFAULT_CONFIG = _REPO_ROOT / "config.example.toml"


def parse_args(argv=None) -> argparse.Namespace:
    """Only options given on the command line end up in the namespace."""
    parser = argparse.ArgumentParser(
        description="Train a MeshGraphNet on deforming_plate (config: TOML, "
                    "options below override it).",
        argument_default=argparse.SUPPRESS)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help=f"TOML config (default: {DEFAULT_CONFIG.relative_to(_REPO_ROOT)})")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir",
                        help="dataset directory with dataset.db / metadata.json / "
                             "normalizer.json ([data].data_dir)")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir",
                        help="output directory ([checkpointing].checkpoint_dir)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", dest="num_dataloader_workers", type=int,
                        help="DataLoader workers ([data].num_dataloader_workers)")
    parser.add_argument("--resume", help="checkpoint to continue from")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = vars(parse_args(argv))
    config_path = args.pop("config", str(DEFAULT_CONFIG))
    config = TrainingConfig.from_toml(config_path, overrides=args)
    run_name = new_run_name()
    setup_logging(Path(config.checkpoint_dir) / f"{run_name}.log")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    data_dir = Path(config.data_dir)
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    normalizer = Normalizer.load(str(data_dir / "normalizer.json"))

    train_loader, val_loader, test_loader = create_dataloaders(config, normalizer)
    config.resolve_lr_decay_steps(len(train_loader))
    model = create_model(config, metadata, normalizer)
    trainer = Trainer(model, config, normalizer, metadata, run_name)

    start_epoch = trainer.load_checkpoint(config.resume) if config.resume else 0
    trainer.train(train_loader, val_loader, start_epoch)
    trainer.evaluate_and_save(test_loader)


if __name__ == "__main__":
    main()
