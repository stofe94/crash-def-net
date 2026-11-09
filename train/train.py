#!/usr/bin/env python3
"""
train.py — Train a model (MeshGraphNets or MGN-T, ``[model] architecture``) on a ``dataset.db``.

All settings come from the TOML config; command-line options override it only
when they are given explicitly.

Usage
-----
    python train/train.py --config examples/quick_start/train.toml
    python train/train.py --config examples/quick_start/train.toml --architecture meshgraphnets
    python train/train.py --epochs 50 --device cuda --data-dir data/deforming_plate_s10

Output in ``[checkpointing].checkpoint_dir``: ``best_model.pt``, periodic
``checkpoint_epoch_N.pt``, ``history.json`` (per-epoch losses), the run's log
and config, and ``test_metrics.json``.
"""

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))   # packages: dataset_preprocessor, models, utils, ...

import torch  # noqa: E402

from dataset_preprocessor.dataset import load_statistics  # noqa: E402
from dataset_preprocessor.export import window_statistics  # noqa: E402
from dataset_preprocessor.graph_features import select_metadata, select_statistics  # noqa: E402
from train.config import ARCHITECTURES, TrainingConfig  # noqa: E402
from train.trainer import (  # noqa: E402
    Trainer,
    checkpoint_normalizer,
    create_dataloaders,
    create_model,
    logger,
    new_run_name,
    setup_logging,
)

DEFAULT_CONFIG = _REPO_ROOT / "examples" / "quick_start" / "train.toml"


def build_parser() -> argparse.ArgumentParser:
    """Command line options; the reference in docs/reference/cli.md is generated from them.
    Only options given on the command line end up in the namespace."""
    parser = argparse.ArgumentParser(
        description="Train a model on an exported dataset (config: TOML, the options "
                    "below override it).",
        argument_default=argparse.SUPPRESS)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help=f"TOML config (default: {DEFAULT_CONFIG.relative_to(_REPO_ROOT).as_posix()})")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir",
                        help="dataset directory with dataset.db / metadata.json / "
                             "normalizer.json ([data].data_dir)")
    parser.add_argument("--architecture", choices=ARCHITECTURES,
                        help="model architecture, sized by its section of the config "
                             "([model].architecture)")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir",
                        help="output directory ([checkpointing].checkpoint_dir)")
    parser.add_argument("--epochs", type=int, help="training epochs ([training].epochs)")
    parser.add_argument("--batch-size", dest="batch_size", type=int,
                        help="graphs per optimizer step ([training].batch_size)")
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float,
                        help="initial learning rate ([training].learning_rate)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"],
                        help="auto: the GPU if available ([training].device)")
    parser.add_argument("--seed", type=int, help="random seed ([training].seed)")
    parser.add_argument("--num-workers", dest="num_dataloader_workers", type=int,
                        help="DataLoader workers ([data].num_dataloader_workers)")
    parser.add_argument("--resume", help="checkpoint to continue from")
    return parser


def main(argv=None) -> None:
    args = vars(build_parser().parse_args(argv))
    config_path = args.pop("config", str(DEFAULT_CONFIG))
    config = TrainingConfig.from_toml(config_path, overrides=args)
    run_name = new_run_name()
    setup_logging(Path(config.checkpoint_dir) / f"{run_name}.log")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    data_dir = Path(config.data_dir)
    with open(data_dir / "metadata.json", encoding="utf-8") as f:
        export_metadata = json.load(f)
    metadata = select_metadata(export_metadata, config.features)
    if config.resume:   # continue with the scale the weights were trained with
        checkpoint = torch.load(config.resume, map_location="cpu", weights_only=False)
        for key in ("node_feature_names", "target_feature_names", "global_feature_names"):
            if checkpoint["metadata"].get(key) != metadata.get(key):
                raise ValueError(f"{config.resume} was trained with {key} "
                                 f"{checkpoint['metadata'].get(key)}, this run has "
                                 f"{metadata.get(key)}: a run continues with its features")
        normalizer, _ = checkpoint_normalizer(
            checkpoint, data_dir, config.resume, config.features,
            windowed=config.start_frame is not None or config.end_frame is not None)
        del checkpoint
    elif config.start_frame is not None or config.end_frame is not None:
        # The scale of the frames this run uses, e.g. without the jump at the impact.
        normalizer = select_statistics(window_statistics(data_dir, config.start_frame,
                                                         config.end_frame),
                                       export_metadata, config.features)
        logger.info(f"Statistics of the node inputs and targets fitted on frames "
                    f"{config.start_frame or 'first'} to {config.end_frame or 'last'} of the "
                    "train split")
    else:
        normalizer = load_statistics(data_dir, config.features)

    train_loader, val_loader, test_loader = create_dataloaders(config, normalizer)
    config.resolve_lr_decay_steps(math.ceil(len(train_loader) / config.accumulation_steps))
    model = create_model(config, metadata, normalizer)
    trainer = Trainer(model, config, normalizer, metadata, run_name)

    start_epoch = trainer.load_checkpoint(config.resume) if config.resume else 0
    trainer.train(train_loader, val_loader, start_epoch)
    trainer.evaluate_and_save(test_loader)


if __name__ == "__main__":
    main()
