#!/usr/bin/env python3
"""
check_protocol.py — Check rollouts against a comparison protocol and compare them
with a reference trajectory by trajectory (docs/comparison-protocol.md).

The first rollout folder is the reference (the MeshGraphNets baseline), every
further one a candidate. For each rollout the training config stored with the
checkpoint is compared with the protocol config:

- fixed settings must be equal: budget, optimizer, schedule, noise, loss,
  validation, model selection and early stopping, seed, ...;
- declared variations are allowed and listed: the architecture and its own
  parameters and the world edge variant;
- machine settings (data loading, device, checkpointing) are ignored.

The evaluated checkpoint must be the selected model (``best_model.pt``, the epoch
flagged ``selected`` in ``history.json``, in older histories the last ``is_best``),
the training must have ended regularly (``end`` in the history; older runs
without it get a note), and every rollout must cover the whole test split of the
same dataset (content fingerprint, for older exports the SHA-256 of the file)
with the same trajectories and steps as the reference. The candidates are then
compared with the reference per trajectory (paired): which model has the lower
error, and how often.

Prints a Markdown report; the exit code is 1 if a rollout violates the protocol.

Usage
-----
    python utils/check_protocol.py rollouts/<baseline> rollouts/<new> [...] \\
        [--exclude test_traj0038] [--protocol protocols/deforming_plate_s8_t40.toml]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset_preprocessor.dataset import dataset_fingerprint  # noqa: E402
from train.config import TOML_KEYS, TrainingConfig  # noqa: E402

DEFAULT_PROTOCOL = _REPO_ROOT / "protocols" / "deforming_plate_s8_t40.toml"
PROTOCOL_SPLIT = "test"
# Display names and paired metrics as in compare_paper.py (not imported from there:
# it loads matplotlib, which this check does not need).
ARCHITECTURE_NAMES = {"mgn_t": "MGN-T", "meshgraphnets": "MGN", "meshgraphnet": "MGN"}
PAIRED_METRICS = [  # key in paper_metrics, title, factor to the unit, unit
    ("q_rmse_all", "q RMSE-all (whole rollout)", 1e3, "mm"),
    ("stress_rmse_all", "σ RMSE-all (whole rollout)", 1e-3, "kPa"),
    ("q_rmse_1", "q RMSE-1 (one step)", 1e3, "mm"),
    ("stress_rmse_1", "σ RMSE-1 (one step)", 1e-3, "kPa"),
]
# Relative differences up to this count as equal: numerical noise between machines
# (the baseline rolled out again on the CPU differed by less than 0.01 %).
EQUAL_TOLERANCE = 1e-3

# Settings of the machine and the bookkeeping, without influence on the result.
MACHINE_FIELDS = {"data_dir", "num_dataloader_workers", "lazy_loading", "cache_frames",
                  "static_cache_mb", "device", "checkpoint_dir", "save_every", "log_every",
                  "resume", "config_file", "config_file_sha256", "config_overrides"}
# Parameters of an architecture: the fields of the sections [model.<architecture>].
ARCHITECTURE_FIELDS = {field for key, field in TOML_KEYS.items()
                       if key.startswith("model.") and key.count(".") >= 2}
# Allowed variations; they are listed with the result.
DECLARED_FIELDS = {"architecture", "world_edges", "world_edge_exclude_hops"} | ARCHITECTURE_FIELDS


@dataclasses.dataclass
class Rollout:
    """A finished rollout folder: ``rollout_info.json`` and its summary."""
    folder: Path
    info: Dict
    summary: Dict

    @property
    def model(self) -> Dict:
        return self.info["model"]

    @property
    def split(self) -> str:
        return self.info["data"]["split"]


def load_rollout(folder: Path) -> Rollout:
    """Raises ValueError unless ``folder`` holds a finished rollout."""
    info_path = folder / "rollout_info.json"
    if not info_path.exists():
        raise ValueError(f"no rollout_info.json in {folder}")
    with open(info_path) as f:
        info = json.load(f)
    summary_name = (info.get("results") or {}).get(
        "summary_file", f"{info['data']['split']}_rollout_summary.json")
    if info.get("status") != "finished" or not (folder / summary_name).exists():
        raise ValueError(f"rollout not finished ({info.get('status')}): {folder}")
    with open(folder / summary_name) as f:
        return Rollout(folder, info, json.load(f))


def number(value: float) -> str:
    """3 significant digits, no exponent."""
    if value == 0:
        return "0"
    decimals = max(2 - int(math.floor(math.log10(abs(value)))), 0)
    return f"{value:.{decimals}f}"


@dataclasses.dataclass
class Check:
    """Findings for one rollout."""
    violations: List[str] = dataclasses.field(default_factory=list)
    variations: List[str] = dataclasses.field(default_factory=list)
    notes: List[str] = dataclasses.field(default_factory=list)


# ── files ────────────────────────────────────────────────────────────────────

_SHA256: Dict[Path, str] = {}


def sha256(path: Path) -> str:
    if path not in _SHA256:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        _SHA256[path] = digest.hexdigest()
    return _SHA256[path]


def dataset_id(record: Dict) -> str:
    """Identity of a recorded dataset.db: its content fingerprint (export format v4)
    or, for older exports, the SHA-256 of the file."""
    return record.get("fingerprint") or record["sha256"]


def _local_dataset_id(db: Path, record: Dict) -> str:
    """The same identity of a local dataset.db, of the kind recorded."""
    if "fingerprint" in record:
        return dataset_fingerprint(db) or ""
    return sha256(db)


def _local_path(recorded: str, *fallback: str) -> List[Path]:
    """The recorded path, then the same name under this repository (the rollout
    may have been made in another clone)."""
    paths = [Path(recorded)]
    if not paths[0].is_absolute():
        paths[0] = _REPO_ROOT / paths[0]
    paths.append(_REPO_ROOT.joinpath(*fallback))
    return paths


def train_samples(rollout: Rollout) -> Optional[int]:
    """Train samples of the rollout's dataset, from a metadata.json next to a
    dataset.db with the recorded fingerprint / SHA-256."""
    data = rollout.info["data"]
    name = Path(data["data_dir"]).name
    for data_dir in _local_path(data["data_dir"], "data", name):
        db, metadata = data_dir / "dataset.db", data_dir / "metadata.json"
        if (db.exists() and metadata.exists()
                and _local_dataset_id(db, data["dataset_db"]) == dataset_id(data["dataset_db"])):
            with open(metadata) as f:
                return json.load(f).get("sample_counts", {}).get("train")
    return None


def training_history(rollout: Rollout) -> Optional[List[Dict]]:
    """history.json of the training run, next to a checkpoint with the recorded SHA-256."""
    checkpoint = rollout.model["checkpoint"]
    name = Path(checkpoint["path"]).name
    for path in _local_path(checkpoint["absolute_path"], "checkpoints",
                            rollout.model["training_run"], name):
        history = path.parent / "history.json"
        if path.exists() and history.exists() and sha256(path) == checkpoint["sha256"]:
            with open(history) as f:
                return json.load(f)
    return None


# ── checks ───────────────────────────────────────────────────────────────────

def check_config(rollout: Rollout, protocol: TrainingConfig, check: Check) -> None:
    """Training config of the rollout's checkpoint against the protocol config."""
    run = TrainingConfig.from_dict(rollout.model["training_config"])
    run_unused, protocol_unused = run.unused_fields(), protocol.unused_fields()
    own_parameters = []   # of an architecture other than the protocol's
    for field in dataclasses.fields(TrainingConfig):
        name = field.name
        if name in MACHINE_FIELDS or name in run_unused:
            continue
        expected, actual = getattr(protocol, name), getattr(run, name)
        if name in protocol_unused:
            # In effect only in the run: a declared variation brings its own settings
            # (e.g. world_edge_exclude_hops); other switches are reported themselves.
            if name in ARCHITECTURE_FIELDS:
                own_parameters.append(f"{name} {actual!r}")
            elif name in DECLARED_FIELDS:
                check.variations.append(f"`{name}`: {actual!r} (not used by the protocol)")
            continue
        if name == "lr_decay_steps" and expected == "auto":
            samples = train_samples(rollout)
            if samples is None:
                check.notes.append(f"`lr_decay_steps` = {actual} not checked against "
                                   "\"auto\": dataset with metadata.json not found")
                continue
            expected = run.epochs * math.ceil(samples / run.batch_size)
            if actual != expected:
                check.violations.append(f"`lr_decay_steps`: {actual}, protocol \"auto\" = "
                                        f"{expected} (epochs x steps per epoch)")
            continue
        if actual == expected:
            continue
        text = f"`{name}`: {actual!r} (protocol: {expected!r})"
        if name in DECLARED_FIELDS:
            check.variations.append(text)
        else:
            check.violations.append(text)
    if own_parameters:
        check.variations.append(f"parameters of `{run.architecture}`: "
                                + ", ".join(own_parameters))


def check_training_run(rollout: Rollout, check: Check) -> None:
    """The evaluated checkpoint must be the selected model (best_model.pt), and
    the training must have ended regularly (early stopping or all epochs)."""
    history = training_history(rollout)
    if not history:
        check.notes.append("selected epoch and end of training not checked: history.json "
                           "of the training run not found")
        return
    # "selected" marks the final choice (the selection by several criteria may
    # return to an earlier epoch); older histories only flag every new best.
    flag = "selected" if any("selected" in record for record in history) else "is_best"
    chosen = [record["epoch"] for record in history if record.get(flag)]
    if not chosen:
        check.notes.append("selected epoch not checked: no selected epoch in history.json")
    elif rollout.model["epoch"] != chosen[-1]:
        check.violations.append(f"evaluated epoch {rollout.model['epoch']}, but the selected "
                                f"model (best_model.pt) is epoch {chosen[-1]}")

    epochs = TrainingConfig.from_dict(rollout.model["training_config"]).epochs
    if any("end" in record for record in history) or len(history) >= epochs:
        return
    text = f"training ended after {len(history)} of {epochs} epochs without a recorded end"
    if flag == "selected":
        # Written by a version that records the end (and flags the selected
        # epoch in every record): the run was aborted.
        check.violations.append(f"{text} (aborted)")
    else:
        check.notes.append(f"{text}: aborted, or trained before the end was recorded")


def check_data(rollout: Rollout, reference: Rollout, check: Check) -> None:
    """Whole test split, same dataset file, trajectories and steps as the reference."""
    data = rollout.info["data"]
    if rollout.split != PROTOCOL_SPLIT:
        check.violations.append(f"split {rollout.split!r} (protocol: {PROTOCOL_SPLIT!r})")
    if data.get("max_sequences") is not None or data.get("max_steps") is not None:
        check.violations.append("rollout limited by --max-sequences / --max-steps")
    if not data.get("matches_training_dataset", True):
        check.violations.append("rolled out on a dataset with other export settings than "
                                "the training dataset")
    if rollout is reference:
        return
    reference_data = reference.info["data"]
    if dataset_id(data["dataset_db"]) != dataset_id(reference_data["dataset_db"]):
        check.violations.append("other dataset_db than the reference (fingerprint / SHA-256)")
    if data["normalizer"]["sha256"] != reference_data["normalizer"]["sha256"]:
        check.violations.append("other normalizer file than the reference (SHA-256)")
    if sorted(rollout.summary["sequences"]) != sorted(reference.summary["sequences"]):
        check.violations.append("other trajectories than the reference")
    steps = len(rollout.summary["position_rmse_per_step"])
    reference_steps = len(reference.summary["position_rmse_per_step"])
    if steps != reference_steps:
        check.violations.append(f"{steps} rollout steps (reference: {reference_steps})")


def check_rollout(rollout: Rollout, reference: Rollout, protocol: TrainingConfig) -> Check:
    check = Check()
    check_config(rollout, protocol, check)
    check_training_run(rollout, check)
    check_data(rollout, reference, check)
    return check


# ── report ───────────────────────────────────────────────────────────────────

def label(rollout: Rollout) -> str:
    architecture = rollout.model["architecture"]
    return (f"{ARCHITECTURE_NAMES.get(architecture, architecture)} Ep. {rollout.model['epoch']}"
            f" — `{rollout.folder.name}`")


def check_section(title: str, rollout: Rollout, check: Check) -> List[str]:
    verdict = "violates the protocol" if check.violations else "conforms to the protocol"
    lines = [f"### {title}: {label(rollout)}", "",
             f"{rollout.model['parameters'] / 1e6:.2f} M parameters — **{verdict}**", ""]
    for heading, items in (("Violations", check.violations),
                           ("Declared variations", check.variations),
                           ("Notes", check.notes)):
        if items:
            lines += [f"{heading}:", ""] + [f"- {item}" for item in items] + [""]
    return lines


def lower_error(reference: float, candidate: float) -> str:
    """"candidate", "reference" or "equal" (within ``EQUAL_TOLERANCE``)."""
    if abs(candidate - reference) <= EQUAL_TOLERANCE * max(abs(reference), abs(candidate)):
        return "equal"
    return "candidate" if candidate < reference else "reference"


def tally(winners: Dict[str, str], sequences: Sequence[str]) -> str:
    """Count of lower errors over ``sequences``; established only if all agree."""
    lower = sum(winners[sid] == "candidate" for sid in sequences)
    higher = sum(winners[sid] == "reference" for sid in sequences)
    n = len(sequences)
    if n > 1 and lower == n:
        verdict = "candidate lower on every trajectory"
    elif n > 1 and higher == n:
        verdict = "reference lower on every trajectory"
    elif lower == higher == 0:
        verdict = "no difference"
    else:
        verdict = "not established (needs every trajectory)"
    return f"candidate lower on {lower} of {n}, reference on {higher} of {n} — {verdict}"


def paired_section(candidate: Rollout, reference: Rollout,
                   exclude: Sequence[str]) -> List[str]:
    sequences = sorted(reference.summary["sequences"])
    kept = [sid for sid in sequences if sid not in exclude]
    lines = [f"### {label(candidate)}", "",
             f"Reference: {label(reference)}. Difference = candidate − reference; "
             "negative = candidate better.", ""]
    for key, title, factor, unit in PAIRED_METRICS:
        lines += [f"**{title}** [{unit}]", "",
                  "| Trajectory | Reference | Candidate | Difference | Lower error |",
                  "|---|---|---|---|---|"]
        winners = {}
        for sid in sequences:
            ref = reference.summary["sequences"][sid]["paper_metrics"][key] * factor
            cand = candidate.summary["sequences"][sid]["paper_metrics"][key] * factor
            winners[sid] = lower_error(ref, cand)
            sign = "+" if cand > ref else "−" if cand < ref else "±"
            name = f"{sid} (excluded)" if sid in exclude else sid
            lines.append(f"| {name} | {number(ref)} | {number(cand)} | "
                         f"{sign}{number(abs(cand - ref))} | {winners[sid]} |")
        lines += ["", f"- All trajectories: {tally(winners, sequences)}"]
        if len(kept) < len(sequences):
            lines.append(f"- Without {', '.join(s for s in sequences if s in exclude)}: "
                         f"{tally(winners, kept)}")
        lines.append("")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check rollouts against a comparison protocol and compare them "
                    "with the reference per trajectory (docs/comparison-protocol.md).")
    parser.add_argument("reference", type=Path, help="rollout folder of the reference")
    parser.add_argument("candidates", nargs="*", type=Path, help="further rollout folders")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL,
                        help="protocol training config (default: %(default)s)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="trajectories counted separately, e.g. test_traj0038")
    args = parser.parse_args(argv)

    try:
        rollouts = [load_rollout(folder) for folder in [args.reference, *args.candidates]]
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    reference, candidates = rollouts[0], rollouts[1:]
    protocol = TrainingConfig.from_toml(str(args.protocol))
    checks = [check_rollout(rollout, reference, protocol) for rollout in rollouts]

    data = reference.info["data"]
    protocol_path = args.protocol.resolve()
    if protocol_path.is_relative_to(_REPO_ROOT):
        protocol_path = protocol_path.relative_to(_REPO_ROOT)
    lines = [f"# Protocol check: {args.protocol.stem}", "",
             f"- Protocol: `{protocol_path}`",
             f"- Dataset: `{Path(data['data_dir']).name}`, dataset.db "
             f"`{dataset_id(data['dataset_db'])[:12]}…`, {reference.split} split, "
             f"{len(reference.summary['sequences'])} trajectories, "
             f"{len(reference.summary['position_rmse_per_step'])} steps", "",
             "## Conformity", ""]
    lines += check_section("Reference", reference, checks[0])
    for rollout, check in zip(candidates, checks[1:]):
        lines += check_section("Candidate", rollout, check)
    if candidates:
        n = len(reference.summary["sequences"])
        chance = (f"With equal models one of them is lower on all {n} by chance in "
                  f"{100 * 2 * 0.5 ** n:.1f} % of cases"
                  + (f", on {n - 1} of {n} in {100 * 2 * n * 0.5 ** n:.1f} %." if n >= 3 else "."))
        lines += ["## Paired comparison with the reference", "",
                  "A difference is established only if one model has the lower error on "
                  f"every trajectory. {chance}", ""]
        for candidate in candidates:
            lines += paired_section(candidate, reference, args.exclude)
    print("\n".join(lines).rstrip())
    return 1 if any(check.violations for check in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
