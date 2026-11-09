#!/usr/bin/env python3
"""
app.py — The CrashDefNet app: the workflow as a local web page, without a terminal.

Starts a web server on this machine and opens ``app.html`` in the browser. At an early
stage: made mainly for a first run (docs/getting-started/quick-start.md), it can already
work with the datasets, configs, models and predictions of the repository
(docs/guides/app.md), in five steps:

1. Raw data: what DeepMind's ``deforming_plate`` holds, with the example plates of
   ``docs/assets/plates/``, and its ``meta.json`` and ``test.tfrecord`` (0.8 GB, 100
   plates) into ``datasets/deforming_plate/raw_dataset/``, if missing.
2. Dataset: any export in ``data/``, or a new one from ``test.tfrecord`` with the
   number of plates per split and every n-th time step.
3. Training: a config of the repository (``CONFIG_PATTERNS``) on the chosen dataset,
   with its architecture and number of epochs; every run into
   ``checkpoints/<architecture>_<time>/``.
4. Prediction: the rollout of any trained model on the test split of its dataset, then
   the animation of its first simulation, so that step 5 opens with the picture.
5. Result: the errors per simulation and the animation of any simulation next to its
   prediction; every animation into
   ``evaluations/<time>_animation_<rollout>/``, whose name links it to the rollout.

Every step runs the command line tool a user would run (export, train, rollout,
animate_rollout) as a child process from the repository root, so the app adds
nothing to the pipeline itself; the page shows its progress and log. One step
runs at a time, and nothing is overwritten: an existing dataset is chosen, not
exported again.

The server listens on 127.0.0.1 only. It answers requests addressed to
127.0.0.1 or localhost, and starts a step only on a JSON request, which a
browser sends from another web page only after a CORS preflight that the server
does not answer: pages open in the browser cannot start steps.

Usage
-----
    python utils/app.py              # opens http://127.0.0.1:8765
    python utils/app.py --port 8800 --no-browser
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.results import current_metric_names  # noqa: E402
from utils.timezone import now  # noqa: E402

PAGE = Path(__file__).with_name("app.html")
RAW_URL = "https://storage.googleapis.com/dm-meshgraphnets/deforming_plate"
RAW_DIR = "datasets/deforming_plate/raw_dataset"
RAW_FILES = ("meta.json", "test.tfrecord")
RAW_TRAJECTORIES = 100   # in test.tfrecord
# The quick start's dataset: EXPORT_DEFAULTS of export.py, [data].data_dir of
# examples/quick_start/train.toml (tests/test_app.py keeps them in step).
QUICKSTART_DATASET = {"n_train": 8, "n_val": 2, "n_test": 2, "time_stride": 10}
QUICKSTART_DATASET_NAME = "deforming_plate_s10_small"
TIME_STRIDES = (1, 2, 5, 10, 20)
CONFIG = "examples/quick_start/train.toml"
# The training configs the page offers, in this order; ``configs/`` holds a user's own.
CONFIG_PATTERNS = (CONFIG, "configs/*.toml", "configs/train/*.toml", "protocols/*.toml")
ARCHITECTURES = ("mgn_t", "meshgraphnets")   # train.config.ARCHITECTURES
EPOCHS = (1, 500)
DEFAULT_EPOCHS = 20   # with examples/quick_start/train.toml, as in examples/quick_start/crashdefnet-quickstart.ipynb
EXAMPLES_DIR = "docs/assets/plates"   # example plates of step 1, also in docs/concepts/graphs.md
# What /files/ serves: GIFs and PNGs below these
FILE_DIRS = ("evaluations", "checkpoints", EXAMPLES_DIR)
FILE_SUFFIXES = {".gif": "image/gif", ".png": "image/png"}

# Training progress: tqdm's bar of the epoch ("Epoch 3:  45%|...| 45/100 [...]").
EPOCH_BAR = re.compile(r"Epoch (\d+):\s*\d+%\|.*?\|\s*(\d+)/(\d+)")
# Export progress: "  [3/12] train test.tfrecord#0: ..."
EXPORT_LINE = re.compile(r"\[(\d+)/(\d+)\]")


class AppError(Exception):
    """A request the app does not carry out; the message is shown on the page."""


class Cancelled(Exception):
    pass


def _json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f, object_hook=current_metric_names)
    except (OSError, ValueError):
        return None


def _toml(path: Path) -> Dict:
    """The settings of a config; empty if unreadable (the page also opens before the
    installation, to say what is missing)."""
    try:
        import toml
        return toml.loads(path.read_text(encoding="utf-8"))
    except (ImportError, OSError, ValueError):
        return {}


def _integer(params: Dict, key: str, low: int, high: int) -> int:
    try:
        value = int(params[key])
    except (KeyError, TypeError, ValueError):
        raise AppError(f"{key}: a whole number is needed") from None
    if not low <= value <= high:
        raise AppError(f"{key}: between {low} and {high}")
    return value


def dataset_name(n_train: int, n_val: int, n_test: int, time_stride: int) -> str:
    """Folder name in data/ of an export with these settings."""
    if dict(n_train=n_train, n_val=n_val, n_test=n_test,
            time_stride=time_stride) == QUICKSTART_DATASET:
        return QUICKSTART_DATASET_NAME
    return f"deforming_plate_s{time_stride}_n{n_train}-{n_val}-{n_test}"


class Job:
    """A step running in the background: a child process or the download."""

    def __init__(self, kind: str, title: str, info: Optional[Dict] = None):
        self.kind, self.title, self.info = kind, title, info or {}
        self.status = "running"   # running | finished | failed | cancelled
        self.lines: deque = deque(maxlen=300)
        self.partial = ""         # the line a progress bar rewrites with \r
        self.progress: Optional[float] = None
        self.detail = ""
        self.bar: Optional[tuple] = None   # training: (epoch, step, steps) of tqdm's bar
        self.result: Dict = {}
        self.error = ""
        self.started, self.ended = time.time(), None
        self.process: Optional[subprocess.Popen] = None
        self.cancel_requested = False

    def to_dict(self) -> Dict:
        log = list(self.lines) + ([self.partial] if self.partial else [])
        return {"kind": self.kind, "title": self.title, "status": self.status, "info": self.info,
                "started": self.started,
                "elapsed": round((self.ended or time.time()) - self.started, 1),
                "progress": self.progress, "detail": self.detail, "result": self.result,
                "error": self.error, "log": log[-200:]}


def _last_segment(text: str) -> str:
    """What a terminal shows of a line that a progress bar rewrote with \\r."""
    parts = [part for part in text.split("\r") if part.strip()]
    return parts[-1] if parts else ""


class App:
    """State of the page and the steps; ``root`` is the repository."""

    def __init__(self, root: Path = _REPO_ROOT):
        self.root = Path(root)
        self.job: Optional[Job] = None
        self.lock = threading.Lock()
        self.device: Optional[str] = None   # None while detecting

    # ── What is there ──────────────────────────────────────────────

    def raw_data(self) -> Dict:
        folder = self.root / RAW_DIR
        files = {name: (folder / name).stat().st_size if (folder / name).is_file() else None
                 for name in RAW_FILES}
        meta = _json(folder / "meta.json") or {}
        length = meta.get("features", {}).get("world_pos", {}).get("shape", [None])[0]
        examples = sorted(path.name for path in (self.root / EXAMPLES_DIR).glob("*.png"))
        return {"folder": RAW_DIR, "files": files, "trajectory_length": length,
                "trajectories": RAW_TRAJECTORIES,
                "complete": all(size is not None for size in files.values()),
                "examples": [f"{EXAMPLES_DIR}/{name}" for name in examples]}

    def datasets(self) -> List[Dict]:
        busy = self.job.info.get("dataset") if self.job and self.job.status == "running" else None
        out = []
        for meta_path in sorted((self.root / "data").glob("*/metadata.json")):
            folder = meta_path.parent
            meta = _json(meta_path)
            if meta is None or not (folder / "dataset.db").is_file() or folder.name == busy:
                continue
            splits = meta.get("splits", {})
            out.append({"name": folder.name, "data_dir": f"data/{folder.name}",
                        "simulations": {split: splits.get(split, {}).get("count", 0)
                                   for split in ("train", "val", "test")},
                        "frames": meta.get("num_steps"), "time_stride": meta.get("time_stride"),
                        "train_samples": meta.get("sample_counts", {}).get("train"),
                        "stress_unit": meta.get("stress_unit", "Pa"),
                        "size_mb": round((folder / "dataset.db").stat().st_size / 1e6, 1)})
        return out

    def configs(self) -> List[Dict]:
        """The training configs of the repository with what the page preselects."""
        out = []
        for pattern in CONFIG_PATTERNS:
            for path in sorted(self.root.glob(pattern)):
                raw = _toml(path)
                name = path.relative_to(self.root).as_posix()
                out.append({"path": name,
                            "architecture": raw.get("model", {}).get("architecture"),
                            "epochs": (DEFAULT_EPOCHS if name == CONFIG else
                                       raw.get("training", {}).get("epochs", DEFAULT_EPOCHS))})
        return out

    def _has_dataset(self, data_dir: str) -> bool:
        return bool(data_dir) and (self.root / data_dir / "dataset.db").is_file()

    def models(self) -> List[Dict]:
        busy = self.job.info.get("model") if self.job and self.job.status == "running" else None
        out = []
        folders = (self.root / "checkpoints").iterdir() if (self.root / "checkpoints").is_dir() else ()
        rollouts = self.rollouts()
        for folder in folders:
            configs = sorted(folder.glob("train_*_config.json"))
            if not configs:
                continue
            config = _json(configs[-1]) or {}
            history = _json(folder / "history.json") or []
            test = _json(folder / "test_metrics.json") or {}
            data_dir = config.get("data_dir", "")
            if not self._has_dataset(data_dir):
                # Trained on another machine, the path is one of that machine: a prediction
                # that found its dataset equal to the training's tells where it is here
                data_dir = next((r["data_dir"] for r in rollouts
                                 if r["model"] == folder.name and r["matches_training_dataset"]
                                 and self._has_dataset(r["data_dir"])), data_dir)
            out.append({
                "name": folder.name, "architecture": config.get("architecture"),
                "data_dir": data_dir, "dataset": Path(data_dir).name,
                "dataset_present": self._has_dataset(data_dir),
                "stress_unit": (_json(self.root / data_dir / "metadata.json") or {}).get(
                    "stress_unit", "Pa") if data_dir else "Pa",
                "epochs": len(history),
                "selected_epoch": next((r["epoch"] for r in history if r.get("selected")), None),
                "has_model": (folder / "best_model.pt").is_file(),
                "finished": bool(test), "training": folder.name == busy,
                "test": test.get("rollout", {}).get("rollout_metrics"),
                "started": configs[-1].stat().st_mtime})
        return sorted(out, key=lambda model: -model["started"])

    def _animations(self) -> Dict[str, Dict[str, str]]:
        """Rollout -> simulation -> GIF below root, the newest of a simulation."""
        out: Dict[str, Dict[str, str]] = {}
        for gif in sorted((self.root / "evaluations").glob("*_animation_*/rollout_*.gif")):
            rollout = gif.parent.name.split("_animation_", 1)[1]
            out.setdefault(rollout, {})[gif.stem.removeprefix("rollout_")] = (
                gif.relative_to(self.root).as_posix())
        return out

    def rollouts(self) -> List[Dict]:
        out = []
        animations = self._animations()
        for info_path in (self.root / "rollouts").glob("*/rollout_info.json"):
            info = _json(info_path)
            if not info or info.get("status") != "finished":
                continue
            model, data, results = (info.get(key, {}) for key in ("model", "data", "results"))
            out.append({"name": info_path.parent.name, "model": model.get("training_run"),
                        "architecture": model.get("architecture"), "epoch": model.get("epoch"),
                        "data_dir": data.get("data_dir", ""),
                        "dataset": Path(data.get("data_dir", "")).name,
                        "matches_training_dataset": data.get("matches_training_dataset", False),
                        "split": data.get("split"), "started": info.get("started", ""),
                        "trajectories": results.get("sequences", []),
                        "gifs": animations.get(info_path.parent.name, {}),
                        "rollout_metrics": results.get("rollout_metrics")})
        return sorted(out, key=lambda rollout: rollout["started"], reverse=True)

    def rollout_detail(self, name: str) -> Dict:
        """Errors and animations of a rollout per simulation."""
        rollout = next((r for r in self.rollouts() if r["name"] == name), None)
        if rollout is None:
            raise AppError(f"no finished prediction {name!r}")
        folder = self.root / "rollouts" / name
        info = _json(folder / "rollout_info.json")
        summary = _json(folder / info["results"].get("summary_file", "")) or {}
        simulations = {sid: {"rollout_metrics": seq.get("rollout_metrics"),
                        "gif": rollout["gifs"].get(sid)}
                  for sid, seq in summary.get("sequences", {}).items()}
        metadata = info["data"].get("metadata", {})
        return {**rollout, "folder": f"rollouts/{name}",
                "stress_unit": metadata.get("stress_unit", "Pa"),
                "quick_start": rollout["dataset"] == QUICKSTART_DATASET_NAME,
                "simulations": simulations}

    def history(self, name: str) -> List[Dict]:
        """Losses per epoch of a training run, for the chart."""
        if not re.fullmatch(r"[\w.-]+", name):
            raise AppError(f"no training run {name!r}")
        keys = ("epoch", "train_loss", "val_loss", "val_rollout_pos_rmse", "selected")
        return [{key: record.get(key) for key in keys}
                for record in _json(self.root / "checkpoints" / name / "history.json") or []]

    def state(self) -> Dict:
        with self.lock:
            job = self.job
            if job and job.kind == "train" and job.status == "running":
                self._training_progress(job)
            return {"device": self.device, "job": job.to_dict() if job else None}

    def lists(self) -> Dict:
        return {"raw": self.raw_data(), "datasets": self.datasets(), "models": self.models(),
                "rollouts": self.rollouts(), "configs": self.configs(),
                "defaults": {**QUICKSTART_DATASET, "time_strides": TIME_STRIDES,
                             "config": CONFIG, "epochs": DEFAULT_EPOCHS, "epoch_range": EPOCHS}}

    # ── Steps ──────────────────────────────────────────────────────

    def export_command(self, params: Dict):
        raw = self.raw_data()
        if not raw["complete"]:
            raise AppError("the raw data is missing: download it in step 1")
        n_train = _integer(params, "n_train", 1, RAW_TRAJECTORIES - 2)
        n_val = _integer(params, "n_val", 1, RAW_TRAJECTORIES - 2)
        n_test = _integer(params, "n_test", 1, RAW_TRAJECTORIES - 2)
        if n_train + n_val + n_test > RAW_TRAJECTORIES:
            raise AppError(f"test.tfrecord holds {RAW_TRAJECTORIES} plates: at most "
                           f"{RAW_TRAJECTORIES} in all three groups together")
        stride = int(params.get("time_stride", 0))
        if stride not in TIME_STRIDES:
            raise AppError(f"time_stride: one of {', '.join(map(str, TIME_STRIDES))}")
        name = dataset_name(n_train, n_val, n_test, stride)
        if (self.root / "data" / name).exists():
            raise AppError(f"data/{name} exists already: choose it in the list")
        num_steps = (raw["trajectory_length"] - 1) // stride + 1   # frames 0, stride, ...
        command = ["-m", "dataset_preprocessor.export", "--out-dir", f"data/{name}",
                   "--n-train", str(n_train), "--n-val", str(n_val), "--n-test", str(n_test),
                   "--time-stride", str(stride), "--num-steps", str(num_steps),
                   "--no-overwrite"]
        return command, {"dataset": name}

    def train_command(self, params: Dict):
        dataset = next((d for d in self.datasets() if d["name"] == params.get("dataset")), None)
        if dataset is None:
            raise AppError("choose a dataset in step 2")
        config = params.get("config", CONFIG)
        if config not in [c["path"] for c in self.configs()]:
            raise AppError(f"no training config {config!r}")
        architecture = params.get("architecture")
        if architecture not in ARCHITECTURES:
            raise AppError(f"architecture: one of {', '.join(ARCHITECTURES)}")
        epochs = _integer(params, "epochs", *EPOCHS)
        name = f"{architecture}_{now():%Y%m%d_%H%M%S}"
        command = ["train/train.py", "--config", config, "--architecture", architecture,
                   "--data-dir", dataset["data_dir"], "--checkpoint-dir", f"checkpoints/{name}",
                   "--epochs", str(epochs)]
        return command, {"model": name, "epochs": epochs}

    def rollout_command(self, params: Dict):
        model = next((m for m in self.models() if m["name"] == params.get("model")), None)
        if model is None or not model["has_model"]:
            raise AppError("choose a trained model")
        if not model["dataset_present"]:
            raise AppError(f"the dataset of the model ({model['data_dir']}) is missing")
        command = ["predict/rollout.py", "--checkpoint", f"checkpoints/{model['name']}/best_model.pt",
                   "--data-dir", model["data_dir"], "--split", "test"]
        return command, {"model": model["name"]}

    def animate_command(self, params: Dict):
        rollout = next((r for r in self.rollouts() if r["name"] == params.get("rollout")), None)
        if rollout is None:
            raise AppError("choose a prediction")
        trajectory = params.get("trajectory")
        if trajectory not in rollout["trajectories"]:
            raise AppError(f"no simulation {trajectory!r} in this prediction")
        stamp, link = f"{now():%Y%m%d_%H%M%S}", f"animation_{rollout['name']}"   # see _animations
        folder, n = Path("evaluations") / f"{stamp}_{link}", 1
        while (self.root / folder).exists():
            n += 1
            folder = folder.with_name(f"{stamp}_{n}_{link}")
        gif = (folder / f"rollout_{trajectory}.gif").as_posix()
        command = ["utils/animate_rollout.py", f"rollouts/{rollout['name']}",
                   "--trajectory", trajectory, "--output", gif]
        return command, {"rollout": rollout["name"], "trajectory": trajectory, "gif": gif}

    def start(self, kind: str, params: Dict) -> None:
        titles = {"export": "Create the dataset", "train": "Training",
                  "rollout": "Prediction", "animate": "Animation"}
        if kind == "download":
            self._start(Job("download", "Download the raw data"), self._download)
            return
        if kind not in titles:
            raise AppError(f"unknown step {kind!r}")
        command, info = getattr(self, f"{kind}_command")(params)
        self._start(Job(kind, titles[kind], info), lambda job: self._run_process(job, command))

    def cancel(self) -> None:
        with self.lock:
            job = self.job
            if job is None or job.status != "running":
                return
            job.cancel_requested = True
            if job.process is not None:
                job.process.terminate()

    def _start(self, job: Job, work: Callable[[Job], None]) -> None:
        with self.lock:
            if self.job and self.job.status == "running":
                raise AppError(f"{self.job.title} is still running")
            self.job = job
        threading.Thread(target=self._finish, args=(job, work), daemon=True).start()

    def _finish(self, job: Job, work: Callable[[Job], None]) -> None:
        before = set(os.listdir(self.root / "rollouts")) if (self.root / "rollouts").is_dir() else set()
        try:
            work(job)
            status = "finished"
        except Cancelled:
            status = "cancelled"
        except Exception as error:   # shown on the page; the server goes on
            job.error = str(error) or type(error).__name__
            status = "failed"
        if job.cancel_requested:
            status = "cancelled"
        if status == "finished":
            job.result = self._result(job, before)
            if job.kind == "rollout":
                self._first_animation(job)
        with self.lock:
            job.status, job.ended, job.partial = status, time.time(), ""
            if job.kind in ("train", "rollout"):   # "epoch k of n" no longer applies
                job.detail = ""

    def _result(self, job: Job, rollouts_before: set) -> Dict:
        if job.kind == "rollout":
            new = sorted(set(os.listdir(self.root / "rollouts")) - rollouts_before)
            return {"rollout": new[-1]} if new else {}
        return dict(job.info)

    def _first_animation(self, job: Job) -> None:
        """So that step 5 opens with a picture. The rollout stands without it (pyvista is
        optional, rendering can fail, it can be cancelled): the reason goes to the log, and
        step 5 offers the animation again."""
        try:
            name = job.result.get("rollout")
            trajectory = self.rollout_detail(name)["trajectories"][0]
            command, _ = self.animate_command({"rollout": name, "trajectory": trajectory})
            job.detail = f"animation of {trajectory}"
            self._run_process(job, command)
        except Exception as error:   # as in _finish: the server goes on
            with self.lock:
                job.lines.append(f"No animation: {error}")

    def _run_process(self, job: Job, command: List[str]) -> None:
        # UTF-8 also on Windows, whose pipes default to the ANSI code page.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen([sys.executable, *command], cwd=self.root, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with self.lock:
            job.process = process
            job.lines.append("$ python " + " ".join(command))
        if job.cancel_requested:
            process.terminate()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending = ""
        while chunk := process.stdout.read1(65536):
            pending += decoder.decode(chunk)
            *lines, pending = pending.split("\n")
            # Keep only the last rewrite of a progress bar line (from its last \r on).
            pending = pending[pending.rstrip("\r").rfind("\r") + 1:]
            with self.lock:
                for line in lines:
                    job.lines.append(_last_segment(line))
                    self._line_progress(job, job.lines[-1])
                job.partial = _last_segment(pending)
                self._line_progress(job, job.partial)
        if process.wait() != 0 and not job.cancel_requested:
            tail = [line for line in job.lines if line.strip()][-1:]
            raise RuntimeError(f"stopped with an error: {tail[0] if tail else process.returncode}")

    def _line_progress(self, job: Job, line: str) -> None:
        if job.kind == "export" and (match := EXPORT_LINE.search(line)):
            done, total = map(int, match.groups())
            job.progress, job.detail = done / total, f"simulation {done} of {total}"
        elif job.kind == "train" and (match := EPOCH_BAR.search(line)):
            job.bar = tuple(map(int, match.groups()))

    def _training_progress(self, job: Job) -> None:
        """Progress from history.json (one record per finished epoch) and the epoch's bar."""
        epochs = job.info["epochs"]
        history = _json(self.root / "checkpoints" / job.info["model"] / "history.json") or []
        done = len(history)
        if history and history[-1].get("end"):   # last epoch done, the test runs
            job.progress, job.detail = 1.0, "test of the selected model on the unseen simulations"
            return
        epoch, step, steps = job.bar or (0, 0, 1)
        within = step / steps if epoch == done + 1 else 0.0
        job.progress = min(1.0, (done + within) / epochs)
        job.detail = f"epoch {min(done + 1, epochs)} of {epochs}"
        times = [r["epoch_time_seconds"] for r in history if r.get("epoch_time_seconds")]
        if times:   # early stopping may end the run sooner
            left = (epochs - done) * sum(times) / len(times) / 60
            job.detail += f", at most {max(1, round(left))} min left"

    def _download(self, job: Job) -> None:
        folder = self.root / RAW_DIR
        folder.mkdir(parents=True, exist_ok=True)
        for name in RAW_FILES:
            target = folder / name
            if target.is_file():
                continue
            job.lines.append(f"{RAW_URL}/{name} -> {RAW_DIR}/{name}")
            part = target.with_name(name + ".part")
            with urllib.request.urlopen(f"{RAW_URL}/{name}", timeout=60) as response, \
                    open(part, "wb") as f:
                total, done = int(response.headers.get("Content-Length") or 0), 0
                while chunk := response.read(1 << 20):
                    if job.cancel_requested:
                        raise Cancelled
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        job.progress = done / total
                        job.detail = f"{name}: {done / 1e6:.0f} of {total / 1e6:.0f} MB"
            os.replace(part, target)

    def detect_device(self) -> None:
        """The device training will use, from a child process: the app does not load PyTorch."""
        probe = ("import torch; print(torch.cuda.get_device_name() "
                 "if torch.cuda.is_available() else '')")
        try:
            result = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                                    text=True, timeout=300)
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is None or result.returncode != 0:
            self.device = "missing: PyTorch is not installed in this Python environment"
        else:
            name = result.stdout.strip()
            self.device = f"GPU ({name})" if name else "CPU (no GPU found)"

    def shutdown(self) -> None:
        self.cancel()


class Handler(BaseHTTPRequestHandler):
    app: App   # set by serve()

    def log_message(self, format, *args):   # no line per request
        pass

    def _local(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ("127.0.0.1", "localhost")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, value, status: int = 200) -> None:
        self._send(status, json.dumps(value).encode(), "application/json")

    def do_GET(self):
        if not self._local():
            return self._send(403, b"", "text/plain")
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/state":
                return self._send_json(self.app.state())
            if url.path == "/api/lists":
                return self._send_json(self.app.lists())
            if url.path == "/api/history":
                return self._send_json(self.app.history(query.get("model", "")))
            if url.path == "/api/rollout":
                return self._send_json(self.app.rollout_detail(query.get("name", "")))
            if url.path.startswith("/files/"):
                return self._send_file(unquote(url.path[len("/files/"):]))
        except AppError as error:
            return self._send_json({"error": str(error)}, 400)
        self._send(404, b"", "text/plain")

    def _send_file(self, relative: str) -> None:
        root = self.app.root.resolve()
        path = (root / relative).resolve()
        allowed = any(path.is_relative_to(root / folder) for folder in FILE_DIRS)
        if not allowed or path.suffix not in FILE_SUFFIXES or not path.is_file():
            return self._send(404, b"", "text/plain")
        self._send(200, path.read_bytes(), FILE_SUFFIXES[path.suffix])

    def do_POST(self):
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        if not self._local() or content_type != "application/json":
            return self._send(403, b"", "text/plain")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            params = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/cancel":
                self.app.cancel()
            elif self.path.startswith("/api/start/"):
                self.app.start(self.path[len("/api/start/"):], params)
            else:
                return self._send(404, b"", "text/plain")
        except (AppError, ValueError) as error:
            return self._send_json({"error": str(error)}, 400)
        self._send_json({"ok": True})


def serve(app: App, port: int) -> ThreadingHTTPServer:
    """The server on 127.0.0.1:``port`` (0: a free port), not yet serving."""
    handler = type("BoundHandler", (Handler,), {"app": app})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def open_browser(url: str) -> None:
    """The page in the default browser. Under WSL that is the browser of Windows, which
    ``webbrowser`` does not reach: it takes gio, which fails there without a Linux browser."""
    explorer = shutil.which("explorer.exe") if os.environ.get("WSL_DISTRO_NAME") else None
    if explorer:
        subprocess.run([explorer, url], check=False)  # its exit code is 1 even when it worked
    else:
        webbrowser.open(url)


def build_parser() -> argparse.ArgumentParser:
    """Command line options; the reference in docs/reference/cli.md is generated from them."""
    parser = argparse.ArgumentParser(
        description="The CrashDefNet app (early stage): the workflow as a local web page, "
                    "made mainly for the quick start: raw data, dataset, training, prediction "
                    "and result, without a terminal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--port", type=int, default=8765,
                        help="port of the page on 127.0.0.1")
    parser.add_argument("--browser", action=argparse.BooleanOptionalAction, default=True,
                        help="open the page in the default browser")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    app = App()
    try:
        server = serve(app, args.port)
    except OSError as error:
        print(f"Port {args.port} is not free ({error}); start with --port <another port>",
              file=sys.stderr)
        return 1
    url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=app.detect_device, daemon=True).start()
    print(f"CrashDefNet app: {url}\nOpen this address in the browser; "
          "stop with Ctrl+C.", flush=True)
    if args.browser:
        threading.Timer(0.5, open_browser, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
