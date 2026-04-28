"""
Run artifact management.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import json
import math
import netrc
import os
from urllib.parse import urlparse
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import flax
import grain
import jax
import numpy as np
import optax
import tiktoken

from config import RunConfig
from data import Batch


class HealthMonitor:
    train_slope_window = 100
    val_slope_window = 5
    spike_window = 128
    min_spike_history = 8
    spike_threshold_std = 6.0

    def __init__(self):
        self.started_at = time.perf_counter()
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.grad_norms: list[float] = []
        self.loss_spike_count = 0
        self.grad_norm_spike_count = 0
        self.any_spike_count = 0
        self.nan_count = 0
        self.steps = 0

    def enrich(self, metrics: dict) -> dict:
        metrics["optim/loss_scale"] = 1.0
        metrics["time/elapsed_sec"] = time.perf_counter() - self.started_at
        row_has_nan = any(_is_nan(value) for value in metrics.values())

        train_loss = _finite_float(metrics.get("train/loss"))
        val_loss = _finite_float(metrics.get("val/loss"))
        grad_norm = _finite_float(metrics.get("train/grad_norm"))
        param_norm = _finite_float(metrics.get("train/param_norm"))

        if train_loss is not None:
            self.steps += 1
            loss_spike = self._is_spike(train_loss, self.train_losses)
            metrics["health/train_loss_spike"] = float(loss_spike)
            if loss_spike:
                self.loss_spike_count += 1
            self.train_losses.append(train_loss)
            self.train_losses = self.train_losses[-max(self.train_slope_window, self.spike_window) :]
            slope = _slope(self.train_losses[-self.train_slope_window :])
            if slope is not None:
                metrics["health/train_loss_slope"] = slope

        if val_loss is not None:
            self.val_losses.append(val_loss)
            self.val_losses = self.val_losses[-self.val_slope_window :]
            slope = _slope(self.val_losses[-self.val_slope_window :])
            if slope is not None:
                metrics["health/val_loss_slope"] = slope

        if train_loss is not None and val_loss is not None:
            metrics["health/train_val_gap"] = val_loss - train_loss

        if grad_norm is not None:
            grad_spike = self._is_spike(grad_norm, self.grad_norms)
            metrics["health/grad_norm_spike"] = float(grad_spike)
            if grad_spike:
                self.grad_norm_spike_count += 1
            self.grad_norms.append(grad_norm)
            self.grad_norms = self.grad_norms[-self.spike_window :]

        if grad_norm is not None and param_norm is not None and param_norm != 0.0:
            metrics["health/grad_param_ratio"] = grad_norm / param_norm

        if train_loss is not None:
            if metrics.get("health/train_loss_spike", 0.0) or metrics.get("health/grad_norm_spike", 0.0):
                self.any_spike_count += 1
            metrics["health/loss_spike_count"] = self.loss_spike_count
            metrics["health/grad_norm_spike_count"] = self.grad_norm_spike_count
            metrics["health/spike_rate"] = self.any_spike_count / self.steps

        if row_has_nan:
            self.nan_count += 1
        metrics["health/nan_count"] = self.nan_count

        return metrics

    def _is_spike(self, value: float, history: list[float]) -> bool:
        window = history[-self.spike_window :]
        if len(window) < self.min_spike_history:
            return False
        mean = sum(window) / len(window)
        variance = sum((item - mean) ** 2 for item in window) / len(window)
        return value > mean + self.spike_threshold_std * math.sqrt(variance)


class RunLogger:
    def __init__(self, run_dir: Path, wandb_module: Any | None = None, wandb_run: Any | None = None):
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.batches_path = run_dir / "batches.jsonl"
        self._health = HealthMonitor()
        self._wandb = wandb_module
        self._wandb_run = wandb_run
        self._sample_table = None
        if self._wandb is not None:
            self._sample_table = self._wandb.Table(columns=["step", "path", "text"], log_mode="MUTABLE")

    def log(self, metrics: dict):
        log_start = time.perf_counter()
        metrics = dict(metrics)
        metrics["time/log_sec"] = 0.0
        self._health.enrich(metrics)
        metrics["time/log_sec"] = time.perf_counter() - log_start

        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, sort_keys=True) + "\n")

        if self._wandb_run is None:
            return

        step = metrics["step"]
        wandb_metrics = {
            key: value
            for key, value in metrics.items()
            if _should_log_to_wandb(key, value)
        }
        if wandb_metrics:
            self._wandb_run.log(wandb_metrics, step=step)

        sample_path = metrics.get("sample/path")
        if sample_path is not None:
            path = Path(sample_path)
            text = path.read_text(encoding="utf-8")
            self._sample_table.add_data(step, str(path), text)
            self._wandb_run.log({"samples": self._sample_table}, step=step)

    def log_batch(self, step: int, batch: Batch):
        record = {
            "step": step,
            "chunk_idx": np.asarray(batch["chunk_idx"]).astype(int).tolist(),
            "token_start": np.asarray(batch["token_start"]).astype(int).tolist(),
            "token_end": np.asarray(batch["token_end"]).astype(int).tolist(),
        }
        with self.batches_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def close(self):
        if self._wandb is not None:
            self._wandb.finish()


def setup_run(
    config_path: str | Path,
    config: RunConfig,
    *,
    resume: bool = False,
) -> RunLogger:
    config_path = Path(config_path)
    run_dir = Path(config.experiment.out_dir) / config.experiment.name

    if resume:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist for resume: {run_dir}")
        wandb_module, wandb_run = _init_wandb(run_dir, config, resume=True)
        return RunLogger(run_dir, wandb_module, wandb_run)

    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. Choose a new [experiment].name "
            "or rerun with --resume."
        )

    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "samples").mkdir()

    shutil.copyfile(config_path, run_dir / "config.toml")
    _write_metadata(run_dir, config_path)
    wandb_module, wandb_run = _init_wandb(run_dir, config, resume=False)
    return RunLogger(run_dir, wandb_module, wandb_run)


def _init_wandb(run_dir: Path, config: RunConfig, *, resume: bool):
    if not config.wandb.enabled:
        return None, None

    import wandb

    _login_wandb(wandb)
    wandb_id_path = run_dir / "wandb_id.txt"
    init_kwargs = {
        "project": config.wandb.project,
        "name": config.experiment.name,
        "config": asdict(config),
        "dir": str(run_dir),
        "tags": config.wandb.tags,
    }
    if config.wandb.entity:
        init_kwargs["entity"] = config.wandb.entity

    if resume:
        if not wandb_id_path.exists():
            raise FileNotFoundError(f"Missing W&B run id for resume: {wandb_id_path}")
        init_kwargs["id"] = wandb_id_path.read_text(encoding="utf-8").strip()
        init_kwargs["resume"] = "must"

    wandb_run = wandb.init(**init_kwargs)
    if not resume:
        run_id = getattr(wandb_run, "id", None)
        if not run_id:
            raise RuntimeError("W&B did not return a run id")
        wandb_id_path.write_text(f"{run_id}\n", encoding="utf-8")

    return wandb, wandb_run


def _login_wandb(wandb_module):
    if os.environ.get("WANDB_MODE") == "offline":
        return

    key = os.environ.get("WANDB_API_KEY")
    if key:
        wandb_module.login(key=key)
        return

    if _has_wandb_netrc_key():
        wandb_module.login(relogin=False)
        return

    key = getpass.getpass("W&B API key: ").strip()
    if not key:
        raise RuntimeError("W&B is enabled but no API key was provided")

    wandb_module.login(key=key)


def _has_wandb_netrc_key() -> bool:
    host = urlparse(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    if host is None:
        return False

    try:
        auth = netrc.netrc().authenticators(host)
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False

    return auth is not None and bool(auth[2])


def _is_scalar(value: Any) -> bool:
    if isinstance(value, (int, float, bool)):
        return True
    if isinstance(value, np.generic):
        return True
    return False


def _should_log_to_wandb(key: str, value: Any) -> bool:
    if key in {"step", "sample/path"} or not _is_scalar(value):
        return False
    if key.endswith("/ppl"):
        return False
    if key.startswith("val/domain/") and not key.endswith("/loss"):
        return False
    if key.startswith("val/") and key.rsplit("/", 1)[-1] in {
        "bytes",
        "elapsed_sec",
        "eval_steps",
        "examples",
        "tokens",
        "tokens_per_sec",
    }:
        return False
    return True


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float, np.generic)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _is_nan(value: Any) -> bool:
    if not isinstance(value, (int, float, np.generic)):
        return False
    return math.isnan(float(value))


def _slope(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    numerator = sum((idx - x_mean) * (value - y_mean) for idx, value in enumerate(values))
    denominator = sum((idx - x_mean) ** 2 for idx in range(n))
    return numerator / denominator


def _write_metadata(run_dir: Path, config_path: Path):
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "flax": flax.__version__,
        "optax": optax.__version__,
        "grain": grain.__version__,
        "tiktoken": tiktoken.__version__,
        "devices": [str(device) for device in jax.devices()],
    }

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
