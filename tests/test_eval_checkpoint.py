import json
import sys

import numpy as np
import optax
import pytest
from flax import nnx

from research.checkpoint import create_checkpoint_manager, save_checkpoint
from research.config import DataConfig, ModelConfig, TrainConfig
from research.data import REQUIRED_EVAL_DOMAINS, make_dataloaders
from research.evals import LossEvalResult
from research.model import Model
from research.utils import eval_checkpoint


def tiny_model_config():
    return ModelConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=1,
        seq_len=8,
        theta=10000.0,
        eps=1e-6,
        tied=False,
    )


def train_config(**overrides):
    values = dict(
        seed=0,
        batch_size=2,
        seq_len=8,
        steps=2,
        lr=0.001,
        decay=0.1,
        log_every=1,
        eval_every=1,
        eval_steps=2,
        checkpoint_every=1,
        keep_last=2,
    )
    values.update(overrides)
    return TrainConfig(**values)


def write_manifest(data_dir):
    (data_dir / "manifest.json").write_text(json.dumps({
        "dtype": "uint32",
        "num_tokens": 128,
        "tokenizer": {"name": "gpt2"},
        "files": {"tokens": {"path": "tokens.bin"}},
        "splits": {
            "train": {"start": 0, "end": 64, "tokens": 64},
            "val": {"start": 64, "end": 128, "tokens": 64},
        },
    }))


def write_eval_domain_pack(root):
    root.mkdir(parents=True)
    np.ones(128, dtype=np.uint16).tofile(root / "token_bytes.bin")
    domains = {}
    for name in REQUIRED_EVAL_DOMAINS:
        domain_dir = root / name
        domain_dir.mkdir()
        np.arange(32, dtype=np.uint32).tofile(domain_dir / "tokens.bin")
        (domain_dir / "manifest.json").write_text(json.dumps({
            "dtype": "uint32",
            "num_tokens": 32,
            "tokenizer": {"name": "gpt2"},
            "files": {"tokens": {"path": "tokens.bin"}},
            "splits": {
                "train": {"start": 0, "end": 0, "tokens": 0},
                "val": {"start": 0, "end": 32, "tokens": 32},
            },
        }))
        domains[name] = {"path": name, "num_tokens": 32}
    (root / "manifest.json").write_text(json.dumps({
        "kind": "eval_domains",
        "tokenizer": {"name": "gpt2"},
        "files": {"token_bytes": {"path": "token_bytes.bin"}},
        "domains": domains,
    }))


def write_run_config(run_dir, data_dir, eval_root=None):
    eval_root = eval_root or run_dir.parent / "eval_domains"
    (run_dir / "config.toml").write_text(
        f"""
[experiment]
name = "unit"
out_dir = "{run_dir.parent}"

[model]
vocab_size = 128
hidden_size = 32
intermediate_size = 64
n_layers = 1
n_heads = 4
n_kv_heads = 1
seq_len = 8
theta = 10000.0
eps = 0.000001
tied = false

[distributed]
enabled = false
device_count = "auto"
axis_name = "data"

[train]
seed = 0
batch_size = 2
seq_len = 8
steps = 2
lr = 0.001
decay = 0.1
log_every = 1
eval_every = 1
eval_steps = 2
checkpoint_every = 1
keep_last = 2

[data]
source = "tokens"
path = "{data_dir}"
tokenizer = "gpt2"

[eval]
domain_root = "{eval_root}"
domain_eval_steps = 1
""",
        encoding="utf-8",
    )


def write_run_metrics(run_dir):
    rows = [
        {
            "step": 0,
            "train/loss": 4.0,
            "val/loss": 4.5,
            "val/bpb": 2.2,
            "train/tokens_seen": 16,
            "health/nan_count": 0,
            "health/loss_spike_count": 0,
            "health/grad_norm_spike_count": 0,
        },
        {
            "step": 1,
            "train/loss": 3.0,
            "val/loss": 3.5,
            "val/bpb": 1.8,
            "train/tokens_seen": 32,
            "health/nan_count": 0,
            "health/loss_spike_count": 0,
            "health/grad_norm_spike_count": 0,
        },
    ]
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_run(tmp_path):
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    np.arange(128, dtype=np.uint32).tofile(data_dir / "tokens.bin")
    write_manifest(data_dir)
    eval_root = tmp_path / "eval_domains"
    write_eval_domain_pack(eval_root)

    run_dir = tmp_path / "runs" / "unit"
    run_dir.mkdir(parents=True)
    write_run_config(run_dir, data_dir, eval_root)
    write_run_metrics(run_dir)

    model_config = tiny_model_config()
    tc = train_config()
    dc = DataConfig(source="tokens", path=str(data_dir), tokenizer="gpt2")
    model = Model(model_config, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(tc.lr), wrt=nnx.Param)
    train_iter, _ = make_dataloaders(dc, tc)
    manager = create_checkpoint_manager(run_dir, keep_last=2)
    save_checkpoint(
        manager,
        next_step=1,
        model=model,
        optimizer=optimizer,
        train_iter=train_iter,
    )
    manager.wait_until_finished()
    return run_dir


def fake_domain_results(*args, **kwargs):
    return {
        name: LossEvalResult(
            loss=1.0,
            ppl=2.0,
            eval_steps=1,
            examples=2,
            tokens=16,
            elapsed_sec=0.1,
            bpb=0.5,
            bytes=32,
        )
        for name in REQUIRED_EVAL_DOMAINS
    }


def test_eval_checkpoint_cli_writes_latest_eval_artifacts(tmp_path, monkeypatch):
    run_dir = make_run(tmp_path)
    monkeypatch.setattr(sys, "argv", ["eval-checkpoint", str(run_dir), "--eval-steps", "1"])
    monkeypatch.setattr(eval_checkpoint, "evaluate_domain_losses", fake_domain_results)

    eval_checkpoint.main()

    metrics_path = run_dir / "evals" / "step_1" / "metrics.json"
    summary_path = run_dir / "evals" / "step_1" / "summary.md"
    run_summary_path = run_dir / "summary" / "run_summary.json"
    scorecard_path = run_dir / "summary" / "scorecard.md"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert summary_path.exists()
    assert run_summary_path.exists()
    assert scorecard_path.exists()
    assert "Native Validation" in summary_path.read_text(encoding="utf-8")
    assert "Domain Validation" in summary_path.read_text(encoding="utf-8")
    assert metrics["run_dir"] == str(run_dir)
    assert metrics["checkpoint_step"] == 1
    assert metrics["eval_steps"] == 1
    assert metrics["examples"] == 2
    assert metrics["tokens"] == 16
    assert metrics["loss"] > 0.0
    assert metrics["ppl"] > 0.0
    assert metrics["bpb"] > 0.0
    assert metrics["bytes"] > 0
    assert metrics["tokens_per_sec"] > 0.0
    assert set(metrics["domains"]) == set(REQUIRED_EVAL_DOMAINS)
    assert metrics["domains"]["web"]["loss"] > 0.0
    refreshed_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    assert refreshed_summary["checkpoint_evals"]["latest"]["checkpoint_step"] == 1


def test_eval_checkpoint_cli_supports_explicit_step(tmp_path, monkeypatch):
    run_dir = make_run(tmp_path)
    monkeypatch.setattr(sys, "argv", ["eval-checkpoint", str(run_dir), "--step", "1", "--eval-steps", "1"])
    monkeypatch.setattr(eval_checkpoint, "evaluate_domain_losses", fake_domain_results)

    eval_checkpoint.main()

    metrics = json.loads((run_dir / "evals" / "step_1" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["checkpoint_step"] == 1


def test_eval_checkpoint_missing_checkpoint_raises_clear_error(tmp_path):
    run_dir = tmp_path / "runs" / "unit"
    run_dir.mkdir(parents=True)
    eval_root = tmp_path / "eval_domains"
    write_eval_domain_pack(eval_root)
    write_run_config(run_dir, tmp_path / "prepared", eval_root)

    with pytest.raises(FileNotFoundError, match="No checkpoint"):
        eval_checkpoint.run_eval(run_dir, step=None, eval_steps=1)
