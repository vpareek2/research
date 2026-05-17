import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from research.config import DataConfig, DistributedConfig, ExperimentConfig, ModelConfig, OptimizerConfig, PrecisionConfig, ProfilingConfig, RunConfig, SamplingConfig, TrainConfig, WandbConfig
from research.distributed import create_distributed_context
from research.evals import loss
from research.lr_schedule import build_lr_schedule
from research.model import Model
from research.optimizers import build_optimizer
from research.pretrain import (
    add_timing_metrics,
    format_metrics_row,
    LiveThroughputTracker,
    maybe_write_completion_summary,
    metric_header,
    print_startup,
    save_final_checkpoint_if_needed,
    sync_metric_scalars,
    train_step,
    tree_l2_norm,
)
from research.profiling import StepTimer


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


def tiny_train_config(**overrides):
    values = dict(
        seed=0,
        batch_size=2,
        seq_len=8,
        steps=4,
        log_every=1,
        eval_every=1,
        eval_steps=1,
        checkpoint_every=2,
        keep_last=2,
    )
    values.update(overrides)
    return TrainConfig(**values)


def tiny_optimizer_config(**overrides):
    values = dict(name="muon", lr=1e-3, weight_decay=0.1)
    values.update(overrides)
    return OptimizerConfig(**values)


def test_loss_is_scalar_and_finite():
    cfg = tiny_model_config()
    model = Model(cfg, rngs=nnx.Rngs(0))
    input_ids = jax.random.randint(jax.random.key(1), (2, 8), 0, cfg.vocab_size)

    value = loss(model, input_ids)

    assert value.shape == ()
    assert bool(jnp.isfinite(value))


def test_loss_is_scalar_and_finite_with_bf16_compute():
    cfg = tiny_model_config()
    precision = PrecisionConfig(compute_dtype="bf16", param_dtype="fp32", loss_dtype="fp32")
    model = Model(cfg, precision=precision, rngs=nnx.Rngs(0))
    input_ids = jax.random.randint(jax.random.key(1), (2, 8), 0, cfg.vocab_size)

    value = loss(model, input_ids)

    assert value.shape == ()
    assert value.dtype == jnp.float32
    assert bool(jnp.isfinite(value))


def test_tree_l2_norm():
    value = tree_l2_norm({"a": jnp.array([3.0, 4.0]), "b": jnp.array([12.0])})

    assert value.shape == ()
    assert float(value) == 13.0


def test_train_step_returns_classic_train_metrics():
    cfg = tiny_model_config()
    train_cfg = tiny_train_config()
    optimizer_cfg = tiny_optimizer_config()
    model = Model(cfg, rngs=nnx.Rngs(0))
    optimizer = build_optimizer(model, cfg, optimizer_cfg, build_lr_schedule(train_cfg, peak_lr=optimizer_cfg.lr))
    input_ids = jax.random.randint(jax.random.key(1), (2, 8), 0, cfg.vocab_size)
    token_bytes = jnp.ones((cfg.vocab_size,), dtype=jnp.uint16)

    value, metrics = train_step(model, optimizer, input_ids, token_bytes)

    assert value.shape == ()
    assert bool(jnp.isfinite(value))
    assert metrics["train/bpb"].shape == ()
    assert bool(jnp.isfinite(metrics["train/bpb"]))
    assert metrics["train/bytes"].shape == ()
    assert int(metrics["train/bytes"]) == input_ids.size - input_ids.shape[0]
    assert metrics["train/grad_norm"].shape == ()
    assert metrics["train/param_norm"].shape == ()
    assert bool(jnp.isfinite(metrics["train/grad_norm"]))
    assert bool(jnp.isfinite(metrics["train/param_norm"]))


def test_sync_metric_scalars_returns_python_floats():
    metrics = sync_metric_scalars(
        {
            "train/loss": jnp.asarray(1.25, dtype=jnp.float32),
            "train/ppl": jnp.asarray(3.5, dtype=jnp.float32),
            "train/grad_norm": jnp.asarray(0.75, dtype=jnp.float32),
            "train/param_norm": jnp.asarray(12.0, dtype=jnp.float32),
            "train/bytes": jnp.asarray(14, dtype=jnp.float32),
            "train/bytes_seen": jnp.asarray(28, dtype=jnp.float32),
            "optim/lr": jnp.asarray(1e-3, dtype=jnp.float32),
        }
    )

    assert set(metrics) == {
        "train/loss",
        "train/ppl",
        "train/grad_norm",
        "train/param_norm",
        "train/bytes",
        "train/bytes_seen",
        "optim/lr",
    }
    assert metrics["train/loss"] == pytest.approx(1.25)
    assert metrics["optim/lr"] == pytest.approx(1e-3)
    assert all(isinstance(value, float) for value in metrics.values())


def test_muon_optimizer_accepts_lr_schedule():
    cfg = tiny_model_config()
    train_cfg = tiny_train_config()
    optimizer_cfg = tiny_optimizer_config()
    model = Model(cfg, rngs=nnx.Rngs(0))
    optimizer = build_optimizer(model, cfg, optimizer_cfg, build_lr_schedule(train_cfg, peak_lr=optimizer_cfg.lr))
    input_ids = jax.random.randint(jax.random.key(1), (2, 8), 0, cfg.vocab_size)
    token_bytes = jnp.ones((cfg.vocab_size,), dtype=jnp.uint16)

    value, metrics = train_step(model, optimizer, input_ids, token_bytes)

    assert value.shape == ()
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(metrics["train/grad_norm"]))


def test_print_startup_outputs_run_summary(capsys):
    config = RunConfig(
        experiment=ExperimentConfig(name="unit", out_dir="runs"),
        model=tiny_model_config(),
        train=tiny_train_config(steps=3),
        optimizer=tiny_optimizer_config(),
        data=DataConfig(path="input.txt", tokenizer="gpt2", val_fraction=0.25),
        sampling=SamplingConfig(enabled=True),
        distributed=DistributedConfig(enabled=False),
        profiling=ProfilingConfig(),
        precision=PrecisionConfig(compute_dtype="bf16", param_dtype="fp32", loss_dtype="fp32"),
        wandb=WandbConfig(enabled=True, project="unit"),
    )
    distributed = create_distributed_context(config.distributed, config.train)

    print_startup(config, resume=True, distributed=distributed)

    output = capsys.readouterr().out
    assert "Pretraining" in output
    assert "run:        unit (resume)" in output
    assert "model:" in output
    assert "distributed: devices=1 global_batch=2 per_device_batch=2 axis=data" in output
    assert "optimizer:  muon peak_lr=0.001 weight_decay=0.1" in output
    assert "schedule:   cosine" in output
    assert "precision:  compute=bf16 params=fp32 loss=fp32" in output
    assert "profiling:  enabled=False profiler=none" in output
    assert "wandb:      on" in output
    assert "Starting training" not in output
    assert "step |" not in output


def test_metric_header_and_row_formatting():
    metrics = {
        "step": 20,
        "train/loss": 9.9001,
        "train/ppl": 19936.25,
        "train/grad_norm": 8.78,
        "time/tokens_per_sec": 30632.0,
        "perf/mfu": 12.3,
        "val/loss": 9.0258,
    }

    assert metric_header() == "  step |       loss |        ppl |      grad |    tok/s |      mfu |        val"
    assert format_metrics_row(metrics) == "    20 |     9.9001 |   19936.25 |      8.78 |    30632 |    12.3% |     9.0258"


def test_metric_row_without_val_loss():
    metrics = {
        "step": 21,
        "train/loss": 8.9149,
        "train/ppl": 7430.12,
        "train/grad_norm": 9.16,
        "time/tokens_per_sec": 20006.0,
    }

    assert format_metrics_row(metrics) == "    21 |     8.9149 |    7430.12 |      9.16 |    20006 |          |           "


def test_add_timing_metrics_reports_loop_and_train_throughput():
    train_cfg = tiny_train_config(batch_size=4, steps=2)
    timer = StepTimer()
    timer.add("step", 0.2)
    timer.add("train_step", 0.1)
    timer.add("data", 0.03)
    metrics = {"step": 0}

    add_timing_metrics(metrics, timer, train_cfg)

    assert metrics["time/data_sec"] == 0.03
    assert metrics["time/train_step_sec"] == 0.1
    assert metrics["time/step_sec"] == 0.2
    assert metrics["time/tokens_per_sec"] == 160.0
    assert metrics["time/train_tokens_per_sec"] == 320.0


def test_add_timing_metrics_uses_train_sync_for_train_throughput():
    train_cfg = tiny_train_config(batch_size=4, steps=2)
    timer = StepTimer()
    timer.add("step", 0.2)
    timer.add("train_step", 0.02)
    timer.add("train_sync", 0.08)
    metrics = {"step": 0}

    add_timing_metrics(metrics, timer, train_cfg)

    assert metrics["time/train_step_sec"] == 0.02
    assert metrics["time/train_sync_sec"] == 0.08
    assert metrics["time/train_tokens_per_sec"] == 320.0


def test_live_throughput_tracker_preserves_first_row_raw_metrics():
    tracker = LiveThroughputTracker()
    metrics = {
        "step": 10,
        "train/tokens_seen": 1000,
        "time/elapsed_sec": 5.0,
        "time/tokens_per_sec": 100.0,
        "time/train_tokens_per_sec": 200.0,
    }

    tracker.update(metrics)

    assert metrics["time/raw_tokens_per_sec"] == 100.0
    assert metrics["time/raw_train_tokens_per_sec"] == 200.0
    assert metrics["time/tokens_per_sec"] == 100.0
    assert metrics["time/train_tokens_per_sec"] == 200.0


def test_live_throughput_tracker_replaces_second_plain_row_with_delta_rate():
    tracker = LiveThroughputTracker()
    tracker.update(
        {
            "step": 10,
            "train/tokens_seen": 1000,
            "time/elapsed_sec": 5.0,
            "time/tokens_per_sec": 100.0,
            "time/train_tokens_per_sec": 200.0,
        }
    )
    metrics = {
        "step": 20,
        "train/tokens_seen": 3000,
        "time/elapsed_sec": 9.0,
        "time/tokens_per_sec": 120.0,
        "time/train_tokens_per_sec": 240.0,
    }

    tracker.update(metrics)

    assert metrics["time/raw_tokens_per_sec"] == 120.0
    assert metrics["time/raw_train_tokens_per_sec"] == 240.0
    assert metrics["time/tokens_per_sec"] == 500.0
    assert metrics["time/train_tokens_per_sec"] == 500.0


def test_live_throughput_tracker_eval_rows_reuse_latest_steady_rate_without_updating_anchor():
    tracker = LiveThroughputTracker()
    tracker.update(
        {
            "step": 10,
            "train/tokens_seen": 1000,
            "time/elapsed_sec": 5.0,
            "time/tokens_per_sec": 100.0,
            "time/train_tokens_per_sec": 200.0,
        }
    )
    tracker.update(
        {
            "step": 20,
            "train/tokens_seen": 3000,
            "time/elapsed_sec": 9.0,
            "time/tokens_per_sec": 120.0,
            "time/train_tokens_per_sec": 240.0,
        }
    )
    eval_metrics = {
        "step": 30,
        "train/tokens_seen": 5000,
        "time/elapsed_sec": 100.0,
        "time/tokens_per_sec": 10.0,
        "time/train_tokens_per_sec": 20.0,
        "val/loss": 3.0,
    }
    next_train_metrics = {
        "step": 40,
        "train/tokens_seen": 5000,
        "time/elapsed_sec": 13.0,
        "time/tokens_per_sec": 130.0,
        "time/train_tokens_per_sec": 260.0,
    }

    tracker.update(eval_metrics)
    tracker.update(next_train_metrics)

    assert eval_metrics["time/tokens_per_sec"] == 500.0
    assert eval_metrics["time/train_tokens_per_sec"] == 500.0
    assert next_train_metrics["time/tokens_per_sec"] == 500.0
    assert next_train_metrics["time/train_tokens_per_sec"] == 500.0


def test_live_throughput_tracker_ignores_non_positive_deltas():
    tracker = LiveThroughputTracker()
    tracker.update(
        {
            "step": 10,
            "train/tokens_seen": 1000,
            "time/elapsed_sec": 5.0,
            "time/tokens_per_sec": 100.0,
            "time/train_tokens_per_sec": 200.0,
        }
    )
    metrics = {
        "step": 20,
        "train/tokens_seen": 900,
        "time/elapsed_sec": 9.0,
        "time/tokens_per_sec": 120.0,
        "time/train_tokens_per_sec": 240.0,
    }

    tracker.update(metrics)

    assert metrics["time/tokens_per_sec"] == 120.0
    assert metrics["time/train_tokens_per_sec"] == 240.0


def test_maybe_write_completion_summary_only_writes_for_completed_runs(monkeypatch, tmp_path):
    calls = []

    def fake_write(run_dir):
        calls.append(run_dir)
        return tmp_path / "run_summary.json", tmp_path / "scorecard.md"

    monkeypatch.setattr("research.pretrain.write_completion_summary", fake_write)

    assert maybe_write_completion_summary(tmp_path / "run", completed=False) is None
    assert calls == []

    result = maybe_write_completion_summary(tmp_path / "run", completed=True)

    assert calls == [tmp_path / "run"]
    assert result == (tmp_path / "run_summary.json", tmp_path / "scorecard.md")


class FakeCheckpointManager:
    def __init__(self, latest_step):
        self._latest_step = latest_step
        self.waits = 0

    def latest_step(self):
        return self._latest_step

    def wait_until_finished(self):
        self.waits += 1


def test_save_final_checkpoint_if_needed_saves_missing_final_step(monkeypatch):
    calls = []
    manager = FakeCheckpointManager(latest_step=2)
    train_cfg = tiny_train_config(batch_size=4, steps=5, checkpoint_every=0)

    def fake_save_checkpoint(manager_arg, **kwargs):
        calls.append((manager_arg, kwargs))

    monkeypatch.setattr("research.pretrain.save_checkpoint", fake_save_checkpoint)

    saved = save_final_checkpoint_if_needed(
        manager,
        train_cfg,
        model="model",
        optimizer="optimizer",
        train_iter="train_iter",
    )

    assert saved is True
    assert manager.waits == 1
    assert calls == [
        (
            manager,
            {
                "next_step": 5,
                "model": "model",
                "optimizer": "optimizer",
                "train_iter": "train_iter",
            },
        )
    ]


def test_save_final_checkpoint_if_needed_skips_existing_final_step(monkeypatch):
    calls = []
    manager = FakeCheckpointManager(latest_step=5)
    train_cfg = tiny_train_config(batch_size=4, steps=5, checkpoint_every=5)
    monkeypatch.setattr("research.pretrain.save_checkpoint", lambda *args, **kwargs: calls.append((args, kwargs)))

    saved = save_final_checkpoint_if_needed(
        manager,
        train_cfg,
        model="model",
        optimizer="optimizer",
        train_iter="train_iter",
    )

    assert saved is False
    assert manager.waits == 0
    assert calls == []
