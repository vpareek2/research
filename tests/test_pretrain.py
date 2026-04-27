import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from config import DataConfig, DistributedConfig, ExperimentConfig, ModelConfig, PrecisionConfig, ProfilingConfig, RunConfig, SamplingConfig, TrainConfig, WandbConfig
from distributed import create_distributed_context
from evals import loss
from lr_schedule import build_lr_schedule
from model import Model
from pretrain import add_timing_metrics, format_metrics_row, make_muon_dimension_numbers, metric_header, print_startup, sync_metric_scalars, train_step, tree_l2_norm
from profiling import StepTimer


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


def test_muon_selector_excludes_vocab_and_non_2d_params():
    cfg = tiny_model_config()
    selector = make_muon_dimension_numbers(cfg)
    params = {
        "internal": jnp.zeros((cfg.hidden_size, cfg.hidden_size)),
        "lm_head": jnp.zeros((cfg.hidden_size, cfg.vocab_size)),
        "embed": jnp.zeros((cfg.vocab_size, cfg.hidden_size)),
        "norm": jnp.zeros((cfg.hidden_size,)),
    }

    labels = selector(params)

    assert isinstance(labels["internal"], optax.contrib.MuonDimensionNumbers)
    assert labels["lm_head"] is None
    assert labels["embed"] is None
    assert labels["norm"] is None


def test_tree_l2_norm():
    value = tree_l2_norm({"a": jnp.array([3.0, 4.0]), "b": jnp.array([12.0])})

    assert value.shape == ()
    assert float(value) == 13.0


def test_train_step_returns_classic_train_metrics():
    cfg = tiny_model_config()
    model = Model(cfg, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)
    input_ids = jax.random.randint(jax.random.key(1), (2, 8), 0, cfg.vocab_size)
    token_bytes = jnp.ones((cfg.vocab_size,), dtype=jnp.uint16)

    value, metrics = train_step(model, optimizer, input_ids, token_bytes)

    assert value.shape == ()
    assert bool(jnp.isfinite(value))
    assert metrics["train/bpb"].shape == ()
    assert bool(jnp.isfinite(metrics["train/bpb"]))
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
            "optim/lr": jnp.asarray(1e-3, dtype=jnp.float32),
        }
    )

    assert set(metrics) == {
        "train/loss",
        "train/ppl",
        "train/grad_norm",
        "train/param_norm",
        "optim/lr",
    }
    assert metrics["train/loss"] == pytest.approx(1.25)
    assert metrics["optim/lr"] == pytest.approx(1e-3)
    assert all(isinstance(value, float) for value in metrics.values())


def test_muon_optimizer_accepts_lr_schedule():
    cfg = tiny_model_config()
    train_cfg = TrainConfig(
        seed=0,
        batch_size=2,
        seq_len=8,
        steps=4,
        lr=1e-3,
        decay=0.1,
        log_every=1,
        eval_every=1,
        eval_steps=1,
        checkpoint_every=2,
        keep_last=2,
    )
    model = Model(cfg, rngs=nnx.Rngs(0))
    tx = optax.contrib.muon(
        learning_rate=build_lr_schedule(train_cfg),
        weight_decay=train_cfg.decay,
        adam_weight_decay=train_cfg.decay,
        muon_weight_dimension_numbers=make_muon_dimension_numbers(cfg),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
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
        train=TrainConfig(
            seed=0,
            batch_size=2,
            seq_len=8,
            steps=3,
            lr=1e-3,
            decay=0.1,
            log_every=1,
            eval_every=1,
            eval_steps=1,
            checkpoint_every=2,
            keep_last=2,
        ),
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
        "val/loss": 9.0258,
    }

    assert metric_header() == "  step |       loss |        ppl |      grad |    tok/s |        val"
    assert format_metrics_row(metrics) == "    20 |     9.9001 |   19936.25 |      8.78 |    30632 |     9.0258"


def test_metric_row_without_val_loss():
    metrics = {
        "step": 21,
        "train/loss": 8.9149,
        "train/ppl": 7430.12,
        "train/grad_norm": 9.16,
        "time/tokens_per_sec": 20006.0,
    }

    assert format_metrics_row(metrics) == "    21 |     8.9149 |    7430.12 |      9.16 |    20006 |           "


def test_add_timing_metrics_reports_loop_and_train_throughput():
    train_cfg = TrainConfig(
        seed=0,
        batch_size=4,
        seq_len=8,
        steps=2,
        lr=1e-3,
        decay=0.1,
        log_every=1,
        eval_every=1,
        eval_steps=1,
        checkpoint_every=2,
        keep_last=2,
    )
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
