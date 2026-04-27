import math

import jax
from flax import nnx

from config import DistributedConfig, ModelConfig, TrainConfig
from distributed import create_distributed_context
from evals import evaluate_loss
from model import Model


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
        eval_steps=1,
        checkpoint_every=1,
        keep_last=2,
    )
    values.update(overrides)
    return TrainConfig(**values)


def test_evaluate_loss_returns_finite_scalar_metrics():
    cfg = tiny_model_config()
    tc = train_config(eval_steps=1)
    distributed = create_distributed_context(DistributedConfig(enabled=False), tc)
    model = Model(cfg, rngs=nnx.Rngs(0))
    batch = {
        "input_ids": jax.random.randint(jax.random.key(1), (tc.batch_size, tc.seq_len), 0, cfg.vocab_size),
    }

    result = evaluate_loss(
        model,
        iter([batch]),
        tc.eval_steps,
        distributed,
        tokens_per_example=tc.seq_len,
    )

    assert math.isfinite(result.loss)
    assert math.isfinite(result.ppl)
    assert result.eval_steps == 1
    assert result.examples == tc.batch_size
    assert result.tokens == tc.batch_size * tc.seq_len
    assert result.tokens_per_sec > 0.0
    assert set(result.to_dict()) == {
        "loss",
        "ppl",
        "eval_steps",
        "examples",
        "tokens",
        "elapsed_sec",
        "tokens_per_sec",
    }


def test_evaluate_loss_rejects_invalid_eval_steps():
    cfg = tiny_model_config()
    tc = train_config(eval_steps=1)
    distributed = create_distributed_context(DistributedConfig(enabled=False), tc)
    model = Model(cfg, rngs=nnx.Rngs(0))
    batch = {
        "input_ids": jax.random.randint(jax.random.key(1), (tc.batch_size, tc.seq_len), 0, cfg.vocab_size),
    }

    try:
        evaluate_loss(model, iter([batch]), 0, distributed, tokens_per_example=tc.seq_len)
    except ValueError as exc:
        assert "eval_steps" in str(exc)
    else:
        raise AssertionError("expected invalid eval_steps to raise")
