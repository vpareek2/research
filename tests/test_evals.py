import math

import jax
import jax.numpy as jnp
from flax import nnx

from config import DistributedConfig, ModelConfig, TrainConfig
from distributed import create_distributed_context
from evals import bpb_from_losses, evaluate_loss
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


def test_bpb_from_losses_uses_target_token_bytes():
    losses = jnp.asarray([[math.log(2.0), math.log(4.0)]], dtype=jnp.float32)
    target_ids = jnp.asarray([[1, 2]], dtype=jnp.int32)
    token_bytes = jnp.asarray([0, 1, 3], dtype=jnp.uint16)

    bpb, byte_count = bpb_from_losses(losses, target_ids, token_bytes)

    assert math.isclose(float(bpb), math.log(8.0) / (math.log(2.0) * 4))
    assert int(byte_count) == 4


def test_evaluate_loss_returns_finite_scalar_metrics():
    cfg = tiny_model_config()
    tc = train_config(eval_steps=1)
    distributed = create_distributed_context(DistributedConfig(enabled=False), tc)
    model = Model(cfg, rngs=nnx.Rngs(0))
    token_bytes = jnp.ones((cfg.vocab_size,), dtype=jnp.uint16)
    batch = {
        "input_ids": jax.random.randint(jax.random.key(1), (tc.batch_size, tc.seq_len), 0, cfg.vocab_size),
    }

    result = evaluate_loss(
        model,
        iter([batch]),
        tc.eval_steps,
        distributed,
        tokens_per_example=tc.seq_len,
        token_bytes=token_bytes,
    )

    assert math.isfinite(result.loss)
    assert math.isfinite(result.ppl)
    assert math.isfinite(result.bpb)
    assert result.bytes == tc.batch_size * (tc.seq_len - 1)
    assert result.eval_steps == 1
    assert result.examples == tc.batch_size
    assert result.tokens == tc.batch_size * tc.seq_len
    assert result.tokens_per_sec > 0.0
    assert set(result.to_dict()) == {
        "loss",
        "ppl",
        "bpb",
        "bytes",
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
    token_bytes = jnp.ones((cfg.vocab_size,), dtype=jnp.uint16)
    batch = {
        "input_ids": jax.random.randint(jax.random.key(1), (tc.batch_size, tc.seq_len), 0, cfg.vocab_size),
    }

    try:
        evaluate_loss(model, iter([batch]), 0, distributed, tokens_per_example=tc.seq_len, token_bytes=token_bytes)
    except ValueError as exc:
        assert "eval_steps" in str(exc)
    else:
        raise AssertionError("expected invalid eval_steps to raise")
