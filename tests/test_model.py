import chex
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from config import ModelConfig
from model import Model, _precompute_rope, rope


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


def test_rope_shapes():
    x = jnp.ones((2, 8, 4, 8), dtype=jnp.float32)
    cos, sin = _precompute_rope(seq_len=8, head_dim=8, theta=10000.0, dtype=x.dtype)
    y = rope(x, cos, sin)

    chex.assert_shape(cos, (8, 4))
    chex.assert_shape(sin, (8, 4))
    chex.assert_shape(y, x.shape)
    assert bool(jnp.all(jnp.isfinite(y)))


def test_model_forward_shape_and_finite():
    cfg = tiny_model_config()
    model = Model(cfg, rngs=nnx.Rngs(0))
    input_ids = jax.random.randint(jax.random.key(0), (2, 8), 0, cfg.vocab_size)

    logits = model(input_ids)

    chex.assert_shape(logits, (2, 8, cfg.vocab_size))
    assert bool(jnp.all(jnp.isfinite(logits)))


def test_model_rejects_overlong_sequence():
    cfg = tiny_model_config()
    model = Model(cfg, rngs=nnx.Rngs(0))
    input_ids = jnp.zeros((1, cfg.seq_len + 1), dtype=jnp.int32)

    with pytest.raises(ValueError, match="exceeds seq_len"):
        model(input_ids)
