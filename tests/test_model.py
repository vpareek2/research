import chex
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from research.config import ModelConfig, PrecisionConfig
from research.kv_cache import init_kv_cache
from research.model import Model, _precompute_rope, rope


def tiny_model_config(**overrides):
    values = {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "n_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 1,
        "seq_len": 8,
        "theta": 10000.0,
        "eps": 1e-6,
        "tied": False,
    }
    values.update(overrides)
    return ModelConfig(**values)


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


def test_model_bf16_compute_keeps_params_fp32():
    cfg = tiny_model_config()
    precision = PrecisionConfig(compute_dtype="bf16", param_dtype="fp32", loss_dtype="fp32")
    model = Model(cfg, precision=precision, rngs=nnx.Rngs(0))
    input_ids = jax.random.randint(jax.random.key(0), (2, 8), 0, cfg.vocab_size)

    logits = model(input_ids)
    params = jax.tree.leaves(nnx.state(model, nnx.Param))

    assert logits.dtype == jnp.bfloat16
    assert params
    assert {param.dtype for param in params} == {jnp.dtype(jnp.float32)}
    assert model.loss_dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(logits)))


def test_model_rejects_overlong_sequence():
    cfg = tiny_model_config()
    model = Model(cfg, rngs=nnx.Rngs(0))
    input_ids = jnp.zeros((1, cfg.seq_len + 1), dtype=jnp.int32)

    with pytest.raises(ValueError, match="exceeds seq_len"):
        model(input_ids)


def test_kv_cache_init_shapes():
    cfg = tiny_model_config(n_layers=2, n_kv_heads=2)
    cache = init_kv_cache(cfg, batch_size=3, dtype=jnp.float32)

    assert len(cache.layers) == 2
    assert cache.length.shape == ()
    assert int(cache.length) == 0
    for layer_cache in cache.layers:
        chex.assert_shape(layer_cache.k, (3, cfg.seq_len, cfg.n_kv_heads, cfg.hidden_size // cfg.n_heads))
        chex.assert_shape(layer_cache.v, (3, cfg.seq_len, cfg.n_kv_heads, cfg.hidden_size // cfg.n_heads))
        assert layer_cache.k.dtype == jnp.float32


def test_cached_decode_matches_full_forward():
    cfg = tiny_model_config(n_kv_heads=2)
    model = Model(cfg, rngs=nnx.Rngs(0))
    input_ids = jax.random.randint(jax.random.key(1), (2, 6), 0, cfg.vocab_size)

    full_logits = model(input_ids)
    cache = init_kv_cache(cfg, batch_size=2, dtype=jnp.float32)
    cached_logits = []
    for i in range(input_ids.shape[1]):
        logits, cache = model.decode_one(input_ids[:, i : i + 1], cache)
        cached_logits.append(logits)
    cached_logits = jnp.stack(cached_logits, axis=1)

    chex.assert_trees_all_close(cached_logits, full_logits, rtol=2e-4, atol=2e-4)
    assert int(cache.length) == input_ids.shape[1]


def test_prefill_matches_full_forward_last_position():
    cfg = tiny_model_config(n_kv_heads=2)
    model = Model(cfg, rngs=nnx.Rngs(0))
    input_ids = jax.random.randint(jax.random.key(2), (2, 5), 0, cfg.vocab_size)
    cache = init_kv_cache(cfg, batch_size=2, dtype=jnp.float32)

    full_logits = model(input_ids)
    prefill_logits, cache = model.prefill(input_ids, cache)

    chex.assert_trees_all_close(prefill_logits, full_logits[:, -1, :], rtol=2e-4, atol=2e-4)
    assert int(cache.length) == input_ids.shape[1]
