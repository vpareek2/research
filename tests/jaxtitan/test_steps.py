import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxtitan.batch import Batch
from jaxtitan.errors import ContractError
from jaxtitan.mesh import build_mesh_context, build_sharding_plan, place_batch
from jaxtitan.models import build_model
from jaxtitan.specs.mesh import MeshSpec
from jaxtitan.specs.model import ModelSpec
from jaxtitan.steps import causal_lm_loss, eval_step, make_eval_step


def test_causal_lm_loss_matches_hand_computed_cross_entropy() -> None:
    logits = jnp.array([[[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]]], dtype=jnp.bfloat16)
    target_ids = jnp.array([[0, 2]], dtype=jnp.int32)
    loss_mask = jnp.array([[True, True]])

    output = causal_lm_loss(logits, target_ids, loss_mask)
    expected = -jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)[0, 0, 0]
    expected += -jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)[0, 1, 2]

    assert jnp.allclose(output.loss_sum, expected)
    assert output.token_count == 2
    assert jnp.allclose(output.loss, expected / 2)


def test_causal_lm_loss_uses_boolean_validity_mask() -> None:
    logits = jnp.array(
        [
            [[3.0, 0.0], [0.0, 3.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ],
        dtype=jnp.float32,
    )
    target_ids = jnp.array([[0, 0], [1, 1]], dtype=jnp.int32)
    loss_mask = jnp.array([[True, False], [False, True]])

    output = causal_lm_loss(logits, target_ids, loss_mask)
    losses = -jax.nn.log_softmax(logits, axis=-1)
    expected = losses[0, 0, 0] + losses[1, 1, 1]

    assert jnp.allclose(output.loss_sum, expected)
    assert output.token_count == 2


def test_causal_lm_loss_rejects_shape_mismatches() -> None:
    logits = jnp.zeros((2, 3, 5), dtype=jnp.float32)

    with pytest.raises(ContractError, match="target_ids"):
        causal_lm_loss(logits, jnp.zeros((2, 2), dtype=jnp.int32), jnp.ones((2, 3), dtype=jnp.bool_))
    with pytest.raises(ContractError, match="loss_mask"):
        causal_lm_loss(logits, jnp.zeros((2, 3), dtype=jnp.int32), jnp.ones((2, 2), dtype=jnp.bool_))
    with pytest.raises(ContractError, match="logits"):
        causal_lm_loss(jnp.zeros((2, 3), dtype=jnp.float32), jnp.zeros((2, 3), dtype=jnp.int32), jnp.ones((2, 3)))


def test_eval_step_returns_numerator_denominator_metrics() -> None:
    built = build_model(_tiny_spec(), seed=0)
    batch = _batch(batch_size=2, seq_len=4, vocab_size=16)

    metrics = eval_step(built.graph, built.state, batch)

    assert metrics.loss_sum.shape == ()
    assert metrics.loss_sum.dtype == jnp.float32
    assert metrics.token_count == 8
    assert metrics.num_batches == 1
    assert metrics.byte_count is None
    assert math.isfinite(float(jax.device_get(metrics.loss_sum)))


def test_eval_step_rejects_bad_batch_shapes_before_compile() -> None:
    built = build_model(_tiny_spec(), seed=0)
    bad_batch = Batch(
        input_ids=jnp.zeros((2, 4), dtype=jnp.int32),
        target_ids=jnp.zeros((2, 3), dtype=jnp.int32),
        loss_mask=jnp.ones((2, 4), dtype=jnp.bool_),
    )

    with pytest.raises(ContractError, match="target_ids"):
        eval_step(built.graph, built.state, bad_batch)


def test_make_eval_step_accepts_host_arrays_and_placed_batch() -> None:
    built = build_model(_tiny_spec(), seed=1)
    step = make_eval_step(built.graph)
    batch = _batch(batch_size=2, seq_len=4, vocab_size=16)

    host_metrics = step(built.state, batch)
    context = build_mesh_context(MeshSpec())
    plan = build_sharding_plan(context)
    placed_metrics = step(built.state, place_batch(batch, plan))

    assert host_metrics.token_count == placed_metrics.token_count
    assert jnp.allclose(host_metrics.loss_sum, placed_metrics.loss_sum)


def test_repeated_compiled_eval_calls_return_stable_shapes_and_dtypes() -> None:
    built = build_model(_tiny_spec(), seed=2)
    step = make_eval_step(built.graph)

    first = step(built.state, _batch(batch_size=2, seq_len=4, vocab_size=16))
    second = step(built.state, _batch(batch_size=2, seq_len=4, vocab_size=16, offset=3))

    assert first.loss_sum.shape == second.loss_sum.shape == ()
    assert first.token_count.shape == second.token_count.shape == ()
    assert first.num_batches.shape == second.num_batches.shape == ()
    assert first.loss_sum.dtype == second.loss_sum.dtype == jnp.float32
    assert first.token_count.dtype == second.token_count.dtype
    assert first.num_batches.dtype == second.num_batches.dtype


def _tiny_spec() -> ModelSpec:
    return ModelSpec(
        name="decoder",
        variant="tiny",
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_heads=2,
        n_kv_heads=1,
        max_seq_len=4,
        compute_dtype="float32",
    )


def _batch(*, batch_size: int, seq_len: int, vocab_size: int, offset: int = 0) -> Batch:
    input_ids = (np.arange(batch_size * seq_len, dtype=np.int32).reshape(batch_size, seq_len) + offset) % vocab_size
    target_ids = (input_ids + 1) % vocab_size
    return Batch(
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=np.ones((batch_size, seq_len), dtype=np.bool_),
    )
