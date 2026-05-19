import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxtitan.batch import Batch
from jaxtitan.errors import ContractError
from jaxtitan.models import build_model
from jaxtitan.optim import build_optimizer
from jaxtitan.specs.model import ModelSpec
from jaxtitan.specs.optimizer import OptimizerSpec, ScheduleSpec
from jaxtitan.steps import initialize_train_state, make_train_step, train_step


def test_initialize_train_state_is_seed_deterministic() -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)

    first = initialize_train_state(built.state, optimizer.transform, seed=123)
    second = initialize_train_state(built.state, optimizer.transform, seed=123)
    third = initialize_train_state(built.state, optimizer.transform, seed=124)

    assert first.step == 0
    assert first.step.dtype == jnp.int32
    assert first.tokens_seen == 0
    assert first.tokens_seen.dtype == jnp.uint32
    assert first.schedule_state is None
    assert _trees_equal(first.opt_state, second.opt_state)
    assert _rng_equal(first.rng, second.rng)
    assert not _rng_equal(first.rng, third.rng)


def test_train_step_updates_model_optimizer_state_and_metrics() -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    state = initialize_train_state(built.state, optimizer.transform, seed=1)
    batch = _batch(batch_size=2, seq_len=4, vocab_size=16)

    next_state, metrics = train_step(built.graph, optimizer, state, batch)

    assert next_state.step == 1
    assert next_state.tokens_seen == 8
    assert _trees_changed(state.model, next_state.model)
    assert _trees_changed(state.opt_state, next_state.opt_state)
    assert _rng_equal(state.rng, next_state.rng)
    assert metrics.loss_sum.shape == ()
    assert metrics.token_count == 8
    assert metrics.lr.shape == ()
    assert metrics.grad_norm.shape == ()
    assert metrics.param_norm.shape == ()
    assert metrics.update_norm.shape == ()
    assert metrics.overflow is None
    assert math.isfinite(float(jax.device_get(metrics.loss_sum)))
    assert math.isfinite(float(jax.device_get(metrics.grad_norm)))
    assert math.isfinite(float(jax.device_get(metrics.param_norm)))
    assert math.isfinite(float(jax.device_get(metrics.update_norm)))


def test_train_state_is_a_jax_pytree() -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    state = initialize_train_state(built.state, optimizer.transform, seed=1)

    leaves = jax.tree.leaves(state)

    assert any(jnp.array_equal(leaf, state.step) for leaf in leaves)
    assert any(jnp.array_equal(leaf, state.tokens_seen) for leaf in leaves)
    assert any(
        getattr(leaf, "dtype", None) == state.rng.train.dtype and jnp.array_equal(leaf, state.rng.train)
        for leaf in leaves
    )
    assert len(leaves) > len(jax.tree.leaves(state.model))


def test_train_step_reports_lr_from_old_step() -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(
        built.state,
        built.metadata,
        schedule=ScheduleSpec(name="constant", peak_lr=0.2, warmup_steps=2),
    )
    state = initialize_train_state(built.state, optimizer.transform, seed=1)
    step = make_train_step(built.graph, optimizer)
    batch = _batch(batch_size=2, seq_len=4, vocab_size=16)

    state, first_metrics = step(state, batch)
    state, second_metrics = step(state, batch)

    assert first_metrics.lr == pytest.approx(jnp.asarray(0.1, dtype=jnp.float32))
    assert second_metrics.lr == pytest.approx(jnp.asarray(0.2, dtype=jnp.float32))
    assert state.step == 2


def test_partial_mask_controls_tokens_seen_and_metric_denominator() -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    state = initialize_train_state(built.state, optimizer.transform, seed=1)
    batch = _batch(batch_size=2, seq_len=4, vocab_size=16)
    batch = Batch(
        input_ids=batch.input_ids,
        target_ids=batch.target_ids,
        loss_mask=np.array([[True, False, True, False], [False, False, False, False]], dtype=np.bool_),
    )

    next_state, metrics = train_step(built.graph, optimizer, state, batch)

    assert metrics.token_count == 2
    assert next_state.tokens_seen == 2
    assert math.isfinite(float(jax.device_get(metrics.loss_sum)))


def test_train_step_rejects_bad_batch_shapes_before_compile() -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    state = initialize_train_state(built.state, optimizer.transform, seed=1)
    bad_batch = Batch(
        input_ids=jnp.zeros((2, 4), dtype=jnp.int32),
        target_ids=jnp.zeros((2, 3), dtype=jnp.int32),
        loss_mask=jnp.ones((2, 4), dtype=jnp.bool_),
    )

    with pytest.raises(ContractError, match="target_ids"):
        train_step(built.graph, optimizer, state, bad_batch)


def test_repeated_compiled_train_calls_return_stable_shapes_and_dtypes() -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    state = initialize_train_state(built.state, optimizer.transform, seed=1)
    step = make_train_step(built.graph, optimizer)

    first_state, first_metrics = step(state, _batch(batch_size=2, seq_len=4, vocab_size=16))
    second_state, second_metrics = step(first_state, _batch(batch_size=2, seq_len=4, vocab_size=16, offset=3))

    assert first_state.step.shape == second_state.step.shape == ()
    assert first_state.tokens_seen.shape == second_state.tokens_seen.shape == ()
    assert first_metrics.loss_sum.shape == second_metrics.loss_sum.shape == ()
    assert first_metrics.token_count.shape == second_metrics.token_count.shape == ()
    assert first_metrics.lr.shape == second_metrics.lr.shape == ()
    assert first_metrics.grad_norm.shape == second_metrics.grad_norm.shape == ()
    assert first_metrics.param_norm.shape == second_metrics.param_norm.shape == ()
    assert first_metrics.update_norm.shape == second_metrics.update_norm.shape == ()
    assert first_metrics.loss_sum.dtype == second_metrics.loss_sum.dtype == jnp.float32
    assert first_metrics.lr.dtype == second_metrics.lr.dtype == jnp.float32
    assert first_metrics.token_count.dtype == second_metrics.token_count.dtype


def _optimizer(model_state, metadata, *, schedule: ScheduleSpec | None = None):
    if schedule is None:
        schedule = ScheduleSpec(peak_lr=1e-3)
    return build_optimizer(
        OptimizerSpec(name="adamw", schedule=schedule, weight_decay=0.01),
        model_state,
        metadata,
    )


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


def _trees_equal(left, right) -> bool:
    return all(
        jnp.array_equal(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _trees_changed(left, right) -> bool:
    return any(
        not jnp.array_equal(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _rng_equal(left, right) -> bool:
    return all(
        jnp.array_equal(getattr(left, name), getattr(right, name))
        for name in ("train", "data", "eval", "sample")
    )
