import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxtitan.batch import Batch
from jaxtitan.errors import ContractError
from jaxtitan.mesh import build_mesh_context, build_sharding_plan, place_accumulated_batch, place_batch, place_replicated
from jaxtitan.models import build_model
from jaxtitan.optim import build_optimizer
from jaxtitan.specs.mesh import MeshSpec
from jaxtitan.specs.model import ModelSpec
from jaxtitan.specs.optimizer import OptimizerSpec, ScheduleSpec
from jaxtitan.steps import initialize_train_state, make_train_step, train_step

FAKE_DEVICE_COUNT = 4


def require_fake_devices() -> None:
    if jax.local_device_count() < FAKE_DEVICE_COUNT:
        pytest.skip("JAX was initialized before fake CPU device flags were set")


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
    assert metrics.microbatch_loss_mean.shape == ()
    assert metrics.microbatch_loss_max.shape == ()
    assert metrics.batch_het.shape == ()
    assert metrics.batch_het == pytest.approx(jnp.asarray(0.0, dtype=jnp.float32))
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


def test_donating_train_step_returns_next_state_with_expected_shape() -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    state = initialize_train_state(built.state, optimizer.transform, seed=1)
    step = make_train_step(
        built.graph,
        optimizer,
        donate_state=True,
        expected_batch_shape=(1, 2, 4),
    )

    next_state, metrics = step(state, _batch(batch_size=2, seq_len=4, vocab_size=16))

    jax.block_until_ready((next_state.step, next_state.tokens_seen, metrics.loss_sum))
    assert next_state.step == 1
    assert next_state.tokens_seen == 8
    assert metrics.token_count == 8
    assert metrics.loss_sum.shape == ()


def test_train_step_expected_shape_guard_normalizes_rank2_batches() -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    state = initialize_train_state(built.state, optimizer.transform, seed=1)
    step = make_train_step(built.graph, optimizer, expected_batch_shape=(2, 2, 4))

    with pytest.raises(ContractError, match="expected compiled shape"):
        step(state, _batch(batch_size=2, seq_len=4, vocab_size=16))


def test_train_step_expected_shape_guard_rejects_wrong_accumulation_shape() -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    state = initialize_train_state(built.state, optimizer.transform, seed=1)
    step = make_train_step(built.graph, optimizer, expected_batch_shape=(2, 2, 4))
    micro = _batch(batch_size=2, seq_len=4, vocab_size=16)
    bad_batch = Batch(
        input_ids=np.stack([micro.input_ids, micro.input_ids, micro.input_ids]),
        target_ids=np.stack([micro.target_ids, micro.target_ids, micro.target_ids]),
        loss_mask=np.ones((3, 2, 4), dtype=np.bool_),
    )

    with pytest.raises(ContractError, match="expected compiled shape"):
        step(state, bad_batch)


def test_train_step_rejects_bad_batch_dtypes_before_compile() -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    state = initialize_train_state(built.state, optimizer.transform, seed=1)
    good_batch = _batch(batch_size=2, seq_len=4, vocab_size=16)

    with pytest.raises(ContractError, match="input_ids must have integer dtype"):
        train_step(
            built.graph,
            optimizer,
            state,
            Batch(
                input_ids=good_batch.input_ids.astype(np.float32),
                target_ids=good_batch.target_ids,
                loss_mask=good_batch.loss_mask,
            ),
        )
    with pytest.raises(ContractError, match="loss_mask must have bool dtype"):
        train_step(
            built.graph,
            optimizer,
            state,
            Batch(
                input_ids=good_batch.input_ids,
                target_ids=good_batch.target_ids,
                loss_mask=good_batch.loss_mask.astype(np.int32),
            ),
        )


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
    assert first_metrics.microbatch_loss_mean.shape == second_metrics.microbatch_loss_mean.shape == ()
    assert first_metrics.microbatch_loss_max.shape == second_metrics.microbatch_loss_max.shape == ()
    assert first_metrics.batch_het.shape == second_metrics.batch_het.shape == ()
    assert first_metrics.loss_sum.dtype == second_metrics.loss_sum.dtype == jnp.float32
    assert first_metrics.lr.dtype == second_metrics.lr.dtype == jnp.float32
    assert first_metrics.token_count.dtype == second_metrics.token_count.dtype


def test_accumulated_train_step_matches_equivalent_large_batch() -> None:
    built = build_model(_tiny_spec(), seed=0)
    accumulated_batch = Batch(
        input_ids=np.stack(
            [
                _batch(batch_size=2, seq_len=4, vocab_size=16, offset=0).input_ids,
                _batch(batch_size=2, seq_len=4, vocab_size=16, offset=8).input_ids,
            ]
        ),
        target_ids=np.stack(
            [
                _batch(batch_size=2, seq_len=4, vocab_size=16, offset=0).target_ids,
                _batch(batch_size=2, seq_len=4, vocab_size=16, offset=8).target_ids,
            ]
        ),
        loss_mask=np.ones((2, 2, 4), dtype=np.bool_),
    )
    large_batch = Batch(
        input_ids=accumulated_batch.input_ids.reshape(4, 4),
        target_ids=accumulated_batch.target_ids.reshape(4, 4),
        loss_mask=accumulated_batch.loss_mask.reshape(4, 4),
    )
    accum_optimizer = _optimizer(built.state, built.metadata)
    large_optimizer = _optimizer(built.state, built.metadata)
    accum_state = initialize_train_state(built.state, accum_optimizer.transform, seed=1)
    large_state = initialize_train_state(built.state, large_optimizer.transform, seed=1)

    next_accum, accum_metrics = make_train_step(built.graph, accum_optimizer)(accum_state, accumulated_batch)
    next_large, large_metrics = make_train_step(built.graph, large_optimizer)(large_state, large_batch)

    assert next_accum.step == 1
    assert next_accum.tokens_seen == 16
    assert accum_metrics.token_count == 16
    assert accum_metrics.lr == large_metrics.lr
    assert accum_metrics.microbatch_loss_max >= accum_metrics.microbatch_loss_mean
    assert accum_metrics.batch_het == pytest.approx(accum_metrics.microbatch_loss_max - accum_metrics.microbatch_loss_mean)
    assert large_metrics.batch_het == pytest.approx(jnp.asarray(0.0, dtype=jnp.float32))
    assert np.allclose(np.asarray(jax.device_get(accum_metrics.loss_sum)), np.asarray(jax.device_get(large_metrics.loss_sum)))
    assert _trees_close(next_accum.model, next_large.model)


def test_block_remat_train_step_matches_plain_with_gradient_accumulation() -> None:
    plain = build_model(_tiny_spec(compute_dtype="float32", remat="none"), seed=0)
    remat = build_model(_tiny_spec(compute_dtype="float32", remat="block"), seed=0)
    batch = _batch(batch_size=2, seq_len=4, vocab_size=16)
    accumulated_batch = Batch(
        input_ids=np.stack([batch.input_ids, batch.input_ids]),
        target_ids=np.stack([batch.target_ids, batch.target_ids]),
        loss_mask=np.ones((2, 2, 4), dtype=np.bool_),
    )
    plain_optimizer = _optimizer(plain.state, plain.metadata)
    remat_optimizer = _optimizer(remat.state, remat.metadata)
    plain_state = initialize_train_state(plain.state, plain_optimizer.transform, seed=1)
    remat_state = initialize_train_state(remat.state, remat_optimizer.transform, seed=1)

    next_plain, plain_metrics = make_train_step(
        plain.graph,
        plain_optimizer,
        donate_state=True,
        expected_batch_shape=(2, 2, 4),
    )(plain_state, accumulated_batch)
    next_remat, remat_metrics = make_train_step(
        remat.graph,
        remat_optimizer,
        donate_state=True,
        expected_batch_shape=(2, 2, 4),
    )(remat_state, accumulated_batch)

    assert _scalar_int(remat_metrics.token_count) == _scalar_int(plain_metrics.token_count) == 16
    assert np.allclose(
        np.asarray(jax.device_get(remat_metrics.loss_sum)),
        np.asarray(jax.device_get(plain_metrics.loss_sum)),
        rtol=1e-5,
        atol=1e-5,
    )
    assert _trees_close(next_remat.model, next_plain.model)


def test_muon_train_step_supports_accumulation_remat_donation_and_sharding() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(compute_dtype="float32", remat="block"), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(context)
    model_state = place_replicated(built.state, plan)
    optimizer = _optimizer(model_state, built.metadata, optimizer_name="muon")
    state = initialize_train_state(model_state, optimizer.transform, seed=1)
    step = make_train_step(
        built.graph,
        optimizer,
        sharding=plan,
        state_template=state,
        donate_state=True,
        expected_batch_shape=(2, 8, 4),
    )
    micro = _batch(batch_size=8, seq_len=4, vocab_size=16)
    batch = Batch(
        input_ids=np.stack([micro.input_ids, micro.input_ids]),
        target_ids=np.stack([micro.target_ids, micro.target_ids]),
        loss_mask=np.ones((2, 8, 4), dtype=np.bool_),
    )

    next_state, metrics = step(state, place_accumulated_batch(batch, plan))

    assert next_state.step == 1
    assert next_state.tokens_seen == 64
    assert metrics.token_count == 64
    assert metrics.loss_sum.shape == ()
    assert {assignment.backend for assignment in optimizer.route_assignments} == {"adamw", "muon"}


def test_train_step_with_data_axis_sharding_reports_global_metrics() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(context)
    model_state = place_replicated(built.state, plan)
    optimizer = _optimizer(model_state, built.metadata)
    state = initialize_train_state(model_state, optimizer.transform, seed=1)
    step = make_train_step(
        built.graph,
        optimizer,
        sharding=plan,
        state_template=state,
        donate_state=True,
        expected_batch_shape=(1, 8, 4),
    )
    batch = _batch(batch_size=8, seq_len=4, vocab_size=16)

    next_state, metrics = step(state, place_batch(batch, plan))

    assert next_state.step == 1
    assert next_state.tokens_seen == 32
    assert metrics.token_count == 32
    assert metrics.loss_sum.shape == ()
    assert metrics.loss_sum.sharding == plan.metrics
    assert metrics.batch_het.shape == ()
    assert metrics.batch_het.sharding == plan.metrics
    assert next_state.step.sharding == plan.replicated
    assert jax.tree.leaves(next_state.model)[0].sharding == plan.replicated


def test_accumulated_train_step_with_data_axis_sharding_reports_global_metrics() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(context)
    model_state = place_replicated(built.state, plan)
    optimizer = _optimizer(model_state, built.metadata)
    state = initialize_train_state(model_state, optimizer.transform, seed=1)
    step = make_train_step(
        built.graph,
        optimizer,
        sharding=plan,
        state_template=state,
        donate_state=True,
        expected_batch_shape=(2, 8, 4),
    )
    micro = _batch(batch_size=8, seq_len=4, vocab_size=16)
    batch = Batch(
        input_ids=np.stack([micro.input_ids, micro.input_ids]),
        target_ids=np.stack([micro.target_ids, micro.target_ids]),
        loss_mask=np.ones((2, 8, 4), dtype=np.bool_),
    )

    next_state, metrics = step(state, place_accumulated_batch(batch, plan))

    assert next_state.step == 1
    assert next_state.tokens_seen == 64
    assert metrics.token_count == 64
    assert metrics.loss_sum.shape == ()
    assert metrics.loss_sum.sharding == plan.metrics
    assert next_state.step.sharding == plan.replicated


def test_data_axis_sharded_train_step_matches_one_device_global_batch() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(), seed=0)
    batch = _batch(batch_size=8, seq_len=4, vocab_size=16)

    one_context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(1,)))
    one_plan = build_sharding_plan(one_context)
    one_model = place_replicated(built.state, one_plan)
    one_optimizer = _optimizer(one_model, built.metadata)
    one_state = initialize_train_state(one_model, one_optimizer.transform, seed=1)
    one_step = make_train_step(built.graph, one_optimizer, sharding=one_plan, state_template=one_state)
    next_one, metrics_one = one_step(one_state, place_batch(batch, one_plan))

    four_context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    four_plan = build_sharding_plan(four_context)
    four_model = place_replicated(built.state, four_plan)
    four_optimizer = _optimizer(four_model, built.metadata)
    four_state = initialize_train_state(four_model, four_optimizer.transform, seed=1)
    four_step = make_train_step(built.graph, four_optimizer, sharding=four_plan, state_template=four_state)
    next_four, metrics_four = four_step(four_state, place_batch(batch, four_plan))

    assert _scalar_int(metrics_four.token_count) == _scalar_int(metrics_one.token_count) == 32
    assert np.allclose(
        np.asarray(jax.device_get(metrics_four.loss_sum)),
        np.asarray(jax.device_get(metrics_one.loss_sum)),
        rtol=1e-5,
        atol=1e-5,
    )
    assert _trees_close(next_four.model, next_one.model)


def _optimizer(model_state, metadata, *, schedule: ScheduleSpec | None = None, optimizer_name: str = "adamw"):
    if schedule is None:
        schedule = ScheduleSpec(peak_lr=1e-3)
    return build_optimizer(
        OptimizerSpec(name=optimizer_name, schedule=schedule, weight_decay=0.01),
        model_state,
        metadata,
    )


def _tiny_spec(**overrides) -> ModelSpec:
    values = {
        "name": "decoder",
        "variant": "tiny",
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_layers": 1,
        "num_heads": 2,
        "n_kv_heads": 1,
        "max_seq_len": 4,
        "compute_dtype": "float32",
    }
    values.update(overrides)
    return ModelSpec(**values)


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


def _trees_close(left, right) -> bool:
    return all(
        np.allclose(
            np.asarray(jax.device_get(left_leaf)),
            np.asarray(jax.device_get(right_leaf)),
            rtol=1e-5,
            atol=1e-5,
        )
        for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _scalar_int(value) -> int:
    return int(np.asarray(jax.device_get(value)).item())


def _rng_equal(left, right) -> bool:
    return all(
        jnp.array_equal(getattr(left, name), getattr(right, name))
        for name in ("train", "data", "eval", "sample")
    )
