import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from jaxtitan.batch import Batch
from jaxtitan.errors import ContractError
from jaxtitan.mesh import (
    build_mesh_context,
    build_sharding_plan,
    place_accumulated_batch,
    place_batch,
    place_model_state,
    place_optimizer_init_state,
    place_replicated,
)
from jaxtitan.models import AuxLoss, ModelOutput, build_model
from jaxtitan.optim import OptimizerBuildResult, RouteAssignment, build_optimizer
import jaxtitan.steps.train as train_module
from jaxtitan.specs.mesh import MeshSpec
from jaxtitan.specs.model import ModelSpec, TrinitySpec
from jaxtitan.specs.optimizer import OptimizerSpec, ScheduleSpec
from jaxtitan.specs.parallelism import ParallelismSpec
from jaxtitan.specs.run import TrainingLossSpec
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
    _assert_optimizer_groups_cover(metrics, built.metadata)


def test_train_step_updates_dense_trinity_model() -> None:
    built = build_model(_tiny_trinity_spec(), seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    state = initialize_train_state(built.state, optimizer.transform, seed=1)
    batch = _batch(batch_size=2, seq_len=4, vocab_size=16)

    next_state, metrics = train_step(built.graph, optimizer, state, batch)

    assert next_state.step == 1
    assert next_state.tokens_seen == 8
    assert _trees_changed(state.model, next_state.model)
    assert _trees_changed(state.opt_state, next_state.opt_state)
    assert metrics.token_count == 8
    assert math.isfinite(float(jax.device_get(metrics.loss_sum)))


def test_train_step_updates_trinity_moe_model_with_adamw() -> None:
    built = build_model(
        _tiny_trinity_spec(
            num_layers=2,
            initial_dense_layers=1,
            norm_policy="afmoe_dual",
            moe={"num_experts": 3, "top_k": 2, "num_shared_experts": 1, "route_scale": 1.25},
        ),
        seed=0,
    )
    optimizer = _optimizer(built.state, built.metadata)
    state = initialize_train_state(built.state, optimizer.transform, seed=1)
    batch = _batch(batch_size=2, seq_len=4, vocab_size=16)

    next_state, metrics = train_step(built.graph, optimizer, state, batch)

    assert next_state.step == 1
    assert next_state.tokens_seen == 8
    assert _trees_changed(state.model, next_state.model)
    assert _trees_changed(state.opt_state, next_state.opt_state)
    assert metrics.token_count == 8
    assert metrics.router_expert_counts.shape == (1, 3)
    assert metrics.router_importance.shape == (1, 3)
    assert jnp.sum(metrics.router_expert_counts) == 2 * 4 * 2
    assert float(jax.device_get(jnp.sum(metrics.router_importance))) == pytest.approx(2 * 4 * 1.25, abs=1e-2)
    assert metrics.router_dead_experts_count >= 0
    assert metrics.router_experts_active_mean <= 3
    assert metrics.router_mean_importance_entropy is not None
    assert metrics.smebu_bias_norm is None
    assert metrics.smebu_momentum_norm is None
    tags = {group["tag"] for group in metrics.optimizer_group_specs}
    assert {"moe_router", "moe_expert_bias", "moe_gate", "moe_up", "moe_down"}.issubset(tags)
    assert math.isfinite(float(jax.device_get(metrics.loss_sum)))


def test_train_step_updates_trinity_moe_model_with_per_expert_muon_routes() -> None:
    built = build_model(
        _tiny_trinity_spec(
            num_layers=2,
            initial_dense_layers=1,
            moe={"num_experts": 3, "top_k": 2, "num_shared_experts": 1},
        ),
        seed=0,
    )
    optimizer = _optimizer(built.state, built.metadata, optimizer_name="muon")
    state = initialize_train_state(built.state, optimizer.transform, seed=1)
    batch = _batch(batch_size=2, seq_len=4, vocab_size=16)

    next_state, metrics = train_step(built.graph, optimizer, state, batch)

    routes = {assignment.tag: assignment for assignment in optimizer.route_assignments if assignment.tag.startswith("moe_")}
    for tag in ("moe_gate", "moe_up", "moe_down"):
        assert routes[tag].backend == "muon"
        assert routes[tag].fallback_reason is None
    for tag in ("moe_shared_gate", "moe_shared_up", "moe_shared_down"):
        assert routes[tag].backend == "muon"
    assert routes["moe_router"].backend == "adamw"
    assert routes["moe_expert_bias"].backend == "adamw"
    assert routes["moe_expert_bias"].weight_decay is False
    assert routes["moe_expert_bias"].fallback_reason == "expert_bias"
    assert next_state.step == 1
    assert next_state.tokens_seen == 8
    assert metrics.token_count == 8
    assert math.isfinite(float(jax.device_get(metrics.loss_sum)))


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


def test_train_step_aux_loss_changes_objective_without_changing_lm_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_apply_model_output(graph, params, input_ids):
        logits = jnp.broadcast_to(jnp.stack([params, -params]), (*input_ids.shape, 2))
        aux_losses = ()
        if graph == "with_aux":
            aux_losses = (AuxLoss(name="synthetic", value=jnp.square(params), weight=0.5),)
        return ModelOutput(logits=logits, aux_losses=aux_losses)

    monkeypatch.setattr(train_module, "apply_model_output", fake_apply_model_output)
    optimizer = OptimizerBuildResult(
        transform=optax.sgd(0.1),
        schedule=lambda step: jnp.asarray(0.1, dtype=jnp.float32),
        adamw_fallback_schedule=None,
        route_assignments=(),
        description="sgd",
    )
    initial_model = jnp.asarray(0.25, dtype=jnp.float32)
    plain_state = initialize_train_state(initial_model, optimizer.transform, seed=1)
    aux_state = initialize_train_state(initial_model, optimizer.transform, seed=1)
    batch = Batch(
        input_ids=jnp.zeros((2, 3), dtype=jnp.int32),
        target_ids=jnp.zeros((2, 3), dtype=jnp.int32),
        loss_mask=jnp.ones((2, 3), dtype=jnp.bool_),
    )

    next_plain, plain_metrics = make_train_step("plain", optimizer)(plain_state, batch)
    next_aux, aux_metrics = make_train_step("with_aux", optimizer)(aux_state, batch)

    assert jnp.allclose(aux_metrics.loss_sum, plain_metrics.loss_sum)
    assert aux_metrics.token_count == plain_metrics.token_count == 6
    assert aux_metrics.aux_loss > 0
    assert aux_metrics.objective > plain_metrics.objective
    assert aux_metrics.total_loss == aux_metrics.objective
    assert next_aux.model < next_plain.model


def test_train_step_z_loss_changes_objective_without_changing_lm_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_apply_model_output(_graph, params, input_ids):
        logits = jnp.broadcast_to(jnp.stack([params, -params]), (*input_ids.shape, 2))
        return ModelOutput(logits=logits)

    monkeypatch.setattr(train_module, "apply_model_output", fake_apply_model_output)
    optimizer = OptimizerBuildResult(
        transform=optax.sgd(0.1),
        schedule=lambda step: jnp.asarray(0.1, dtype=jnp.float32),
        adamw_fallback_schedule=None,
        route_assignments=(),
        description="sgd",
    )
    initial_model = jnp.asarray(0.25, dtype=jnp.float32)
    plain_state = initialize_train_state(initial_model, optimizer.transform, seed=1)
    z_state = initialize_train_state(initial_model, optimizer.transform, seed=1)
    batch = Batch(
        input_ids=jnp.zeros((2, 3), dtype=jnp.int32),
        target_ids=jnp.zeros((2, 3), dtype=jnp.int32),
        loss_mask=jnp.ones((2, 3), dtype=jnp.bool_),
    )

    _next_plain, plain_metrics = make_train_step("plain", optimizer)(plain_state, batch)
    _next_z, z_metrics = make_train_step(
        "plain",
        optimizer,
        loss=TrainingLossSpec(z_loss_weight=0.1),
    )(z_state, batch)

    assert jnp.allclose(z_metrics.loss_sum, plain_metrics.loss_sum)
    assert z_metrics.token_count == plain_metrics.token_count == 6
    assert z_metrics.z_loss > 0
    assert z_metrics.total_loss > plain_metrics.total_loss


def test_optimizer_group_diagnostics_report_zero_gradients_and_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    route = RouteAssignment(
        path=("w",),
        tag="synthetic",
        backend="adamw",
        weight_decay=True,
    )
    batch = Batch(
        input_ids=jnp.zeros((2, 3), dtype=jnp.int32),
        target_ids=jnp.zeros((2, 3), dtype=jnp.int32),
        loss_mask=jnp.ones((2, 3), dtype=jnp.bool_),
    )

    def constant_output(_graph, _params, input_ids):
        logits = jnp.zeros((*input_ids.shape, 2), dtype=jnp.float32)
        return ModelOutput(logits=logits)

    monkeypatch.setattr(train_module, "apply_model_output", constant_output)
    zero_grad_optimizer = OptimizerBuildResult(
        transform=optax.sgd(0.1),
        schedule=lambda step: jnp.asarray(0.1, dtype=jnp.float32),
        adamw_fallback_schedule=None,
        route_assignments=(route,),
        description="sgd",
    )
    zero_grad_state = initialize_train_state({"w": jnp.asarray([0.5], dtype=jnp.float32)}, zero_grad_optimizer.transform, seed=1)
    _next_zero_grad, zero_grad_metrics = make_train_step("constant", zero_grad_optimizer)(zero_grad_state, batch)

    assert zero_grad_metrics.optimizer_group_specs[0]["group"] == "synthetic:adamw"
    assert np.allclose(np.asarray(jax.device_get(zero_grad_metrics.optimizer_group_grad_norms)), [0.0])
    assert np.allclose(np.asarray(jax.device_get(zero_grad_metrics.optimizer_group_update_norms)), [0.0])

    def parameter_output(_graph, params, input_ids):
        logits = jnp.broadcast_to(jnp.stack([params["w"][0], -params["w"][0]]), (*input_ids.shape, 2))
        return ModelOutput(logits=logits)

    monkeypatch.setattr(train_module, "apply_model_output", parameter_output)
    zero_update_optimizer = OptimizerBuildResult(
        transform=optax.sgd(0.0),
        schedule=lambda step: jnp.asarray(0.0, dtype=jnp.float32),
        adamw_fallback_schedule=None,
        route_assignments=(route,),
        description="sgd",
    )
    zero_update_state = initialize_train_state(
        {"w": jnp.asarray([0.5], dtype=jnp.float32)},
        zero_update_optimizer.transform,
        seed=1,
    )
    _next_zero_update, zero_update_metrics = make_train_step("parameter", zero_update_optimizer)(
        zero_update_state,
        batch,
    )

    assert float(jax.device_get(zero_update_metrics.optimizer_group_grad_norms[0])) > 0.0
    assert float(jax.device_get(zero_update_metrics.optimizer_group_update_norms[0])) == pytest.approx(0.0)


def test_smebu_updates_expert_bias_and_momentum_after_train_step() -> None:
    spec = _tiny_trinity_spec(
        num_layers=2,
        initial_dense_layers=1,
        moe={
            "num_experts": 3,
            "top_k": 2,
            "balance": {
                "name": "smebu",
                "load_lr": 1e-2,
                "momentum": 0.5,
                "clamp": 2.0,
                "sequence_aux_loss_weight": 1e-4,
            },
        },
    )
    built = build_model(spec, seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    state = initialize_train_state(
        built.state,
        optimizer.transform,
        seed=1,
        moe_balance_spec=spec.trinity.moe.balance,
    )
    bias_path = next(item.path for item in built.metadata if item.tag == "moe_expert_bias")

    next_state, metrics = make_train_step(built.graph, optimizer)(state, _batch(batch_size=2, seq_len=4, vocab_size=16))

    assert next_state.moe_balance is not None
    bias = _state_value_by_path(next_state.model, bias_path)
    momentum = next_state.moe_balance.layers[0].momentum
    assert not jnp.allclose(momentum, jnp.zeros_like(momentum))
    assert jnp.allclose(bias, momentum, atol=1e-6)
    assert metrics.moe_aux_loss > 0
    assert metrics.router_max_vio is not None
    assert metrics.smebu_bias_norm is not None
    assert metrics.smebu_momentum_norm is not None


def test_smebu_gradient_accumulation_aggregates_counts_once_per_step() -> None:
    spec = _tiny_trinity_spec(
        num_layers=2,
        initial_dense_layers=1,
        moe={"num_experts": 3, "top_k": 2, "balance": {"name": "smebu", "load_lr": 1e-2}},
    )
    built = build_model(spec, seed=0)
    optimizer = _optimizer(built.state, built.metadata)
    initial_state = initialize_train_state(
        built.state,
        optimizer.transform,
        seed=1,
        moe_balance_spec=spec.trinity.moe.balance,
    )
    batch = _batch(batch_size=2, seq_len=4, vocab_size=16)
    accumulated_batch = Batch(
        input_ids=np.stack([batch.input_ids, batch.input_ids]),
        target_ids=np.stack([batch.target_ids, batch.target_ids]),
        loss_mask=np.ones((2, 2, 4), dtype=np.bool_),
    )

    next_accum, metrics = make_train_step(built.graph, optimizer)(initial_state, accumulated_batch)
    after_first, single_metrics = make_train_step(built.graph, optimizer)(initial_state, batch)

    assert metrics.token_count == 16
    assert jnp.allclose(metrics.router_expert_counts, single_metrics.router_expert_counts * 2)
    assert jnp.allclose(metrics.router_importance, single_metrics.router_importance * 2)
    assert jnp.allclose(next_accum.moe_balance.layers[0].momentum, after_first.moe_balance.layers[0].momentum)


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


def test_ep_train_step_shards_routed_experts_and_uses_per_expert_muon() -> None:
    require_fake_devices()
    built = build_model(
        _tiny_trinity_spec(
            num_layers=2,
            initial_dense_layers=1,
            moe={"num_experts": 4, "top_k": 2},
            compute_dtype="float32",
            remat="block",
        ),
        seed=0,
    )
    context = build_mesh_context(MeshSpec(axis_names=("data", "ep"), axis_sizes=(1, 4)))
    plan = build_sharding_plan(
        context,
        parallelism=ParallelismSpec(mode="ddp", expert_parallel=True),
        param_layouts=built.param_layouts,
        expert_layouts=built.expert_layouts,
    )
    model_state = place_model_state(built.state, plan)
    optimizer = _optimizer(model_state, built.metadata, optimizer_name="muon")
    state = initialize_train_state(model_state, optimizer.transform, seed=1)
    step = make_train_step(
        built.graph,
        optimizer,
        sharding=plan,
        state_template=state,
        donate_state=True,
        expected_batch_shape=(1, 4, 4),
    )
    batch = _batch(batch_size=4, seq_len=4, vocab_size=16)

    next_state, metrics = step(state, place_batch(batch, plan))

    routes = {assignment.tag: assignment for assignment in optimizer.route_assignments if assignment.tag.startswith("moe_")}
    for tag in ("moe_gate", "moe_up", "moe_down"):
        assert routes[tag].backend == "muon"
        assert routes[tag].resolution_reason == "expert_axis_sharded_full_matrices"
    assert next_state.step == 1
    assert next_state.tokens_seen == 16
    assert metrics.token_count == 16
    assert metrics.loss_sum.shape == ()
    assert any(
        getattr(leaf, "sharding", None).spec == jax.sharding.PartitionSpec("ep", None, None)
        for leaf in jax.tree.leaves(next_state.model)
        if hasattr(getattr(leaf, "sharding", None), "spec")
    )


def test_fsdp_ep_train_step_uses_dion2_for_dense_and_muon_for_routed_experts() -> None:
    require_fake_devices()
    built = build_model(
        _tiny_trinity_spec(
            num_layers=2,
            initial_dense_layers=1,
            moe={"num_experts": 4, "top_k": 2, "num_shared_experts": 1},
            compute_dtype="float32",
        ),
        seed=0,
    )
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp", "ep"), axis_sizes=(1, 2, 2)))
    plan = build_sharding_plan(
        context,
        parallelism=ParallelismSpec(mode="fsdp", expert_parallel=True),
        param_layouts=built.param_layouts,
        expert_layouts=built.expert_layouts,
    )
    model_state = place_model_state(built.state, plan)
    optimizer = _optimizer(model_state, built.metadata, optimizer_name="muon")
    state = initialize_train_state(model_state, optimizer.transform, seed=1)
    step = make_train_step(
        built.graph,
        optimizer,
        sharding=plan,
        state_template=state,
        donate_state=True,
        expected_batch_shape=(1, 4, 4),
    )
    batch = _batch(batch_size=4, seq_len=4, vocab_size=16)

    next_state, metrics = step(state, place_batch(batch, plan))

    routes = {assignment.tag: assignment.backend for assignment in optimizer.route_assignments}
    assert routes["attention_q"] == "dion2"
    assert routes["moe_gate"] == "muon"
    assert routes["moe_up"] == "muon"
    assert routes["moe_down"] == "muon"
    assert next_state.step == 1
    assert next_state.tokens_seen == 16
    assert metrics.token_count == 16
    assert metrics.loss_sum.shape == ()


def test_expert_region_fsdp_train_step_runs_with_adamw() -> None:
    require_fake_devices()
    built = build_model(
        _tiny_trinity_spec(
            num_layers=2,
            initial_dense_layers=1,
            moe={"num_experts": 4, "top_k": 2, "expert_intermediate_size": 16, "num_shared_experts": 1},
            compute_dtype="float32",
        ),
        seed=0,
    )
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp", "ep", "expert_fsdp"), axis_sizes=(1, 1, 2, 2)))
    plan = build_sharding_plan(
        context,
        parallelism=ParallelismSpec(mode="fsdp", expert_parallel=True),
        param_layouts=built.param_layouts,
        expert_layouts=built.expert_layouts,
    )
    model_state = place_model_state(built.state, plan)
    optimizer = _optimizer(model_state, built.metadata, optimizer_name="adamw")
    state = initialize_train_state(model_state, optimizer.transform, seed=1)
    step = make_train_step(
        built.graph,
        optimizer,
        sharding=plan,
        state_template=state,
        donate_state=True,
        expected_batch_shape=(1, 4, 4),
    )
    batch = _batch(batch_size=4, seq_len=4, vocab_size=16)

    next_state, metrics = step(state, place_batch(batch, plan))

    assert plan.expert_parallel_axis == "ep"
    assert plan.expert_fsdp_axis == "expert_fsdp"
    assert next_state.step == 1
    assert next_state.tokens_seen == 16
    assert metrics.token_count == 16
    assert metrics.loss_sum.shape == ()
    assert any(
        getattr(leaf, "sharding", None).spec == jax.sharding.PartitionSpec("ep", None, "expert_fsdp")
        for leaf in jax.tree.leaves(next_state.model)
        if hasattr(getattr(leaf, "sharding", None), "spec")
    )


def test_data_axis_rdep_train_step_runs_with_adamw() -> None:
    require_fake_devices()
    built = build_model(
        _tiny_trinity_spec(
            num_layers=2,
            initial_dense_layers=1,
            moe={"num_experts": 4, "top_k": 2, "num_shared_experts": 1},
            compute_dtype="float32",
        ),
        seed=0,
    )
    context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(4,)))
    plan = build_sharding_plan(
        context,
        parallelism=ParallelismSpec(mode="ddp", expert_parallel=True, expert_parallel_axis="data"),
        param_layouts=built.param_layouts,
        expert_layouts=built.expert_layouts,
    )
    model_state = place_model_state(built.state, plan)
    optimizer = _optimizer(model_state, built.metadata, optimizer_name="adamw")
    state = initialize_train_state(model_state, optimizer.transform, seed=1)
    step = make_train_step(
        built.graph,
        optimizer,
        sharding=plan,
        state_template=state,
        donate_state=True,
        expected_batch_shape=(1, 4, 4),
    )
    batch = _batch(batch_size=4, seq_len=4, vocab_size=16)

    next_state, metrics = step(state, place_batch(batch, plan))

    assert plan.expert_parallel_axis == "data"
    assert plan.expert_parallel_axis_sharing == "shared_with_data"
    assert plan.expert_parallel_dispatcher == "rdep_static"
    assert next_state.step == 1
    assert next_state.tokens_seen == 16
    assert metrics.token_count == 16
    assert metrics.loss_sum.shape == ()
    assert any(
        getattr(leaf, "sharding", None).spec == jax.sharding.PartitionSpec("data", None, None)
        for leaf in jax.tree.leaves(next_state.model)
        if hasattr(getattr(leaf, "sharding", None), "spec")
    )


def test_folded_fsdp_ep_train_step_uses_dion2_for_dense_and_muon_for_routed_experts() -> None:
    require_fake_devices()
    built = build_model(
        _tiny_trinity_spec(
            num_layers=2,
            initial_dense_layers=1,
            moe={"num_experts": 4, "top_k": 2, "num_shared_experts": 1},
            compute_dtype="float32",
        ),
        seed=0,
    )
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 4)))
    plan = build_sharding_plan(
        context,
        parallelism=ParallelismSpec(mode="fsdp", expert_parallel=True),
        param_layouts=built.param_layouts,
        expert_layouts=built.expert_layouts,
    )
    model_state = place_model_state(built.state, plan)
    optimizer = _optimizer(model_state, built.metadata, optimizer_name="muon")
    state = initialize_train_state(model_state, optimizer.transform, seed=1)
    step = make_train_step(
        built.graph,
        optimizer,
        sharding=plan,
        state_template=state,
        donate_state=True,
        expected_batch_shape=(1, 4, 4),
    )
    batch = _batch(batch_size=4, seq_len=4, vocab_size=16)

    next_state, metrics = step(state, place_batch(batch, plan))

    routes = {assignment.tag: assignment for assignment in optimizer.route_assignments}
    assert plan.expert_parallel_axis == "fsdp"
    assert plan.expert_parallel_axis_sharing == "shared_with_fsdp"
    assert routes["attention_q"].backend == "dion2"
    assert routes["moe_shared_gate"].backend == "dion2"
    for tag in ("moe_gate", "moe_up", "moe_down"):
        assert routes[tag].backend == "muon"
        assert routes[tag].resolution_reason == "expert_axis_sharded_full_matrices"
    assert next_state.step == 1
    assert next_state.tokens_seen == 16
    assert metrics.token_count == 16
    assert metrics.loss_sum.shape == ()


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


def test_tensor_parallel_train_step_matches_replicated_global_batch() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(hidden_size=16, intermediate_size=32, num_heads=4, n_kv_heads=4), seed=0)
    batch = _batch(batch_size=4, seq_len=4, vocab_size=16)

    one_context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(1,)))
    one_plan = build_sharding_plan(one_context)
    one_model = place_replicated(built.state, one_plan)
    one_optimizer = _optimizer(one_model, built.metadata)
    one_state = initialize_train_state(one_model, one_optimizer.transform, seed=1)
    one_step = make_train_step(built.graph, one_optimizer, sharding=one_plan, state_template=one_state)
    next_one, metrics_one = one_step(one_state, place_batch(batch, one_plan))

    tp_context = build_mesh_context(MeshSpec(axis_names=("data", "tp"), axis_sizes=(2, 2)))
    tp_plan = build_sharding_plan(
        tp_context,
        parallelism=ParallelismSpec(tensor_parallel=True),
        param_layouts=built.param_layouts,
    )
    tp_model = place_model_state(built.state, tp_plan)
    tp_optimizer = _optimizer(tp_model, built.metadata)
    tp_state = initialize_train_state(tp_model, tp_optimizer.transform, seed=1)
    tp_step = make_train_step(built.graph, tp_optimizer, sharding=tp_plan, state_template=tp_state)
    next_tp, metrics_tp = tp_step(tp_state, place_batch(batch, tp_plan))

    assert _scalar_int(metrics_tp.token_count) == _scalar_int(metrics_one.token_count) == 16
    assert np.allclose(
        np.asarray(jax.device_get(metrics_tp.loss_sum)),
        np.asarray(jax.device_get(metrics_one.loss_sum)),
        rtol=1e-5,
        atol=1e-5,
    )
    assert any(getattr(leaf, "sharding", None).spec == jax.sharding.PartitionSpec(None, "tp") for leaf in jax.tree.leaves(next_tp.model))
    assert _trees_close(next_tp.model, next_one.model)


def test_tensor_parallel_trinity_moe_train_step_uses_shared_expert_tp_policy() -> None:
    require_fake_devices()
    built = build_model(
        _tiny_trinity_spec(
            moe={"num_experts": 4, "top_k": 2, "num_shared_experts": 1},
        ),
        seed=0,
    )
    context = build_mesh_context(MeshSpec(axis_names=("data", "tp"), axis_sizes=(2, 2)))
    plan = build_sharding_plan(
        context,
        parallelism=ParallelismSpec(tensor_parallel=True),
        param_layouts=built.param_layouts,
        expert_layouts=built.expert_layouts,
    )
    model_state = place_model_state(built.state, plan)
    optimizer = _optimizer(model_state, built.metadata)
    state = initialize_train_state(model_state, optimizer.transform, seed=1)
    step = make_train_step(built.graph, optimizer, sharding=plan, state_template=state)
    batch = _batch(batch_size=4, seq_len=4, vocab_size=16)

    next_state, metrics = step(state, place_batch(batch, plan))

    assert next_state.step == 1
    assert _scalar_int(metrics.token_count) == 16
    assert metrics.router_expert_counts.shape == (1, 4)
    by_tag = {layout.tag: plan.param_shardings[layout.path] for layout in built.param_layouts}
    assert by_tag["moe_gate"].spec == jax.sharding.PartitionSpec()
    assert by_tag["moe_shared_gate"].spec == jax.sharding.PartitionSpec(None, "tp")
    assert any(getattr(leaf, "sharding", None).spec == jax.sharding.PartitionSpec(None, "tp") for leaf in jax.tree.leaves(next_state.model))
    assert math.isfinite(float(jax.device_get(metrics.loss_sum)))


def test_fsdp_train_step_reports_global_metrics_and_sharded_state() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(hidden_size=16, intermediate_size=32, num_heads=4, n_kv_heads=4), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 4)))
    plan = build_sharding_plan(context, parallelism=ParallelismSpec(mode="fsdp"), param_layouts=built.param_layouts)
    model_state = place_model_state(built.state, plan)
    optimizer = _optimizer(model_state, built.metadata)
    state = initialize_train_state(model_state, optimizer.transform, seed=1)
    step = make_train_step(
        built.graph,
        optimizer,
        sharding=plan,
        state_template=state,
        donate_state=True,
        expected_batch_shape=(1, 4, 4),
    )
    batch = _batch(batch_size=4, seq_len=4, vocab_size=16)

    next_state, metrics = step(state, place_batch(batch, plan))

    assert next_state.step == 1
    assert next_state.tokens_seen == 16
    assert metrics.token_count == 16
    assert metrics.loss_sum.shape == ()
    assert metrics.loss_sum.sharding == plan.metrics
    assert next_state.step.sharding == plan.replicated
    assert jax.tree.leaves(next_state.model)[0].sharding == plan.replicated
    assert any(getattr(leaf, "sharding", None).spec == jax.sharding.PartitionSpec(None, "fsdp") for leaf in jax.tree.leaves(next_state.model))


def test_fsdp_train_step_auto_resolves_muon_to_dion2() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(hidden_size=16, intermediate_size=32, num_heads=4, n_kv_heads=4), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 4)))
    plan = build_sharding_plan(context, parallelism=ParallelismSpec(mode="fsdp"), param_layouts=built.param_layouts)
    model_state = place_model_state(built.state, plan)
    optimizer = _optimizer(model_state, built.metadata, optimizer_name="muon")
    state = initialize_train_state(model_state, optimizer.transform, seed=1)
    step = make_train_step(
        built.graph,
        optimizer,
        sharding=plan,
        state_template=state,
        donate_state=True,
        expected_batch_shape=(1, 4, 4),
    )
    batch = _batch(batch_size=4, seq_len=4, vocab_size=16)

    next_state, metrics = step(state, place_batch(batch, plan))

    assert {assignment.backend for assignment in optimizer.route_assignments} == {"adamw", "dion2"}
    assert next_state.tokens_seen == 16
    assert metrics.token_count == 16
    assert metrics.loss_sum.shape == ()


def test_zero2_train_step_keeps_model_replicated_and_optimizer_sharded() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(hidden_size=16, intermediate_size=32, num_heads=4, n_kv_heads=4), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 4)))
    plan = build_sharding_plan(context, parallelism=ParallelismSpec(mode="zero2"), param_layouts=built.param_layouts)
    model_state = place_model_state(built.state, plan)
    optimizer_init_state = place_optimizer_init_state(built.state, plan)
    optimizer = _optimizer(optimizer_init_state, built.metadata)
    state = initialize_train_state(
        model_state,
        optimizer.transform,
        seed=1,
        optimizer_init_model_state=optimizer_init_state,
    )
    step = make_train_step(
        built.graph,
        optimizer,
        sharding=plan,
        state_template=state,
        donate_state=True,
        expected_batch_shape=(1, 4, 4),
    )
    batch = _batch(batch_size=4, seq_len=4, vocab_size=16)

    next_state, metrics = step(state, place_batch(batch, plan))

    assert next_state.step == 1
    assert next_state.tokens_seen == 16
    assert metrics.token_count == 16
    assert metrics.loss_sum.shape == ()
    assert metrics.loss_sum.sharding == plan.metrics
    assert all(getattr(leaf, "sharding", None) == plan.replicated for leaf in jax.tree.leaves(next_state.model))
    assert any(
        getattr(leaf, "sharding", None).spec == jax.sharding.PartitionSpec(None, "fsdp")
        for leaf in jax.tree.leaves(next_state.opt_state)
        if hasattr(getattr(leaf, "sharding", None), "spec")
    )


def test_zero2_train_step_auto_resolves_muon_to_dion2() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(hidden_size=16, intermediate_size=32, num_heads=4, n_kv_heads=4), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 4)))
    plan = build_sharding_plan(context, parallelism=ParallelismSpec(mode="zero2"), param_layouts=built.param_layouts)
    model_state = place_model_state(built.state, plan)
    optimizer_init_state = place_optimizer_init_state(built.state, plan)
    optimizer = _optimizer(optimizer_init_state, built.metadata, optimizer_name="muon")
    state = initialize_train_state(
        model_state,
        optimizer.transform,
        seed=1,
        optimizer_init_model_state=optimizer_init_state,
    )
    step = make_train_step(
        built.graph,
        optimizer,
        sharding=plan,
        state_template=state,
        donate_state=True,
        expected_batch_shape=(1, 4, 4),
    )
    batch = _batch(batch_size=4, seq_len=4, vocab_size=16)

    next_state, metrics = step(state, place_batch(batch, plan))

    assert {assignment.backend for assignment in optimizer.route_assignments} == {"adamw", "dion2"}
    assert all(getattr(leaf, "sharding", None) == plan.replicated for leaf in jax.tree.leaves(next_state.model))
    assert next_state.tokens_seen == 16
    assert metrics.token_count == 16


def test_fsdp_train_step_matches_ddp_global_batch_loss() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(hidden_size=16, intermediate_size=32, num_heads=4, n_kv_heads=4), seed=0)
    batch = _batch(batch_size=4, seq_len=4, vocab_size=16)

    ddp_context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(1,)))
    ddp_plan = build_sharding_plan(ddp_context, parallelism=ParallelismSpec(mode="ddp"), param_layouts=built.param_layouts)
    ddp_model = place_model_state(built.state, ddp_plan)
    ddp_optimizer = _optimizer(ddp_model, built.metadata)
    ddp_state = initialize_train_state(ddp_model, ddp_optimizer.transform, seed=1)
    ddp_step = make_train_step(built.graph, ddp_optimizer, sharding=ddp_plan, state_template=ddp_state)
    _, ddp_metrics = ddp_step(ddp_state, place_batch(batch, ddp_plan))

    fsdp_context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 4)))
    fsdp_plan = build_sharding_plan(fsdp_context, parallelism=ParallelismSpec(mode="fsdp"), param_layouts=built.param_layouts)
    fsdp_model = place_model_state(built.state, fsdp_plan)
    fsdp_optimizer = _optimizer(fsdp_model, built.metadata)
    fsdp_state = initialize_train_state(fsdp_model, fsdp_optimizer.transform, seed=1)
    fsdp_step = make_train_step(built.graph, fsdp_optimizer, sharding=fsdp_plan, state_template=fsdp_state)
    _, fsdp_metrics = fsdp_step(fsdp_state, place_batch(batch, fsdp_plan))

    assert _scalar_int(fsdp_metrics.token_count) == _scalar_int(ddp_metrics.token_count) == 16
    assert np.allclose(
        np.asarray(jax.device_get(fsdp_metrics.loss_sum)),
        np.asarray(jax.device_get(ddp_metrics.loss_sum)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_zero2_train_step_matches_ddp_global_batch_loss() -> None:
    require_fake_devices()
    built = build_model(_tiny_spec(hidden_size=16, intermediate_size=32, num_heads=4, n_kv_heads=4), seed=0)
    batch = _batch(batch_size=4, seq_len=4, vocab_size=16)

    ddp_context = build_mesh_context(MeshSpec(axis_names=("data",), axis_sizes=(1,)))
    ddp_plan = build_sharding_plan(ddp_context, parallelism=ParallelismSpec(mode="ddp"), param_layouts=built.param_layouts)
    ddp_model = place_model_state(built.state, ddp_plan)
    ddp_optimizer = _optimizer(ddp_model, built.metadata)
    ddp_state = initialize_train_state(ddp_model, ddp_optimizer.transform, seed=1)
    ddp_step = make_train_step(built.graph, ddp_optimizer, sharding=ddp_plan, state_template=ddp_state)
    next_ddp, ddp_metrics = ddp_step(ddp_state, place_batch(batch, ddp_plan))

    zero2_context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 4)))
    zero2_plan = build_sharding_plan(zero2_context, parallelism=ParallelismSpec(mode="zero2"), param_layouts=built.param_layouts)
    zero2_model = place_model_state(built.state, zero2_plan)
    zero2_optimizer_init = place_optimizer_init_state(built.state, zero2_plan)
    zero2_optimizer = _optimizer(zero2_optimizer_init, built.metadata)
    zero2_state = initialize_train_state(
        zero2_model,
        zero2_optimizer.transform,
        seed=1,
        optimizer_init_model_state=zero2_optimizer_init,
    )
    zero2_step = make_train_step(built.graph, zero2_optimizer, sharding=zero2_plan, state_template=zero2_state)
    next_zero2, zero2_metrics = zero2_step(zero2_state, place_batch(batch, zero2_plan))

    assert _scalar_int(zero2_metrics.token_count) == _scalar_int(ddp_metrics.token_count) == 16
    assert np.allclose(
        np.asarray(jax.device_get(zero2_metrics.loss_sum)),
        np.asarray(jax.device_get(ddp_metrics.loss_sum)),
        rtol=1e-5,
        atol=1e-5,
    )
    assert _trees_close(next_zero2.model, next_ddp.model, rtol=2e-5, atol=2e-5)


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


def _tiny_trinity_spec(**overrides) -> ModelSpec:
    trinity_values = {
        "initial_dense_layers": 1,
        "local_window": 4,
        "local_layers_per_global": 1,
        "norm_policy": "depth_scaled_sandwich",
        "moe": None,
    }
    for key in tuple(overrides):
        if key in trinity_values:
            trinity_values[key] = overrides.pop(key)
    values = {
        "name": "trinity",
        "variant": "tiny",
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_layers": 2,
        "num_heads": 2,
        "n_kv_heads": 1,
        "max_seq_len": 4,
        "compute_dtype": "float32",
        "trinity": TrinitySpec(**trinity_values),
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


def _trees_close(left, right, *, rtol: float = 1e-5, atol: float = 1e-5) -> bool:
    return all(
        np.allclose(
            np.asarray(jax.device_get(left_leaf)),
            np.asarray(jax.device_get(right_leaf)),
            rtol=rtol,
            atol=atol,
        )
        for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _assert_optimizer_groups_cover(metrics, metadata) -> None:
    groups = metrics.optimizer_group_specs
    assert groups
    assert sum(group["leaf_count"] for group in groups) == len(metadata)
    assert sum(group["parameter_count"] for group in groups) == sum(item.count for item in metadata)
    group_grad_norms = np.asarray(jax.device_get(metrics.optimizer_group_grad_norms))
    group_update_norms = np.asarray(jax.device_get(metrics.optimizer_group_update_norms))
    group_param_norms = np.asarray(jax.device_get(metrics.optimizer_group_param_norms))
    assert group_grad_norms.shape == group_update_norms.shape == group_param_norms.shape == (len(groups),)
    assert np.sqrt(np.sum(np.square(group_grad_norms))) == pytest.approx(float(jax.device_get(metrics.grad_norm)))
    assert np.sqrt(np.sum(np.square(group_update_norms))) == pytest.approx(float(jax.device_get(metrics.update_norm)))
    assert np.sqrt(np.sum(np.square(group_param_norms))) == pytest.approx(float(jax.device_get(metrics.param_norm)))


def _scalar_int(value) -> int:
    return int(np.asarray(jax.device_get(value)).item())


def _state_value_by_path(state, target_path: tuple[str, ...]):
    for path, variable in nnx.to_flat_state(state):
        if tuple(str(part) for part in path) == target_path:
            return variable.get_value()
    raise AssertionError(f"state path {'.'.join(target_path)} not found")


def _rng_equal(left, right) -> bool:
    return all(
        jnp.array_equal(getattr(left, name), getattr(right, name))
        for name in ("train", "data", "eval", "sample")
    )
