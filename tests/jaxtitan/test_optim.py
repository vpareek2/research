import jax
import jax.numpy as jnp
import pytest

from jaxtitan.errors import ContractError
from jaxtitan.mesh import build_mesh_context, build_sharding_plan, place_model_state, place_optimizer_init_state
from jaxtitan.models import ParamMetadata, build_model
from jaxtitan.optim import (
    build_lr_schedule,
    build_optimizer,
    describe_optimizer,
    muon_policy_constants,
    optimizer_policy_summary,
    zeropower_via_newton_schulz,
)
from jaxtitan.optim.dion2 import dion2_policy_constants, dion2_transform, polar_express, select_dion2_slices
from jaxtitan.specs.mesh import MeshSpec
from jaxtitan.specs.model import ModelSpec, TrinitySpec
from jaxtitan.specs.optimizer import OptimizerSpec, ParamRouteRule, ScheduleSpec
from jaxtitan.specs.parallelism import ParallelismSpec

FAKE_DEVICE_COUNT = 4


def require_fake_devices() -> None:
    if jax.local_device_count() < FAKE_DEVICE_COUNT:
        pytest.skip("JAX was initialized before fake CPU device flags were set")


def test_constant_schedule_supports_warmup() -> None:
    schedule = build_lr_schedule(ScheduleSpec(name="constant", peak_lr=0.1, warmup_steps=2))

    assert _scalar(schedule(0)) == pytest.approx(0.05)
    assert _scalar(schedule(1)) == pytest.approx(0.1)
    assert _scalar(schedule(2)) == pytest.approx(0.1)


def test_cosine_schedule_supports_warmup_and_min_lr() -> None:
    schedule = build_lr_schedule(
        ScheduleSpec(name="cosine", peak_lr=1.0, warmup_steps=1, total_steps=5, min_lr_ratio=0.1)
    )

    assert _scalar(schedule(0)) == pytest.approx(1.0)
    assert _scalar(schedule(1)) == pytest.approx(1.0)
    assert _scalar(schedule(4)) == pytest.approx(0.1)


def test_wsd_schedule_requires_explicit_stable_steps() -> None:
    with pytest.raises(ContractError, match="stable_steps"):
        build_lr_schedule(ScheduleSpec(name="wsd", peak_lr=1.0, total_steps=10))

    schedule = build_lr_schedule(
        ScheduleSpec(name="wsd", peak_lr=1.0, warmup_steps=1, stable_steps=2, total_steps=6, min_lr_ratio=0.1)
    )

    assert _scalar(schedule(0)) == pytest.approx(1.0)
    assert _scalar(schedule(1)) == pytest.approx(1.0)
    assert _scalar(schedule(2)) == pytest.approx(1.0)
    assert _scalar(schedule(5)) == pytest.approx(0.1)


def test_schedule_rejects_missing_or_invalid_total_steps() -> None:
    with pytest.raises(ContractError, match="total_steps"):
        build_lr_schedule(ScheduleSpec(name="cosine", peak_lr=1.0))
    with pytest.raises(ContractError, match="total_steps"):
        ScheduleSpec(name="cosine", peak_lr=1.0, total_steps=0)
    with pytest.raises(ContractError, match="less than total_steps"):
        build_lr_schedule(ScheduleSpec(name="cosine", peak_lr=1.0, warmup_steps=4, total_steps=4))


def test_adamw_build_init_and_update_accept_nnx_model_state() -> None:
    result = build_model(_tiny_spec(), seed=0)
    built = build_optimizer(
        OptimizerSpec(name="adamw", schedule=ScheduleSpec(peak_lr=1e-3), weight_decay=0.1),
        result.state,
        result.metadata,
    )

    opt_state = built.transform.init(result.state)
    grads = jax.tree.map(jnp.ones_like, result.state)
    updates, next_opt_state = built.transform.update(grads, opt_state, params=result.state)

    assert len(built.route_assignments) == len(result.metadata)
    assert {assignment.path for assignment in built.route_assignments} == {item.path for item in result.metadata}
    assert {assignment.backend for assignment in built.route_assignments} == {"adamw"}
    assert len(jax.tree.leaves(updates)) == len(jax.tree.leaves(result.state))
    assert len(jax.tree.leaves(next_opt_state)) == len(jax.tree.leaves(opt_state))
    assert any(jnp.any(leaf != 0) for leaf in jax.tree.leaves(updates))


def test_muon_newton_schulz_is_finite_shape_preserving_and_deterministic() -> None:
    tall = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 10.0
    wide = jnp.arange(15, dtype=jnp.float32).reshape(3, 5) / 10.0
    expert_stack = jnp.stack([tall, tall + 1.0], axis=0)

    first = zeropower_via_newton_schulz(tall)
    second = zeropower_via_newton_schulz(tall)
    wide_result = zeropower_via_newton_schulz(wide)
    expert_result = zeropower_via_newton_schulz(expert_stack)

    assert first.shape == tall.shape
    assert wide_result.shape == wide.shape
    assert expert_result.shape == expert_stack.shape
    assert jnp.all(jnp.isfinite(first))
    assert jnp.all(jnp.isfinite(wide_result))
    assert jnp.all(jnp.isfinite(expert_result))
    assert jnp.all(first == second)
    assert jnp.allclose(expert_result[0], first)


def test_dion2_selects_expected_rows_and_columns() -> None:
    value = jnp.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0, 2.0],
            [0.5, 0.5, 0.5, 0.5],
            [4.0, 1.0, -1.0, 2.0],
        ],
        dtype=jnp.float32,
    )

    rows, row_indices = select_dion2_slices(value, select_axis=0, fraction=0.5)
    cols, col_indices = select_dion2_slices(value, select_axis=1, fraction=0.5)

    assert set(map(int, row_indices.tolist())) == {1, 3}
    assert rows.shape == (2, 4)
    assert set(map(int, col_indices.tolist())) == {0, 3}
    assert cols.shape == (4, 2)


def test_polar_express_is_finite_shape_preserving_and_deterministic() -> None:
    tall = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 10.0
    wide = jnp.arange(15, dtype=jnp.float32).reshape(3, 5) / 10.0

    first = polar_express(tall)
    second = polar_express(tall)
    wide_result = polar_express(wide)

    assert first.shape == tall.shape
    assert wide_result.shape == wide.shape
    assert jnp.all(jnp.isfinite(first))
    assert jnp.all(jnp.isfinite(wide_result))
    assert jnp.all(first == second)


def test_dion2_update_touches_only_selected_rows_without_weight_decay() -> None:
    params = {"w": jnp.ones((8, 4), dtype=jnp.float32)}
    grads = {
        "w": jnp.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0, 2.0],
                [0.5, 0.5, 0.5, 0.5],
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [-1.0, 1.0, -1.0, 1.0],
                [4.0, 2.0, -2.0, 2.0],
            ],
            dtype=jnp.float32,
        )
    }
    transform = dion2_transform(lambda _count: jnp.asarray(1.0, dtype=jnp.float32), weight_decay=0.0, select_axis=0)

    state = transform.init(params)
    updates, next_state = transform.update(grads, state, params=params)

    changed_rows = jnp.any(updates["w"] != 0, axis=1)
    assert changed_rows.tolist() == [False, True, False, False, False, False, False, True]
    assert jnp.allclose(next_state.momentum["w"][1], grads["w"][1] * dion2_policy_constants()["ef_decay"])
    assert jnp.allclose(next_state.momentum["w"][7], grads["w"][7] * dion2_policy_constants()["ef_decay"])
    assert jnp.allclose(next_state.momentum["w"][0], grads["w"][0])
    assert jnp.allclose(next_state.momentum["w"][2], grads["w"][2])


def test_muon_primary_routes_hidden_matrices_to_muon_with_adamw_fallback() -> None:
    result = build_model(_tiny_spec(), seed=0)
    built = build_optimizer(
        OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3), weight_decay=0.1),
        result.state,
        result.metadata,
    )

    assignments = {assignment.path: assignment for assignment in built.route_assignments}
    tags_by_backend = {}
    for assignment in built.route_assignments:
        tags_by_backend.setdefault(assignment.backend, set()).add(assignment.tag)

    assert tags_by_backend["muon"] == {
        "attention_q",
        "attention_k",
        "attention_v",
        "attention_o",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
    }
    assert "embedding" in tags_by_backend["adamw"]
    assert "lm_head" in tags_by_backend["adamw"]
    assert "final_norm" in tags_by_backend["adamw"]
    assert assignments[("embed", "embedding")].fallback_reason == "embedding"
    assert assignments[("lm_head", "kernel")].fallback_reason == "lm_head"
    assert assignments[("norm", "scale")].fallback_reason == "norm"


def test_muon_routes_moe_matrices_and_excludes_expert_bias_weight_decay() -> None:
    result = build_model(
        _tiny_trinity_spec(
            num_layers=2,
            initial_dense_layers=1,
            moe={"num_experts": 3, "top_k": 2, "num_shared_experts": 1},
        ),
        seed=0,
    )
    built = build_optimizer(
        OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3), weight_decay=0.1),
        result.state,
        result.metadata,
    )

    assignments = {assignment.tag: assignment for assignment in built.route_assignments if assignment.tag.startswith("moe_")}
    assert assignments["moe_shared_gate"].backend == "muon"
    assert assignments["moe_shared_up"].backend == "muon"
    assert assignments["moe_shared_down"].backend == "muon"
    assert assignments["moe_gate"].backend == "muon"
    assert assignments["moe_up"].backend == "muon"
    assert assignments["moe_down"].backend == "muon"
    assert assignments["moe_gate"].fallback_reason is None
    assert assignments["moe_up"].fallback_reason is None
    assert assignments["moe_down"].fallback_reason is None
    assert assignments["moe_router"].backend == "adamw"
    assert assignments["moe_expert_bias"].backend == "adamw"
    assert assignments["moe_expert_bias"].weight_decay is False
    assert assignments["moe_expert_bias"].fallback_reason == "expert_bias"


def test_optimizer_policy_summary_records_distributed_safety() -> None:
    result = build_model(_tiny_spec(), seed=0)
    muon = build_optimizer(
        OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3), weight_decay=0.1),
        result.state,
        result.metadata,
    )

    adamw_policy = optimizer_policy_summary(OptimizerSpec(name="adamw", schedule=ScheduleSpec(peak_lr=1e-3)))
    muon_policy = optimizer_policy_summary(
        OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3)),
        muon.route_assignments,
    )

    assert adamw_policy["distributed_policy"] == {
        "optimizer_state": "elementwise_shard_safe",
        "gradient_update": "elementwise_shard_safe",
        "zero2_fsdp": "supported",
    }
    assert adamw_policy["adamw"]["distributed_policy"] == "elementwise_shard_safe"
    assert muon_policy["distributed_policy"] == {
        "optimizer_state": "replicated_or_expert_axis_muon_sharded_dion2_or_dist_muon_exact",
        "gradient_update": "muon_when_complete_matrix_dion2_for_fsdp_dist_muon_exact_for_tp",
        "zero2_fsdp": "auto_dion2",
    }
    assert muon_policy["muon"]["newton_schulz_precision"] == "bfloat16"
    assert muon_policy["muon"]["distributed_policy"] == "replicated_or_auto_dion2_when_sharded"
    assert muon_policy["muon"]["distributed_matrix_update"] == "auto_dion2_or_dist_muon_exact"
    assert muon_policy["muon"]["rank3_expert_policy"] == "per_expert_full_matrix_when_complete_local"
    assert muon_policy["dion2"]["fraction"] == 0.25
    assert muon_policy["auto_routing"]["active"] is False
    route_policies = {route["backend"]: route["distributed_policy"] for route in muon_policy["routes"]}
    assert route_policies == {
        "adamw": "elementwise_shard_safe",
        "muon": "complete_matrix_or_per_expert_complete_matrix",
    }


def test_rank3_moe_expert_muon_updates_selected_expert_matrices() -> None:
    params = {"expert": jnp.ones((2, 3, 4), dtype=jnp.float32)}
    grads = {"expert": jnp.ones((2, 3, 4), dtype=jnp.float32)}
    metadata = (
        ParamMetadata(path=("expert",), shape=(2, 3, 4), dtype="float32", count=24, tag="moe_gate"),
    )
    built = build_optimizer(
        OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3), weight_decay=0.0),
        params,
        metadata,
    )

    opt_state = built.transform.init(params)
    updates, next_opt_state = built.transform.update(grads, opt_state, params=params)

    assert built.route_assignments[0].backend == "muon"
    assert updates["expert"].shape == params["expert"].shape
    assert jnp.any(updates["expert"] != 0)
    momentum_leaf = jax.tree.leaves(next_opt_state)[1]
    assert momentum_leaf.shape == params["expert"].shape


def test_rank3_moe_expert_muon_rejects_matrix_axis_sharding() -> None:
    require_fake_devices()
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 4)))
    sharded = jax.device_put(
        jnp.ones((2, 4, 8), dtype=jnp.float32),
        jax.sharding.NamedSharding(context.mesh, jax.sharding.PartitionSpec(None, "fsdp", None)),
    )
    params = {"expert": sharded}
    metadata = (
        ParamMetadata(path=("expert",), shape=(2, 4, 8), dtype="float32", count=64, tag="moe_gate"),
    )

    with pytest.raises(ContractError, match="complete per-expert matrices"):
        build_optimizer(
            OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3), weight_decay=0.0),
            params,
            metadata,
        )


def test_sharded_muon_routes_auto_resolve_to_dion2() -> None:
    require_fake_devices()
    result = build_model(_tiny_spec(hidden_size=16, intermediate_size=32, num_heads=4, n_kv_heads=4), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp"), axis_sizes=(1, 4)))
    plan = build_sharding_plan(context, parallelism=ParallelismSpec(mode="zero2"), param_layouts=result.param_layouts)
    optimizer_init_state = place_optimizer_init_state(result.state, plan)

    built = build_optimizer(
        OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3), weight_decay=0.1),
        optimizer_init_state,
        result.metadata,
    )
    policy = optimizer_policy_summary(
        OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3)),
        built.route_assignments,
        parallelism_mode="zero2",
        fsdp_axis_size=4,
    )

    assert {assignment.backend for assignment in built.route_assignments} == {"adamw", "dion2"}
    assert policy["route_counts"] == {"adamw": 7, "dion2": 7}
    assert policy["auto_routing"]["active"] is True
    assert policy["auto_routing"]["muon_sharded_matrix_backend"] == "dion2"
    assert policy["auto_routing"]["muon_tp_sharded_matrix_backend"] == "dist_muon_exact"
    dion2_routes = [assignment for assignment in built.route_assignments if assignment.backend == "dion2"]
    assert {assignment.requested_backend for assignment in dion2_routes} == {"muon"}
    assert {assignment.resolution_reason for assignment in dion2_routes} == {"fsdp_sharded_optimizer_state"}
    assert all(assignment.auto_resolved for assignment in dion2_routes)
    assert {assignment.matrix_axis for assignment in dion2_routes} == {0, 1}


def test_tensor_parallel_muon_routes_to_exact_distributed_muon() -> None:
    require_fake_devices()
    result = build_model(_tiny_spec(hidden_size=16, intermediate_size=32, num_heads=4, n_kv_heads=4), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data", "tp"), axis_sizes=(1, 4)))
    plan = build_sharding_plan(
        context,
        parallelism=ParallelismSpec(tensor_parallel=True),
        param_layouts=result.param_layouts,
    )
    model_state = place_model_state(result.state, plan)

    built = build_optimizer(
        OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3), weight_decay=0.1),
        model_state,
        result.metadata,
    )
    policy = optimizer_policy_summary(
        OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3)),
        built.route_assignments,
    )

    assert {assignment.backend for assignment in built.route_assignments} == {"adamw", "dist_muon_exact"}
    exact_routes = [assignment for assignment in built.route_assignments if assignment.backend == "dist_muon_exact"]
    assert {assignment.requested_backend for assignment in exact_routes} == {"muon"}
    assert {assignment.resolution_reason for assignment in exact_routes} == {"tp_sharded_matrix_exact_muon"}
    assert all(assignment.auto_resolved for assignment in exact_routes)
    assert {assignment.matrix_axis for assignment in exact_routes} == {0, 1}
    assert policy["route_counts"] == {"adamw": 7, "dist_muon_exact": 7}
    assert policy["dist_muon_exact"] == {
        "distributed_policy": "reference_logical_matrix_exact",
        "exact": True,
        "correctness_status": "four_h100_acceptance_passed",
        "approximation": "none",
        "performance": "replicate_logical_matrix_reference",
        "newton_schulz_precision": "bfloat16",
        "replicated_model_axis_reduction": "pmean",
        "auto_selected_for": "tp_sharded_muon_matrix_routes",
    }
    assert {assignment.logical_shape for assignment in exact_routes}
    assert {assignment.sharded_model_axes for assignment in exact_routes} == {("tp",)}
    assert {assignment.replicated_model_axes for assignment in exact_routes} == {()}
    assert all(assignment.partition_spec is not None for assignment in exact_routes)


@pytest.mark.parametrize("mode", ["fsdp", "zero2"])
def test_tensor_parallel_muon_takes_precedence_over_fsdp_dion2_route(mode: str) -> None:
    require_fake_devices()
    result = build_model(_tiny_spec(hidden_size=16, intermediate_size=32, num_heads=4, n_kv_heads=4), seed=0)
    context = build_mesh_context(MeshSpec(axis_names=("data", "fsdp", "tp"), axis_sizes=(1, 2, 2)))
    plan = build_sharding_plan(
        context,
        parallelism=ParallelismSpec(mode=mode, tensor_parallel=True),
        param_layouts=result.param_layouts,
    )
    optimizer_state = place_optimizer_init_state(result.state, plan)

    built = build_optimizer(
        OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3), weight_decay=0.1),
        optimizer_state,
        result.metadata,
    )
    policy = optimizer_policy_summary(
        OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3)),
        built.route_assignments,
        parallelism_mode=mode,
        fsdp_axis_size=2,
        tp_axis_size=2,
    )

    assert {assignment.backend for assignment in built.route_assignments} == {"adamw", "dist_muon_exact"}
    assert policy["route_counts"] == {"adamw": 7, "dist_muon_exact": 7}
    assert policy["auto_routing"]["active"] is True
    exact_routes = [assignment for assignment in built.route_assignments if assignment.backend == "dist_muon_exact"]
    assert {assignment.resolution_reason for assignment in exact_routes} == {"tp_sharded_matrix_exact_muon"}
    assert {assignment.matrix_axis for assignment in exact_routes} == {0, 1}
    assert {assignment.sharded_model_axes for assignment in exact_routes}
    assert any(assignment.replicated_model_axes == ("fsdp",) for assignment in exact_routes)


@pytest.mark.parametrize(
    "partition_spec",
    [
        pytest.param(jax.sharding.PartitionSpec("tp", None), id="row_sharded"),
        pytest.param(jax.sharding.PartitionSpec(None, "tp"), id="column_sharded"),
    ],
)
def test_exact_distributed_muon_update_matches_replicated_muon_for_tp_shards(partition_spec) -> None:
    require_fake_devices()
    full_params = {"w": jnp.arange(24, dtype=jnp.float32).reshape(6, 4) / 10.0}
    full_grads = {"w": jnp.arange(24, dtype=jnp.float32).reshape(6, 4) / 20.0}
    metadata = (ParamMetadata(path=("w",), shape=(6, 4), dtype="float32", count=24, tag="attention_q"),)
    spec = OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3), weight_decay=0.0)

    replicated = build_optimizer(spec, full_params, metadata)
    replicated_params = full_params
    replicated_state = replicated.transform.init(replicated_params)

    context = build_mesh_context(MeshSpec(axis_names=("data", "tp"), axis_sizes=(1, 2)))
    sharding = jax.sharding.NamedSharding(context.mesh, partition_spec)
    sharded_params = {"w": jax.device_put(full_params["w"], sharding)}
    sharded_grads = {"w": jax.device_put(full_grads["w"], sharding)}
    distributed = build_optimizer(spec, sharded_params, metadata)
    distributed_params = sharded_params
    distributed_state = distributed.transform.init(distributed_params)

    assert distributed.route_assignments[0].backend == "dist_muon_exact"
    for _ in range(5):
        replicated_updates, replicated_state = replicated.transform.update(
            full_grads,
            replicated_state,
            params=replicated_params,
        )
        distributed_updates, distributed_state = distributed.transform.update(
            sharded_grads,
            distributed_state,
            params=distributed_params,
        )
        assert jnp.allclose(distributed_updates["w"], replicated_updates["w"], rtol=1e-5, atol=1e-5)
        replicated_params = jax.tree.map(lambda param, update: param + update, replicated_params, replicated_updates)
        distributed_params = jax.tree.map(lambda param, update: param + update, distributed_params, distributed_updates)
        assert jnp.allclose(distributed_params["w"], replicated_params["w"], rtol=1e-5, atol=1e-5)


def test_muon_build_init_and_update_accept_nnx_model_state() -> None:
    result = build_model(_tiny_spec(), seed=0)
    built = build_optimizer(
        OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3), weight_decay=0.1),
        result.state,
        result.metadata,
    )

    opt_state = built.transform.init(result.state)
    grads = jax.tree.map(jnp.ones_like, result.state)
    updates, next_opt_state = built.transform.update(grads, opt_state, params=result.state)

    assert {assignment.backend for assignment in built.route_assignments} == {"adamw", "muon"}
    assert len(jax.tree.leaves(updates)) == len(jax.tree.leaves(result.state))
    assert len(jax.tree.leaves(next_opt_state)) == len(jax.tree.leaves(opt_state))
    assert any(jnp.any(leaf != 0) for leaf in jax.tree.leaves(updates))


def test_muon_adamw_fallback_can_use_separate_schedule() -> None:
    params = {
        "matrix": jnp.ones((2, 3), dtype=jnp.float32),
        "head": jnp.ones((3,), dtype=jnp.float32),
    }
    grads = jax.tree.map(jnp.zeros_like, params)
    metadata = (
        ParamMetadata(path=("matrix",), shape=(2, 3), dtype="float32", count=6, tag="attention_q"),
        ParamMetadata(path=("head",), shape=(3,), dtype="float32", count=3, tag="lm_head"),
    )
    built = build_optimizer(
        OptimizerSpec(
            name="muon",
            schedule=ScheduleSpec(peak_lr=0.02),
            adamw_fallback_schedule=ScheduleSpec(peak_lr=0.001),
            weight_decay=0.1,
        ),
        params,
        metadata,
    )

    opt_state = built.transform.init(params)
    updates, _next_opt_state = built.transform.update(grads, opt_state, params=params)

    assert built.adamw_fallback_schedule is not None
    assert updates["head"] == pytest.approx(jnp.full((3,), -0.0001))


def test_muon_momentum_matches_reference_convention() -> None:
    params = {"w": jnp.ones((2, 3), dtype=jnp.float32)}
    grads = {"w": jnp.full((2, 3), 2.0, dtype=jnp.float32)}
    metadata = (ParamMetadata(path=("w",), shape=(2, 3), dtype="float32", count=6, tag="attention_q"),)
    built = build_optimizer(
        OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1e-3)),
        params,
        metadata,
    )

    opt_state = built.transform.init(params)
    _updates, next_opt_state = built.transform.update(grads, opt_state, params=params)
    momentum_leaf = jax.tree.leaves(next_opt_state)[1]

    expected = (1.0 - muon_policy_constants()["momentum"]) * grads["w"]
    assert jnp.allclose(momentum_leaf, expected)


def test_muon_rejects_eligible_tags_that_are_not_matrices() -> None:
    params = {"w": jnp.asarray([1.0, 2.0])}
    metadata = (ParamMetadata(path=("w",), shape=(2,), dtype="float32", count=2, tag="attention_q"),)

    with pytest.raises(ContractError, match="Muon route"):
        build_optimizer(
            OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1.0)),
            params,
            metadata,
        )


def test_grad_clip_norm_changes_adamw_update_deterministically() -> None:
    params = {"w": jnp.asarray([1.0, 1.0])}
    metadata = (ParamMetadata(path=("w",), shape=(2,), dtype="float32", count=2, tag="weight"),)
    unclipped = build_optimizer(
        OptimizerSpec(name="adamw", schedule=ScheduleSpec(peak_lr=1.0), weight_decay=0.0),
        params,
        metadata,
    ).transform
    clipped = build_optimizer(
        OptimizerSpec(name="adamw", schedule=ScheduleSpec(peak_lr=1.0), weight_decay=0.0, grad_clip_norm=1.0),
        params,
        metadata,
    ).transform
    first_grads = {"w": jnp.asarray([300.0, 400.0])}
    second_grads = {"w": jnp.asarray([1.0, 0.0])}

    unclipped_state = unclipped.init(params)
    _, unclipped_state = unclipped.update(first_grads, unclipped_state, params=params)
    unclipped_updates, _ = unclipped.update(second_grads, unclipped_state, params=params)

    clipped_state = clipped.init(params)
    _, clipped_state = clipped.update(first_grads, clipped_state, params=params)
    clipped_updates, _ = clipped.update(second_grads, clipped_state, params=params)

    unclipped_norm = jnp.linalg.norm(unclipped_updates["w"])
    clipped_norm = jnp.linalg.norm(clipped_updates["w"])
    assert clipped_norm != pytest.approx(unclipped_norm)


def test_route_rules_can_disable_weight_decay_for_a_tag() -> None:
    params = {"w": jnp.asarray([2.0])}
    metadata = (ParamMetadata(path=("w",), shape=(1,), dtype="float32", count=1, tag="no_decay"),)
    spec = OptimizerSpec(
        name="adamw",
        schedule=ScheduleSpec(peak_lr=1.0),
        weight_decay=0.1,
        route_rules=(ParamRouteRule(tag="no_decay", transform="adamw", weight_decay=False),),
    )
    built = build_optimizer(spec, params, metadata)

    opt_state = built.transform.init(params)
    updates, _ = built.transform.update({"w": jnp.asarray([0.0])}, opt_state, params=params)

    assert built.route_assignments[0].weight_decay is False
    assert updates["w"] == pytest.approx(jnp.asarray([0.0]))


def test_route_rules_reject_unknown_tags_and_backends() -> None:
    params = {"w": jnp.asarray([1.0])}
    metadata = (ParamMetadata(path=("w",), shape=(1,), dtype="float32", count=1, tag="weight"),)

    with pytest.raises(ContractError, match="does not match"):
        build_optimizer(
            OptimizerSpec(
                name="adamw",
                schedule=ScheduleSpec(peak_lr=1.0),
                route_rules=(ParamRouteRule(tag="missing", transform="adamw"),),
            ),
            params,
            metadata,
        )
    with pytest.raises(ContractError, match="runtime adapter"):
        build_optimizer(
            OptimizerSpec(
                name="adamw",
                schedule=ScheduleSpec(peak_lr=1.0),
                route_rules=(ParamRouteRule(tag="weight", transform="soap"),),
            ),
            params,
            metadata,
        )


def test_build_optimizer_rejects_stale_metadata_paths() -> None:
    params = {"w": jnp.asarray([1.0])}
    metadata = (ParamMetadata(path=("missing",), shape=(1,), dtype="float32", count=1, tag="weight"),)

    with pytest.raises(ContractError, match="missing model parameter path"):
        build_optimizer(
            OptimizerSpec(name="adamw", schedule=ScheduleSpec(peak_lr=1.0)),
            params,
            metadata,
        )


def test_unsupported_optimizer_names_fail_at_runtime_build() -> None:
    params = {"w": jnp.asarray([1.0])}
    metadata = (ParamMetadata(path=("w",), shape=(1,), dtype="float32", count=1, tag="weight"),)
    spec = OptimizerSpec(name="soap", schedule=ScheduleSpec(peak_lr=1.0))

    with pytest.raises(ContractError, match="runtime adapter"):
        build_optimizer(spec, params, metadata)


def test_describe_optimizer_includes_backend_schedule_and_adamw_defaults() -> None:
    description = describe_optimizer(
        OptimizerSpec(
            name="adamw",
            schedule=ScheduleSpec(name="cosine", peak_lr=0.001, warmup_steps=2, total_steps=10),
            weight_decay=0.1,
            grad_clip_norm=1.0,
        )
    )

    assert "adamw" in description
    assert "schedule=cosine" in description
    assert "peak_lr=0.001" in description
    assert "weight_decay=0.1" in description
    assert "grad_clip_norm=1" in description
    assert "adamw_b1=0.9" in description
    assert "adamw_eps=1e-08" in description


def test_describe_optimizer_includes_muon_policy_constants() -> None:
    description = describe_optimizer(
        OptimizerSpec(
            name="muon",
            schedule=ScheduleSpec(peak_lr=0.02),
            adamw_fallback_schedule=ScheduleSpec(peak_lr=0.001),
        )
    )

    assert "muon" in description
    assert "peak_lr=0.02" in description
    assert "adamw_fallback_schedule=constant:peak_lr=0.001" in description
    assert "muon_momentum=0.95" in description
    assert "muon_ns_steps=5" in description
    assert "muon_scale_mode=match_rms_adamw" in description
    assert "adamw_fallback=true" in description


def _tiny_spec(**overrides) -> ModelSpec:
    values = {
        "name": "decoder",
        "variant": "tiny",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_layers": 1,
        "num_heads": 4,
        "n_kv_heads": 2,
        "max_seq_len": 8,
        "param_dtype": "float32",
        "compute_dtype": "bfloat16",
    }
    values.update(overrides)
    return ModelSpec(**values)


def _tiny_trinity_spec(**overrides) -> ModelSpec:
    trinity_values = {
        "initial_dense_layers": 1,
        "local_window": 8,
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
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_layers": 2,
        "num_heads": 4,
        "n_kv_heads": 2,
        "max_seq_len": 8,
        "param_dtype": "float32",
        "compute_dtype": "bfloat16",
        "trinity": TrinitySpec(**trinity_values),
    }
    values.update(overrides)
    return ModelSpec(**values)


def _scalar(value) -> float:
    return float(jnp.asarray(value).item())
