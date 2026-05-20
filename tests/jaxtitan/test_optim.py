import jax
import jax.numpy as jnp
import pytest

from jaxtitan.errors import ContractError
from jaxtitan.models import ParamMetadata, build_model
from jaxtitan.optim import (
    build_lr_schedule,
    build_optimizer,
    describe_optimizer,
    muon_policy_constants,
    zeropower_via_newton_schulz,
)
from jaxtitan.specs.model import ModelSpec
from jaxtitan.specs.optimizer import OptimizerSpec, ParamRouteRule, ScheduleSpec


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

    first = zeropower_via_newton_schulz(tall)
    second = zeropower_via_newton_schulz(tall)
    wide_result = zeropower_via_newton_schulz(wide)

    assert first.shape == tall.shape
    assert wide_result.shape == wide.shape
    assert jnp.all(jnp.isfinite(first))
    assert jnp.all(jnp.isfinite(wide_result))
    assert jnp.all(first == second)


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
    description = describe_optimizer(OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=0.001)))

    assert "muon" in description
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


def _scalar(value) -> float:
    return float(jnp.asarray(value).item())
