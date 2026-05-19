import jax
import jax.numpy as jnp
import pytest

from jaxtitan.errors import ContractError
from jaxtitan.models import ParamMetadata, build_model
from jaxtitan.optim import build_lr_schedule, build_optimizer, describe_optimizer
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
    spec = OptimizerSpec(name="muon", schedule=ScheduleSpec(peak_lr=1.0))

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
