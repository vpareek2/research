"""Optimizer build boundary.

Jaxtitan keeps the runtime contract smaller than any one optimizer library:
compiled steps need an object with ``init`` and ``update`` plus opaque state.
Optax is the first backend adapter, not the public training abstraction.
"""

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
import math
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import optax

from jaxtitan.errors import ContractError
from jaxtitan.models import ParamMetadata
from jaxtitan.optim.dion2 import dion2_policy_constants, dion2_transform
from jaxtitan.optim.muon import (
    MuonLeafExecutionPlan,
    distributed_muon_transform,
    muon_policy_constants,
    muon_transform,
)
from jaxtitan.optim.muon_policy import (
    muon_shape_policy_constants,
    select_muon_execution,
)
from jaxtitan.specs.optimizer import OptimizerSpec, ScheduleSpec

PyTree = Any
ADAMW_B1 = 0.9
ADAMW_B2 = 0.999
ADAMW_EPS = 1e-8
_SUPPORTED_RUNTIME_BACKENDS = {"adamw", "muon"}
_INTERNAL_RUNTIME_BACKENDS = {"dion2", "dist_muon"}
_MUON_GRAM_BUCKET_MAX_BYTES = 32 * 1024 * 1024
# Next optimizer backends to evaluate after Muon has run artifacts: Aurora, Scion, and SOAP.
_MUON_MATRIX_TAGS = frozenset(
    {
        "attention_q",
        "attention_k",
        "attention_v",
        "attention_o",
        "attention_gate",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
        "moe_shared_gate",
        "moe_shared_up",
        "moe_shared_down",
    }
)
_MUON_EXPERT_TAGS = frozenset({"moe_gate", "moe_up", "moe_down"})
_MUON_TAGS = _MUON_MATRIX_TAGS | _MUON_EXPERT_TAGS
_NORM_TAGS = frozenset(
    {
        "attention_q_norm",
        "attention_k_norm",
        "attention_pre_norm",
        "attention_post_norm",
        "block_pre_norm",
        "block_post_norm",
        "ffn_pre_norm",
        "ffn_post_norm",
        "final_norm",
    }
)


class OptimizerTransform(Protocol):
    """Tiny optimizer protocol consumed by future compiled train steps."""

    def init(self, params: PyTree) -> PyTree:
        """Initialize opaque optimizer state for a parameter PyTree."""

    def update(self, grads: PyTree, state: PyTree, *, params: PyTree | None = None) -> tuple[PyTree, PyTree]:
        """Transform gradients into parameter updates and next optimizer state."""


@dataclass(frozen=True, slots=True)
class RouteAssignment:
    """Resolved optimizer route for one parameter leaf."""

    path: tuple[str, ...]
    tag: str
    backend: str
    weight_decay: bool
    fallback_reason: str | None = None
    requested_backend: str | None = None
    auto_resolved: bool = False
    resolution_reason: str | None = None
    matrix_axis: int | None = None
    logical_shape: tuple[int, ...] = ()
    sharded_model_axes: tuple[str, ...] = ()
    replicated_model_axes: tuple[str, ...] = ()
    partition_spec: str | None = None


@dataclass(frozen=True, slots=True)
class OptimizerBuildResult:
    """Built optimizer runtime plus reproducibility metadata."""

    transform: OptimizerTransform
    schedule: Callable[[Any], jax.Array]
    adamw_fallback_schedule: Callable[[Any], jax.Array] | None
    route_assignments: tuple[RouteAssignment, ...]
    description: str
    muon_execution_plans: tuple[MuonLeafExecutionPlan, ...] = ()


def build_lr_schedule(spec: ScheduleSpec) -> Callable[[Any], jax.Array]:
    """Build a JAX learning-rate schedule from a static schedule spec."""

    if spec.name == "constant":
        return _constant_schedule(peak_lr=spec.peak_lr, warmup_steps=spec.warmup_steps)
    if spec.name == "cosine":
        total_steps = _required_total_steps(spec)
        _require_decay_steps(spec.name, total_steps=total_steps, warmup_steps=spec.warmup_steps, stable_steps=0)
        return _cosine_schedule(
            peak_lr=spec.peak_lr,
            min_lr=spec.peak_lr * spec.min_lr_ratio,
            total_steps=total_steps,
            warmup_steps=spec.warmup_steps,
        )
    if spec.name == "wsd":
        total_steps = _required_total_steps(spec)
        stable_steps = _required_stable_steps(spec)
        _require_decay_steps(
            spec.name,
            total_steps=total_steps,
            warmup_steps=spec.warmup_steps,
            stable_steps=stable_steps,
        )
        return _wsd_schedule(
            peak_lr=spec.peak_lr,
            min_lr=spec.peak_lr * spec.min_lr_ratio,
            total_steps=total_steps,
            warmup_steps=spec.warmup_steps,
            stable_steps=stable_steps,
        )
    raise ContractError(f"unsupported optimizer.schedule.name {spec.name!r}")


def build_optimizer(
    spec: OptimizerSpec,
    model_state: PyTree,
    metadata: Iterable[ParamMetadata],
    *,
    runtime_parameter_state: PyTree | None = None,
    gradient_shardings: PyTree | None = None,
) -> OptimizerBuildResult:
    """Build the first Jaxtitan optimizer runtime boundary."""

    if spec.name not in _SUPPORTED_RUNTIME_BACKENDS:
        raise ContractError(
            f"optimizer.name {spec.name!r} is valid config but has no Jaxtitan runtime adapter yet; "
            f"supported runtime backends: {sorted(_SUPPORTED_RUNTIME_BACKENDS)}"
        )

    metadata = tuple(metadata)
    runtime_parameter_state = model_state if runtime_parameter_state is None else runtime_parameter_state
    if jax.tree.structure(runtime_parameter_state) != jax.tree.structure(model_state):
        raise ContractError("runtime parameter state must match optimizer-init model state structure")
    if gradient_shardings is not None and jax.tree.structure(gradient_shardings) != jax.tree.structure(model_state):
        raise ContractError("gradient sharding tree must match optimizer-init model state structure")
    params_by_path = _params_by_metadata_path(model_state)
    assignments = _route_assignments(spec, metadata, params_by_path)
    muon_execution_plans = _build_muon_execution_plans(
        assignments,
        optimizer_init_state=model_state,
        runtime_parameter_state=runtime_parameter_state,
        gradient_shardings=gradient_shardings,
        requested_mode=spec.muon_tp_mode,
    )
    schedule = build_lr_schedule(spec.schedule)
    adamw_fallback_schedule = None
    if spec.adamw_fallback_schedule is not None:
        adamw_fallback_schedule = build_lr_schedule(spec.adamw_fallback_schedule)
    _validate_assignment_paths(model_state, assignments)

    transforms = []
    if spec.grad_clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(spec.grad_clip_norm))

    if spec.name == "adamw":
        transforms.append(_adamw_transform(schedule, spec, assignments, model_state))
    elif spec.name == "muon":
        transforms.extend(
            _muon_primary_transforms(
                schedule,
                adamw_fallback_schedule if adamw_fallback_schedule is not None else schedule,
                spec,
                assignments,
                model_state,
                muon_execution_plans,
            )
        )
    else:
        raise ContractError(
            f"optimizer.name {spec.name!r} is valid config but has no Jaxtitan runtime adapter yet; "
            f"supported runtime backends: {sorted(_SUPPORTED_RUNTIME_BACKENDS)}"
        )

    transform = optax.chain(*transforms)
    return OptimizerBuildResult(
        transform=transform,
        schedule=schedule,
        adamw_fallback_schedule=adamw_fallback_schedule,
        route_assignments=assignments,
        muon_execution_plans=muon_execution_plans,
        description=describe_optimizer(spec),
    )


def describe_optimizer(spec: OptimizerSpec) -> str:
    """Return a stable summary for local run manifests and logs."""

    clip = "none" if spec.grad_clip_norm is None else f"{spec.grad_clip_norm:g}"
    total = "none" if spec.schedule.total_steps is None else str(spec.schedule.total_steps)
    stable = "none" if spec.schedule.stable_steps is None else str(spec.schedule.stable_steps)
    fallback_schedule = _schedule_description(spec.adamw_fallback_schedule)
    return (
        f"{spec.name} schedule={spec.schedule.name} peak_lr={spec.schedule.peak_lr:g} "
        f"warmup_steps={spec.schedule.warmup_steps} total_steps={total} "
        f"min_lr_ratio={spec.schedule.min_lr_ratio:g} stable_steps={stable} "
        f"adamw_fallback_schedule={fallback_schedule} "
        f"weight_decay={spec.weight_decay:g} grad_clip_norm={clip} "
        f"adamw_b1={ADAMW_B1:g} adamw_b2={ADAMW_B2:g} adamw_eps={ADAMW_EPS:g}"
        f"{_muon_description_suffix(spec)}"
    )


def optimizer_policy_summary(
    spec: OptimizerSpec,
    assignments: Iterable[RouteAssignment] | None = None,
    *,
    execution_plans: Iterable[MuonLeafExecutionPlan] | None = None,
    parallelism_mode: str | None = None,
    fsdp_axis_size: int | None = None,
    tp_axis_size: int | None = None,
) -> dict[str, Any]:
    """Return a stable optimizer policy payload for artifacts and compatibility checks."""

    assignments = None if assignments is None else tuple(assignments)
    auto_routing_active = _auto_routing_active(
        spec,
        assignments=assignments,
        parallelism_mode=parallelism_mode,
        fsdp_axis_size=fsdp_axis_size,
        tp_axis_size=tp_axis_size,
    )
    payload = {
        "name": spec.name,
        "requested_name": spec.name,
        "schedule": _schedule_payload(spec.schedule),
        "adamw_fallback_schedule": None
        if spec.adamw_fallback_schedule is None
        else _schedule_payload(spec.adamw_fallback_schedule),
        "supported_runtime_backends": sorted(_SUPPORTED_RUNTIME_BACKENDS),
        "internal_runtime_backends": sorted(_INTERNAL_RUNTIME_BACKENDS),
        "distributed_policy": _distributed_policy_payload(spec.name),
        "adamw": {
            "b1": ADAMW_B1,
            "b2": ADAMW_B2,
            "eps": ADAMW_EPS,
            "distributed_policy": "elementwise_shard_safe",
        },
        "muon": {
            **muon_policy_constants(),
            "tp_mode": spec.muon_tp_mode,
            "distributed_policy": "replicated_or_auto_dion2_when_sharded",
            "distributed_matrix_update": "auto_dion2_or_dist_muon",
            "rank3_expert_policy": "per_expert_full_matrix_when_complete_local",
            "rank3_split_matrix_policy": "unsupported_until_explicit_distributed_expert_matrix_optimizer",
            "hidden_matrix_tags": sorted(_MUON_MATRIX_TAGS),
            "routed_expert_matrix_tags": sorted(_MUON_EXPERT_TAGS),
            "fallback_tags": ["embedding", "lm_head", *sorted(_NORM_TAGS)],
        },
        "dion2": {
            **dion2_policy_constants(),
            "distributed_policy": "fsdp_sharded_matrix",
            "auto_selected_for": "sharded_muon_matrix_routes",
        },
        "dist_muon": _dist_muon_policy_payload(spec),
        "auto_routing": {
            "muon_sharded_matrix_backend": "dion2",
            "muon_tp_sharded_matrix_backend": "dist_muon",
            "active": auto_routing_active,
        },
        "route_counts": None,
        "fallback_counts": None,
        "routes": None,
    }
    if assignments is not None:
        route_counts: dict[str, int] = {}
        fallback_counts: dict[str, int] = {}
        routes = []
        for assignment in assignments:
            route_counts[assignment.backend] = route_counts.get(assignment.backend, 0) + 1
            if assignment.fallback_reason is not None:
                fallback_counts[assignment.fallback_reason] = fallback_counts.get(assignment.fallback_reason, 0) + 1
            routes.append(
                {
                    "path": list(assignment.path),
                    "tag": assignment.tag,
                    "backend": assignment.backend,
                    "requested_backend": assignment.requested_backend,
                    "weight_decay": assignment.weight_decay,
                    "fallback_reason": assignment.fallback_reason,
                    "distributed_policy": _route_distributed_policy(
                        assignment.backend,
                        muon_tp_mode=spec.muon_tp_mode,
                    ),
                    "auto_resolved": assignment.auto_resolved,
                    "resolution_reason": assignment.resolution_reason,
                    "matrix_axis": assignment.matrix_axis,
                    "logical_shape": list(assignment.logical_shape),
                    "sharded_model_axes": list(assignment.sharded_model_axes),
                    "replicated_model_axes": list(assignment.replicated_model_axes),
                    "partition_spec": assignment.partition_spec,
                }
            )
        payload["route_counts"] = dict(sorted(route_counts.items()))
        payload["fallback_counts"] = dict(sorted(fallback_counts.items()))
        payload["routes"] = routes
    if execution_plans is not None:
        payload["dist_muon"]["leaf_execution_plans"] = [
            _execution_plan_payload(plan)
            for plan in execution_plans
        ]
    return payload


def _dist_muon_policy_payload(spec: OptimizerSpec) -> dict[str, Any]:
    common = {
        "requested_mode": spec.muon_tp_mode,
        "newton_schulz_precision": muon_policy_constants()["newton_schulz_precision"],
        "replicated_model_axis_reduction": "pmean",
        "auto_selected_for": "tp_sharded_muon_matrix_routes",
    }
    if spec.muon_tp_mode == "duplicated":
        return {
            **common,
            "distributed_policy": "duplicated_full_logical_matrix",
            "exact": True,
            "correctness_status": "four_h100_acceptance_passed",
            "approximation": "none",
            "execution": "duplicated",
            "performance": "single_bf16_logical_matrix_gather",
            "numerical_contract": "accepted_bfloat16_reference",
        }
    return {
        **common,
        "distributed_policy": "hybrid_shape_topology_sharded_gram_collectives",
        "shape_topology_policy": muon_shape_policy_constants(),
        "exact": False,
        "correctness_status": "local_fake_device_acceptance_passed",
        "approximation": "floating_point_reduction_order",
        "execution": "host_static_per_cohort",
        "performance": "duplicated_or_one_norm_plus_five_gram_reductions_with_optional_exchange",
        "numerical_contract": "deterministic_close_not_bitwise",
    }


def _execution_plan_payload(plan: MuonLeafExecutionPlan) -> dict[str, Any]:
    def role_payload(
        sharding: jax.sharding.NamedSharding,
        replica_axes: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "partition_spec": repr(sharding.spec),
            "replica_axes": list(replica_axes),
        }

    return {
        "path": list(plan.path),
        "logical_shape": list(plan.logical_shape),
        "tp_partition_dim": plan.tp_partition_dim,
        "transpose_for_shape": plan.transpose_for_shape,
        "canonical_tp_dim": plan.canonical_tp_dim,
        "requested_mode": plan.requested_mode,
        "execution": plan.execution,
        "fallback_reason": plan.fallback_reason,
        "bucket_id": plan.bucket_id,
        "weight_decay": plan.weight_decay,
        "selection": {
            "policy_version": plan.policy_version,
            "eligible_executions": list(plan.eligible_executions),
            "selected_execution": plan.execution,
            "selection_reason": plan.selection_reason,
            "short_dimension": plan.short_dimension,
            "long_dimension": plan.long_dimension,
            "tp_size": plan.tp_size,
            "cohort_size": plan.cohort_size,
            "gram_side": plan.gram_side,
            "modeled_costs": dict(plan.modeled_costs),
        },
        "roles": {
            "parameter": role_payload(plan.parameter_sharding, plan.parameter_replica_axes),
            "gradient": role_payload(plan.gradient_sharding, plan.gradient_replica_axes),
            "momentum": role_payload(plan.momentum_sharding, plan.momentum_replica_axes),
            "update": role_payload(plan.update_sharding, plan.update_replica_axes),
        },
    }


def _auto_routing_active(
    spec: OptimizerSpec,
    *,
    assignments: tuple[RouteAssignment, ...] | None,
    parallelism_mode: str | None,
    fsdp_axis_size: int | None,
    tp_axis_size: int | None,
) -> bool:
    if assignments is not None:
        return any(assignment.auto_resolved for assignment in assignments)
    fsdp_active = bool(
        spec.name == "muon"
        and parallelism_mode in {"zero2", "fsdp"}
        and fsdp_axis_size is not None
        and fsdp_axis_size > 1
    )
    tp_active = bool(spec.name == "muon" and tp_axis_size is not None and tp_axis_size > 1)
    return fsdp_active or tp_active


def _route_assignments(
    spec: OptimizerSpec,
    metadata: tuple[ParamMetadata, ...],
    params_by_path: dict[tuple[str, ...], Any],
) -> tuple[RouteAssignment, ...]:
    known_tags = {item.tag for item in metadata}
    known_paths = set()
    for item in metadata:
        if item.path in known_paths:
            raise ContractError(f"optimizer metadata contains duplicate parameter path {'.'.join(item.path)!r}")
        known_paths.add(item.path)
    routes_by_tag = {}
    for rule in spec.route_rules:
        if rule.tag not in known_tags:
            raise ContractError(f"optimizer route tag {rule.tag!r} does not match any model parameter metadata tag")
        if rule.tag in routes_by_tag:
            raise ContractError(f"optimizer route tag {rule.tag!r} is configured more than once")
        if rule.transform not in _SUPPORTED_RUNTIME_BACKENDS:
            raise ContractError(
                f"optimizer route transform {rule.transform!r} has no Jaxtitan runtime adapter yet; "
                f"supported runtime backends: {sorted(_SUPPORTED_RUNTIME_BACKENDS)}"
            )
        routes_by_tag[rule.tag] = rule

    assignments = []
    for item in metadata:
        rule = routes_by_tag.get(item.tag)
        requested_backend = _default_backend(spec, item) if rule is None else rule.transform
        leaf = params_by_path.get(item.path)
        if leaf is None:
            raise ContractError(f"optimizer metadata is missing model parameter path {'.'.join(item.path)!r}")
        backend, matrix_axis, resolution_reason = _resolve_backend(requested_backend, item, leaf)
        weight_decay = _default_weight_decay(item) if rule is None else bool(rule.weight_decay and item.tag != "moe_expert_bias")
        _validate_route(item, backend)
        assignments.append(
            RouteAssignment(
                path=item.path,
                tag=item.tag,
                backend=backend,
                weight_decay=weight_decay,
                fallback_reason=_fallback_reason(spec, item, backend, rule is not None),
                requested_backend=requested_backend,
                auto_resolved=backend != requested_backend,
                resolution_reason=resolution_reason,
                matrix_axis=matrix_axis,
                logical_shape=tuple(item.shape) if backend == "dist_muon" else (),
                sharded_model_axes=_route_model_parallel_topology(leaf)[0]
                if backend == "dist_muon"
                else (),
                replicated_model_axes=_route_model_parallel_topology(leaf)[1]
                if backend == "dist_muon"
                else (),
                partition_spec=_partition_spec_text(leaf) if backend == "dist_muon" else None,
            )
        )
    return tuple(assignments)


def _distributed_policy_payload(name: str) -> dict[str, str]:
    if name == "adamw":
        return {
            "optimizer_state": "elementwise_shard_safe",
            "gradient_update": "elementwise_shard_safe",
            "zero2_fsdp": "supported",
        }
    if name == "muon":
        return {
            "optimizer_state": "replicated_or_expert_axis_muon_sharded_dion2_or_dist_muon",
            "gradient_update": "muon_when_complete_matrix_dion2_for_fsdp_dist_muon_for_tp",
            "zero2_fsdp": "auto_dion2",
        }
    return {
        "optimizer_state": "unknown",
        "gradient_update": "unknown",
        "zero2_fsdp": "unsupported",
    }


def _route_distributed_policy(backend: str, *, muon_tp_mode: str) -> str:
    if backend == "adamw":
        return "elementwise_shard_safe"
    if backend == "muon":
        return "complete_matrix_or_per_expert_complete_matrix"
    if backend == "dion2":
        return "fsdp_sharded_matrix"
    if backend == "dist_muon":
        return (
            "duplicated_full_logical_matrix"
            if muon_tp_mode == "duplicated"
            else "hybrid_shape_topology_sharded_gram_collectives"
        )
    return "unknown"


def _adamw_transform(
    schedule: Callable[[Any], jax.Array],
    spec: OptimizerSpec,
    assignments: tuple[RouteAssignment, ...],
    params: PyTree,
) -> optax.GradientTransformationExtraArgs:
    return optax.adamw(
        learning_rate=schedule,
        b1=ADAMW_B1,
        b2=ADAMW_B2,
        eps=ADAMW_EPS,
        weight_decay=spec.weight_decay,
        mask=_mask_from_assignments(params, assignments, lambda assignment: assignment.weight_decay),
    )


def _muon_primary_transforms(
    schedule: Callable[[Any], jax.Array],
    adamw_fallback_schedule: Callable[[Any], jax.Array],
    spec: OptimizerSpec,
    assignments: tuple[RouteAssignment, ...],
    params: PyTree,
    execution_plans: tuple[MuonLeafExecutionPlan, ...],
) -> list[optax.GradientTransformationExtraArgs]:
    transforms = []
    muon_decay_mask = _mask_from_assignments(
        params,
        assignments,
        lambda assignment: assignment.backend == "muon" and assignment.weight_decay,
    )
    muon_no_decay_mask = _mask_from_assignments(
        params,
        assignments,
        lambda assignment: assignment.backend == "muon" and not assignment.weight_decay,
    )
    dion2_row_decay_mask = _mask_from_assignments(
        params,
        assignments,
        lambda assignment: assignment.backend == "dion2" and assignment.matrix_axis == 0 and assignment.weight_decay,
    )
    dion2_row_no_decay_mask = _mask_from_assignments(
        params,
        assignments,
        lambda assignment: assignment.backend == "dion2" and assignment.matrix_axis == 0 and not assignment.weight_decay,
    )
    dion2_col_decay_mask = _mask_from_assignments(
        params,
        assignments,
        lambda assignment: assignment.backend == "dion2" and assignment.matrix_axis == 1 and assignment.weight_decay,
    )
    dion2_col_no_decay_mask = _mask_from_assignments(
        params,
        assignments,
        lambda assignment: assignment.backend == "dion2" and assignment.matrix_axis == 1 and not assignment.weight_decay,
    )
    dist_muon_mask = _mask_from_assignments(
        params,
        assignments,
        lambda assignment: assignment.backend == "dist_muon",
    )
    adamw_mask = _mask_from_assignments(params, assignments, lambda assignment: assignment.backend == "adamw")
    adamw_decay_mask = _mask_from_assignments(
        params,
        assignments,
        lambda assignment: assignment.backend == "adamw" and assignment.weight_decay,
    )
    if _mask_any(muon_decay_mask):
        transforms.append(
            optax.masked(
                _replica_aware_muon_transform(
                    schedule,
                    weight_decay=spec.weight_decay,
                    params=params,
                    mask=muon_decay_mask,
                ),
                muon_decay_mask,
                mask_compatible_extra_args=True,
            )
        )
    if _mask_any(muon_no_decay_mask):
        transforms.append(
            optax.masked(
                _replica_aware_muon_transform(
                    schedule,
                    weight_decay=0.0,
                    params=params,
                    mask=muon_no_decay_mask,
                ),
                muon_no_decay_mask,
                mask_compatible_extra_args=True,
            )
        )
    if _mask_any(dion2_row_decay_mask):
        transforms.append(
            optax.masked(
                dion2_transform(schedule, weight_decay=spec.weight_decay, select_axis=0),
                dion2_row_decay_mask,
                mask_compatible_extra_args=True,
            )
        )
    if _mask_any(dion2_row_no_decay_mask):
        transforms.append(
            optax.masked(
                dion2_transform(schedule, weight_decay=0.0, select_axis=0),
                dion2_row_no_decay_mask,
                mask_compatible_extra_args=True,
            )
        )
    if _mask_any(dion2_col_decay_mask):
        transforms.append(
            optax.masked(
                dion2_transform(schedule, weight_decay=spec.weight_decay, select_axis=1),
                dion2_col_decay_mask,
                mask_compatible_extra_args=True,
            )
        )
    if _mask_any(dion2_col_no_decay_mask):
        transforms.append(
            optax.masked(
                dion2_transform(schedule, weight_decay=0.0, select_axis=1),
                dion2_col_no_decay_mask,
                mask_compatible_extra_args=True,
            )
        )
    if _mask_any(dist_muon_mask):
        transforms.append(
            optax.masked(
                distributed_muon_transform(
                    schedule,
                    weight_decay=spec.weight_decay,
                    execution_plans=_masked_execution_plans(
                        params,
                        dist_muon_mask,
                        execution_plans,
                    ),
                ),
                dist_muon_mask,
                mask_compatible_extra_args=True,
            )
        )
    if _mask_any(adamw_mask):
        transforms.append(
            optax.masked(
                optax.adamw(
                    learning_rate=adamw_fallback_schedule,
                    b1=ADAMW_B1,
                    b2=ADAMW_B2,
                    eps=ADAMW_EPS,
                    weight_decay=spec.weight_decay,
                    mask=adamw_decay_mask,
                ),
                adamw_mask,
                mask_compatible_extra_args=True,
            )
        )
    return transforms


def _default_backend(spec: OptimizerSpec, item: ParamMetadata) -> str:
    if spec.name == "adamw":
        return "adamw"
    if spec.name == "muon":
        return "muon" if item.tag in _MUON_TAGS else "adamw"
    return spec.name


def _resolve_backend(requested_backend: str, item: ParamMetadata, leaf: Any) -> tuple[str, int | None, str | None]:
    if requested_backend != "muon":
        return requested_backend, None, None
    if item.tag in _MUON_EXPERT_TAGS:
        return _resolve_expert_muon_backend(item, leaf)
    if item.tag not in _MUON_MATRIX_TAGS:
        return requested_backend, None, None
    tp_matrix_axis = _single_matrix_axis(leaf, "tp", item.path)
    if tp_matrix_axis is not None:
        return "dist_muon", tp_matrix_axis, "tp_sharded_matrix_muon"
    matrix_axis = _fsdp_matrix_axis(leaf)
    if matrix_axis is None:
        return "muon", None, None
    return "dion2", matrix_axis, "fsdp_sharded_optimizer_state"


def _resolve_expert_muon_backend(item: ParamMetadata, leaf: Any) -> tuple[str, int | None, str | None]:
    if len(item.shape) != 3:
        return "muon", None, None
    sharded_axes = _sharded_axes(leaf, {"data", "fsdp", "tp", "ep", "expert_fsdp"})
    matrix_sharded_axes = sorted(axis for axis in sharded_axes if axis in {1, 2})
    if matrix_sharded_axes:
        raise ContractError(
            f"Muon routed expert route for parameter {'.'.join(item.path)!r} requires complete per-expert "
            f"matrices; sharded matrix axes {matrix_sharded_axes} are unsupported until an explicit "
            "distributed expert matrix optimizer exists"
        )
    if 0 in sharded_axes:
        return "muon", None, "expert_axis_sharded_full_matrices"
    return "muon", None, None


def _validate_route(item: ParamMetadata, backend: str) -> None:
    if backend == "muon" and item.tag in _MUON_EXPERT_TAGS:
        if len(item.shape) != 3:
            raise ContractError(
                f"Muon routed expert route for parameter {'.'.join(item.path)!r} requires a rank-3 "
                f"expert matrix stack, got shape {item.shape}"
            )
        return
    if backend in {"muon", "dion2", "dist_muon"} and len(item.shape) != 2:
        display_backend = {"muon": "Muon", "dion2": "Dion2", "dist_muon": "Distributed Muon"}.get(backend, backend)
        raise ContractError(
            f"{display_backend} route for parameter {'.'.join(item.path)!r} requires a rank-2 matrix, "
            f"got shape {item.shape}"
        )
    if item.tag in _MUON_MATRIX_TAGS and len(item.shape) != 2:
        raise ContractError(
            f"Muon-eligible parameter {'.'.join(item.path)!r} has tag {item.tag!r} but shape {item.shape}"
        )
    if item.tag in _MUON_EXPERT_TAGS and len(item.shape) != 3:
        raise ContractError(
            f"Muon routed expert parameter {'.'.join(item.path)!r} has tag {item.tag!r} but shape {item.shape}"
        )


def _fallback_reason(spec: OptimizerSpec, item: ParamMetadata, backend: str, explicit_rule: bool) -> str | None:
    if spec.name != "muon" or backend != "adamw":
        return None
    if explicit_rule:
        return "route_rule"
    if item.tag == "embedding":
        return "embedding"
    if item.tag == "lm_head":
        return "lm_head"
    if item.tag in _NORM_TAGS:
        return "norm"
    if item.tag == "moe_expert_bias":
        return "expert_bias"
    if len(item.shape) != 2:
        return "rank_not_two"
    return "not_hidden_matrix"


def _default_weight_decay(item: ParamMetadata) -> bool:
    return item.tag != "moe_expert_bias"


def _validate_assignment_paths(params: PyTree, assignments: tuple[RouteAssignment, ...]) -> None:
    assignments_by_path = {assignment.path: assignment for assignment in assignments}
    param_paths = {_metadata_path_from_jax_path(path) for path, _value in jax.tree_util.tree_flatten_with_path(params)[0]}
    assignment_paths = set(assignments_by_path)
    missing = sorted(param_paths - assignment_paths)
    extra = sorted(assignment_paths - param_paths)
    if missing:
        raise ContractError(f"optimizer metadata is missing model parameter path {'.'.join(missing[0])!r}")
    if extra:
        raise ContractError(f"optimizer metadata has stale parameter path {'.'.join(extra[0])!r}")


def _params_by_metadata_path(params: PyTree) -> dict[tuple[str, ...], Any]:
    return {_metadata_path_from_jax_path(path): value for path, value in jax.tree_util.tree_flatten_with_path(params)[0]}


def _build_muon_execution_plans(
    assignments: tuple[RouteAssignment, ...],
    *,
    optimizer_init_state: PyTree,
    runtime_parameter_state: PyTree,
    gradient_shardings: PyTree | None,
    requested_mode: str,
) -> tuple[MuonLeafExecutionPlan, ...]:
    optimizer_by_path = _params_by_metadata_path(optimizer_init_state)
    runtime_by_path = _params_by_metadata_path(runtime_parameter_state)
    gradient_by_path = (
        None
        if gradient_shardings is None
        else {
            _metadata_path_from_jax_path(path): value
            for path, value in jax.tree_util.tree_flatten_with_path(gradient_shardings)[0]
        }
    )
    plans = []
    for assignment in assignments:
        if assignment.backend != "dist_muon":
            continue
        parameter_sharding = _require_named_sharding(runtime_by_path[assignment.path], assignment.path, "parameter")
        momentum_sharding = _require_named_sharding(optimizer_by_path[assignment.path], assignment.path, "momentum")
        gradient_sharding = (
            parameter_sharding
            if gradient_by_path is None
            else _require_named_sharding_value(gradient_by_path[assignment.path], assignment.path, "gradient")
        )
        update_sharding = gradient_sharding
        tp_partition_dim = _single_named_sharding_axis(momentum_sharding, "tp", assignment.path)
        if tp_partition_dim is None:
            raise ContractError(
                f"distributed Muon execution plan for {'.'.join(assignment.path)!r} requires a TP-sharded momentum"
            )
        rows, columns = assignment.logical_shape
        transpose_for_shape = rows > columns
        canonical_tp_dim = 1 - tp_partition_dim if transpose_for_shape else tp_partition_dim
        plans.append(
            MuonLeafExecutionPlan(
                path=assignment.path,
                logical_shape=(rows, columns),
                parameter_sharding=parameter_sharding,
                gradient_sharding=gradient_sharding,
                momentum_sharding=momentum_sharding,
                update_sharding=update_sharding,
                parameter_replica_axes=_replica_axes_for_sharding(parameter_sharding),
                gradient_replica_axes=_replica_axes_for_sharding(gradient_sharding),
                momentum_replica_axes=_replica_axes_for_sharding(momentum_sharding),
                update_replica_axes=_replica_axes_for_sharding(update_sharding),
                tp_partition_dim=tp_partition_dim,
                transpose_for_shape=transpose_for_shape,
                canonical_tp_dim=canonical_tp_dim,
                requested_mode=requested_mode,
                execution="duplicated",
                fallback_reason=None,
                bucket_id=-1,
                weight_decay=assignment.weight_decay,
            )
        )
    planned = _select_muon_shape_policy(tuple(plans))
    return _assign_muon_bucket_ids(planned)


def _select_muon_shape_policy(
    plans: tuple[MuonLeafExecutionPlan, ...],
) -> tuple[MuonLeafExecutionPlan, ...]:
    """Bind host-static execution decisions after compatible cohorts are known."""

    cohort_counts = Counter(_muon_policy_cohort_key(plan) for plan in plans)
    selected = []
    for plan in plans:
        tp_size = int(plan.momentum_sharding.mesh.shape["tp"])
        decision = select_muon_execution(
            requested_mode=plan.requested_mode,
            canonical_tp_dim=plan.canonical_tp_dim,
            logical_shape=plan.logical_shape,
            tp_size=tp_size,
            cohort_size=cohort_counts[_muon_policy_cohort_key(plan)],
        )
        fallback_reason = (
            decision.selection_reason
            if plan.requested_mode == "distributed"
            and decision.execution == "duplicated"
            else None
        )
        selected.append(
            replace(
                plan,
                execution=decision.execution,
                fallback_reason=fallback_reason,
                policy_version=decision.policy_version,
                eligible_executions=decision.eligible_executions,
                selection_reason=decision.selection_reason,
                short_dimension=decision.short_dimension,
                long_dimension=decision.long_dimension,
                tp_size=decision.tp_size,
                cohort_size=decision.cohort_size,
                gram_side=decision.gram_side,
                modeled_costs=decision.modeled_costs,
            )
        )
    return tuple(selected)


def _muon_policy_cohort_key(plan: MuonLeafExecutionPlan) -> tuple[Any, ...]:
    """Return the role-free compatibility key used for policy population."""

    mesh = plan.gradient_sharding.mesh
    return (
        tuple(sorted(plan.logical_shape)),
        plan.canonical_tp_dim,
        int(mesh.shape["tp"]),
        tuple(mesh.axis_names),
        tuple(mesh.shape.items()),
        repr(plan.gradient_sharding.spec),
        repr(plan.parameter_sharding.spec),
        repr(plan.momentum_sharding.spec),
        repr(plan.update_sharding.spec),
        plan.parameter_replica_axes,
        plan.gradient_replica_axes,
        plan.momentum_replica_axes,
        plan.update_replica_axes,
        plan.transpose_for_shape,
    )


def _assign_muon_bucket_ids(
    plans: tuple[MuonLeafExecutionPlan, ...],
) -> tuple[MuonLeafExecutionPlan, ...]:
    """Greedily assign deterministic bounded buckets to compatible leaves."""

    bucket_states: dict[tuple[Any, ...], tuple[int, int]] = {}
    next_bucket_id = 0
    assigned_by_path: dict[tuple[str, ...], MuonLeafExecutionPlan] = {}
    for plan in sorted(plans, key=lambda item: item.path):
        if plan.execution == "duplicated":
            assigned_by_path[plan.path] = plan
            continue
        key = (
            plan.execution,
            tuple(plan.gradient_sharding.mesh.axis_names),
            tuple(plan.gradient_sharding.mesh.shape.items()),
            repr(plan.gradient_sharding.spec),
            repr(plan.parameter_sharding.spec),
            repr(plan.momentum_sharding.spec),
            repr(plan.update_sharding.spec),
            plan.parameter_replica_axes,
            plan.gradient_replica_axes,
            plan.momentum_replica_axes,
            plan.update_replica_axes,
            plan.transpose_for_shape,
            plan.canonical_tp_dim,
            "bfloat16",
            "float32",
        )
        gram_dimension = (
            max(plan.logical_shape)
            if plan.execution == "distributed_large_gram"
            else min(plan.logical_shape)
        )
        payload_bytes = gram_dimension * gram_dimension * jnp.dtype(jnp.bfloat16).itemsize
        bucket_id, bucket_bytes = bucket_states.get(key, (-1, 0))
        if bucket_id < 0 or bucket_bytes + payload_bytes > _MUON_GRAM_BUCKET_MAX_BYTES:
            bucket_id = next_bucket_id
            next_bucket_id += 1
            bucket_bytes = 0
        bucket_states[key] = (bucket_id, bucket_bytes + payload_bytes)
        assigned_by_path[plan.path] = replace(plan, bucket_id=bucket_id)
    return tuple(assigned_by_path[plan.path] for plan in plans)


def _require_named_sharding(value: Any, path: tuple[str, ...], role: str) -> jax.sharding.NamedSharding:
    return _require_named_sharding_value(getattr(value, "sharding", None), path, role)


def _require_named_sharding_value(
    value: Any,
    path: tuple[str, ...],
    role: str,
) -> jax.sharding.NamedSharding:
    if not isinstance(value, jax.sharding.NamedSharding):
        raise ContractError(
            f"distributed Muon execution plan for {'.'.join(path)!r} requires static NamedSharding for {role}"
        )
    return value


def _single_named_sharding_axis(
    sharding: jax.sharding.NamedSharding,
    axis_name: str,
    path: tuple[str, ...],
) -> int | None:
    axes = [index for index, axis in enumerate(tuple(sharding.spec)) if _axis_contains(axis, axis_name)]
    if not axes:
        return None
    if len(axes) != 1:
        raise ContractError(
            f"distributed Muon execution plan for {'.'.join(path)!r} expects one {axis_name} matrix axis, "
            f"got {sharding.spec}"
        )
    return axes[0]


def _replica_axes_for_sharding(sharding: jax.sharding.NamedSharding) -> tuple[str, ...]:
    partitioned_axes = set()
    for axis in tuple(sharding.spec):
        if isinstance(axis, str):
            partitioned_axes.add(axis)
        elif isinstance(axis, tuple):
            partitioned_axes.update(str(name) for name in axis)
    return tuple(
        str(name)
        for name, size in sharding.mesh.shape.items()
        if name in {"fsdp", "tp", "ep", "expert_fsdp"}
        and int(size) > 1
        and name not in partitioned_axes
    )


def _fsdp_matrix_axis(leaf: Any) -> int | None:
    return _single_matrix_axis(leaf, "fsdp")


def _single_matrix_axis(leaf: Any, axis_name: str, path: tuple[str, ...] | None = None) -> int | None:
    axes = _partition_axes(leaf)
    if axes is None:
        return None
    sharded_axes = [idx for idx, axis in enumerate(axes) if _axis_contains(axis, axis_name)]
    if not sharded_axes:
        return None
    if len(sharded_axes) != 1:
        name = "parameter" if path is None else f"parameter {'.'.join(path)!r}"
        raise ContractError(f"Muon matrix route for {name} expects at most one {axis_name}-sharded parameter axis, got {axes}")
    return sharded_axes[0]


def _axis_sharded(leaf: Any, axis_name: str) -> bool:
    axes = _partition_axes(leaf)
    return False if axes is None else any(_axis_contains(axis, axis_name) for axis in axes)


def _sharded_axes(leaf: Any, names: set[str]) -> set[int]:
    axes = _partition_axes(leaf)
    if axes is None:
        return set()
    return {idx for idx, axis in enumerate(axes) if any(_axis_contains(axis, name) for name in names)}


def _partition_axes(leaf: Any) -> tuple[Any, ...] | None:
    sharding = getattr(leaf, "sharding", None)
    spec = getattr(sharding, "spec", None)
    if spec is None:
        return None
    return tuple(spec)


def _axis_contains(axis: Any, name: str) -> bool:
    if axis == name:
        return True
    if isinstance(axis, tuple):
        return name in axis
    return False


def _route_model_parallel_topology(leaf: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sharding = getattr(leaf, "sharding", None)
    spec = getattr(sharding, "spec", None)
    mesh = getattr(sharding, "mesh", None)
    if spec is None or mesh is None:
        return (), ()
    partitioned_axes = set()
    for axis in tuple(spec):
        if isinstance(axis, str):
            partitioned_axes.add(axis)
        elif isinstance(axis, tuple):
            partitioned_axes.update(str(name) for name in axis)
    active_axes = tuple(
        str(name)
        for name, size in mesh.shape.items()
        if name in {"fsdp", "tp", "ep", "expert_fsdp"} and int(size) > 1
    )
    return (
        tuple(name for name in active_axes if name in partitioned_axes),
        tuple(name for name in active_axes if name not in partitioned_axes),
    )


def _partition_spec_text(leaf: Any) -> str | None:
    sharding = getattr(leaf, "sharding", None)
    spec = getattr(sharding, "spec", None)
    return None if spec is None else repr(spec)


def _mask_from_assignments(
    params: PyTree,
    assignments: tuple[RouteAssignment, ...],
    predicate: Callable[[RouteAssignment], bool],
) -> PyTree:
    assignments_by_path = {assignment.path: assignment for assignment in assignments}

    def leaf_mask(path, _value):
        metadata_path = _metadata_path_from_jax_path(path)
        return bool(predicate(assignments_by_path[metadata_path]))

    return jax.tree_util.tree_map_with_path(leaf_mask, params)


def _mask_any(mask: PyTree) -> bool:
    return any(jax.tree.leaves(mask))


def _masked_parameter_shardings(params: PyTree, mask: PyTree) -> PyTree:
    def leaf_sharding(param: Any, selected: bool) -> Any:
        if not selected:
            return optax.MaskedNode()
        sharding = getattr(param, "sharding", None)
        if not isinstance(sharding, jax.sharding.NamedSharding):
            raise ContractError("distributed Muon requires statically known NamedSharding for every selected leaf")
        return sharding

    return jax.tree.map(leaf_sharding, params, mask)


def _masked_execution_plans(
    params: PyTree,
    mask: PyTree,
    execution_plans: tuple[MuonLeafExecutionPlan, ...],
) -> PyTree:
    plans_by_path = {plan.path: plan for plan in execution_plans}

    def leaf_plan(path, _param: Any, selected: bool) -> Any:
        if not selected:
            return optax.MaskedNode()
        metadata_path = _metadata_path_from_jax_path(path)
        plan = plans_by_path.get(metadata_path)
        if plan is None:
            raise ContractError(f"missing distributed Muon execution plan for {'.'.join(metadata_path)!r}")
        return plan

    return jax.tree_util.tree_map_with_path(leaf_plan, params, mask)


def _replica_aware_muon_transform(
    schedule: Callable[[Any], jax.Array],
    *,
    weight_decay: float,
    params: PyTree,
    mask: PyTree,
) -> optax.GradientTransformationExtraArgs:
    selected = [
        param
        for param, include in zip(jax.tree.leaves(params), jax.tree.leaves(mask), strict=True)
        if include
    ]
    replica_aware = any(
        isinstance(getattr(param, "sharding", None), jax.sharding.NamedSharding)
        and bool(_route_model_parallel_topology(param)[1])
        for param in selected
    )
    if not replica_aware:
        return muon_transform(schedule, weight_decay=weight_decay)
    return muon_transform(
        schedule,
        weight_decay=weight_decay,
        synchronize_model_replicas=True,
        parameter_shardings=_masked_parameter_shardings(params, mask),
    )


def _muon_description_suffix(spec: OptimizerSpec) -> str:
    if spec.name != "muon":
        return ""
    constants = muon_policy_constants()
    return (
        f" muon_momentum={constants['momentum']:g} "
        f"muon_nesterov={str(constants['nesterov']).lower()} "
        f"muon_ns_steps={constants['newton_schulz_steps']} "
        f"muon_ns_coefficients={constants['newton_schulz_coefficients']} "
        f"muon_scale_mode={constants['scale_mode']} "
        f"muon_rms_match_scale={constants['rms_match_scale']:g} "
        f"muon_tp_mode={spec.muon_tp_mode} "
        "muon_rank3_expert=per_expert_full_matrix "
        "adamw_fallback=true"
    )


def _schedule_description(spec: ScheduleSpec | None) -> str:
    if spec is None:
        return "same"
    total = "none" if spec.total_steps is None else str(spec.total_steps)
    stable = "none" if spec.stable_steps is None else str(spec.stable_steps)
    return (
        f"{spec.name}:peak_lr={spec.peak_lr:g}:warmup_steps={spec.warmup_steps}:"
        f"total_steps={total}:min_lr_ratio={spec.min_lr_ratio:g}:stable_steps={stable}"
    )


def _schedule_payload(spec: ScheduleSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "peak_lr": spec.peak_lr,
        "warmup_steps": spec.warmup_steps,
        "total_steps": spec.total_steps,
        "min_lr_ratio": spec.min_lr_ratio,
        "stable_steps": spec.stable_steps,
    }


def _metadata_path_from_jax_path(path) -> tuple[str, ...]:
    parts = []
    for key in path:
        name = getattr(key, "key", None)
        if name is None:
            name = getattr(key, "name", None)
        if name == "value":
            continue
        parts.append(str(name))
    return tuple(parts)


def _required_total_steps(spec: ScheduleSpec) -> int:
    if spec.total_steps is None:
        raise ContractError(f"optimizer.schedule.total_steps is required for {spec.name!r} schedule")
    return spec.total_steps


def _required_stable_steps(spec: ScheduleSpec) -> int:
    if spec.stable_steps is None:
        raise ContractError("optimizer.schedule.stable_steps is required for 'wsd' schedule")
    return spec.stable_steps


def _require_decay_steps(name: str, *, total_steps: int, warmup_steps: int, stable_steps: int) -> None:
    if warmup_steps + stable_steps >= total_steps:
        raise ContractError(
            f"optimizer.schedule.{name} requires warmup_steps + stable_steps to be less than total_steps; "
            f"got warmup_steps={warmup_steps}, stable_steps={stable_steps}, total_steps={total_steps}"
        )


def _linear_warmup(count, *, peak_lr: float, warmup_steps: int):
    if warmup_steps == 0:
        return jnp.asarray(peak_lr, dtype=jnp.float32)
    return peak_lr * jnp.minimum((count + 1.0) / warmup_steps, 1.0)


def _cosine_decay(count, *, peak_lr: float, min_lr: float, decay_steps: int):
    if decay_steps == 1:
        return jnp.asarray(min_lr, dtype=jnp.float32)
    progress = jnp.clip(count / (decay_steps - 1), 0.0, 1.0)
    cosine = 0.5 * (1.0 + jnp.cos(jnp.asarray(math.pi, dtype=jnp.float32) * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def _constant_schedule(*, peak_lr: float, warmup_steps: int):
    def schedule(count):
        count = jnp.asarray(count, dtype=jnp.float32)
        warmup_lr = _linear_warmup(count, peak_lr=peak_lr, warmup_steps=warmup_steps)
        return jnp.where(count < warmup_steps, warmup_lr, jnp.asarray(peak_lr, dtype=jnp.float32))

    return schedule


def _cosine_schedule(*, peak_lr: float, min_lr: float, total_steps: int, warmup_steps: int):
    decay_steps = total_steps - warmup_steps

    def schedule(count):
        count = jnp.asarray(count, dtype=jnp.float32)
        warmup_lr = _linear_warmup(count, peak_lr=peak_lr, warmup_steps=warmup_steps)
        decay_lr = _cosine_decay(
            count - warmup_steps,
            peak_lr=peak_lr,
            min_lr=min_lr,
            decay_steps=decay_steps,
        )
        return jnp.where(count < warmup_steps, warmup_lr, decay_lr)

    return schedule


def _wsd_schedule(*, peak_lr: float, min_lr: float, total_steps: int, warmup_steps: int, stable_steps: int):
    decay_start = warmup_steps + stable_steps
    decay_steps = total_steps - decay_start

    def schedule(count):
        count = jnp.asarray(count, dtype=jnp.float32)
        warmup_lr = _linear_warmup(count, peak_lr=peak_lr, warmup_steps=warmup_steps)
        decay_lr = _cosine_decay(
            count - decay_start,
            peak_lr=peak_lr,
            min_lr=min_lr,
            decay_steps=decay_steps,
        )
        return jnp.where(
            count < warmup_steps,
            warmup_lr,
            jnp.where(count < decay_start, jnp.asarray(peak_lr, dtype=jnp.float32), decay_lr),
        )

    return schedule
