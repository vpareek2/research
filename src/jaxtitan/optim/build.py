"""Optimizer build boundary.

Jaxtitan keeps the runtime contract smaller than any one optimizer library:
compiled steps need an object with ``init`` and ``update`` plus opaque state.
Optax is the first backend adapter, not the public training abstraction.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import optax

from jaxtitan.errors import ContractError
from jaxtitan.models import ParamMetadata
from jaxtitan.optim.dion2 import dion2_policy_constants, dion2_transform
from jaxtitan.optim.muon import muon_policy_constants, muon_transform
from jaxtitan.specs.optimizer import OptimizerSpec, ScheduleSpec

PyTree = Any
ADAMW_B1 = 0.9
ADAMW_B2 = 0.999
ADAMW_EPS = 1e-8
_SUPPORTED_RUNTIME_BACKENDS = {"adamw", "muon"}
_INTERNAL_RUNTIME_BACKENDS = {"dion2"}
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


@dataclass(frozen=True, slots=True)
class OptimizerBuildResult:
    """Built optimizer runtime plus reproducibility metadata."""

    transform: OptimizerTransform
    schedule: Callable[[Any], jax.Array]
    adamw_fallback_schedule: Callable[[Any], jax.Array] | None
    route_assignments: tuple[RouteAssignment, ...]
    description: str


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
) -> OptimizerBuildResult:
    """Build the first Jaxtitan optimizer runtime boundary."""

    if spec.name not in _SUPPORTED_RUNTIME_BACKENDS:
        raise ContractError(
            f"optimizer.name {spec.name!r} is valid config but has no Jaxtitan runtime adapter yet; "
            f"supported runtime backends: {sorted(_SUPPORTED_RUNTIME_BACKENDS)}"
        )

    metadata = tuple(metadata)
    params_by_path = _params_by_metadata_path(model_state)
    assignments = _route_assignments(spec, metadata, params_by_path)
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
    parallelism_mode: str | None = None,
    fsdp_axis_size: int | None = None,
) -> dict[str, Any]:
    """Return a stable optimizer policy payload for artifacts and compatibility checks."""

    assignments = None if assignments is None else tuple(assignments)
    auto_routing_active = _auto_routing_active(
        spec,
        assignments=assignments,
        parallelism_mode=parallelism_mode,
        fsdp_axis_size=fsdp_axis_size,
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
            "distributed_policy": "replicated_or_auto_dion2_when_sharded",
            "distributed_matrix_update": "auto_dion2",
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
        "auto_routing": {
            "muon_sharded_matrix_backend": "dion2",
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
                    "distributed_policy": _route_distributed_policy(assignment.backend),
                    "auto_resolved": assignment.auto_resolved,
                    "resolution_reason": assignment.resolution_reason,
                    "matrix_axis": assignment.matrix_axis,
                }
            )
        payload["route_counts"] = dict(sorted(route_counts.items()))
        payload["fallback_counts"] = dict(sorted(fallback_counts.items()))
        payload["routes"] = routes
    return payload


def _auto_routing_active(
    spec: OptimizerSpec,
    *,
    assignments: tuple[RouteAssignment, ...] | None,
    parallelism_mode: str | None,
    fsdp_axis_size: int | None,
) -> bool:
    if assignments is not None:
        return any(assignment.auto_resolved for assignment in assignments)
    return bool(
        spec.name == "muon"
        and parallelism_mode in {"zero2", "fsdp"}
        and fsdp_axis_size is not None
        and fsdp_axis_size > 1
    )


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
            "optimizer_state": "replicated_or_expert_axis_muon_and_sharded_dion2",
            "gradient_update": "muon_when_complete_matrix_dion2_when_rank2_sharded",
            "zero2_fsdp": "auto_dion2",
        }
    return {
        "optimizer_state": "unknown",
        "gradient_update": "unknown",
        "zero2_fsdp": "unsupported",
    }


def _route_distributed_policy(backend: str) -> str:
    if backend == "adamw":
        return "elementwise_shard_safe"
    if backend == "muon":
        return "complete_matrix_or_per_expert_complete_matrix"
    if backend == "dion2":
        return "fsdp_sharded_matrix"
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
    adamw_mask = _mask_from_assignments(params, assignments, lambda assignment: assignment.backend == "adamw")
    adamw_decay_mask = _mask_from_assignments(
        params,
        assignments,
        lambda assignment: assignment.backend == "adamw" and assignment.weight_decay,
    )
    if _mask_any(muon_decay_mask):
        transforms.append(
            optax.masked(
                muon_transform(schedule, weight_decay=spec.weight_decay),
                muon_decay_mask,
                mask_compatible_extra_args=True,
            )
        )
    if _mask_any(muon_no_decay_mask):
        transforms.append(
            optax.masked(
                muon_transform(schedule, weight_decay=0.0),
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
    if _axis_sharded(leaf, "tp"):
        raise ContractError(
            f"Muon matrix route for parameter {'.'.join(item.path)!r} is not supported with tensor-parallel "
            "sharding yet; use optimizer.name='adamw'"
        )
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
    if backend in {"muon", "dion2"} and len(item.shape) != 2:
        display_backend = {"muon": "Muon", "dion2": "Dion2"}.get(backend, backend)
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


def _fsdp_matrix_axis(leaf: Any) -> int | None:
    axes = _partition_axes(leaf)
    if axes is None:
        return None
    fsdp_axes = [idx for idx, axis in enumerate(axes) if _axis_contains(axis, "fsdp")]
    if not fsdp_axes:
        return None
    if len(fsdp_axes) != 1:
        raise ContractError(f"Muon matrix route expects at most one fsdp-sharded parameter axis, got {axes}")
    return fsdp_axes[0]


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
