"""Native JAX Muon transform."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple
import math

import jax
import jax.numpy as jnp
import optax
from jax.sharding import NamedSharding, PartitionSpec as P

PyTree = Any

_MODEL_PARALLEL_AXES = frozenset({"fsdp", "tp", "ep", "expert_fsdp"})

MUON_MOMENTUM = 0.95
MUON_NESTEROV = True
MUON_NS_STEPS = 5
MUON_NS_EPS = 1e-7
MUON_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
MUON_NS_PRECISION = "bfloat16"
MUON_RMS_MATCH_SCALE = 0.2
MUON_SCALE_MODE = "match_rms_adamw"


class MuonState(NamedTuple):
    """Opaque optimizer state for the Muon transform."""

    count: jax.Array
    momentum: PyTree


@dataclass(frozen=True, slots=True)
class _MuonLeafResult:
    update: jax.Array
    momentum: jax.Array


@dataclass(frozen=True, slots=True)
class MuonLeafExecutionPlan:
    """Host-static sharding and execution contract for one distributed Muon leaf."""

    path: tuple[str, ...]
    logical_shape: tuple[int, int]
    parameter_sharding: NamedSharding
    gradient_sharding: NamedSharding
    momentum_sharding: NamedSharding
    update_sharding: NamedSharding
    parameter_replica_axes: tuple[str, ...]
    gradient_replica_axes: tuple[str, ...]
    momentum_replica_axes: tuple[str, ...]
    update_replica_axes: tuple[str, ...]
    tp_partition_dim: int
    transpose_for_shape: bool
    canonical_tp_dim: int
    execution: str
    bucket_id: int


def muon_transform(
    learning_rate: Callable[[Any], jax.Array],
    *,
    weight_decay: float,
    newton_schulz_precision: str = MUON_NS_PRECISION,
    reference_logical_matrix: bool = False,
    synchronize_model_replicas: bool = False,
    parameter_shardings: PyTree | None = None,
) -> optax.GradientTransformationExtraArgs:
    """Build a matrix-only Muon gradient transformation."""

    _validate_newton_schulz_precision(newton_schulz_precision)

    def init_fn(params: PyTree) -> MuonState:
        return MuonState(
            count=jnp.asarray(0, dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
        )

    def update_fn(updates: PyTree, state: MuonState, params: PyTree | None = None, **extra_args) -> tuple[PyTree, MuonState]:
        del extra_args
        if params is None:
            raise ValueError("Muon update requires params")
        base_lr = learning_rate(state.count)

        def update_leaf(
            grad: jax.Array,
            momentum: jax.Array,
            param: jax.Array,
            sharding: NamedSharding | None = None,
        ) -> jax.Array:
            if synchronize_model_replicas or reference_logical_matrix:
                grad = _average_model_replicas(grad, sharding)
                momentum = _average_model_replicas(momentum, sharding)
                param = _average_model_replicas(param, sharding)
            next_momentum = MUON_MOMENTUM * momentum + (1.0 - MUON_MOMENTUM) * grad
            muon_input = (
                (1.0 - MUON_MOMENTUM) * grad + MUON_MOMENTUM * next_momentum
                if MUON_NESTEROV
                else next_momentum
            )
            if reference_logical_matrix:
                logical_input = _all_gather_logical_matrix(
                    muon_input.astype(jnp.bfloat16),
                    sharding,
                )
                orthogonalized = zeropower_via_newton_schulz(
                    logical_input,
                    precision=newton_schulz_precision,
                ).astype(muon_input.dtype)
                orthogonalized = _constrain_like(orthogonalized, sharding)
            else:
                orthogonalized = zeropower_via_newton_schulz(
                    muon_input,
                    precision=newton_schulz_precision,
                )
            adjusted_lr = base_lr * _rms_match_scale(param.shape)
            update = -adjusted_lr.astype(orthogonalized.dtype) * orthogonalized
            if weight_decay != 0.0:
                update = update - base_lr.astype(param.dtype) * weight_decay * param
            if not (synchronize_model_replicas or reference_logical_matrix):
                return update
            return _average_model_replicas(update, sharding)

        def momentum_leaf(
            grad: jax.Array,
            momentum: jax.Array,
            sharding: NamedSharding | None = None,
        ) -> jax.Array:
            if synchronize_model_replicas or reference_logical_matrix:
                grad = _average_model_replicas(grad, sharding)
                momentum = _average_model_replicas(momentum, sharding)
            if synchronize_model_replicas or reference_logical_matrix:
                next_momentum = MUON_MOMENTUM * momentum + (1.0 - MUON_MOMENTUM) * grad
                return _average_model_replicas(_constrain_like(next_momentum, sharding), sharding)
            return MUON_MOMENTUM * momentum + (1.0 - MUON_MOMENTUM) * grad

        if synchronize_model_replicas or reference_logical_matrix:
            if parameter_shardings is None:
                raise ValueError("replica-aware Muon requires static parameter shardings")
            is_sharding = lambda value: isinstance(value, NamedSharding)
            next_updates = jax.tree.map(
                update_leaf,
                updates,
                state.momentum,
                params,
                parameter_shardings,
                is_leaf=is_sharding,
            )
            next_momentum = jax.tree.map(
                momentum_leaf,
                updates,
                state.momentum,
                parameter_shardings,
                is_leaf=is_sharding,
            )
        else:
            next_updates = jax.tree.map(update_leaf, updates, state.momentum, params)
            next_momentum = jax.tree.map(momentum_leaf, updates, state.momentum)
        next_state = MuonState(count=optax.safe_increment(state.count), momentum=next_momentum)
        return next_updates, next_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def distributed_muon_transform(
    learning_rate: Callable[[Any], jax.Array],
    *,
    weight_decay: float,
    execution_plans: PyTree | None = None,
    parameter_shardings: PyTree | None = None,
) -> optax.GradientTransformationExtraArgs:
    """Build the planned exact distributed-Muon transform."""

    if execution_plans is None:
        if parameter_shardings is None:
            raise ValueError("distributed Muon requires static execution plans")
        return muon_transform(
            learning_rate,
            weight_decay=weight_decay,
            newton_schulz_precision=MUON_NS_PRECISION,
            reference_logical_matrix=True,
            synchronize_model_replicas=True,
            parameter_shardings=parameter_shardings,
        )

    def init_fn(params: PyTree) -> MuonState:
        return MuonState(
            count=jnp.asarray(0, dtype=jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
        )

    def update_fn(
        updates: PyTree,
        state: MuonState,
        params: PyTree | None = None,
        **extra_args,
    ) -> tuple[PyTree, MuonState]:
        del extra_args
        if params is None:
            raise ValueError("distributed Muon update requires params")
        base_lr = learning_rate(state.count)

        def update_leaf(
            grad: jax.Array,
            momentum: jax.Array,
            param: jax.Array,
            plan: MuonLeafExecutionPlan,
        ) -> _MuonLeafResult:
            grad = _average_declared_replicas(
                grad,
                plan.gradient_sharding,
                plan.gradient_replica_axes,
            )
            momentum = _average_declared_replicas(
                momentum,
                plan.momentum_sharding,
                plan.momentum_replica_axes,
            )
            param = _average_declared_replicas(
                param,
                plan.parameter_sharding,
                plan.parameter_replica_axes,
            )
            next_momentum = MUON_MOMENTUM * momentum + (1.0 - MUON_MOMENTUM) * grad
            muon_input = (
                (1.0 - MUON_MOMENTUM) * grad + MUON_MOMENTUM * next_momentum
                if MUON_NESTEROV
                else next_momentum
            )
            orthogonalized = _planned_zeropower(muon_input, plan)
            adjusted_lr = base_lr * _rms_match_scale(plan.logical_shape)
            update = -adjusted_lr.astype(orthogonalized.dtype) * orthogonalized
            if weight_decay != 0.0:
                param_for_update = jax.lax.with_sharding_constraint(param, plan.update_sharding)
                update = update - base_lr.astype(param_for_update.dtype) * weight_decay * param_for_update
            update = jax.lax.with_sharding_constraint(update, plan.update_sharding)
            next_momentum = jax.lax.with_sharding_constraint(next_momentum, plan.momentum_sharding)
            update = _average_declared_replicas(
                update,
                plan.update_sharding,
                plan.update_replica_axes,
            )
            next_momentum = _average_declared_replicas(
                next_momentum,
                plan.momentum_sharding,
                plan.momentum_replica_axes,
            )
            return _MuonLeafResult(update=update, momentum=next_momentum)

        is_plan = lambda value: isinstance(value, MuonLeafExecutionPlan)
        leaf_results = jax.tree.map(
            update_leaf,
            updates,
            state.momentum,
            params,
            execution_plans,
            is_leaf=is_plan,
        )
        is_result = lambda value: isinstance(value, _MuonLeafResult)
        next_updates = jax.tree.map(lambda result: result.update, leaf_results, is_leaf=is_result)
        next_momentum = jax.tree.map(lambda result: result.momentum, leaf_results, is_leaf=is_result)
        return next_updates, MuonState(
            count=optax.safe_increment(state.count),
            momentum=next_momentum,
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _planned_zeropower(
    value: jax.Array,
    plan: MuonLeafExecutionPlan,
) -> jax.Array:
    if plan.execution != "reference_once":
        raise ValueError(f"unsupported distributed Muon execution {plan.execution!r}")
    logical_input = _all_gather_logical_matrix(
        value.astype(jnp.bfloat16),
        plan.gradient_sharding,
    )
    result = zeropower_via_newton_schulz(
        logical_input,
        precision=MUON_NS_PRECISION,
    ).astype(value.dtype)
    return jax.lax.with_sharding_constraint(result, plan.update_sharding)


def _average_declared_replicas(
    value: jax.Array,
    sharding: NamedSharding,
    replica_axes: tuple[str, ...],
) -> jax.Array:
    if not replica_axes:
        return value
    return jax.shard_map(
        lambda local: jax.lax.pmean(local, replica_axes),
        mesh=sharding.mesh,
        in_specs=sharding.spec,
        out_specs=sharding.spec,
        check_vma=False,
    )(value)


def zeropower_via_newton_schulz(value: jax.Array, *, precision: str = MUON_NS_PRECISION) -> jax.Array:
    """Approximate the zeroth power of a matrix or expert-axis matrix stack."""

    if len(value.shape) not in {2, 3}:
        raise ValueError(f"Muon Newton-Schulz expects rank-2 or expert-axis rank-3 arrays, got shape {value.shape}")
    _validate_newton_schulz_precision(precision)
    original_dtype = value.dtype
    compute_dtype = jnp.bfloat16 if precision == "bfloat16" else jnp.float32
    x = value.astype(compute_dtype)
    norm_axes = (-2, -1) if x.ndim == 3 else None
    x = x / jnp.maximum(
        jnp.linalg.norm(x, axis=norm_axes, keepdims=norm_axes is not None),
        jnp.asarray(MUON_NS_EPS, dtype=x.dtype),
    )
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = jnp.swapaxes(x, -1, -2)
    a, b, c = MUON_NS_COEFFICIENTS
    for _ in range(MUON_NS_STEPS):
        xx_t = x @ jnp.swapaxes(x, -1, -2)
        update = b * xx_t + c * (xx_t @ xx_t)
        x = a * x + update @ x
    if transposed:
        x = jnp.swapaxes(x, -1, -2)
    return x.astype(original_dtype)


def muon_policy_constants() -> dict[str, Any]:
    """Return stable Muon constants recorded in artifacts."""

    return {
        "momentum": MUON_MOMENTUM,
        "nesterov": MUON_NESTEROV,
        "newton_schulz_steps": MUON_NS_STEPS,
        "newton_schulz_eps": MUON_NS_EPS,
        "newton_schulz_coefficients": list(MUON_NS_COEFFICIENTS),
        "newton_schulz_precision": MUON_NS_PRECISION,
        "scale_mode": MUON_SCALE_MODE,
        "rms_match_scale": MUON_RMS_MATCH_SCALE,
    }


def _rms_match_scale(shape: tuple[int, ...]) -> jax.Array:
    matrix_shape = shape[-2:] if len(shape) == 3 else shape
    return jnp.asarray(MUON_RMS_MATCH_SCALE * math.sqrt(max(matrix_shape)), dtype=jnp.float32)


def _validate_newton_schulz_precision(precision: str) -> None:
    if precision not in {"bfloat16", "float32"}:
        raise ValueError(f"unsupported Muon Newton-Schulz precision {precision!r}")


def _all_gather_logical_matrix(value: jax.Array, sharding: NamedSharding | None) -> jax.Array:
    if not isinstance(sharding, NamedSharding):
        return value
    if _is_replicated_named_sharding(sharding):
        return value
    tp_dimensions = [
        index
        for index, axis in enumerate(tuple(sharding.spec))
        if axis == "tp" or (isinstance(axis, tuple) and "tp" in axis)
    ]
    if len(tp_dimensions) != 1:
        raise ValueError(f"distributed Muon requires exactly one TP-sharded matrix dimension, got {sharding.spec}")
    tp_dimension = tp_dimensions[0]

    def gather_bfloat16(local: jax.Array) -> jax.Array:
        if local.dtype != jnp.bfloat16:
            raise ValueError(f"reference_once expects a bfloat16 Muon input, got {local.dtype}")
        # Preserve the two-byte collective payload through XLA algebraic
        # simplification. Without the bitcast, the CPU SPMD optimizer legally
        # hoists the preceding FP32-to-BF16 conversion past all_gather.
        bits = jax.lax.bitcast_convert_type(local, jnp.uint16)
        gathered_bits = jax.lax.all_gather(
            bits,
            "tp",
            axis=tp_dimension,
            tiled=True,
        )
        return jax.lax.bitcast_convert_type(gathered_bits, jnp.bfloat16)

    return jax.shard_map(
        gather_bfloat16,
        mesh=sharding.mesh,
        in_specs=sharding.spec,
        out_specs=P(),
        # JAX 0.10 does not infer that tiled all_gather makes every TP
        # participant hold the same logical matrix.
        check_vma=False,
    )(value)


def _constrain_like(value: jax.Array, sharding: NamedSharding | None) -> jax.Array:
    if not isinstance(sharding, NamedSharding):
        return value
    return jax.lax.with_sharding_constraint(value, sharding)


def _is_replicated_named_sharding(sharding: NamedSharding) -> bool:
    return all(axis is None for axis in tuple(sharding.spec))


def _average_model_replicas(value: jax.Array, sharding: NamedSharding | None) -> jax.Array:
    """Make implicit non-data model-axis replicas physically identical."""

    if not isinstance(sharding, NamedSharding):
        return value
    replica_axes = _replicated_model_axes(sharding)
    if not replica_axes:
        return value
    return jax.shard_map(
        lambda local: jax.lax.pmean(local, replica_axes),
        mesh=sharding.mesh,
        in_specs=sharding.spec,
        out_specs=sharding.spec,
        check_vma=False,
    )(value)


def _replicated_model_axes(sharding: NamedSharding) -> tuple[str, ...]:
    partitioned_axes = set()
    for axis in tuple(sharding.spec):
        if isinstance(axis, str):
            partitioned_axes.add(axis)
        elif isinstance(axis, tuple):
            partitioned_axes.update(str(name) for name in axis)
    return tuple(
        str(name)
        for name, size in sharding.mesh.shape.items()
        if name in _MODEL_PARALLEL_AXES and int(size) > 1 and name not in partitioned_axes
    )
