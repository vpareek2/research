"""Native JAX Muon transform."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple
import math

import jax
import jax.numpy as jnp
import optax
from jax.experimental.xla_metadata import set_xla_metadata
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
    requested_mode: str
    execution: str
    fallback_reason: str | None
    bucket_id: int
    weight_decay: bool


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
    """Build the planned distributed-Muon transform."""

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

        def prepare_leaf(
            grad: jax.Array,
            momentum: jax.Array,
            param: jax.Array,
            plan: MuonLeafExecutionPlan,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
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
            return muon_input, next_momentum, param

        is_plan = lambda value: isinstance(value, MuonLeafExecutionPlan)
        plan_leaves, plan_treedef = jax.tree_util.tree_flatten(
            execution_plans,
            is_leaf=is_plan,
        )
        update_leaves, update_treedef = jax.tree_util.tree_flatten(updates)
        momentum_leaves, momentum_treedef = jax.tree_util.tree_flatten(state.momentum)
        param_leaves, param_treedef = jax.tree_util.tree_flatten(params)
        if not (
            update_treedef == momentum_treedef == param_treedef == plan_treedef
            and len(update_leaves) == len(plan_leaves)
        ):
            raise ValueError("distributed Muon execution plans must match the selected parameter tree")

        prepared = [
            prepare_leaf(grad, momentum, param, plan)
            for grad, momentum, param, plan in zip(
                update_leaves,
                momentum_leaves,
                param_leaves,
                plan_leaves,
                strict=True,
            )
        ]
        orthogonalized: list[jax.Array | None] = [None] * len(plan_leaves)
        distributed_buckets: dict[int, list[int]] = {}
        for index, plan in enumerate(plan_leaves):
            if plan.execution == "duplicated":
                orthogonalized[index] = _planned_zeropower(prepared[index][0], plan)
            else:
                distributed_buckets.setdefault(plan.bucket_id, []).append(index)
        for indices in distributed_buckets.values():
            bucket_results = _bucketed_distributed_zeropower(
                tuple(prepared[index][0] for index in indices),
                tuple(plan_leaves[index] for index in indices),
            )
            for index, result in zip(indices, bucket_results, strict=True):
                orthogonalized[index] = result

        def finish_leaf(
            prepared_leaf: tuple[jax.Array, jax.Array, jax.Array],
            orthogonalized_leaf: jax.Array | None,
            plan: MuonLeafExecutionPlan,
        ) -> _MuonLeafResult:
            if orthogonalized_leaf is None:
                raise ValueError(f"missing distributed Muon result for {'.'.join(plan.path)!r}")
            _muon_input, next_momentum, param = prepared_leaf
            adjusted_lr = base_lr * _rms_match_scale(plan.logical_shape)
            update = -adjusted_lr.astype(orthogonalized_leaf.dtype) * orthogonalized_leaf
            if plan.weight_decay and weight_decay != 0.0:
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

        leaf_results = [
            finish_leaf(prepared_leaf, orthogonalized_leaf, plan)
            for prepared_leaf, orthogonalized_leaf, plan in zip(
                prepared,
                orthogonalized,
                plan_leaves,
                strict=True,
            )
        ]
        next_updates = update_treedef.unflatten([result.update for result in leaf_results])
        next_momentum = momentum_treedef.unflatten([result.momentum for result in leaf_results])
        return next_updates, MuonState(
            count=optax.safe_increment(state.count),
            momentum=next_momentum,
        )

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def _planned_zeropower(
    value: jax.Array,
    plan: MuonLeafExecutionPlan,
) -> jax.Array:
    if plan.execution == "duplicated":
        logical_input = _all_gather_logical_matrix(
            value.astype(jnp.bfloat16),
            plan.gradient_sharding,
        )
        result = zeropower_via_newton_schulz(
            logical_input,
            precision=MUON_NS_PRECISION,
        ).astype(value.dtype)
        return jax.lax.with_sharding_constraint(result, plan.update_sharding)
    if plan.execution in {"distributed_direct", "distributed_exchange"}:
        return _distributed_zeropower(value, plan)
    raise ValueError(f"unsupported distributed Muon execution {plan.execution!r}")


def _distributed_zeropower(
    value: jax.Array,
    plan: MuonLeafExecutionPlan,
) -> jax.Array:
    """Run one Megatron-style Newton–Schulz update."""

    return _bucketed_distributed_zeropower((value,), (plan,))[0]


def _bucketed_distributed_zeropower(
    values: tuple[jax.Array, ...],
    plans: tuple[MuonLeafExecutionPlan, ...],
) -> tuple[jax.Array, ...]:
    """Share norm and Gram reductions across compatible TP-sharded matrices."""

    if not values or len(values) != len(plans):
        raise ValueError("distributed Muon bucket requires equally sized non-empty values and plans")
    mesh = plans[0].gradient_sharding.mesh
    execution = plans[0].execution
    bucket_id = plans[0].bucket_id
    if execution not in {"distributed_direct", "distributed_exchange"}:
        raise ValueError(f"unsupported bucketed distributed Muon execution {execution!r}")
    if any(plan.gradient_sharding.mesh != mesh or plan.execution != execution for plan in plans):
        raise ValueError("distributed Muon bucket contains incompatible execution plans")

    def kernel(*locals_: jax.Array) -> tuple[jax.Array, ...]:
        local_floats = tuple(local.astype(jnp.float32) for local in locals_)
        local_square_sums = jnp.stack(
            [jnp.sum(jnp.square(local), dtype=jnp.float32) for local in local_floats]
        )
        with set_xla_metadata(muon_op="norm", muon_bucket=str(bucket_id)):
            global_square_sums = jax.lax.psum(local_square_sums, "tp")
        xs = []
        for index, (local_float, plan) in enumerate(zip(local_floats, plans, strict=True)):
            norm = jnp.sqrt(global_square_sums[index])
            x = (
                local_float
                / jnp.maximum(norm, jnp.asarray(MUON_NS_EPS, dtype=jnp.float32))
            ).astype(jnp.bfloat16)
            if plan.transpose_for_shape:
                x = jnp.swapaxes(x, -1, -2)
            if execution == "distributed_exchange":
                with set_xla_metadata(muon_op="exchange_forward", muon_bucket=str(bucket_id)):
                    x = jax.lax.all_to_all(
                        x,
                        "tp",
                        split_axis=1,
                        concat_axis=0,
                        tiled=True,
                    )
            xs.append(x)
        a, b, c = MUON_NS_COEFFICIENTS
        for _ in range(MUON_NS_STEPS):
            gram_shapes = tuple((x.shape[0], x.shape[0]) for x in xs)
            gram_sizes = tuple(rows * columns for rows, columns in gram_shapes)
            local_grams = tuple(
                (x @ jnp.swapaxes(x, -1, -2)).reshape(-1)
                for x in xs
            )
            with set_xla_metadata(muon_op="gram", muon_bucket=str(bucket_id)):
                reduced_grams = jax.lax.psum(jnp.concatenate(local_grams), "tp")
            next_xs = []
            offset = 0
            for x, gram_shape, gram_size in zip(xs, gram_shapes, gram_sizes, strict=True):
                gram = jax.lax.dynamic_slice_in_dim(
                    reduced_grams,
                    offset,
                    gram_size,
                ).reshape(gram_shape)
                gram2 = gram @ gram
                next_xs.append(a * x + (b * gram + c * gram2) @ x)
                offset += gram_size
            xs = next_xs
        results = []
        for x, local, plan in zip(xs, locals_, plans, strict=True):
            if execution == "distributed_exchange":
                with set_xla_metadata(muon_op="exchange_reverse", muon_bucket=str(bucket_id)):
                    x = jax.lax.all_to_all(
                        x,
                        "tp",
                        split_axis=0,
                        concat_axis=1,
                        tiled=True,
                    )
            if plan.transpose_for_shape:
                x = jnp.swapaxes(x, -1, -2)
            results.append(x.astype(local.dtype))
        return tuple(results)

    return jax.shard_map(
        kernel,
        mesh=mesh,
        in_specs=tuple(plan.gradient_sharding.spec for plan in plans),
        out_specs=tuple(plan.update_sharding.spec for plan in plans),
        check_vma=True,
    )(*values)


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
