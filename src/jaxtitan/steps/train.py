"""Training step boundary."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from jaxtitan.batch import Batch
from jaxtitan.errors import ContractError
from jaxtitan.metrics import StepMetrics
from jaxtitan.mesh import ShardingPlan, gradient_shardings_like, replicated_shardings_like
from jaxtitan.models import apply_model_output
from jaxtitan.models.execution import ModelExecutionContext, feature_parallel_activation
from jaxtitan.optim import OptimizerBuildResult, OptimizerTransform
from jaxtitan.specs.model import MoeBalanceSpec
from jaxtitan.specs.run import TrainingLossSpec
from jaxtitan.state import RngState, TrainState
from jaxtitan.steps.eval import tensor_parallel_causal_lm_loss
from jaxtitan.steps.moe_balance import (
    apply_moe_balance_update,
    initialize_moe_balance_state,
    router_counts_from_stats,
    router_importance_from_stats,
)


def initialize_train_state(
    model_state: Any,
    optimizer_transform: OptimizerTransform,
    seed: int,
    *,
    optimizer_init_model_state: Any | None = None,
    moe_balance_spec: MoeBalanceSpec | None = None,
) -> TrainState:
    """Initialize explicit train state from model state and an optimizer transform."""

    init_model_state = model_state if optimizer_init_model_state is None else optimizer_init_model_state
    if jax.tree.structure(init_model_state) != jax.tree.structure(model_state):
        raise ContractError("optimizer init model state must match model state structure")
    train_key, data_key, eval_key, sample_key = jax.random.split(jax.random.key(seed), 4)
    return TrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        tokens_seen=jnp.asarray(0, dtype=jnp.uint32),
        model=model_state,
        opt_state=optimizer_transform.init(init_model_state),
        rng=RngState(train=train_key, data=data_key, eval=eval_key, sample=sample_key),
        schedule_state=None,
        moe_balance=initialize_moe_balance_state(model_state, moe_balance_spec),
    )


def train_step(
    graph: Any,
    optimizer: OptimizerBuildResult,
    state: TrainState,
    batch: Batch,
) -> tuple[TrainState, StepMetrics]:
    """Run one compiled train step for a model graph, optimizer, state, and batch."""

    return make_train_step(graph, optimizer)(state, batch)


def make_train_step(
    graph: Any,
    optimizer: OptimizerBuildResult,
    *,
    sharding: ShardingPlan | None = None,
    state_template: TrainState | None = None,
    donate_state: bool = False,
    expected_batch_shape: tuple[int, int, int] | None = None,
    loss: TrainingLossSpec | None = None,
) -> Callable[[TrainState, Batch], tuple[TrainState, StepMetrics]]:
    """Create a compiled train callable bound to a static graph and optimizer."""

    loss = TrainingLossSpec() if loss is None else loss
    z_loss_weight = float(loss.z_loss_weight)
    if sharding is not None and state_template is None:
        raise ContractError("state_template is required when compiling train step with explicit shardings")
    state_sharding = None if sharding is None else replicated_shardings_like(state_template, sharding)
    gradient_sharding = None if sharding is None else gradient_shardings_like(state_template.model, sharding)
    execution = _model_execution_context(sharding)
    in_shardings = None
    out_shardings = None
    if sharding is not None:
        in_shardings = (
            state_sharding,
            sharding.batch.accumulated_input_ids,
            sharding.batch.accumulated_target_ids,
            sharding.batch.accumulated_loss_mask,
        )
        out_shardings = (state_sharding, *([sharding.metrics] * 33))
    optimizer_group_templates, optimizer_group_index_by_path = _optimizer_group_templates(
        optimizer.route_assignments,
    )
    optimizer_group_count = len(optimizer_group_templates)

    def _compiled_impl(
        state: TrainState,
        input_ids: Any,
        target_ids: Any,
        loss_mask: Any,
    ) -> tuple[Any, ...]:
        grad_zero = jax.tree.map(jnp.zeros_like, state.model)

        def microbatch_grad(params: Any, micro_input_ids: Any, micro_target_ids: Any, micro_loss_mask: Any):
            def objective_sum_fn(loss_params: Any):
                output = (
                    apply_model_output(graph, loss_params, micro_input_ids)
                    if execution is None
                    else apply_model_output(graph, loss_params, micro_input_ids, execution=execution)
                )
                loss = tensor_parallel_causal_lm_loss(output.logits, micro_target_ids, micro_loss_mask, execution)
                aux_loss = _aux_loss_value(output.aux_losses)
                moe_aux_loss = _aux_loss_value(output.aux_losses, name_prefix="moe_")
                z_loss_sum = _z_loss_sum(output.logits, micro_loss_mask, execution) * jnp.asarray(
                    z_loss_weight,
                    dtype=jnp.float32,
                )
                objective_sum = loss.loss_sum + z_loss_sum + aux_loss * loss.token_count.astype(jnp.float32)
                router_counts = router_counts_from_stats(output.router_stats, state.moe_balance)
                router_importance = router_importance_from_stats(output.router_stats, state.moe_balance)
                return objective_sum, (
                    loss.loss_sum,
                    loss.token_count,
                    aux_loss,
                    moe_aux_loss,
                    z_loss_sum,
                    router_counts,
                    router_importance,
                )

            (
                objective_sum,
                (loss_sum, token_count, aux_loss, moe_aux_loss, z_loss_sum, router_counts, router_importance),
            ), grads = jax.value_and_grad(objective_sum_fn, has_aux=True)(params)
            return (
                loss_sum,
                token_count,
                aux_loss,
                moe_aux_loss,
                z_loss_sum,
                router_counts,
                router_importance,
                objective_sum,
                grads,
            )

        def accumulate(carry: tuple[Any, ...], micro: tuple[Any, Any, Any]):
            (
                grad_accum,
                loss_sum_accum,
                token_count_accum,
                objective_sum_accum,
                aux_sum_accum,
                moe_aux_sum_accum,
                z_loss_sum_accum,
            ) = carry
            micro_input_ids, micro_target_ids, micro_loss_mask = micro
            (
                loss_sum,
                token_count,
                aux_loss,
                moe_aux_loss,
                z_loss_sum,
                router_counts,
                router_importance,
                objective_sum,
                grads,
            ) = microbatch_grad(state.model, micro_input_ids, micro_target_ids, micro_loss_mask)
            return (
                jax.tree.map(lambda total, grad: total + grad, grad_accum, grads),
                loss_sum_accum + loss_sum,
                token_count_accum + token_count,
                objective_sum_accum + objective_sum,
                aux_sum_accum + aux_loss * token_count.astype(jnp.float32),
                moe_aux_sum_accum + moe_aux_loss * token_count.astype(jnp.float32),
                z_loss_sum_accum + z_loss_sum,
            ), (loss_sum, token_count, router_counts, router_importance)

        (
            grad_sum,
            loss_sum,
            token_count,
            objective_sum,
            aux_sum,
            moe_aux_sum,
            z_loss_sum,
        ), (micro_loss_sums, micro_token_counts, micro_router_counts, micro_router_importance) = jax.lax.scan(
            accumulate,
            (
                grad_zero,
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            (input_ids, target_ids, loss_mask),
        )
        router_counts = jnp.sum(micro_router_counts, axis=0)
        router_importance = jnp.sum(micro_router_importance, axis=0)
        grad_denominator = jnp.asarray(token_count, dtype=jnp.float32)
        grads = jax.tree.map(lambda grad: grad / grad_denominator, grad_sum)
        if gradient_sharding is not None:
            grads = _constrain_like(grads, gradient_sharding)
            grads = _average_model_replicas(grads, gradient_sharding)
        updates, next_opt_state = optimizer.transform.update(grads, state.opt_state, params=state.model)
        if gradient_sharding is not None:
            updates = _constrain_like(updates, gradient_sharding)
        next_model = jax.tree.map(lambda param, update: param + update, state.model, updates)
        next_model, next_moe_balance, moe_balance_metrics = apply_moe_balance_update(
            next_model,
            state.moe_balance,
            router_counts,
        )
        next_step = state.step + jnp.asarray(1, dtype=state.step.dtype)
        next_tokens_seen = state.tokens_seen + token_count.astype(state.tokens_seen.dtype)
        next_state = state.replace(
            step=next_step,
            tokens_seen=next_tokens_seen,
            model=next_model,
            opt_state=next_opt_state,
            moe_balance=next_moe_balance,
        )
        lr = optimizer.schedule(state.step)
        grad_norm = _tree_l2_norm(grads)
        param_norm = _tree_l2_norm(next_model)
        update_norm = _tree_l2_norm(updates)
        group_grad_norms, group_update_norms, group_param_norms = _optimizer_group_norms(
            grads,
            updates,
            next_model,
            optimizer_group_index_by_path,
            optimizer_group_count,
        )
        micro_losses = micro_loss_sums / micro_token_counts.astype(jnp.float32)
        microbatch_loss_mean = jnp.mean(micro_losses)
        microbatch_loss_max = jnp.max(micro_losses)
        batch_het = microbatch_loss_max - microbatch_loss_mean
        objective = objective_sum / grad_denominator
        aux_loss = aux_sum / grad_denominator
        moe_aux_loss = moe_aux_sum / grad_denominator
        z_loss = z_loss_sum / grad_denominator
        total_loss = objective
        router_diagnostics = _router_diagnostics(router_counts, router_importance)
        return (
            next_state,
            loss_sum,
            token_count,
            lr,
            grad_norm,
            param_norm,
            update_norm,
            group_grad_norms,
            group_update_norms,
            group_param_norms,
            objective,
            aux_loss,
            z_loss,
            moe_aux_loss,
            total_loss,
            microbatch_loss_mean,
            microbatch_loss_max,
            batch_het,
            router_counts,
            router_importance,
            router_diagnostics["max_vio"],
            router_diagnostics["load_min"],
            router_diagnostics["load_max"],
            router_diagnostics["load_entropy"],
            router_diagnostics["mean_load_cv"],
            router_diagnostics["std_load_cv"],
            router_diagnostics["mean_load_entropy"],
            router_diagnostics["min_load_entropy"],
            router_diagnostics["dead_experts_count"],
            router_diagnostics["experts_active_mean"],
            router_diagnostics["mean_importance_cv"],
            router_diagnostics["mean_importance_entropy"],
            moe_balance_metrics.bias_norm,
            moe_balance_metrics.momentum_norm,
        )

    _compiled = jax.jit(
        _compiled_impl,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        donate_argnums=(0,) if donate_state else (),
    )

    def _train(state: TrainState, batch: Batch) -> tuple[TrainState, StepMetrics]:
        _validate_train_batch(batch, expected_batch_shape=expected_batch_shape)
        (
            next_state,
            loss_sum,
            token_count,
            lr,
            grad_norm,
            param_norm,
            update_norm,
            optimizer_group_grad_norms,
            optimizer_group_update_norms,
            optimizer_group_param_norms,
            objective,
            aux_loss,
            z_loss,
            moe_aux_loss,
            total_loss,
            microbatch_loss_mean,
            microbatch_loss_max,
            batch_het,
            router_expert_counts,
            router_importance,
            router_max_vio,
            router_load_min,
            router_load_max,
            router_load_entropy,
            router_mean_load_cv,
            router_std_load_cv,
            router_mean_load_entropy,
            router_min_load_entropy,
            router_dead_experts_count,
            router_experts_active_mean,
            router_mean_importance_cv,
            router_mean_importance_entropy,
            smebu_bias_norm,
            smebu_momentum_norm,
        ) = _compiled(
            state,
            _ensure_accumulation_axis(batch.input_ids),
            _ensure_accumulation_axis(batch.target_ids),
            _ensure_accumulation_axis(batch.loss_mask),
        )
        router_active = router_expert_counts.shape[0] > 0
        balance_active = state.moe_balance is not None
        optimizer_group_specs = _optimizer_group_specs_for_model(
            next_state.model,
            optimizer.route_assignments,
            optimizer_group_templates,
        )
        metrics = StepMetrics(
            loss_sum=loss_sum,
            token_count=token_count,
            lr=lr,
            grad_norm=grad_norm,
            param_norm=param_norm,
            update_norm=update_norm,
            optimizer_group_specs=optimizer_group_specs,
            optimizer_group_grad_norms=optimizer_group_grad_norms,
            optimizer_group_update_norms=optimizer_group_update_norms,
            optimizer_group_param_norms=optimizer_group_param_norms,
            overflow=None,
            objective=objective,
            aux_loss=aux_loss,
            z_loss=z_loss,
            moe_aux_loss=moe_aux_loss,
            total_loss=total_loss,
            router_expert_counts=None if not router_active else router_expert_counts,
            router_importance=None if not router_active else router_importance,
            router_max_vio=None if not router_active else router_max_vio,
            router_load_min=None if not router_active else router_load_min,
            router_load_max=None if not router_active else router_load_max,
            router_load_entropy=None if not router_active else router_load_entropy,
            router_mean_load_cv=None if not router_active else router_mean_load_cv,
            router_std_load_cv=None if not router_active else router_std_load_cv,
            router_mean_load_entropy=None if not router_active else router_mean_load_entropy,
            router_min_load_entropy=None if not router_active else router_min_load_entropy,
            router_dead_experts_count=None if not router_active else router_dead_experts_count,
            router_experts_active_mean=None if not router_active else router_experts_active_mean,
            router_mean_importance_cv=None if not router_active else router_mean_importance_cv,
            router_mean_importance_entropy=None if not router_active else router_mean_importance_entropy,
            smebu_bias_norm=None if not balance_active else smebu_bias_norm,
            smebu_momentum_norm=None if not balance_active else smebu_momentum_norm,
            aux_metrics=(),
            microbatch_loss_mean=microbatch_loss_mean,
            microbatch_loss_max=microbatch_loss_max,
            batch_het=batch_het,
        )
        return next_state, metrics

    return _train


def _model_execution_context(sharding: ShardingPlan | None) -> ModelExecutionContext | None:
    if sharding is None or (
        not sharding.parallelism.expert_parallel
        and not sharding.parallelism.tensor_parallel
        and not sharding.parallelism.context_parallel
    ):
        return None
    if sharding.parallelism.expert_parallel and sharding.expert_parallel_axis is None:
        raise ContractError("expert parallel sharding plan is missing a resolved expert axis")
    return ModelExecutionContext(
        expert_parallel_mesh=sharding.mesh.mesh if sharding.parallelism.expert_parallel else None,
        expert_parallel_axis_name=sharding.expert_parallel_axis or "ep",
        expert_fsdp_axis_name=sharding.expert_fsdp_axis,
        expert_parallel_dispatcher=sharding.expert_parallel_dispatcher or "all_to_all",
        tensor_parallel_mesh=sharding.mesh.mesh if sharding.parallelism.tensor_parallel else None,
        tensor_parallel_axis_name=sharding.tensor_parallel_axis or "tp",
        context_parallel_mesh=sharding.mesh.mesh if sharding.parallelism.context_parallel else None,
        context_parallel_axis_name=sharding.context_parallel_axis or "cp",
    )


def _validate_train_batch(batch: Batch, *, expected_batch_shape: tuple[int, int, int] | None) -> None:
    input_shape = _shape(batch.input_ids)
    if len(input_shape) not in {2, 3}:
        raise ContractError(f"batch.input_ids must have rank 2 or 3, got shape {input_shape}")
    if _shape(batch.target_ids) != input_shape:
        raise ContractError(f"batch.target_ids shape {_shape(batch.target_ids)} must equal input_ids shape {input_shape}")
    if _shape(batch.loss_mask) != input_shape:
        raise ContractError(f"batch.loss_mask shape {_shape(batch.loss_mask)} must equal input_ids shape {input_shape}")
    if not _is_integer_dtype(batch.input_ids):
        raise ContractError(f"batch.input_ids must have integer dtype, got {_dtype(batch.input_ids)}")
    if not _is_integer_dtype(batch.target_ids):
        raise ContractError(f"batch.target_ids must have integer dtype, got {_dtype(batch.target_ids)}")
    if _dtype(batch.loss_mask) != jnp.dtype(jnp.bool_):
        raise ContractError(f"batch.loss_mask must have bool dtype, got {_dtype(batch.loss_mask)}")
    normalized_shape = input_shape if len(input_shape) == 3 else (1, *input_shape)
    if expected_batch_shape is not None and normalized_shape != expected_batch_shape:
        raise ContractError(
            f"train batch shape {normalized_shape} must match expected compiled shape {expected_batch_shape}"
        )


def _ensure_accumulation_axis(value: Any) -> Any:
    if len(_shape(value)) == 2:
        return jnp.asarray(value)[None, ...]
    return value


def _constrain_like(tree: Any, shardings: Any) -> Any:
    return jax.tree.map(lambda leaf, sharding: jax.lax.with_sharding_constraint(leaf, sharding), tree, shardings)


def _average_model_replicas(tree: Any, shardings: Any) -> Any:
    def average(leaf: Any, sharding: Any) -> Any:
        if not isinstance(sharding, jax.sharding.NamedSharding):
            return leaf
        partitioned_axes = set()
        for axis in tuple(sharding.spec):
            if isinstance(axis, str):
                partitioned_axes.add(axis)
            elif isinstance(axis, tuple):
                partitioned_axes.update(str(name) for name in axis)
        replica_axes = tuple(
            str(name)
            for name, size in sharding.mesh.shape.items()
            if name in {"fsdp", "tp", "ep", "expert_fsdp"}
            and int(size) > 1
            and name not in partitioned_axes
        )
        if not replica_axes:
            return leaf
        return jax.shard_map(
            lambda local: jax.lax.pmean(local, replica_axes),
            mesh=sharding.mesh,
            in_specs=sharding.spec,
            out_specs=sharding.spec,
            check_vma=False,
        )(leaf)

    return jax.tree.map(
        average,
        tree,
        shardings,
        is_leaf=lambda value: isinstance(value, jax.sharding.NamedSharding),
    )


def _optimizer_group_templates(
    assignments: Any,
) -> tuple[tuple[dict[str, Any], ...], dict[tuple[str, ...], int]]:
    assignments = tuple(assignments)
    if not assignments:
        return (), {}
    groups: list[dict[str, Any]] = []
    group_by_key: dict[tuple[str, str], int] = {}
    group_index_by_path: dict[tuple[str, ...], int] = {}
    for assignment in assignments:
        key = (assignment.tag, assignment.backend)
        group_index = group_by_key.get(key)
        if group_index is None:
            group_index = len(groups)
            group_by_key[key] = group_index
            groups.append(
                {
                    "group": f"{assignment.tag}:{assignment.backend}",
                    "tag": assignment.tag,
                    "backend": assignment.backend,
                    "weight_decay_enabled_count": 0,
                    "auto_resolved_count": 0,
                }
            )
        if assignment.weight_decay:
            groups[group_index]["weight_decay_enabled_count"] += 1
        if assignment.auto_resolved:
            groups[group_index]["auto_resolved_count"] += 1
        group_index_by_path[assignment.path] = group_index
    return tuple(groups), group_index_by_path


def _optimizer_group_specs_for_model(
    model: Any,
    assignments: Any,
    templates: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    assignments = tuple(assignments)
    if not assignments:
        return ()
    assignments_by_path = {assignment.path: assignment for assignment in assignments}
    group_by_key = {(template["tag"], template["backend"]): idx for idx, template in enumerate(templates)}
    leaf_counts = [0 for _template in templates]
    parameter_counts = [0 for _template in templates]
    model_paths = set()
    for path, leaf in jax.tree_util.tree_flatten_with_path(model)[0]:
        metadata_path = _metadata_path_from_jax_path(path)
        model_paths.add(metadata_path)
        assignment = assignments_by_path.get(metadata_path)
        if assignment is None:
            raise ContractError(f"optimizer diagnostics missing route for model parameter {'.'.join(metadata_path)!r}")
        group_index = group_by_key[(assignment.tag, assignment.backend)]
        leaf_counts[group_index] += 1
        parameter_counts[group_index] += int(jnp.size(leaf))
    assignment_paths = set(assignments_by_path)
    missing = sorted(assignment_paths - model_paths)
    if missing:
        raise ContractError(f"optimizer diagnostics route has stale parameter path {'.'.join(missing[0])!r}")
    specs = []
    for idx, template in enumerate(templates):
        specs.append(
            {
                **template,
                "leaf_count": leaf_counts[idx],
                "parameter_count": parameter_counts[idx],
            }
        )
    return tuple(specs)


def _optimizer_group_norms(
    grads: Any,
    updates: Any,
    params: Any,
    group_index_by_path: dict[tuple[str, ...], int],
    group_count: int,
) -> tuple[Any, Any, Any]:
    if group_count == 0:
        empty = jnp.zeros((0,), dtype=jnp.float32)
        return empty, empty, empty
    grad_squares = jnp.zeros((group_count,), dtype=jnp.float32)
    update_squares = jnp.zeros((group_count,), dtype=jnp.float32)
    param_squares = jnp.zeros((group_count,), dtype=jnp.float32)
    grad_pairs = jax.tree_util.tree_flatten_with_path(grads)[0]
    update_leaves = jax.tree.leaves(updates)
    param_leaves = jax.tree.leaves(params)
    for (path, grad), update, param in zip(grad_pairs, update_leaves, param_leaves, strict=True):
        metadata_path = _metadata_path_from_jax_path(path)
        group_index = group_index_by_path.get(metadata_path)
        if group_index is None:
            raise ContractError(f"optimizer diagnostics missing route for gradient {'.'.join(metadata_path)!r}")
        grad_squares = grad_squares.at[group_index].add(_leaf_square_norm(grad))
        update_squares = update_squares.at[group_index].add(_leaf_square_norm(update))
        param_squares = param_squares.at[group_index].add(_leaf_square_norm(param))
    return jnp.sqrt(grad_squares), jnp.sqrt(update_squares), jnp.sqrt(param_squares)


def _leaf_square_norm(value: Any) -> Any:
    return jnp.sum(jnp.square(jnp.asarray(value, dtype=jnp.float32)))


def _metadata_path_from_jax_path(path: Any) -> tuple[str, ...]:
    parts = []
    for key in path:
        name = getattr(key, "key", None)
        if name is None:
            name = getattr(key, "name", None)
        if name == "value":
            continue
        parts.append(str(name))
    return tuple(parts)


def _aux_loss_value(aux_losses: Any, *, name_prefix: str | None = None):
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for aux_loss in aux_losses:
        if name_prefix is not None and not aux_loss.name.startswith(name_prefix):
            continue
        total = total + jnp.asarray(aux_loss.value, dtype=jnp.float32) * jnp.asarray(
            aux_loss.weight,
            dtype=jnp.float32,
        )
    return total


def _z_loss_sum(logits: Any, loss_mask: Any, execution: ModelExecutionContext | None = None):
    if execution is not None and (execution.tensor_parallel_enabled or execution.context_parallel_enabled):
        logits = feature_parallel_activation(logits, execution)
    log_z = jax.nn.logsumexp(jnp.asarray(logits, dtype=jnp.float32), axis=-1)
    mask = jnp.asarray(loss_mask, dtype=jnp.bool_)
    return jnp.sum(jnp.where(mask, jnp.square(log_z), 0.0))


def _router_diagnostics(counts: Any, importance: Any) -> dict[str, Any]:
    counts = jnp.asarray(counts, dtype=jnp.float32)
    importance = jnp.asarray(importance, dtype=jnp.float32)
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    if counts.shape[0] == 0 or counts.shape[1] == 0:
        return {
            "max_vio": zero,
            "load_min": zero,
            "load_max": zero,
            "load_entropy": zero,
            "mean_load_cv": zero,
            "std_load_cv": zero,
            "mean_load_entropy": zero,
            "min_load_entropy": zero,
            "dead_experts_count": zero,
            "experts_active_mean": zero,
            "mean_importance_cv": zero,
            "mean_importance_entropy": zero,
        }
    layer_totals = jnp.sum(counts, axis=-1)
    layer_means = layer_totals / jnp.asarray(counts.shape[-1], dtype=jnp.float32)
    load_min = jnp.min(counts)
    load_max = jnp.max(counts)
    max_vio = jnp.max((jnp.max(counts, axis=-1) - layer_means) / jnp.maximum(layer_means, 1e-6))
    load_entropy = _row_entropy(counts)
    importance_entropy = _row_entropy(importance)
    load_cv = _row_cv(counts)
    importance_cv = _row_cv(importance)
    return {
        "max_vio": max_vio,
        "load_min": load_min,
        "load_max": load_max,
        "load_entropy": jnp.mean(load_entropy),
        "mean_load_cv": jnp.mean(load_cv),
        "std_load_cv": jnp.std(load_cv),
        "mean_load_entropy": jnp.mean(load_entropy),
        "min_load_entropy": jnp.min(load_entropy),
        "dead_experts_count": jnp.sum(counts <= 0.0),
        "experts_active_mean": jnp.mean(jnp.sum(counts > 0.0, axis=-1).astype(jnp.float32)),
        "mean_importance_cv": jnp.mean(importance_cv),
        "mean_importance_entropy": jnp.mean(importance_entropy),
    }


def _row_cv(values: Any) -> Any:
    means = jnp.mean(values, axis=-1)
    return jnp.where(means > 0.0, jnp.std(values, axis=-1) / means * 100.0, 0.0)


def _row_entropy(values: Any) -> Any:
    totals = jnp.sum(values, axis=-1, keepdims=True)
    probabilities = values / jnp.maximum(totals, jnp.asarray(1e-6, dtype=jnp.float32))
    safe_probabilities = jnp.where(probabilities > 0.0, probabilities, 1.0)
    return -jnp.sum(jnp.where(probabilities > 0.0, probabilities * jnp.log(safe_probabilities), 0.0), axis=-1)


def _tree_l2_norm(tree: Any):
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    total = sum(jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32))) for leaf in leaves)
    return jnp.sqrt(total)


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in jnp.shape(value))


def _dtype(value: Any) -> Any:
    return jnp.asarray(value).dtype


def _is_integer_dtype(value: Any) -> bool:
    return bool(jnp.issubdtype(_dtype(value), jnp.integer))
