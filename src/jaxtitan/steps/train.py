"""Training step boundary."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from jaxtitan.batch import Batch
from jaxtitan.errors import ContractError
from jaxtitan.metrics import StepMetrics
from jaxtitan.mesh import ShardingPlan, gradient_shardings_like, replicated_shardings_like
from jaxtitan.models import apply_model
from jaxtitan.optim import OptimizerBuildResult, OptimizerTransform
from jaxtitan.state import RngState, TrainState
from jaxtitan.steps.eval import causal_lm_loss


def initialize_train_state(
    model_state: Any,
    optimizer_transform: OptimizerTransform,
    seed: int,
    *,
    optimizer_init_model_state: Any | None = None,
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
) -> Callable[[TrainState, Batch], tuple[TrainState, StepMetrics]]:
    """Create a compiled train callable bound to a static graph and optimizer."""

    if sharding is not None and state_template is None:
        raise ContractError("state_template is required when compiling train step with explicit shardings")
    state_sharding = None if sharding is None else replicated_shardings_like(state_template, sharding)
    gradient_sharding = (
        None
        if sharding is None or sharding.parallelism.mode != "zero2"
        else gradient_shardings_like(state_template.model, sharding)
    )
    in_shardings = None
    out_shardings = None
    if sharding is not None:
        in_shardings = (
            state_sharding,
            sharding.batch.accumulated_input_ids,
            sharding.batch.accumulated_target_ids,
            sharding.batch.accumulated_loss_mask,
        )
        out_shardings = (
            state_sharding,
            sharding.metrics,
            sharding.metrics,
            sharding.metrics,
            sharding.metrics,
            sharding.metrics,
            sharding.metrics,
            sharding.metrics,
            sharding.metrics,
            sharding.metrics,
        )

    def _compiled_impl(
        state: TrainState,
        input_ids: Any,
        target_ids: Any,
        loss_mask: Any,
    ) -> tuple[TrainState, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
        grad_zero = jax.tree.map(jnp.zeros_like, state.model)

        def microbatch_grad(params: Any, micro_input_ids: Any, micro_target_ids: Any, micro_loss_mask: Any):
            def loss_sum_fn(loss_params: Any):
                logits = apply_model(graph, loss_params, micro_input_ids)
                loss = causal_lm_loss(logits, micro_target_ids, micro_loss_mask)
                return loss.loss_sum, loss.token_count

            (loss_sum, token_count), grads = jax.value_and_grad(loss_sum_fn, has_aux=True)(params)
            return loss_sum, token_count, grads

        def accumulate(carry: tuple[Any, Any, Any], micro: tuple[Any, Any, Any]):
            grad_accum, loss_sum_accum, token_count_accum = carry
            micro_input_ids, micro_target_ids, micro_loss_mask = micro
            loss_sum, token_count, grads = microbatch_grad(
                state.model,
                micro_input_ids,
                micro_target_ids,
                micro_loss_mask,
            )
            return (
                jax.tree.map(lambda total, grad: total + grad, grad_accum, grads),
                loss_sum_accum + loss_sum,
                token_count_accum + token_count,
            ), (loss_sum, token_count)

        (grad_sum, loss_sum, token_count), (micro_loss_sums, micro_token_counts) = jax.lax.scan(
            accumulate,
            (
                grad_zero,
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            (input_ids, target_ids, loss_mask),
        )
        grad_denominator = jnp.asarray(token_count, dtype=jnp.float32)
        grads = jax.tree.map(lambda grad: grad / grad_denominator, grad_sum)
        if gradient_sharding is not None:
            grads = _constrain_like(grads, gradient_sharding)
        updates, next_opt_state = optimizer.transform.update(grads, state.opt_state, params=state.model)
        if gradient_sharding is not None:
            updates = _constrain_like(updates, gradient_sharding)
        next_model = jax.tree.map(lambda param, update: param + update, state.model, updates)
        next_step = state.step + jnp.asarray(1, dtype=state.step.dtype)
        next_tokens_seen = state.tokens_seen + token_count.astype(state.tokens_seen.dtype)
        next_state = state.replace(
            step=next_step,
            tokens_seen=next_tokens_seen,
            model=next_model,
            opt_state=next_opt_state,
        )
        lr = optimizer.schedule(state.step)
        grad_norm = _tree_l2_norm(grads)
        param_norm = _tree_l2_norm(next_model)
        update_norm = _tree_l2_norm(updates)
        micro_losses = micro_loss_sums / micro_token_counts.astype(jnp.float32)
        microbatch_loss_mean = jnp.mean(micro_losses)
        microbatch_loss_max = jnp.max(micro_losses)
        batch_het = microbatch_loss_max - microbatch_loss_mean
        return (
            next_state,
            loss_sum,
            token_count,
            lr,
            grad_norm,
            param_norm,
            update_norm,
            microbatch_loss_mean,
            microbatch_loss_max,
            batch_het,
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
            microbatch_loss_mean,
            microbatch_loss_max,
            batch_het,
        ) = _compiled(
            state,
            _ensure_accumulation_axis(batch.input_ids),
            _ensure_accumulation_axis(batch.target_ids),
            _ensure_accumulation_axis(batch.loss_mask),
        )
        metrics = StepMetrics(
            loss_sum=loss_sum,
            token_count=token_count,
            lr=lr,
            grad_norm=grad_norm,
            param_norm=param_norm,
            update_norm=update_norm,
            overflow=None,
            microbatch_loss_mean=microbatch_loss_mean,
            microbatch_loss_max=microbatch_loss_max,
            batch_het=batch_het,
        )
        return next_state, metrics

    return _train


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
