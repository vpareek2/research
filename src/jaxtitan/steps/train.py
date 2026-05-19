"""Training step boundary."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from jaxtitan.batch import Batch
from jaxtitan.metrics import StepMetrics
from jaxtitan.models import apply_model
from jaxtitan.optim import OptimizerBuildResult, OptimizerTransform
from jaxtitan.state import RngState, TrainState
from jaxtitan.steps.eval import _validate_batch, causal_lm_loss


def initialize_train_state(model_state: Any, optimizer_transform: OptimizerTransform, seed: int) -> TrainState:
    """Initialize explicit train state from model state and an optimizer transform."""

    train_key, data_key, eval_key, sample_key = jax.random.split(jax.random.key(seed), 4)
    return TrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        tokens_seen=jnp.asarray(0, dtype=jnp.uint32),
        model=model_state,
        opt_state=optimizer_transform.init(model_state),
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
) -> Callable[[TrainState, Batch], tuple[TrainState, StepMetrics]]:
    """Create a compiled train callable bound to a static graph and optimizer."""

    @jax.jit
    def _compiled(
        state: TrainState,
        input_ids: Any,
        target_ids: Any,
        loss_mask: Any,
    ) -> tuple[TrainState, Any, Any, Any, Any, Any, Any]:
        def loss_fn(params):
            logits = apply_model(graph, params, input_ids)
            loss = causal_lm_loss(logits, target_ids, loss_mask)
            return loss.loss, (loss.loss_sum, loss.token_count)

        (_loss, (loss_sum, token_count)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.model)
        updates, next_opt_state = optimizer.transform.update(grads, state.opt_state, params=state.model)
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
        return next_state, loss_sum, token_count, lr, grad_norm, param_norm, update_norm

    def _train(state: TrainState, batch: Batch) -> tuple[TrainState, StepMetrics]:
        _validate_batch(batch)
        (
            next_state,
            loss_sum,
            token_count,
            lr,
            grad_norm,
            param_norm,
            update_norm,
        ) = _compiled(
            state,
            batch.input_ids,
            batch.target_ids,
            batch.loss_mask,
        )
        metrics = StepMetrics(
            loss_sum=loss_sum,
            token_count=token_count,
            lr=lr,
            grad_norm=grad_norm,
            param_norm=param_norm,
            update_norm=update_norm,
            overflow=None,
        )
        return next_state, metrics

    return _train


def _tree_l2_norm(tree: Any):
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    total = sum(jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32))) for leaf in leaves)
    return jnp.sqrt(total)
