"""Causal LM loss and evaluation step boundaries."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from jaxtitan.batch import Batch
from jaxtitan.errors import ContractError
from jaxtitan.metrics import EvalMetrics
from jaxtitan.mesh import ShardingPlan, replicated_shardings_like
from jaxtitan.models import apply_model


@dataclass(frozen=True, slots=True)
class LossOutput:
    """Masked causal LM loss numerator and denominator."""

    loss: Any
    loss_sum: Any
    token_count: Any


def causal_lm_loss(logits: Any, target_ids: Any, loss_mask: Any) -> LossOutput:
    """Compute masked next-token cross entropy from aligned logits and targets."""

    _validate_loss_shapes(logits, target_ids, loss_mask)
    log_probs = jax.nn.log_softmax(jnp.asarray(logits, dtype=jnp.float32), axis=-1)
    target_ids = jnp.asarray(target_ids, dtype=jnp.int32)
    valid = jnp.asarray(loss_mask, dtype=jnp.bool_)
    per_token_loss = -jnp.take_along_axis(log_probs, target_ids[..., None], axis=-1)[..., 0]
    loss_sum = jnp.where(valid, per_token_loss, 0.0).sum()
    token_count = valid.sum()
    return LossOutput(loss=loss_sum / token_count, loss_sum=loss_sum, token_count=token_count)


def eval_step(graph: Any, state: Any, batch: Batch) -> EvalMetrics:
    """Run one compiled eval step for a model graph/state and a Batch."""

    return make_eval_step(graph)(state, batch)


def make_eval_step(
    graph: Any,
    *,
    sharding: ShardingPlan | None = None,
    state_template: Any | None = None,
) -> Callable[[Any, Batch], EvalMetrics]:
    """Create a compiled eval callable bound to a static model graph."""

    if sharding is not None and state_template is None:
        raise ContractError("state_template is required when compiling eval step with explicit shardings")
    state_sharding = None if sharding is None else replicated_shardings_like(state_template, sharding)
    in_shardings = None
    out_shardings = None
    if sharding is not None:
        in_shardings = (
            state_sharding,
            sharding.batch.input_ids,
            sharding.batch.target_ids,
            sharding.batch.loss_mask,
        )
        out_shardings = (sharding.metrics, sharding.metrics)

    def _compiled_impl(state: Any, input_ids: Any, target_ids: Any, loss_mask: Any) -> tuple[Any, Any]:
        logits = apply_model(graph, state, input_ids)
        loss = causal_lm_loss(logits, target_ids, loss_mask)
        return loss.loss_sum, loss.token_count

    _compiled = jax.jit(_compiled_impl, in_shardings=in_shardings, out_shardings=out_shardings)

    def _eval(state: Any, batch: Batch) -> EvalMetrics:
        _validate_batch(batch)
        loss_sum, token_count = _compiled(state, batch.input_ids, batch.target_ids, batch.loss_mask)
        return EvalMetrics(
            loss_sum=loss_sum,
            token_count=token_count,
            num_batches=jnp.asarray(1, dtype=token_count.dtype),
            byte_count=None,
        )

    return _eval


def _validate_batch(batch: Batch) -> None:
    _validate_rank2(batch.input_ids, "batch.input_ids")
    _validate_rank2(batch.target_ids, "batch.target_ids")
    _validate_rank2(batch.loss_mask, "batch.loss_mask")
    input_shape = _shape(batch.input_ids)
    if _shape(batch.target_ids) != input_shape:
        raise ContractError(f"batch.target_ids shape {_shape(batch.target_ids)} must equal input_ids shape {input_shape}")
    if _shape(batch.loss_mask) != input_shape:
        raise ContractError(f"batch.loss_mask shape {_shape(batch.loss_mask)} must equal input_ids shape {input_shape}")


def _validate_loss_shapes(logits: Any, target_ids: Any, loss_mask: Any) -> None:
    if len(_shape(logits)) != 3:
        raise ContractError(f"logits must have shape [batch, seq, vocab], got {_shape(logits)}")
    _validate_rank2(target_ids, "target_ids")
    _validate_rank2(loss_mask, "loss_mask")
    expected = _shape(logits)[:2]
    if _shape(target_ids) != expected:
        raise ContractError(f"target_ids shape {_shape(target_ids)} must equal logits batch/seq shape {expected}")
    if _shape(loss_mask) != expected:
        raise ContractError(f"loss_mask shape {_shape(loss_mask)} must equal logits batch/seq shape {expected}")


def _validate_rank2(value: Any, name: str) -> None:
    if len(_shape(value)) != 2:
        raise ContractError(f"{name} must have rank 2, got shape {_shape(value)}")


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in jnp.shape(value))
