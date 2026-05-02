"""
Reusable language-model evaluation helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Iterator

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from research.data import Batch
from research.distributed import DistributedContext, shard_batch
from research.model import Model


@dataclass(frozen=True)
class LossEvalResult:
    loss: float
    ppl: float
    bpb: float
    bytes: int
    eval_steps: int
    examples: int
    tokens: int
    elapsed_sec: float

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens / self.elapsed_sec if self.elapsed_sec > 0.0 else 0.0

    def to_dict(self) -> dict[str, float | int]:
        data = asdict(self)
        data["tokens_per_sec"] = self.tokens_per_sec
        return data


def token_losses(model: Model, input_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :].astype(model.loss_dtype)
    shift_labels = input_ids[:, 1:]
    return optax.softmax_cross_entropy_with_integer_labels(
        shift_logits,
        shift_labels,
    )


def loss(model: Model, input_ids: jax.Array) -> jax.Array:
    return token_losses(model, input_ids).mean()


def bpb_from_losses(losses: jax.Array, target_ids: jax.Array, token_bytes: jax.Array) -> tuple[jax.Array, jax.Array]:
    byte_counts = token_bytes[target_ids]
    valid = byte_counts > 0
    total_nats = jnp.where(valid, losses, 0.0).sum()
    total_bytes = byte_counts.sum()
    return total_nats / (math.log(2) * total_bytes), total_bytes


def loss_with_bpb(model: Model, input_ids: jax.Array, token_bytes: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
    losses = token_losses(model, input_ids)
    bpb, byte_count = bpb_from_losses(losses, input_ids[:, 1:], token_bytes)
    return losses.mean(), {"bpb": bpb, "bytes": byte_count}


@nnx.jit
def eval_step(model: Model, input_ids: jax.Array) -> jax.Array:
    return loss(model, input_ids)


@nnx.jit
def eval_step_sums(model: Model, input_ids: jax.Array, token_bytes: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    losses = token_losses(model, input_ids)
    target_ids = input_ids[:, 1:]
    byte_counts = token_bytes[target_ids]
    valid = byte_counts > 0
    total_nats = jnp.where(valid, losses, 0.0).sum()
    total_bytes = byte_counts.sum()
    return losses.sum(), jnp.asarray(losses.size), total_nats, total_bytes


def evaluate_loss(
    model: Model,
    val_iter: Iterator[Batch],
    eval_steps: int,
    distributed: DistributedContext,
    *,
    tokens_per_example: int,
    token_bytes: jax.Array,
) -> LossEvalResult:
    if eval_steps <= 0:
        raise ValueError(f"eval_steps must be positive, got {eval_steps}")

    start = time.perf_counter()
    loss_sums = []
    loss_counts = []
    nats_sums = []
    byte_sums = []
    for _ in range(eval_steps):
        input_ids = shard_batch(next(val_iter), distributed)["input_ids"]
        loss_sum, loss_count, nats_sum, byte_sum = eval_step_sums(model, input_ids, token_bytes)
        loss_sums.append(loss_sum)
        loss_counts.append(loss_count)
        nats_sums.append(nats_sum)
        byte_sums.append(byte_sum)

    mean_loss = jnp.sum(jnp.asarray(loss_sums)) / jnp.sum(jnp.asarray(loss_counts))
    total_nats = jnp.sum(jnp.asarray(nats_sums))
    total_bytes = jnp.sum(jnp.asarray(byte_sums))
    bpb = total_nats / (math.log(2) * total_bytes)
    loss_value = float(jax.device_get(mean_loss))
    ppl = float(jax.device_get(jnp.exp(mean_loss)))
    bpb_value = float(jax.device_get(bpb))
    byte_count = int(jax.device_get(total_bytes))
    elapsed_sec = time.perf_counter() - start
    examples = eval_steps * distributed.global_batch_size
    tokens = examples * tokens_per_example

    return LossEvalResult(
        loss=loss_value,
        ppl=ppl,
        bpb=bpb_value,
        bytes=byte_count,
        eval_steps=eval_steps,
        examples=examples,
        tokens=tokens,
        elapsed_sec=elapsed_sec,
    )


def evaluate_domain_losses(
    model: Model,
    domain_iters: dict[str, Iterator[Batch]],
    eval_steps: int,
    distributed: DistributedContext,
    *,
    tokens_per_example: int,
    token_bytes: jax.Array,
) -> dict[str, LossEvalResult]:
    return {
        name: evaluate_loss(
            model,
            domain_iter,
            eval_steps,
            distributed,
            tokens_per_example=tokens_per_example,
            token_bytes=token_bytes,
        )
        for name, domain_iter in domain_iters.items()
    }
