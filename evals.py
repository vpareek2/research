"""
Reusable language-model evaluation helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Iterator

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from data import Batch
from distributed import DistributedContext, shard_batch
from model import Model


@dataclass(frozen=True)
class LossEvalResult:
    loss: float
    ppl: float
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


def loss(model: Model, input_ids: jax.Array) -> jax.Array:
    logits = model(input_ids)
    shift_logits = logits[:, :-1, :].astype(model.loss_dtype)
    shift_labels = input_ids[:, 1:]
    return optax.softmax_cross_entropy_with_integer_labels(
        shift_logits,
        shift_labels,
    ).mean()


@nnx.jit
def eval_step(model: Model, input_ids: jax.Array) -> jax.Array:
    return loss(model, input_ids)


def evaluate_loss(
    model: Model,
    val_iter: Iterator[Batch],
    eval_steps: int,
    distributed: DistributedContext,
    *,
    tokens_per_example: int,
) -> LossEvalResult:
    if eval_steps <= 0:
        raise ValueError(f"eval_steps must be positive, got {eval_steps}")

    start = time.perf_counter()
    losses = [
        eval_step(model, shard_batch(next(val_iter), distributed)["input_ids"])
        for _ in range(eval_steps)
    ]
    mean_loss = jnp.mean(jnp.asarray(losses))
    loss_value = float(jax.device_get(mean_loss))
    ppl = float(jax.device_get(jnp.exp(mean_loss)))
    elapsed_sec = time.perf_counter() - start
    examples = eval_steps * distributed.global_batch_size
    tokens = examples * tokens_per_example

    return LossEvalResult(
        loss=loss_value,
        ppl=ppl,
        eval_steps=eval_steps,
        examples=examples,
        tokens=tokens,
        elapsed_sec=elapsed_sec,
    )
