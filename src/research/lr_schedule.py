"""
Learning rate schedule helpers.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax.numpy as jnp

from research.config import TrainConfig


def ratio_to_steps(total_steps: int, ratio: float) -> int:
    if ratio == 0.0:
        return 0
    return max(1, int(round(total_steps * ratio)))


def build_lr_schedule(train_config: TrainConfig) -> Callable[[int], jnp.ndarray]:
    if train_config.steps <= 0:
        raise ValueError(f"train.steps must be positive, got {train_config.steps}")

    schedule = train_config.lr_schedule
    warmup_steps = ratio_to_steps(train_config.steps, schedule.warmup_ratio)
    min_lr = train_config.lr * schedule.min_lr_ratio

    if schedule.type == "cosine":
        return _cosine_schedule(
            peak_lr=train_config.lr,
            min_lr=min_lr,
            total_steps=train_config.steps,
            warmup_steps=warmup_steps,
        )

    if schedule.type == "wsd":
        stable_steps = ratio_to_steps(train_config.steps, schedule.stable_ratio)
        if warmup_steps + stable_steps >= train_config.steps:
            raise ValueError(
                "WSD schedule requires warmup_ratio + stable_ratio to leave at "
                f"least one decay step; got warmup_steps={warmup_steps}, "
                f"stable_steps={stable_steps}, train.steps={train_config.steps}"
            )
        return _wsd_schedule(
            peak_lr=train_config.lr,
            min_lr=min_lr,
            total_steps=train_config.steps,
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
        )

    raise ValueError(f"Unknown lr schedule type: {schedule.type}")


def describe_lr_schedule(train_config: TrainConfig) -> str:
    schedule = train_config.lr_schedule
    if schedule.type == "cosine":
        warmup_steps = ratio_to_steps(train_config.steps, schedule.warmup_ratio)
        return (
            f"cosine warmup_ratio={schedule.warmup_ratio:g} "
            f"warmup_steps={warmup_steps} min_lr_ratio={schedule.min_lr_ratio:g}"
        )

    stable_steps = ratio_to_steps(train_config.steps, schedule.stable_ratio)
    warmup_steps = ratio_to_steps(train_config.steps, schedule.warmup_ratio)
    decay_steps = train_config.steps - warmup_steps - stable_steps
    return (
        f"wsd warmup_ratio={schedule.warmup_ratio:g} warmup_steps={warmup_steps} "
        f"stable_ratio={schedule.stable_ratio:g} stable_steps={stable_steps} "
        f"decay_steps={decay_steps} min_lr_ratio={schedule.min_lr_ratio:g}"
    )


def _linear_warmup(count, *, peak_lr: float, warmup_steps: int):
    if warmup_steps == 0:
        return jnp.asarray(peak_lr, dtype=jnp.float32)
    return peak_lr * jnp.minimum((count + 1.0) / warmup_steps, 1.0)


def _cosine_decay(count, *, peak_lr: float, min_lr: float, decay_steps: int):
    if decay_steps <= 0:
        return jnp.asarray(min_lr, dtype=jnp.float32)
    if decay_steps == 1:
        progress = jnp.asarray(1.0, dtype=jnp.float32)
    else:
        progress = jnp.clip(count / (decay_steps - 1), 0.0, 1.0)
    cosine = 0.5 * (1.0 + jnp.cos(jnp.asarray(math.pi, dtype=jnp.float32) * progress))
    return min_lr + (peak_lr - min_lr) * cosine


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
            jnp.where(count < decay_start, peak_lr, decay_lr),
        )

    return schedule
