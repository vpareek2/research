"""Metric contracts returned by compiled steps."""

from dataclasses import dataclass
from typing import Any

ArrayScalar = Any


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Numerator/denominator training metrics.

    Host code derives presentation metrics such as mean loss and tokens/sec.
    """

    loss_sum: ArrayScalar
    token_count: ArrayScalar
    lr: ArrayScalar
    grad_norm: ArrayScalar | None = None
    param_norm: ArrayScalar | None = None
    update_norm: ArrayScalar | None = None
    overflow: ArrayScalar | None = None
    microbatch_loss_mean: ArrayScalar | None = None
    microbatch_loss_max: ArrayScalar | None = None
    batch_het: ArrayScalar | None = None


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    """Numerator/denominator evaluation metrics."""

    loss_sum: ArrayScalar
    token_count: ArrayScalar
    num_batches: ArrayScalar
    byte_count: ArrayScalar | None = None
