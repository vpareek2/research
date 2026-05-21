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
    objective: ArrayScalar | None = None
    aux_loss: ArrayScalar | None = None
    z_loss: ArrayScalar | None = None
    moe_aux_loss: ArrayScalar | None = None
    total_loss: ArrayScalar | None = None
    router_expert_counts: ArrayScalar | None = None
    router_importance: ArrayScalar | None = None
    router_max_vio: ArrayScalar | None = None
    router_load_min: ArrayScalar | None = None
    router_load_max: ArrayScalar | None = None
    router_load_entropy: ArrayScalar | None = None
    router_mean_load_cv: ArrayScalar | None = None
    router_std_load_cv: ArrayScalar | None = None
    router_mean_load_entropy: ArrayScalar | None = None
    router_min_load_entropy: ArrayScalar | None = None
    router_dead_experts_count: ArrayScalar | None = None
    router_experts_active_mean: ArrayScalar | None = None
    router_mean_importance_cv: ArrayScalar | None = None
    router_mean_importance_entropy: ArrayScalar | None = None
    smebu_bias_norm: ArrayScalar | None = None
    smebu_momentum_norm: ArrayScalar | None = None
    optimizer_group_specs: Any = ()
    optimizer_group_grad_norms: ArrayScalar | None = None
    optimizer_group_update_norms: ArrayScalar | None = None
    optimizer_group_param_norms: ArrayScalar | None = None
    aux_metrics: Any = ()
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
