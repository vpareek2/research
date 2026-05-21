"""Structured model forward outputs."""

from typing import Any

from flax import struct


@struct.dataclass
class AuxLoss:
    """Auxiliary objective term emitted by a model."""

    name: str = struct.field(pytree_node=False)
    value: Any
    weight: Any = 1.0


@struct.dataclass
class AuxMetric:
    """Auxiliary scalar metric emitted by a model."""

    name: str = struct.field(pytree_node=False)
    value: Any


@struct.dataclass
class ModelOutput:
    """Full-sequence model output consumed by train and eval steps."""

    logits: Any
    aux_losses: tuple[AuxLoss, ...] = ()
    aux_metrics: tuple[AuxMetric, ...] = ()


def ensure_model_output(value: Any) -> ModelOutput:
    """Normalize legacy logits-only model outputs into ModelOutput."""

    if isinstance(value, ModelOutput):
        return value
    return ModelOutput(logits=value)
