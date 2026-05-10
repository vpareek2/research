"""
Optimizer factory.
"""

from __future__ import annotations

from flax import nnx

from research.config import ModelConfig, OptimizerConfig
from research.optimizers.mixed import mixed_matrix_adamw
from research.optimizers.routing import classify_param_tree


def build_optimizer(model: nnx.Module, model_config: ModelConfig, optimizer_config: OptimizerConfig, learning_rate):
    params = nnx.state(model, nnx.Param)
    labels = classify_param_tree(params, model_config)
    tx = mixed_matrix_adamw(labels, optimizer_config, learning_rate)
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def describe_optimizer(config: OptimizerConfig) -> str:
    return f"{config.name} peak_lr={config.lr:g} weight_decay={config.weight_decay:g}"
