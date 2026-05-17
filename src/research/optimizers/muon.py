"""
Muon matrix optimizer direction.
"""

from __future__ import annotations

from research.config import OptimizerConfig
from research.optimizers.mixed import matrix_momentum_direction, polar


def muon_direction(grad, mu, count, config: OptimizerConfig):
    muon = config.muon
    direction = matrix_momentum_direction(grad, mu, count, beta=muon.beta, nesterov=muon.nesterov)
    return polar(direction, config)
