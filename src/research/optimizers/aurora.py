"""
Aurora matrix optimizer directions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from research.config import OptimizerConfig
from research.optimizers.mixed import polar


def aurora_direction(matrix, config: OptimizerConfig):
    if matrix.ndim != 2:
        raise ValueError(f"Aurora expects a matrix update, got shape={matrix.shape}")
    if matrix.shape[0] == matrix.shape[1]:
        return polar(matrix, config)
    return aurora_balanced_polar(matrix, config)


def aurora_balanced_polar(matrix, config: OptimizerConfig):
    aurora = config.aurora
    x = matrix.astype(jnp.float32)
    transposed = x.shape[0] < x.shape[1]
    if transposed:
        x = x.T
    rows, cols = x.shape
    target_row_sq = cols / rows
    row_norm = jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), aurora.eps)
    diagonal = 1.0 / row_norm
    update = polar(diagonal * x, config)
    for _ in range(1, aurora.pp_iterations):
        row_sq = jnp.maximum(jnp.sum(jnp.square(update.astype(jnp.float32)), axis=-1, keepdims=True), aurora.eps * aurora.eps)
        diagonal = diagonal * jnp.power(target_row_sq / row_sq, aurora.pp_beta)
        update = polar(diagonal * x, config)
    if transposed:
        update = update.T
    return update.astype(matrix.dtype)


def riemannian_aurora_direction(matrix, config: OptimizerConfig):
    if matrix.ndim != 2:
        raise ValueError(f"Riemannian Aurora expects a matrix update, got shape={matrix.shape}")
    if matrix.shape[0] == matrix.shape[1]:
        return polar(matrix, config)
    return riemannian_balanced_polar(matrix, config)


def riemannian_balanced_polar(matrix, config: OptimizerConfig):
    aurora = config.riemannian_aurora
    x = matrix.astype(jnp.float32)
    transposed = x.shape[0] < x.shape[1]
    if transposed:
        x = x.T
    rows, cols = x.shape
    target_row_sq = cols / rows
    target_row_norm = jnp.sqrt(target_row_sq)
    update = polar(x, config).astype(jnp.float32)

    for _ in range(aurora.outer_steps):
        utg = update.T @ x
        stiefel_correction = 0.5 * (utg + utg.T)
        rhs = jnp.sum(x * update, axis=-1) - jnp.sum((update @ stiefel_correction) * update, axis=-1)
        rhs = rhs - jnp.mean(rhs)
        multipliers = solve_row_norm_multipliers(update, target_row_sq, rhs, max_iter=aurora.cg_steps)
        multipliers = multipliers - jnp.mean(multipliers)
        tangent_correction = stiefel_correction - update.T @ (multipliers[:, None] * update)
        tangent = x - update @ tangent_correction - multipliers[:, None] * update
        candidate = update + aurora.riemannian_eta * tangent
        candidate = jnp.where(jnp.all(jnp.isfinite(candidate)), candidate, update)
        for _ in range(aurora.retraction_steps):
            row_norm = jnp.maximum(jnp.linalg.norm(candidate, axis=-1, keepdims=True), aurora.eps)
            candidate = candidate * (target_row_norm / row_norm)
            candidate = polar(candidate, config).astype(jnp.float32)
        update = candidate

    if transposed:
        update = update.T
    return update.astype(matrix.dtype)


def solve_row_norm_multipliers(update, target_row_sq, rhs, *, max_iter: int):
    row_sq_sq = jnp.square(jnp.sum(jnp.square(update), axis=-1))
    regularizer = jnp.maximum(jnp.max(row_sq_sq) - target_row_sq + 1e-3, 0.0)
    effective_target = target_row_sq + regularizer

    def matvec(vector):
        middle = update.T @ (vector[:, None] * update)
        return effective_target * vector - jnp.sum((update @ middle) * update, axis=-1)

    initial = (
        jnp.zeros_like(rhs),
        rhs,
        rhs,
        jnp.sum(jnp.square(rhs)),
        jnp.asarray(False),
    )
    rhs_norm = jnp.maximum(jnp.linalg.norm(rhs), 1e-12)

    def step(_, state):
        value, residual, direction, residual_sq, done = state
        product = matvec(direction)
        denom = jnp.sum(direction * product)
        active = (~done) & jnp.isfinite(denom) & (denom >= 1e-30)
        alpha = jnp.where(active, residual_sq / denom, 0.0)
        next_value = value + alpha * direction
        next_residual = residual - alpha * product
        next_residual_sq = jnp.sum(jnp.square(next_residual))
        converged = (~jnp.isfinite(next_residual_sq)) | (jnp.sqrt(next_residual_sq) < 1e-8 * rhs_norm)
        beta = next_residual_sq / jnp.maximum(residual_sq, 1e-30)
        next_direction = next_residual + beta * direction
        next_done = done | (~active) | converged
        value = jnp.where(active, next_value, value)
        residual = jnp.where(active, next_residual, residual)
        direction = jnp.where(active, next_direction, direction)
        residual_sq = jnp.where(active, next_residual_sq, residual_sq)
        return value, residual, direction, residual_sq, next_done

    value, _, _, _, _ = jax.lax.fori_loop(0, max_iter, step, initial)
    return jnp.where(jnp.all(jnp.isfinite(value)), value, jnp.zeros_like(value))
