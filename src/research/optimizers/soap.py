"""
SOAP optimizer.

This follows the reference implementation's Adam-in-Shampoo-eigenbasis update
while keeping JAX optimizer state shapes static.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from research.config import OptimizerConfig, SOAPOptimizerConfig


@jax.tree_util.register_pytree_node_class
class SOAPAxes:
    def __init__(self, axes: tuple[jax.Array, ...]):
        self.axes = axes

    def tree_flatten(self):
        return self.axes, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(tuple(children))


class SOAPState(NamedTuple):
    count: jax.Array
    initialized: jax.Array
    exp_avg: object
    exp_avg_sq: object
    gg: object
    q: object


def soap(labels, config: OptimizerConfig, learning_rate: Callable[[int], jax.Array]) -> optax.GradientTransformation:
    del labels

    def init_fn(params):
        zeros = jax.tree.map(jnp.zeros_like, params)
        gg = jax.tree.map(lambda p: init_preconditioner_axes(p, config.soap), params)
        q = jax.tree.map(lambda p: init_q_axes(p, config.soap), params)
        return SOAPState(
            count=jnp.zeros([], dtype=jnp.int32),
            initialized=jnp.asarray(False),
            exp_avg=zeros,
            exp_avg_sq=zeros,
            gg=gg,
            q=q,
        )

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("soap requires params for decoupled weight decay")
        lr = learning_rate(state.count)
        grad_leaves, treedef = jax.tree_util.tree_flatten(updates)
        param_leaves, _ = jax.tree_util.tree_flatten(params)
        exp_avg_leaves, _ = jax.tree_util.tree_flatten(state.exp_avg)
        exp_avg_sq_leaves, _ = jax.tree_util.tree_flatten(state.exp_avg_sq)
        gg_axis_leaves, gg_treedef = jax.tree_util.tree_flatten(state.gg)
        q_axis_leaves, q_treedef = jax.tree_util.tree_flatten(state.q)
        gg_leaves = group_axis_leaves(param_leaves, gg_axis_leaves)
        q_leaves = group_axis_leaves(param_leaves, q_axis_leaves)

        next_updates = []
        next_exp_avg = []
        next_exp_avg_sq = []
        next_gg_axes = []
        next_q_axes = []
        for grad, param, exp_avg, exp_avg_sq, gg, q in zip(
            grad_leaves,
            param_leaves,
            exp_avg_leaves,
            exp_avg_sq_leaves,
            gg_leaves,
            q_leaves,
            strict=True,
        ):
            leaf_update, leaf_exp_avg, leaf_exp_avg_sq, leaf_gg, leaf_q = soap_leaf_update(
                grad,
                param,
                exp_avg,
                exp_avg_sq,
                gg,
                q,
                count=state.count,
                initialized=state.initialized,
                lr=lr,
                optimizer_config=config,
            )
            next_updates.append(leaf_update)
            next_exp_avg.append(leaf_exp_avg)
            next_exp_avg_sq.append(leaf_exp_avg_sq)
            next_gg_axes.extend(leaf_gg.axes)
            next_q_axes.extend(leaf_q.axes)

        normal_step = state.initialized
        next_count = jnp.where(normal_step, state.count + jnp.asarray(1, dtype=jnp.int32), state.count)
        return (
            jax.tree_util.tree_unflatten(treedef, next_updates),
            SOAPState(
                count=next_count,
                initialized=jnp.asarray(True),
                exp_avg=jax.tree_util.tree_unflatten(treedef, next_exp_avg),
                exp_avg_sq=jax.tree_util.tree_unflatten(treedef, next_exp_avg_sq),
                gg=jax.tree_util.tree_unflatten(gg_treedef, next_gg_axes),
                q=jax.tree_util.tree_unflatten(q_treedef, next_q_axes),
            ),
        )

    return optax.GradientTransformation(init_fn, update_fn)


def is_soap_axes(value):
    return isinstance(value, SOAPAxes)


def group_axis_leaves(param_leaves, axis_leaves) -> list[SOAPAxes]:
    grouped = []
    index = 0
    for param in param_leaves:
        axis_count = max(param.ndim, 1)
        grouped.append(SOAPAxes(tuple(axis_leaves[index : index + axis_count])))
        index += axis_count
    if index != len(axis_leaves):
        raise ValueError("SOAP axis state does not match parameter tree")
    return grouped


def init_preconditioner_axes(param, config: SOAPOptimizerConfig) -> SOAPAxes:
    axes = []
    if param.ndim == 1:
        if config.precondition_1d and param.shape[0] <= config.max_precond_dim:
            axes.append(jnp.zeros((param.shape[0], param.shape[0]), dtype=jnp.float32))
        else:
            axes.append(skipped_axis())
    else:
        for dim in param.shape:
            if dim <= config.max_precond_dim:
                axes.append(jnp.zeros((dim, dim), dtype=jnp.float32))
            else:
                axes.append(skipped_axis())
    return SOAPAxes(tuple(axes))


def init_q_axes(param, config: SOAPOptimizerConfig) -> SOAPAxes:
    axes = []
    for gg in init_preconditioner_axes(param, config).axes:
        if is_skipped_axis(gg):
            axes.append(skipped_axis())
        else:
            axes.append(jnp.eye(gg.shape[0], dtype=jnp.float32))
    return SOAPAxes(tuple(axes))


def skipped_axis():
    return jnp.zeros((1, 2), dtype=jnp.float32)


def is_skipped_axis(axis):
    return axis.shape[0] != axis.shape[1]


def soap_leaf_update(
    grad,
    param,
    exp_avg,
    exp_avg_sq,
    gg: SOAPAxes,
    q: SOAPAxes,
    *,
    count,
    initialized,
    lr,
    optimizer_config: OptimizerConfig,
):
    config = optimizer_config.soap
    shampoo_beta = config.b2 if config.shampoo_beta < 0.0 else config.shampoo_beta
    gg_from_grad = update_preconditioner_stats(grad, gg, shampoo_beta=shampoo_beta)
    initial_q = get_orthogonal_matrix(gg_from_grad)

    step = count + jnp.asarray(1, dtype=jnp.int32)
    grad_projected = project(grad, q)
    next_exp_avg = config.b1 * exp_avg + (1.0 - config.b1) * grad_projected
    next_exp_avg_sq = config.b2 * exp_avg_sq + (1.0 - config.b2) * jnp.square(grad_projected)
    denom = jnp.sqrt(next_exp_avg_sq) + config.eps
    step_size = lr
    if config.correct_bias:
        bias_correction1 = 1.0 - config.b1**step
        bias_correction2 = 1.0 - config.b2**step
        step_size = step_size * jnp.sqrt(bias_correction2) / bias_correction1
    direction = project_back(next_exp_avg / denom, q)
    if config.normalize_grads:
        direction = direction / jnp.sqrt(jnp.mean(jnp.square(direction)) + 1e-30)
    param_update = -step_size * direction
    if optimizer_config.weight_decay > 0.0:
        param_update = (param + param_update) * (1.0 - lr * optimizer_config.weight_decay) - param

    refresh_q = (step % config.precondition_frequency) == 0
    exp_avg_original = project_back(next_exp_avg, q)
    refreshed_q, refreshed_exp_avg_sq = get_orthogonal_matrix_qr(
        gg_from_grad,
        q,
        next_exp_avg_sq,
        refresh=refresh_q,
    )
    refreshed_exp_avg = project(exp_avg_original, refreshed_q)

    zero_update = jnp.zeros_like(param)
    return (
        jnp.where(initialized, param_update, zero_update),
        jnp.where(initialized, refreshed_exp_avg, exp_avg),
        jnp.where(initialized, refreshed_exp_avg_sq, exp_avg_sq),
        gg_from_grad,
        tree_where_axes(initialized, refreshed_q, initial_q),
    )


def update_preconditioner_stats(grad, gg: SOAPAxes, *, shampoo_beta: float) -> SOAPAxes:
    axes = []
    for idx, current in enumerate(gg.axes):
        if is_skipped_axis(current):
            axes.append(current)
            continue
        contract_axes = tuple(axis for axis in range(grad.ndim) if axis != idx)
        outer = jnp.tensordot(grad.astype(jnp.float32), grad.astype(jnp.float32), axes=(contract_axes, contract_axes))
        axes.append(shampoo_beta * current + (1.0 - shampoo_beta) * outer)
    return SOAPAxes(tuple(axes))


def project(value, q: SOAPAxes):
    result = value
    for axis in q.axes:
        if is_skipped_axis(axis):
            result = jnp.moveaxis(result, 0, -1)
        else:
            result = jnp.tensordot(result, axis.astype(result.dtype), axes=([0], [0]))
    return result


def project_back(value, q: SOAPAxes):
    result = value
    for axis in q.axes:
        if is_skipped_axis(axis):
            result = jnp.moveaxis(result, 0, -1)
        else:
            result = jnp.tensordot(result, axis.astype(result.dtype), axes=([0], [1]))
    return result


def get_orthogonal_matrix(gg: SOAPAxes) -> SOAPAxes:
    axes = []
    for axis in gg.axes:
        if is_skipped_axis(axis):
            axes.append(axis)
            continue
        _, q = jnp.linalg.eigh(axis + 1e-30 * jnp.eye(axis.shape[0], dtype=axis.dtype))
        axes.append(jnp.flip(q, axis=1))
    return SOAPAxes(tuple(axes))


def get_orthogonal_matrix_qr(gg: SOAPAxes, q: SOAPAxes, exp_avg_sq, *, refresh):
    axes = []
    next_exp_avg_sq = exp_avg_sq
    for idx, (stat, basis) in enumerate(zip(gg.axes, q.axes, strict=True)):
        if is_skipped_axis(stat):
            axes.append(stat)
            continue
        estimated_eigenvalues = jnp.diag(basis.T @ stat @ basis)
        sort_idx = jnp.argsort(estimated_eigenvalues)[::-1]
        sorted_basis = basis[:, sort_idx]
        sorted_exp_avg_sq = jnp.take(next_exp_avg_sq, sort_idx, axis=idx)
        power_iter = stat @ sorted_basis
        refreshed_basis, _ = jnp.linalg.qr(power_iter)
        axes.append(jnp.where(refresh, refreshed_basis, basis))
        next_exp_avg_sq = jnp.where(refresh, sorted_exp_avg_sq, next_exp_avg_sq)
    return SOAPAxes(tuple(axes)), next_exp_avg_sq


def tree_where_axes(condition, if_true: SOAPAxes, if_false: SOAPAxes) -> SOAPAxes:
    return SOAPAxes(tuple(jnp.where(condition, true, false) for true, false in zip(if_true.axes, if_false.axes, strict=True)))
