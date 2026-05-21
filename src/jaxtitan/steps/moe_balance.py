"""MoE balancing state and update helpers."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from jaxtitan.errors import ContractError
from jaxtitan.specs.model import MoeBalanceSpec


@struct.dataclass
class MoeLayerBalanceState:
    """Non-gradient balancing state for one MoE layer."""

    path: tuple[str, ...] = struct.field(pytree_node=False)
    momentum: Any


@struct.dataclass
class MoeBalanceState:
    """Non-gradient MoE balancing state carried by TrainState."""

    name: str = struct.field(pytree_node=False)
    load_lr: float = struct.field(pytree_node=False)
    momentum_factor: float = struct.field(pytree_node=False)
    clamp: float = struct.field(pytree_node=False)
    layers: tuple[MoeLayerBalanceState, ...]


@struct.dataclass
class MoeBalanceMetrics:
    """Scalar diagnostics produced by a balance update."""

    max_vio: Any
    load_min: Any
    load_max: Any
    load_entropy: Any
    bias_norm: Any
    momentum_norm: Any


def initialize_moe_balance_state(model_state: Any, balance: MoeBalanceSpec | None) -> MoeBalanceState | None:
    """Initialize non-gradient MoE balancing state from model expert-bias leaves."""

    if balance is None or balance.name == "none":
        return None
    if balance.name != "smebu":
        raise ContractError(f"unsupported MoE balance policy {balance.name!r}")
    layers = []
    for path, leaf in _expert_bias_leaves(model_state):
        layers.append(MoeLayerBalanceState(path=path, momentum=jnp.zeros_like(leaf, dtype=jnp.float32)))
    if not layers:
        raise ContractError("SMEBU balance requires at least one MoE expert_bias parameter")
    return MoeBalanceState(
        name=balance.name,
        load_lr=float(balance.load_lr),
        momentum_factor=float(balance.momentum),
        clamp=float(balance.clamp),
        layers=tuple(layers),
    )


def zero_router_counts(balance: MoeBalanceState | None) -> Any:
    """Return a zero count array matching the balance state layer/expert shape."""

    if balance is None:
        return jnp.zeros((0, 0), dtype=jnp.float32)
    return jnp.stack([jnp.zeros_like(layer.momentum, dtype=jnp.float32) for layer in balance.layers])


def router_counts_from_stats(router_stats: tuple[Any, ...], balance: MoeBalanceState | None) -> Any:
    """Return layer-major expert counts from model router stats."""

    if not router_stats:
        return jnp.zeros((0, 0), dtype=jnp.float32)
    if balance is not None and len(router_stats) != len(balance.layers):
        raise ContractError(
            f"MoE balance expected {len(balance.layers)} router stat entries, got {len(router_stats)}"
        )
    return jnp.stack([jnp.asarray(stats.expert_counts, dtype=jnp.float32) for stats in router_stats])


def router_importance_from_stats(router_stats: tuple[Any, ...], balance: MoeBalanceState | None) -> Any:
    """Return layer-major selected-router-weight sums from model router stats."""

    if not router_stats:
        return jnp.zeros((0, 0), dtype=jnp.float32)
    if balance is not None and len(router_stats) != len(balance.layers):
        raise ContractError(
            f"MoE balance expected {len(balance.layers)} router stat entries, got {len(router_stats)}"
        )
    return jnp.stack([jnp.asarray(stats.importance, dtype=jnp.float32) for stats in router_stats])


def apply_moe_balance_update(
    model_state: Any,
    balance: MoeBalanceState | None,
    router_counts: Any,
) -> tuple[Any, MoeBalanceState | None, MoeBalanceMetrics]:
    """Apply one non-gradient balancing update to model expert bias leaves."""

    zero = jnp.asarray(0.0, dtype=jnp.float32)
    if balance is None:
        return model_state, None, MoeBalanceMetrics(zero, zero, zero, zero, zero, zero)
    if balance.name != "smebu":
        raise ContractError(f"unsupported MoE balance policy {balance.name!r}")
    if router_counts.shape != (len(balance.layers), balance.layers[0].momentum.shape[0]):
        raise ContractError(
            "router count shape must match MoE balance state, "
            f"got {router_counts.shape}"
        )

    updates_by_path = {}
    next_layers = []
    bias_norm_terms = []
    momentum_norm_terms = []
    max_vios = []
    load_mins = []
    load_maxes = []
    entropies = []
    for layer, counts in zip(balance.layers, router_counts, strict=True):
        count_sum = jnp.sum(counts)
        mean = count_sum / jnp.asarray(counts.shape[0], dtype=jnp.float32)
        violation = (mean - counts) / jnp.maximum(mean, jnp.asarray(1e-6, dtype=jnp.float32))
        delta = jnp.asarray(balance.load_lr, dtype=jnp.float32) * jnp.tanh(
            jnp.asarray(balance.clamp, dtype=jnp.float32) * violation
        )
        delta = delta - jnp.mean(delta)
        next_momentum = (
            jnp.asarray(balance.momentum_factor, dtype=jnp.float32) * layer.momentum
            + (1.0 - jnp.asarray(balance.momentum_factor, dtype=jnp.float32)) * delta
        )
        updates_by_path[layer.path] = next_momentum
        next_layers.append(layer.replace(momentum=next_momentum))

        load_min = jnp.min(counts)
        load_max = jnp.max(counts)
        probabilities = counts / jnp.maximum(count_sum, jnp.asarray(1e-6, dtype=jnp.float32))
        safe_probabilities = jnp.where(probabilities > 0, probabilities, 1.0)
        entropy = -jnp.sum(jnp.where(probabilities > 0, probabilities * jnp.log(safe_probabilities), 0.0))
        max_vio = (load_max - mean) / jnp.maximum(mean, jnp.asarray(1e-6, dtype=jnp.float32))
        load_mins.append(load_min)
        load_maxes.append(load_max)
        max_vios.append(max_vio)
        entropies.append(entropy)
        momentum_norm_terms.append(jnp.sum(jnp.square(next_momentum)))

    next_model = _update_expert_biases(model_state, updates_by_path, bias_norm_terms)
    next_balance = balance.replace(layers=tuple(next_layers))
    bias_norm = jnp.sqrt(sum(bias_norm_terms, zero))
    momentum_norm = jnp.sqrt(sum(momentum_norm_terms, zero))
    metrics = MoeBalanceMetrics(
        max_vio=jnp.max(jnp.stack(max_vios)),
        load_min=jnp.min(jnp.stack(load_mins)),
        load_max=jnp.max(jnp.stack(load_maxes)),
        load_entropy=jnp.mean(jnp.stack(entropies)),
        bias_norm=bias_norm,
        momentum_norm=momentum_norm,
    )
    return next_model, next_balance, metrics


def _expert_bias_leaves(model_state: Any) -> list[tuple[tuple[str, ...], Any]]:
    leaves = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(model_state)[0]:
        metadata_path = _metadata_path_from_jax_path(path)
        if metadata_path and metadata_path[-1] == "expert_bias":
            leaves.append((metadata_path, leaf))
    return sorted(leaves, key=lambda item: item[0])


def _update_expert_biases(model_state: Any, updates_by_path: dict[tuple[str, ...], Any], bias_norm_terms: list[Any]) -> Any:
    def update(path, leaf):
        metadata_path = _metadata_path_from_jax_path(path)
        update_value = updates_by_path.get(metadata_path)
        if update_value is None:
            return leaf
        next_leaf = jnp.asarray(leaf, dtype=jnp.float32) + update_value
        bias_norm_terms.append(jnp.sum(jnp.square(next_leaf)))
        return next_leaf.astype(jnp.asarray(leaf).dtype)

    return jax.tree_util.tree_map_with_path(update, model_state)


def _metadata_path_from_jax_path(path) -> tuple[str, ...]:
    parts = []
    for key in path:
        name = getattr(key, "key", None)
        if name is None:
            name = getattr(key, "name", None)
        if name == "value":
            continue
        parts.append(str(name))
    return tuple(parts)
