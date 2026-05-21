"""Sparse feed-forward components."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from jaxtitan.errors import ContractError
from jaxtitan.models.components.dtypes import dtype_from_name
from jaxtitan.models.output import AuxLoss, RouterStats
from jaxtitan.specs.model import ModelSpec, TrinityMoeSpec


@dataclass(frozen=True, slots=True)
class RouterOutput:
    """Selected expert ids and normalized routing weights."""

    expert_ids: jax.Array
    weights: jax.Array
    scores: jax.Array


class SigmoidTopKRouter(nnx.Module):
    """Deterministic sigmoid top-k router."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        dtype: Any,
        param_dtype: Any,
        rngs: nnx.Rngs,
        route_scale: float = 1.0,
        kernel_init: Any | None = None,
    ):
        if top_k <= 0 or num_experts <= 0 or top_k > num_experts:
            raise ContractError("router top_k must be positive and <= num_experts")
        if route_scale <= 0.0:
            raise ContractError(f"router route_scale must be positive, got {route_scale}")
        linear_kwargs = {} if kernel_init is None else {"kernel_init": kernel_init}
        self.proj = nnx.Linear(
            hidden_size,
            num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            **linear_kwargs,
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.route_scale = float(route_scale)

    def __call__(self, x: jax.Array, expert_bias: jax.Array | None = None) -> RouterOutput:
        logits = jnp.asarray(self.proj(x), dtype=jnp.float32)
        scores = jax.nn.sigmoid(logits)
        if expert_bias is None:
            expert_bias = jnp.zeros((self.num_experts,), dtype=jnp.float32)
        expert_bias = jnp.asarray(expert_bias, dtype=jnp.float32)
        if expert_bias.shape != (self.num_experts,):
            raise ContractError(f"expert_bias must have shape [{self.num_experts}], got {expert_bias.shape}")
        _biased_scores, expert_ids = jax.lax.top_k(scores + expert_bias, self.top_k)
        top_scores = jnp.take_along_axis(scores, expert_ids, axis=-1)
        denominator = jnp.sum(top_scores, axis=-1, keepdims=True) + jnp.asarray(1e-20, dtype=jnp.float32)
        weights = (top_scores / denominator) * jnp.asarray(self.route_scale, dtype=jnp.float32)
        return RouterOutput(
            expert_ids=jnp.asarray(expert_ids, dtype=jnp.int32),
            weights=jnp.asarray(weights, dtype=x.dtype),
            scores=scores,
        )


class ExpertSwiGLU(nnx.Module):
    """Selected-expert SwiGLU feed-forward block."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        *,
        dtype: Any,
        param_dtype: Any,
        rngs: nnx.Rngs,
        kernel_init: Any | None = None,
    ):
        initializer = nnx.initializers.lecun_normal() if kernel_init is None else kernel_init
        self.gate = nnx.Param(initializer(rngs.params(), (num_experts, hidden_size, intermediate_size), param_dtype))
        self.up = nnx.Param(initializer(rngs.params(), (num_experts, hidden_size, intermediate_size), param_dtype))
        self.down = nnx.Param(initializer(rngs.params(), (num_experts, intermediate_size, hidden_size), param_dtype))
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.dtype = dtype
        self.param_dtype = param_dtype

    def __call__(self, x: jax.Array, expert_ids: jax.Array, weights: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=self.dtype)
        expert_ids = jnp.asarray(expert_ids, dtype=jnp.int32)
        weights = jnp.asarray(weights, dtype=self.dtype)
        if expert_ids.shape != weights.shape:
            raise ContractError(f"expert_ids shape {expert_ids.shape} must equal weights shape {weights.shape}")
        if expert_ids.shape[:-1] != x.shape[:-1]:
            raise ContractError("expert routing shape must match input batch and sequence dimensions")

        gate = jnp.asarray(self.gate[...], dtype=self.dtype)[expert_ids]
        up = jnp.asarray(self.up[...], dtype=self.dtype)[expert_ids]
        down = jnp.asarray(self.down[...], dtype=self.dtype)[expert_ids]
        gate_x = jnp.einsum("...h,...khi->...ki", x, gate)
        up_x = jnp.einsum("...h,...khi->...ki", x, up)
        expert_hidden = jax.nn.silu(gate_x) * up_x
        expert_output = jnp.einsum("...ki,...kih->...kh", expert_hidden, down)
        return jnp.sum(expert_output * weights[..., None], axis=-2)


class SharedSwiGLU(nnx.Module):
    """Always-on shared expert path with ordinary dense FFN matrices."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        dtype: Any,
        param_dtype: Any,
        rngs: nnx.Rngs,
        kernel_init: Any | None = None,
    ):
        linear_kwargs = {} if kernel_init is None else {"kernel_init": kernel_init}
        self.gate = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            **linear_kwargs,
        )
        self.up = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            **linear_kwargs,
        )
        self.down = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            **linear_kwargs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))


class SparseMoE(nnx.Module):
    """Sparse selected-expert feed-forward module."""

    def __init__(self, spec: ModelSpec, moe: TrinityMoeSpec, rngs: nnx.Rngs, *, kernel_init: Any | None = None):
        if moe.expert_intermediate_size is None:
            raise ContractError("model.trinity.moe.expert_intermediate_size must be resolved before model construction")
        dtype = dtype_from_name(spec.compute_dtype)
        param_dtype = dtype_from_name(spec.param_dtype)
        self.router = SigmoidTopKRouter(
            spec.hidden_size,
            moe.num_experts,
            moe.top_k,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            route_scale=moe.route_scale,
            kernel_init=kernel_init,
        )
        self.expert_bias = nnx.Param(jnp.zeros((moe.num_experts,), dtype=jnp.float32))
        self.balance_name = moe.balance.name
        self.sequence_aux_loss_weight = float(moe.balance.sequence_aux_loss_weight)
        self.experts = ExpertSwiGLU(
            spec.hidden_size,
            moe.expert_intermediate_size,
            moe.num_experts,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=kernel_init,
        )
        if moe.num_shared_experts > 0:
            self.shared_experts = SharedSwiGLU(
                spec.hidden_size,
                moe.expert_intermediate_size * moe.num_shared_experts,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                kernel_init=kernel_init,
            )

    def route(self, x: jax.Array) -> RouterOutput:
        return self.router(x, expert_bias=jax.lax.stop_gradient(self.expert_bias[...]))

    def __call__(self, x: jax.Array) -> jax.Array:
        route = self.route(x)
        output = self.experts(x, route.expert_ids, route.weights)
        if hasattr(self, "shared_experts"):
            output = output + self.shared_experts(x)
        return output

    def forward_with_output(
        self,
        x: jax.Array,
        *,
        name: str,
        layer_index: int,
    ) -> tuple[jax.Array, tuple[AuxLoss, ...], tuple[RouterStats, ...]]:
        route = self.route(x)
        output = self.experts(x, route.expert_ids, route.weights)
        if hasattr(self, "shared_experts"):
            output = output + self.shared_experts(x)
        stats = _router_stats(
            name=name,
            layer_index=layer_index,
            expert_ids=route.expert_ids,
            weights=route.weights,
            num_experts=self.experts.num_experts,
        )
        aux_losses = ()
        if self.balance_name == "smebu" and self.sequence_aux_loss_weight > 0.0:
            aux_losses = (
                AuxLoss(
                    name="moe_sequence_balance",
                    value=_sequence_balance_loss(route.expert_ids, route.scores, top_k=self.router.top_k),
                    weight=jnp.asarray(self.sequence_aux_loss_weight, dtype=jnp.float32),
                ),
            )
        return output, aux_losses, (stats,)


def _router_stats(
    *,
    name: str,
    layer_index: int,
    expert_ids: jax.Array,
    weights: jax.Array,
    num_experts: int,
) -> RouterStats:
    selected = jax.nn.one_hot(expert_ids, num_experts, dtype=jnp.float32)
    reduce_axes = tuple(range(expert_ids.ndim))
    counts = jnp.sum(selected, axis=reduce_axes)
    importance = jnp.sum(selected * jnp.asarray(weights, dtype=jnp.float32)[..., None], axis=reduce_axes)
    total = jnp.sum(counts)
    mean = total / jnp.asarray(num_experts, dtype=jnp.float32)
    load_min = jnp.min(counts)
    load_max = jnp.max(counts)
    max_vio = (load_max - mean) / jnp.maximum(mean, jnp.asarray(1e-6, dtype=jnp.float32))
    probabilities = counts / jnp.maximum(total, jnp.asarray(1e-6, dtype=jnp.float32))
    safe_probabilities = jnp.where(probabilities > 0, probabilities, 1.0)
    entropy = -jnp.sum(jnp.where(probabilities > 0, probabilities * jnp.log(safe_probabilities), 0.0))
    return RouterStats(
        name=name,
        layer_index=layer_index,
        expert_counts=counts,
        importance=importance,
        total_assignments=total,
        load_min=load_min,
        load_max=load_max,
        load_mean=mean,
        max_vio=max_vio,
        load_entropy=entropy,
    )


def _sequence_balance_loss(expert_ids: jax.Array, scores: jax.Array, *, top_k: int) -> jax.Array:
    if scores.ndim != 3:
        raise ContractError(f"router scores must have shape [batch, seq, experts], got {scores.shape}")
    batch, seq_len, num_experts = scores.shape
    selected = jnp.sum(jax.nn.one_hot(expert_ids, num_experts, dtype=jnp.float32), axis=-2)
    selected_per_sequence = jnp.sum(selected, axis=1)
    f_i = (num_experts / (top_k * seq_len)) * selected_per_sequence
    normalized_scores = scores / jnp.maximum(
        jnp.sum(scores, axis=-1, keepdims=True),
        jnp.asarray(1e-20, dtype=jnp.float32),
    )
    p_i = jnp.mean(normalized_scores, axis=1)
    return jnp.sum(f_i * p_i) / jnp.asarray(batch, dtype=jnp.float32)
