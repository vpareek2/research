"""Sparse feed-forward components."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from jaxtitan.errors import ContractError
from jaxtitan.models.components.dtypes import dtype_from_name
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
        kernel_init: Any | None = None,
    ):
        if top_k <= 0 or num_experts <= 0 or top_k > num_experts:
            raise ContractError("router top_k must be positive and <= num_experts")
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

    def __call__(self, x: jax.Array) -> RouterOutput:
        scores = jax.nn.sigmoid(jnp.asarray(self.proj(x), dtype=jnp.float32))
        top_scores, expert_ids = jax.lax.top_k(scores, self.top_k)
        denominator = jnp.maximum(jnp.sum(top_scores, axis=-1, keepdims=True), jnp.asarray(1e-9, dtype=jnp.float32))
        weights = top_scores / denominator
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
            kernel_init=kernel_init,
        )
        self.experts = ExpertSwiGLU(
            spec.hidden_size,
            moe.expert_intermediate_size,
            moe.num_experts,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=kernel_init,
        )

    def route(self, x: jax.Array) -> RouterOutput:
        return self.router(x)

    def __call__(self, x: jax.Array) -> jax.Array:
        route = self.router(x)
        return self.experts(x, route.expert_ids, route.weights)
