"""Dense Trinity model recipe assembled from reusable components."""

import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from jaxtitan.errors import ContractError
from jaxtitan.models.components.attention import FullAttentionContext
from jaxtitan.models.components.blocks import TrinityDenseBlock
from jaxtitan.models.components.dtypes import dtype_from_name
from jaxtitan.models.components.init import truncated_normal_init
from jaxtitan.models.components.norm import build_rms_norm
from jaxtitan.models.components.position import precompute_rope
from jaxtitan.models.execution import apply_layer
from jaxtitan.specs.model import ModelSpec, TrinitySpec


class TrinityModel(nnx.Module):
    """Dense-only Trinity recipe.

    The recipe encodes Trinity's dense block shape while leaving MoE layers for
    a later slice.
    """

    def __init__(self, spec: ModelSpec, rngs: nnx.Rngs):
        if spec.tied_embeddings:
            raise ContractError("model.tied_embeddings is not supported yet")
        trinity = _trinity_spec(spec)
        self.spec = spec
        self.trinity = trinity
        self.layer_attention = tuple(_layer_attention_kind(index, trinity) for index in range(spec.num_layers))
        dtype = dtype_from_name(spec.compute_dtype)
        param_dtype = dtype_from_name(spec.param_dtype)
        init_std = trinity.init_std if trinity.init_std is not None else 0.5 / math.sqrt(spec.hidden_size)
        initializer = truncated_normal_init(init_std)
        self.embed = nnx.Embed(
            spec.vocab_size,
            spec.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            embedding_init=initializer,
            rngs=rngs,
        )
        self.layers = nnx.List(
            [
                TrinityDenseBlock(
                    spec,
                    rngs=rngs,
                    attention_position="none" if kind == "global" else "rope",
                    attention_mask="causal" if kind == "global" else "sliding_window",
                    local_window=None if kind == "global" else trinity.local_window,
                    qk_norm=trinity.qk_norm,
                    attention_gate=trinity.attention_gate,
                    kernel_init=initializer,
                )
                for kind in self.layer_attention
            ]
        )
        self.norm = build_rms_norm(spec, rngs=rngs)
        self.lm_head = nnx.Linear(
            spec.hidden_size,
            spec.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=initializer,
            rngs=rngs,
        )

    def __call__(self, input_ids: Any) -> jax.Array:
        input_ids = jnp.asarray(input_ids)
        if input_ids.ndim != 2:
            raise ContractError(f"input_ids must have shape [batch, seq], got {input_ids.shape}")
        _, seq_len = input_ids.shape
        if seq_len > self.spec.max_seq_len:
            raise ContractError(f"input sequence length {seq_len} exceeds model.max_seq_len={self.spec.max_seq_len}")

        x = self._embed(input_ids)
        cos, sin = precompute_rope(
            seq_len=seq_len,
            head_dim=self.spec.hidden_size // self.spec.num_heads,
            theta=self.spec.rope_theta,
            dtype=x.dtype,
        )
        for kind, layer in zip(self.layer_attention, self.layers, strict=True):
            context = FullAttentionContext(cos=None, sin=None) if kind == "global" else FullAttentionContext(cos=cos, sin=sin)
            x = apply_layer(layer, x, context, remat=self.spec.remat)
        return self.lm_head(self.norm(x))

    def prefill(self, input_ids: Any, positions: Any, attention_mask: Any, cache: Any) -> tuple[jax.Array, jax.Array, Any]:
        input_ids = jnp.asarray(input_ids)
        positions = jnp.asarray(positions)
        attention_mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
        if input_ids.ndim != 2:
            raise ContractError(f"input_ids must have shape [batch, seq], got {input_ids.shape}")
        _, seq_len = input_ids.shape
        if positions.shape != input_ids.shape:
            raise ContractError(f"positions shape {positions.shape} must equal input_ids shape {input_ids.shape}")
        if attention_mask.shape != input_ids.shape:
            raise ContractError(
                f"attention_mask shape {attention_mask.shape} must equal input_ids shape {input_ids.shape}"
            )
        if seq_len > self.spec.max_seq_len:
            raise ContractError(f"input sequence length {seq_len} exceeds model.max_seq_len={self.spec.max_seq_len}")

        x = self._embed(input_ids)
        for layer_index, layer in enumerate(self.layers):
            x, cache = layer.prefill(x, positions, attention_mask, cache, layer_index)
        logits = self.lm_head(self.norm(x))
        return logits, logits[:, -1, :], cache

    def decode_one(self, token_ids: Any, positions: Any, attention_mask: Any, cache: Any) -> tuple[jax.Array, Any]:
        token_ids = jnp.asarray(token_ids)
        positions = jnp.asarray(positions)
        attention_mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
        if token_ids.ndim != 1:
            raise ContractError(f"token_ids must have shape [batch], got {token_ids.shape}")
        if positions.shape != token_ids.shape:
            raise ContractError(f"positions shape {positions.shape} must equal token_ids shape {token_ids.shape}")
        if attention_mask.ndim != 2 or attention_mask.shape[0] != token_ids.shape[0]:
            raise ContractError("attention_mask must have shape [batch, max_cache_len]")

        x = self._embed(token_ids[:, None])
        for layer_index, layer in enumerate(self.layers):
            x, cache = layer.decode_one(x, positions, attention_mask, cache, layer_index)
        logits = self.lm_head(self.norm(x))[:, 0, :]
        return logits, cache

    def _embed(self, input_ids: Any) -> jax.Array:
        x = self.embed(input_ids)
        if self.trinity.embedding_scale == "sqrt_hidden":
            return x * jnp.asarray(math.sqrt(self.spec.hidden_size), dtype=x.dtype)
        raise ContractError(f"unsupported Trinity embedding scale {self.trinity.embedding_scale!r}")


def _trinity_spec(spec: ModelSpec) -> TrinitySpec:
    if spec.trinity is None:
        raise ContractError("model.name='trinity' requires [model.trinity]")
    if not isinstance(spec.trinity, TrinitySpec):
        raise ContractError("model.trinity must resolve to TrinitySpec")
    return spec.trinity


def _layer_attention_kind(index: int, trinity: TrinitySpec) -> str:
    cycle = trinity.local_layers_per_global + 1
    return "global" if (index + 1) % cycle == 0 else "local"
