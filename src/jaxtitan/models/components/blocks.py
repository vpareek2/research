"""Transformer block components."""

import math
from typing import Any

import jax
from flax import nnx

from jaxtitan.models.components.attention import (
    DecodeAttentionContext,
    FullAttentionContext,
    GroupedQueryAttention,
    PrefillAttentionContext,
)
from jaxtitan.models.components.ffn import DecoderSwiGLU
from jaxtitan.models.components.norm import build_rms_norm
from jaxtitan.specs.model import ModelSpec


class DecoderBlock(nnx.Module):
    """Single decoder transformer block."""

    def __init__(self, spec: ModelSpec, rngs: nnx.Rngs):
        self.pre_norm = build_rms_norm(spec, rngs=rngs)
        self.attn = GroupedQueryAttention(spec, rngs=rngs)
        self.post_norm = build_rms_norm(spec, rngs=rngs)
        self.mlp = DecoderSwiGLU(spec, rngs=rngs)

    def __call__(self, x: jax.Array, context: FullAttentionContext) -> jax.Array:
        x = x + self.attn(self.pre_norm(x), context)
        x = x + self.mlp(self.post_norm(x))
        return x

    def prefill(
        self,
        x: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
        cache: Any,
        layer_index: int,
    ) -> tuple[jax.Array, Any]:
        context = PrefillAttentionContext(
            positions=positions,
            attention_mask=attention_mask,
            cache=cache,
            layer_index=layer_index,
        )
        attn_out, cache = self.attn.prefill(self.pre_norm(x), context)
        x = x + attn_out
        x = x + self.mlp(self.post_norm(x))
        return x, cache

    def decode_one(
        self,
        x: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
        cache: Any,
        layer_index: int,
    ) -> tuple[jax.Array, Any]:
        context = DecodeAttentionContext(
            positions=positions,
            attention_mask=attention_mask,
            cache=cache,
            layer_index=layer_index,
        )
        attn_out, cache = self.attn.decode_one(self.pre_norm(x), context)
        x = x + attn_out
        x = x + self.mlp(self.post_norm(x))
        return x, cache


class TrinityDenseBlock(nnx.Module):
    """Dense Trinity-style transformer block with depth-scaled sandwich norms."""

    def __init__(
        self,
        spec: ModelSpec,
        rngs: nnx.Rngs,
        *,
        attention_position: str,
        attention_mask: str,
        local_window: int | None,
        qk_norm: bool,
        attention_gate: bool,
        kernel_init: Any,
    ):
        post_scale = 1.0 / math.sqrt(spec.num_layers)
        self.attn_pre_norm = build_rms_norm(spec, rngs=rngs)
        self.attn = GroupedQueryAttention(
            spec,
            rngs=rngs,
            position=attention_position,
            mask=attention_mask,
            local_window=local_window,
            qk_norm=qk_norm,
            gate=attention_gate,
            kernel_init=kernel_init,
        )
        self.attn_post_norm = build_rms_norm(spec, rngs=rngs, scale_init_value=post_scale)
        self.ffn_pre_norm = build_rms_norm(spec, rngs=rngs)
        self.mlp = DecoderSwiGLU(spec, rngs=rngs, kernel_init=kernel_init)
        self.ffn_post_norm = build_rms_norm(spec, rngs=rngs, scale_init_value=post_scale)

    def __call__(self, x: jax.Array, context: FullAttentionContext) -> jax.Array:
        x = x + self.attn_post_norm(self.attn(self.attn_pre_norm(x), context))
        x = x + self.ffn_post_norm(self.mlp(self.ffn_pre_norm(x)))
        return x

    def prefill(
        self,
        x: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
        cache: Any,
        layer_index: int,
    ) -> tuple[jax.Array, Any]:
        context = PrefillAttentionContext(
            positions=positions,
            attention_mask=attention_mask,
            cache=cache,
            layer_index=layer_index,
        )
        attn_out, cache = self.attn.prefill(self.attn_pre_norm(x), context)
        x = x + self.attn_post_norm(attn_out)
        x = x + self.ffn_post_norm(self.mlp(self.ffn_pre_norm(x)))
        return x, cache

    def decode_one(
        self,
        x: jax.Array,
        positions: jax.Array,
        attention_mask: jax.Array,
        cache: Any,
        layer_index: int,
    ) -> tuple[jax.Array, Any]:
        context = DecodeAttentionContext(
            positions=positions,
            attention_mask=attention_mask,
            cache=cache,
            layer_index=layer_index,
        )
        attn_out, cache = self.attn.decode_one(self.attn_pre_norm(x), context)
        x = x + self.attn_post_norm(attn_out)
        x = x + self.ffn_post_norm(self.mlp(self.ffn_pre_norm(x)))
        return x, cache
