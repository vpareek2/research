"""Reusable model components assembled by model recipe modules."""

from jaxtitan.models.components.attention import (
    DecodeAttentionContext,
    FullAttentionContext,
    GroupedQueryAttention,
    PrefillAttentionContext,
    scaled_dot_product_attention,
)
from jaxtitan.models.components.blocks import DecoderBlock, TrinityDenseBlock, TrinityMoEBlock
from jaxtitan.models.components.dtypes import dtype_from_name
from jaxtitan.models.components.ffn import DecoderSwiGLU
from jaxtitan.models.components.init import truncated_normal_init
from jaxtitan.models.components.masks import cache_attention_mask, full_sequence_attention_mask
from jaxtitan.models.components.moe import (
    AllToAllExpertDispatcher,
    ExpertParallelDispatcher,
    ExpertSwiGLU,
    LocalExpertDispatcher,
    RdepStaticExpertDispatcher,
    RouterOutput,
    SigmoidTopKRouter,
    SparseMoE,
)
from jaxtitan.models.components.norm import build_rms_norm
from jaxtitan.models.components.position import apply_rope, apply_rope_at_positions, precompute_rope

__all__ = [
    "cache_attention_mask",
    "AllToAllExpertDispatcher",
    "DecodeAttentionContext",
    "DecoderBlock",
    "DecoderSwiGLU",
    "ExpertParallelDispatcher",
    "ExpertSwiGLU",
    "FullAttentionContext",
    "full_sequence_attention_mask",
    "GroupedQueryAttention",
    "LocalExpertDispatcher",
    "PrefillAttentionContext",
    "RdepStaticExpertDispatcher",
    "RouterOutput",
    "SigmoidTopKRouter",
    "SparseMoE",
    "TrinityDenseBlock",
    "TrinityMoEBlock",
    "apply_rope",
    "apply_rope_at_positions",
    "build_rms_norm",
    "dtype_from_name",
    "precompute_rope",
    "scaled_dot_product_attention",
    "truncated_normal_init",
]
