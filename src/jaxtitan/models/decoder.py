"""Flax NNX decoder-only language model boundary."""

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from jaxtitan.errors import ContractError
from jaxtitan.models.components.attention import FullAttentionContext
from jaxtitan.models.components.blocks import DecoderBlock
from jaxtitan.models.components.dtypes import dtype_from_name
from jaxtitan.models.components.position import precompute_rope
from jaxtitan.models.execution import apply_layer
from jaxtitan.models.output import ModelOutput, ensure_model_output
from jaxtitan.specs.model import ModelSpec


@dataclass(frozen=True, slots=True)
class ParamMetadata:
    """Stable metadata for one model parameter leaf."""

    path: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    count: int
    tag: str


@dataclass(frozen=True, slots=True)
class ParamLayout:
    """Model-owned parameter layout policy for distributed placement."""

    path: tuple[str, ...]
    tag: str
    shape: tuple[int, ...]
    logical_axes: tuple[str, ...]
    fsdp_axis: int | None


@dataclass(frozen=True, slots=True)
class ModelBuildResult:
    """Explicit model graph/state split plus parameter metadata."""

    graph: Any
    state: Any
    metadata: tuple[ParamMetadata, ...]
    param_layouts: tuple[ParamLayout, ...]


def build_model(spec: ModelSpec, seed: int) -> ModelBuildResult:
    """Build a supported model and return explicit NNX graph/state pieces."""

    if spec.name == "decoder":
        model = DecoderModel(spec, rngs=nnx.Rngs(seed))
    elif spec.name == "trinity":
        from jaxtitan.models.trinity import TrinityModel

        model = TrinityModel(spec, rngs=nnx.Rngs(seed))
    else:
        raise ContractError(f"unsupported model.name {spec.name!r}; expected 'decoder' or 'trinity'")
    graph, state = nnx.split(model)
    metadata = parameter_metadata(state)
    return ModelBuildResult(graph=graph, state=state, metadata=metadata, param_layouts=parameter_layouts(metadata))


def apply_model_output(graph: Any, state: Any, input_ids: Any) -> ModelOutput:
    """Apply a split model graph/state and return the structured output."""

    model = nnx.merge(graph, state)
    return ensure_model_output(model(input_ids))


def apply_model(graph: Any, state: Any, input_ids: Any) -> jax.Array:
    """Apply a split model graph/state to token ids."""

    return apply_model_output(graph, state, input_ids).logits


def prefill_model(
    graph: Any,
    state: Any,
    input_ids: Any,
    positions: Any,
    attention_mask: Any,
    cache: Any,
) -> tuple[jax.Array, jax.Array, Any]:
    """Apply model prefill and update a KV cache."""

    model = nnx.merge(graph, state)
    return model.prefill(input_ids, positions, attention_mask, cache)


def decode_model(
    graph: Any,
    state: Any,
    token_ids: Any,
    positions: Any,
    attention_mask: Any,
    cache: Any,
) -> tuple[jax.Array, Any]:
    """Apply one-token decode and update a KV cache."""

    model = nnx.merge(graph, state)
    return model.decode_one(token_ids, positions, attention_mask, cache)


def count_parameters(metadata: tuple[ParamMetadata, ...]) -> int:
    """Count parameters from metadata, not by re-walking model state."""

    return sum(item.count for item in metadata)


def parameter_metadata(state: Any) -> tuple[ParamMetadata, ...]:
    """Build metadata for every parameter leaf in an NNX State."""

    items = []
    for raw_path, variable in nnx.to_flat_state(state):
        value = variable.get_value()
        shape = tuple(int(dim) for dim in value.shape)
        path = tuple(str(part) for part in raw_path)
        items.append(
            ParamMetadata(
                path=path,
                shape=shape,
                dtype=str(value.dtype),
                count=reduce(mul, shape, 1),
                tag=_tag_for_path(path),
            )
        )
    return tuple(items)


def parameter_layouts(metadata: tuple[ParamMetadata, ...]) -> tuple[ParamLayout, ...]:
    """Build model-owned sharding layout metadata from semantic parameter metadata."""

    return tuple(_layout_for_metadata(item) for item in metadata)


class DecoderModel(nnx.Module):
    """Decoder-only language model recipe built from reusable components."""

    def __init__(self, spec: ModelSpec, rngs: nnx.Rngs):
        if spec.tied_embeddings:
            raise ContractError("model.tied_embeddings is not supported yet")
        self.spec = spec
        dtype = dtype_from_name(spec.compute_dtype)
        param_dtype = dtype_from_name(spec.param_dtype)
        self.embed = nnx.Embed(
            spec.vocab_size,
            spec.hidden_size,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.layers = nnx.List([DecoderBlock(spec, rngs=rngs) for _ in range(spec.num_layers)])
        self.norm = nnx.RMSNorm(
            spec.hidden_size,
            epsilon=spec.norm_epsilon,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.lm_head = nnx.Linear(
            spec.hidden_size,
            spec.vocab_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(self, input_ids: Any) -> jax.Array:
        input_ids = jnp.asarray(input_ids)
        if input_ids.ndim != 2:
            raise ContractError(f"input_ids must have shape [batch, seq], got {input_ids.shape}")
        _, seq_len = input_ids.shape
        if seq_len > self.spec.max_seq_len:
            raise ContractError(f"input sequence length {seq_len} exceeds model.max_seq_len={self.spec.max_seq_len}")

        x = self.embed(input_ids)
        cos, sin = precompute_rope(
            seq_len=seq_len,
            head_dim=self.spec.hidden_size // self.spec.num_heads,
            theta=self.spec.rope_theta,
            dtype=x.dtype,
        )
        context = FullAttentionContext(cos=cos, sin=sin)
        for layer in self.layers:
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

        x = self.embed(input_ids)
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

        x = self.embed(token_ids[:, None])
        for layer_index, layer in enumerate(self.layers):
            x, cache = layer.decode_one(x, positions, attention_mask, cache, layer_index)
        logits = self.lm_head(self.norm(x))[:, 0, :]
        return logits, cache


def _tag_for_path(path: tuple[str, ...]) -> str:
    if path[0] == "embed":
        return "embedding"
    if path[0] == "lm_head":
        return "lm_head"
    if path[0] == "norm":
        return "final_norm"
    if "attn_pre_norm" in path:
        return "attention_pre_norm"
    if "attn_post_norm" in path:
        return "attention_post_norm"
    if "ffn_pre_norm" in path:
        return "ffn_pre_norm"
    if "ffn_post_norm" in path:
        return "ffn_post_norm"
    if "pre_norm" in path:
        return "block_pre_norm"
    if "post_norm" in path:
        return "block_post_norm"
    if "attn" in path:
        return _attention_tag(path)
    if "mlp" in path and (
        "router" in path or "experts" in path or "shared_experts" in path or "expert_bias" in path
    ):
        return _moe_tag(path)
    if "mlp" in path:
        return _mlp_tag(path)
    raise ContractError(f"unrecognized decoder parameter path {'.'.join(path)}")


def _attention_tag(path: tuple[str, ...]) -> str:
    for component, tag in (
        ("q_norm", "attention_q_norm"),
        ("k_norm", "attention_k_norm"),
        ("q", "attention_q"),
        ("k", "attention_k"),
        ("v", "attention_v"),
        ("o", "attention_o"),
        ("gate", "attention_gate"),
    ):
        if component in path:
            return tag
    raise ContractError(f"unrecognized attention parameter path {'.'.join(path)}")


def _mlp_tag(path: tuple[str, ...]) -> str:
    for component, tag in (("gate", "mlp_gate"), ("up", "mlp_up"), ("down", "mlp_down")):
        if component in path:
            return tag
    raise ContractError(f"unrecognized MLP parameter path {'.'.join(path)}")


def _moe_tag(path: tuple[str, ...]) -> str:
    if "expert_bias" in path:
        return "moe_expert_bias"
    if "router" in path:
        return "moe_router"
    if "shared_experts" in path:
        for component, tag in (
            ("gate", "moe_shared_gate"),
            ("up", "moe_shared_up"),
            ("down", "moe_shared_down"),
        ):
            if component in path:
                return tag
    for component, tag in (("gate", "moe_gate"), ("up", "moe_up"), ("down", "moe_down")):
        if component in path:
            return tag
    raise ContractError(f"unrecognized MoE parameter path {'.'.join(path)}")


def _layout_for_metadata(item: ParamMetadata) -> ParamLayout:
    logical_axes, fsdp_axis = _layout_policy(item)
    if len(logical_axes) != len(item.shape):
        raise ContractError(
            f"parameter layout for {'.'.join(item.path)!r} has rank {len(logical_axes)}, "
            f"but parameter shape has rank {len(item.shape)}"
        )
    if fsdp_axis is not None and not (0 <= fsdp_axis < len(item.shape)):
        raise ContractError(f"parameter layout for {'.'.join(item.path)!r} has invalid fsdp axis {fsdp_axis}")
    return ParamLayout(
        path=item.path,
        tag=item.tag,
        shape=item.shape,
        logical_axes=logical_axes,
        fsdp_axis=fsdp_axis,
    )


def _layout_policy(item: ParamMetadata) -> tuple[tuple[str, ...], int | None]:
    tag = item.tag
    if tag == "embedding":
        return ("vocab", "hidden"), None
    if tag == "lm_head":
        return ("hidden", "vocab"), 0
    if tag in {"attention_q", "attention_k", "attention_v", "attention_gate"}:
        return ("hidden_in", "hidden_out"), 1
    if tag == "attention_o":
        return ("hidden_in", "hidden_out"), 0
    if tag in {"mlp_gate", "mlp_up"}:
        return ("hidden", "intermediate"), 1
    if tag == "mlp_down":
        return ("intermediate", "hidden"), 0
    if tag == "moe_router":
        return ("hidden", "expert"), None
    if tag == "moe_expert_bias":
        return ("expert",), None
    if tag in {"moe_gate", "moe_up"}:
        return ("expert", "hidden", "intermediate"), None
    if tag == "moe_down":
        return ("expert", "intermediate", "hidden"), None
    if tag in {"moe_shared_gate", "moe_shared_up"}:
        return ("hidden", "intermediate"), 1
    if tag == "moe_shared_down":
        return ("intermediate", "hidden"), 0
    if tag in {
        "attention_q_norm",
        "attention_k_norm",
        "attention_pre_norm",
        "attention_post_norm",
        "block_pre_norm",
        "block_post_norm",
        "ffn_pre_norm",
        "ffn_post_norm",
        "final_norm",
    }:
        return tuple("norm" for _dim in item.shape), None
    return tuple("replicated" for _dim in item.shape), None
