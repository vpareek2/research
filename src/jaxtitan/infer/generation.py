"""KV-cache-native token generation."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.sharding import NamedSharding, PartitionSpec as P

from jaxtitan.batch import DecodeBatch, PrefillBatch
from jaxtitan.errors import ContractError
from jaxtitan.infer.core import InferenceState
from jaxtitan.mesh import ShardingPlan
from jaxtitan.models import decode_model, dtype_from_name, prefill_model
from jaxtitan.models.execution import ModelExecutionContext
from jaxtitan.specs.generation import GenerationSpec
from jaxtitan.specs.model import ModelSpec
from jaxtitan.state import RngState


@struct.dataclass
class KVCache:
    """Layer-major functional KV cache."""

    keys: Any
    values: Any
    lengths: Any
    max_cache_len: int = struct.field(pytree_node=False)
    rope_theta: float = struct.field(pytree_node=False)


@struct.dataclass
class PrefillOutput:
    """Prefill logits and updated cache."""

    logits: Any
    next_token_logits: Any
    cache: KVCache


@struct.dataclass
class DecodeOutput:
    """One-token decode logits and updated cache."""

    logits: Any
    cache: KVCache


@struct.dataclass
class SampleOutput:
    """Sampled token ids, logprobs, and updated inference state."""

    token_ids: Any
    logprobs: Any
    state: InferenceState


@struct.dataclass
class GenerationResult:
    """Token-native autoregressive generation result."""

    generated_ids: Any
    full_ids: Any
    logprobs: Any
    state: InferenceState
    cache: KVCache


def init_kv_cache(
    model_spec: ModelSpec,
    batch_size: int,
    max_cache_len: int,
    dtype: Any | None = None,
    *,
    sharding: ShardingPlan | None = None,
) -> KVCache:
    """Initialize a zeroed KV cache from model shape metadata."""

    if batch_size <= 0:
        raise ContractError(f"batch_size must be positive, got {batch_size}")
    if max_cache_len <= 0:
        raise ContractError(f"max_cache_len must be positive, got {max_cache_len}")
    if max_cache_len > model_spec.max_seq_len:
        raise ContractError(
            f"max_cache_len {max_cache_len} exceeds model.max_seq_len={model_spec.max_seq_len}"
        )
    if sharding is not None and sharding.parallelism.context_parallel:
        if max_cache_len % sharding.context_parallel_axis_size != 0:
            raise ContractError(
                f"max_cache_len {max_cache_len} must be divisible by cp axis size "
                f"{sharding.context_parallel_axis_size}"
            )
    if dtype is None:
        cache_dtype = dtype_from_name(model_spec.compute_dtype)
    elif isinstance(dtype, str):
        cache_dtype = dtype_from_name(dtype)
    else:
        cache_dtype = dtype
    head_dim = model_spec.hidden_size // model_spec.num_heads
    shape = (model_spec.num_layers, batch_size, max_cache_len, model_spec.n_kv_heads, head_dim)
    keys = jnp.zeros(shape, dtype=cache_dtype)
    values = jnp.zeros(shape, dtype=cache_dtype)
    lengths = jnp.zeros((batch_size,), dtype=jnp.int32)
    if sharding is not None:
        keys, values, lengths = _place_kv_cache_arrays(keys, values, lengths, sharding)
    return KVCache(
        keys=keys,
        values=values,
        lengths=lengths,
        max_cache_len=max_cache_len,
        rope_theta=model_spec.rope_theta,
    )


def make_prefill_step(
    graph: Any,
    *,
    sharding: ShardingPlan | None = None,
) -> Callable[[InferenceState, PrefillBatch, KVCache], PrefillOutput]:
    """Create a compiled prefill callable bound to a model graph."""

    execution = _model_execution_context(sharding)

    @jax.jit
    def _compiled(
        model_state: Any,
        input_ids: Any,
        positions: Any,
        attention_mask: Any,
        cache: KVCache,
    ) -> tuple[Any, Any, KVCache]:
        return prefill_model(
            graph,
            model_state,
            input_ids,
            positions,
            attention_mask,
            cache,
            execution=execution,
        )

    def _prefill(state: InferenceState, batch: PrefillBatch, cache: KVCache) -> PrefillOutput:
        _validate_prefill_batch(batch, cache)
        logits, next_token_logits, next_cache = _compiled(
            state.model,
            batch.input_ids,
            batch.positions,
            batch.attention_mask,
            cache,
        )
        return PrefillOutput(logits=logits, next_token_logits=next_token_logits, cache=next_cache)

    return _prefill


def make_decode_step(
    graph: Any,
    *,
    sharding: ShardingPlan | None = None,
) -> Callable[[InferenceState, DecodeBatch, KVCache], DecodeOutput]:
    """Create a compiled one-token decode callable bound to a model graph."""

    execution = _model_execution_context(sharding, include_context_parallel=False)

    @jax.jit
    def _compiled(
        model_state: Any,
        token_ids: Any,
        positions: Any,
        attention_mask: Any,
        cache: KVCache,
    ) -> tuple[Any, KVCache]:
        return decode_model(
            graph,
            model_state,
            token_ids,
            positions,
            attention_mask,
            cache,
            execution=execution,
        )

    def _decode(state: InferenceState, batch: DecodeBatch, cache: KVCache) -> DecodeOutput:
        _validate_decode_batch(batch, cache)
        logits, next_cache = _compiled(
            state.model,
            batch.token_ids,
            batch.positions,
            batch.attention_mask,
            cache,
        )
        return DecodeOutput(logits=logits, cache=next_cache)

    return _decode


def prefill(
    graph: Any,
    state: InferenceState,
    batch: PrefillBatch,
    cache: KVCache,
    *,
    sharding: ShardingPlan | None = None,
) -> PrefillOutput:
    """Run compiled prefill for a graph/state/cache."""

    return make_prefill_step(graph, sharding=sharding)(state, batch, cache)


def decode_one(
    graph: Any,
    state: InferenceState,
    batch: DecodeBatch,
    cache: KVCache,
    *,
    sharding: ShardingPlan | None = None,
) -> DecodeOutput:
    """Run compiled one-token decode for a graph/state/cache."""

    return make_decode_step(graph, sharding=sharding)(state, batch, cache)


def sample_next_token(logits: Any, state: InferenceState, spec: GenerationSpec) -> SampleOutput:
    """Sample next token ids from logits with explicit RNG state updates."""

    _validate_sampling_spec(spec)
    logits = jnp.asarray(logits, dtype=jnp.float32)
    if logits.ndim != 2:
        raise ContractError(f"logits must have shape [batch, vocab], got {logits.shape}")
    sample_key, next_sample_key = jax.random.split(state.rng.sample)
    scaled = logits / jnp.asarray(spec.temperature, dtype=jnp.float32)
    filtered = _top_k_logits(scaled, spec.top_k)
    if spec.top_k == 1:
        token_ids = jnp.argmax(filtered, axis=-1).astype(jnp.int32)
    else:
        token_ids = jax.random.categorical(sample_key, filtered, axis=-1).astype(jnp.int32)
    log_probs = jax.nn.log_softmax(filtered, axis=-1)
    logprobs = jnp.take_along_axis(log_probs, token_ids[:, None], axis=-1)[:, 0]
    rng = state.rng.replace(sample=next_sample_key)
    return SampleOutput(token_ids=token_ids, logprobs=logprobs, state=state.replace(rng=rng))


def generate_tokens(
    graph: Any,
    state: InferenceState,
    model_spec: ModelSpec,
    prompt_ids: Any,
    spec: GenerationSpec,
    *,
    sharding: ShardingPlan | None = None,
) -> GenerationResult:
    """Generate token ids from prompt token ids using prefill/decode."""

    _validate_sampling_spec(spec)
    prompt_ids = jnp.asarray(prompt_ids, dtype=jnp.int32)
    if prompt_ids.ndim != 2:
        raise ContractError(f"prompt_ids must have shape [batch, prompt_len], got {prompt_ids.shape}")
    batch_size, prompt_len = prompt_ids.shape
    total_len = prompt_len + spec.max_new_tokens
    if total_len > model_spec.max_seq_len:
        raise ContractError(
            f"prompt_len + max_new_tokens ({total_len}) exceeds model.max_seq_len={model_spec.max_seq_len}"
        )
    padded_prompt_len = _pad_to_cp_multiple(prompt_len, sharding)
    padded_total_len = _pad_to_cp_multiple(total_len, sharding)
    cache = init_kv_cache(model_spec, batch_size=batch_size, max_cache_len=padded_total_len, sharding=sharding)
    padded_prompt_ids = _pad_prompt_ids(prompt_ids, padded_prompt_len)
    positions = jnp.broadcast_to(jnp.arange(padded_prompt_len, dtype=jnp.int32)[None, :], padded_prompt_ids.shape)
    prompt_mask = jnp.broadcast_to(
        (jnp.arange(padded_prompt_len, dtype=jnp.int32) < prompt_len)[None, :],
        padded_prompt_ids.shape,
    )
    prefill_batch = PrefillBatch(
        input_ids=_place_prefill_array(padded_prompt_ids, sharding),
        positions=_place_prefill_array(positions, sharding),
        attention_mask=_place_prefill_array(prompt_mask, sharding),
    )
    prefill_step = make_prefill_step(graph, sharding=sharding)
    decode_step = make_decode_step(graph, sharding=sharding)
    prefill_out = prefill_step(state, prefill_batch, cache)
    next_logits = prefill_out.logits[:, prompt_len - 1, :]
    cache = prefill_out.cache
    current_state = state
    generated = []
    logprobs = []
    for offset in range(spec.max_new_tokens):
        sample = sample_next_token(next_logits, current_state, spec)
        current_state = sample.state
        generated.append(sample.token_ids)
        logprobs.append(sample.logprobs)
        decode_positions = jnp.full((batch_size,), prompt_len + offset, dtype=jnp.int32)
        attention_mask = jnp.arange(padded_total_len, dtype=jnp.int32)[None, :] <= decode_positions[:, None]
        decode_batch = DecodeBatch(
            token_ids=_place_batch_vector(sample.token_ids, sharding),
            positions=_place_batch_vector(decode_positions, sharding),
            attention_mask=_place_decode_mask(attention_mask, sharding),
        )
        decode_out = decode_step(current_state, decode_batch, cache)
        next_logits = decode_out.logits
        cache = decode_out.cache
    generated_ids = jnp.stack(generated, axis=1)
    generated_logprobs = jnp.stack(logprobs, axis=1)
    return GenerationResult(
        generated_ids=generated_ids,
        full_ids=jnp.concatenate([prompt_ids, generated_ids], axis=1),
        logprobs=generated_logprobs,
        state=current_state,
        cache=cache,
    )


def _pad_to_cp_multiple(length: int, sharding: ShardingPlan | None) -> int:
    if sharding is None or not sharding.parallelism.context_parallel:
        return length
    axis_size = sharding.context_parallel_axis_size
    return ((length + axis_size - 1) // axis_size) * axis_size


def _pad_prompt_ids(prompt_ids: Any, padded_len: int) -> Any:
    if prompt_ids.shape[1] == padded_len:
        return prompt_ids
    pad_width = padded_len - prompt_ids.shape[1]
    return jnp.pad(prompt_ids, ((0, 0), (0, pad_width)))


def _place_prefill_array(value: Any, sharding: ShardingPlan | None) -> Any:
    if sharding is None:
        return value
    return jax.device_put(value, sharding.batch.input_ids)


def _place_batch_vector(value: Any, sharding: ShardingPlan | None) -> Any:
    if sharding is None:
        return value
    return jax.device_put(value, _batch_vector_sharding(sharding))


def _place_decode_mask(value: Any, sharding: ShardingPlan | None) -> Any:
    if sharding is None:
        return value
    return jax.device_put(value, _decode_mask_sharding(sharding))


def _place_kv_cache_arrays(keys: Any, values: Any, lengths: Any, sharding: ShardingPlan) -> tuple[Any, Any, Any]:
    cache_sharding = _kv_array_sharding(sharding)
    return (
        jax.device_put(keys, cache_sharding),
        jax.device_put(values, cache_sharding),
        jax.device_put(lengths, _batch_vector_sharding(sharding)),
    )


def _kv_array_sharding(sharding: ShardingPlan) -> NamedSharding:
    seq_axis = sharding.context_parallel_axis if sharding.parallelism.context_parallel else None
    return NamedSharding(sharding.mesh.mesh, P(None, "data", seq_axis, None, None))


def _batch_vector_sharding(sharding: ShardingPlan) -> NamedSharding:
    return NamedSharding(sharding.mesh.mesh, P("data"))


def _decode_mask_sharding(sharding: ShardingPlan) -> NamedSharding:
    seq_axis = sharding.context_parallel_axis if sharding.parallelism.context_parallel else None
    return NamedSharding(sharding.mesh.mesh, P("data", seq_axis))


def _model_execution_context(
    sharding: ShardingPlan | None,
    *,
    include_context_parallel: bool = True,
) -> ModelExecutionContext | None:
    if sharding is None or (
        not sharding.parallelism.expert_parallel
        and not sharding.parallelism.tensor_parallel
        and (not include_context_parallel or not sharding.parallelism.context_parallel)
    ):
        return None
    if sharding.parallelism.expert_parallel and sharding.expert_parallel_axis is None:
        raise ContractError("expert parallel sharding plan is missing a resolved expert axis")
    return ModelExecutionContext(
        expert_parallel_mesh=sharding.mesh.mesh if sharding.parallelism.expert_parallel else None,
        expert_parallel_axis_name=sharding.expert_parallel_axis or "ep",
        expert_fsdp_axis_name=sharding.expert_fsdp_axis,
        expert_parallel_dispatcher=sharding.expert_parallel_dispatcher or "all_to_all",
        tensor_parallel_mesh=sharding.mesh.mesh if sharding.parallelism.tensor_parallel else None,
        tensor_parallel_axis_name=sharding.tensor_parallel_axis or "tp",
        context_parallel_mesh=sharding.mesh.mesh
        if include_context_parallel and sharding.parallelism.context_parallel
        else None,
        context_parallel_axis_name=sharding.context_parallel_axis or "cp",
    )


def _validate_sampling_spec(spec: GenerationSpec) -> None:
    if spec.top_p is not None:
        raise ContractError("generation.top_p is not supported by token-native generation yet")


def _top_k_logits(logits: Any, top_k: int | None) -> Any:
    if top_k is None:
        return logits
    if top_k > logits.shape[-1]:
        raise ContractError(f"generation.top_k={top_k} exceeds vocab size {logits.shape[-1]}")
    values, _indices = jax.lax.top_k(logits, top_k)
    threshold = values[:, -1:]
    return jnp.where(logits >= threshold, logits, jnp.finfo(jnp.float32).min)


def _validate_prefill_batch(batch: PrefillBatch, cache: KVCache) -> None:
    input_shape = _rank2(batch.input_ids, "prefill.input_ids")
    if _rank2(batch.positions, "prefill.positions") != input_shape:
        raise ContractError("prefill.positions shape must equal prefill.input_ids shape")
    if _rank2(batch.attention_mask, "prefill.attention_mask") != input_shape:
        raise ContractError("prefill.attention_mask shape must equal prefill.input_ids shape")
    if input_shape[0] != cache.keys.shape[1]:
        raise ContractError(f"prefill batch size {input_shape[0]} must equal cache batch size {cache.keys.shape[1]}")
    _validate_positions(batch.positions, cache, "prefill.positions")


def _validate_decode_batch(batch: DecodeBatch, cache: KVCache) -> None:
    token_shape = _rank1(batch.token_ids, "decode.token_ids")
    if _rank1(batch.positions, "decode.positions") != token_shape:
        raise ContractError("decode.positions shape must equal decode.token_ids shape")
    mask_shape = _rank2(batch.attention_mask, "decode.attention_mask")
    if mask_shape != (token_shape[0], cache.max_cache_len):
        raise ContractError(
            f"decode.attention_mask shape {mask_shape} must equal "
            f"(batch, max_cache_len)=({token_shape[0]}, {cache.max_cache_len})"
        )
    if token_shape[0] != cache.keys.shape[1]:
        raise ContractError(f"decode batch size {token_shape[0]} must equal cache batch size {cache.keys.shape[1]}")
    _validate_positions(batch.positions, cache, "decode.positions")


def _rank1(value: Any, name: str) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in jnp.shape(value))
    if len(shape) != 1:
        raise ContractError(f"{name} must have rank 1, got shape {shape}")
    return shape


def _rank2(value: Any, name: str) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in jnp.shape(value))
    if len(shape) != 2:
        raise ContractError(f"{name} must have rank 2, got shape {shape}")
    return shape


def _validate_positions(value: Any, cache: KVCache, name: str) -> None:
    positions = np.asarray(value)
    if positions.size and positions.min() < 0:
        raise ContractError(f"{name} must be non-negative")
    if positions.size and positions.max() >= cache.max_cache_len:
        raise ContractError(f"{name} contains position outside max_cache_len={cache.max_cache_len}")
