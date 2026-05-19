"""Inference state and checkpoint restore boundaries."""

from jaxtitan.infer.checkpoint import InferenceRestore, restore_inference_checkpoint
from jaxtitan.infer.core import (
    InferenceMetadata,
    InferenceState,
    apply_inference_model,
    inference_from_train_state,
    initialize_inference_state,
)
from jaxtitan.infer.generation import (
    DecodeOutput,
    GenerationResult,
    KVCache,
    PrefillOutput,
    SampleOutput,
    decode_one,
    generate_tokens,
    init_kv_cache,
    make_decode_step,
    make_prefill_step,
    prefill,
    sample_next_token,
)

__all__ = [
    "DecodeOutput",
    "GenerationResult",
    "InferenceMetadata",
    "InferenceRestore",
    "InferenceState",
    "KVCache",
    "PrefillOutput",
    "SampleOutput",
    "apply_inference_model",
    "decode_one",
    "generate_tokens",
    "init_kv_cache",
    "inference_from_train_state",
    "initialize_inference_state",
    "make_decode_step",
    "make_prefill_step",
    "prefill",
    "restore_inference_checkpoint",
    "sample_next_token",
]
