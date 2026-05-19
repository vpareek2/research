"""Public inference state and forward contracts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
from flax import struct

from jaxtitan.errors import ContractError
from jaxtitan.models import apply_model
from jaxtitan.specs.mesh import MeshSpec
from jaxtitan.specs.model import ModelSpec
from jaxtitan.state import RngState, TrainState


@struct.dataclass
class InferenceState:
    """Device-relevant model state for inference and rollout paths."""

    model: Any
    rng: RngState


@dataclass(frozen=True, slots=True)
class InferenceMetadata:
    """Host-only provenance for an inference state."""

    run_id: str | None
    checkpoint_step: int | None
    checkpoint_path: Path | None
    tokens_seen: int | None
    model_spec: ModelSpec
    mesh_spec: MeshSpec
    runtime_fingerprint: str | None

    def __post_init__(self) -> None:
        if self.checkpoint_path is not None:
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))


def initialize_inference_state(
    model_state: Any,
    *,
    seed: int | None = None,
    rng: RngState | None = None,
) -> InferenceState:
    """Initialize inference state from model params and exactly one RNG source."""

    if (seed is None) == (rng is None):
        raise ContractError("initialize_inference_state requires exactly one of seed or rng")
    return InferenceState(model=model_state, rng=_rng_from_seed(seed) if seed is not None else rng)


def inference_from_train_state(train_state: TrainState, *, rng: RngState | None = None) -> InferenceState:
    """Extract inference-only state from a training state."""

    return InferenceState(model=train_state.model, rng=train_state.rng if rng is None else rng)


def apply_inference_model(graph: Any, state: InferenceState, input_ids: Any) -> Any:
    """Apply a split model graph and inference state to token ids."""

    return apply_model(graph, state.model, input_ids)


def _rng_from_seed(seed: int) -> RngState:
    if seed < 0:
        raise ContractError(f"inference seed must be non-negative, got {seed}")
    train_key, data_key, eval_key, sample_key = jax.random.split(jax.random.key(seed), 4)
    return RngState(train=train_key, data=data_key, eval=eval_key, sample=sample_key)
