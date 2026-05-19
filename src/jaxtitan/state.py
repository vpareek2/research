"""Explicit dynamic state contracts."""

from dataclasses import dataclass
from typing import Any

from flax import struct

PyTree = Any


@struct.dataclass
class RngState:
    """Explicit RNG streams for deterministic JAX execution."""

    train: PyTree
    data: PyTree
    eval: PyTree
    sample: PyTree


@struct.dataclass
class TrainState:
    """Device-relevant training state passed through compiled steps."""

    step: PyTree
    tokens_seen: PyTree
    model: PyTree
    opt_state: PyTree
    rng: RngState
    schedule_state: PyTree | None = None


@dataclass(frozen=True, slots=True)
class DatasetState:
    """Checkpointable prepared-token cursor state."""

    shard_index: int
    token_offset: int
    epoch: int
    shuffle_state: int | None = None


@dataclass(frozen=True, slots=True)
class HostState:
    """Host-only state that should not be passed through JAX transforms."""

    dataset: DatasetState
    last_checkpoint_step: int
    wallclock_start_ns: int
    run_id: str
