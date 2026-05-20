"""Explicit dynamic state contracts."""

from dataclasses import dataclass
from pathlib import Path
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
class DataPipelineState:
    """Host-checkpointable state for the canonical training data pipeline."""

    schema_version: int
    backend: str
    backend_version: str | None
    split: str
    order: str
    worker_count: int
    prefetch: bool
    manifest_path: Path
    manifest_sha256: str
    tokenizer_id: str
    seq_len: int
    batch_size: int
    num_records: int
    next_record_index: int
    token_offset: int
    epoch: int
    sampler_summary: str
    source_summary: str
    grain_state: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "grain_state", dict(self.grain_state))


@dataclass(frozen=True, slots=True)
class HostState:
    """Host-only state that should not be passed through JAX transforms."""

    dataset: DataPipelineState
    last_checkpoint_step: int
    wallclock_start_ns: int
    run_id: str
