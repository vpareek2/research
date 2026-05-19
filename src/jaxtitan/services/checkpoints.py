"""Checkpoint service contracts."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from jaxtitan.specs.run import RunSpec
from jaxtitan.state import DatasetState, TrainState


class CheckpointService(Protocol):
    """Host-side checkpoint service protocol."""

    def restore_or_initialize(
        self,
        initial_state: TrainState,
        initial_dataset: DatasetState,
        spec: RunSpec,
    ) -> tuple[TrainState, DatasetState, Mapping[str, Any]]: ...

    def save(
        self,
        state: TrainState,
        dataset: DatasetState,
        metadata: Mapping[str, Any],
    ) -> None: ...

    def latest(self) -> Path | None: ...
