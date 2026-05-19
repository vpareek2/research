"""Artifact service contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from jaxtitan.specs.run import RunSpec


class ArtifactWriter(Protocol):
    """Host-side writer for canonical local artifacts."""

    def write_config(self, source_toml: str, resolved: RunSpec) -> None: ...

    def append_event(self, event: Mapping[str, Any]) -> None: ...

    def append_train_metrics(self, row: Mapping[str, Any]) -> None: ...

    def append_eval_metrics(self, row: Mapping[str, Any]) -> None: ...

    def write_summary(self, summary: Mapping[str, Any]) -> None: ...
