"""Static-shape batch contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ArrayTree = Any


@dataclass(frozen=True, slots=True)
class Batch:
    """Training batch contract.

    Arrays must be fixed shape for a compiled step. Shape validation belongs to
    runtime builders because this contract intentionally does not import JAX.
    """

    input_ids: ArrayTree
    target_ids: ArrayTree
    loss_mask: ArrayTree
    doc_ids: ArrayTree | None = None


@dataclass(frozen=True, slots=True)
class EvalBatch:
    """Evaluation batch contract."""

    input_ids: ArrayTree
    target_ids: ArrayTree
    loss_mask: ArrayTree


@dataclass(frozen=True, slots=True)
class PrefillBatch:
    """Generation prefill batch contract."""

    input_ids: ArrayTree
    positions: ArrayTree
    attention_mask: ArrayTree


@dataclass(frozen=True, slots=True)
class DecodeBatch:
    """Single-token decode batch contract."""

    token_ids: ArrayTree
    positions: ArrayTree
    attention_mask: ArrayTree
