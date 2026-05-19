"""Deterministic prepared-token data service."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from jaxtitan.batch import Batch
from jaxtitan.data.manifest import PreparedDatasetManifest, TokenShard, validate_dataset_manifest
from jaxtitan.errors import ContractError
from jaxtitan.state import DatasetState

SplitName = Literal["train", "val"]


@dataclass(frozen=True, slots=True)
class BatchProvenance:
    """Host-side provenance for one deterministic token batch."""

    split: str
    epoch: int
    token_start: int
    token_end: int
    examples: int
    target_tokens: int
    row_start_offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _OpenShard:
    info: TokenShard
    tokens: np.memmap


class PreparedDataService:
    """Sequential host-side reader for validated prepared-token datasets."""

    def __init__(
        self,
        *,
        manifest: PreparedDatasetManifest,
        split: SplitName,
        seq_len: int,
        batch_size: int,
    ) -> None:
        if seq_len <= 0:
            raise ContractError(f"seq_len must be positive, got {seq_len}")
        if batch_size <= 0:
            raise ContractError(f"batch_size must be positive, got {batch_size}")
        if split not in {"train", "val"}:
            raise ContractError(f"split must be 'train' or 'val', got {split!r}")

        token_split = manifest.train if split == "train" else manifest.val
        self.manifest = manifest
        self.split = split
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.split_start = token_split.start
        self.split_end = token_split.end
        self._shards = _open_shards(manifest)
        self._shard_starts = tuple(shard.info.start for shard in self._shards)

        if not self._batch_fits(self.split_start):
            required = batch_size * seq_len + 1
            raise ContractError(
                f"{split} split has {token_split.tokens} tokens, but one batch requires at least {required}"
            )

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        *,
        tokenizer_id: str | None,
        split: SplitName,
        seq_len: int,
        batch_size: int,
    ) -> "PreparedDataService":
        manifest = validate_dataset_manifest(path, tokenizer_id=tokenizer_id)
        return cls(manifest=manifest, split=split, seq_len=seq_len, batch_size=batch_size)

    def initial_state(self) -> DatasetState:
        return DatasetState(
            shard_index=self._shard_index_for_offset(self.split_start),
            token_offset=self.split_start,
            epoch=0,
            shuffle_state=None,
        )

    def next_batch(self, state: DatasetState, *, repeat: bool = False) -> tuple[Batch, DatasetState, BatchProvenance]:
        offset = self._next_batch_offset(state, repeat=repeat)
        epoch = state.epoch if offset == state.token_offset else state.epoch + 1

        row_starts = tuple(offset + row * self.seq_len for row in range(self.batch_size))
        examples = []
        for row_start in row_starts:
            tokens = self._read(row_start, row_start + self.seq_len + 1)
            examples.append(tokens)

        stacked = np.stack(examples)
        input_ids = stacked[:, :-1].astype(np.int32, copy=False)
        target_ids = stacked[:, 1:].astype(np.int32, copy=False)
        loss_mask = np.ones((self.batch_size, self.seq_len), dtype=np.bool_)
        token_end = offset + self.batch_size * self.seq_len

        next_state = DatasetState(
            shard_index=self._shard_index_for_offset(token_end),
            token_offset=token_end,
            epoch=epoch,
            shuffle_state=None,
        )
        provenance = BatchProvenance(
            split=self.split,
            epoch=epoch,
            token_start=offset,
            token_end=token_end,
            examples=self.batch_size,
            target_tokens=self.batch_size * self.seq_len,
            row_start_offsets=row_starts,
        )
        return Batch(input_ids=input_ids, target_ids=target_ids, loss_mask=loss_mask), next_state, provenance

    def _next_batch_offset(self, state: DatasetState, *, repeat: bool) -> int:
        if state.shuffle_state is not None:
            raise ContractError("PreparedDataService does not support shuffled DatasetState")
        if state.token_offset < self.split_start or state.token_offset > self.split_end:
            raise ContractError(
                f"dataset token_offset={state.token_offset} is outside {self.split} split "
                f"[{self.split_start}, {self.split_end}]"
            )
        if self._batch_fits(state.token_offset):
            return state.token_offset
        if not repeat:
            raise StopIteration(f"not enough tokens left in {self.split} split for one full batch")
        return self.split_start

    def _batch_fits(self, offset: int) -> bool:
        return offset + self.batch_size * self.seq_len + 1 <= self.split_end

    def _read(self, start: int, end: int) -> np.ndarray:
        return _read_token_range_from_open_shards(self._shards, self._shard_starts, start, end, total_tokens=self.manifest.num_tokens)

    def _shard_index_for_offset(self, offset: int) -> int:
        if offset == self.manifest.num_tokens:
            return len(self._shards) - 1
        for idx, shard in enumerate(self._shards):
            if shard.info.start <= offset < shard.info.end:
                return idx
        raise ContractError(f"token offset {offset} is outside prepared dataset shard bounds")


def read_token_range(manifest: PreparedDatasetManifest, start: int, end: int) -> np.ndarray:
    """Read an absolute token interval from a prepared dataset manifest."""

    open_shards = _open_shards(manifest)
    shard_starts = tuple(shard.info.start for shard in open_shards)
    return _read_token_range_from_open_shards(open_shards, shard_starts, start, end, total_tokens=manifest.num_tokens)


def _open_shards(manifest: PreparedDatasetManifest) -> tuple[_OpenShard, ...]:
    data_dir = manifest.manifest_path.parent
    return tuple(
        _OpenShard(
            info=shard,
            tokens=np.memmap(data_dir / shard.path, dtype=np.uint32, mode="r"),
        )
        for shard in manifest.shards
    )


def _read_token_range_from_open_shards(
    shards: tuple[_OpenShard, ...],
    shard_starts: tuple[int, ...],
    start: int,
    end: int,
    *,
    total_tokens: int,
) -> np.ndarray:
    if start < 0 or end < start or end > total_tokens:
        raise IndexError(f"token range is outside dataset bounds: start={start}, end={end}, total={total_tokens}")
    if end == start:
        return np.asarray([], dtype=np.uint32)

    pieces = []
    cursor = start
    while cursor < end:
        shard_idx = bisect.bisect_right(shard_starts, cursor) - 1
        if shard_idx < 0 or shard_idx >= len(shards):
            raise IndexError(f"token range starts outside shards: start={start}, end={end}")
        shard = shards[shard_idx]
        shard_start = shard.info.start
        shard_end = shard.info.end
        if cursor >= shard_end:
            raise IndexError(f"token range crosses a gap before token {cursor}")
        take_end = min(end, shard_end)
        local_start = cursor - shard_start
        local_end = take_end - shard_start
        pieces.append(np.asarray(shard.tokens[local_start:local_end], dtype=np.uint32))
        cursor = take_end
    return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
