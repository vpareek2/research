"""Prepared-token shard reading primitives."""

import bisect
from dataclasses import dataclass

import numpy as np

from jaxtitan.data.manifest import PreparedDatasetManifest, TokenShard


@dataclass(frozen=True, slots=True)
class _OpenShard:
    info: TokenShard
    tokens: np.memmap


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
