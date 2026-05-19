from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from jaxtitan.data import PreparedDataService, PreparedDatasetManifest, TokenShard, TokenSplit, read_token_range
from jaxtitan.errors import ContractError
from jaxtitan.state import DatasetState


def test_read_token_range_within_one_shard(prepared_dataset_factory: Callable[..., Path]) -> None:
    manifest_path = _service_manifest(prepared_dataset_factory)
    service = PreparedDataService.from_manifest(
        manifest_path,
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=3,
        batch_size=2,
    )

    values = read_token_range(service.manifest, 1, 4)

    np.testing.assert_array_equal(values, np.asarray([1, 2, 3], dtype=np.uint32))


def test_read_token_range_across_shards(prepared_dataset_factory: Callable[..., Path]) -> None:
    manifest_path = _service_manifest(prepared_dataset_factory)
    service = PreparedDataService.from_manifest(
        manifest_path,
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=3,
        batch_size=2,
    )

    values = read_token_range(service.manifest, 8, 13)

    np.testing.assert_array_equal(values, np.asarray([8, 9, 10, 11, 12], dtype=np.uint32))


def test_read_token_range_empty_and_out_of_bounds(prepared_dataset_factory: Callable[..., Path]) -> None:
    manifest_path = _service_manifest(prepared_dataset_factory)
    service = PreparedDataService.from_manifest(
        manifest_path,
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=3,
        batch_size=2,
    )

    assert read_token_range(service.manifest, 5, 5).tolist() == []
    with pytest.raises(IndexError, match="outside dataset bounds"):
        read_token_range(service.manifest, -1, 1)
    with pytest.raises(IndexError, match="outside dataset bounds"):
        read_token_range(service.manifest, 0, service.manifest.num_tokens + 1)


def test_read_token_range_detects_shard_gap(prepared_dataset_factory: Callable[..., Path]) -> None:
    manifest_path = _service_manifest(prepared_dataset_factory)
    service = PreparedDataService.from_manifest(
        manifest_path,
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=3,
        batch_size=2,
    )
    manifest = PreparedDatasetManifest(
        manifest_path=service.manifest.manifest_path,
        schema_version=2,
        kind="training_tokens",
        dtype="uint32",
        tokenizer_id="toy-tokenizer",
        num_tokens=30,
        train=TokenSplit(start=0, end=25, tokens=25),
        val=TokenSplit(start=25, end=30, tokens=5),
        shards=(
            TokenShard(path=Path("tokens-00000.bin"), start=0, end=4, tokens=4, bytes=16),
            TokenShard(path=Path("tokens-00001.bin"), start=6, end=16, tokens=10, bytes=40),
        ),
        token_bytes_path=Path("token_bytes.bin"),
        token_bytes_sha256=None,
        manifest_sha256="manual",
    )

    with pytest.raises(IndexError, match="gap"):
        read_token_range(manifest, 3, 7)


def test_services_select_train_and_val_split_bounds(prepared_dataset_factory: Callable[..., Path]) -> None:
    manifest_path = _service_manifest(prepared_dataset_factory)

    train = PreparedDataService.from_manifest(
        manifest_path,
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=4,
        batch_size=2,
    )
    val = PreparedDataService.from_manifest(
        manifest_path,
        tokenizer_id="toy-tokenizer",
        split="val",
        seq_len=2,
        batch_size=1,
    )

    assert train.split_start == 0
    assert train.split_end == 25
    assert val.split_start == 25
    assert val.split_end == 30


def test_next_batch_emits_shifted_fixed_shape_numpy_arrays(prepared_dataset_factory: Callable[..., Path]) -> None:
    service = _service(prepared_dataset_factory, seq_len=3, batch_size=2)

    batch, next_state, provenance = service.next_batch(service.initial_state())

    assert batch.input_ids.shape == (2, 3)
    assert batch.target_ids.shape == (2, 3)
    assert batch.loss_mask.shape == (2, 3)
    assert batch.input_ids.dtype == np.int32
    assert batch.target_ids.dtype == np.int32
    assert batch.loss_mask.dtype == np.bool_
    np.testing.assert_array_equal(batch.input_ids, np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32))
    np.testing.assert_array_equal(batch.target_ids, np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    assert batch.loss_mask.all()
    assert next_state == DatasetState(shard_index=0, token_offset=6, epoch=0, shuffle_state=None)
    assert provenance.split == "train"
    assert provenance.epoch == 0
    assert provenance.token_start == 0
    assert provenance.token_end == 6
    assert provenance.examples == 2
    assert provenance.target_tokens == 6
    assert provenance.row_start_offsets == (0, 3)


def test_state_advances_across_multiple_batches(prepared_dataset_factory: Callable[..., Path]) -> None:
    service = _service(prepared_dataset_factory, seq_len=4, batch_size=2)
    state = service.initial_state()

    _, state, first = service.next_batch(state)
    _, state, second = service.next_batch(state)
    _, state, third = service.next_batch(state)

    assert first.row_start_offsets == (0, 4)
    assert second.row_start_offsets == (8, 12)
    assert third.row_start_offsets == (16, 20)
    assert state.token_offset == 24
    assert state.epoch == 0


def test_repeat_false_raises_at_split_end(prepared_dataset_factory: Callable[..., Path]) -> None:
    service = _service(prepared_dataset_factory, seq_len=4, batch_size=2)
    state = service.initial_state()
    for _ in range(3):
        _, state, _ = service.next_batch(state)

    with pytest.raises(StopIteration):
        service.next_batch(state, repeat=False)


def test_repeat_true_resets_without_wrapping_inside_batch(prepared_dataset_factory: Callable[..., Path]) -> None:
    service = _service(prepared_dataset_factory, seq_len=4, batch_size=2)
    state = service.initial_state()
    for _ in range(3):
        _, state, _ = service.next_batch(state)

    batch, state, provenance = service.next_batch(state, repeat=True)

    assert provenance.epoch == 1
    assert provenance.row_start_offsets == (0, 4)
    assert state.token_offset == 8
    assert state.epoch == 1
    np.testing.assert_array_equal(batch.input_ids[0], np.asarray([0, 1, 2, 3], dtype=np.int32))


def test_service_construction_fails_when_split_too_small(prepared_dataset: Path) -> None:
    with pytest.raises(ContractError, match="one batch requires"):
        PreparedDataService.from_manifest(
            prepared_dataset,
            tokenizer_id="toy-tokenizer",
            split="val",
            seq_len=2,
            batch_size=1,
        )


def _service(prepared_dataset_factory: Callable[..., Path], *, seq_len: int, batch_size: int) -> PreparedDataService:
    return PreparedDataService.from_manifest(
        _service_manifest(prepared_dataset_factory),
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=seq_len,
        batch_size=batch_size,
    )


def _service_manifest(prepared_dataset_factory: Callable[..., Path]) -> Path:
    return prepared_dataset_factory(
        "service",
        shard_token_groups=(tuple(range(0, 10)), tuple(range(10, 20)), tuple(range(20, 30))),
        train_tokens=25,
    )
