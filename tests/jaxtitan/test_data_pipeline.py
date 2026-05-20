from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from jaxtitan.data import (
    PreparedDatasetManifest,
    PreparedTokenDataSource,
    PreparedTokenGrainPipeline,
    TokenShard,
    TokenSplit,
    data_pipeline_compat_payload,
    data_pipeline_state_from_mapping,
    data_pipeline_state_to_dict,
    load_dataset_manifest,
    read_token_range,
)
from jaxtitan.errors import ContractError


def test_read_token_range_within_one_shard(prepared_dataset_factory: Callable[..., Path]) -> None:
    manifest = load_dataset_manifest(_pipeline_manifest(prepared_dataset_factory))

    values = read_token_range(manifest, 1, 4)

    np.testing.assert_array_equal(values, np.asarray([1, 2, 3], dtype=np.uint32))


def test_read_token_range_across_shards(prepared_dataset_factory: Callable[..., Path]) -> None:
    manifest = load_dataset_manifest(_pipeline_manifest(prepared_dataset_factory))

    values = read_token_range(manifest, 8, 13)

    np.testing.assert_array_equal(values, np.asarray([8, 9, 10, 11, 12], dtype=np.uint32))


def test_read_token_range_empty_and_out_of_bounds(prepared_dataset_factory: Callable[..., Path]) -> None:
    manifest = load_dataset_manifest(_pipeline_manifest(prepared_dataset_factory))

    assert read_token_range(manifest, 5, 5).tolist() == []
    with pytest.raises(IndexError, match="outside dataset bounds"):
        read_token_range(manifest, -1, 1)
    with pytest.raises(IndexError, match="outside dataset bounds"):
        read_token_range(manifest, 0, manifest.num_tokens + 1)


def test_read_token_range_detects_shard_gap(prepared_dataset_factory: Callable[..., Path]) -> None:
    manifest = load_dataset_manifest(_pipeline_manifest(prepared_dataset_factory))
    gapped = PreparedDatasetManifest(
        manifest_path=manifest.manifest_path,
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
        read_token_range(gapped, 3, 7)


def test_data_source_reads_shifted_examples_and_provenance(prepared_dataset_factory: Callable[..., Path]) -> None:
    source = PreparedTokenDataSource.from_manifest(
        _pipeline_manifest(prepared_dataset_factory),
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=4,
    )

    first = source[0]
    cross_shard = source[3]

    assert len(source) == 6
    assert source.split_start == 0
    assert source.split_end == 25
    np.testing.assert_array_equal(first["input_ids"], np.asarray([0, 1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(first["target_ids"], np.asarray([1, 2, 3, 4], dtype=np.int32))
    assert first["loss_mask"].dtype == np.bool_
    assert int(first["token_start"]) == 0
    assert int(first["token_end"]) == 4
    np.testing.assert_array_equal(cross_shard["input_ids"], np.asarray([12, 13, 14, 15], dtype=np.int32))
    np.testing.assert_array_equal(cross_shard["target_ids"], np.asarray([13, 14, 15, 16], dtype=np.int32))
    with pytest.raises(IndexError):
        source[len(source)]


def test_pipeline_sequential_batches_are_fixed_shape_and_shifted(prepared_dataset_factory: Callable[..., Path]) -> None:
    pipeline = _pipeline(prepared_dataset_factory, seq_len=3, batch_size=2)
    state = pipeline.initial_state()

    result = pipeline.next_batch(state)

    assert result.batch.input_ids.shape == (2, 3)
    assert result.batch.target_ids.shape == (2, 3)
    assert result.batch.loss_mask.shape == (2, 3)
    assert result.batch.input_ids.dtype == np.int32
    assert result.batch.target_ids.dtype == np.int32
    assert result.batch.loss_mask.dtype == np.bool_
    np.testing.assert_array_equal(result.batch.input_ids, np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32))
    np.testing.assert_array_equal(result.batch.target_ids, np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    assert result.batch.loss_mask.all()
    assert result.state.backend == "grain"
    assert result.state.next_record_index == 2
    assert result.state.token_offset == 6
    assert result.state.epoch == 0
    assert result.provenance.split == "train"
    assert result.provenance.epoch == 0
    assert result.provenance.token_start == 0
    assert result.provenance.token_end == 6
    assert result.provenance.examples == 2
    assert result.provenance.target_tokens == 6
    assert result.provenance.row_start_offsets == (0, 3)
    pipeline.close()


def test_pipeline_state_advances_and_exhausts_without_repeat(prepared_dataset_factory: Callable[..., Path]) -> None:
    pipeline = _pipeline(prepared_dataset_factory, seq_len=4, batch_size=2)
    state = pipeline.initial_state()

    first = pipeline.next_batch(state)
    second = pipeline.next_batch(first.state)
    third = pipeline.next_batch(second.state)

    assert first.provenance.row_start_offsets == (0, 4)
    assert second.provenance.row_start_offsets == (8, 12)
    assert third.provenance.row_start_offsets == (16, 20)
    assert third.state.next_record_index == 6
    assert third.state.token_offset == 24
    with pytest.raises(StopIteration):
        pipeline.next_batch(third.state)
    pipeline.close()


def test_pipeline_construction_fails_when_split_too_small(prepared_dataset: Path) -> None:
    with pytest.raises(ContractError, match="one batch requires"):
        PreparedTokenGrainPipeline.from_manifest(
            prepared_dataset,
            tokenizer_id="toy-tokenizer",
            split="val",
            seq_len=2,
            batch_size=1,
        )


def test_data_pipeline_state_json_round_trips(prepared_dataset_factory: Callable[..., Path]) -> None:
    pipeline = _pipeline(prepared_dataset_factory, seq_len=4, batch_size=2)
    state = pipeline.initial_state()

    raw = data_pipeline_state_to_dict(state)
    restored = data_pipeline_state_from_mapping(raw)

    assert raw["manifest_path"].endswith("manifest.json")
    assert restored == state
    assert pipeline.state_from_json(raw) == state
    pipeline.close()


def test_grain_restore_reproduces_exact_next_batch(prepared_dataset_factory: Callable[..., Path]) -> None:
    first_pipeline = _pipeline(prepared_dataset_factory, seq_len=4, batch_size=2)
    state = first_pipeline.initial_state()
    first = first_pipeline.next_batch(state)
    expected_second = first_pipeline.next_batch(first.state)

    restored_pipeline = _pipeline(prepared_dataset_factory, seq_len=4, batch_size=2)
    restored_second = restored_pipeline.next_batch(first.state)

    np.testing.assert_array_equal(restored_second.batch.input_ids, expected_second.batch.input_ids)
    np.testing.assert_array_equal(restored_second.batch.target_ids, expected_second.batch.target_ids)
    np.testing.assert_array_equal(restored_second.batch.loss_mask, expected_second.batch.loss_mask)
    assert restored_second.provenance == expected_second.provenance
    first_pipeline.close()
    restored_pipeline.close()


def test_pipeline_compat_summary_changes_for_identity_fields(prepared_dataset_factory: Callable[..., Path]) -> None:
    first_manifest = _pipeline_manifest(prepared_dataset_factory)
    second_manifest = prepared_dataset_factory(
        "pipeline-other-tokenizer",
        tokenizer_id="other-tokenizer",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )

    base = data_pipeline_compat_payload(
        first_manifest,
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=4,
        batch_size=2,
    )
    assert data_pipeline_compat_payload(first_manifest, tokenizer_id="toy-tokenizer", split="val", seq_len=4, batch_size=1) != base
    assert data_pipeline_compat_payload(first_manifest, tokenizer_id="toy-tokenizer", split="train", seq_len=3, batch_size=2) != base
    assert data_pipeline_compat_payload(first_manifest, tokenizer_id="toy-tokenizer", split="train", seq_len=4, batch_size=1) != base
    assert data_pipeline_compat_payload(second_manifest, tokenizer_id="other-tokenizer", split="train", seq_len=4, batch_size=2) != base


def _pipeline(prepared_dataset_factory: Callable[..., Path], *, seq_len: int, batch_size: int) -> PreparedTokenGrainPipeline:
    return PreparedTokenGrainPipeline.from_manifest(
        _pipeline_manifest(prepared_dataset_factory),
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=seq_len,
        batch_size=batch_size,
    )


def _pipeline_manifest(prepared_dataset_factory: Callable[..., Path]) -> Path:
    return prepared_dataset_factory(
        "pipeline",
        shard_token_groups=(tuple(range(0, 10)), tuple(range(10, 20)), tuple(range(20, 30))),
        train_tokens=25,
    )
