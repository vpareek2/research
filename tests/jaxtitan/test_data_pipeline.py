from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from jaxtitan.data import (
    PreparedDatasetManifest,
    PreparedTokenDataSource,
    PreparedTokenDocumentBufferPipeline,
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
from jaxtitan.runtime.training import _combine_provenance, _stack_accumulated_batches


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
        schema_version=1,
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
        documents=None,
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


def test_data_source_emits_document_ids_for_manifest_with_offsets(
    prepared_dataset_factory: Callable[..., Path],
) -> None:
    source = PreparedTokenDataSource.from_manifest(
        _document_manifest(prepared_dataset_factory),
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=4,
    )

    first = source[0]
    second = source[1]
    cross_shard = source[3]

    assert source.document_aware is True
    assert source.document_count == 4
    assert int(first["doc_id"]) == 0
    assert int(second["doc_id"]) == 0
    assert int(cross_shard["doc_id"]) == 2


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
    assert result.batch.doc_ids is None
    np.testing.assert_array_equal(result.batch.input_ids, np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32))
    np.testing.assert_array_equal(result.batch.target_ids, np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32))
    assert result.batch.loss_mask.all()
    assert result.state.backend == "grain"
    assert result.state.order == "sequential"
    assert result.state.shuffle_seed is None
    assert result.state.worker_count == 0
    assert result.state.worker_buffer_size == 1
    assert result.state.prefetch is False
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


def test_pipeline_batches_document_ids_when_manifest_is_document_aware(
    prepared_dataset_factory: Callable[..., Path],
) -> None:
    pipeline = PreparedTokenGrainPipeline.from_manifest(
        _document_manifest(prepared_dataset_factory),
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=4,
        batch_size=3,
    )
    state = pipeline.initial_state()

    result = pipeline.next_batch(state)

    assert result.batch.doc_ids is not None
    assert result.batch.doc_ids.shape == (3,)
    assert result.batch.doc_ids.dtype == np.int32
    np.testing.assert_array_equal(result.batch.doc_ids, np.asarray([0, 0, 1], dtype=np.int32))
    assert result.provenance.row_doc_ids == (0, 0, 1)
    description = pipeline.describe()
    assert description["source_summary"].startswith("PreparedTokenDataSource(")
    assert "document_aware=True" in description["source_summary"]
    assert description["document_aware"] is True
    assert description["document_count"] == 4
    assert description["document_offsets_path"] == "document_offsets.u64"
    pipeline.close()


def test_gradient_accumulation_combines_document_provenance(
    prepared_dataset_factory: Callable[..., Path],
) -> None:
    pipeline = PreparedTokenGrainPipeline.from_manifest(
        _document_manifest(prepared_dataset_factory),
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=4,
        batch_size=2,
    )
    state = pipeline.initial_state()
    first = pipeline.next_batch(state)
    second = pipeline.next_batch(first.state)

    batch = _stack_accumulated_batches([first.batch, second.batch])
    provenance = _combine_provenance([first.provenance, second.provenance])

    assert batch.doc_ids is not None
    assert batch.doc_ids.shape == (2, 2)
    np.testing.assert_array_equal(batch.doc_ids, np.asarray([[0, 0], [1, 2]], dtype=np.int32))
    assert provenance.row_doc_ids == (0, 0, 1, 2)
    pipeline.close()


def test_gradient_accumulation_rejects_mixed_document_batches(
    prepared_dataset_factory: Callable[..., Path],
) -> None:
    document_pipeline = PreparedTokenGrainPipeline.from_manifest(
        _document_manifest(prepared_dataset_factory),
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=4,
        batch_size=2,
    )
    token_pipeline = _pipeline(prepared_dataset_factory, seq_len=4, batch_size=2)
    document_batch = document_pipeline.next_batch(document_pipeline.initial_state())
    token_batch = token_pipeline.next_batch(token_pipeline.initial_state())

    with pytest.raises(ContractError, match="doc_ids"):
        _stack_accumulated_batches([document_batch.batch, token_batch.batch])
    with pytest.raises(ContractError, match="document ids"):
        _combine_provenance([document_batch.provenance, token_batch.provenance])
    document_pipeline.close()
    token_pipeline.close()


def test_document_buffer_batches_are_seeded_and_mask_boundaries(
    prepared_dataset_factory: Callable[..., Path],
) -> None:
    first = _document_buffer_pipeline(prepared_dataset_factory, shuffle_seed=11)
    second = _document_buffer_pipeline(prepared_dataset_factory, shuffle_seed=11)
    third = _document_buffer_pipeline(prepared_dataset_factory, shuffle_seed=12)

    first_result = first.next_batch(first.initial_state())
    second_result = second.next_batch(second.initial_state())
    third_result = third.next_batch(third.initial_state())

    np.testing.assert_array_equal(first_result.batch.input_ids, second_result.batch.input_ids)
    np.testing.assert_array_equal(first_result.batch.target_ids, second_result.batch.target_ids)
    np.testing.assert_array_equal(first_result.batch.loss_mask, second_result.batch.loss_mask)
    np.testing.assert_array_equal(first_result.batch.doc_ids, second_result.batch.doc_ids)
    assert first_result.provenance == second_result.provenance
    assert first_result.batch.doc_ids is not None
    assert first_result.provenance.row_doc_ids == tuple(int(value) for value in first_result.batch.doc_ids.tolist())
    assert not first_result.batch.loss_mask.all()
    assert not np.array_equal(first_result.batch.input_ids, third_result.batch.input_ids)
    first.close()
    second.close()
    third.close()


def test_document_buffer_restore_reproduces_exact_next_batch(
    prepared_dataset_factory: Callable[..., Path],
) -> None:
    pipeline = _document_buffer_pipeline(prepared_dataset_factory, shuffle_seed=7)
    state = pipeline.initial_state()
    first = pipeline.next_batch(state)
    expected_second = pipeline.next_batch(first.state)

    restored = _document_buffer_pipeline(prepared_dataset_factory, shuffle_seed=7)
    restored_second = restored.next_batch(first.state)

    np.testing.assert_array_equal(restored_second.batch.input_ids, expected_second.batch.input_ids)
    np.testing.assert_array_equal(restored_second.batch.target_ids, expected_second.batch.target_ids)
    np.testing.assert_array_equal(restored_second.batch.loss_mask, expected_second.batch.loss_mask)
    np.testing.assert_array_equal(restored_second.batch.doc_ids, expected_second.batch.doc_ids)
    assert restored_second.provenance == expected_second.provenance
    pipeline.close()
    restored.close()


def test_document_buffer_requires_document_offsets(prepared_dataset_factory: Callable[..., Path]) -> None:
    with pytest.raises(ContractError, match="document offsets"):
        PreparedTokenDocumentBufferPipeline.from_manifest(
            _pipeline_manifest(prepared_dataset_factory),
            tokenizer_id="toy-tokenizer",
            split="train",
            seq_len=4,
            batch_size=2,
            shuffle_seed=1,
            document_buffer_size=2,
            document_refill_size=2,
        )


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


@pytest.mark.parametrize("order_kwargs", [{}, {"order": "shuffle", "shuffle_seed": 123}])
def test_grain_restore_reproduces_exact_next_batch(
    prepared_dataset_factory: Callable[..., Path],
    order_kwargs: dict,
) -> None:
    first_pipeline = _pipeline(prepared_dataset_factory, seq_len=4, batch_size=2)
    if order_kwargs:
        first_pipeline.close()
        first_pipeline = _pipeline(prepared_dataset_factory, seq_len=4, batch_size=2, **order_kwargs)
    state = first_pipeline.initial_state()
    first = first_pipeline.next_batch(state)
    expected_second = first_pipeline.next_batch(first.state)

    restored_pipeline = _pipeline(prepared_dataset_factory, seq_len=4, batch_size=2, **order_kwargs)
    restored_second = restored_pipeline.next_batch(first.state)

    np.testing.assert_array_equal(restored_second.batch.input_ids, expected_second.batch.input_ids)
    np.testing.assert_array_equal(restored_second.batch.target_ids, expected_second.batch.target_ids)
    np.testing.assert_array_equal(restored_second.batch.loss_mask, expected_second.batch.loss_mask)
    assert restored_second.provenance == expected_second.provenance
    first_pipeline.close()
    restored_pipeline.close()


def test_shuffle_order_is_seeded_and_deterministic(prepared_dataset_factory: Callable[..., Path]) -> None:
    first = _pipeline(prepared_dataset_factory, seq_len=4, batch_size=2, order="shuffle", shuffle_seed=7)
    second = _pipeline(prepared_dataset_factory, seq_len=4, batch_size=2, order="shuffle", shuffle_seed=7)
    third = _pipeline(prepared_dataset_factory, seq_len=4, batch_size=2, order="shuffle", shuffle_seed=8)

    first_result = first.next_batch(first.initial_state())
    second_result = second.next_batch(second.initial_state())
    third_result = third.next_batch(third.initial_state())

    np.testing.assert_array_equal(first_result.batch.input_ids, second_result.batch.input_ids)
    assert first_result.provenance.row_start_offsets == second_result.provenance.row_start_offsets
    assert first_result.provenance.row_start_offsets != (0, 4)
    assert first_result.provenance.row_start_offsets != third_result.provenance.row_start_offsets
    first.close()
    second.close()
    third.close()


def test_pipeline_rejects_invalid_loader_policy(prepared_dataset_factory: Callable[..., Path]) -> None:
    with pytest.raises(ContractError, match="shuffle_seed"):
        _pipeline(prepared_dataset_factory, seq_len=4, batch_size=2, order="shuffle")
    with pytest.raises(ContractError, match="worker_buffer_size"):
        _pipeline(prepared_dataset_factory, seq_len=4, batch_size=2, worker_buffer_size=0)


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
    assert data_pipeline_compat_payload(first_manifest, tokenizer_id="toy-tokenizer", split="train", seq_len=4, batch_size=2, order="shuffle", shuffle_seed=123) != base
    assert data_pipeline_compat_payload(first_manifest, tokenizer_id="toy-tokenizer", split="train", seq_len=4, batch_size=2, worker_count=1) != base
    assert data_pipeline_compat_payload(first_manifest, tokenizer_id="toy-tokenizer", split="train", seq_len=4, batch_size=2, worker_buffer_size=2) != base
    assert data_pipeline_compat_payload(first_manifest, tokenizer_id="toy-tokenizer", split="train", seq_len=4, batch_size=2, prefetch=True) != base
    assert data_pipeline_compat_payload(second_manifest, tokenizer_id="other-tokenizer", split="train", seq_len=4, batch_size=2) != base


def _pipeline(
    prepared_dataset_factory: Callable[..., Path],
    *,
    seq_len: int,
    batch_size: int,
    order: str = "sequential",
    shuffle_seed: int | None = None,
    worker_count: int = 0,
    worker_buffer_size: int = 1,
    prefetch: bool = False,
) -> PreparedTokenGrainPipeline:
    return PreparedTokenGrainPipeline.from_manifest(
        _pipeline_manifest(prepared_dataset_factory),
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=seq_len,
        batch_size=batch_size,
        order=order,
        shuffle_seed=shuffle_seed,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
        prefetch=prefetch,
    )


def _pipeline_manifest(prepared_dataset_factory: Callable[..., Path]) -> Path:
    return prepared_dataset_factory(
        "pipeline",
        shard_token_groups=(tuple(range(0, 10)), tuple(range(10, 20)), tuple(range(20, 30))),
        train_tokens=25,
    )


def _document_manifest(prepared_dataset_factory: Callable[..., Path]) -> Path:
    return prepared_dataset_factory(
        "pipeline-documents",
        shard_token_groups=(tuple(range(0, 10)), tuple(range(10, 20)), tuple(range(20, 30))),
        train_tokens=25,
        document_offsets=(0, 8, 12, 21, 30),
    )


def _document_buffer_manifest(prepared_dataset_factory: Callable[..., Path]) -> Path:
    return prepared_dataset_factory(
        "pipeline-document-buffer",
        shard_token_groups=(tuple(range(0, 40)),),
        train_tokens=30,
        document_offsets=(0, 3, 6, 9, 12, 15, 20, 25, 30, 40),
    )


def _document_buffer_pipeline(
    prepared_dataset_factory: Callable[..., Path],
    *,
    shuffle_seed: int,
) -> PreparedTokenDocumentBufferPipeline:
    return PreparedTokenDocumentBufferPipeline.from_manifest(
        _document_buffer_manifest(prepared_dataset_factory),
        tokenizer_id="toy-tokenizer",
        split="train",
        seq_len=4,
        batch_size=2,
        shuffle_seed=shuffle_seed,
        document_buffer_size=2,
        document_refill_size=2,
    )
