"""Hugging Face streaming text pipeline for runtime training."""

from collections.abc import Mapping
from dataclasses import asdict
import json
import os
from typing import Any

from importlib import metadata as importlib_metadata
import numpy as np
import tiktoken

from jaxtitan.batch import Batch
from jaxtitan.data.pipeline import BatchProvenance, PipelineBatch, data_pipeline_state_to_dict
from jaxtitan.errors import ContractError
from jaxtitan.specs.data import HFStreamingSpec
from jaxtitan.state import DataPipelineState

HF_STREAMING_BACKEND = "hf_streaming"
HF_STREAMING_STATE_SCHEMA_VERSION = 2
HF_STREAMING_ORDER = "sequential"
HF_STREAMING_WORKER_COUNT = 0
HF_STREAMING_WORKER_BUFFER_SIZE = 1
HF_STREAMING_PREFETCH = False
HF_STREAMING_NUM_RECORDS_UNKNOWN = -1


class HFStreamingTextPipeline:
    """Runtime HF row stream -> Jaxtitan token batches."""

    def __init__(
        self,
        *,
        source: HFStreamingSpec,
        tokenizer_id: str,
        seq_len: int,
        batch_size: int,
    ) -> None:
        if seq_len <= 0:
            raise ContractError(f"seq_len must be positive, got {seq_len}")
        if batch_size <= 0:
            raise ContractError(f"batch_size must be positive, got {batch_size}")
        if not tokenizer_id:
            raise ContractError("tokenizer_id is required for HF streaming data")
        self.source = source
        self.tokenizer_id = tokenizer_id
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.split = source.split
        self.order = HF_STREAMING_ORDER
        self.shuffle_seed = None
        self.worker_count = HF_STREAMING_WORKER_COUNT
        self.worker_buffer_size = HF_STREAMING_WORKER_BUFFER_SIZE
        self.prefetch = HF_STREAMING_PREFETCH
        self.backend_version = datasets_version()
        self.tokenizer = _load_tokenizer(tokenizer_id)
        self.source_summary = json.dumps(_source_summary(source), sort_keys=True, separators=(",", ":"))
        self.sampler_summary = "hf_streaming_sequential"
        self.split_start = 0
        self.split_end = 0
        self.split_tokens = 0
        self._dataset: Any | None = None
        self._iterator: Any | None = None
        self._current_state: DataPipelineState | None = None
        self._hf_state: dict[str, Any] = {}
        self._current_doc_tokens: list[int] | None = None
        self._current_doc_offset = 0
        self._pending_tokens: list[int] = []
        self._rows_seen = 0
        self._docs_seen = 0
        self._next_record_index = 0

    def initial_state(self) -> DataPipelineState:
        self._dataset = _load_hf_dataset(self.source)
        self._iterator = iter(self._dataset)
        self._hf_state = _jsonable(self._dataset.state_dict())
        self._current_doc_tokens = None
        self._current_doc_offset = 0
        self._pending_tokens = []
        self._rows_seen = 0
        self._docs_seen = 0
        self._next_record_index = 0
        state = self._state_from_current()
        self._current_state = state
        return state

    def next_batch(self, state: DataPipelineState) -> PipelineBatch:
        self._validate_state(state)
        if self._current_state != state:
            self._restore_state(state)
        inputs = []
        targets = []
        masks = []
        row_offsets = []
        for _idx in range(self.batch_size):
            start = self._next_record_index * self.seq_len
            record = self._next_record()
            inputs.append(record[:-1])
            targets.append(record[1:])
            masks.append(np.ones((self.seq_len,), dtype=np.bool_))
            row_offsets.append(start)
            self._next_record_index += 1
        batch = Batch(
            input_ids=np.asarray(inputs, dtype=np.int32),
            target_ids=np.asarray(targets, dtype=np.int32),
            loss_mask=np.asarray(masks, dtype=np.bool_),
            doc_ids=None,
        )
        next_state = self._state_from_current()
        self._current_state = next_state
        return PipelineBatch(
            batch=batch,
            state=next_state,
            provenance=BatchProvenance(
                split=self.split,
                epoch=0,
                token_start=min(row_offsets),
                token_end=max(row_offsets) + self.seq_len,
                examples=self.batch_size,
                target_tokens=self.batch_size * self.seq_len,
                row_start_offsets=tuple(row_offsets),
                row_doc_ids=None,
            ),
        )

    def state_to_json(self, state: DataPipelineState) -> dict[str, Any]:
        self._validate_state(state)
        return data_pipeline_state_to_dict(state)

    def state_from_json(self, raw: Mapping[str, Any]) -> DataPipelineState:
        from jaxtitan.data.pipeline import data_pipeline_state_from_mapping

        state = data_pipeline_state_from_mapping(raw)
        self._validate_state(state)
        return state

    def describe(self) -> dict[str, Any]:
        return _jsonable(
            {
                "schema_version": 1,
                "backend": HF_STREAMING_BACKEND,
                "backend_version": self.backend_version,
                "state_schema_version": HF_STREAMING_STATE_SCHEMA_VERSION,
                "split": self.split,
                "order": self.order,
                "shuffle": False,
                "shuffle_seed": None,
                "num_epochs": 1,
                "worker_count": self.worker_count,
                "worker_buffer_size": self.worker_buffer_size,
                "prefetch": self.prefetch,
                "drop_remainder": True,
                "batch_size": self.batch_size,
                "seq_len": self.seq_len,
                "num_records": None,
                "manifest_path": None,
                "manifest_sha256": None,
                "tokenizer_id": self.tokenizer_id,
                "split_start": None,
                "split_end": None,
                "split_tokens": None,
                "document_aware": False,
                "document_count": None,
                "document_offsets_path": None,
                "document_offsets_sha256": None,
                "source": _source_summary(self.source),
                "source_summary": self.source_summary,
                "sampler_summary": self.sampler_summary,
                "exact_resume": True,
                "rows_seen": self._rows_seen,
                "documents_seen": self._docs_seen,
                "tokens_emitted": self._next_record_index * self.seq_len,
            }
        )

    def close(self) -> None:
        self._iterator = None
        self._dataset = None

    def _next_record(self) -> list[int]:
        self._fill_pending(self.seq_len + 1)
        record = self._pending_tokens[: self.seq_len + 1]
        self._pending_tokens = self._pending_tokens[self.seq_len :]
        return record

    def _fill_pending(self, target_size: int) -> None:
        while len(self._pending_tokens) < target_size:
            if self._current_doc_tokens is None or self._current_doc_offset >= len(self._current_doc_tokens):
                self._read_next_document()
            remaining = len(self._current_doc_tokens) - self._current_doc_offset
            take = min(target_size - len(self._pending_tokens), remaining)
            self._pending_tokens.extend(
                self._current_doc_tokens[self._current_doc_offset : self._current_doc_offset + take]
            )
            self._current_doc_offset += take
            if self._current_doc_offset >= len(self._current_doc_tokens):
                self._current_doc_tokens = None
                self._current_doc_offset = 0
                if self._dataset is not None:
                    self._hf_state = _jsonable(self._dataset.state_dict())

    def _read_next_document(self) -> None:
        if self._dataset is None or self._iterator is None:
            self._restore_state(self.initial_state())
        while True:
            self._hf_state = _jsonable(self._dataset.state_dict())
            try:
                row = next(self._iterator)
            except StopIteration as exc:
                raise StopIteration("HF streaming source ended before one full batch could be produced") from exc
            self._rows_seen += 1
            tokens = self._tokens_from_row(row)
            if not tokens:
                self._hf_state = _jsonable(self._dataset.state_dict())
                continue
            self._docs_seen += 1
            self._current_doc_tokens = tokens
            self._current_doc_offset = 0
            return

    def _tokens_from_row(self, row: Any) -> list[int]:
        if not isinstance(row, Mapping):
            raise ContractError(f"HF streaming rows must be mappings, got {type(row).__name__}")
        if self.source.text_column not in row:
            raise ContractError(f"HF streaming row is missing text column {self.source.text_column!r}")
        value = row[self.source.text_column]
        if not isinstance(value, str):
            raise ContractError(f"HF streaming text column {self.source.text_column!r} must contain strings")
        tokens = self.tokenizer.encode(value)
        if self.source.append_eot:
            tokens.append(self.tokenizer.eot_token)
        return tokens

    def _state_from_current(self) -> DataPipelineState:
        return DataPipelineState(
            schema_version=HF_STREAMING_STATE_SCHEMA_VERSION,
            backend=HF_STREAMING_BACKEND,
            backend_version=self.backend_version,
            split=self.split,
            order=self.order,
            shuffle_seed=None,
            worker_count=self.worker_count,
            worker_buffer_size=self.worker_buffer_size,
            prefetch=self.prefetch,
            manifest_path=None,
            manifest_sha256=None,
            tokenizer_id=self.tokenizer_id,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            num_records=HF_STREAMING_NUM_RECORDS_UNKNOWN,
            next_record_index=self._next_record_index,
            token_offset=self._next_record_index * self.seq_len,
            epoch=0,
            sampler_summary=self.sampler_summary,
            source_summary=self.source_summary,
            grain_state={},
            stream_state={
                "hf_state": _jsonable(self._hf_state),
                "current_doc_active": self._current_doc_tokens is not None,
                "current_doc_token_offset": self._current_doc_offset,
                "pending_tokens": list(self._pending_tokens),
                "rows_seen": self._rows_seen,
                "docs_seen": self._docs_seen,
                "tokens_emitted": self._next_record_index * self.seq_len,
            },
        )

    def _restore_state(self, state: DataPipelineState) -> None:
        stream_state = state.stream_state
        hf_state = _require_mapping(stream_state.get("hf_state"), "stream_state.hf_state")
        self._dataset = _load_hf_dataset(self.source)
        self._dataset.load_state_dict(dict(hf_state))
        self._iterator = iter(self._dataset)
        self._hf_state = _jsonable(hf_state)
        self._pending_tokens = _int_list(stream_state.get("pending_tokens", []), "stream_state.pending_tokens")
        self._rows_seen = _required_int(stream_state, "rows_seen", "stream_state")
        self._docs_seen = _required_int(stream_state, "docs_seen", "stream_state")
        self._next_record_index = state.next_record_index
        self._current_doc_tokens = None
        self._current_doc_offset = 0
        if bool(stream_state.get("current_doc_active", False)):
            try:
                row = next(self._iterator)
            except StopIteration as exc:
                raise ContractError("HF streaming state points at a missing current document") from exc
            tokens = self._tokens_from_row(row)
            offset = _required_int(stream_state, "current_doc_token_offset", "stream_state")
            if offset < 0 or offset > len(tokens):
                raise ContractError(
                    f"stream_state.current_doc_token_offset={offset} is outside current document tokens"
                )
            self._current_doc_tokens = tokens
            self._current_doc_offset = offset
        self._current_state = state

    def _validate_state(self, state: DataPipelineState) -> None:
        checks = (
            ("schema_version", state.schema_version, HF_STREAMING_STATE_SCHEMA_VERSION),
            ("backend", state.backend, HF_STREAMING_BACKEND),
            ("backend_version", state.backend_version, self.backend_version),
            ("split", state.split, self.split),
            ("order", state.order, self.order),
            ("shuffle_seed", state.shuffle_seed, None),
            ("worker_count", state.worker_count, self.worker_count),
            ("worker_buffer_size", state.worker_buffer_size, self.worker_buffer_size),
            ("prefetch", state.prefetch, self.prefetch),
            ("manifest_path", state.manifest_path, None),
            ("manifest_sha256", state.manifest_sha256, None),
            ("tokenizer_id", state.tokenizer_id, self.tokenizer_id),
            ("seq_len", state.seq_len, self.seq_len),
            ("batch_size", state.batch_size, self.batch_size),
            ("num_records", state.num_records, HF_STREAMING_NUM_RECORDS_UNKNOWN),
            ("sampler_summary", state.sampler_summary, self.sampler_summary),
            ("source_summary", state.source_summary, self.source_summary),
        )
        for name, actual, wanted in checks:
            if actual != wanted:
                raise ContractError(f"HF streaming state {name} mismatch: state={actual!r} pipeline={wanted!r}")
        if state.next_record_index < 0:
            raise ContractError(f"HF streaming next_record_index must be non-negative, got {state.next_record_index}")
        if state.token_offset != state.next_record_index * self.seq_len:
            raise ContractError("HF streaming token_offset must match next_record_index * seq_len")
        if state.epoch != 0:
            raise ContractError("HF streaming supports only epoch=0 in this slice")
        _require_mapping(state.stream_state.get("hf_state"), "stream_state.hf_state")


def build_hf_streaming_pipeline(
    source: HFStreamingSpec,
    *,
    tokenizer_id: str | None,
    seq_len: int,
    batch_size: int,
    order: str,
    shuffle_seed: int | None,
    worker_count: int,
    worker_buffer_size: int,
    prefetch: bool,
) -> HFStreamingTextPipeline:
    if tokenizer_id is None:
        raise ContractError("data.tokenizer_id is required when data.mode='hf_streaming'")
    if order != HF_STREAMING_ORDER:
        raise ContractError("HF streaming supports only data.order='sequential'")
    if shuffle_seed is not None:
        raise ContractError("HF streaming does not support data.shuffle_seed")
    if worker_count != 0 or worker_buffer_size != 1 or prefetch:
        raise ContractError("HF streaming requires worker_count=0, worker_buffer_size=1, and prefetch=false")
    return HFStreamingTextPipeline(source=source, tokenizer_id=tokenizer_id, seq_len=seq_len, batch_size=batch_size)


def hf_streaming_compat_payload(
    source: HFStreamingSpec,
    *,
    tokenizer_id: str,
    seq_len: int,
    batch_size: int,
) -> dict[str, Any]:
    return _jsonable(
        {
            "backend": HF_STREAMING_BACKEND,
            "backend_version": datasets_version(),
            "state_schema_version": HF_STREAMING_STATE_SCHEMA_VERSION,
            "source": _source_summary(source),
            "tokenizer_id": tokenizer_id,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "order": HF_STREAMING_ORDER,
            "exact_resume": True,
        }
    )


def datasets_version() -> str | None:
    try:
        return importlib_metadata.version("datasets")
    except importlib_metadata.PackageNotFoundError:
        return None


def _load_hf_dataset(source: HFStreamingSpec) -> Any:
    _configure_hf_imports()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ContractError("HF streaming requires the 'datasets' package") from exc
    kwargs: dict[str, Any] = {
        "split": source.split,
        "streaming": True,
        "revision": source.revision,
    }
    if source.data_dir is not None:
        kwargs["data_dir"] = source.data_dir
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        kwargs["token"] = token
    try:
        if source.name is None:
            return load_dataset(source.dataset, **kwargs)
        return load_dataset(source.dataset, source.name, **kwargs)
    except Exception as exc:
        raise ContractError(f"failed to load HF streaming dataset {source.dataset!r}: {exc}") from exc


def _configure_hf_imports() -> None:
    os.environ.setdefault("USE_TORCH", "0")
    _patch_multiprocess_resource_tracker()


def _patch_multiprocess_resource_tracker() -> None:
    try:
        from multiprocess import resource_tracker
    except Exception:
        return
    current = resource_tracker.ResourceTracker.__del__
    if getattr(current, "__name__", "") == "_jaxtitan_streaming_resource_tracker_del":
        return

    def _jaxtitan_streaming_resource_tracker_del(self) -> None:
        try:
            self._stop(use_blocking_lock=False)
        except AttributeError as exc:
            if "_recursion_count" not in str(exc):
                raise

    resource_tracker.ResourceTracker.__del__ = _jaxtitan_streaming_resource_tracker_del


def _load_tokenizer(name: str) -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:
        raise ContractError(f"unsupported tiktoken tokenizer {name!r}: {exc}") from exc


def _source_summary(source: HFStreamingSpec) -> dict[str, Any]:
    return _jsonable(asdict(source))


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a JSON object")
    return value


def _required_int(raw: Mapping[str, Any], key: str, name: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"{name}.{key} must be an integer")
    return value


def _int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ContractError(f"{name} must be a list of integers")
    return list(value)
