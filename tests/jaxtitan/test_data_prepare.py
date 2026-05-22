import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tiktoken

from jaxtitan.data import (
    PreparedTokenDocumentBufferPipeline,
    load_hf_texts,
    load_prepare_config,
    prepare_config_from_mapping,
    prepare_dataset,
    prepare_result_to_dict,
    validate_dataset_manifest,
)
from jaxtitan.data.prepare import PrepareSourceConfig
from jaxtitan.errors import ConfigError, ContractError


def test_load_prepare_config_parses_hf_source(tmp_path: Path) -> None:
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(_prepare_config(tmp_path / "prepared"), encoding="utf-8")

    config = load_prepare_config(config_path)

    assert config.source.type == "hf"
    assert config.source.dataset == "HuggingFaceFW/fineweb"
    assert config.source.name == "sample-10BT"
    assert config.source.split == "train"
    assert config.source.text_column == "text"
    assert config.source.streaming is True
    assert config.tokenizer.name == "gpt2"
    assert config.tokenizer.append_eot is True
    assert config.output.max_tokens == 64
    assert config.output.val_fraction == 0.25
    assert config.output.shard_tokens == 7
    assert config.tokenization.workers == 1
    assert config.tokenization.batch_docs == 2
    assert config.tokenization.queue_batches == 1


@pytest.mark.parametrize("source_type", ["parquet", "jsonl", "text"])
def test_load_prepare_config_parses_local_sources(tmp_path: Path, source_type: str) -> None:
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(_local_prepare_config(tmp_path / "prepared", source_type, ["~/data/*.txt"]), encoding="utf-8")

    config = load_prepare_config(config_path)

    assert config.source.type == source_type
    assert config.source.paths == ("~/data/*.txt",)
    assert config.source.dataset is None
    assert config.source.split is None
    assert config.source.text_column == "text"


@pytest.mark.parametrize(
    "raw,match",
    [
        ({"tokenizer": {"name": "gpt2"}, "output": {"path": "x", "max_tokens": 1}}, "missing required"),
        (
            {
                "source": {"type": "hf", "dataset": "d", "split": "train"},
                "tokenizer": {"name": "gpt2"},
                "output": {"path": "x"},
            },
            "output.max_tokens",
        ),
        (
            {
                "source": {"type": "hf", "dataset": "d", "split": "train"},
                "tokenizer": {"name": "gpt2"},
                "output": {"path": "x", "max_tokens": 1, "val_fraction": 1.0},
            },
            "val_fraction",
        ),
        (
            {
                "source": {"type": "text"},
                "tokenizer": {"name": "gpt2"},
                "output": {"path": "x", "max_tokens": 1},
            },
            "source.paths",
        ),
        (
            {
                "source": {"type": "csv", "paths": ["data.csv"]},
                "tokenizer": {"name": "gpt2"},
                "output": {"path": "x", "max_tokens": 1},
            },
            "source.type",
        ),
    ],
)
def test_prepare_config_rejects_invalid_values(raw: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        prepare_config_from_mapping(raw)


def test_load_hf_texts_passes_expected_dataset_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"content": "one"}, {"content": "two"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    monkeypatch.setenv("HF_TOKEN", "secret-token")
    source = PrepareSourceConfig(
        type="hf",
        dataset="bigcode/the-stack-dedup",
        name="data",
        data_dir="python",
        split="train",
        text_column="content",
        streaming=False,
    )

    texts = list(load_hf_texts(source))

    assert texts == ["one", "two"]
    assert os.environ["USE_TORCH"] == "0"
    assert calls == [
        (
            ("bigcode/the-stack-dedup", "data"),
            {
                "split": "train",
                "streaming": False,
                "data_dir": "python",
                "token": "secret-token",
            },
        )
    ]


def test_load_hf_texts_rejects_missing_text_column(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda *args, **kwargs: [{"other": "value"}]),
    )
    source = PrepareSourceConfig(type="hf", dataset="dataset", split="train", text_column="text")

    with pytest.raises(ContractError, match="missing text column"):
        list(load_hf_texts(source))


def test_prepare_dataset_writes_valid_manifest_shards_and_document_offsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = ["hello", "world", "goodbye"]
    monkeypatch.setattr("jaxtitan.data.prepare.load_hf_texts", lambda source: docs)
    config_path = tmp_path / "prepare.toml"
    output = tmp_path / "prepared"
    config_path.write_text(_prepare_config(output, max_tokens=64, shard_tokens=5), encoding="utf-8")

    result = prepare_dataset(config_path, quiet=True)
    manifest = validate_dataset_manifest(result.manifest.manifest_path, tokenizer_id="gpt2", verify_checksums=True)

    tokenizer = tiktoken.get_encoding("gpt2")
    expected_lengths = [len(tokenizer.encode(doc)) + 1 for doc in docs]
    expected_offsets = np.asarray([0, *np.cumsum(expected_lengths).tolist()], dtype=np.uint64)
    actual_offsets = np.fromfile(output / "document_offsets.u64", dtype=np.uint64)
    all_tokens = np.concatenate([np.fromfile(output / shard.path, dtype=np.uint32) for shard in manifest.shards])

    assert manifest.num_tokens == int(sum(expected_lengths))
    assert manifest.train_tokens == int(manifest.num_tokens * 0.75)
    assert manifest.val_tokens == manifest.num_tokens - manifest.train_tokens
    assert manifest.shard_count > 1
    assert manifest.documents is not None
    assert manifest.documents.count == len(docs)
    np.testing.assert_array_equal(actual_offsets, expected_offsets)
    assert int(all_tokens[expected_offsets[1] - 1]) == tokenizer.eot_token
    assert prepare_result_to_dict(result)["source"]["dataset"] == "HuggingFaceFW/fineweb"


def test_prepare_dataset_writes_local_parquet_source(tmp_path: Path) -> None:
    source_path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"text": ["hello", "world"], "other": [1, 2]}), source_path)
    config = load_prepare_config_from_text(
        tmp_path,
        _local_prepare_config(tmp_path / "prepared-parquet", "parquet", [source_path.as_posix()]),
    )

    result = prepare_dataset(config, quiet=True)
    manifest = validate_dataset_manifest(result.manifest.manifest_path, tokenizer_id="gpt2", verify_checksums=True)

    assert manifest.documents is not None
    assert manifest.documents.count == 2
    assert result.raw_manifest["source"]["type"] == "parquet"
    assert result.raw_manifest["source"]["paths"] == [source_path.as_posix()]
    assert result.raw_manifest["source"]["resolved_file_count"] == 1
    assert result.raw_manifest["source"]["resolved_total_bytes"] == source_path.stat().st_size


def test_prepare_dataset_writes_local_jsonl_source(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"text":"alpha"}\n{"text":"beta"}\n', encoding="utf-8")
    config = load_prepare_config_from_text(
        tmp_path,
        _local_prepare_config(tmp_path / "prepared-jsonl", "jsonl", [source_path.as_posix()]),
    )

    result = prepare_dataset(config, quiet=True)
    manifest = validate_dataset_manifest(result.manifest.manifest_path, tokenizer_id="gpt2", verify_checksums=True)

    assert manifest.documents is not None
    assert manifest.documents.count == 2
    assert result.raw_manifest["source"]["type"] == "jsonl"


def test_prepare_dataset_writes_local_text_source_one_line_per_document(tmp_path: Path) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_text("first\n\nthird\n", encoding="utf-8")
    config = load_prepare_config_from_text(
        tmp_path,
        _local_prepare_config(tmp_path / "prepared-text", "text", [source_path.as_posix()]),
    )

    result = prepare_dataset(config, quiet=True)
    offsets = np.fromfile(tmp_path / "prepared-text" / "document_offsets.u64", dtype=np.uint64)

    assert result.manifest.documents is not None
    assert result.manifest.documents.count == 3
    assert len(offsets) == 4
    assert result.raw_manifest["source"]["type"] == "text"
    assert result.raw_manifest["source"]["text_column"] is None


def test_prepare_dataset_local_sources_are_sorted_and_globbed(tmp_path: Path) -> None:
    second = tmp_path / "b.jsonl"
    first = tmp_path / "a.jsonl"
    second.write_text('{"text":"second"}\n', encoding="utf-8")
    first.write_text('{"text":"first"}\n', encoding="utf-8")
    config = load_prepare_config_from_text(
        tmp_path,
        _local_prepare_config(tmp_path / "prepared-glob", "jsonl", [(tmp_path / "*.jsonl").as_posix()]),
    )

    result = prepare_dataset(config, quiet=True)

    assert result.raw_manifest["source"]["resolved_file_count"] == 2
    assert result.raw_manifest["source"]["resolved_total_bytes"] == first.stat().st_size + second.stat().st_size


def test_prepare_dataset_rejects_bad_local_sources(tmp_path: Path) -> None:
    config = load_prepare_config_from_text(
        tmp_path,
        _local_prepare_config(tmp_path / "bad-glob", "jsonl", [(tmp_path / "*.missing").as_posix()]),
    )
    with pytest.raises(ContractError, match="did not match"):
        prepare_dataset(config, quiet=True)

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text('{"other":"value"}\n', encoding="utf-8")
    config = load_prepare_config_from_text(
        tmp_path,
        _local_prepare_config(tmp_path / "bad-jsonl", "jsonl", [bad_jsonl.as_posix()]),
    )
    with pytest.raises(ContractError, match="missing text column"):
        prepare_dataset(config, quiet=True)

    bad_parquet = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"other": ["value"]}), bad_parquet)
    config = load_prepare_config_from_text(
        tmp_path,
        _local_prepare_config(tmp_path / "bad-parquet", "parquet", [bad_parquet.as_posix()]),
    )
    with pytest.raises(ContractError, match="missing text column"):
        prepare_dataset(config, quiet=True)


def test_prepare_dataset_respects_max_tokens_and_document_buffer_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("jaxtitan.data.prepare.load_hf_texts", lambda source: ["alpha beta gamma", "delta"])
    config_path = tmp_path / "prepare.toml"
    output = tmp_path / "prepared"
    config_path.write_text(_prepare_config(output, max_tokens=5, shard_tokens=3), encoding="utf-8")

    result = prepare_dataset(config_path, quiet=True)
    offsets = np.fromfile(output / "document_offsets.u64", dtype=np.uint64)
    pipeline = PreparedTokenDocumentBufferPipeline.from_manifest(
        result.manifest.manifest_path,
        tokenizer_id="gpt2",
        split="train",
        seq_len=2,
        batch_size=1,
        shuffle_seed=123,
        document_buffer_size=1,
        document_refill_size=1,
    )

    assert result.manifest.num_tokens == 5
    assert int(offsets[-1]) == 5
    batch = pipeline.next_batch(pipeline.initial_state())
    assert batch.batch.input_ids.shape == (1, 2)
    assert batch.batch.doc_ids is not None
    pipeline.close()


def test_prepare_dataset_rejects_existing_output_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("jaxtitan.data.prepare.load_hf_texts", lambda source: ["hello"])
    output = tmp_path / "prepared"
    output.mkdir()
    config = load_prepare_config_from_text(tmp_path, _prepare_config(output, max_tokens=8))

    with pytest.raises(ContractError, match="already exists"):
        prepare_dataset(config, quiet=True)

    result = prepare_dataset(config, overwrite=True, quiet=True)
    assert result.manifest.manifest_path == output / "manifest.json"


def test_prepare_dataset_rejects_zero_token_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("jaxtitan.data.prepare.load_hf_texts", lambda source: [""])
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(_prepare_config(tmp_path / "prepared", max_tokens=8, append_eot=False), encoding="utf-8")

    with pytest.raises(ContractError, match="zero tokens"):
        prepare_dataset(config_path, quiet=True)


def load_prepare_config_from_text(tmp_path: Path, text: str):
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(text, encoding="utf-8")
    return load_prepare_config(config_path)


def _prepare_config(
    output: Path,
    *,
    max_tokens: int = 64,
    shard_tokens: int = 7,
    append_eot: bool = True,
) -> str:
    return f"""
[source]
type = "hf"
dataset = "HuggingFaceFW/fineweb"
name = "sample-10BT"
split = "train"
text_column = "text"
streaming = true

[tokenizer]
name = "gpt2"
append_eot = {str(append_eot).lower()}

[output]
path = "{output.as_posix()}"
max_tokens = {max_tokens}
val_fraction = 0.25
shard_tokens = {shard_tokens}

[tokenization]
workers = 1
batch_docs = 2
queue_batches = 1
"""


def _local_prepare_config(output: Path, source_type: str, paths: list[str]) -> str:
    paths_toml = ", ".join(json.dumps(path) for path in paths)
    return f"""
[source]
type = "{source_type}"
paths = [{paths_toml}]
text_column = "text"

[tokenizer]
name = "gpt2"
append_eot = true

[output]
path = "{output.as_posix()}"
max_tokens = 64
val_fraction = 0.25
shard_tokens = 8

[tokenization]
workers = 1
batch_docs = 2
queue_batches = 1
"""
