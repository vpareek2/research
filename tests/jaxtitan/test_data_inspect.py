import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from jaxtitan.data import (
    data_config_snippet,
    data_inspection_to_dict,
    data_inspection_to_json,
    format_data_inspection,
    inspect_dataset_manifest,
    load_prepare_config,
    prepare_dataset,
)
from jaxtitan.errors import ContractError


def test_inspect_generated_hf_manifest_human_and_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("jaxtitan.data.prepare.load_hf_texts", lambda source: ["hello", "world"])
    config = load_prepare_config_from_text(tmp_path, _prepare_config(tmp_path / "prepared"))
    result = prepare_dataset(config, quiet=True)

    inspection = inspect_dataset_manifest(result.manifest.manifest_path, tokenizer_id="gpt2", seq_len=2)
    payload = data_inspection_to_dict(inspection)
    text = format_data_inspection(inspection)

    assert payload["source"]["dataset"] == "HuggingFaceFW/fineweb"
    assert payload["tokenizer"]["append_eot"] is True
    assert payload["tokens"]["total"] == result.manifest.num_tokens
    assert payload["documents"]["count"] == 2
    assert payload["shards"]["checksums_present"] is True
    assert payload["records"]["train"] == max(0, (result.manifest.train_tokens - 1) // 2)
    assert 'order = "document_buffer"' in payload["data_config_toml"]
    assert "training config:" in text
    assert "HuggingFaceFW/fineweb" in text
    assert json.loads(data_inspection_to_json(inspection))["manifest"]["tokenizer_id"] == "gpt2"


def test_inspect_generated_local_manifest_human_and_json(tmp_path: Path) -> None:
    source_path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"text": ["alpha", "beta"]}), source_path)
    config = load_prepare_config_from_text(
        tmp_path,
        _local_prepare_config(tmp_path / "prepared", "parquet", [source_path.as_posix()]),
    )
    result = prepare_dataset(config, quiet=True)

    inspection = inspect_dataset_manifest(result.manifest.manifest_path, tokenizer_id="gpt2", seq_len=2)
    payload = data_inspection_to_dict(inspection)
    text = format_data_inspection(inspection)

    assert payload["source"]["type"] == "parquet"
    assert payload["source"]["resolved_file_count"] == 1
    assert payload["source"]["resolved_total_bytes"] == source_path.stat().st_size
    assert payload["documents"]["count"] == 2
    assert "source: type=parquet files=1" in text
    assert json.loads(data_inspection_to_json(inspection))["source"]["type"] == "parquet"


def test_inspect_tolerates_older_manifest_without_source(
    prepared_dataset: Path,
) -> None:
    inspection = inspect_dataset_manifest(prepared_dataset, tokenizer_id="toy-tokenizer", seq_len=4)
    payload = data_inspection_to_dict(inspection)
    text = format_data_inspection(inspection)

    assert payload["source"] is None
    assert payload["records"] == {"seq_len": 4, "train": 1, "val": 0}
    assert payload["data_config_toml"] == data_config_snippet(inspection.manifest)
    assert 'order = "sequential"' in payload["data_config_toml"]
    assert "dataset=None" in text


def test_inspect_rejects_tokenizer_and_checksum_mismatch(
    prepared_dataset_factory,
) -> None:
    manifest_path = prepared_dataset_factory("inspect-bad-checksum")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["shards"][0]["sha256"] = "bad"
    manifest_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ContractError, match="does not match config tokenizer"):
        inspect_dataset_manifest(manifest_path, tokenizer_id="wrong-tokenizer")
    with pytest.raises(ContractError, match="checksum mismatch"):
        inspect_dataset_manifest(manifest_path, tokenizer_id="toy-tokenizer", verify_checksums=True)


def test_inspect_rejects_invalid_seq_len(prepared_dataset: Path) -> None:
    with pytest.raises(ContractError, match="seq-len"):
        inspect_dataset_manifest(prepared_dataset, seq_len=0)


def load_prepare_config_from_text(tmp_path: Path, text: str):
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(text, encoding="utf-8")
    return load_prepare_config(config_path)


def _prepare_config(output: Path) -> str:
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
