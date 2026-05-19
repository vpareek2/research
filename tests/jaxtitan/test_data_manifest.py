from collections.abc import Callable
import json
from pathlib import Path

import pytest

from jaxtitan.data import (
    dataset_manifest_sha256,
    dataset_manifest_summary,
    load_dataset_manifest,
    validate_dataset_manifest,
)
from jaxtitan.errors import ContractError


def test_load_dataset_manifest_returns_normalized_dataclass(prepared_dataset: Path) -> None:
    manifest = load_dataset_manifest(prepared_dataset)

    assert manifest.schema_version == 2
    assert manifest.kind == "training_tokens"
    assert manifest.dtype == "uint32"
    assert manifest.tokenizer_id == "toy-tokenizer"
    assert manifest.num_tokens == 8
    assert manifest.train_tokens == 6
    assert manifest.val_tokens == 2
    assert manifest.shard_count == 2
    assert manifest.shards[0].path == Path("tokens-00000.bin")
    assert manifest.token_bytes_path == Path("token_bytes.bin")
    assert manifest.manifest_sha256 == dataset_manifest_sha256(prepared_dataset)

    summary = dataset_manifest_summary(manifest)
    assert summary["manifest_path"] == prepared_dataset.as_posix()
    assert summary["total_tokens"] == 8
    assert summary["token_bytes_path"] == "token_bytes.bin"


@pytest.mark.parametrize(
    ("update", "match"),
    [
        (lambda data: data.update(schema_version=1), "schema_version=2"),
        (lambda data: data.update(kind="eval_domain"), "kind must be training_tokens"),
        (lambda data: data.update(dtype="uint16"), "dtype must be uint32"),
    ],
)
def test_manifest_top_level_schema_failures(
    prepared_dataset_factory: Callable[[str], Path], update, match: str
) -> None:
    manifest_path = prepared_dataset_factory("bad-top-level")
    data = _read_manifest(manifest_path)
    update(data)
    _write_manifest(manifest_path, data)

    with pytest.raises(ContractError, match=match):
        validate_dataset_manifest(manifest_path, tokenizer_id="toy-tokenizer")


def test_tokenizer_mismatch_raises(prepared_dataset: Path) -> None:
    with pytest.raises(ContractError, match="does not match config tokenizer"):
        validate_dataset_manifest(prepared_dataset, tokenizer_id="wrong-tokenizer")


def test_missing_shard_file_raises(prepared_dataset_factory: Callable[[str], Path]) -> None:
    manifest_path = prepared_dataset_factory("missing-shard")
    (manifest_path.parent / "tokens-00000.bin").unlink()

    with pytest.raises(ContractError, match="file does not exist"):
        validate_dataset_manifest(manifest_path, tokenizer_id="toy-tokenizer")


def test_non_contiguous_shards_raise(prepared_dataset_factory: Callable[[str], Path]) -> None:
    manifest_path = prepared_dataset_factory("bad-shard-bounds")
    data = _read_manifest(manifest_path)
    data["shards"][1]["start"] = 5
    _write_manifest(manifest_path, data)

    with pytest.raises(ContractError, match="contiguous"):
        validate_dataset_manifest(manifest_path, tokenizer_id="toy-tokenizer")


def test_shard_file_length_mismatch_raises(prepared_dataset_factory: Callable[[str], Path]) -> None:
    manifest_path = prepared_dataset_factory("bad-file-length")
    (manifest_path.parent / "tokens-00000.bin").write_bytes(b"\x01\x00\x00\x00")

    with pytest.raises(ContractError, match="file length"):
        validate_dataset_manifest(manifest_path, tokenizer_id="toy-tokenizer")


@pytest.mark.parametrize(
    ("update", "match"),
    [
        (lambda data: data["splits"].pop("train"), "splits.train"),
        (lambda data: data["splits"]["val"].update(start=5, tokens=3), "overlap"),
        (lambda data: data["splits"]["train"].update(tokens=99), "end-start"),
    ],
)
def test_bad_splits_raise(prepared_dataset_factory: Callable[[str], Path], update, match: str) -> None:
    manifest_path = prepared_dataset_factory("bad-split")
    data = _read_manifest(manifest_path)
    update(data)
    _write_manifest(manifest_path, data)

    with pytest.raises(ContractError, match=match):
        validate_dataset_manifest(manifest_path, tokenizer_id="toy-tokenizer")


def test_missing_token_bytes_raises(prepared_dataset_factory: Callable[[str], Path]) -> None:
    manifest_path = prepared_dataset_factory("missing-token-bytes")
    (manifest_path.parent / "token_bytes.bin").unlink()

    with pytest.raises(ContractError, match="token_bytes"):
        validate_dataset_manifest(manifest_path, tokenizer_id="toy-tokenizer")


def test_checksums_are_optional_then_enforced(prepared_dataset_factory: Callable[[str], Path]) -> None:
    manifest_path = prepared_dataset_factory("bad-checksum")
    data = _read_manifest(manifest_path)
    data["shards"][0]["sha256"] = "bad"
    _write_manifest(manifest_path, data)

    assert validate_dataset_manifest(manifest_path, tokenizer_id="toy-tokenizer").num_tokens == 8
    with pytest.raises(ContractError, match="checksum mismatch"):
        validate_dataset_manifest(manifest_path, tokenizer_id="toy-tokenizer", verify_checksums=True)


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_manifest(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
