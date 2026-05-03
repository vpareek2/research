import json
from pathlib import Path

import numpy as np
import pytest
import tiktoken

from research.config import DataConfig, EvalConfig, TrainConfig
from research.data import (
    REQUIRED_EVAL_DOMAINS,
    ShardedTokenDataset,
    TokenDataset,
    TokenMemmapDataset,
    build_token_bytes,
    default_eval_domain_root,
    eval_domain_root,
    load_eval_domain_token_bytes,
    load_token_manifest,
    load_token_bytes,
    make_dataloaders,
    make_eval_domain_dataloaders,
    make_val_dataloader,
    split_tokens,
    validate_eval_domain_pack,
    validate_token_manifest,
)


def train_config(**overrides):
    values = dict(
        seed=0,
        batch_size=2,
        seq_len=8,
        steps=2,
        lr=0.001,
        decay=0.1,
        log_every=1,
        eval_every=1,
        eval_steps=2,
        checkpoint_every=2,
        keep_last=2,
    )
    values.update(overrides)
    return TrainConfig(**values)


def data_config(tmp_path, text=None, val_fraction=0.25):
    path = tmp_path / "input.txt"
    if text is None:
        text = "hello world " * 512
    path.write_text(text)
    return DataConfig(path=str(path), tokenizer="gpt2", val_fraction=val_fraction)


def write_manifest(data_dir, *, num_tokens=128, train=(0, 64), val=(64, 128), tokenizer="gpt2", dtype="uint32", token_bytes=False):
    files = {}
    if token_bytes:
        files["token_bytes"] = {"path": "token_bytes.bin"}
    (data_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "kind": "training_tokens",
        "dtype": dtype,
        "num_tokens": num_tokens,
        "tokenizer": {"name": tokenizer},
        "files": files,
        "shards": [{"path": "tokens.bin", "start": 0, "end": num_tokens, "tokens": num_tokens, "bytes": num_tokens * 4}],
        "splits": {
            "train": {"start": train[0], "end": train[1], "tokens": train[1] - train[0]},
            "val": {"start": val[0], "end": val[1], "tokens": val[1] - val[0]},
        },
    }))


def write_eval_domain_pack(root, *, tokenizer="gpt2", tokens_per_domain=32):
    root.mkdir(parents=True)
    np.arange(128, dtype=np.uint16).tofile(root / "token_bytes.bin")
    domains = {}
    for name in REQUIRED_EVAL_DOMAINS:
        domain_dir = root / name
        domain_dir.mkdir()
        np.arange(tokens_per_domain, dtype=np.uint32).tofile(domain_dir / "tokens.bin")
        (domain_dir / "manifest.json").write_text(json.dumps({
            "dtype": "uint32",
            "num_tokens": tokens_per_domain,
            "tokenizer": {"name": tokenizer},
            "files": {"tokens": {"path": "tokens.bin"}},
            "splits": {
                "train": {"start": 0, "end": 0, "tokens": 0},
                "val": {"start": 0, "end": tokens_per_domain, "tokens": tokens_per_domain},
            },
        }))
        domains[name] = {"path": name, "num_tokens": tokens_per_domain}
    (root / "manifest.json").write_text(json.dumps({
        "kind": "eval_domains",
        "tokenizer": {"name": tokenizer},
        "files": {"token_bytes": {"path": "token_bytes.bin"}},
        "domains": domains,
    }))


def test_token_dataset_chunks_with_provenance():
    dataset = TokenDataset(np.arange(20, dtype=np.int32), seq_len=6)

    assert len(dataset) == 3
    first = dataset[0]
    third = dataset[2]
    np.testing.assert_array_equal(first["input_ids"], np.arange(6, dtype=np.int32))
    np.testing.assert_array_equal(third["input_ids"], np.arange(12, 18, dtype=np.int32))
    assert first["chunk_idx"] == 0
    assert first["token_start"] == 0
    assert first["token_end"] == 6
    assert third["chunk_idx"] == 2
    assert third["token_start"] == 12
    assert third["token_end"] == 18


def test_split_tokens_is_deterministic():
    tokens = np.arange(100, dtype=np.int32)
    train, val = split_tokens(tokens, 0.2)

    np.testing.assert_array_equal(train, np.arange(80, dtype=np.int32))
    np.testing.assert_array_equal(val, np.arange(80, 100, dtype=np.int32))


def test_build_token_bytes_zeros_eot_and_counts_normal_tokens():
    tokenizer = tiktoken.get_encoding("gpt2")
    token_bytes = build_token_bytes(tokenizer)
    hello_token = tokenizer.encode("hello")[0]

    assert token_bytes[tokenizer.eot_token] == 0
    assert token_bytes[hello_token] == len(tokenizer.decode_single_token_bytes(hello_token))


def test_load_token_bytes_prefers_prepared_table(tmp_path):
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    np.arange(128, dtype=np.uint32).tofile(data_dir / "tokens.bin")
    expected = np.arange(128, dtype=np.uint16)
    expected.tofile(data_dir / "token_bytes.bin")
    write_manifest(data_dir, token_bytes=True)
    dc = DataConfig(source="tokens", path=str(data_dir), tokenizer="gpt2")

    np.testing.assert_array_equal(load_token_bytes(dc), expected)


def test_default_eval_domain_root_uses_tokenizer_name():
    assert default_eval_domain_root("gpt2") == Path("data/eval_domains/gpt2")
    assert default_eval_domain_root("org/tokenizer") == Path("data/eval_domains/org__tokenizer")


def test_eval_domain_root_allows_config_override():
    dc = DataConfig(source="tokens", path="data/prepared", tokenizer="gpt2")

    assert eval_domain_root(EvalConfig(domain_root="custom/eval"), dc) == Path("custom/eval")


def test_dataloaders_shapes_and_val_determinism(tmp_path):
    dc = data_config(tmp_path)
    tc = train_config()

    train_iter, val_iter = make_dataloaders(dc, tc)
    train_batch = next(train_iter)
    val_batch = next(val_iter)
    val_a = next(make_val_dataloader(dc, tc))
    val_b = next(make_val_dataloader(dc, tc))

    assert train_batch["input_ids"].shape == (tc.batch_size, tc.seq_len)
    assert val_batch["input_ids"].shape == (tc.batch_size, tc.seq_len)
    assert train_batch["chunk_idx"].shape == (tc.batch_size,)
    assert train_batch["token_start"].shape == (tc.batch_size,)
    assert train_batch["token_end"].shape == (tc.batch_size,)
    assert train_batch["input_ids"].dtype == np.int32
    assert val_batch["input_ids"].dtype == np.int32
    np.testing.assert_array_equal(np.asarray(val_a["input_ids"]), np.asarray(val_b["input_ids"]))
    np.testing.assert_array_equal(np.asarray(val_a["chunk_idx"]), np.asarray(val_b["chunk_idx"]))


def test_oversized_eval_steps_raises(tmp_path):
    dc = data_config(tmp_path, text="hello world " * 64, val_fraction=0.25)
    tc = train_config(eval_steps=10_000)

    with pytest.raises(ValueError, match="Validation set has"):
        make_dataloaders(dc, tc)


def test_token_memmap_dataset_chunks_with_provenance(tmp_path):
    token_path = tmp_path / "tokens.bin"
    np.arange(20, dtype=np.uint32).tofile(token_path)

    dataset = TokenMemmapDataset(token_path, seq_len=6, start=6, end=20)

    assert len(dataset) == 2
    first = dataset[0]
    np.testing.assert_array_equal(first["input_ids"], np.arange(6, 12, dtype=np.int32))
    assert first["chunk_idx"] == 0
    assert first["token_start"] == 6
    assert first["token_end"] == 12


def test_prepared_token_dataloaders_shapes_and_val_determinism(tmp_path):
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    np.arange(128, dtype=np.uint32).tofile(data_dir / "tokens.bin")
    write_manifest(data_dir)
    dc = DataConfig(source="tokens", path=str(data_dir), tokenizer="gpt2")
    tc = train_config(eval_steps=2)

    train_iter, val_iter = make_dataloaders(dc, tc)
    train_batch = next(train_iter)
    val_batch = next(val_iter)
    val_a = next(make_val_dataloader(dc, tc))
    val_b = next(make_val_dataloader(dc, tc))

    assert train_batch["input_ids"].shape == (tc.batch_size, tc.seq_len)
    assert val_batch["input_ids"].shape == (tc.batch_size, tc.seq_len)
    assert train_batch["input_ids"].dtype == np.int32
    assert val_batch["input_ids"].dtype == np.int32
    np.testing.assert_array_equal(np.asarray(val_a["input_ids"]), np.asarray(val_b["input_ids"]))
    np.testing.assert_array_equal(np.asarray(val_a["chunk_idx"]), np.asarray(val_b["chunk_idx"]))


def test_sharded_token_dataset_reads_across_shard_boundaries(tmp_path):
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    np.arange(10, dtype=np.uint32).tofile(data_dir / "tokens-00000.bin")
    np.arange(10, 20, dtype=np.uint32).tofile(data_dir / "tokens-00001.bin")
    (data_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "kind": "training_tokens",
        "dtype": "uint32",
        "num_tokens": 20,
        "tokenizer": {"name": "gpt2"},
        "files": {},
        "shards": [
            {"path": "tokens-00000.bin", "start": 0, "end": 10, "tokens": 10, "bytes": 40},
            {"path": "tokens-00001.bin", "start": 10, "end": 20, "tokens": 10, "bytes": 40},
        ],
        "splits": {
            "train": {"start": 6, "end": 18, "tokens": 12},
            "val": {"start": 18, "end": 20, "tokens": 2},
        },
    }))
    dc = DataConfig(source="tokens", path=str(data_dir), tokenizer="gpt2")
    manifest = validate_token_manifest(data_dir, load_token_manifest(data_dir), dc)
    dataset = ShardedTokenDataset(data_dir, manifest, seq_len=8, start=6, end=18)

    first = dataset[0]

    np.testing.assert_array_equal(first["input_ids"], np.arange(6, 14, dtype=np.int32))
    assert first["token_start"] == 6
    assert first["token_end"] == 14


@pytest.mark.parametrize(
    "manifest_kwargs,match",
    [
        ({"dtype": "uint16"}, "dtype"),
        ({"tokenizer": "cl100k_base"}, "tokenizer"),
        ({"num_tokens": 127}, "file length"),
        ({"train": (0, 80), "val": (64, 128)}, "overlap"),
        ({"train": (0, 129)}, "invalid bounds"),
    ],
)
def test_prepared_token_manifest_validation_raises(tmp_path, manifest_kwargs, match):
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    np.arange(128, dtype=np.uint32).tofile(data_dir / "tokens.bin")
    write_manifest(data_dir, **manifest_kwargs)
    dc = DataConfig(source="tokens", path=str(data_dir), tokenizer="gpt2")
    tc = train_config(eval_steps=2)

    with pytest.raises((ValueError, FileNotFoundError), match=match):
        make_dataloaders(dc, tc)


def test_prepared_token_manifest_missing_token_file_raises(tmp_path):
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    write_manifest(data_dir, num_tokens=0, train=(0, 0), val=(0, 0))
    dc = DataConfig(source="tokens", path=str(data_dir), tokenizer="gpt2")
    tc = train_config(eval_steps=2)

    with pytest.raises(FileNotFoundError, match="token shard"):
        make_dataloaders(dc, tc)


def test_eval_domain_pack_validation_and_dataloaders(tmp_path):
    eval_root = tmp_path / "eval_domains"
    write_eval_domain_pack(eval_root, tokens_per_domain=64)
    dc = DataConfig(source="tokens", path=str(tmp_path / "prepared"), tokenizer="gpt2")
    ec = EvalConfig(domain_root=str(eval_root), domain_eval_steps=2)
    tc = train_config(batch_size=2, seq_len=8)

    manifest = validate_eval_domain_pack(ec, dc)
    dataloaders = make_eval_domain_dataloaders(ec, dc, tc)
    token_bytes = load_eval_domain_token_bytes(ec, dc)

    assert set(manifest["domains"]) == set(REQUIRED_EVAL_DOMAINS)
    assert set(dataloaders) == set(REQUIRED_EVAL_DOMAINS)
    assert token_bytes.dtype == np.uint16
    batch = next(dataloaders["web"])
    assert batch["input_ids"].shape == (tc.batch_size, tc.seq_len)


def test_eval_domain_pack_uses_default_root_when_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    eval_root = default_eval_domain_root("gpt2")
    write_eval_domain_pack(eval_root, tokens_per_domain=64)
    dc = DataConfig(source="tokens", path=str(tmp_path / "prepared"), tokenizer="gpt2")
    tc = train_config(batch_size=2, seq_len=8)

    dataloaders = make_eval_domain_dataloaders(EvalConfig(domain_eval_steps=2), dc, tc)

    assert set(dataloaders) == set(REQUIRED_EVAL_DOMAINS)


def test_eval_domain_pack_tokenizer_mismatch_raises(tmp_path):
    eval_root = tmp_path / "eval_domains"
    write_eval_domain_pack(eval_root, tokenizer="cl100k_base")
    dc = DataConfig(source="tokens", path=str(tmp_path / "prepared"), tokenizer="gpt2")

    with pytest.raises(ValueError, match="tokenizer"):
        validate_eval_domain_pack(EvalConfig(domain_root=str(eval_root)), dc)


def test_eval_domain_pack_insufficient_tokens_raises(tmp_path):
    eval_root = tmp_path / "eval_domains"
    write_eval_domain_pack(eval_root, tokens_per_domain=8)
    dc = DataConfig(source="tokens", path=str(tmp_path / "prepared"), tokenizer="gpt2")
    ec = EvalConfig(domain_root=str(eval_root), domain_eval_steps=2)
    tc = train_config(batch_size=2, seq_len=8)

    with pytest.raises(ValueError, match="Eval domain"):
        make_eval_domain_dataloaders(ec, dc, tc)
