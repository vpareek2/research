import json

import numpy as np
import pytest

from config import DataConfig, TrainConfig
from data import TokenDataset, TokenMemmapDataset, make_dataloaders, make_val_dataloader, split_tokens


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
    (data_dir / "manifest.json").write_text(json.dumps({
        "files": {"tokens": {"path": "tokens.bin"}},
        "splits": {
            "train": {"start": 0, "end": 64, "tokens": 64},
            "val": {"start": 64, "end": 128, "tokens": 64},
        },
    }))
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
