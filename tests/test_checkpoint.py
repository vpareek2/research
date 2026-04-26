import numpy as np
import optax
from flax import nnx

from checkpoint import create_checkpoint_manager, restore_latest_checkpoint, save_checkpoint
from config import DataConfig, ModelConfig, TrainConfig
from data import make_dataloaders
from model import Model


def tiny_model_config():
    return ModelConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=1,
        seq_len=8,
        theta=10000.0,
        eps=1e-6,
        tied=False,
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
        eval_steps=1,
        checkpoint_every=1,
        keep_last=2,
    )
    values.update(overrides)
    return TrainConfig(**values)


def data_config(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("hello world " * 512)
    return DataConfig(path=str(path), tokenizer="gpt2", val_fraction=0.25)


def test_checkpoint_restores_next_step_and_train_iterator(tmp_path):
    model_config = tiny_model_config()
    tc = train_config()
    dc = data_config(tmp_path)

    model = Model(model_config, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(tc.lr), wrt=nnx.Param)
    train_iter, _ = make_dataloaders(dc, tc)

    _ = next(train_iter)
    manager = create_checkpoint_manager(tmp_path / "run", keep_last=2)
    save_checkpoint(
        manager,
        next_step=1,
        model=model,
        optimizer=optimizer,
        train_iter=train_iter,
    )
    manager.wait_until_finished()

    expected_next_batch = next(train_iter)

    restored_model = Model(model_config, rngs=nnx.Rngs(1))
    restored_optimizer = nnx.Optimizer(restored_model, optax.adamw(tc.lr), wrt=nnx.Param)
    restored_train_iter, _ = make_dataloaders(dc, tc)
    restored_next_step = restore_latest_checkpoint(
        manager,
        model=restored_model,
        optimizer=restored_optimizer,
        train_iter=restored_train_iter,
    )
    actual_next_batch = next(restored_train_iter)

    assert restored_next_step == 1
    np.testing.assert_array_equal(
        np.asarray(actual_next_batch["input_ids"]),
        np.asarray(expected_next_batch["input_ids"]),
    )
    np.testing.assert_array_equal(
        np.asarray(actual_next_batch["chunk_idx"]),
        np.asarray(expected_next_batch["chunk_idx"]),
    )
