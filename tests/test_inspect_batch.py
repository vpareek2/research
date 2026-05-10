import json

import numpy as np
import tiktoken

from research.config import DataConfig, ExperimentConfig, ModelConfig, OptimizerConfig, RunConfig, SamplingConfig, TrainConfig
from research.utils.inspect_batch import inspect_batch


def _write_run_config(run_dir, data_path):
    config = RunConfig(
        experiment=ExperimentConfig(name="unit", out_dir=str(run_dir.parent)),
        model=ModelConfig(
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
        ),
        train=TrainConfig(
            seed=0,
            batch_size=1,
            seq_len=8,
            steps=1,
            log_every=1,
            eval_every=1,
            eval_steps=1,
            checkpoint_every=1,
            keep_last=1,
        ),
        optimizer=OptimizerConfig(name="muon", lr=0.001, weight_decay=0.1),
        data=DataConfig(path=str(data_path), tokenizer="gpt2", val_fraction=0.25),
        sampling=SamplingConfig(),
    )
    (run_dir / "config.toml").write_text(
        f"""
[experiment]
name = "{config.experiment.name}"
out_dir = "{config.experiment.out_dir}"

[model]
vocab_size = {config.model.vocab_size}
hidden_size = {config.model.hidden_size}
intermediate_size = {config.model.intermediate_size}
n_layers = {config.model.n_layers}
n_heads = {config.model.n_heads}
n_kv_heads = {config.model.n_kv_heads}
seq_len = {config.model.seq_len}
theta = {config.model.theta}
eps = {config.model.eps}
tied = false

[train]
seed = {config.train.seed}
batch_size = {config.train.batch_size}
seq_len = {config.train.seq_len}
steps = {config.train.steps}
log_every = {config.train.log_every}
eval_every = {config.train.eval_every}
eval_steps = {config.train.eval_steps}
checkpoint_every = {config.train.checkpoint_every}
keep_last = {config.train.keep_last}

[optimizer]
name = "{config.optimizer.name}"
lr = {config.optimizer.lr}
weight_decay = {config.optimizer.weight_decay}

[data]
path = "{config.data.path}"
tokenizer = "{config.data.tokenizer}"
val_fraction = {config.data.val_fraction}
""".strip()
    )


def test_inspect_batch_writes_default_file(tmp_path):
    data_path = tmp_path / "input.txt"
    data_path.write_text("hello world " * 64)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "samples").mkdir()
    _write_run_config(run_dir, data_path)
    (run_dir / "batches.jsonl").write_text(
        json.dumps(
            {
                "step": 3,
                "chunk_idx": [0],
                "token_start": [0],
                "token_end": [4],
            }
        )
        + "\n"
    )

    out_file = inspect_batch(run_dir, 3)

    assert out_file == run_dir / "samples" / "batch_step_000003.txt"
    out = out_file.read_text()
    assert "step: 3" in out
    assert "chunk_idx: 0" in out
    assert "hello" in out


def test_inspect_batch_writes_explicit_file(tmp_path):
    data_path = tmp_path / "input.txt"
    data_path.write_text("hello world " * 64)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run_config(run_dir, data_path)
    (run_dir / "batches.jsonl").write_text(
        json.dumps(
            {
                "step": 3,
                "chunk_idx": [0],
                "token_start": [0],
                "token_end": [4],
            }
        )
        + "\n"
    )
    out_file = tmp_path / "batch.txt"

    result = inspect_batch(run_dir, 3, out_file)

    assert result == out_file
    assert "hello" in out_file.read_text()


def test_inspect_batch_reads_prepared_token_data(tmp_path):
    tokenizer = tiktoken.get_encoding("gpt2")
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    tokens = np.asarray(tokenizer.encode("prepared hello world"), dtype=np.uint32)
    tokens.tofile(data_dir / "tokens.bin")
    (data_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "kind": "training_tokens",
        "dtype": "uint32",
        "num_tokens": len(tokens),
        "tokenizer": {"name": "gpt2"},
        "files": {},
        "shards": [{"path": "tokens.bin", "start": 0, "end": len(tokens), "tokens": len(tokens), "bytes": len(tokens) * 4}],
        "splits": {
            "train": {"start": 0, "end": len(tokens), "tokens": len(tokens)},
            "val": {"start": len(tokens), "end": len(tokens), "tokens": 0},
        },
    }))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "samples").mkdir()
    config_text = f"""
[experiment]
name = "unit"
out_dir = "{tmp_path}"

[model]
vocab_size = 128
hidden_size = 32
intermediate_size = 64
n_layers = 1
n_heads = 4
n_kv_heads = 1
seq_len = 8
theta = 10000.0
eps = 0.000001
tied = false

[train]
seed = 0
batch_size = 1
seq_len = 8
steps = 1
log_every = 1
eval_every = 1
eval_steps = 1
checkpoint_every = 1
keep_last = 1

[optimizer]
name = "muon"
lr = 0.001
weight_decay = 0.1

[data]
source = "tokens"
path = "{data_dir}"
tokenizer = "gpt2"

[sampling]
enabled = false
""".strip()
    (run_dir / "config.toml").write_text(config_text)
    (run_dir / "batches.jsonl").write_text(
        json.dumps({"step": 0, "chunk_idx": [0], "token_start": [0], "token_end": [len(tokens)]}) + "\n"
    )

    out_file = inspect_batch(run_dir, 0)

    assert "prepared hello world" in out_file.read_text()
