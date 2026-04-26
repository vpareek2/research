import json
import sys
import types

import numpy as np
import pytest

from config import (
    DataConfig,
    ExperimentConfig,
    WandbConfig,
    ModelConfig,
    RunConfig,
    SamplingConfig,
    TrainConfig,
)
from run import setup_run


def make_config(out_dir, *, wandb=False):
    return RunConfig(
        experiment=ExperimentConfig(name="unit", out_dir=str(out_dir)),
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
            batch_size=2,
            seq_len=8,
            steps=2,
            lr=1e-3,
            decay=0.1,
            log_every=1,
            eval_every=1,
            eval_steps=1,
            checkpoint_every=2,
            keep_last=2,
        ),
        data=DataConfig(path="input.txt", tokenizer="gpt2", val_fraction=0.25),
        sampling=SamplingConfig(),
        wandb=WandbConfig(
            enabled=wandb,
            project="unit-project",
            entity="unit-entity",
            tags=["unit"],
        ),
    )


def make_fake_wandb():
    class FakeRun:
        id = "fake-run-id"

        def __init__(self):
            self.logs = []

        def log(self, data, step=None):
            self.logs.append((data, step))

    class FakeTable:
        def __init__(self, columns, log_mode=None):
            self.columns = columns
            self.log_mode = log_mode
            self.rows = []

        def add_data(self, *row):
            self.rows.append(row)

    fake_run = FakeRun()
    fake = types.SimpleNamespace()
    fake.init_calls = []
    fake.finish_calls = 0
    fake.login_calls = []
    fake.run = fake_run
    fake.tables = []

    def init(**kwargs):
        fake.init_calls.append(kwargs)
        return fake_run

    def login(**kwargs):
        fake.login_calls.append(kwargs)

    def table(columns, log_mode=None):
        value = FakeTable(columns, log_mode=log_mode)
        fake.tables.append(value)
        return value

    def finish():
        fake.finish_calls += 1

    fake.init = init
    fake.login = login
    fake.Table = table
    fake.finish = finish
    return fake


def test_setup_run_creates_artifacts_and_logs_metrics_and_batches(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[experiment]\nname = 'unit'\nout_dir = 'runs'\n")
    config = make_config(tmp_path / "runs")

    logger = setup_run(config_path, config)
    logger.log({"step": 0, "train/loss": 1.25})
    logger.log_batch(
        0,
        {
            "chunk_idx": np.asarray([3, 7]),
            "token_start": np.asarray([24, 56]),
            "token_end": np.asarray([32, 64]),
        },
    )

    run_dir = tmp_path / "runs" / "unit"
    assert (run_dir / "config.toml").read_text() == config_path.read_text()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "samples").is_dir()

    metrics = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(metrics) == 1
    assert json.loads(metrics[0]) == {"step": 0, "train/loss": 1.25}

    batches = (run_dir / "batches.jsonl").read_text().splitlines()
    assert len(batches) == 1
    assert json.loads(batches[0]) == {
        "step": 0,
        "chunk_idx": [3, 7],
        "token_start": [24, 56],
        "token_end": [32, 64],
    }


def test_setup_run_rejects_duplicate_run_dir(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[experiment]\nname = 'unit'\nout_dir = 'runs'\n")
    config = make_config(tmp_path / "runs")

    setup_run(config_path, config)
    with pytest.raises(FileExistsError):
        setup_run(config_path, config)


def test_setup_run_resume_requires_existing_run_dir(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[experiment]\nname = 'unit'\nout_dir = 'runs'\n")
    config = make_config(tmp_path / "runs")

    with pytest.raises(FileNotFoundError):
        setup_run(config_path, config, resume=True)

    logger = setup_run(config_path, config)
    resumed = setup_run(config_path, config, resume=True)
    assert resumed.run_dir == logger.run_dir


def test_wandb_disabled_does_not_import_or_initialize(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    config_path = tmp_path / "config.toml"
    config_path.write_text("[experiment]\nname = 'unit'\nout_dir = 'runs'\n")
    config = make_config(tmp_path / "runs", wandb=False)

    logger = setup_run(config_path, config)
    logger.log({"step": 0, "train/loss": 1.0})
    logger.close()

    assert "wandb" not in sys.modules


def test_wandb_first_run_logs_scalars_and_samples(tmp_path, monkeypatch):
    fake_wandb = make_fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    config_path = tmp_path / "config.toml"
    config_path.write_text("[experiment]\nname = 'unit'\nout_dir = 'runs'\n")
    config = make_config(tmp_path / "runs", wandb=True)

    logger = setup_run(config_path, config)
    sample_path = logger.run_dir / "samples" / "sample_step_000000.txt"
    sample_path.write_text("sample text", encoding="utf-8")
    logger.log({"step": 0, "train/loss": 1.0, "sample/path": str(sample_path)})
    logger.close()

    assert fake_wandb.login_calls == [{"key": "test-key"}]
    assert (logger.run_dir / "wandb_id.txt").read_text().strip() == "fake-run-id"
    assert fake_wandb.init_calls[0]["project"] == "unit-project"
    assert fake_wandb.init_calls[0]["entity"] == "unit-entity"
    assert fake_wandb.init_calls[0]["tags"] == ["unit"]
    assert fake_wandb.init_calls[0]["name"] == "unit"
    assert fake_wandb.run.logs[0] == ({"train/loss": 1.0}, 0)
    sample_log, sample_step = fake_wandb.run.logs[1]
    assert sample_step == 0
    assert "samples" in sample_log
    assert fake_wandb.tables[0].log_mode == "MUTABLE"
    assert fake_wandb.tables[0].rows == [(0, str(sample_path), "sample text")]
    assert fake_wandb.finish_calls == 1


def test_wandb_resume_uses_stored_run_id(tmp_path, monkeypatch):
    fake_wandb = make_fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    config_path = tmp_path / "config.toml"
    config_path.write_text("[experiment]\nname = 'unit'\nout_dir = 'runs'\n")
    config = make_config(tmp_path / "runs", wandb=True)

    first = setup_run(config_path, config)
    first.close()
    resumed = setup_run(config_path, config, resume=True)
    resumed.close()

    assert fake_wandb.init_calls[1]["id"] == "fake-run-id"
    assert fake_wandb.init_calls[1]["resume"] == "must"


def test_wandb_resume_requires_stored_run_id(tmp_path, monkeypatch):
    fake_wandb = make_fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    config_path = tmp_path / "config.toml"
    config_path.write_text("[experiment]\nname = 'unit'\nout_dir = 'runs'\n")
    config = make_config(tmp_path / "runs", wandb=True)
    run_dir = tmp_path / "runs" / "unit"
    run_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        setup_run(config_path, config, resume=True)


def test_wandb_enabled_prompts_for_missing_key(tmp_path, monkeypatch):
    fake_wandb = make_fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("getpass.getpass", lambda prompt: "prompt-key")
    config_path = tmp_path / "config.toml"
    config_path.write_text("[experiment]\nname = 'unit'\nout_dir = 'runs'\n")
    config = make_config(tmp_path / "runs", wandb=True)

    logger = setup_run(config_path, config)
    logger.close()

    assert fake_wandb.login_calls == [{"key": "prompt-key"}]


def test_wandb_enabled_reuses_saved_netrc_key(tmp_path, monkeypatch):
    fake_wandb = make_fake_wandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    netrc_path = tmp_path / ".netrc"
    netrc_path.write_text("machine api.wandb.ai login user password saved-key\n", encoding="utf-8")
    netrc_path.chmod(0o600)
    config_path = tmp_path / "config.toml"
    config_path.write_text("[experiment]\nname = 'unit'\nout_dir = 'runs'\n")
    config = make_config(tmp_path / "runs", wandb=True)

    logger = setup_run(config_path, config)
    logger.close()

    assert fake_wandb.login_calls == [{"relogin": False}]
