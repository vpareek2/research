from pathlib import Path

import pytest

from config import DataConfig, load_config


def write_config(path: Path, *, include_wandb: bool = True):
    wandb_section = """
[wandb]
enabled = true
project = "unit-project"
entity = "unit-entity"
tags = ["smoke", "unit"]
""" if include_wandb else ""
    path.write_text(
        f"""
[experiment]
name = "unit"
out_dir = "runs"

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
batch_size = 2
seq_len = 8
steps = 2
lr = 0.001
decay = 0.1
log_every = 1
eval_every = 1
eval_steps = 1
checkpoint_every = 2
keep_last = 2

[data]
path = "input.txt"
tokenizer = "gpt2"
val_fraction = 0.25

[sampling]
enabled = true
prompt = "ROMEO:"
max_new_tokens = 8
temperature = 0.0
top_k = 10
{wandb_section}
""".strip()
    )


def test_load_config(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(config_path)

    config = load_config(config_path)

    assert config.experiment.name == "unit"
    assert config.model.hidden_size == 32
    assert config.train.eval_steps == 1
    assert config.train.checkpoint_every == 2
    assert config.train.keep_last == 2
    assert config.data.val_fraction == 0.25
    assert config.sampling.enabled is True
    assert config.sampling.prompt == "ROMEO:"
    assert config.sampling.max_new_tokens == 8
    assert config.wandb.enabled is True
    assert config.wandb.project == "unit-project"
    assert config.wandb.entity == "unit-entity"
    assert config.wandb.tags == ["smoke", "unit"]


def test_wandb_config_defaults_to_disabled(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(config_path, include_wandb=False)

    config = load_config(config_path)

    assert config.wandb.enabled is False
    assert config.wandb.project == "data-research"
    assert config.wandb.entity == ""
    assert config.wandb.tags == []


@pytest.mark.parametrize("val_fraction", [0.0, 1.0, -0.1, 1.1])
def test_invalid_val_fraction_raises(val_fraction):
    with pytest.raises(ValueError):
        DataConfig(path="input.txt", tokenizer="gpt2", val_fraction=val_fraction)
