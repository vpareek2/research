import json

import pytest

from config import load_config
from train_budget import build_budget, format_budget, steps_for_tokens, steps_per_epoch, tokens_per_step


def write_config(path):
    path.write_text(
        """
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
""".strip(),
        encoding="utf-8",
    )


def test_steps_for_tokens_rounds_up():
    assert steps_for_tokens(100, 32) == 4
    assert steps_for_tokens(96, 32) == 3
    assert steps_for_tokens(0, 32) == 0


def test_steps_per_epoch_drops_partial_step():
    assert steps_per_epoch(100, 32) == 3
    assert steps_per_epoch(96, 32) == 3


def test_build_budget_for_text_config_with_target_tokens(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    config = load_config(config_path)

    budget = build_budget(config, target_tokens=100)

    assert tokens_per_step(config) == 16
    assert budget.tokens_per_step == 16
    assert budget.configured_steps == 2
    assert budget.configured_tokens == 32
    assert budget.target_tokens == 100
    assert budget.target_steps == 7
    assert budget.train_tokens is None
    assert budget.steps_per_epoch is None
    assert budget.configured_epochs is None


def test_build_budget_infers_prepared_token_epoch(tmp_path):
    data_dir = tmp_path / "prepared"
    data_dir.mkdir()
    (data_dir / "tokens.bin").write_bytes(b"")
    (data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "files": {"tokens": {"path": "tokens.bin"}},
                "splits": {
                    "train": {"start": 0, "end": 100, "tokens": 100},
                    "val": {"start": 100, "end": 120, "tokens": 20},
                },
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace(
        """[data]
path = "input.txt"
tokenizer = "gpt2"
val_fraction = 0.25""",
        "[data]\n"
        "source = \"tokens\"\n"
        f"path = \"{data_dir}\"\n"
        "tokenizer = \"gpt2\"",
    )
    config_path.write_text(text, encoding="utf-8")
    config = load_config(config_path)

    budget = build_budget(config, target_tokens=200)

    assert budget.train_tokens == 100
    assert budget.steps_per_epoch == 6
    assert budget.usable_epoch_tokens == 96
    assert budget.configured_epochs == pytest.approx(0.32)
    assert budget.target_steps == 13
    assert budget.target_epochs == pytest.approx(2.0)


def test_format_budget_includes_configured_target_and_epoch_fields(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    config = load_config(config_path)
    budget = build_budget(config, target_tokens=100)

    output = format_budget(config, budget)

    assert "Training Budget" in output
    assert "tokens_per_step:   16" in output
    assert "configured_tokens:   32" in output
    assert "target_steps:        7" in output
    assert "steps_per_epoch:     n/a" in output
