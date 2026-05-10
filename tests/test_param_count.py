import argparse

import jax
from flax import nnx

from research.config import ModelConfig
from research.model import Model
from research.utils.param_count import config_from_args, count_params, format_breakdown


def tiny_model_config(**overrides):
    values = dict(
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
    values.update(overrides)
    return ModelConfig(**values)


def _actual_param_count(model: Model) -> int:
    return sum(leaf.size for leaf in jax.tree.leaves(nnx.state(model, nnx.Param)))


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
log_every = 1
eval_every = 1
eval_steps = 1
checkpoint_every = 2
keep_last = 2

[optimizer]
name = "muon"
lr = 0.001
weight_decay = 0.1

[data]
path = "input.txt"
tokenizer = "gpt2"
val_fraction = 0.25
""".strip()
    )


def test_count_params_matches_instantiated_model():
    config = tiny_model_config()
    model = Model(config, rngs=nnx.Rngs(0))

    assert count_params(config).total == _actual_param_count(model)


def test_count_params_breakdown():
    breakdown = count_params(tiny_model_config())

    assert breakdown.token_embedding == 4096
    assert breakdown.lm_head == 4096
    assert breakdown.attention == 2576
    assert breakdown.mlp == 6144
    assert breakdown.norms == 96
    assert breakdown.total == 17008


def test_tied_embeddings_remove_lm_head_from_count():
    untied = count_params(tiny_model_config(tied=False))
    tied = count_params(tiny_model_config(tied=True))

    assert tied.lm_head == 0
    assert tied.total == untied.total - untied.lm_head


def test_format_breakdown_includes_total_and_major_splits():
    config = tiny_model_config()
    output = format_breakdown(config, count_params(config))

    assert "token_embedding" in output
    assert "attention" in output
    assert "mlp" in output
    assert "total" in output
    assert "17,008" in output


def test_config_from_args_loads_config_and_applies_overrides(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    args = argparse.Namespace(
        config=str(config_path),
        vocab_size=None,
        hidden_size=64,
        intermediate_size=None,
        n_layers=3,
        n_heads=None,
        n_kv_heads=None,
        seq_len=None,
        theta=1_000_000.0,
        eps=1e-6,
        tied=True,
    )

    config = config_from_args(args)

    assert config.vocab_size == 128
    assert config.hidden_size == 64
    assert config.intermediate_size == 64
    assert config.n_layers == 3
    assert config.tied is True
