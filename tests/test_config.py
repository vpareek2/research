from pathlib import Path

import jax.numpy as jnp
import pytest

from config import DataConfig, DistributedConfig, LRScheduleConfig, PrecisionConfig, dtype_from_name, load_config


def write_config(
    path: Path,
    *,
    include_wandb: bool = True,
    lr_schedule_section: str = "",
    precision_section: str = "",
    distributed_section: str = "",
):
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
{precision_section}
{distributed_section}

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
{lr_schedule_section}

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
    assert config.train.lr_schedule.type == "cosine"
    assert config.train.lr_schedule.warmup_ratio == 0.01
    assert config.train.lr_schedule.min_lr_ratio == 0.1
    assert config.train.lr_schedule.stable_ratio == 0.80
    assert config.precision.compute_dtype == "fp32"
    assert config.precision.param_dtype == "fp32"
    assert config.precision.loss_dtype == "fp32"
    assert config.distributed.enabled is True
    assert config.distributed.device_count == "auto"
    assert config.distributed.axis_name == "data"
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


def test_explicit_lr_schedule_config(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(
        config_path,
        lr_schedule_section="""
[train.lr_schedule]
type = "wsd"
warmup_ratio = 0.02
stable_ratio = 0.75
min_lr_ratio = 0.05
""",
    )

    config = load_config(config_path)

    assert config.train.lr_schedule.type == "wsd"
    assert config.train.lr_schedule.warmup_ratio == 0.02
    assert config.train.lr_schedule.stable_ratio == 0.75
    assert config.train.lr_schedule.min_lr_ratio == 0.05


def test_explicit_precision_config(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(
        config_path,
        precision_section="""
[precision]
compute_dtype = "bf16"
param_dtype = "fp32"
loss_dtype = "fp32"
""",
    )

    config = load_config(config_path)

    assert config.precision.compute_dtype == "bf16"
    assert config.precision.param_dtype == "fp32"
    assert config.precision.loss_dtype == "fp32"


def test_explicit_distributed_config(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(
        config_path,
        distributed_section="""
[distributed]
enabled = true
device_count = "auto"
axis_name = "data"
""",
    )

    config = load_config(config_path)

    assert config.distributed.enabled is True
    assert config.distributed.device_count == "auto"
    assert config.distributed.axis_name == "data"


def test_distributed_can_be_disabled(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(
        config_path,
        distributed_section="""
[distributed]
enabled = false
device_count = 8
axis_name = "data"
""",
    )

    config = load_config(config_path)

    assert config.distributed.enabled is False
    assert config.distributed.device_count == 8


def test_mismatched_model_and_train_seq_len_raises(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("seq_len = 8\nsteps = 2", "seq_len = 4\nsteps = 2")
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="model.seq_len"):
        load_config(config_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 0),
        ("seq_len", 0),
        ("steps", 0),
        ("log_every", 0),
        ("eval_every", 0),
        ("eval_steps", 0),
        ("keep_last", 0),
    ],
)
def test_invalid_positive_train_fields_raise(tmp_path, field, value):
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    originals = {
        "batch_size": "batch_size = 2",
        "seq_len": "seq_len = 8\nsteps = 2",
        "steps": "steps = 2",
        "log_every": "log_every = 1",
        "eval_every": "eval_every = 1",
        "eval_steps": "eval_steps = 1",
        "keep_last": "keep_last = 2",
    }
    replacements = {
        "batch_size": f"batch_size = {value}",
        "seq_len": f"seq_len = {value}\nsteps = 2",
        "steps": f"steps = {value}",
        "log_every": f"log_every = {value}",
        "eval_every": f"eval_every = {value}",
        "eval_steps": f"eval_steps = {value}",
        "keep_last": f"keep_last = {value}",
    }
    text = config_path.read_text(encoding="utf-8").replace(originals[field], replacements[field])
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=f"train.{field}"):
        load_config(config_path)


@pytest.mark.parametrize("val_fraction", [0.0, 1.0, -0.1, 1.1])
def test_invalid_val_fraction_raises(val_fraction):
    with pytest.raises(ValueError):
        DataConfig(path="input.txt", tokenizer="gpt2", val_fraction=val_fraction)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": "linear"},
        {"warmup_ratio": -0.1},
        {"warmup_ratio": 1.0},
        {"min_lr_ratio": -0.1},
        {"min_lr_ratio": 1.1},
        {"stable_ratio": -0.1},
        {"stable_ratio": 1.0},
    ],
)
def test_invalid_lr_schedule_config_raises(kwargs):
    with pytest.raises(ValueError):
        LRScheduleConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"compute_dtype": "fp16"},
        {"param_dtype": "float32"},
        {"loss_dtype": "int32"},
    ],
)
def test_invalid_precision_config_raises(kwargs):
    with pytest.raises(ValueError):
        PrecisionConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": "yes"},
        {"device_count": 0},
        {"device_count": -1},
        {"device_count": "all"},
        {"axis_name": ""},
    ],
)
def test_invalid_distributed_config_raises(kwargs):
    with pytest.raises(ValueError):
        DistributedConfig(**kwargs)


def test_dtype_from_name():
    assert jnp.dtype(dtype_from_name("fp32")) == jnp.dtype(jnp.float32)
    assert jnp.dtype(dtype_from_name("bf16")) == jnp.dtype(jnp.bfloat16)
    with pytest.raises(ValueError):
        dtype_from_name("fp16")
