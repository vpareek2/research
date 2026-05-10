from pathlib import Path

import jax.numpy as jnp
import pytest

from research.config import (
    AuroraOptimizerConfig,
    DataConfig,
    DistributedConfig,
    LRScheduleConfig,
    OptimizerConfig,
    PrecisionConfig,
    ProfilingConfig,
    RiemannianAuroraOptimizerConfig,
    SOAPOptimizerConfig,
    TargetConfig,
    dtype_from_name,
    load_config,
)


def write_config(
    path: Path,
    *,
    include_wandb: bool = True,
    lr_schedule_section: str = "",
    precision_section: str = "",
    distributed_section: str = "",
    profiling_section: str = "",
    eval_section: str = "",
    target_section: str = "",
    data_extra: str = "",
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
{profiling_section}
{eval_section}
{target_section}

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
{lr_schedule_section}

[optimizer]
name = "muon"
lr = 0.001
weight_decay = 0.1

[data]
path = "input.txt"
tokenizer = "gpt2"
val_fraction = 0.25
{data_extra}

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
    assert config.target.tokens == 2_000_000_000
    assert config.train.checkpoint_every == 2
    assert config.train.keep_last == 2
    assert config.train.lr_schedule.type == "cosine"
    assert config.train.lr_schedule.warmup_ratio == 0.01
    assert config.train.lr_schedule.min_lr_ratio == 0.1
    assert config.train.lr_schedule.stable_ratio == 0.80
    assert config.optimizer.name == "muon"
    assert config.optimizer.lr == 0.001
    assert config.optimizer.weight_decay == 0.1
    assert config.optimizer.adamw.b1 == 0.9
    assert config.optimizer.adamw.b2 == 0.999
    assert config.optimizer.muon.ns_steps == 5
    assert config.optimizer.aurora.pp_iterations == 2
    assert config.optimizer.aurora.pp_beta == 0.5
    assert config.optimizer.riemannian_aurora.outer_steps == 3
    assert config.optimizer.riemannian_aurora.cg_steps == 20
    assert config.optimizer.soap.b1 == 0.95
    assert config.optimizer.soap.b2 == 0.95
    assert config.optimizer.soap.precondition_frequency == 10
    assert config.precision.compute_dtype == "fp32"
    assert config.precision.param_dtype == "fp32"
    assert config.precision.loss_dtype == "fp32"
    assert config.distributed.enabled is True
    assert config.distributed.device_count == "auto"
    assert config.distributed.axis_name == "data"
    assert config.profiling.enabled is False
    assert config.profiling.profiler == "none"
    assert config.profiling.start_step == 100
    assert config.profiling.steps == 5
    assert config.profiling.output_dir == "profiles"
    assert config.eval.domain_root is None
    assert config.eval.domain_eval_steps is None
    assert config.eval.prepare_config == "configs/data/eval_domains.toml"
    assert config.data.val_fraction == 0.25
    assert config.data.prepare_config is None
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


def test_explicit_profiling_config(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(
        config_path,
        profiling_section="""
[profiling]
enabled = false
profiler = "none"
start_step = 200
steps = 7
output_dir = "profiles/unit"
""",
    )

    config = load_config(config_path)

    assert config.profiling.enabled is False
    assert config.profiling.profiler == "none"
    assert config.profiling.start_step == 200
    assert config.profiling.steps == 7
    assert config.profiling.output_dir == "profiles/unit"


def test_explicit_eval_config(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(
        config_path,
        eval_section="""
[eval]
domain_root = "data/eval_domains/custom"
domain_eval_steps = 25
prepare_config = "configs/data/custom_eval_domains.toml"
""",
    )

    config = load_config(config_path)

    assert config.eval.domain_root == "data/eval_domains/custom"
    assert config.eval.domain_eval_steps == 25
    assert config.eval.prepare_config == "configs/data/custom_eval_domains.toml"


def test_explicit_target_and_data_prepare_config(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(
        config_path,
        target_section="""
[target]
tokens = 12345
""",
        data_extra='prepare_config = "configs/data/unit.toml"',
    )

    config = load_config(config_path)

    assert config.target.tokens == 12345
    assert config.data.prepare_config == "configs/data/unit.toml"


def test_target_tokens_must_be_positive():
    with pytest.raises(ValueError, match="target.tokens"):
        TargetConfig(tokens=0)


def test_missing_optimizer_section_raises(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace(
        """
[optimizer]
name = "muon"
lr = 0.001
weight_decay = 0.1
""",
        "",
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[optimizer\]"):
        load_config(config_path)


def test_old_train_lr_and_decay_raise(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace("steps = 2", "steps = 2\nlr = 0.001\ndecay = 0.1", 1)
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="train.decay, train.lr"):
        load_config(config_path)


@pytest.mark.parametrize("name", ["aurora", "riemannian_aurora", "soap"])
def test_loads_matrix_optimizer_variants(tmp_path, name):
    config_path = tmp_path / "config.toml"
    write_config(config_path)
    text = config_path.read_text(encoding="utf-8").replace('name = "muon"', f'name = "{name}"', 1)
    config_path.write_text(text, encoding="utf-8")

    config = load_config(config_path)

    assert config.optimizer.name == name


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "not_real"},
        {"lr": 0.0},
        {"weight_decay": -0.1},
    ],
)
def test_invalid_optimizer_config_raises(kwargs):
    values = {"name": "muon", "lr": 0.001, "weight_decay": 0.1}
    values.update(kwargs)
    with pytest.raises(ValueError):
        OptimizerConfig(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beta": -0.1},
        {"beta": 1.0},
        {"nesterov": "yes"},
        {"pp_iterations": 0},
        {"pp_beta": 0.0},
        {"eps": 0.0},
    ],
)
def test_invalid_aurora_config_raises(kwargs):
    with pytest.raises(ValueError):
        AuroraOptimizerConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beta": -0.1},
        {"beta": 1.0},
        {"nesterov": "yes"},
        {"outer_steps": 0},
        {"cg_steps": 0},
        {"riemannian_eta": 0.0},
        {"retraction_steps": 0},
        {"eps": 0.0},
    ],
)
def test_invalid_riemannian_aurora_config_raises(kwargs):
    with pytest.raises(ValueError):
        RiemannianAuroraOptimizerConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"b1": -0.1},
        {"b1": 1.0},
        {"b2": -0.1},
        {"b2": 1.0},
        {"shampoo_beta": -0.5},
        {"shampoo_beta": 1.0},
        {"eps": 0.0},
        {"precondition_frequency": 0},
        {"max_precond_dim": 0},
        {"precondition_1d": "yes"},
        {"normalize_grads": "yes"},
        {"correct_bias": "yes"},
    ],
)
def test_invalid_soap_config_raises(kwargs):
    with pytest.raises(ValueError):
        SOAPOptimizerConfig(**kwargs)


def test_jax_profiling_config_parses(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(
        config_path,
        profiling_section="""
[profiling]
enabled = true
profiler = "jax"
start_step = 10
steps = 3
output_dir = "profiles"
""",
    )

    config = load_config(config_path)

    assert config.profiling.enabled is True
    assert config.profiling.profiler == "jax"
    assert config.profiling.start_step == 10
    assert config.profiling.steps == 3


def test_nsys_profiling_config_parses(tmp_path):
    config_path = tmp_path / "config.toml"
    write_config(
        config_path,
        profiling_section="""
[profiling]
enabled = true
profiler = "nsys"
start_step = 10
steps = 3
output_dir = "profiles"
""",
    )

    config = load_config(config_path)

    assert config.profiling.enabled is True
    assert config.profiling.profiler == "nsys"
    assert config.profiling.start_step == 10
    assert config.profiling.steps == 3


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": "yes"},
        {"profiler": "xplane"},
        {"start_step": -1},
        {"start_step": 0.5},
        {"steps": 0},
        {"steps": -1},
        {"output_dir": ""},
    ],
)
def test_invalid_profiling_config_raises(kwargs):
    with pytest.raises(ValueError):
        ProfilingConfig(**kwargs)


def test_dtype_from_name():
    assert jnp.dtype(dtype_from_name("fp32")) == jnp.dtype(jnp.float32)
    assert jnp.dtype(dtype_from_name("bf16")) == jnp.dtype(jnp.bfloat16)
    with pytest.raises(ValueError):
        dtype_from_name("fp16")
