from pathlib import Path
from hashlib import sha256
import json

import pytest

from jaxtitan.config import load_config, resolved_config_sha256, run_spec_to_json, source_config_sha256
from jaxtitan.errors import ConfigError

MINIMAL_CONFIG = """
[run]
id = "smoke"
seed = 11
output_dir = "runs"

[model]
name = "decoder"
variant = "tiny"
vocab_size = 32000
hidden_size = 128
intermediate_size = 512
num_layers = 2
num_heads = 4
max_seq_len = 64

[optimizer]
name = "adamw"
weight_decay = 0.1

[optimizer.schedule]
name = "constant"
peak_lr = 0.001

[data]
train_manifest = "data/train/manifest.json"
tokenizer_id = "toy-tokenizer"

[training]
seq_len = 64
global_batch_size = 2
target_tokens = 128
log_every_steps = 1
checkpoint_every_steps = 10

[mesh]
axis_names = ["data"]
axis_sizes = [1]
"""


def test_load_config_resolves_minimal_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)

    spec = load_config(config_path)

    assert spec.run_id == "smoke"
    assert spec.seed == 11
    assert spec.model.hidden_size == 128
    assert spec.model.intermediate_size == 512
    assert spec.model.rope_theta == 1_000_000.0
    assert spec.model.norm_epsilon == 1e-6
    assert spec.model.tied_embeddings is False
    assert spec.model.remat == "none"
    assert spec.optimizer.schedule.peak_lr == 0.001
    assert spec.data.train_manifest == Path("data/train/manifest.json")
    assert spec.data.order == "sequential"
    assert spec.data.shuffle_seed is None
    assert spec.data.worker_count == 0
    assert spec.data.worker_buffer_size == 1
    assert spec.data.prefetch is False
    assert spec.training.target_tokens == 128
    assert spec.training.gradient_accumulation_steps == 1
    assert spec.mesh.axis_names == ("data",)
    assert spec.artifacts.root == Path("runs")


def test_load_config_accepts_explicit_model_runtime_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace(
            "max_seq_len = 64",
            "\n".join(
                [
                    "max_seq_len = 64",
                    "n_kv_heads = 2",
                    "rope_theta = 10000.0",
                    "norm_epsilon = 0.00001",
                    'param_dtype = "bfloat16"',
                    'compute_dtype = "float32"',
                    'remat = "block"',
                ]
            ),
        )
    )

    spec = load_config(config_path)

    assert spec.model.n_kv_heads == 2
    assert spec.model.rope_theta == 10000.0
    assert spec.model.norm_epsilon == 0.00001
    assert spec.model.param_dtype == "bfloat16"
    assert spec.model.compute_dtype == "float32"
    assert spec.model.remat == "block"


def test_load_config_rejects_invalid_remat_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(MINIMAL_CONFIG.replace('variant = "tiny"', 'variant = "tiny"\nremat = "layer"'))

    with pytest.raises(ConfigError, match="model.remat"):
        load_config(config_path)


def test_load_config_accepts_validation_eval(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        MINIMAL_CONFIG
        + """
[[evals]]
name = "validation"
every_steps = 10
num_batches = 2
"""
    )

    spec = load_config(config_path)

    assert len(spec.evals) == 1
    assert spec.evals[0].name == "validation"
    assert spec.evals[0].every_steps == 10
    assert spec.evals[0].num_batches == 2


def test_load_config_accepts_gradient_accumulation_steps(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace(
            "global_batch_size = 2",
            "global_batch_size = 2\ngradient_accumulation_steps = 4",
        ).replace("target_tokens = 128", "target_tokens = 512")
    )

    spec = load_config(config_path)

    assert spec.training.gradient_accumulation_steps == 4


def test_load_config_accepts_data_loader_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace(
            'tokenizer_id = "toy-tokenizer"',
            "\n".join(
                [
                    'tokenizer_id = "toy-tokenizer"',
                    'order = "shuffle"',
                    "shuffle_seed = 123",
                    "worker_count = 2",
                    "worker_buffer_size = 3",
                    "prefetch = true",
                ]
            ),
        )
    )

    spec = load_config(config_path)

    assert spec.data.order == "shuffle"
    assert spec.data.shuffle_seed == 123
    assert spec.data.worker_count == 2
    assert spec.data.worker_buffer_size == 3
    assert spec.data.prefetch is True


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ('order = "rsdb"', "data.order"),
        ('order = "shuffle"', "shuffle_seed"),
        ('order = "sequential"\nshuffle_seed = 1', "shuffle_seed"),
        ('worker_count = -1', "worker_count"),
        ('worker_buffer_size = 0', "worker_buffer_size"),
    ],
)
def test_load_config_rejects_invalid_data_loader_policy(tmp_path: Path, replacement: str, match: str) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(MINIMAL_CONFIG.replace('tokenizer_id = "toy-tokenizer"', f'tokenizer_id = "toy-tokenizer"\n{replacement}'))

    with pytest.raises(ConfigError, match=match):
        load_config(config_path)


def test_load_config_rejects_cross_spec_mismatch(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(MINIMAL_CONFIG.replace("max_seq_len = 64", "max_seq_len = 32"))

    with pytest.raises(ConfigError, match="max_seq_len"):
        load_config(config_path)


def test_run_spec_json_and_hashes_are_stable(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(MINIMAL_CONFIG)
    spec = load_config(config_path)

    resolved_json = run_spec_to_json(spec)
    decoded = json.loads(resolved_json)

    assert decoded["run_id"] == "smoke"
    assert decoded["data"]["train_manifest"] == "data/train/manifest.json"
    assert decoded["data"]["order"] == "sequential"
    assert decoded["data"]["worker_buffer_size"] == 1
    assert resolved_json == run_spec_to_json(spec)
    assert resolved_config_sha256(spec) == sha256(resolved_json.encode("utf-8")).hexdigest()
    assert source_config_sha256(config_path) == sha256(MINIMAL_CONFIG.encode("utf-8")).hexdigest()
