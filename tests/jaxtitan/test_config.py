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
    assert spec.parallelism.mode == "ddp"
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


def test_load_config_accepts_trinity_recipe_section(tmp_path: Path) -> None:
    config_path = tmp_path / "trinity.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        + """
[model.trinity]
initial_dense_layers = 2
local_window = 32
local_layers_per_global = 3
attention_gate = true
qk_norm = true
norm_policy = "depth_scaled_sandwich"
embedding_scale = "sqrt_hidden"
init_std = 0.02
"""
    )

    spec = load_config(config_path)

    assert spec.model.name == "trinity"
    assert spec.model.trinity is not None
    assert spec.model.trinity.initial_dense_layers == 2
    assert spec.model.trinity.local_window == 32
    assert spec.model.trinity.local_layers_per_global == 3
    assert spec.model.trinity.attention_gate is True
    assert spec.model.trinity.qk_norm is True
    assert spec.model.trinity.init_std == 0.02


def test_load_config_accepts_trinity_moe_section(tmp_path: Path) -> None:
    config_path = tmp_path / "trinity-moe.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3
norm_policy = "afmoe_dual"

[model.trinity.moe]
num_experts = 4
top_k = 2
num_shared_experts = 2
route_scale = 1.5
"""
    )

    spec = load_config(config_path)

    assert spec.model.trinity is not None
    assert spec.model.trinity.moe is not None
    assert spec.model.trinity.moe.num_experts == 4
    assert spec.model.trinity.moe.top_k == 2
    assert spec.model.trinity.moe.expert_intermediate_size == spec.model.intermediate_size
    assert spec.model.trinity.moe.num_shared_experts == 2
    assert spec.model.trinity.moe.route_scale == 1.5
    assert spec.model.trinity.norm_policy == "afmoe_dual"


def test_load_config_rejects_missing_trinity_section(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-trinity.toml"
    config_path.write_text(MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"'))

    with pytest.raises(ConfigError, match="model.trinity"):
        load_config(config_path)


def test_load_config_rejects_trinity_section_for_decoder(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-decoder.toml"
    config_path.write_text(
        MINIMAL_CONFIG
        + """
[model.trinity]
initial_dense_layers = 2
local_window = 32
local_layers_per_global = 3
"""
    )

    with pytest.raises(ConfigError, match="model.trinity"):
        load_config(config_path)


def test_load_config_rejects_incomplete_trinity_section(tmp_path: Path) -> None:
    config_path = tmp_path / "incomplete-trinity.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        + """
[model.trinity]
initial_dense_layers = 2
local_window = 32
"""
    )

    with pytest.raises(ConfigError, match="model.trinity.local_layers_per_global"):
        load_config(config_path)


def test_load_config_rejects_invalid_trinity_moe_section(tmp_path: Path) -> None:
    base = (
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 2
top_k = 3
"""
    )
    config_path = tmp_path / "bad-moe.toml"
    config_path.write_text(base)

    with pytest.raises(ConfigError, match="top_k"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_shared_experts", "-1", "num_shared_experts"),
        ("route_scale", "0.0", "route_scale"),
        ("route_scale", '"large"', "route_scale"),
    ],
)
def test_load_config_rejects_invalid_afmoe_fields(tmp_path: Path, field: str, value: str, message: str) -> None:
    config_path = tmp_path / "bad-afmoe.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        + f"""
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 4
top_k = 2
{field} = {value}
"""
    )

    with pytest.raises(ConfigError, match=message):
        load_config(config_path)


def test_load_config_rejects_invalid_trinity_norm_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-norm-policy.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        + """
[model.trinity]
initial_dense_layers = 2
local_window = 32
local_layers_per_global = 3
norm_policy = "rmsnorm"
"""
    )

    with pytest.raises(ConfigError, match="norm_policy"):
        load_config(config_path)


def test_load_config_accepts_muon_fallback_schedule(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "adamw"', 'name = "muon"', 1)
        + """
[optimizer.adamw_fallback_schedule]
name = "constant"
peak_lr = 0.0006
"""
    )

    spec = load_config(config_path)

    assert spec.optimizer.name == "muon"
    assert spec.optimizer.adamw_fallback_schedule is not None
    assert spec.optimizer.adamw_fallback_schedule.peak_lr == 0.0006


def test_load_config_accepts_explicit_parallelism_modes(tmp_path: Path) -> None:
    ddp_path = tmp_path / "ddp.toml"
    ddp_path.write_text(MINIMAL_CONFIG + "\n[parallelism]\nmode = \"ddp\"\n")
    fsdp_path = tmp_path / "fsdp.toml"
    fsdp_path.write_text(
        MINIMAL_CONFIG.replace('axis_names = ["data"]', 'axis_names = ["data", "fsdp"]').replace(
            "axis_sizes = [1]",
            "axis_sizes = [1, 4]",
        )
        + "\n[parallelism]\nmode = \"fsdp\"\n"
    )
    zero2_path = tmp_path / "zero2.toml"
    zero2_path.write_text(
        MINIMAL_CONFIG.replace('axis_names = ["data"]', 'axis_names = ["data", "fsdp"]').replace(
            "axis_sizes = [1]",
            "axis_sizes = [1, 4]",
        )
        + "\n[parallelism]\nmode = \"zero2\"\n"
    )

    assert load_config(ddp_path).parallelism.mode == "ddp"
    assert load_config(fsdp_path).parallelism.mode == "fsdp"
    assert load_config(zero2_path).parallelism.mode == "zero2"


def test_load_config_rejects_invalid_parallelism_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(MINIMAL_CONFIG + "\n[parallelism]\nmode = \"zero2\"\n")

    with pytest.raises(ConfigError, match="parallelism.mode"):
        load_config(config_path)


def test_load_config_rejects_ddp_with_fsdp_axis_size_greater_than_one(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('axis_names = ["data"]', 'axis_names = ["data", "fsdp"]').replace(
            "axis_sizes = [1]",
            "axis_sizes = [1, 4]",
        )
        + "\n[parallelism]\nmode = \"ddp\"\n"
    )

    with pytest.raises(ConfigError, match="parallelism.mode='ddp'"):
        load_config(config_path)


def test_load_config_rejects_fsdp_without_fsdp_axis(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(MINIMAL_CONFIG + "\n[parallelism]\nmode = \"fsdp\"\n")

    with pytest.raises(ConfigError, match="requires a mesh fsdp axis"):
        load_config(config_path)


def test_load_config_rejects_zero2_without_fsdp_axis(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(MINIMAL_CONFIG + "\n[parallelism]\nmode = \"zero2\"\n")

    with pytest.raises(ConfigError, match="requires a mesh fsdp axis"):
        load_config(config_path)


@pytest.mark.parametrize("mode", ["fsdp", "zero2"])
def test_load_config_allows_muon_when_fsdp_axis_is_noop(tmp_path: Path, mode: str) -> None:
    config_path = tmp_path / f"{mode}.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "adamw"', 'name = "muon"', 1)
        .replace('axis_names = ["data"]', 'axis_names = ["data", "fsdp"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 1]")
        + f'\n[parallelism]\nmode = "{mode}"\n'
    )

    spec = load_config(config_path)

    assert spec.optimizer.name == "muon"
    assert spec.parallelism.mode == mode


@pytest.mark.parametrize("mode", ["fsdp", "zero2"])
def test_load_config_accepts_muon_with_real_fsdp_axis(tmp_path: Path, mode: str) -> None:
    config_path = tmp_path / f"{mode}-muon.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "adamw"', 'name = "muon"', 1)
        .replace('axis_names = ["data"]', 'axis_names = ["data", "fsdp"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 4]")
        + f'\n[parallelism]\nmode = "{mode}"\n'
    )

    spec = load_config(config_path)

    assert spec.optimizer.name == "muon"
    assert spec.parallelism.mode == mode


def test_load_config_rejects_public_dion2_optimizer_name(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-dion2.toml"
    config_path.write_text(MINIMAL_CONFIG.replace('name = "adamw"', 'name = "dion2"', 1))

    with pytest.raises(ConfigError, match="optimizer.name"):
        load_config(config_path)


def test_load_config_rejects_fallback_schedule_for_non_muon(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        MINIMAL_CONFIG
        + """
[optimizer.adamw_fallback_schedule]
name = "constant"
peak_lr = 0.0006
"""
    )

    with pytest.raises(ConfigError, match="adamw_fallback_schedule"):
        load_config(config_path)


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


def test_load_config_accepts_document_buffer_policy(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace(
            'tokenizer_id = "toy-tokenizer"',
            "\n".join(
                [
                    'tokenizer_id = "toy-tokenizer"',
                    'order = "document_buffer"',
                    "shuffle_seed = 123",
                    "document_buffer_size = 4",
                    "document_refill_size = 2",
                ]
            ),
        )
    )

    spec = load_config(config_path)

    assert spec.data.order == "document_buffer"
    assert spec.data.shuffle_seed == 123
    assert spec.data.document_buffer_size == 4
    assert spec.data.document_refill_size == 2


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        ('order = "rsdb"', "data.order"),
        ('order = "shuffle"', "shuffle_seed"),
        ('order = "document_buffer"', "shuffle_seed"),
        ('order = "document_buffer"\nshuffle_seed = 1', "document_buffer_size"),
        ('order = "document_buffer"\nshuffle_seed = 1\ndocument_buffer_size = 4', "document_refill_size"),
        (
            'order = "document_buffer"\nshuffle_seed = 1\ndocument_buffer_size = 4\ndocument_refill_size = 2\nprefetch = true',
            "prefetch",
        ),
        ('document_buffer_size = 4', "document buffer settings"),
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
    assert decoded["data"]["document_buffer_size"] is None
    assert resolved_json == run_spec_to_json(spec)
    assert resolved_config_sha256(spec) == sha256(resolved_json.encode("utf-8")).hexdigest()
    assert source_config_sha256(config_path) == sha256(MINIMAL_CONFIG.encode("utf-8")).hexdigest()
