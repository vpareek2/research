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
    assert spec.parallelism.tensor_parallel is False
    assert spec.parallelism.context_parallel is False
    assert spec.parallelism.expert_parallel is False
    assert spec.artifacts.root == Path("runs")
    assert spec.artifacts.wandb_enabled is False
    assert spec.artifacts.wandb_project == "jaxtitan"
    assert spec.artifacts.wandb_entity is None
    assert spec.artifacts.wandb_group is None
    assert spec.artifacts.wandb_tags == ()
    assert spec.artifacts.wandb_mode == "online"
    assert spec.profiling.enabled is False
    assert spec.profiling.trace_start_step == 3
    assert spec.profiling.trace_steps == 2
    assert spec.profiling.create_perfetto_trace is True
    assert spec.profiling.create_perfetto_link is False
    assert spec.kernels.enabled is False
    assert spec.kernels.strict is False
    assert spec.kernels.compile == "lazy"


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
    assert spec.model.trinity.moe.balance.name == "none"
    assert spec.training.loss.z_loss_weight == 0.0
    assert spec.model.trinity.norm_policy == "afmoe_dual"


def test_load_config_accepts_trinity_moe_balance_and_training_loss(tmp_path: Path) -> None:
    config_path = tmp_path / "trinity-moe-balance.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 4
top_k = 2

[model.trinity.moe.balance]
name = "smebu"
load_lr = 0.0007
momentum = 0.6
clamp = 3.0
sequence_aux_loss_weight = 0.0002

[training.loss]
z_loss_weight = 0.000001
"""
    )

    spec = load_config(config_path)

    assert spec.model.trinity is not None
    assert spec.model.trinity.moe is not None
    balance = spec.model.trinity.moe.balance
    assert balance.name == "smebu"
    assert balance.load_lr == pytest.approx(7e-4)
    assert balance.momentum == pytest.approx(0.6)
    assert balance.clamp == pytest.approx(3.0)
    assert balance.sequence_aux_loss_weight == pytest.approx(2e-4)
    assert spec.training.loss.z_loss_weight == pytest.approx(1e-6)


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
        ("balance.name", '"unknown"', "balance.name"),
        ("balance.load_lr", "0.0", "balance.load_lr"),
        ("balance.momentum", "0.0", "balance.momentum"),
        ("balance.clamp", "0.0", "balance.clamp"),
        ("balance.sequence_aux_loss_weight", "-1.0", "sequence_aux_loss_weight"),
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


def test_load_config_rejects_invalid_training_loss(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-loss.toml"
    config_path.write_text(
        MINIMAL_CONFIG
        + """
[training.loss]
z_loss_weight = -1.0
"""
    )

    with pytest.raises(ConfigError, match="z_loss_weight"):
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


def test_load_config_accepts_wandb_artifacts_section(tmp_path: Path) -> None:
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(
        MINIMAL_CONFIG
        + """
[artifacts]
wandb_enabled = true
wandb_project = "small-lm"
wandb_entity = "lab"
wandb_group = "ablations"
wandb_tags = ["smoke", "muon"]
wandb_mode = "offline"
"""
    )

    spec = load_config(config_path)

    assert spec.artifacts.wandb_enabled is True
    assert spec.artifacts.wandb_project == "small-lm"
    assert spec.artifacts.wandb_entity == "lab"
    assert spec.artifacts.wandb_group == "ablations"
    assert spec.artifacts.wandb_tags == ("smoke", "muon")
    assert spec.artifacts.wandb_mode == "offline"


@pytest.mark.parametrize(
    ("artifact_block", "match"),
    [
        ('wandb_project = ""', "wandb_project"),
        ('wandb_mode = "dryrun"', "wandb_mode"),
        ('wandb_tags = ["ok", ""]', "wandb_tags"),
        ('wandb_tags = "smoke"', "wandb_tags"),
    ],
)
def test_load_config_rejects_invalid_wandb_artifacts(tmp_path: Path, artifact_block: str, match: str) -> None:
    config_path = tmp_path / "bad-wandb.toml"
    config_path.write_text(
        MINIMAL_CONFIG
        + f"""
[artifacts]
{artifact_block}
"""
    )

    with pytest.raises(ConfigError, match=match):
        load_config(config_path)


def test_load_config_accepts_profiling_section(tmp_path: Path) -> None:
    config_path = tmp_path / "profiled.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace("target_tokens = 128", "target_tokens = 512")
        + """
[profiling]
enabled = true
trace_start_step = 2
trace_steps = 3
create_perfetto_trace = true
create_perfetto_link = false
"""
    )

    spec = load_config(config_path)

    assert spec.profiling.enabled is True
    assert spec.profiling.trace_start_step == 2
    assert spec.profiling.trace_steps == 3
    assert spec.profiling.trace_end_step == 4
    assert spec.profiling.create_perfetto_trace is True
    assert spec.profiling.create_perfetto_link is False


@pytest.mark.parametrize(
    ("profiling_block", "match"),
    [
        ('enabled = "yes"', "profiling.enabled"),
        ("trace_start_step = 0", "trace_start_step"),
        ("trace_steps = 0", "trace_steps"),
        ('create_perfetto_trace = "true"', "create_perfetto_trace"),
        ('create_perfetto_link = "false"', "create_perfetto_link"),
    ],
)
def test_load_config_rejects_invalid_profiling_section(
    tmp_path: Path,
    profiling_block: str,
    match: str,
) -> None:
    config_path = tmp_path / "bad-profiling.toml"
    config_path.write_text(
        MINIMAL_CONFIG
        + f"""
[profiling]
{profiling_block}
"""
    )

    with pytest.raises(ConfigError, match=match):
        load_config(config_path)


def test_load_config_rejects_unreachable_profiling_start_step(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-profiling-window.toml"
    config_path.write_text(
        MINIMAL_CONFIG
        + """
[profiling]
enabled = true
trace_start_step = 2
trace_steps = 1
"""
    )

    with pytest.raises(ConfigError, match="trace_start_step"):
        load_config(config_path)


def test_load_config_accepts_kernel_backend_section(tmp_path: Path) -> None:
    config_path = tmp_path / "kernels.toml"
    config_path.write_text(
        MINIMAL_CONFIG
        + """
[kernels]
enabled = true
strict = false
compile = "ahead_of_time"
"""
    )

    spec = load_config(config_path)

    assert spec.kernels.enabled is True
    assert spec.kernels.strict is False
    assert spec.kernels.compile == "ahead_of_time"


@pytest.mark.parametrize(
    ("kernel_block", "match"),
    [
        ('enabled = "yes"', "kernels.enabled"),
        ('strict = "no"', "kernels.strict"),
        ('compile = "now"', "kernels.compile"),
    ],
)
def test_load_config_rejects_invalid_kernel_backend_section(
    tmp_path: Path,
    kernel_block: str,
    match: str,
) -> None:
    config_path = tmp_path / "bad-kernels.toml"
    config_path.write_text(
        MINIMAL_CONFIG
        + f"""
[kernels]
{kernel_block}
"""
    )

    with pytest.raises(ConfigError, match=match):
        load_config(config_path)


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


def test_load_config_accepts_tensor_parallelism(tmp_path: Path) -> None:
    config_path = tmp_path / "tp.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('axis_names = ["data"]', 'axis_names = ["data", "tp"]').replace(
            "axis_sizes = [1]",
            "axis_sizes = [1, 4]",
        )
        + "\n[parallelism]\ntensor_parallel = true\n"
    )

    spec = load_config(config_path)

    assert spec.parallelism.tensor_parallel is True
    assert spec.mesh.axis_names == ("data", "tp")


def test_load_config_accepts_context_parallelism(tmp_path: Path) -> None:
    config_path = tmp_path / "cp.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('axis_names = ["data"]', 'axis_names = ["data", "cp"]').replace(
            "axis_sizes = [1]",
            "axis_sizes = [1, 4]",
        )
        + "\n[parallelism]\ncontext_parallel = true\n"
    )

    spec = load_config(config_path)

    assert spec.parallelism.context_parallel is True
    assert spec.mesh.axis_names == ("data", "cp")


def test_load_config_rejects_cp_axis_without_context_parallel(tmp_path: Path) -> None:
    config_path = tmp_path / "cp-disabled.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('axis_names = ["data"]', 'axis_names = ["data", "cp"]').replace(
            "axis_sizes = [1]",
            "axis_sizes = [1, 2]",
        )
    )

    with pytest.raises(ConfigError, match="context_parallel"):
        load_config(config_path)


def test_load_config_rejects_context_parallel_without_cp_axis(tmp_path: Path) -> None:
    config_path = tmp_path / "cp-missing.toml"
    config_path.write_text(MINIMAL_CONFIG + "\n[parallelism]\ncontext_parallel = true\n")

    with pytest.raises(ConfigError, match="mesh cp axis"):
        load_config(config_path)


def test_load_config_rejects_context_parallel_non_divisible_sequence_length(tmp_path: Path) -> None:
    config_path = tmp_path / "cp-bad-seq.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace("seq_len = 64", "seq_len = 62")
        .replace('axis_names = ["data"]', 'axis_names = ["data", "cp"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 4]")
        + "\n[parallelism]\ncontext_parallel = true\n"
    )

    with pytest.raises(ConfigError, match="training.seq_len"):
        load_config(config_path)


def test_load_config_rejects_tp_axis_without_tensor_parallel(tmp_path: Path) -> None:
    config_path = tmp_path / "tp-disabled.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('axis_names = ["data"]', 'axis_names = ["data", "tp"]').replace(
            "axis_sizes = [1]",
            "axis_sizes = [1, 2]",
        )
    )

    with pytest.raises(ConfigError, match="tensor_parallel"):
        load_config(config_path)


def test_load_config_rejects_tensor_parallel_without_tp_axis(tmp_path: Path) -> None:
    config_path = tmp_path / "tp-missing.toml"
    config_path.write_text(MINIMAL_CONFIG + "\n[parallelism]\ntensor_parallel = true\n")

    with pytest.raises(ConfigError, match="mesh tp axis"):
        load_config(config_path)


def test_load_config_rejects_tensor_parallel_non_divisible_heads(tmp_path: Path) -> None:
    config_path = tmp_path / "tp-bad-heads.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('axis_names = ["data"]', 'axis_names = ["data", "tp"]').replace(
            "axis_sizes = [1]",
            "axis_sizes = [1, 3]",
        )
        + "\n[parallelism]\ntensor_parallel = true\n"
    )

    with pytest.raises(ConfigError, match="model.hidden_size|model.intermediate_size|model.num_heads"):
        load_config(config_path)


def test_load_config_rejects_tensor_parallel_non_divisible_vocab(tmp_path: Path) -> None:
    config_path = tmp_path / "tp-bad-vocab.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace("vocab_size = 32000", "vocab_size = 32001")
        .replace('axis_names = ["data"]', 'axis_names = ["data", "tp"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 2]")
        + "\n[parallelism]\ntensor_parallel = true\n"
    )

    with pytest.raises(ConfigError, match="model.vocab_size"):
        load_config(config_path)


def test_load_config_rejects_tensor_parallel_non_divisible_sequence_length(tmp_path: Path) -> None:
    config_path = tmp_path / "tp-bad-seq.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace("seq_len = 64", "seq_len = 62")
        .replace('axis_names = ["data"]', 'axis_names = ["data", "tp"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 4]")
        + "\n[parallelism]\ntensor_parallel = true\n"
    )

    with pytest.raises(ConfigError, match="training.seq_len"):
        load_config(config_path)


def test_load_config_accepts_muon_with_tensor_parallel(tmp_path: Path) -> None:
    config_path = tmp_path / "tp-muon.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "adamw"', 'name = "muon"', 1)
        .replace('axis_names = ["data"]', 'axis_names = ["data", "tp"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 2]")
        + "\n[parallelism]\ntensor_parallel = true\n"
    )

    spec = load_config(config_path)

    assert spec.optimizer.name == "muon"
    assert spec.parallelism.tensor_parallel is True


def test_load_config_accepts_trinity_moe_with_tensor_parallel_adamw(tmp_path: Path) -> None:
    config_path = tmp_path / "tp-moe.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        .replace('axis_names = ["data"]', 'axis_names = ["data", "tp"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 2]")
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 8
top_k = 2
num_shared_experts = 1

[parallelism]
tensor_parallel = true
"""
    )

    spec = load_config(config_path)

    assert spec.model.name == "trinity"
    assert spec.model.trinity is not None and spec.model.trinity.moe is not None
    assert spec.parallelism.tensor_parallel is True
    assert spec.optimizer.name == "adamw"


def test_load_config_accepts_trinity_moe_with_tensor_and_expert_parallel(tmp_path: Path) -> None:
    config_path = tmp_path / "tp-ep-moe.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        .replace('axis_names = ["data"]', 'axis_names = ["data", "tp", "ep"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 2, 2]")
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 8
top_k = 2
num_shared_experts = 1

[parallelism]
tensor_parallel = true
expert_parallel = true
"""
    )

    spec = load_config(config_path)

    assert spec.parallelism.tensor_parallel is True
    assert spec.parallelism.expert_parallel is True
    assert spec.mesh.axis_names == ("data", "tp", "ep")


def test_load_config_accepts_expert_parallelism_with_ep_axis(tmp_path: Path) -> None:
    config_path = tmp_path / "ep.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        .replace('axis_names = ["data"]', 'axis_names = ["data", "ep"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 4]")
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 8
top_k = 2

[parallelism]
mode = "ddp"
expert_parallel = true
"""
    )

    spec = load_config(config_path)

    assert spec.parallelism.mode == "ddp"
    assert spec.parallelism.expert_parallel is True
    assert spec.parallelism.expert_parallel_axis == "auto"
    assert spec.mesh.axis_names == ("data", "ep")


def test_load_config_accepts_folded_fsdp_expert_parallelism(tmp_path: Path) -> None:
    config_path = tmp_path / "folded-fsdp-ep.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        .replace('axis_names = ["data"]', 'axis_names = ["data", "fsdp"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 2]")
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 8
top_k = 2

[parallelism]
mode = "fsdp"
expert_parallel = true
"""
    )

    spec = load_config(config_path)

    assert spec.parallelism.mode == "fsdp"
    assert spec.parallelism.expert_parallel is True
    assert spec.parallelism.expert_parallel_axis == "auto"
    assert spec.mesh.axis_names == ("data", "fsdp")


def test_load_config_accepts_explicit_folded_expert_parallel_axis(tmp_path: Path) -> None:
    config_path = tmp_path / "explicit-folded.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        .replace('axis_names = ["data"]', 'axis_names = ["data", "fsdp"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 2]")
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 8
top_k = 2

[parallelism]
mode = "fsdp"
expert_parallel = true
expert_parallel_axis = "fsdp"
"""
    )

    spec = load_config(config_path)

    assert spec.parallelism.expert_parallel_axis == "fsdp"


def test_load_config_accepts_expert_region_fsdp(tmp_path: Path) -> None:
    config_path = tmp_path / "expert-region-fsdp.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        .replace("intermediate_size = 512", "intermediate_size = 512")
        .replace('axis_names = ["data"]', 'axis_names = ["data", "fsdp", "ep", "expert_fsdp"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 1, 2, 2]")
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 8
top_k = 2
expert_intermediate_size = 512

[parallelism]
mode = "fsdp"
expert_parallel = true
expert_parallel_axis = "ep"
"""
    )

    spec = load_config(config_path)

    assert spec.parallelism.mode == "fsdp"
    assert spec.parallelism.expert_parallel is True
    assert spec.parallelism.expert_parallel_axis == "ep"
    assert spec.mesh.axis_names == ("data", "fsdp", "ep", "expert_fsdp")


def test_load_config_rejects_expert_region_fsdp_with_muon(tmp_path: Path) -> None:
    config_path = tmp_path / "expert-region-fsdp-muon.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        .replace('name = "adamw"', 'name = "muon"', 1)
        .replace('axis_names = ["data"]', 'axis_names = ["data", "fsdp", "ep", "expert_fsdp"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 1, 2, 2]")
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 8
top_k = 2
expert_intermediate_size = 512

[parallelism]
mode = "fsdp"
expert_parallel = true
expert_parallel_axis = "ep"
"""
    )

    with pytest.raises(ConfigError, match="optimizer.name='muon'"):
        load_config(config_path)


def test_load_config_rejects_expert_region_fsdp_without_dedicated_ep_axis(tmp_path: Path) -> None:
    config_path = tmp_path / "expert-region-fsdp-folded.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        .replace('axis_names = ["data"]', 'axis_names = ["data", "fsdp", "expert_fsdp"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 2, 2]")
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 8
top_k = 2
expert_intermediate_size = 512

[parallelism]
mode = "fsdp"
expert_parallel = true
"""
    )

    with pytest.raises(ConfigError, match="expert_fsdp axis"):
        load_config(config_path)


def test_load_config_rejects_expert_region_fsdp_non_divisible_width(tmp_path: Path) -> None:
    config_path = tmp_path / "expert-region-fsdp-bad-width.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        .replace('axis_names = ["data"]', 'axis_names = ["data", "fsdp", "ep", "expert_fsdp"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 1, 2, 2]")
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 8
top_k = 2
expert_intermediate_size = 513

[parallelism]
mode = "fsdp"
expert_parallel = true
expert_parallel_axis = "ep"
"""
    )

    with pytest.raises(ConfigError, match="expert_intermediate_size"):
        load_config(config_path)


def test_load_config_rejects_unused_ep_axis(tmp_path: Path) -> None:
    config_path = tmp_path / "unused-ep.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('axis_names = ["data"]', 'axis_names = ["data", "ep"]').replace(
            "axis_sizes = [1]",
            "axis_sizes = [1, 2]",
        )
    )

    with pytest.raises(ConfigError, match="expert_parallel"):
        load_config(config_path)


def test_load_config_rejects_expert_parallel_without_ep_axis(tmp_path: Path) -> None:
    config_path = tmp_path / "missing-ep.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 8
top_k = 2

[parallelism]
expert_parallel = true
"""
    )

    with pytest.raises(ConfigError, match="ep axis"):
        load_config(config_path)


def test_load_config_accepts_data_axis_rdep_expert_parallel(tmp_path: Path) -> None:
    config_path = tmp_path / "data-rdep.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        .replace("axis_sizes = [1]", "axis_sizes = [2]")
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 8
top_k = 2

[parallelism]
expert_parallel = true
expert_parallel_axis = "data"
"""
    )

    spec = load_config(config_path)

    assert spec.parallelism.expert_parallel is True
    assert spec.parallelism.expert_parallel_axis == "data"
    assert spec.mesh.axis_names == ("data",)


def test_load_config_rejects_invalid_expert_parallel_axis(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-expert-axis.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        .replace('axis_names = ["data"]', 'axis_names = ["data", "ep"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 2]")
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 8
top_k = 2

[parallelism]
expert_parallel = true
expert_parallel_axis = "bad"
"""
    )

    with pytest.raises(ConfigError, match="expert_parallel_axis"):
        load_config(config_path)


def test_load_config_rejects_expert_parallel_axis_without_expert_parallel(tmp_path: Path) -> None:
    config_path = tmp_path / "unused-expert-axis.toml"
    config_path.write_text(MINIMAL_CONFIG + '\n[parallelism]\nexpert_parallel_axis = "fsdp"\n')

    with pytest.raises(ConfigError, match="expert_parallel_axis"):
        load_config(config_path)


def test_load_config_rejects_folded_expert_parallel_for_ddp(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-folded-ddp.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        .replace('axis_names = ["data"]', 'axis_names = ["data", "fsdp"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 1]")
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 8
top_k = 2

[parallelism]
mode = "ddp"
expert_parallel = true
expert_parallel_axis = "fsdp"
"""
    )

    with pytest.raises(ConfigError, match="expert_parallel_axis='fsdp'"):
        load_config(config_path)


def test_load_config_rejects_expert_parallel_without_moe(tmp_path: Path) -> None:
    config_path = tmp_path / "dense-ep.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('axis_names = ["data"]', 'axis_names = ["data", "ep"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 2]")
        + """
[parallelism]
expert_parallel = true
"""
    )

    with pytest.raises(ConfigError, match="Trinity MoE"):
        load_config(config_path)


def test_load_config_rejects_expert_count_not_divisible_by_ep_axis(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-ep.toml"
    config_path.write_text(
        MINIMAL_CONFIG.replace('name = "decoder"', 'name = "trinity"')
        .replace('axis_names = ["data"]', 'axis_names = ["data", "ep"]')
        .replace("axis_sizes = [1]", "axis_sizes = [1, 4]")
        + """
[model.trinity]
initial_dense_layers = 1
local_window = 32
local_layers_per_global = 3

[model.trinity.moe]
num_experts = 6
top_k = 2

[parallelism]
expert_parallel = true
"""
    )

    with pytest.raises(ConfigError, match="num_experts"):
        load_config(config_path)


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


def test_load_config_accepts_hf_streaming_data(tmp_path: Path) -> None:
    config_path = tmp_path / "streaming.toml"
    config_path.write_text(_streaming_config_text())

    spec = load_config(config_path)

    assert spec.data.mode == "hf_streaming"
    assert spec.data.train_manifest is None
    assert spec.data.tokenizer_id == "gpt2"
    assert spec.data.order == "sequential"
    assert spec.data.hf_streaming is not None
    assert spec.data.hf_streaming.dataset == "HuggingFaceFW/fineweb"
    assert spec.data.hf_streaming.name == "sample-10BT"
    assert spec.data.hf_streaming.split == "train"
    assert spec.data.hf_streaming.revision == "abc123"
    assert spec.data.hf_streaming.text_column == "text"
    assert spec.data.hf_streaming.append_eot is True


def test_load_config_accepts_hf_streaming_with_prepared_validation_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "streaming-eval.toml"
    config_path.write_text(
        _streaming_config_text(validation_manifest='validation_manifest = "data/val/manifest.json"')
        + """
[[evals]]
name = "validation"
every_steps = 10
num_batches = 1
"""
    )

    spec = load_config(config_path)

    assert spec.data.mode == "hf_streaming"
    assert spec.data.validation_manifest == Path("data/val/manifest.json")
    assert len(spec.evals) == 1


@pytest.mark.parametrize(
    ("data_section", "match"),
    [
        (
            """
[data]
mode = "hf_streaming"
tokenizer_id = "gpt2"
order = "sequential"

[data.hf_streaming]
dataset = "HuggingFaceFW/fineweb"
name = "sample-10BT"
split = "train"
text_column = "text"
""",
            "revision",
        ),
        (
            """
[data]
mode = "hf_streaming"
train_manifest = "data/train/manifest.json"
tokenizer_id = "gpt2"
order = "sequential"

[data.hf_streaming]
dataset = "HuggingFaceFW/fineweb"
name = "sample-10BT"
split = "train"
revision = "abc123"
text_column = "text"
""",
            "train_manifest",
        ),
        (
            """
[data]
mode = "hf_streaming"
tokenizer_id = "gpt2"
order = "document_buffer"
shuffle_seed = 1
document_buffer_size = 4
document_refill_size = 2

[data.hf_streaming]
dataset = "HuggingFaceFW/fineweb"
name = "sample-10BT"
split = "train"
revision = "abc123"
text_column = "text"
""",
            "hf_streaming",
        ),
        (
            """
[data]
mode = "hf_streaming"
tokenizer_id = "gpt2"
order = "sequential"
worker_count = 1

[data.hf_streaming]
dataset = "HuggingFaceFW/fineweb"
name = "sample-10BT"
split = "train"
revision = "abc123"
text_column = "text"
""",
            "worker_count",
        ),
    ],
)
def test_load_config_rejects_invalid_hf_streaming_data(
    tmp_path: Path,
    data_section: str,
    match: str,
) -> None:
    config_path = tmp_path / "bad-streaming.toml"
    config_path.write_text(_replace_data_section(MINIMAL_CONFIG, data_section))

    with pytest.raises(ConfigError, match=match):
        load_config(config_path)


def test_load_config_rejects_hf_streaming_eval_without_validation_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "bad-streaming-eval.toml"
    config_path.write_text(
        _streaming_config_text()
        + """
[[evals]]
name = "validation"
every_steps = 10
num_batches = 1
"""
    )

    with pytest.raises(ConfigError, match="validation_manifest"):
        load_config(config_path)


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


def _streaming_config_text(*, validation_manifest: str = "") -> str:
    return _replace_data_section(
        MINIMAL_CONFIG,
        f"""
[data]
mode = "hf_streaming"
tokenizer_id = "gpt2"
{validation_manifest}
order = "sequential"

[data.hf_streaming]
dataset = "HuggingFaceFW/fineweb"
name = "sample-10BT"
split = "train"
revision = "abc123"
text_column = "text"
append_eot = true
""",
    )


def _replace_data_section(config: str, replacement: str) -> str:
    start = config.index("[data]")
    end = config.index("[training]")
    return config[:start] + replacement.strip() + "\n\n" + config[end:]
