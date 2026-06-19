from pathlib import Path
from types import SimpleNamespace

import pytest

from jaxtitan.config import load_config
from jaxtitan.errors import ContractError
from jaxtitan.runtime.resume import (
    build_resume_compat,
    checkpoint_metadata,
    validate_resume_compat,
    validate_resume_metadata,
)
from jaxtitan.runtime.training import _with_runtime_schedule_steps
from jaxtitan.state import DataPipelineState, HostState


def test_resume_fingerprint_is_stable_for_identical_specs(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = _manifest(prepared_dataset_factory, "stable")
    first = _runtime_spec(tmp_path, manifest)
    second = _runtime_spec(tmp_path, manifest)

    assert build_resume_compat(first) == build_resume_compat(second)


def test_resume_fingerprint_ignores_safe_runtime_controls(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = _manifest(prepared_dataset_factory, "controls")
    first = _runtime_spec(tmp_path, manifest, target_tokens=128, log_every_steps=1, checkpoint_every_steps=10)
    second = _runtime_spec(tmp_path, manifest, target_tokens=256, log_every_steps=5, checkpoint_every_steps=20)

    assert build_resume_compat(first).runtime_fingerprint == build_resume_compat(second).runtime_fingerprint


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 16},
        {"remat": "block"},
        {"optimizer_name": "muon"},
        {"weight_decay": 0.2},
        {"axis_sizes": (2,)},
        {"axis_names": ("data", "fsdp"), "axis_sizes": (1, 4), "parallelism_mode": "fsdp", "hidden_size": 16, "intermediate_size": 32, "num_heads": 4, "n_kv_heads": 4},
        {"axis_names": ("data", "fsdp"), "axis_sizes": (1, 4), "parallelism_mode": "zero2", "hidden_size": 16, "intermediate_size": 32, "num_heads": 4, "n_kv_heads": 4},
        {"seed": 12},
        {"precision": "fp32"},
        {"seq_len": 2},
        {"global_batch_size": 1},
        {"gradient_accumulation_steps": 2},
        {"z_loss_weight": 1e-6},
        {"data_order": "shuffle", "shuffle_seed": 123},
        {"worker_count": 1},
        {"worker_buffer_size": 2},
        {"prefetch": True},
    ],
)
def test_resume_fingerprint_changes_for_unsafe_fields(
    tmp_path: Path,
    prepared_dataset_factory,
    kwargs: dict,
) -> None:
    manifest = _manifest(prepared_dataset_factory, "unsafe")
    base = _runtime_spec(tmp_path, manifest)
    changed = _runtime_spec(tmp_path, manifest, **kwargs)

    assert build_resume_compat(base).runtime_fingerprint != build_resume_compat(changed).runtime_fingerprint


def test_resume_fingerprint_changes_for_tokenizer_identity(tmp_path: Path, prepared_dataset_factory) -> None:
    first_manifest = _manifest(prepared_dataset_factory, "tokenizer-first")
    second_manifest = prepared_dataset_factory(
        "tokenizer-second",
        tokenizer_id="other-tokenizer",
        shard_token_groups=(tuple(range(0, 30)),),
        train_tokens=25,
    )
    base = _runtime_spec(tmp_path, first_manifest)
    changed = _runtime_spec(tmp_path, second_manifest, tokenizer_id="other-tokenizer")

    assert build_resume_compat(base).runtime_fingerprint != build_resume_compat(changed).runtime_fingerprint


def test_resume_fingerprint_changes_for_data_manifest_hash(tmp_path: Path, prepared_dataset_factory) -> None:
    first_manifest = _manifest(prepared_dataset_factory, "first")
    second_manifest = prepared_dataset_factory("second", shard_token_groups=(tuple(range(10, 40)),), train_tokens=25)
    first = _runtime_spec(tmp_path, first_manifest)
    second = _runtime_spec(tmp_path, second_manifest)

    assert build_resume_compat(first).runtime_fingerprint != build_resume_compat(second).runtime_fingerprint


def test_resume_fingerprint_changes_for_document_buffer_policy(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = prepared_dataset_factory(
        "document-buffer",
        shard_token_groups=(tuple(range(0, 80)),),
        train_tokens=48,
        document_offsets=(0, 3, 6, 9, 12, 20, 32, 48, 80),
    )
    base = _runtime_spec(
        tmp_path,
        manifest,
        data_order="document_buffer",
        shuffle_seed=123,
        document_buffer_size=3,
        document_refill_size=2,
    )
    changed = _runtime_spec(
        tmp_path,
        manifest,
        data_order="document_buffer",
        shuffle_seed=123,
        document_buffer_size=4,
        document_refill_size=2,
    )

    assert build_resume_compat(base).runtime_fingerprint != build_resume_compat(changed).runtime_fingerprint


def test_resume_fingerprint_changes_for_moe_balance_policy(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = _manifest(prepared_dataset_factory, "moe-balance")
    fixed_bias = _runtime_spec(tmp_path, manifest, trinity_moe_balance_name="none")
    smebu = _runtime_spec(tmp_path, manifest, trinity_moe_balance_name="smebu")
    changed_aux_weight = _runtime_spec(
        tmp_path,
        manifest,
        trinity_moe_balance_name="smebu",
        sequence_aux_loss_weight=2e-4,
    )

    assert build_resume_compat(fixed_bias).runtime_fingerprint != build_resume_compat(smebu).runtime_fingerprint
    assert build_resume_compat(smebu).runtime_fingerprint != build_resume_compat(changed_aux_weight).runtime_fingerprint


def test_resume_fingerprint_changes_for_expert_parallel_axis(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = _manifest(prepared_dataset_factory, "ep")
    ep_two = _runtime_spec(
        tmp_path,
        manifest,
        axis_names=("data", "ep"),
        axis_sizes=(1, 2),
        expert_parallel=True,
        trinity_moe_num_experts=4,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        n_kv_heads=4,
    )
    ep_four = _runtime_spec(
        tmp_path,
        manifest,
        axis_names=("data", "ep"),
        axis_sizes=(1, 4),
        expert_parallel=True,
        trinity_moe_num_experts=4,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        n_kv_heads=4,
    )

    assert build_resume_compat(ep_two).runtime_fingerprint != build_resume_compat(ep_four).runtime_fingerprint
    assert build_resume_compat(ep_two).payload["parallelism"]["expert_parallel"] is True
    assert build_resume_compat(ep_two).payload["parallelism"]["expert_parallel_policy"] == {
        "enabled": True,
        "axis": "ep",
        "axis_size": 2,
        "axis_sharing": "dedicated_ep",
        "expert_fsdp_axis": None,
        "expert_fsdp_axis_size": 1,
        "expert_fsdp_axis_sharing": None,
        "num_experts": 4,
        "experts_per_rank": 2,
        "dispatcher_backend": "all_to_all",
        "capacity_policy": "strict_dropless_static_source_buckets",
        "token_partition": "assignment_index_mod_ep",
        "combine_policy": "reverse_all_to_all_then_psum",
    }


def test_resume_fingerprint_changes_for_data_axis_rdep(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = _manifest(prepared_dataset_factory, "rdep")
    product_ep = _runtime_spec(
        tmp_path,
        manifest,
        axis_names=("data", "ep"),
        axis_sizes=(1, 2),
        expert_parallel=True,
        trinity_moe_num_experts=4,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        n_kv_heads=4,
    )
    rdep = _runtime_spec(
        tmp_path,
        manifest,
        axis_names=("data",),
        axis_sizes=(2,),
        expert_parallel=True,
        expert_parallel_axis="data",
        trinity_moe_num_experts=4,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        n_kv_heads=4,
    )

    rdep_payload = build_resume_compat(rdep).payload["parallelism"]["expert_parallel_policy"]
    assert build_resume_compat(product_ep).runtime_fingerprint != build_resume_compat(rdep).runtime_fingerprint
    assert rdep_payload["axis"] == "data"
    assert rdep_payload["axis_sharing"] == "shared_with_data"
    assert rdep_payload["dispatcher_backend"] == "rdep_static"
    assert rdep_payload["token_partition"] == "route_row_source_data_axis"
    assert rdep_payload["combine_policy"] == "return_by_route_row_identity"
    assert rdep_payload["route_row_identity"] == "((source_rank * T) + token) * top_k + slot"


def test_resume_fingerprint_changes_for_folded_expert_parallel_axis(
    tmp_path: Path,
    prepared_dataset_factory,
) -> None:
    manifest = _manifest(prepared_dataset_factory, "folded-ep")
    product_ep = _runtime_spec(
        tmp_path,
        manifest,
        axis_names=("data", "ep"),
        axis_sizes=(1, 2),
        expert_parallel=True,
        trinity_moe_num_experts=4,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        n_kv_heads=4,
    )
    folded_ep = _runtime_spec(
        tmp_path,
        manifest,
        axis_names=("data", "fsdp"),
        axis_sizes=(1, 2),
        parallelism_mode="fsdp",
        expert_parallel=True,
        trinity_moe_num_experts=4,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        n_kv_heads=4,
    )

    product_payload = build_resume_compat(product_ep).payload
    folded_payload = build_resume_compat(folded_ep).payload

    assert product_payload["parallelism"]["expert_parallel_policy"]["axis"] == "ep"
    assert folded_payload["parallelism"]["expert_parallel_policy"]["axis"] == "fsdp"
    assert folded_payload["parallelism"]["expert_parallel_policy"]["axis_sharing"] == "shared_with_fsdp"
    assert build_resume_compat(product_ep).runtime_fingerprint != build_resume_compat(folded_ep).runtime_fingerprint


def test_resume_fingerprint_changes_for_expert_region_fsdp_axis(
    tmp_path: Path,
    prepared_dataset_factory,
) -> None:
    manifest = _manifest(prepared_dataset_factory, "expert-fsdp")
    product_ep = _runtime_spec(
        tmp_path,
        manifest,
        axis_names=("data", "fsdp", "ep"),
        axis_sizes=(1, 2, 2),
        parallelism_mode="fsdp",
        expert_parallel=True,
        trinity_moe_num_experts=4,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        n_kv_heads=4,
    )
    expert_fsdp = _runtime_spec(
        tmp_path,
        manifest,
        axis_names=("data", "fsdp", "ep", "expert_fsdp"),
        axis_sizes=(1, 1, 2, 2),
        parallelism_mode="fsdp",
        expert_parallel=True,
        trinity_moe_num_experts=4,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        n_kv_heads=4,
    )

    product_payload = build_resume_compat(product_ep).payload
    expert_fsdp_payload = build_resume_compat(expert_fsdp).payload

    assert product_payload["parallelism"]["expert_fsdp_policy"]["enabled"] is False
    assert expert_fsdp_payload["parallelism"]["expert_fsdp_policy"] == {
        "enabled": True,
        "axis": "expert_fsdp",
        "axis_size": 2,
        "axis_sharing": "expert_region_internal",
    }
    assert build_resume_compat(product_ep).runtime_fingerprint != build_resume_compat(expert_fsdp).runtime_fingerprint


def test_resume_metadata_contains_compatibility_payload(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = _manifest(prepared_dataset_factory, "metadata")
    spec = _runtime_spec(tmp_path, manifest)
    metadata = checkpoint_metadata(
        spec,
        {"step": 3, "tokens_seen": 384, "loss": 1.25},
        reason="interval",
    )

    assert metadata["schema_version"] == 1
    assert metadata["compat_version"] == 1
    assert metadata["run_id"] == "smoke"
    assert metadata["checkpoint"] == {"step": 3, "tokens_seen": 384, "reason": "interval"}
    assert metadata["metrics"] == {"train_loss": 1.25, "eval_loss": None}
    assert metadata["runtime_fingerprint"] == build_resume_compat(spec).runtime_fingerprint
    assert metadata["compatibility"] == build_resume_compat(spec).payload
    assert metadata["compatibility"]["optimizer"]["name"] == "adamw"
    assert metadata["compatibility"]["optimizer"]["policy"]["name"] == "adamw"
    assert metadata["compatibility"]["optimizer"]["policy"]["muon"]["scale_mode"] == "match_rms_adamw"
    assert metadata["compatibility"]["optimizer"]["policy"]["muon"]["newton_schulz_precision"] == "bfloat16"
    assert metadata["compatibility"]["optimizer"]["policy"]["muon"]["distributed_policy"] == "replicated_or_auto_dion2_when_sharded"
    assert metadata["compatibility"]["optimizer"]["policy"]["dion2"]["fraction"] == 0.25
    assert metadata["compatibility"]["optimizer"]["policy"]["distributed_policy"]["zero2_fsdp"] == "supported"
    assert metadata["compatibility"]["optimizer"]["policy"]["auto_routing"]["active"] is False
    assert metadata["compatibility"]["parallelism"]["mode"] == "ddp"
    assert metadata["mutable_controls"] == {
        "target_tokens": 128,
        "log_every_steps": 1,
        "checkpoint_every_steps": 10,
    }


@pytest.mark.parametrize("mode", ["fsdp", "zero2"])
def test_resume_compat_marks_sharded_muon_auto_dion2(tmp_path: Path, prepared_dataset_factory, mode: str) -> None:
    manifest = _manifest(prepared_dataset_factory, f"{mode}-dion2")
    spec = _runtime_spec(
        tmp_path,
        manifest,
        optimizer_name="muon",
        axis_names=("data", "fsdp"),
        axis_sizes=(1, 4),
        parallelism_mode=mode,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        n_kv_heads=4,
    )
    compat = build_resume_compat(spec)

    assert compat.payload["optimizer"]["policy"]["auto_routing"] == {
        "active": True,
        "muon_sharded_matrix_backend": "dion2",
    }
    assert compat.payload["optimizer"]["policy"]["dion2"]["orthogonalizer"] == "polar_express"


def test_resume_metadata_rejects_malformed_version(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = _manifest(prepared_dataset_factory, "bad-version")
    spec = _runtime_spec(tmp_path, manifest)
    metadata = checkpoint_metadata(spec, {"step": 1, "tokens_seen": 128, "loss": 1.0}, reason="interval")
    metadata["schema_version"] = 0

    with pytest.raises(ContractError, match="schema_version"):
        validate_resume_metadata(metadata, spec)


def test_resume_metadata_names_mismatched_field(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = _manifest(prepared_dataset_factory, "mismatch")
    checkpoint_spec = _runtime_spec(tmp_path, manifest)
    current_spec = _runtime_spec(tmp_path, manifest, hidden_size=16)
    metadata = checkpoint_metadata(checkpoint_spec, {"step": 1, "tokens_seen": 128, "loss": 1.0}, reason="interval")

    with pytest.raises(ContractError, match=r"compatibility\.model\.hidden_size"):
        validate_resume_metadata(metadata, current_spec)


def test_resume_restore_rejects_counter_mismatch(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = _manifest(prepared_dataset_factory, "counter-mismatch")
    spec = _runtime_spec(tmp_path, manifest)
    metadata = checkpoint_metadata(spec, {"step": 3, "tokens_seen": 24, "loss": 1.0}, reason="interval")
    dataset_state = _dataset_state(token_offset=24, next_record_index=6)
    restored = SimpleNamespace(
        metadata=metadata,
        step=3,
        train_state=SimpleNamespace(step=3, tokens_seen=16),
        dataset_state=dataset_state,
        host_state=HostState(dataset=dataset_state, last_checkpoint_step=3, wallclock_start_ns=123, run_id="smoke"),
    )

    with pytest.raises(ContractError, match="tokens_seen mismatch"):
        validate_resume_compat(restored, spec)


def test_auto_cosine_total_steps_changes_fingerprint_when_target_changes(
    tmp_path: Path,
    prepared_dataset_factory,
) -> None:
    manifest = _manifest(prepared_dataset_factory, "cosine-auto")
    first = _runtime_spec(tmp_path, manifest, schedule_name="cosine", total_steps=None, target_tokens=128)
    second = _runtime_spec(tmp_path, manifest, schedule_name="cosine", total_steps=None, target_tokens=256)

    assert first.optimizer.schedule.total_steps == 16
    assert second.optimizer.schedule.total_steps == 32
    assert build_resume_compat(first).runtime_fingerprint != build_resume_compat(second).runtime_fingerprint


def test_explicit_cosine_total_steps_allows_target_change(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = _manifest(prepared_dataset_factory, "cosine-explicit")
    first = _runtime_spec(tmp_path, manifest, schedule_name="cosine", total_steps=32, target_tokens=128)
    second = _runtime_spec(tmp_path, manifest, schedule_name="cosine", total_steps=32, target_tokens=256)

    assert build_resume_compat(first).runtime_fingerprint == build_resume_compat(second).runtime_fingerprint


def test_hf_streaming_data_identity_is_in_resume_fingerprint(tmp_path: Path, prepared_dataset_factory) -> None:
    prepared = _runtime_spec(tmp_path, _manifest(prepared_dataset_factory, "prepared-vs-streaming"))
    first = _streaming_runtime_spec(tmp_path, revision="abc123")
    second = _streaming_runtime_spec(tmp_path, revision="def456")
    first_payload = build_resume_compat(first).payload

    assert first_payload["data"]["mode"] == "hf_streaming"
    assert first_payload["data"]["train_manifest"] is None
    assert first_payload["data"]["training_pipeline"]["backend"] == "hf_streaming"
    assert first_payload["data"]["training_pipeline"]["source"]["revision"] == "abc123"
    assert build_resume_compat(first).runtime_fingerprint != build_resume_compat(second).runtime_fingerprint
    assert build_resume_compat(first).runtime_fingerprint != build_resume_compat(prepared).runtime_fingerprint


def _runtime_spec(tmp_path: Path, train_manifest: Path, **kwargs):
    config_path = tmp_path / f"resume-{len(list(tmp_path.glob('resume-*.toml')))}.toml"
    config_path.write_text(_config_text(train_manifest, **kwargs))
    return _with_runtime_schedule_steps(load_config(config_path))


def _streaming_runtime_spec(tmp_path: Path, *, revision: str):
    config_path = tmp_path / f"streaming-resume-{len(list(tmp_path.glob('streaming-resume-*.toml')))}.toml"
    config_path.write_text(_streaming_config_text(revision=revision))
    return _with_runtime_schedule_steps(load_config(config_path))


def _manifest(prepared_dataset_factory, name: str) -> Path:
    return prepared_dataset_factory(name, shard_token_groups=(tuple(range(0, 30)),), train_tokens=25)


def _config_text(
    train_manifest: Path,
    *,
    seed: int = 11,
    hidden_size: int = 8,
    intermediate_size: int = 16,
    num_layers: int = 1,
    num_heads: int = 2,
    n_kv_heads: int = 1,
    remat: str = "none",
    weight_decay: float = 0.0,
    optimizer_name: str = "adamw",
    schedule_name: str = "constant",
    total_steps: int | None = None,
    tokenizer_id: str = "toy-tokenizer",
    precision: str = "bf16",
    seq_len: int = 4,
    global_batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    z_loss_weight: float = 0.0,
    target_tokens: int = 128,
    log_every_steps: int = 1,
    checkpoint_every_steps: int = 10,
    axis_names: tuple[str, ...] = ("data",),
    axis_sizes: tuple[int, ...] = (1,),
    parallelism_mode: str = "ddp",
    data_order: str = "sequential",
    shuffle_seed: int | None = None,
    worker_count: int = 0,
    worker_buffer_size: int = 1,
    prefetch: bool = False,
    document_buffer_size: int | None = None,
    document_refill_size: int | None = None,
    trinity_moe_balance_name: str | None = None,
    trinity_moe_num_experts: int = 3,
    sequence_aux_loss_weight: float = 1e-4,
    expert_parallel: bool = False,
    expert_parallel_axis: str = "auto",
) -> str:
    total_steps_line = "" if total_steps is None else f"total_steps = {total_steps}\n"
    shuffle_seed_line = "" if shuffle_seed is None else f"shuffle_seed = {shuffle_seed}\n"
    document_buffer_size_line = "" if document_buffer_size is None else f"document_buffer_size = {document_buffer_size}\n"
    document_refill_size_line = "" if document_refill_size is None else f"document_refill_size = {document_refill_size}\n"
    model_name = "decoder"
    trinity_block = ""
    if trinity_moe_balance_name is not None or expert_parallel:
        balance_name = "none" if trinity_moe_balance_name is None else trinity_moe_balance_name
        model_name = "trinity"
        num_layers = 2
        trinity_block = f"""
[model.trinity]
initial_dense_layers = 1
local_window = 4
local_layers_per_global = 1

[model.trinity.moe]
num_experts = {trinity_moe_num_experts}
top_k = 2

[model.trinity.moe.balance]
name = "{balance_name}"
sequence_aux_loss_weight = {sequence_aux_loss_weight}
"""
    return f"""
[run]
id = "smoke"
seed = {seed}
output_dir = "runs"

[model]
name = "{model_name}"
variant = "tiny"
vocab_size = 64
hidden_size = {hidden_size}
intermediate_size = {intermediate_size}
num_layers = {num_layers}
num_heads = {num_heads}
n_kv_heads = {n_kv_heads}
max_seq_len = 4
compute_dtype = "float32"
remat = "{remat}"
{trinity_block}

[optimizer]
name = "{optimizer_name}"
weight_decay = {weight_decay}

[optimizer.schedule]
name = "{schedule_name}"
peak_lr = 0.001
{total_steps_line}
[data]
train_manifest = "{train_manifest.as_posix()}"
tokenizer_id = "{tokenizer_id}"
order = "{data_order}"
{shuffle_seed_line}worker_count = {worker_count}
worker_buffer_size = {worker_buffer_size}
prefetch = {str(prefetch).lower()}
{document_buffer_size_line}{document_refill_size_line}

[training]
seq_len = {seq_len}
global_batch_size = {global_batch_size}
gradient_accumulation_steps = {gradient_accumulation_steps}
target_tokens = {target_tokens}
precision = "{precision}"
log_every_steps = {log_every_steps}
checkpoint_every_steps = {checkpoint_every_steps}

[training.loss]
z_loss_weight = {z_loss_weight}

[mesh]
axis_names = [{", ".join(f'"{name}"' for name in axis_names)}]
axis_sizes = [{", ".join(str(size) for size in axis_sizes)}]

[parallelism]
mode = "{parallelism_mode}"
expert_parallel = {str(expert_parallel).lower()}
expert_parallel_axis = "{expert_parallel_axis}"
"""


def _streaming_config_text(*, revision: str) -> str:
    return f"""
[run]
id = "smoke"
seed = 11
output_dir = "runs"

[model]
name = "decoder"
variant = "tiny"
vocab_size = 50257
hidden_size = 8
intermediate_size = 16
num_layers = 1
num_heads = 2
n_kv_heads = 1
max_seq_len = 4
compute_dtype = "float32"

[optimizer]
name = "adamw"
weight_decay = 0.0

[optimizer.schedule]
name = "constant"
peak_lr = 0.001

[data]
mode = "hf_streaming"
tokenizer_id = "gpt2"
order = "sequential"

[data.hf_streaming]
dataset = "mock/dataset"
name = "mock-config"
split = "train"
revision = "{revision}"
text_column = "text"
append_eot = true

[training]
seq_len = 4
global_batch_size = 2
target_tokens = 128
precision = "bf16"
log_every_steps = 1
checkpoint_every_steps = 10

[training.loss]
z_loss_weight = 0.0

[mesh]
axis_names = ["data"]
axis_sizes = [1]

[parallelism]
mode = "ddp"
"""


def _dataset_state(*, token_offset: int, next_record_index: int) -> DataPipelineState:
    return DataPipelineState(
        schema_version=2,
        backend="grain",
        backend_version="0.2.16",
        split="train",
        order="sequential",
        shuffle_seed=None,
        worker_count=0,
        worker_buffer_size=1,
        prefetch=False,
        manifest_path="data/train/manifest.json",
        manifest_sha256="hash",
        tokenizer_id="toy-tokenizer",
        seq_len=4,
        batch_size=2,
        num_records=100,
        next_record_index=next_record_index,
        token_offset=token_offset,
        epoch=0,
        sampler_summary="sampler",
        source_summary="source",
        grain_state={"version": 2, "last_seen_indices": {"0": next_record_index - 1}},
    )
