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
from jaxtitan.state import DatasetState, HostState


def test_resume_fingerprint_is_stable_for_identical_specs(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = prepared_dataset_factory("stable")
    first = _runtime_spec(tmp_path, manifest)
    second = _runtime_spec(tmp_path, manifest)

    assert build_resume_compat(first) == build_resume_compat(second)


def test_resume_fingerprint_ignores_safe_runtime_controls(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = prepared_dataset_factory("controls")
    first = _runtime_spec(tmp_path, manifest, target_tokens=128, log_every_steps=1, checkpoint_every_steps=10)
    second = _runtime_spec(tmp_path, manifest, target_tokens=256, log_every_steps=5, checkpoint_every_steps=20)

    assert build_resume_compat(first).runtime_fingerprint == build_resume_compat(second).runtime_fingerprint


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 16},
        {"remat": "block"},
        {"weight_decay": 0.2},
        {"axis_sizes": (2,)},
        {"seed": 12},
        {"precision": "fp32"},
        {"seq_len": 2},
        {"global_batch_size": 1},
        {"gradient_accumulation_steps": 2},
        {"tokenizer_id": "other-tokenizer"},
    ],
)
def test_resume_fingerprint_changes_for_unsafe_fields(
    tmp_path: Path,
    prepared_dataset_factory,
    kwargs: dict,
) -> None:
    manifest = prepared_dataset_factory("unsafe")
    base = _runtime_spec(tmp_path, manifest)
    changed = _runtime_spec(tmp_path, manifest, **kwargs)

    assert build_resume_compat(base).runtime_fingerprint != build_resume_compat(changed).runtime_fingerprint


def test_resume_fingerprint_changes_for_data_manifest_hash(tmp_path: Path, prepared_dataset_factory) -> None:
    first_manifest = prepared_dataset_factory("first")
    second_manifest = prepared_dataset_factory("second", shard_token_groups=(tuple(range(10, 18)),))
    first = _runtime_spec(tmp_path, first_manifest)
    second = _runtime_spec(tmp_path, second_manifest)

    assert build_resume_compat(first).runtime_fingerprint != build_resume_compat(second).runtime_fingerprint


def test_resume_metadata_contains_compatibility_payload(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = prepared_dataset_factory("metadata")
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
    assert metadata["mutable_controls"] == {
        "target_tokens": 128,
        "log_every_steps": 1,
        "checkpoint_every_steps": 10,
    }


def test_resume_metadata_rejects_malformed_version(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = prepared_dataset_factory("bad-version")
    spec = _runtime_spec(tmp_path, manifest)
    metadata = checkpoint_metadata(spec, {"step": 1, "tokens_seen": 128, "loss": 1.0}, reason="interval")
    metadata["schema_version"] = 0

    with pytest.raises(ContractError, match="schema_version"):
        validate_resume_metadata(metadata, spec)


def test_resume_metadata_names_mismatched_field(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = prepared_dataset_factory("mismatch")
    checkpoint_spec = _runtime_spec(tmp_path, manifest)
    current_spec = _runtime_spec(tmp_path, manifest, hidden_size=16)
    metadata = checkpoint_metadata(checkpoint_spec, {"step": 1, "tokens_seen": 128, "loss": 1.0}, reason="interval")

    with pytest.raises(ContractError, match=r"compatibility\.model\.hidden_size"):
        validate_resume_metadata(metadata, current_spec)


def test_resume_restore_rejects_counter_mismatch(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = prepared_dataset_factory("counter-mismatch")
    spec = _runtime_spec(tmp_path, manifest)
    metadata = checkpoint_metadata(spec, {"step": 3, "tokens_seen": 24, "loss": 1.0}, reason="interval")
    dataset_state = DatasetState(shard_index=0, token_offset=24, epoch=0, shuffle_state=None)
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
    manifest = prepared_dataset_factory("cosine-auto")
    first = _runtime_spec(tmp_path, manifest, schedule_name="cosine", total_steps=None, target_tokens=128)
    second = _runtime_spec(tmp_path, manifest, schedule_name="cosine", total_steps=None, target_tokens=256)

    assert first.optimizer.schedule.total_steps == 16
    assert second.optimizer.schedule.total_steps == 32
    assert build_resume_compat(first).runtime_fingerprint != build_resume_compat(second).runtime_fingerprint


def test_explicit_cosine_total_steps_allows_target_change(tmp_path: Path, prepared_dataset_factory) -> None:
    manifest = prepared_dataset_factory("cosine-explicit")
    first = _runtime_spec(tmp_path, manifest, schedule_name="cosine", total_steps=32, target_tokens=128)
    second = _runtime_spec(tmp_path, manifest, schedule_name="cosine", total_steps=32, target_tokens=256)

    assert build_resume_compat(first).runtime_fingerprint == build_resume_compat(second).runtime_fingerprint


def _runtime_spec(tmp_path: Path, train_manifest: Path, **kwargs):
    config_path = tmp_path / f"resume-{len(list(tmp_path.glob('resume-*.toml')))}.toml"
    config_path.write_text(_config_text(train_manifest, **kwargs))
    return _with_runtime_schedule_steps(load_config(config_path))


def _config_text(
    train_manifest: Path,
    *,
    seed: int = 11,
    hidden_size: int = 8,
    remat: str = "none",
    weight_decay: float = 0.0,
    schedule_name: str = "constant",
    total_steps: int | None = None,
    tokenizer_id: str = "toy-tokenizer",
    precision: str = "bf16",
    seq_len: int = 4,
    global_batch_size: int = 2,
    gradient_accumulation_steps: int = 1,
    target_tokens: int = 128,
    log_every_steps: int = 1,
    checkpoint_every_steps: int = 10,
    axis_sizes: tuple[int, ...] = (1,),
) -> str:
    total_steps_line = "" if total_steps is None else f"total_steps = {total_steps}\n"
    return f"""
[run]
id = "smoke"
seed = {seed}
output_dir = "runs"

[model]
name = "decoder"
variant = "tiny"
vocab_size = 64
hidden_size = {hidden_size}
intermediate_size = 16
num_layers = 1
num_heads = 2
n_kv_heads = 1
max_seq_len = 4
compute_dtype = "float32"
remat = "{remat}"

[optimizer]
name = "adamw"
weight_decay = {weight_decay}

[optimizer.schedule]
name = "{schedule_name}"
peak_lr = 0.001
{total_steps_line}
[data]
train_manifest = "{train_manifest.as_posix()}"
tokenizer_id = "{tokenizer_id}"

[training]
seq_len = {seq_len}
global_batch_size = {global_batch_size}
gradient_accumulation_steps = {gradient_accumulation_steps}
target_tokens = {target_tokens}
precision = "{precision}"
log_every_steps = {log_every_steps}
checkpoint_every_steps = {checkpoint_every_steps}

[mesh]
axis_names = ["data"]
axis_sizes = [{", ".join(str(size) for size in axis_sizes)}]
"""
