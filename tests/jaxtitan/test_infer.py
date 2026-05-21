import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from jaxtitan.batch import Batch, DecodeBatch, PrefillBatch
from jaxtitan.errors import ContractError
from jaxtitan.infer import (
    InferenceState,
    apply_inference_model,
    decode_one,
    generate_tokens,
    init_kv_cache,
    inference_from_train_state,
    initialize_inference_state,
    prefill,
    restore_inference_checkpoint,
    sample_next_token,
)
from jaxtitan.models import apply_model, build_model
from jaxtitan.optim import build_optimizer
from jaxtitan.runtime import run_training
from jaxtitan.specs.generation import GenerationSpec
from jaxtitan.specs.model import ModelSpec, TrinitySpec
from jaxtitan.specs.optimizer import OptimizerSpec, ScheduleSpec
from jaxtitan.state import RngState
from jaxtitan.steps import initialize_train_state, make_train_step


def test_inference_state_is_pytree_with_model_and_rng_leaves() -> None:
    built = build_model(_tiny_spec(), seed=0)
    state = initialize_inference_state(built.state, seed=123)

    leaves = jax.tree.leaves(state)
    model_leaves = jax.tree.leaves(state.model)
    rng_leaves = jax.tree.leaves(state.rng)

    assert isinstance(state, InferenceState)
    assert len(leaves) == len(model_leaves) + len(rng_leaves)
    assert len(model_leaves) > 0
    assert len(rng_leaves) == 4


def test_initialize_inference_state_requires_exactly_one_rng_source() -> None:
    built = build_model(_tiny_spec(), seed=0)
    rng = _rng(7)

    with pytest.raises(ContractError, match="exactly one"):
        initialize_inference_state(built.state)
    with pytest.raises(ContractError, match="exactly one"):
        initialize_inference_state(built.state, seed=1, rng=rng)

    seeded = initialize_inference_state(built.state, seed=1)
    explicit = initialize_inference_state(built.state, rng=rng)
    assert len(jax.tree.leaves(seeded.rng)) == 4
    assert _trees_equal(explicit.rng, rng)


def test_inference_from_train_state_drops_training_only_state_and_handles_rng_override() -> None:
    built = build_model(_tiny_spec(), seed=0)
    optimizer = build_optimizer(
        OptimizerSpec(name="adamw", schedule=ScheduleSpec(peak_lr=1e-3), weight_decay=0.0),
        built.state,
        built.metadata,
    )
    train_state = initialize_train_state(built.state, optimizer.transform, seed=1)
    next_train_state, _ = make_train_step(built.graph, optimizer)(train_state, _batch())
    override_rng = _rng(99)

    preserved = inference_from_train_state(next_train_state)
    overridden = inference_from_train_state(next_train_state, rng=override_rng)

    assert _trees_equal(preserved.model, next_train_state.model)
    assert _trees_equal(preserved.rng, next_train_state.rng)
    assert _trees_equal(overridden.rng, override_rng)
    assert not hasattr(preserved, "opt_state")
    assert not hasattr(preserved, "schedule_state")


def test_apply_inference_model_matches_model_apply() -> None:
    built = build_model(_tiny_spec(), seed=0)
    state = initialize_inference_state(built.state, seed=3)
    input_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)

    expected = apply_model(built.graph, built.state, input_ids)
    actual = apply_inference_model(built.graph, state, input_ids)

    assert jnp.allclose(actual, expected)


def test_init_kv_cache_creates_expected_shapes_and_zero_lengths() -> None:
    spec = _tiny_spec(max_seq_len=8, compute_dtype="float32")

    cache = init_kv_cache(spec, batch_size=3, max_cache_len=6)

    assert cache.keys.shape == (1, 3, 6, 1, 4)
    assert cache.values.shape == (1, 3, 6, 1, 4)
    assert cache.keys.dtype == jnp.float32
    assert cache.values.dtype == jnp.float32
    assert jnp.array_equal(cache.lengths, jnp.zeros((3,), dtype=jnp.int32))
    with pytest.raises(ContractError, match="max_seq_len"):
        init_kv_cache(spec, batch_size=1, max_cache_len=9)


def test_prefill_updates_cache_and_matches_full_forward_logits() -> None:
    spec = _tiny_spec(max_seq_len=8, compute_dtype="float32")
    built = build_model(spec, seed=0)
    state = initialize_inference_state(built.state, seed=1)
    prompt = jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(3, dtype=jnp.int32)[None, :], prompt.shape)
    cache = init_kv_cache(spec, batch_size=2, max_cache_len=5)

    output = prefill(
        built.graph,
        state,
        PrefillBatch(input_ids=prompt, positions=positions, attention_mask=jnp.ones_like(prompt, dtype=jnp.bool_)),
        cache,
    )

    expected = apply_model(built.graph, built.state, prompt)
    assert output.logits.shape == expected.shape
    assert jnp.allclose(output.logits, expected, atol=1e-5)
    assert jnp.allclose(output.next_token_logits, expected[:, -1, :], atol=1e-5)
    assert jnp.array_equal(output.cache.lengths, jnp.asarray([3, 3], dtype=jnp.int32))
    assert jnp.any(output.cache.keys[:, :, :3] != 0)
    assert jnp.all(output.cache.keys[:, :, 3:] == 0)


def test_decode_updates_single_position_and_matches_full_forward_logits() -> None:
    spec = _tiny_spec(max_seq_len=8, compute_dtype="float32")
    built = build_model(spec, seed=0)
    state = initialize_inference_state(built.state, seed=1)
    prompt = jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(3, dtype=jnp.int32)[None, :], prompt.shape)
    cache = init_kv_cache(spec, batch_size=2, max_cache_len=5)
    prefill_output = prefill(
        built.graph,
        state,
        PrefillBatch(input_ids=prompt, positions=positions, attention_mask=jnp.ones_like(prompt, dtype=jnp.bool_)),
        cache,
    )
    token_ids = jnp.asarray([7, 8], dtype=jnp.int32)
    decode_positions = jnp.asarray([3, 3], dtype=jnp.int32)
    attention_mask = jnp.broadcast_to((jnp.arange(5) <= 3)[None, :], (2, 5))

    decoded = decode_one(
        built.graph,
        state,
        DecodeBatch(token_ids=token_ids, positions=decode_positions, attention_mask=attention_mask),
        prefill_output.cache,
    )

    full = apply_model(built.graph, built.state, jnp.concatenate([prompt, token_ids[:, None]], axis=1))
    assert decoded.logits.shape == (2, spec.vocab_size)
    assert jnp.allclose(decoded.logits, full[:, -1, :], atol=1e-5)
    assert jnp.array_equal(decoded.cache.lengths, jnp.asarray([4, 4], dtype=jnp.int32))
    assert jnp.any(decoded.cache.keys[:, :, 3] != 0)
    assert jnp.all(decoded.cache.keys[:, :, 4] == 0)


def test_dense_trinity_prefill_decode_matches_full_forward_logits() -> None:
    spec = _tiny_trinity_spec(max_seq_len=8, compute_dtype="float32")
    built = build_model(spec, seed=0)
    state = initialize_inference_state(built.state, seed=1)
    prompt = jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(3, dtype=jnp.int32)[None, :], prompt.shape)
    cache = init_kv_cache(spec, batch_size=2, max_cache_len=5)
    prefill_output = prefill(
        built.graph,
        state,
        PrefillBatch(input_ids=prompt, positions=positions, attention_mask=jnp.ones_like(prompt, dtype=jnp.bool_)),
        cache,
    )
    token_ids = jnp.asarray([7, 8], dtype=jnp.int32)
    decode_positions = jnp.asarray([3, 3], dtype=jnp.int32)
    attention_mask = jnp.broadcast_to((jnp.arange(5) <= 3)[None, :], (2, 5))

    decoded = decode_one(
        built.graph,
        state,
        DecodeBatch(token_ids=token_ids, positions=decode_positions, attention_mask=attention_mask),
        prefill_output.cache,
    )

    full = apply_model(built.graph, built.state, jnp.concatenate([prompt, token_ids[:, None]], axis=1))
    assert jnp.allclose(prefill_output.logits, apply_model(built.graph, built.state, prompt), atol=1e-5)
    assert jnp.allclose(decoded.logits, full[:, -1, :], atol=1e-5)


def test_prefill_and_decode_reject_invalid_shapes_and_positions() -> None:
    spec = _tiny_spec(max_seq_len=8, compute_dtype="float32")
    built = build_model(spec, seed=0)
    state = initialize_inference_state(built.state, seed=1)
    cache = init_kv_cache(spec, batch_size=2, max_cache_len=4)
    prompt = jnp.asarray([[1, 2], [3, 4]], dtype=jnp.int32)

    with pytest.raises(ContractError, match="rank 2"):
        prefill(
            built.graph,
            state,
            PrefillBatch(
                input_ids=jnp.asarray([1, 2], dtype=jnp.int32),
                positions=jnp.asarray([0, 1], dtype=jnp.int32),
                attention_mask=jnp.asarray([True, True]),
            ),
            cache,
        )
    with pytest.raises(ContractError, match="outside max_cache_len"):
        prefill(
            built.graph,
            state,
            PrefillBatch(
                input_ids=prompt,
                positions=jnp.asarray([[0, 4], [0, 1]], dtype=jnp.int32),
                attention_mask=jnp.ones_like(prompt, dtype=jnp.bool_),
            ),
            cache,
        )
    with pytest.raises(ContractError, match="rank 1"):
        decode_one(
            built.graph,
            state,
            DecodeBatch(
                token_ids=jnp.asarray([[1], [2]], dtype=jnp.int32),
                positions=jnp.asarray([[0], [0]], dtype=jnp.int32),
                attention_mask=jnp.ones((2, 4), dtype=jnp.bool_),
            ),
            cache,
        )


def test_top_k_one_sampling_matches_argmax_and_rejects_top_p() -> None:
    built = build_model(_tiny_spec(), seed=0)
    state = initialize_inference_state(built.state, seed=5)
    logits = jnp.asarray([[0.1, 0.5, 0.2], [2.0, -1.0, 0.0]], dtype=jnp.float32)

    sample = sample_next_token(logits, state, GenerationSpec(max_new_tokens=1, top_k=1))

    assert jnp.array_equal(sample.token_ids, jnp.asarray([1, 0], dtype=jnp.int32))
    assert sample.logprobs.shape == (2,)
    assert not jnp.array_equal(sample.state.rng.sample, state.rng.sample)
    with pytest.raises(ContractError, match="top_p"):
        sample_next_token(logits, state, GenerationSpec(max_new_tokens=1, top_p=0.9))


def test_generate_tokens_is_deterministic_for_same_seed_and_updates_cache_lengths() -> None:
    spec = _tiny_spec(max_seq_len=8, compute_dtype="float32")
    built = build_model(spec, seed=0)
    left = initialize_inference_state(built.state, seed=11)
    right = initialize_inference_state(built.state, seed=11)
    prompt = jnp.asarray([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    generation = GenerationSpec(max_new_tokens=2, temperature=1.0, top_k=3)

    first = generate_tokens(built.graph, left, spec, prompt, generation)
    second = generate_tokens(built.graph, right, spec, prompt, generation)

    assert first.generated_ids.shape == (2, 2)
    assert first.full_ids.shape == (2, 5)
    assert first.logprobs.shape == (2, 2)
    assert jnp.array_equal(first.generated_ids, second.generated_ids)
    assert jnp.allclose(first.logprobs, second.logprobs)
    assert jnp.array_equal(first.cache.lengths, jnp.asarray([5, 5], dtype=jnp.int32))
    assert not jnp.array_equal(first.state.rng.sample, left.rng.sample)


def test_generate_tokens_rejects_prompt_overflow_and_top_p() -> None:
    spec = _tiny_spec(max_seq_len=4, compute_dtype="float32")
    built = build_model(spec, seed=0)
    state = initialize_inference_state(built.state, seed=11)
    prompt = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)

    with pytest.raises(ContractError, match="max_seq_len"):
        generate_tokens(built.graph, state, spec, prompt, GenerationSpec(max_new_tokens=2, top_k=1))
    with pytest.raises(ContractError, match="top_p"):
        generate_tokens(built.graph, state, spec, prompt, GenerationSpec(max_new_tokens=1, top_p=0.9))


def test_restore_inference_checkpoint_latest_best_and_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("infer-restore", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=16, checkpoint_every_steps=1))
    run_training(config_path)
    run_dir = tmp_path / "runs" / "loop"
    index = json.loads((run_dir / "checkpoints" / "index.json").read_text())

    latest = restore_inference_checkpoint(run_dir, "latest")
    best = restore_inference_checkpoint(run_dir, "best")
    explicit = restore_inference_checkpoint(run_dir, "000002")

    assert isinstance(latest.state, InferenceState)
    assert latest.metadata.run_id == "loop"
    assert latest.metadata.checkpoint_step == 2
    assert latest.metadata.checkpoint_path == Path("checkpoints/000002")
    assert latest.metadata.tokens_seen == 16
    assert latest.metadata.runtime_fingerprint
    assert latest.metadata.model_spec == latest.run_spec.model
    assert latest.metadata.mesh_spec == latest.run_spec.mesh
    assert not hasattr(latest.state, "opt_state")
    assert not hasattr(latest, "dataset_state")
    assert best.metadata.checkpoint_step == index["best_eval_step"]
    assert explicit.metadata.checkpoint_step == 2


def test_restore_inference_checkpoint_rejects_missing_selectors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("infer-missing", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=8, checkpoint_every_steps=1))
    run_training(config_path)
    run_dir = tmp_path / "runs" / "loop"
    index_path = run_dir / "checkpoints" / "index.json"
    index = json.loads(index_path.read_text())
    for record in index["records"]:
        record["eval_loss"] = None
    index_path.write_text(json.dumps(index, sort_keys=True))

    with pytest.raises(ContractError, match="best validation checkpoint"):
        restore_inference_checkpoint(run_dir, "best")
    with pytest.raises(ContractError, match="checkpoint selector"):
        restore_inference_checkpoint(run_dir, "middle")
    with pytest.raises(ContractError, match="not retained"):
        restore_inference_checkpoint(run_dir, "999")


def test_restore_inference_checkpoint_rejects_compatibility_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prepared_dataset_factory,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = prepared_dataset_factory("infer-compat", shard_token_groups=(tuple(range(0, 50)),), train_tokens=25)
    config_path = tmp_path / "jaxtitan.toml"
    config_path.write_text(_training_config(manifest, target_tokens=8, checkpoint_every_steps=1))
    run_training(config_path)
    resolved_path = tmp_path / "runs" / "loop" / "config" / "resolved.json"
    resolved = json.loads(resolved_path.read_text())
    resolved["model"]["hidden_size"] = 16
    resolved_path.write_text(json.dumps(resolved, sort_keys=True))

    with pytest.raises(ContractError, match=r"compatibility\.model\.hidden_size"):
        restore_inference_checkpoint(tmp_path / "runs" / "loop", "latest")


def _rng(seed: int) -> RngState:
    train_key, data_key, eval_key, sample_key = jax.random.split(jax.random.key(seed), 4)
    return RngState(train=train_key, data=data_key, eval=eval_key, sample=sample_key)


def _tiny_spec(**overrides) -> ModelSpec:
    values = {
        "name": "decoder",
        "variant": "tiny",
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_layers": 1,
        "num_heads": 2,
        "n_kv_heads": 1,
        "max_seq_len": 4,
        "compute_dtype": "float32",
    }
    values.update(overrides)
    return ModelSpec(**values)


def _tiny_trinity_spec(**overrides) -> ModelSpec:
    trinity = TrinitySpec(
        initial_dense_layers=1,
        local_window=8,
        local_layers_per_global=1,
    )
    values = {
        "name": "trinity",
        "variant": "tiny",
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_layers": 2,
        "num_heads": 2,
        "n_kv_heads": 1,
        "max_seq_len": 8,
        "compute_dtype": "float32",
        "trinity": trinity,
    }
    values.update(overrides)
    return ModelSpec(**values)


def _batch() -> Batch:
    input_ids = jnp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
    target_ids = (input_ids + 1) % 16
    return Batch(input_ids=input_ids, target_ids=target_ids, loss_mask=jnp.ones((2, 4), dtype=jnp.bool_))


def _trees_equal(left, right) -> bool:
    return all(
        jnp.array_equal(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _training_config(train_manifest: Path, *, target_tokens: int, checkpoint_every_steps: int) -> str:
    return f"""
[run]
id = "loop"
seed = 7
output_dir = "runs"

[model]
name = "decoder"
variant = "tiny"
vocab_size = 64
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
train_manifest = "{train_manifest.as_posix()}"
tokenizer_id = "toy-tokenizer"

[training]
seq_len = 4
global_batch_size = 2
target_tokens = {target_tokens}
log_every_steps = 1
checkpoint_every_steps = {checkpoint_every_steps}

[mesh]
axis_names = ["data"]
axis_sizes = [1]

[[evals]]
name = "validation"
every_steps = 1
num_batches = 1
"""
