import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jaxtitan.errors import ContractError
from jaxtitan.models import apply_model, build_model, count_parameters, dtype_from_name
from jaxtitan.models.components import DecoderBlock, DecoderSwiGLU, GroupedQueryAttention
from jaxtitan.specs.model import ModelSpec


def test_model_spec_validates_decoder_runtime_fields() -> None:
    with pytest.raises(ContractError, match="intermediate_size"):
        _tiny_spec(intermediate_size=0)
    with pytest.raises(ContractError, match="tied_embeddings"):
        _tiny_spec(tied_embeddings=True)
    with pytest.raises(ContractError, match="param_dtype"):
        _tiny_spec(param_dtype="fp32")
    with pytest.raises(ContractError, match="compute_dtype"):
        _tiny_spec(compute_dtype="bf16")
    with pytest.raises(ContractError, match="remat"):
        _tiny_spec(remat="layer")


def test_dtype_from_name_resolves_supported_dtypes() -> None:
    assert dtype_from_name("float32") == jnp.float32
    assert dtype_from_name("bfloat16") == jnp.bfloat16
    with pytest.raises(ContractError, match="unsupported dtype"):
        dtype_from_name("fp16")


def test_build_model_is_seed_deterministic() -> None:
    spec = _tiny_spec()

    first = build_model(spec, seed=123)
    second = build_model(spec, seed=123)
    third = build_model(spec, seed=124)

    first_leaves = jax.tree.leaves(nnx.to_pure_dict(first.state))
    second_leaves = jax.tree.leaves(nnx.to_pure_dict(second.state))
    third_leaves = jax.tree.leaves(nnx.to_pure_dict(third.state))
    assert all(jnp.array_equal(left, right) for left, right in zip(first_leaves, second_leaves, strict=True))
    assert any(not jnp.array_equal(left, right) for left, right in zip(first_leaves, third_leaves, strict=True))


def test_apply_model_returns_fixed_shape_logits() -> None:
    result = build_model(_tiny_spec(vocab_size=48, compute_dtype="bfloat16"), seed=0)
    input_ids = jnp.arange(16, dtype=jnp.int32).reshape(2, 8)

    logits = apply_model(result.graph, result.state, input_ids)

    assert logits.shape == (2, 8, 48)
    assert logits.dtype == jnp.bfloat16


def test_apply_model_rejects_sequences_longer_than_model_limit() -> None:
    result = build_model(_tiny_spec(max_seq_len=4), seed=0)
    input_ids = jnp.arange(5, dtype=jnp.int32).reshape(1, 5)

    with pytest.raises(ContractError, match="max_seq_len"):
        apply_model(result.graph, result.state, input_ids)


def test_apply_model_rejects_sequences_longer_than_model_limit_with_remat() -> None:
    result = build_model(_tiny_spec(max_seq_len=4, remat="block"), seed=0)
    input_ids = jnp.arange(5, dtype=jnp.int32).reshape(1, 5)

    with pytest.raises(ContractError, match="max_seq_len"):
        apply_model(result.graph, result.state, input_ids)


def test_block_remat_matches_plain_forward_and_metadata() -> None:
    plain = build_model(_tiny_spec(num_layers=2, compute_dtype="float32", remat="none"), seed=0)
    remat = build_model(_tiny_spec(num_layers=2, compute_dtype="float32", remat="block"), seed=0)
    input_ids = jnp.arange(16, dtype=jnp.int32).reshape(2, 8)

    plain_logits = apply_model(plain.graph, plain.state, input_ids)
    remat_logits = apply_model(remat.graph, remat.state, input_ids)

    assert remat.metadata == plain.metadata
    assert jnp.allclose(remat_logits, plain_logits, rtol=1e-6, atol=1e-6)


def test_decoder_recipe_assembles_reusable_components() -> None:
    result = build_model(_tiny_spec(), seed=0)
    model = nnx.merge(result.graph, result.state)
    layer = model.layers[0]

    assert isinstance(layer, DecoderBlock)
    assert isinstance(layer.attn, GroupedQueryAttention)
    assert isinstance(layer.mlp, DecoderSwiGLU)


def test_model_parameter_dtype_follows_spec() -> None:
    result = build_model(_tiny_spec(param_dtype="bfloat16", compute_dtype="float32"), seed=0)

    for leaf in jax.tree.leaves(nnx.to_pure_dict(result.state)):
        assert leaf.dtype == jnp.bfloat16
    logits = apply_model(result.graph, result.state, jnp.arange(8, dtype=jnp.int32).reshape(1, 8))
    assert logits.dtype == jnp.float32


def test_metadata_covers_every_parameter_leaf_once() -> None:
    result = build_model(_tiny_spec(num_layers=2), seed=0)
    flat_state = nnx.to_flat_state(result.state)

    assert len(result.metadata) == len(flat_state)
    assert len(result.param_layouts) == len(flat_state)
    for metadata, (path, variable) in zip(result.metadata, flat_state, strict=True):
        value = variable.get_value()
        assert metadata.path == tuple(str(part) for part in path)
        assert metadata.shape == tuple(value.shape)
        assert metadata.dtype == str(value.dtype)
        assert metadata.count == value.size
    assert {layout.path for layout in result.param_layouts} == {item.path for item in result.metadata}
    assert count_parameters(result.metadata) == sum(leaf.size for leaf in jax.tree.leaves(nnx.to_pure_dict(result.state)))


def test_decoder_param_layouts_record_fsdp_policy() -> None:
    result = build_model(_tiny_spec(), seed=0)
    layouts = {item.tag: item for item in result.param_layouts}

    for tag in ("embedding", "attention_q_norm", "attention_k_norm", "block_pre_norm", "block_post_norm", "final_norm"):
        assert layouts[tag].fsdp_axis is None
    for tag in ("attention_q", "attention_k", "attention_v", "mlp_gate", "mlp_up"):
        assert layouts[tag].fsdp_axis == 1
    for tag in ("attention_o", "mlp_down", "lm_head"):
        assert layouts[tag].fsdp_axis == 0

    for layout in result.param_layouts:
        assert len(layout.shape) == len(layout.logical_axes)
        if layout.fsdp_axis is not None:
            assert 0 <= layout.fsdp_axis < len(layout.shape)


def test_metadata_has_optimizer_routing_tags() -> None:
    result = build_model(_tiny_spec(), seed=0)

    assert {item.tag for item in result.metadata} == {
        "embedding",
        "attention_q",
        "attention_k",
        "attention_v",
        "attention_o",
        "attention_q_norm",
        "attention_k_norm",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
        "block_pre_norm",
        "block_post_norm",
        "final_norm",
        "lm_head",
    }


def test_build_model_rejects_unknown_model_name() -> None:
    with pytest.raises(ContractError, match="unsupported model.name"):
        build_model(_tiny_spec(name="encoder"), seed=0)


def _tiny_spec(**overrides) -> ModelSpec:
    values = {
        "name": "decoder",
        "variant": "tiny",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_layers": 1,
        "num_heads": 4,
        "n_kv_heads": 2,
        "max_seq_len": 8,
        "param_dtype": "float32",
        "compute_dtype": "bfloat16",
    }
    values.update(overrides)
    return ModelSpec(**values)
