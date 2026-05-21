import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jaxtitan.errors import ContractError
from jaxtitan.models import ModelOutput, apply_model, apply_model_output, build_model, count_parameters, dtype_from_name
from jaxtitan.models.components import (
    DecoderBlock,
    DecoderSwiGLU,
    ExpertSwiGLU,
    GroupedQueryAttention,
    SigmoidTopKRouter,
    SparseMoE,
    TrinityDenseBlock,
    TrinityMoEBlock,
    full_sequence_attention_mask,
)
from jaxtitan.models.components.moe import _sequence_balance_loss
from jaxtitan.specs.model import ModelSpec, TrinityMoeSpec, TrinitySpec


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


def test_build_dense_trinity_is_seed_deterministic() -> None:
    spec = _tiny_trinity_spec()

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


def test_apply_model_output_wraps_logits_and_matches_apply_model() -> None:
    result = build_model(_tiny_spec(vocab_size=48, compute_dtype="bfloat16"), seed=0)
    input_ids = jnp.arange(16, dtype=jnp.int32).reshape(2, 8)

    output = apply_model_output(result.graph, result.state, input_ids)

    assert isinstance(output, ModelOutput)
    assert jnp.array_equal(output.logits, apply_model(result.graph, result.state, input_ids))
    assert output.aux_losses == ()
    assert output.aux_metrics == ()
    assert jax.tree.leaves(output)


def test_dense_trinity_apply_model_returns_fixed_shape_logits() -> None:
    result = build_model(_tiny_trinity_spec(vocab_size=48, compute_dtype="bfloat16"), seed=0)
    input_ids = jnp.arange(16, dtype=jnp.int32).reshape(2, 8)

    logits = apply_model(result.graph, result.state, input_ids)

    assert logits.shape == (2, 8, 48)
    assert logits.dtype == jnp.bfloat16


def test_trinity_moe_apply_model_returns_fixed_shape_logits() -> None:
    result = build_model(_tiny_trinity_spec(num_layers=2, initial_dense_layers=1, moe={"num_experts": 3, "top_k": 2}), seed=0)
    input_ids = jnp.arange(16, dtype=jnp.int32).reshape(2, 8)

    logits = apply_model(result.graph, result.state, input_ids)

    assert logits.shape == (2, 8, 32)
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


def test_dense_trinity_recipe_assembles_dense_blocks_and_layer_pattern() -> None:
    result = build_model(_tiny_trinity_spec(num_layers=4, local_layers_per_global=2), seed=0)
    model = nnx.merge(result.graph, result.state)

    assert model.layer_attention == ("local", "local", "global", "local")
    assert all(isinstance(layer, TrinityDenseBlock) for layer in model.layers)
    assert [layer.attn.position for layer in model.layers] == ["rope", "rope", "none", "rope"]
    assert [layer.attn.mask for layer in model.layers] == ["sliding_window", "sliding_window", "causal", "sliding_window"]


def test_trinity_moe_recipe_uses_dense_prefix_then_moe_layers() -> None:
    result = build_model(_tiny_trinity_spec(num_layers=3, initial_dense_layers=1, moe={"num_experts": 4, "top_k": 2}), seed=0)
    model = nnx.merge(result.graph, result.state)

    assert model.layer_kind == ("dense", "moe", "moe")
    assert isinstance(model.layers[0], TrinityDenseBlock)
    assert all(isinstance(layer, TrinityMoEBlock) for layer in model.layers[1:])
    assert all(isinstance(layer.mlp, SparseMoE) for layer in model.layers[1:])


def test_trinity_afmoe_dual_norm_policy_uses_unit_post_norm_scale() -> None:
    result = build_model(
        _tiny_trinity_spec(
            num_layers=2,
            initial_dense_layers=1,
            norm_policy="afmoe_dual",
            moe={"num_experts": 3, "top_k": 2, "num_shared_experts": 1},
        ),
        seed=0,
    )
    model = nnx.merge(result.graph, result.state)

    for layer in model.layers:
        assert jnp.allclose(layer.attn_post_norm.scale[...], jnp.ones((model.spec.hidden_size,), dtype=jnp.float32))
        assert jnp.allclose(layer.ffn_post_norm.scale[...], jnp.ones((model.spec.hidden_size,), dtype=jnp.float32))


def test_sigmoid_top_k_router_is_deterministic_and_normalizes_weights() -> None:
    router = SigmoidTopKRouter(
        hidden_size=2,
        num_experts=3,
        top_k=2,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    router.proj.kernel[...] = jnp.asarray([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    x = jnp.asarray([[[1.0, 0.0], [0.5, 0.0]]], dtype=jnp.float32)

    first = router(x)
    second = router(x)

    assert jnp.array_equal(first.expert_ids, second.expert_ids)
    assert jnp.array_equal(first.expert_ids[0, 0], jnp.asarray([2, 1], dtype=jnp.int32))
    assert jnp.allclose(jnp.sum(first.weights, axis=-1), jnp.ones((1, 2), dtype=jnp.float32))
    assert jnp.all(first.weights > 0)


def test_sigmoid_top_k_router_uses_bias_for_selection_and_unbiased_scores_for_weights() -> None:
    router = SigmoidTopKRouter(
        hidden_size=2,
        num_experts=3,
        top_k=2,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
        route_scale=1.5,
    )
    router.proj.kernel[...] = jnp.asarray([[2.0, 0.0, -2.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    x = jnp.asarray([[[1.0, 0.0]]], dtype=jnp.float32)
    bias = jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32)

    routed = router(x, expert_bias=bias)

    unbiased_scores = jax.nn.sigmoid(jnp.asarray([2.0, 0.0, -2.0], dtype=jnp.float32))
    expected_ids = jnp.asarray([[[2, 0]]], dtype=jnp.int32)
    expected_selected_scores = jnp.asarray([unbiased_scores[2], unbiased_scores[0]], dtype=jnp.float32)
    expected_weights = (expected_selected_scores / jnp.sum(expected_selected_scores)) * 1.5
    assert jnp.array_equal(routed.expert_ids, expected_ids)
    assert jnp.allclose(routed.weights[0, 0], expected_weights)
    assert jnp.allclose(jnp.sum(routed.weights, axis=-1), jnp.asarray([[1.5]], dtype=jnp.float32))


def test_sequence_balance_loss_matches_manual_calculation() -> None:
    expert_ids = jnp.asarray([[[0, 1], [1, 2]]], dtype=jnp.int32)
    scores = jnp.asarray([[[0.2, 0.5, 0.3], [0.4, 0.1, 0.5]]], dtype=jnp.float32)

    actual = _sequence_balance_loss(expert_ids, scores, top_k=2)

    selected_counts = jnp.asarray([[1.0, 2.0, 1.0]], dtype=jnp.float32)
    f_i = (3 / (2 * 2)) * selected_counts
    p_i = jnp.mean(scores / jnp.sum(scores, axis=-1, keepdims=True), axis=1)
    expected = jnp.sum(f_i * p_i)
    assert jnp.allclose(actual, expected)


def test_expert_swiglu_matches_manual_selected_expert_calculation() -> None:
    experts = ExpertSwiGLU(
        hidden_size=2,
        intermediate_size=2,
        num_experts=2,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    experts.gate[...] = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.0], [0.0, 0.5]],
        ],
        dtype=jnp.float32,
    )
    experts.up[...] = jnp.asarray(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ],
        dtype=jnp.float32,
    )
    experts.down[...] = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, -1.0]],
        ],
        dtype=jnp.float32,
    )
    x = jnp.asarray([[[1.0, 2.0]]], dtype=jnp.float32)
    expert_ids = jnp.asarray([[[1, 0]]], dtype=jnp.int32)
    weights = jnp.asarray([[[0.25, 0.75]]], dtype=jnp.float32)

    actual = experts(x, expert_ids, weights)

    selected_gate = experts.gate[...][expert_ids]
    selected_up = experts.up[...][expert_ids]
    selected_down = experts.down[...][expert_ids]
    gate = jnp.einsum("...h,...khi->...ki", x, selected_gate)
    up = jnp.einsum("...h,...khi->...ki", x, selected_up)
    hidden = jax.nn.silu(gate) * up
    outputs = jnp.einsum("...ki,...kih->...kh", hidden, selected_down)
    expected = jnp.sum(outputs * weights[..., None], axis=-2)
    assert jnp.allclose(actual, expected)


def test_sparse_moe_adds_shared_expert_output() -> None:
    spec = _tiny_trinity_spec(num_layers=2, initial_dense_layers=1, moe={"num_experts": 1, "top_k": 1, "num_shared_experts": 1})
    moe = SparseMoE(spec, spec.trinity.moe, rngs=nnx.Rngs(0))
    assert moe.shared_experts is not None
    moe.experts.gate[...] = jnp.zeros_like(moe.experts.gate[...])
    moe.experts.up[...] = jnp.zeros_like(moe.experts.up[...])
    moe.experts.down[...] = jnp.zeros_like(moe.experts.down[...])
    x = jnp.asarray([[[1.0] * spec.hidden_size]], dtype=jnp.float32)

    actual = moe(x)

    assert jnp.allclose(actual, moe.shared_experts(x))


def test_trinity_moe_output_emits_router_stats_and_aux_loss_when_balanced() -> None:
    result = build_model(
        _tiny_trinity_spec(
            num_layers=2,
            initial_dense_layers=1,
            moe={"num_experts": 3, "top_k": 2, "balance": {"name": "smebu"}},
        ),
        seed=0,
    )
    input_ids = jnp.arange(8, dtype=jnp.int32).reshape(2, 4)

    output = apply_model_output(result.graph, result.state, input_ids)

    assert isinstance(output, ModelOutput)
    assert output.logits.shape == (2, 4, 32)
    assert len(output.router_stats) == 1
    assert len(output.aux_losses) == 1
    stats = output.router_stats[0]
    assert stats.expert_counts.shape == (3,)
    assert stats.importance.shape == (3,)
    assert jnp.sum(stats.expert_counts) == 2 * 4 * 2
    assert float(jax.device_get(jnp.sum(stats.importance))) == pytest.approx(2 * 4, abs=1e-2)
    assert stats.max_vio.shape == ()
    assert output.aux_losses[0].name == "moe_sequence_balance"


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


def test_dense_trinity_metadata_covers_every_parameter_leaf_once() -> None:
    result = build_model(_tiny_trinity_spec(num_layers=2), seed=0)
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


def test_dense_trinity_metadata_has_expected_dense_tags() -> None:
    result = build_model(_tiny_trinity_spec(num_layers=1), seed=0)

    assert {item.tag for item in result.metadata} == {
        "embedding",
        "attention_q",
        "attention_k",
        "attention_v",
        "attention_o",
        "attention_gate",
        "attention_q_norm",
        "attention_k_norm",
        "attention_pre_norm",
        "attention_post_norm",
        "mlp_gate",
        "mlp_up",
        "mlp_down",
        "ffn_pre_norm",
        "ffn_post_norm",
        "final_norm",
        "lm_head",
    }


def test_trinity_moe_metadata_has_expected_sparse_tags() -> None:
    result = build_model(
        _tiny_trinity_spec(
            num_layers=2,
            initial_dense_layers=1,
            moe={"num_experts": 3, "top_k": 2, "num_shared_experts": 1},
        ),
        seed=0,
    )

    assert {
        "moe_router",
        "moe_expert_bias",
        "moe_gate",
        "moe_up",
        "moe_down",
        "moe_shared_gate",
        "moe_shared_up",
        "moe_shared_down",
    }.issubset({item.tag for item in result.metadata})
    expert_bias = [item for item in result.metadata if item.tag == "moe_expert_bias"]
    assert len(expert_bias) == 1
    assert expert_bias[0].shape == (3,)
    assert expert_bias[0].dtype == "float32"
    assert len(result.metadata) == len(nnx.to_flat_state(result.state))
    assert count_parameters(result.metadata) == sum(leaf.size for leaf in jax.tree.leaves(nnx.to_pure_dict(result.state)))


def test_dense_trinity_param_layouts_record_fsdp_policy() -> None:
    result = build_model(_tiny_trinity_spec(), seed=0)
    layouts = {item.tag: item for item in result.param_layouts}

    for tag in (
        "embedding",
        "attention_q_norm",
        "attention_k_norm",
        "attention_pre_norm",
        "attention_post_norm",
        "ffn_pre_norm",
        "ffn_post_norm",
        "final_norm",
    ):
        assert layouts[tag].fsdp_axis is None
    for tag in ("attention_q", "attention_k", "attention_v", "attention_gate", "mlp_gate", "mlp_up"):
        assert layouts[tag].fsdp_axis == 1
    for tag in ("attention_o", "mlp_down", "lm_head"):
        assert layouts[tag].fsdp_axis == 0


def test_trinity_moe_param_layouts_leave_sparse_experts_replicated() -> None:
    result = build_model(
        _tiny_trinity_spec(
            num_layers=2,
            initial_dense_layers=1,
            moe={"num_experts": 3, "top_k": 2, "num_shared_experts": 1},
        ),
        seed=0,
    )
    layouts = {item.tag: item for item in result.param_layouts}

    for tag in ("moe_router", "moe_expert_bias", "moe_gate", "moe_up", "moe_down"):
        assert layouts[tag].fsdp_axis is None
    assert layouts["moe_shared_gate"].fsdp_axis == 1
    assert layouts["moe_shared_up"].fsdp_axis == 1
    assert layouts["moe_shared_down"].fsdp_axis == 0


def test_sliding_window_mask_limits_attention_span() -> None:
    mask = full_sequence_attention_mask(5, mode="sliding_window", local_window=2)[0]
    expected = jnp.asarray(
        [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [False, True, True, False, False],
            [False, False, True, True, False],
            [False, False, False, True, True],
        ],
        dtype=jnp.bool_,
    )

    assert jnp.array_equal(mask, expected)


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


def _tiny_trinity_spec(**overrides) -> ModelSpec:
    trinity_values = {
        "initial_dense_layers": 1,
        "local_window": 8,
        "local_layers_per_global": 3,
        "norm_policy": "depth_scaled_sandwich",
        "moe": None,
    }
    for key in tuple(overrides):
        if key in trinity_values:
            trinity_values[key] = overrides.pop(key)
    values = {
        "name": "trinity",
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
        "trinity": TrinitySpec(**trinity_values),
    }
    values.update(overrides)
    return ModelSpec(**values)
