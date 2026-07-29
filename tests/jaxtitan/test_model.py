import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jaxtitan.errors import ContractError
from jaxtitan.models import ModelExecutionContext, ModelOutput, apply_model, apply_model_output, build_model, count_parameters, dtype_from_name
from jaxtitan.models.execution import ragged_dot_pallas_triton_available, sequence_parallel_activation
from jaxtitan.models.components import (
    AllToAllExpertDispatcher,
    DecoderBlock,
    DecoderSwiGLU,
    ExpertParallelDispatcher,
    ExpertSwiGLU,
    GroupedQueryAttention,
    LocalExpertDispatcher,
    RdepStaticExpertDispatcher,
    SigmoidTopKRouter,
    SparseMoE,
    TrinityDenseBlock,
    TrinityMoEBlock,
    full_sequence_attention_mask,
)
from jaxtitan.models.components.moe import _all_to_all_expert_swiglu, _sequence_balance_loss
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


def test_local_expert_dispatcher_matches_expert_swiglu_reference() -> None:
    experts = ExpertSwiGLU(
        hidden_size=2,
        intermediate_size=2,
        num_experts=2,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    experts.gate[...] = jnp.arange(8, dtype=jnp.float32).reshape(2, 2, 2) / 8.0
    experts.up[...] = jnp.arange(8, 16, dtype=jnp.float32).reshape(2, 2, 2) / 8.0
    experts.down[...] = jnp.arange(16, 24, dtype=jnp.float32).reshape(2, 2, 2) / 8.0
    dispatcher = LocalExpertDispatcher()
    x = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=jnp.float32)
    expert_ids = jnp.asarray([[[1, 0], [0, 1]]], dtype=jnp.int32)
    weights = jnp.asarray([[[0.25, 0.75], [0.6, 0.4]]], dtype=jnp.float32)

    dispatched = dispatcher(experts, x, expert_ids, weights)
    reference = experts(x, expert_ids, weights)

    assert jnp.allclose(dispatched, reference)


def test_expert_parallel_dispatcher_matches_local_dispatcher_on_ep_mesh() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape((1, 4)), ("data", "ep"))
    experts = ExpertSwiGLU(
        hidden_size=3,
        intermediate_size=5,
        num_experts=4,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    gate = jnp.arange(4 * 3 * 5, dtype=jnp.float32).reshape(4, 3, 5) / 37.0
    up = jnp.arange(4 * 3 * 5, dtype=jnp.float32).reshape(4, 3, 5)[::-1] / 41.0
    down = jnp.arange(4 * 5 * 3, dtype=jnp.float32).reshape(4, 5, 3) / 43.0
    experts.gate[...] = jax.device_put(gate, NamedSharding(mesh, P("ep", None, None)))
    experts.up[...] = jax.device_put(up, NamedSharding(mesh, P("ep", None, None)))
    experts.down[...] = jax.device_put(down, NamedSharding(mesh, P("ep", None, None)))
    x = jax.device_put(
        jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3) / 17.0,
        NamedSharding(mesh, P("data", None, None)),
    )
    expert_ids = jax.device_put(
        jnp.asarray(
            [
                [[0, 1], [2, 3], [1, 3], [0, 2]],
                [[3, 1], [0, 2], [2, 1], [3, 0]],
            ],
            dtype=jnp.int32,
        ),
        NamedSharding(mesh, P("data", None, None)),
    )
    weights = jax.device_put(
        jnp.asarray(
            [
                [[0.25, 0.75], [0.5, 0.5], [0.4, 0.6], [0.2, 0.8]],
                [[0.55, 0.45], [0.3, 0.7], [0.65, 0.35], [0.1, 0.9]],
            ],
            dtype=jnp.float32,
        ),
        NamedSharding(mesh, P("data", None, None)),
    )
    expected = LocalExpertDispatcher()(experts, x, expert_ids, weights)

    actual = ExpertParallelDispatcher(mesh)(experts, x, expert_ids, weights)

    assert actual.sharding.spec == P("data", None, None)
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_all_to_all_expert_dispatcher_matches_local_dispatcher_on_ep_mesh() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape((1, 4)), ("data", "ep"))
    experts = ExpertSwiGLU(
        hidden_size=3,
        intermediate_size=5,
        num_experts=4,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    gate = jnp.arange(4 * 3 * 5, dtype=jnp.float32).reshape(4, 3, 5) / 37.0
    up = jnp.arange(4 * 3 * 5, dtype=jnp.float32).reshape(4, 3, 5)[::-1] / 41.0
    down = jnp.arange(4 * 5 * 3, dtype=jnp.float32).reshape(4, 5, 3) / 43.0
    experts.gate[...] = jax.device_put(gate, NamedSharding(mesh, P("ep", None, None)))
    experts.up[...] = jax.device_put(up, NamedSharding(mesh, P("ep", None, None)))
    experts.down[...] = jax.device_put(down, NamedSharding(mesh, P("ep", None, None)))
    x = jax.device_put(
        jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3) / 17.0,
        NamedSharding(mesh, P("data", None, None)),
    )
    expert_ids = jax.device_put(
        jnp.asarray(
            [
                [[0, 1], [2, 3], [1, 3], [0, 2]],
                [[3, 1], [0, 2], [2, 1], [3, 0]],
            ],
            dtype=jnp.int32,
        ),
        NamedSharding(mesh, P("data", None, None)),
    )
    weights = jax.device_put(
        jnp.asarray(
            [
                [[0.25, 0.75], [0.5, 0.5], [0.4, 0.6], [0.2, 0.8]],
                [[0.55, 0.45], [0.3, 0.7], [0.65, 0.35], [0.1, 0.9]],
            ],
            dtype=jnp.float32,
        ),
        NamedSharding(mesh, P("data", None, None)),
    )
    expected = LocalExpertDispatcher()(experts, x, expert_ids, weights)

    actual = AllToAllExpertDispatcher(mesh)(experts, x, expert_ids, weights)

    assert actual.sharding.spec == P("data", None, None)
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_all_to_all_expert_dispatcher_matches_local_dispatcher_on_ep_expert_fsdp_mesh() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape((1, 2, 2)), ("data", "ep", "expert_fsdp"))
    expected_experts = ExpertSwiGLU(
        hidden_size=3,
        intermediate_size=6,
        num_experts=4,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    actual_experts = ExpertSwiGLU(
        hidden_size=3,
        intermediate_size=6,
        num_experts=4,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(1),
    )
    gate = jnp.arange(4 * 3 * 6, dtype=jnp.float32).reshape(4, 3, 6) / 37.0
    up = jnp.arange(4 * 3 * 6, dtype=jnp.float32).reshape(4, 3, 6)[::-1] / 41.0
    down = jnp.arange(4 * 6 * 3, dtype=jnp.float32).reshape(4, 6, 3) / 43.0
    expected_experts.gate[...] = gate
    expected_experts.up[...] = up
    expected_experts.down[...] = down
    actual_experts.gate[...] = jax.device_put(gate, NamedSharding(mesh, P("ep", None, "expert_fsdp")))
    actual_experts.up[...] = jax.device_put(up, NamedSharding(mesh, P("ep", None, "expert_fsdp")))
    actual_experts.down[...] = jax.device_put(down, NamedSharding(mesh, P("ep", "expert_fsdp", None)))
    x = jax.device_put(
        jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3) / 17.0,
        NamedSharding(mesh, P("data", None, None)),
    )
    expert_ids = jax.device_put(
        jnp.asarray(
            [
                [[0, 1], [2, 3], [1, 3], [0, 2]],
                [[3, 1], [0, 2], [2, 1], [3, 0]],
            ],
            dtype=jnp.int32,
        ),
        NamedSharding(mesh, P("data", None, None)),
    )
    weights = jax.device_put(
        jnp.asarray(
            [
                [[0.25, 0.75], [0.5, 0.5], [0.4, 0.6], [0.2, 0.8]],
                [[0.55, 0.45], [0.3, 0.7], [0.65, 0.35], [0.1, 0.9]],
            ],
            dtype=jnp.float32,
        ),
        NamedSharding(mesh, P("data", None, None)),
    )
    expected = LocalExpertDispatcher()(expected_experts, x, expert_ids, weights)

    actual = AllToAllExpertDispatcher(mesh, expert_fsdp_axis_name="expert_fsdp")(actual_experts, x, expert_ids, weights)

    assert actual.sharding.spec == P("data", None, None)
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_all_to_all_expert_dispatcher_matches_local_dispatcher_on_data_ep_mesh() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape((2, 2)), ("data", "ep"))
    experts = ExpertSwiGLU(
        hidden_size=3,
        intermediate_size=5,
        num_experts=4,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    gate = jnp.arange(4 * 3 * 5, dtype=jnp.float32).reshape(4, 3, 5) / 37.0
    up = jnp.arange(4 * 3 * 5, dtype=jnp.float32).reshape(4, 3, 5)[::-1] / 41.0
    down = jnp.arange(4 * 5 * 3, dtype=jnp.float32).reshape(4, 5, 3) / 43.0
    experts.gate[...] = jax.device_put(gate, NamedSharding(mesh, P("ep", None, None)))
    experts.up[...] = jax.device_put(up, NamedSharding(mesh, P("ep", None, None)))
    experts.down[...] = jax.device_put(down, NamedSharding(mesh, P("ep", None, None)))
    x = jax.device_put(
        jnp.arange(4 * 3 * 3, dtype=jnp.float32).reshape(4, 3, 3) / 17.0,
        NamedSharding(mesh, P("data", None, None)),
    )
    expert_ids = jax.device_put(
        jnp.asarray(
            [
                [[0, 1], [2, 3], [1, 3]],
                [[3, 1], [0, 2], [2, 1]],
                [[1, 0], [3, 2], [0, 3]],
                [[2, 0], [1, 3], [3, 2]],
            ],
            dtype=jnp.int32,
        ),
        NamedSharding(mesh, P("data", None, None)),
    )
    weights = jax.device_put(
        jnp.asarray(
            [
                [[0.25, 0.75], [0.5, 0.5], [0.4, 0.6]],
                [[0.55, 0.45], [0.3, 0.7], [0.65, 0.35]],
                [[0.2, 0.8], [0.6, 0.4], [0.45, 0.55]],
                [[0.7, 0.3], [0.35, 0.65], [0.15, 0.85]],
            ],
            dtype=jnp.float32,
        ),
        NamedSharding(mesh, P("data", None, None)),
    )
    expected = LocalExpertDispatcher()(experts, x, expert_ids, weights)

    actual = AllToAllExpertDispatcher(mesh)(experts, x, expert_ids, weights)

    assert actual.sharding.spec == P("data", None, None)
    assert jnp.allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("mesh_shape", "axis_names", "expert_fsdp_axis", "context_axis", "batch_size"),
    [
        pytest.param((1, 4), ("data", "ep"), None, None, 1, id="ep4"),
        pytest.param((2, 2), ("data", "ep"), None, None, 4, id="data2_ep2"),
        pytest.param((1, 2, 2), ("data", "tp", "ep"), "tp", None, 1, id="tp2_ep2"),
        pytest.param((1, 2, 2), ("data", "cp", "ep"), None, "cp", 1, id="cp2_ep2"),
        pytest.param(
            (1, 2, 2),
            ("data", "ep", "expert_fsdp"),
            "expert_fsdp",
            None,
            1,
            id="ep2_expert_fsdp2",
        ),
    ],
)
def test_all_to_all_expert_dispatcher_preserves_edge_case_outputs_and_gradients(
    mesh_shape,
    axis_names,
    expert_fsdp_axis,
    context_axis,
    batch_size,
) -> None:
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape(mesh_shape), axis_names)
    num_experts = 4
    hidden_size = 3
    intermediate_size = 4
    seq_len = 6
    top_k = 2
    x = jnp.arange(batch_size * seq_len * hidden_size, dtype=jnp.float32).reshape(
        batch_size,
        seq_len,
        hidden_size,
    ) / 17.0
    gate = jnp.arange(num_experts * hidden_size * intermediate_size, dtype=jnp.float32).reshape(
        num_experts,
        hidden_size,
        intermediate_size,
    ) / 37.0
    up = jnp.flip(gate, axis=-1) / 1.3
    down = jnp.arange(num_experts * intermediate_size * hidden_size, dtype=jnp.float32).reshape(
        num_experts,
        intermediate_size,
        hidden_size,
    ) / 43.0
    weights = jnp.tile(jnp.asarray([0.3, 0.7], dtype=jnp.float32), (batch_size, seq_len, 1))
    assignment_count = batch_size * seq_len * top_k
    balanced = jnp.arange(assignment_count, dtype=jnp.int32).reshape(batch_size, seq_len, top_k) % num_experts
    no_expert_three = balanced % 3
    all_one_expert = jnp.zeros_like(balanced)
    duplicate_routes = jnp.repeat(
        (jnp.arange(batch_size * seq_len, dtype=jnp.int32) % num_experts).reshape(batch_size, seq_len, 1),
        top_k,
        axis=-1,
    )

    activation_spec = P("data", context_axis, None)
    sharded_x = jax.device_put(x, NamedSharding(mesh, activation_spec))
    sharded_weights = jax.device_put(weights, NamedSharding(mesh, activation_spec))
    sharded_gate = jax.device_put(gate, NamedSharding(mesh, P("ep", None, expert_fsdp_axis)))
    sharded_up = jax.device_put(up, NamedSharding(mesh, P("ep", None, expert_fsdp_axis)))
    sharded_down = jax.device_put(down, NamedSharding(mesh, P("ep", expert_fsdp_axis, None)))

    def reference_output(local_x, local_ids, local_weights, local_gate, local_up, local_down):
        selected_gate = local_gate[local_ids]
        selected_up = local_up[local_ids]
        selected_down = local_down[local_ids]
        gate_x = jnp.einsum("...h,...khi->...ki", local_x, selected_gate)
        up_x = jnp.einsum("...h,...khi->...ki", local_x, selected_up)
        outputs = jnp.einsum("...ki,...kih->...kh", jax.nn.silu(gate_x) * up_x, selected_down)
        return jnp.sum(outputs * local_weights[..., None], axis=-2)

    def distributed_output(local_x, local_ids, local_weights, local_gate, local_up, local_down):
        return _all_to_all_expert_swiglu(
            x=local_x,
            expert_ids=local_ids,
            weights=local_weights,
            gate=local_gate,
            up=local_up,
            down=local_down,
            mesh=mesh,
            axis_name="ep",
            expert_fsdp_axis_name=expert_fsdp_axis,
            context_parallel_axis_name=context_axis,
        )

    differentiable_args = (0, 2, 3, 4, 5)
    cotangent = jnp.arange(x.size, dtype=jnp.float32).reshape(x.shape) / float(x.size)

    def reference_loss(*args):
        return jnp.sum(reference_output(*args) * cotangent)

    def distributed_loss(*args):
        return jnp.sum(distributed_output(*args) * cotangent)

    compiled_reference_output = jax.jit(reference_output)
    compiled_distributed_output = jax.jit(distributed_output)
    compiled_reference_grad = jax.jit(jax.grad(reference_loss, argnums=differentiable_args))
    compiled_distributed_grad = jax.jit(jax.grad(distributed_loss, argnums=differentiable_args))
    for ids in (balanced, no_expert_three, all_one_expert, duplicate_routes):
        sharded_ids = jax.device_put(ids, NamedSharding(mesh, activation_spec))
        expected_output = compiled_reference_output(
            x,
            ids,
            weights,
            gate,
            up,
            down,
        )
        expected_gradients = compiled_reference_grad(x, ids, weights, gate, up, down)
        actual_output = compiled_distributed_output(
            sharded_x,
            sharded_ids,
            sharded_weights,
            sharded_gate,
            sharded_up,
            sharded_down,
        )
        actual_gradients = compiled_distributed_grad(
            sharded_x,
            sharded_ids,
            sharded_weights,
            sharded_gate,
            sharded_up,
            sharded_down,
        )
        repeated_output = compiled_distributed_output(
            sharded_x,
            sharded_ids,
            sharded_weights,
            sharded_gate,
            sharded_up,
            sharded_down,
        )
        repeated_gradients = compiled_distributed_grad(
            sharded_x,
            sharded_ids,
            sharded_weights,
            sharded_gate,
            sharded_up,
            sharded_down,
        )

        assert ids.size == assignment_count
        assert actual_output.sharding.spec == activation_spec
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(repeated_output, actual_output)
        for actual_gradient, expected_gradient, repeated_gradient in zip(
            actual_gradients,
            expected_gradients,
            repeated_gradients,
            strict=True,
        ):
            np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(repeated_gradient, actual_gradient)
            assert bool(jnp.all(jnp.isfinite(actual_gradient)))
            expected_array = np.asarray(expected_gradient)
            for shard in actual_gradient.addressable_shards:
                np.testing.assert_allclose(shard.data, expected_array[shard.index], rtol=1e-5, atol=1e-5)


def test_all_to_all_expert_dispatcher_matches_multistep_reference_updates() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape((1, 4)), ("data", "ep"))
    num_experts = 4
    hidden_size = 3
    intermediate_size = 4
    x = jnp.arange(1 * 6 * hidden_size, dtype=jnp.float32).reshape(1, 6, hidden_size) / 17.0
    expert_ids = jnp.arange(1 * 6 * 2, dtype=jnp.int32).reshape(1, 6, 2) % num_experts
    weights = jnp.tile(jnp.asarray([0.3, 0.7], dtype=jnp.float32), (1, 6, 1))
    gate = jnp.arange(num_experts * hidden_size * intermediate_size, dtype=jnp.float32).reshape(
        num_experts,
        hidden_size,
        intermediate_size,
    ) / 37.0
    up = jnp.flip(gate, axis=-1) / 1.3
    down = jnp.arange(num_experts * intermediate_size * hidden_size, dtype=jnp.float32).reshape(
        num_experts,
        intermediate_size,
        hidden_size,
    ) / 43.0
    sharded_x = jax.device_put(x, NamedSharding(mesh, P("data", None, None)))
    sharded_ids = jax.device_put(expert_ids, NamedSharding(mesh, P("data", None, None)))
    sharded_weights = jax.device_put(weights, NamedSharding(mesh, P("data", None, None)))
    parameter_specs = (P("ep", None, None), P("ep", None, None), P("ep", None, None))

    def reference_loss(local_gate, local_up, local_down):
        selected_gate = local_gate[expert_ids]
        selected_up = local_up[expert_ids]
        selected_down = local_down[expert_ids]
        gate_x = jnp.einsum("...h,...khi->...ki", x, selected_gate)
        up_x = jnp.einsum("...h,...khi->...ki", x, selected_up)
        output = jnp.einsum("...ki,...kih->...kh", jax.nn.silu(gate_x) * up_x, selected_down)
        return jnp.mean(jnp.square(jnp.sum(output * weights[..., None], axis=-2)))

    def distributed_loss(local_gate, local_up, local_down):
        output = _all_to_all_expert_swiglu(
            x=sharded_x,
            expert_ids=sharded_ids,
            weights=sharded_weights,
            gate=local_gate,
            up=local_up,
            down=local_down,
            mesh=mesh,
            axis_name="ep",
            expert_fsdp_axis_name=None,
            context_parallel_axis_name=None,
        )
        return jnp.mean(jnp.square(output))

    reference_grad = jax.jit(jax.grad(reference_loss, argnums=(0, 1, 2)))
    distributed_grad = jax.jit(jax.grad(distributed_loss, argnums=(0, 1, 2)))
    expected_params = (gate, up, down)
    actual_params = tuple(
        jax.device_put(param, NamedSharding(mesh, spec))
        for param, spec in zip(expected_params, parameter_specs, strict=True)
    )
    learning_rate = jnp.asarray(1e-3, dtype=jnp.float32)
    for _ in range(3):
        expected_gradients = reference_grad(*expected_params)
        actual_gradients = distributed_grad(*actual_params)
        expected_params = tuple(
            param - learning_rate * gradient
            for param, gradient in zip(expected_params, expected_gradients, strict=True)
        )
        actual_params = tuple(
            param - learning_rate * gradient
            for param, gradient in zip(actual_params, actual_gradients, strict=True)
        )

    for actual_param, expected_param in zip(actual_params, expected_params, strict=True):
        np.testing.assert_allclose(actual_param, expected_param, rtol=1e-5, atol=1e-5)
        assert bool(jnp.all(jnp.isfinite(actual_param)))


def test_rdep_static_expert_dispatcher_matches_local_dispatcher_on_data_mesh() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape((4,)), ("data",))
    expected_experts = ExpertSwiGLU(
        hidden_size=3,
        intermediate_size=5,
        num_experts=4,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    actual_experts = ExpertSwiGLU(
        hidden_size=3,
        intermediate_size=5,
        num_experts=4,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(1),
    )
    gate = jnp.arange(4 * 3 * 5, dtype=jnp.float32).reshape(4, 3, 5) / 37.0
    up = jnp.arange(4 * 3 * 5, dtype=jnp.float32).reshape(4, 3, 5)[::-1] / 41.0
    down = jnp.arange(4 * 5 * 3, dtype=jnp.float32).reshape(4, 5, 3) / 43.0
    expected_experts.gate[...] = gate
    expected_experts.up[...] = up
    expected_experts.down[...] = down
    actual_experts.gate[...] = jax.device_put(gate, NamedSharding(mesh, P("data", None, None)))
    actual_experts.up[...] = jax.device_put(up, NamedSharding(mesh, P("data", None, None)))
    actual_experts.down[...] = jax.device_put(down, NamedSharding(mesh, P("data", None, None)))
    x = jax.device_put(
        jnp.arange(4 * 3 * 3, dtype=jnp.float32).reshape(4, 3, 3) / 17.0,
        NamedSharding(mesh, P("data", None, None)),
    )
    expert_ids = jax.device_put(
        jnp.asarray(
            [
                [[0, 1], [0, 1], [0, 1]],
                [[1, 0], [1, 0], [1, 0]],
                [[2, 3], [3, 2], [2, 3]],
                [[3, 2], [2, 3], [3, 2]],
            ],
            dtype=jnp.int32,
        ),
        NamedSharding(mesh, P("data", None, None)),
    )
    weights = jax.device_put(
        jnp.asarray(
            [
                [[0.25, 0.75], [0.5, 0.5], [0.4, 0.6]],
                [[0.55, 0.45], [0.3, 0.7], [0.65, 0.35]],
                [[0.2, 0.8], [0.6, 0.4], [0.45, 0.55]],
                [[0.7, 0.3], [0.35, 0.65], [0.15, 0.85]],
            ],
            dtype=jnp.float32,
        ),
        NamedSharding(mesh, P("data", None, None)),
    )
    expected = LocalExpertDispatcher()(expected_experts, x, expert_ids, weights)

    actual = RdepStaticExpertDispatcher(mesh)(actual_experts, x, expert_ids, weights)

    assert actual.sharding.spec == P("data", None, None)
    assert jnp.allclose(actual, expected, atol=1e-5)


def test_rdep_static_expert_dispatcher_lowers_collectives() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape((4,)), ("data",))
    experts = ExpertSwiGLU(
        hidden_size=2,
        intermediate_size=2,
        num_experts=4,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((4, 2, 2), dtype=jnp.float32)
    expert_ids = jnp.asarray([[[0, 1], [2, 3]], [[1, 0], [3, 2]], [[2, 3], [0, 1]], [[3, 2], [1, 0]]], dtype=jnp.int32)
    weights = jnp.ones((4, 2, 2), dtype=jnp.float32) / 2.0

    jaxpr = str(jax.make_jaxpr(lambda hidden, ids, route_weights: RdepStaticExpertDispatcher(mesh)(experts, hidden, ids, route_weights))(x, expert_ids, weights))

    assert "all_to_all" in jaxpr


def test_all_to_all_expert_dispatcher_lowers_collectives() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape((1, 4)), ("data", "ep"))
    experts = ExpertSwiGLU(
        hidden_size=2,
        intermediate_size=2,
        num_experts=4,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((1, 2, 2), dtype=jnp.float32)
    expert_ids = jnp.asarray([[[0, 1], [2, 3]]], dtype=jnp.int32)
    weights = jnp.ones((1, 2, 2), dtype=jnp.float32) / 2.0

    jaxpr = str(jax.make_jaxpr(lambda hidden, ids, route_weights: AllToAllExpertDispatcher(mesh)(experts, hidden, ids, route_weights))(x, expert_ids, weights))

    assert jax.config.jax_ragged_dot_use_gpu_pallas_triton_lowering is ragged_dot_pallas_triton_available()
    if jax.default_backend() == "cpu":
        assert "all_to_all" in jaxpr
    else:
        assert jaxpr.count("ragged_all_to_all[") == 2
    assert jaxpr.count("ragged_dot_general[") == 3


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


def test_apply_model_output_uses_expert_parallel_context() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape((1, 4)), ("data", "ep"))
    result = build_model(
        _tiny_trinity_spec(
            num_layers=2,
            initial_dense_layers=1,
            compute_dtype="float32",
            moe={"num_experts": 4, "top_k": 2},
        ),
        seed=0,
    )
    input_ids = jax.device_put(
        jnp.arange(16, dtype=jnp.int32).reshape(2, 8),
        NamedSharding(mesh, P("data", None)),
    )
    expected = apply_model_output(result.graph, result.state, input_ids)

    actual = apply_model_output(
        result.graph,
        result.state,
        input_ids,
        execution=ModelExecutionContext(expert_parallel_mesh=mesh),
    )

    assert jnp.allclose(actual.logits, expected.logits, atol=1e-5)
    assert len(actual.router_stats) == len(expected.router_stats)


def test_apply_model_output_uses_tensor_parallel_context() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:2], dtype=object).reshape((1, 2)), ("data", "tp"))
    result = build_model(_tiny_spec(compute_dtype="float32"), seed=0)
    input_ids = jax.device_put(
        jnp.arange(16, dtype=jnp.int32).reshape(2, 8),
        NamedSharding(mesh, P("data", None)),
    )
    expected = apply_model_output(result.graph, result.state, input_ids)

    actual = apply_model_output(
        result.graph,
        result.state,
        input_ids,
        execution=ModelExecutionContext(tensor_parallel_mesh=mesh),
    )

    assert jnp.allclose(actual.logits, expected.logits, atol=1e-5)
    assert getattr(actual.logits, "sharding", None).spec == P("data", None, "tp")


@pytest.mark.parametrize("axis_name", ["tp", "cp"])
def test_distributed_model_preserves_bfloat16_compute_dtype(axis_name: str) -> None:
    mesh = Mesh(np.asarray(jax.devices()[:2], dtype=object).reshape((1, 2)), ("data", axis_name))
    result = build_model(_tiny_spec(param_dtype="float32", compute_dtype="bfloat16"), seed=0)
    input_spec = P("data", axis_name if axis_name == "cp" else None)
    input_ids = jax.device_put(
        jnp.arange(16, dtype=jnp.int32).reshape(2, 8),
        NamedSharding(mesh, input_spec),
    )
    expected = apply_model_output(result.graph, result.state, input_ids)
    execution = (
        ModelExecutionContext(tensor_parallel_mesh=mesh)
        if axis_name == "tp"
        else ModelExecutionContext(context_parallel_mesh=mesh)
    )

    actual = apply_model_output(
        result.graph,
        result.state,
        input_ids,
        execution=execution,
    )

    assert actual.logits.dtype == expected.logits.dtype == jnp.bfloat16
    assert jnp.allclose(actual.logits, expected.logits, atol=2e-2)


def test_apply_model_output_uses_context_parallel_context() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:2], dtype=object).reshape((1, 2)), ("data", "cp"))
    result = build_model(_tiny_spec(compute_dtype="float32"), seed=0)
    input_ids = jax.device_put(
        jnp.arange(16, dtype=jnp.int32).reshape(2, 8),
        NamedSharding(mesh, P("data", "cp")),
    )
    expected = apply_model_output(result.graph, result.state, input_ids)

    actual = apply_model_output(
        result.graph,
        result.state,
        input_ids,
        execution=ModelExecutionContext(context_parallel_mesh=mesh),
    )

    assert jnp.allclose(actual.logits, expected.logits, atol=1e-5)
    assert getattr(actual.logits, "sharding", None).spec == P("data", "cp", None)


def test_apply_model_output_uses_context_and_tensor_parallel_context() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape((1, 2, 2)), ("data", "cp", "tp"))
    result = build_model(_tiny_spec(compute_dtype="float32"), seed=0)
    input_ids = jax.device_put(
        jnp.arange(16, dtype=jnp.int32).reshape(2, 8),
        NamedSharding(mesh, P("data", "cp")),
    )
    expected = apply_model_output(result.graph, result.state, input_ids)

    actual = apply_model_output(
        result.graph,
        result.state,
        input_ids,
        execution=ModelExecutionContext(tensor_parallel_mesh=mesh, context_parallel_mesh=mesh),
    )

    assert jnp.allclose(actual.logits, expected.logits, atol=1e-5)
    assert getattr(actual.logits, "sharding", None).spec == P("data", "cp", "tp")


def test_sequence_parallel_activation_uses_sequence_axis() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:2], dtype=object).reshape((1, 2)), ("data", "tp"))
    x = jax.device_put(
        jnp.arange(2 * 8 * 4, dtype=jnp.float32).reshape(2, 8, 4),
        NamedSharding(mesh, P("data", None, None)),
    )

    actual = sequence_parallel_activation(x, ModelExecutionContext(tensor_parallel_mesh=mesh))

    assert getattr(actual, "sharding", None).spec == P("data", "tp", None)
    assert jnp.allclose(actual, x)


def test_context_parallel_activation_takes_precedence_over_sequence_parallel() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape((1, 2, 2)), ("data", "cp", "tp"))
    x = jax.device_put(
        jnp.arange(2 * 8 * 4, dtype=jnp.float32).reshape(2, 8, 4),
        NamedSharding(mesh, P("data", None, None)),
    )

    actual = sequence_parallel_activation(
        x,
        ModelExecutionContext(tensor_parallel_mesh=mesh, context_parallel_mesh=mesh),
    )

    assert getattr(actual, "sharding", None).spec == P("data", "cp", None)
    assert jnp.allclose(actual, x)


def test_trinity_apply_model_output_uses_vocab_parallel_lm_head() -> None:
    mesh = Mesh(np.asarray(jax.devices()[:2], dtype=object).reshape((1, 2)), ("data", "tp"))
    result = build_model(_tiny_trinity_spec(compute_dtype="float32"), seed=0)
    input_ids = jax.device_put(
        jnp.arange(16, dtype=jnp.int32).reshape(2, 8),
        NamedSharding(mesh, P("data", None)),
    )
    expected = apply_model_output(result.graph, result.state, input_ids)

    actual = apply_model_output(
        result.graph,
        result.state,
        input_ids,
        execution=ModelExecutionContext(tensor_parallel_mesh=mesh),
    )

    assert jnp.allclose(actual.logits, expected.logits, atol=1e-5)
    assert getattr(actual.logits, "sharding", None).spec == P("data", None, "tp")


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

    assert layouts["lm_head"].tp_axis == 1


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
    assert layouts["lm_head"].tp_axis == 1


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
        assert layouts[tag].tp_axis is None
    assert layouts["moe_shared_gate"].fsdp_axis == 1
    assert layouts["moe_shared_up"].fsdp_axis == 1
    assert layouts["moe_shared_down"].fsdp_axis == 0
    assert layouts["moe_shared_gate"].tp_axis == 1
    assert layouts["moe_shared_up"].tp_axis == 1
    assert layouts["moe_shared_down"].tp_axis == 0
    expert_layouts = {item.tag: item for item in result.expert_layouts}
    assert set(expert_layouts) == {"moe_gate", "moe_up", "moe_down"}
    for tag in ("moe_gate", "moe_up", "moe_down"):
        assert expert_layouts[tag].expert_axis == 0
        assert expert_layouts[tag].matrix_axes == (1, 2)
    assert expert_layouts["moe_gate"].fsdp_axis == 2
    assert expert_layouts["moe_up"].fsdp_axis == 2
    assert expert_layouts["moe_down"].fsdp_axis == 1


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
