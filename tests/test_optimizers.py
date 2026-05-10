import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from research.config import AuroraOptimizerConfig, ModelConfig, OptimizerConfig, RiemannianAuroraOptimizerConfig, SOAPOptimizerConfig, TrainConfig
from research.lr_schedule import build_lr_schedule
from research.model import Model
from research.optimizers import OptimClass, build_optimizer, iter_param_infos
from research.optimizers.aurora import (
    aurora_balanced_polar,
    aurora_direction,
    riemannian_aurora_direction,
    riemannian_balanced_polar,
)
from research.optimizers.mixed import (
    mixed_matrix_adamw,
    orthogonalize_newton_schulz,
)
from research.optimizers.routing import classify_param_tree
from research.optimizers.soap import (
    get_orthogonal_matrix,
    init_preconditioner_axes,
    project,
    project_back,
    soap as soap_transform,
)
from research.pretrain import train_step


def tiny_model_config():
    return ModelConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=1,
        seq_len=8,
        theta=10000.0,
        eps=1e-6,
        tied=False,
    )


def train_config():
    return TrainConfig(
        seed=0,
        batch_size=2,
        seq_len=8,
        steps=2,
        log_every=1,
        eval_every=1,
        eval_steps=1,
        checkpoint_every=2,
        keep_last=2,
    )


def optimizer_config(**overrides):
    values = dict(name="muon", lr=1e-3, weight_decay=0.1)
    values.update(overrides)
    return OptimizerConfig(**values)


def path_map(infos):
    return {".".join(info.path): info for info in infos}


def test_param_routing_assigns_generic_classes():
    cfg = tiny_model_config()
    model = Model(cfg, rngs=nnx.Rngs(0))

    infos = path_map(iter_param_infos(nnx.state(model, nnx.Param), cfg))

    assert infos["embed.embedding"].optim_class == OptimClass.EMBEDDING
    assert infos["lm_head.kernel"].optim_class == OptimClass.OUTPUT
    assert infos["layers.0.attn.q.kernel"].optim_class == OptimClass.MATRIX
    assert infos["layers.0.attn.k.kernel"].optim_class == OptimClass.MATRIX
    assert infos["layers.0.mlp.gate.kernel"].optim_class == OptimClass.MATRIX
    assert infos["layers.0.pre_norm.scale"].optim_class == OptimClass.VECTOR
    assert infos["norm.scale"].optim_class == OptimClass.VECTOR
    assert infos["layers.0.attn.q.kernel"].tags == frozenset({"attention"})
    assert infos["layers.0.mlp.up.kernel"].tags == frozenset({"mlp"})


def test_newton_schulz_returns_finite_matrix_with_same_shape():
    matrix = jnp.arange(32, dtype=jnp.float32).reshape(4, 8)

    update = orthogonalize_newton_schulz(
        matrix,
        ns_steps=5,
        ns_coeffs=jnp.asarray((3.4445, -4.775, 2.0315), dtype=jnp.float32),
        eps=1e-8,
    )

    assert update.shape == matrix.shape
    assert bool(jnp.all(jnp.isfinite(update)))


def test_custom_muon_optimizer_runs_train_step():
    cfg = tiny_model_config()
    tc = train_config()
    oc = optimizer_config()
    model = Model(cfg, rngs=nnx.Rngs(0))
    optimizer = build_optimizer(model, cfg, oc, build_lr_schedule(tc, peak_lr=oc.lr))
    input_ids = jax.random.randint(jax.random.key(1), (tc.batch_size, tc.seq_len), 0, cfg.vocab_size)
    token_bytes = jnp.ones((cfg.vocab_size,), dtype=jnp.uint16)

    value, metrics = train_step(model, optimizer, input_ids, token_bytes)

    assert value.shape == ()
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(metrics["train/grad_norm"]))


def test_custom_adamw_optimizer_runs_train_step():
    cfg = tiny_model_config()
    tc = train_config()
    oc = optimizer_config(name="adamw")
    model = Model(cfg, rngs=nnx.Rngs(0))
    optimizer = build_optimizer(model, cfg, oc, build_lr_schedule(tc, peak_lr=oc.lr))
    input_ids = jax.random.randint(jax.random.key(1), (tc.batch_size, tc.seq_len), 0, cfg.vocab_size)
    token_bytes = jnp.ones((cfg.vocab_size,), dtype=jnp.uint16)

    value, metrics = train_step(model, optimizer, input_ids, token_bytes)

    assert value.shape == ()
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(metrics["train/grad_norm"]))


def test_soap_optimizer_runs_train_step():
    cfg = tiny_model_config()
    tc = train_config()
    oc = optimizer_config(name="soap")
    model = Model(cfg, rngs=nnx.Rngs(0))
    optimizer = build_optimizer(model, cfg, oc, build_lr_schedule(tc, peak_lr=oc.lr))
    input_ids = jax.random.randint(jax.random.key(1), (tc.batch_size, tc.seq_len), 0, cfg.vocab_size)
    token_bytes = jnp.ones((cfg.vocab_size,), dtype=jnp.uint16)

    value, metrics = train_step(model, optimizer, input_ids, token_bytes)

    assert value.shape == ()
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(metrics["train/grad_norm"]))


def test_soap_preconditioner_layout_preconditions_small_dims_and_skips_large_or_1d():
    config = optimizer_config(name="soap", soap=SOAPOptimizerConfig(max_precond_dim=8))

    matrix_axes = init_preconditioner_axes(jnp.zeros((4, 3), dtype=jnp.float32), config.soap)
    large_axes = init_preconditioner_axes(jnp.zeros((9, 3), dtype=jnp.float32), config.soap)
    vector_axes = init_preconditioner_axes(jnp.zeros((4,), dtype=jnp.float32), config.soap)

    assert [axis.shape for axis in matrix_axes.axes] == [(4, 4), (3, 3)]
    assert [axis.shape for axis in large_axes.axes] == [(1, 2), (3, 3)]
    assert [axis.shape for axis in vector_axes.axes] == [(1, 2)]


def test_soap_precondition_1d_enables_small_vector_preconditioner():
    config = optimizer_config(name="soap", soap=SOAPOptimizerConfig(precondition_1d=True, max_precond_dim=8))

    axes = init_preconditioner_axes(jnp.zeros((4,), dtype=jnp.float32), config.soap)

    assert [axis.shape for axis in axes.axes] == [(4, 4)]


def test_soap_project_back_recovers_preconditioned_and_skipped_axis_tensor():
    config = optimizer_config(name="soap", soap=SOAPOptimizerConfig(max_precond_dim=8))
    value = jnp.arange(36, dtype=jnp.float32).reshape(9, 4)
    gg = init_preconditioner_axes(value, config.soap)
    q = get_orthogonal_matrix(
        type(gg)(
            (
                gg.axes[0],
                jnp.asarray([[2.0, 0.2, 0.0, 0.0], [0.2, 1.5, 0.1, 0.0], [0.0, 0.1, 1.0, 0.3], [0.0, 0.0, 0.3, 0.5]]),
            )
        )
    )

    recovered = project_back(project(value, q), q)

    assert jnp.allclose(recovered, value, rtol=1e-5, atol=1e-5)


def test_soap_first_update_initializes_preconditioner_and_skips_param_update():
    config = optimizer_config(name="soap", weight_decay=0.0)
    params = {"w": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10.0}
    grads = {"w": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 100.0 + 0.01}
    tx = soap_transform(None, config, lambda count: jnp.asarray(1e-3, dtype=jnp.float32))

    updates, state = tx.update(grads, tx.init(params), params)
    q = state.q["w"].axes

    assert bool(state.initialized)
    assert int(state.count) == 0
    assert jnp.allclose(updates["w"], jnp.zeros_like(params["w"]))
    assert float(jnp.sum(jnp.abs(state.gg["w"].axes[0]))) > 0.0
    assert jnp.allclose(q[0].T @ q[0], jnp.eye(3), rtol=1e-5, atol=1e-5)
    assert jnp.allclose(q[1].T @ q[1], jnp.eye(4), rtol=1e-5, atol=1e-5)


def test_soap_second_update_is_finite_nonzero_and_keeps_static_state_shapes():
    config = optimizer_config(name="soap", weight_decay=0.0)
    params = {"w": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10.0}
    grads = {"w": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 100.0 + 0.01}
    tx = soap_transform(None, config, lambda count: jnp.asarray(1e-3, dtype=jnp.float32))
    _, state = tx.update(grads, tx.init(params), params)

    updates, state = tx.update(grads, state, params)

    assert int(state.count) == 1
    assert bool(jnp.all(jnp.isfinite(updates["w"])))
    assert float(jnp.linalg.norm(updates["w"])) > 0.0
    assert [axis.shape for axis in state.gg["w"].axes] == [(3, 3), (4, 4)]
    assert [axis.shape for axis in state.q["w"].axes] == [(3, 3), (4, 4)]


def test_soap_normalize_grads_normalizes_update_rms():
    config = optimizer_config(
        name="soap",
        lr=1.0,
        weight_decay=0.0,
        soap=SOAPOptimizerConfig(normalize_grads=True, correct_bias=False),
    )
    params = {"w": jnp.arange(4, dtype=jnp.float32)}
    grads = {"w": jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)}
    tx = soap_transform(None, config, lambda count: jnp.asarray(1.0, dtype=jnp.float32))
    _, state = tx.update(grads, tx.init(params), params)

    updates, _ = tx.update(grads, state, params)

    assert jnp.allclose(jnp.sqrt(jnp.mean(jnp.square(updates["w"]))), 1.0, rtol=1e-5, atol=1e-5)


def test_soap_unpreconditioned_1d_matches_reference_adamw_after_first_skip():
    config = optimizer_config(
        name="soap",
        lr=0.01,
        weight_decay=0.1,
        soap=SOAPOptimizerConfig(b1=0.5, b2=0.25, correct_bias=True),
    )
    params = {"w": jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)}
    grads = {"w": jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float32)}
    tx = soap_transform(None, config, lambda count: jnp.asarray(config.lr, dtype=jnp.float32))
    _, state = tx.update(grads, tx.init(params), params)

    updates, _ = tx.update(grads, state, params)

    step = 1
    m = (1.0 - config.soap.b1) * grads["w"]
    v = (1.0 - config.soap.b2) * jnp.square(grads["w"])
    step_size = config.lr * jnp.sqrt(1.0 - config.soap.b2**step) / (1.0 - config.soap.b1**step)
    direction = m / (jnp.sqrt(v) + config.soap.eps)
    adam_update = -step_size * direction
    expected = (params["w"] + adam_update) * (1.0 - config.lr * config.weight_decay) - params["w"]
    assert jnp.allclose(updates["w"], expected, rtol=1e-6, atol=1e-8)


def test_soap_matches_torch_reference_on_tiny_2d_second_step():
    torch = pytest.importorskip("torch")
    ref_path = Path(__file__).resolve().parents[1] / "ref" / "SOAP" / "soap.py"
    spec = importlib.util.spec_from_file_location("soap_ref", ref_path)
    soap_ref = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(soap_ref)

    initial = jnp.arange(9, dtype=jnp.float32).reshape(3, 3) / 10.0
    grad1 = jnp.asarray(
        [[0.11, 0.03, 0.07], [0.02, 0.13, 0.05], [0.09, 0.04, 0.17]],
        dtype=jnp.float32,
    )
    grad2 = jnp.asarray(
        [[0.05, 0.12, 0.01], [0.08, 0.02, 0.15], [0.04, 0.11, 0.09]],
        dtype=jnp.float32,
    )
    config = optimizer_config(
        name="soap",
        lr=0.003,
        weight_decay=0.01,
        soap=SOAPOptimizerConfig(b1=0.95, b2=0.95, precondition_frequency=10),
    )
    tx = soap_transform(None, config, lambda count: jnp.asarray(config.lr, dtype=jnp.float32))
    params = {"w": initial}
    _, state = tx.update({"w": grad1}, tx.init(params), params)
    updates, _ = tx.update({"w": grad2}, state, params)
    jax_param = params["w"] + updates["w"]

    torch_param = torch.nn.Parameter(torch.tensor(initial.tolist(), dtype=torch.float32))
    optim = soap_ref.SOAP(
        [torch_param],
        lr=config.lr,
        betas=(config.soap.b1, config.soap.b2),
        weight_decay=config.weight_decay,
        precondition_frequency=config.soap.precondition_frequency,
        max_precond_dim=config.soap.max_precond_dim,
        precondition_1d=config.soap.precondition_1d,
        normalize_grads=config.soap.normalize_grads,
        correct_bias=config.soap.correct_bias,
    )
    torch_param.grad = torch.tensor(grad1.tolist(), dtype=torch.float32)
    optim.step()
    torch_param.grad = torch.tensor(grad2.tolist(), dtype=torch.float32)
    optim.step()

    assert jnp.allclose(jax_param, jnp.asarray(torch_param.detach().numpy()), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("shape", [(4, 4), (8, 4), (4, 8)])
def test_aurora_direction_returns_finite_matrix_with_same_shape(shape):
    matrix = jnp.arange(shape[0] * shape[1], dtype=jnp.float32).reshape(shape) / 100.0 + 0.01

    update = aurora_direction(matrix, optimizer_config(name="aurora"))

    assert update.shape == matrix.shape
    assert bool(jnp.all(jnp.isfinite(update)))


def test_aurora_balancing_improves_rectangular_row_norm_uniformity():
    matrix = jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 100.0 + 0.01
    matrix = matrix.at[0].multiply(10.0)
    config = optimizer_config(name="aurora")

    plain = orthogonalize_newton_schulz(
        matrix,
        ns_steps=config.muon.ns_steps,
        ns_coeffs=jnp.asarray(config.muon.ns_coeffs, dtype=matrix.dtype),
        eps=config.muon.eps,
    )
    balanced = aurora_balanced_polar(matrix, config)

    plain_spread = jnp.std(jnp.sum(jnp.square(plain), axis=-1))
    balanced_spread = jnp.std(jnp.sum(jnp.square(balanced), axis=-1))
    assert balanced_spread < plain_spread


def test_aurora_single_preconditioned_polar_iteration_matches_manual_call():
    matrix = jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 100.0 + 0.01
    config = optimizer_config(name="aurora", aurora=AuroraOptimizerConfig(pp_iterations=1))
    row_norm = jnp.maximum(jnp.linalg.norm(matrix, axis=-1, keepdims=True), config.aurora.eps)
    expected = orthogonalize_newton_schulz(
        matrix / row_norm,
        ns_steps=config.muon.ns_steps,
        ns_coeffs=jnp.asarray(config.muon.ns_coeffs, dtype=matrix.dtype),
        eps=config.muon.eps,
    )

    update = aurora_balanced_polar(matrix, config)

    assert jnp.allclose(update, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("shape", [(8, 4), (4, 8)])
def test_riemannian_aurora_direction_returns_finite_matrix_with_same_shape(shape):
    matrix = jnp.arange(shape[0] * shape[1], dtype=jnp.float32).reshape(shape) / 100.0 + 0.01
    config = optimizer_config(
        name="riemannian_aurora",
        riemannian_aurora=RiemannianAuroraOptimizerConfig(outer_steps=1, cg_steps=4, retraction_steps=1),
    )

    update = riemannian_aurora_direction(matrix, config)

    assert update.shape == matrix.shape
    assert bool(jnp.all(jnp.isfinite(update)))


def test_riemannian_aurora_balanced_polar_approximately_satisfies_constraints():
    matrix = jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 100.0 + 0.01
    config = optimizer_config(
        name="riemannian_aurora",
        riemannian_aurora=RiemannianAuroraOptimizerConfig(outer_steps=2, cg_steps=8, retraction_steps=2),
    )

    update = riemannian_balanced_polar(matrix, config)

    assert update.shape == matrix.shape
    assert bool(jnp.all(jnp.isfinite(update)))
    assert jnp.allclose(update.T @ update, jnp.eye(4), rtol=0.2, atol=0.2)
    assert jnp.std(jnp.sum(jnp.square(update), axis=-1)) < 0.2


@pytest.mark.parametrize("name", ["aurora", "riemannian_aurora"])
def test_matrix_optimizer_variants_run_train_step(name):
    cfg = tiny_model_config()
    tc = train_config()
    oc = optimizer_config(name=name)
    model = Model(cfg, rngs=nnx.Rngs(0))
    optimizer = build_optimizer(model, cfg, oc, build_lr_schedule(tc, peak_lr=oc.lr))
    input_ids = jax.random.randint(jax.random.key(1), (tc.batch_size, tc.seq_len), 0, cfg.vocab_size)
    token_bytes = jnp.ones((cfg.vocab_size,), dtype=jnp.uint16)

    value, metrics = train_step(model, optimizer, input_ids, token_bytes)

    assert value.shape == ()
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(metrics["train/grad_norm"]))


@pytest.mark.parametrize("name", ["aurora", "riemannian_aurora"])
def test_aurora_variants_keep_non_matrix_leaves_on_adamw(name):
    cfg = tiny_model_config()
    model = Model(cfg, rngs=nnx.Rngs(0))
    params = nnx.state(model, nnx.Param)
    grads = jax.tree.map(
        lambda param: (jnp.arange(param.size, dtype=jnp.float32).reshape(param.shape) + 1.0) / 1000.0,
        params,
    )
    learning_rate = lambda count: jnp.asarray(1e-3, dtype=jnp.float32)
    labels = classify_param_tree(params, cfg)
    adam_tx = mixed_matrix_adamw(labels, optimizer_config(name="adamw"), learning_rate)
    aurora_tx = mixed_matrix_adamw(labels, optimizer_config(name=name), learning_rate)

    adam_updates, _ = adam_tx.update(grads, adam_tx.init(params), params)
    aurora_updates, _ = aurora_tx.update(grads, aurora_tx.init(params), params)

    for label, adam_leaf, aurora_leaf in zip(
        jax.tree.leaves(labels),
        jax.tree.leaves(adam_updates),
        jax.tree.leaves(aurora_updates),
        strict=True,
    ):
        if label != OptimClass.MATRIX.value:
            assert jnp.allclose(adam_leaf, aurora_leaf, rtol=1e-6, atol=1e-8)


def test_custom_muon_matches_old_optax_muon_one_step_updates():
    cfg = tiny_model_config()
    model = Model(cfg, rngs=nnx.Rngs(0))
    params = nnx.state(model, nnx.Param)
    grads = jax.tree.map(
        lambda param: (jnp.arange(param.size, dtype=jnp.float32).reshape(param.shape) + 1.0) / 1000.0,
        params,
    )
    learning_rate = lambda count: jnp.asarray(1e-3, dtype=jnp.float32)

    def old_muon_dimension_numbers(params):
        return jax.tree.map(
            lambda param: optax.contrib.MuonDimensionNumbers()
            if param.ndim == 2 and cfg.vocab_size not in param.shape
            else None,
            params,
        )

    old_tx = optax.contrib.muon(
        learning_rate=learning_rate,
        weight_decay=0.1,
        adam_weight_decay=0.1,
        muon_weight_dimension_numbers=old_muon_dimension_numbers,
    )
    new_tx = mixed_matrix_adamw(
        classify_param_tree(params, cfg),
        optimizer_config(),
        learning_rate,
    )

    old_updates, _ = old_tx.update(grads, old_tx.init(params), params)
    new_updates, _ = new_tx.update(grads, new_tx.init(params), params)

    for old_leaf, new_leaf in zip(jax.tree.leaves(old_updates), jax.tree.leaves(new_updates), strict=True):
        assert jnp.allclose(old_leaf, new_leaf, rtol=1e-5, atol=1e-7)
