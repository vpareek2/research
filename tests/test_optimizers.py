import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from research.config import AuroraOptimizerConfig, ModelConfig, OptimizerConfig, RiemannianAuroraOptimizerConfig, TrainConfig
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
