import jax
import jax.numpy as jnp
import optax
from flax import nnx

from research.config import ModelConfig, OptimizerConfig, TrainConfig
from research.lr_schedule import build_lr_schedule
from research.model import Model
from research.optimizers import OptimClass, build_optimizer, iter_param_infos
from research.optimizers.muon import mixed_muon_adamw, orthogonalize_newton_schulz
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
    new_tx = mixed_muon_adamw(
        classify_param_tree(params, cfg),
        optimizer_config(),
        learning_rate,
    )

    old_updates, _ = old_tx.update(grads, old_tx.init(params), params)
    new_updates, _ = new_tx.update(grads, new_tx.init(params), params)

    for old_leaf, new_leaf in zip(jax.tree.leaves(old_updates), jax.tree.leaves(new_updates), strict=True):
        assert jnp.allclose(old_leaf, new_leaf, rtol=1e-5, atol=1e-7)
