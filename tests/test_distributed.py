import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} --xla_force_host_platform_device_count=4".strip()

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from research.config import DistributedConfig, ModelConfig, OptimizerConfig, TrainConfig
from research.distributed import create_distributed_context, place_replicated_state, shard_batch
from research.evals import eval_step
from research.lr_schedule import build_lr_schedule
from research.model import Model
from research.optimizers import build_optimizer
from research.pretrain import train_step


FAKE_DEVICE_COUNT = 4


def require_fake_devices():
    if jax.device_count() < FAKE_DEVICE_COUNT:
        pytest.skip(
            "JAX was initialized before tests/test_distributed.py could set "
            "--xla_force_host_platform_device_count=4"
        )


def make_data_mesh():
    require_fake_devices()
    return Mesh(jax.devices()[:FAKE_DEVICE_COUNT], ("data",))


def per_device_batch(global_batch: int, device_count: int) -> int:
    if global_batch % device_count != 0:
        raise ValueError("global batch size must divide evenly across devices")
    return global_batch // device_count


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


def train_config(batch_size: int = 8):
    return TrainConfig(
        seed=0,
        batch_size=batch_size,
        seq_len=8,
        steps=2,
        log_every=1,
        eval_every=1,
        eval_steps=1,
        checkpoint_every=2,
        keep_last=2,
    )


def optimizer_config():
    return OptimizerConfig(name="muon", lr=0.001, weight_decay=0.1)


def test_fake_cpu_device_availability():
    require_fake_devices()

    assert jax.device_count() >= FAKE_DEVICE_COUNT


def test_batch_axis_sharding_splits_global_batch():
    mesh = make_data_mesh()
    sharding = NamedSharding(mesh, P("data", None))
    batch = jnp.arange(64, dtype=jnp.uint32).reshape(8, 8)

    placed = jax.device_put(batch, sharding)

    assert placed.shape == (8, 8)
    assert placed.sharding == sharding
    assert len(placed.addressable_shards) == FAKE_DEVICE_COUNT
    assert {shard.data.shape for shard in placed.addressable_shards} == {(2, 8)}


def test_replicated_state_keeps_full_shape_on_each_device():
    mesh = make_data_mesh()
    sharding = NamedSharding(mesh, P())
    state = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)

    placed = jax.device_put(state, sharding)

    assert placed.shape == (4, 4)
    assert placed.sharding == sharding
    assert len(placed.addressable_shards) == FAKE_DEVICE_COUNT
    assert {shard.data.shape for shard in placed.addressable_shards} == {(4, 4)}


def test_jitted_train_like_step_uses_replicated_state_and_sharded_batch():
    mesh = make_data_mesh()
    state_sharding = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P("data", None))

    @jax.jit(
        in_shardings=(state_sharding, batch_sharding),
        out_shardings=(state_sharding, None),
    )
    def train_like_step(weight, input_ids):
        batch_mean = input_ids.astype(jnp.float32).mean()
        loss = jnp.square(weight - batch_mean)
        next_weight = weight - 0.01 * (2.0 * (weight - batch_mean))
        return next_weight, loss

    weight = jax.device_put(jnp.asarray(1.0, dtype=jnp.float32), state_sharding)
    batch = jax.device_put(jnp.arange(64, dtype=jnp.uint32).reshape(8, 8), batch_sharding)

    next_weight, loss = train_like_step(weight, batch)

    assert next_weight.shape == ()
    assert next_weight.sharding == state_sharding
    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))
    assert float(next_weight) != pytest.approx(float(weight))


def test_global_batch_must_divide_evenly_across_devices():
    assert per_device_batch(global_batch=8, device_count=4) == 2

    with pytest.raises(ValueError, match="global batch size"):
        per_device_batch(global_batch=10, device_count=4)


def test_create_distributed_context_reports_batch_math():
    require_fake_devices()

    context = create_distributed_context(DistributedConfig(device_count=4), train_config(batch_size=8))

    assert context.device_count == 4
    assert context.global_batch_size == 8
    assert context.per_device_batch_size == 2
    assert context.tokens_per_step == 64
    assert context.axis_name == "data"
    assert context.replicated_sharding == NamedSharding(context.mesh, P())
    assert context.batch_sharding == NamedSharding(context.mesh, P("data", None))


def test_disabled_distributed_context_uses_one_device():
    context = create_distributed_context(DistributedConfig(enabled=False, device_count=4), train_config(batch_size=8))

    assert context.device_count == 1
    assert context.per_device_batch_size == 8


def test_create_distributed_context_rejects_non_divisible_batch():
    require_fake_devices()

    with pytest.raises(ValueError, match="train.batch_size"):
        create_distributed_context(DistributedConfig(device_count=4), train_config(batch_size=10))


def test_create_distributed_context_rejects_too_many_devices():
    require_fake_devices()

    with pytest.raises(ValueError, match="exceeds available"):
        create_distributed_context(DistributedConfig(device_count=jax.device_count() + 1), train_config(batch_size=8))


def test_shard_batch_places_rank_specific_arrays():
    context = create_distributed_context(DistributedConfig(device_count=4), train_config(batch_size=8))
    batch = {
        "input_ids": jnp.arange(64, dtype=jnp.uint32).reshape(8, 8),
        "chunk_idx": jnp.arange(8, dtype=jnp.int32),
    }

    placed = shard_batch(batch, context)

    assert placed["input_ids"].sharding == context.batch_sharding
    assert {shard.data.shape for shard in placed["input_ids"].addressable_shards} == {(2, 8)}
    assert {shard.data.shape for shard in placed["chunk_idx"].addressable_shards} == {(2,)}


def test_shard_batch_rejects_wrong_leading_dim():
    context = create_distributed_context(DistributedConfig(device_count=4), train_config(batch_size=8))

    with pytest.raises(ValueError, match="global_batch_size"):
        shard_batch({"input_ids": jnp.zeros((4, 8), dtype=jnp.uint32)}, context)


def test_replicated_state_placement_and_sharded_train_eval_steps():
    context = create_distributed_context(DistributedConfig(device_count=4), train_config(batch_size=8))
    tc = train_config(batch_size=8)
    oc = optimizer_config()
    cfg = tiny_model_config()
    model = Model(cfg, rngs=nnx.Rngs(0))
    optimizer = build_optimizer(model, cfg, oc, build_lr_schedule(tc, peak_lr=oc.lr))
    place_replicated_state(model, optimizer, context)

    param_leaf = jax.tree.leaves(nnx.state(model, nnx.Param))[0]
    optimizer_leaf = jax.tree.leaves(nnx.state(optimizer))[0]
    assert param_leaf.sharding == context.replicated_sharding
    assert optimizer_leaf.sharding == context.replicated_sharding

    batch = shard_batch(
        {"input_ids": jax.random.randint(jax.random.key(1), (8, 8), 0, cfg.vocab_size)},
        context,
    )
    token_bytes = jax.device_put(jnp.ones((cfg.vocab_size,), dtype=jnp.uint16), context.replicated_sharding)

    train_loss, train_metrics = train_step(model, optimizer, batch["input_ids"], token_bytes)
    val_loss = eval_step(model, batch["input_ids"])

    assert train_loss.shape == ()
    assert val_loss.shape == ()
    assert bool(jnp.isfinite(train_loss))
    assert bool(jnp.isfinite(val_loss))
    assert train_metrics["train/grad_norm"].shape == ()
    assert bool(jnp.isfinite(train_metrics["train/grad_norm"]))
