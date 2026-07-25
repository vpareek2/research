import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "scripts"
    / "jaxtitan"
    / "audit_dist_muon_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_dist_muon_checkpoint",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def require_fake_devices() -> None:
    if jax.local_device_count() < 4:
        pytest.skip("JAX was initialized before fake CPU device flags were set")


def test_checkpoint_audit_accepts_identical_physical_replicas() -> None:
    require_fake_devices()
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape(2, 2), ("fsdp", "tp"))
    sharding = NamedSharding(mesh, P(None, "tp"))
    value = jax.device_put(jnp.arange(16, dtype=jnp.float32).reshape(4, 4), sharding)

    audit = MODULE.audit_tree({"w": value})

    assert audit["gate"] is True
    assert audit["finite"] is True
    assert audit["array_count"] == 1
    assert audit["replicated_array_count"] == 1
    assert audit["max_replica_abs_diff"] == 0.0


def test_checkpoint_audit_rejects_replica_disagreement() -> None:
    require_fake_devices()
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape(2, 2), ("fsdp", "tp"))
    sharding = NamedSharding(mesh, P(None, "tp"))
    value = jax.device_put(jnp.arange(16, dtype=jnp.float32).reshape(4, 4), sharding)
    disagreeing = jax.shard_map(
        lambda local: local + jax.lax.axis_index("fsdp").astype(local.dtype),
        mesh=mesh,
        in_specs=sharding.spec,
        out_specs=sharding.spec,
        check_vma=False,
    )(value)

    audit = MODULE.audit_tree({"w": disagreeing})

    assert audit["gate"] is False
    assert audit["replicas_equal"] is False
    assert audit["max_replica_abs_diff"] == pytest.approx(1.0)
    assert audit["replica_disagreement_paths"]


def test_checkpoint_audit_rejects_nonfinite_state() -> None:
    require_fake_devices()
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object).reshape(2, 2), ("fsdp", "tp"))
    sharding = NamedSharding(mesh, P(None, "tp"))
    value = jax.device_put(
        jnp.ones((4, 4), dtype=jnp.float32).at[0, 0].set(jnp.nan),
        sharding,
    )

    audit = MODULE.audit_tree({"w": value})

    assert audit["gate"] is False
    assert audit["finite"] is False
    assert audit["nonfinite_paths"]


def test_checkpoint_audit_accepts_tree_without_nontrivial_replica_axis() -> None:
    require_fake_devices()
    mesh = Mesh(np.asarray(jax.devices()[:4], dtype=object), ("tp",))
    sharding = NamedSharding(mesh, P("tp", None))
    value = jax.device_put(jnp.arange(16, dtype=jnp.float32).reshape(4, 4), sharding)

    audit = MODULE.audit_tree({"w": value})

    assert audit["gate"] is True
    assert audit["replicated_array_count"] == 0
    assert audit["max_replica_abs_diff"] == 0.0
