import jax.numpy as jnp
import pytest

from research.train_debug import DEBUG_NANS_ENV, debug_nans_enabled, find_nonfinite_leaf, raise_for_nonfinite_training_state


def test_debug_nans_enabled(monkeypatch):
    monkeypatch.delenv(DEBUG_NANS_ENV, raising=False)
    assert not debug_nans_enabled()

    monkeypatch.setenv(DEBUG_NANS_ENV, "1")
    assert debug_nans_enabled()

    monkeypatch.setenv(DEBUG_NANS_ENV, "true")
    assert debug_nans_enabled()


def test_find_nonfinite_leaf_reports_first_float_leaf():
    tree = {
        "ok": jnp.asarray([1.0, 2.0]),
        "bad": {
            "weights": jnp.asarray([1.0, float("nan"), float("inf")]),
        },
    }

    leaf = find_nonfinite_leaf(tree, tree_name="model")

    assert leaf is not None
    assert leaf.tree == "model"
    assert leaf.path == "bad.weights"
    assert leaf.shape == (3,)
    assert leaf.dtype == "float32"
    assert leaf.nan_count == 1
    assert leaf.inf_count == 1
    assert leaf.max_finite_abs == 1.0


def test_find_nonfinite_leaf_ignores_integer_arrays():
    tree = {"tokens": jnp.asarray([1, 2, 3]), "weights": jnp.asarray([1.0])}

    assert find_nonfinite_leaf(tree) is None


def test_raise_for_nonfinite_training_state_reports_model_and_optimizer():
    model = {"params": jnp.asarray([float("nan")])}
    optimizer = {"state": jnp.asarray([float("inf")])}

    with pytest.raises(RuntimeError) as exc:
        raise_for_nonfinite_training_state(
            7,
            {"train/loss": jnp.asarray(float("nan")), "train/grad_norm": jnp.asarray(1.0)},
            model=model,
            optimizer=optimizer,
        )

    message = str(exc.value)
    assert "Nonfinite training state at step 7" in message
    assert "train/loss: nan" in message
    assert "first nonfinite model leaf" in message
    assert "model.params" in message
    assert "first nonfinite optimizer leaf" in message
    assert "optimizer.state" in message


def test_raise_for_nonfinite_training_state_reports_state_with_finite_metrics():
    with pytest.raises(RuntimeError) as exc:
        raise_for_nonfinite_training_state(
            3,
            {"train/loss": jnp.asarray(1.0), "train/grad_norm": jnp.asarray(1.0)},
            model={"params": jnp.asarray([1.0])},
            optimizer={"state": jnp.asarray([float("nan")])},
        )

    message = str(exc.value)
    assert "all tracked metrics finite" in message
    assert "optimizer.state" in message


def test_raise_for_nonfinite_training_state_returns_for_finite_metrics():
    raise_for_nonfinite_training_state(
        7,
        {"train/loss": jnp.asarray(1.0), "train/grad_norm": jnp.asarray(1.0)},
        model={"params": jnp.asarray([1.0])},
        optimizer={"state": jnp.asarray([1.0])},
    )
