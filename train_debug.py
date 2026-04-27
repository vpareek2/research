"""
Optional training debug helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


DEBUG_NANS_ENV = "TRAIN_DEBUG_NANS"


@dataclass(frozen=True)
class NonfiniteLeaf:
    tree: str
    path: str
    shape: tuple[int, ...]
    dtype: str
    nan_count: int
    inf_count: int
    max_finite_abs: float

    def format(self) -> str:
        return (
            f"{self.tree}.{self.path}: "
            f"shape={self.shape} dtype={self.dtype} "
            f"nan_count={self.nan_count} inf_count={self.inf_count} "
            f"max_finite_abs={self.max_finite_abs:g}"
        )


def debug_nans_enabled() -> bool:
    return os.environ.get(DEBUG_NANS_ENV, "").lower() in {"1", "true", "yes", "on"}


def raise_for_nonfinite_training_state(step: int, metrics: dict[str, Any], *, model: Any, optimizer: Any):
    bad_metrics = {key: _to_float(value) for key, value in metrics.items() if _is_nonfinite_scalar(value)}
    model_leaf = find_nonfinite_leaf(model, tree_name="model")
    optimizer_leaf = find_nonfinite_leaf(optimizer, tree_name="optimizer")
    if not bad_metrics and model_leaf is None and optimizer_leaf is None:
        return

    details = [
        f"Nonfinite training state at step {step}.",
        "metrics:",
    ]
    if bad_metrics:
        details.extend(f"  {key}: {value}" for key, value in bad_metrics.items())
    else:
        details.append("  all tracked metrics finite")
    if model_leaf is not None:
        details.extend(["first nonfinite model leaf:", f"  {model_leaf.format()}"])
    if optimizer_leaf is not None:
        details.extend(["first nonfinite optimizer leaf:", f"  {optimizer_leaf.format()}"])
    if model_leaf is None and optimizer_leaf is None:
        details.append("No nonfinite model or optimizer leaf found.")

    raise RuntimeError("\n".join(details))


def find_nonfinite_leaf(tree: Any, *, tree_name: str = "tree") -> NonfiniteLeaf | None:
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not _is_float_array(leaf):
            continue

        array = jnp.asarray(leaf)
        is_nan = jnp.isnan(array)
        is_inf = jnp.isinf(array)
        has_nonfinite = jnp.any(is_nan | is_inf)
        if not bool(jax.device_get(has_nonfinite)):
            continue

        finite_abs = jnp.where(jnp.isfinite(array), jnp.abs(array), 0.0)
        return NonfiniteLeaf(
            tree=tree_name,
            path=_format_path(path),
            shape=tuple(int(dim) for dim in array.shape),
            dtype=str(array.dtype),
            nan_count=int(jax.device_get(jnp.sum(is_nan))),
            inf_count=int(jax.device_get(jnp.sum(is_inf))),
            max_finite_abs=float(jax.device_get(jnp.max(finite_abs))),
        )

    return None


def _is_float_array(value: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False
    return np.issubdtype(np.dtype(dtype), np.floating) or np.issubdtype(np.dtype(dtype), np.complexfloating)


def _is_nonfinite_scalar(value: Any) -> bool:
    scalar = _to_float(value)
    return scalar is not None and not math.isfinite(scalar)


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float, np.generic)):
        return float(value)
    if getattr(value, "shape", None) == ():
        return float(jax.device_get(value))
    return None


def _format_path(path: tuple[Any, ...]) -> str:
    if not path:
        return "<root>"
    return ".".join(_format_path_entry(entry) for entry in path)


def _format_path_entry(entry: Any) -> str:
    key = getattr(entry, "key", None)
    if key is not None:
        return str(key)
    idx = getattr(entry, "idx", None)
    if idx is not None:
        return str(idx)
    name = getattr(entry, "name", None)
    if name is not None:
        return str(name)
    return str(entry)
