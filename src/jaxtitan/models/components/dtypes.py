"""Model component dtype helpers."""

from typing import Any

import jax.numpy as jnp

from jaxtitan.errors import ContractError


def dtype_from_name(name: str) -> Any:
    """Resolve supported Jaxtitan model dtype names."""

    if name == "float32":
        return jnp.float32
    if name == "bfloat16":
        return jnp.bfloat16
    raise ContractError(f"unsupported dtype {name!r}; expected 'float32' or 'bfloat16'")
