"""Model execution policy helpers."""

from typing import Any

import jax

from jaxtitan.errors import ContractError


def apply_layer(layer: Any, *args: Any, remat: str) -> Any:
    """Apply one model layer under the requested execution policy."""

    if remat == "none":
        return layer(*args)
    if remat == "block":
        return jax.checkpoint(layer)(*args)
    raise ContractError(f"unsupported model.remat policy {remat!r}")
