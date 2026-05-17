"""
Parameter routing for optimizer selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import jax

from research.config import ModelConfig


class OptimClass(StrEnum):
    MATRIX = "matrix"
    VECTOR = "vector"
    EMBEDDING = "embedding"
    OUTPUT = "output"
    OTHER = "other"


@dataclass(frozen=True)
class ParamInfo:
    path: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: Any
    optim_class: OptimClass
    tags: frozenset[str]


def classify_param_tree(params, model_config: ModelConfig):
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    labels = []
    for path, value in leaves:
        pieces = _path_pieces(path)
        labels.append(classify_param(pieces, value.shape, model_config).value)
    return jax.tree_util.tree_unflatten(treedef, labels)


def iter_param_infos(params, model_config: ModelConfig) -> list[ParamInfo]:
    infos = []
    for path, value in jax.tree_util.tree_flatten_with_path(params)[0]:
        pieces = _path_pieces(path)
        infos.append(
            ParamInfo(
                path=pieces,
                shape=tuple(value.shape),
                dtype=value.dtype,
                optim_class=classify_param(pieces, value.shape, model_config),
                tags=tags_for_path(pieces),
            )
        )
    return infos


def classify_param(path: tuple[str, ...], shape: tuple[int, ...], model_config: ModelConfig) -> OptimClass:
    if _is_embedding_path(path):
        return OptimClass.EMBEDDING
    if _is_output_path(path):
        return OptimClass.OUTPUT
    if len(shape) == 1:
        return OptimClass.VECTOR
    if len(shape) == 2 and model_config.vocab_size not in shape:
        return OptimClass.MATRIX
    return OptimClass.OTHER


def tags_for_path(path: tuple[str, ...]) -> frozenset[str]:
    tags = set()
    path_set = set(path)
    if "embed" in path_set:
        tags.add("embedding")
    if "lm_head" in path_set:
        tags.add("output")
    if "attn" in path_set:
        tags.add("attention")
    if "mlp" in path_set:
        tags.add("mlp")
    if any("norm" in piece for piece in path):
        tags.add("norm")
    return frozenset(tags)


def _path_pieces(path) -> tuple[str, ...]:
    pieces = []
    for key in path:
        name = getattr(key, "key", None)
        if name is None:
            name = getattr(key, "name", None)
        if name == "value":
            continue
        pieces.append(str(name))
    return tuple(pieces)


def _is_embedding_path(path: tuple[str, ...]) -> bool:
    return len(path) >= 2 and path[0] == "embed" and path[1] == "embedding"


def _is_output_path(path: tuple[str, ...]) -> bool:
    return len(path) >= 2 and path[0] == "lm_head" and path[1] == "kernel"
