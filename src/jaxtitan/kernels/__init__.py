"""Kernel backend registry and wrappers."""

from jaxtitan.kernels.build import (
    compile_kernel_plan,
    enrich_kernel_plan_with_cache,
    format_compile_result,
    load_cache_manifest,
)
from jaxtitan.kernels.registry import (
    format_kernel_plan,
    format_kernel_registry,
    kernel_plan,
    kernel_plan_to_json,
    kernel_registry,
    kernel_registry_payload,
    require_kernel_plan_supported,
)

__all__ = [
    "compile_kernel_plan",
    "enrich_kernel_plan_with_cache",
    "format_kernel_plan",
    "format_kernel_registry",
    "format_compile_result",
    "kernel_plan",
    "kernel_plan_to_json",
    "kernel_registry",
    "kernel_registry_payload",
    "load_cache_manifest",
    "require_kernel_plan_supported",
]
