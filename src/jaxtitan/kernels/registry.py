"""Internal kernel registry and config resolution.

This module is intentionally diagnostic-only in the first backend slice. It
describes Jaxtitan-owned kernel candidates and resolves whether the current run
would use them, but it does not load shared objects or alter model execution.
"""

from dataclasses import dataclass
import json
from typing import Any

from jaxtitan.errors import ContractError
from jaxtitan.specs.run import RunSpec


@dataclass(frozen=True, slots=True)
class KernelRegistration:
    """Static metadata for one Jaxtitan-owned kernel candidate."""

    op: str
    implementation: str
    status: str
    train_capable: bool
    eval_capable: bool
    inference_capable: bool
    notes: str


_REGISTRY: tuple[KernelRegistration, ...] = (
    KernelRegistration(
        op="rmsnorm",
        implementation="thunderkittens",
        status="standalone_poc",
        train_capable=False,
        eval_capable=False,
        inference_capable=False,
        notes="CUDA standalone POC exists; JAX FFI integration is not active.",
    ),
    KernelRegistration(
        op="swiglu",
        implementation="thunderkittens",
        status="planned",
        train_capable=False,
        eval_capable=False,
        inference_capable=False,
        notes="Candidate fused FFN kernel; no implementation registered.",
    ),
    KernelRegistration(
        op="attention",
        implementation="thunderkittens",
        status="planned",
        train_capable=False,
        eval_capable=False,
        inference_capable=False,
        notes="Candidate attention kernel; no implementation registered.",
    ),
    KernelRegistration(
        op="moe_dispatch_combine",
        implementation="thunderkittens",
        status="planned",
        train_capable=False,
        eval_capable=False,
        inference_capable=False,
        notes="Candidate MoE routing kernel; no implementation registered.",
    ),
    KernelRegistration(
        op="routed_expert_ffn",
        implementation="thunderkittens",
        status="planned",
        train_capable=False,
        eval_capable=False,
        inference_capable=False,
        notes="Candidate routed expert FFN kernel; no implementation registered.",
    ),
    KernelRegistration(
        op="optimizer_matrix_update",
        implementation="thunderkittens",
        status="planned",
        train_capable=False,
        eval_capable=False,
        inference_capable=False,
        notes="Candidate optimizer math kernel; no implementation registered.",
    ),
)


def kernel_registry() -> tuple[KernelRegistration, ...]:
    """Return known Jaxtitan-owned kernel candidates."""

    return _REGISTRY


def kernel_registry_payload() -> dict[str, Any]:
    """Return a JSON-safe registry summary."""

    return {
        "schema_version": 1,
        "kernels": [_registration_payload(item) for item in _REGISTRY],
    }


def kernel_plan(spec: RunSpec, *, device_kind: str | None = None) -> dict[str, Any]:
    """Resolve the kernel plan for a run.

    Stage 1 has no active JAX FFI kernels. When kernels are enabled, every
    target operation falls back to XLA with an explicit reason.
    """

    requested_ops = _requested_ops(spec)
    registry = {item.op: item for item in _REGISTRY}
    decisions = []
    active: dict[str, str] = {}
    fallback: dict[str, str] = {}
    unavailable: dict[str, str] = {}
    disabled: dict[str, str] = {}

    for op in requested_ops:
        registration = registry.get(op)
        if registration is None:
            reason = "not_registered"
            decisions.append(_decision(op, "xla", reason, registration=None))
            fallback[op] = reason
            unavailable[op] = reason
            continue
        if not spec.kernels.enabled:
            reason = "kernels_disabled"
            decisions.append(_decision(op, "xla", reason, registration=registration))
            fallback[op] = reason
            disabled[op] = reason
            continue
        reason = "no_jax_ffi_implementation"
        decisions.append(_decision(op, "xla", reason, registration=registration))
        fallback[op] = reason
        unavailable[op] = reason

    return {
        "schema_version": 1,
        "enabled": spec.kernels.enabled,
        "strict": spec.kernels.strict,
        "compile": spec.kernels.compile,
        "mode": "auto" if spec.kernels.enabled else "xla",
        "device_kind": device_kind,
        "registry_version": 1,
        "target_ops": list(requested_ops),
        "active": active,
        "fallback": fallback,
        "disabled": disabled,
        "unavailable": unavailable,
        "active_count": len(active),
        "fallback_count": len(fallback),
        "disabled_count": len(disabled),
        "unavailable_count": len(unavailable),
        "decisions": decisions,
    }


def kernel_plan_to_json(plan: dict[str, Any]) -> str:
    """Serialize a kernel plan as canonical JSON."""

    return json.dumps(plan, sort_keys=True, separators=(",", ":"))


def require_kernel_plan_supported(plan: dict[str, Any]) -> None:
    """Fail if strict kernel mode requires unavailable kernels."""

    if not plan.get("enabled") or not plan.get("strict"):
        return
    if plan.get("fallback_count", 0) == 0:
        return
    unavailable = plan.get("unavailable", {})
    if not isinstance(unavailable, dict):
        unavailable = {}
    ops = ", ".join(sorted(str(op) for op in unavailable)) or "unknown"
    raise ContractError(
        "kernels.strict=true requires validated Jaxtitan kernels for all target ops; "
        f"unavailable ops: {ops}"
    )


def format_kernel_registry(payload: dict[str, Any]) -> str:
    """Format kernel registry entries for humans."""

    lines = ["kernels:"]
    kernels = payload.get("kernels", [])
    if not kernels:
        lines.append("  none")
        return "\n".join(lines)
    for item in kernels:
        lines.append(
            "  "
            f"{item['op']}: impl={item['implementation']} status={item['status']} "
            f"train={item['train_capable']} eval={item['eval_capable']} infer={item['inference_capable']}"
        )
    return "\n".join(lines)


def format_kernel_plan(plan: dict[str, Any]) -> str:
    """Format one resolved kernel plan for humans."""

    lines = [
        (
            "kernels: "
            f"enabled={plan['enabled']} strict={plan['strict']} compile={plan['compile']} "
            f"active={plan['active_count']} fallback={plan['fallback_count']} "
            f"unavailable={plan['unavailable_count']} device={plan['device_kind']}"
        )
    ]
    if "cached_count" in plan:
        lines[0] += (
            f" cache={plan['cache_dir']} cached={plan['cached_count']} "
            f"missing_cache={plan['missing_cache_count']} stale_cache={plan['stale_cache_count']}"
        )
    for decision in plan["decisions"]:
        cache_status = "" if "cache_status" not in decision else f" cache={decision['cache_status']}"
        lines.append(
            "  "
            f"{decision['op']}: backend={decision['backend']} reason={decision['reason']} "
            f"status={decision['status']}{cache_status}"
        )
    return "\n".join(lines)


def _requested_ops(spec: RunSpec) -> tuple[str, ...]:
    ops = ["rmsnorm", "attention", "swiglu"]
    trinity = spec.model.trinity
    if spec.model.name == "trinity" and trinity is not None and trinity.moe is not None:
        ops.extend(["moe_dispatch_combine", "routed_expert_ffn"])
    if spec.optimizer.name == "muon":
        ops.append("optimizer_matrix_update")
    return tuple(dict.fromkeys(ops))


def _decision(
    op: str,
    backend: str,
    reason: str,
    *,
    registration: KernelRegistration | None,
) -> dict[str, Any]:
    return {
        "op": op,
        "backend": backend,
        "reason": reason,
        "implementation": None if registration is None else registration.implementation,
        "status": None if registration is None else registration.status,
        "train_capable": False if registration is None else registration.train_capable,
        "eval_capable": False if registration is None else registration.eval_capable,
        "inference_capable": False if registration is None else registration.inference_capable,
    }


def _registration_payload(registration: KernelRegistration) -> dict[str, Any]:
    return {
        "op": registration.op,
        "implementation": registration.implementation,
        "status": registration.status,
        "train_capable": registration.train_capable,
        "eval_capable": registration.eval_capable,
        "inference_capable": registration.inference_capable,
        "notes": registration.notes,
    }
