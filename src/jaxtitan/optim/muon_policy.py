"""Host-static shape/topology policy for distributed Muon."""

from dataclasses import dataclass
from typing import Any

from jaxtitan.errors import ContractError

MUON_SHAPE_POLICY_VERSION = "shape_topology_v1"
MUON_POLICY_BFLOAT16_BYTES = 2
MUON_POLICY_NS_STEPS = 5
MUON_DIRECT_SINGLETON_LIMIT_BYTES = 384 * 1024
MUON_DIRECT_BUCKET_LIMIT_BYTES = 1024 * 1024
MUON_EXCHANGE_LATENCY_ALLOWANCE_BYTES = 6 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class MuonExecutionDecision:
    """Portable execution decision derived before JAX tracing."""

    policy_version: str
    eligible_executions: tuple[str, ...]
    execution: str
    selection_reason: str
    short_dimension: int
    long_dimension: int
    tp_size: int
    cohort_size: int
    gram_side: str
    modeled_costs: tuple[tuple[str, int], ...]


def select_muon_execution(
    *,
    requested_mode: str,
    canonical_tp_dim: int,
    logical_shape: tuple[int, int],
    tp_size: int,
    cohort_size: int,
) -> MuonExecutionDecision:
    """Select one statically compiled Muon execution without model-role input."""

    rows, columns = logical_shape
    if rows <= 0 or columns <= 0:
        raise ContractError(f"distributed Muon requires positive matrix dimensions, got {logical_shape}")
    if tp_size <= 1:
        raise ContractError(f"distributed Muon shape policy requires TP size greater than one, got {tp_size}")
    if cohort_size <= 0:
        raise ContractError(f"distributed Muon shape policy requires a positive cohort size, got {cohort_size}")
    if canonical_tp_dim not in {0, 1}:
        raise ContractError(f"distributed Muon canonical TP dimension must be 0 or 1, got {canonical_tp_dim}")
    if requested_mode not in {"duplicated", "distributed"}:
        raise ContractError(f"unsupported Muon TP mode {requested_mode!r}")

    short_dimension = min(logical_shape)
    long_dimension = max(logical_shape)
    common = {
        "policy_version": MUON_SHAPE_POLICY_VERSION,
        "short_dimension": short_dimension,
        "long_dimension": long_dimension,
        "tp_size": tp_size,
        "cohort_size": cohort_size,
    }
    logical_matrix_bytes = MUON_POLICY_BFLOAT16_BYTES * short_dimension * long_dimension
    if requested_mode == "duplicated":
        return MuonExecutionDecision(
            **common,
            eligible_executions=("duplicated",),
            execution="duplicated",
            selection_reason="requested_duplicated",
            gram_side="logical_matrix",
            modeled_costs=(("logical_matrix_bytes", logical_matrix_bytes),),
        )

    if canonical_tp_dim == 1:
        direct_pressure_bytes = _ceil_div(
            MUON_POLICY_BFLOAT16_BYTES * short_dimension**3,
            tp_size * long_dimension,
        )
        direct_limit_bytes = (
            MUON_DIRECT_SINGLETON_LIMIT_BYTES
            if cohort_size == 1
            else MUON_DIRECT_BUCKET_LIMIT_BYTES
        )
        use_direct = direct_pressure_bytes <= direct_limit_bytes
        execution = "distributed_direct" if use_direct else "duplicated"
        return MuonExecutionDecision(
            **common,
            eligible_executions=("duplicated", "distributed_direct"),
            execution=execution,
            selection_reason=(
                "aligned_direct_pressure_within_limit"
                if use_direct
                else "aligned_direct_pressure_exceeds_limit"
            ),
            gram_side="small" if use_direct else "logical_matrix",
            modeled_costs=(
                ("direct_pressure_bytes", direct_pressure_bytes),
                ("direct_limit_bytes", direct_limit_bytes),
                (
                    "small_gram_bytes",
                    MUON_POLICY_BFLOAT16_BYTES * short_dimension**2,
                ),
                ("logical_matrix_bytes", logical_matrix_bytes),
            ),
        )

    eligible = ["duplicated", "distributed_large_gram"]
    large_gram_extra_bytes = (
        MUON_POLICY_NS_STEPS
        * MUON_POLICY_BFLOAT16_BYTES
        * (long_dimension**2 - short_dimension**2)
    )
    large_gram_bytes = MUON_POLICY_BFLOAT16_BYTES * long_dimension**2
    costs = [
        ("large_gram_extra_bytes", large_gram_extra_bytes),
        ("large_gram_bytes", large_gram_bytes),
        ("logical_matrix_bytes", logical_matrix_bytes),
    ]
    if long_dimension % tp_size:
        return MuonExecutionDecision(
            **common,
            eligible_executions=tuple(eligible),
            execution="distributed_large_gram",
            selection_reason="exchange_ineligible_long_dimension_not_divisible",
            gram_side="large",
            modeled_costs=tuple(costs),
        )

    eligible.append("distributed_exchange")
    exchange_bytes = (
        2
        * MUON_POLICY_BFLOAT16_BYTES
        * short_dimension
        * long_dimension
        // tp_size
    )
    exchange_latency_allowance_bytes = MUON_EXCHANGE_LATENCY_ALLOWANCE_BYTES // tp_size
    use_large_gram = (
        large_gram_extra_bytes
        <= exchange_bytes + exchange_latency_allowance_bytes
    )
    costs.extend(
        (
            ("exchange_bytes", exchange_bytes),
            ("exchange_latency_allowance_bytes", exchange_latency_allowance_bytes),
            (
                "small_gram_bytes",
                MUON_POLICY_BFLOAT16_BYTES * short_dimension**2,
            ),
        )
    )
    return MuonExecutionDecision(
        **common,
        eligible_executions=tuple(eligible),
        execution=(
            "distributed_large_gram"
            if use_large_gram
            else "distributed_exchange"
        ),
        selection_reason=(
            "large_gram_within_exchange_cost"
            if use_large_gram
            else "exchange_lower_modeled_cost"
        ),
        gram_side="large" if use_large_gram else "small",
        modeled_costs=tuple(costs),
    )


def muon_shape_policy_constants() -> dict[str, Any]:
    """Return the stable policy constants recorded in runtime artifacts."""

    return {
        "version": MUON_SHAPE_POLICY_VERSION,
        "bfloat16_bytes": MUON_POLICY_BFLOAT16_BYTES,
        "newton_schulz_steps": MUON_POLICY_NS_STEPS,
        "direct_singleton_limit_bytes": MUON_DIRECT_SINGLETON_LIMIT_BYTES,
        "direct_bucket_limit_bytes": MUON_DIRECT_BUCKET_LIMIT_BYTES,
        "exchange_latency_allowance_bytes": MUON_EXCHANGE_LATENCY_ALLOWANCE_BYTES,
        "selection_inputs": [
            "logical_shape",
            "canonical_tp_dim",
            "tp_size",
            "compatible_cohort_size",
        ],
    }


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)
