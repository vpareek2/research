"""Sparse feed-forward components."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from jaxtitan.errors import ContractError
from jaxtitan.models.components.dtypes import dtype_from_name
from jaxtitan.models.execution import (
    ModelExecutionContext,
    column_parallel_linear,
    ragged_dot_pallas_triton_available,
    row_parallel_linear,
)
from jaxtitan.models.output import AuxLoss, RouterStats
from jaxtitan.specs.model import ModelSpec, TrinityMoeSpec


# JAX 0.10.0 exposes this config flag even in wheels that omit the internal
# Pallas ragged-dot lowering module. Enable it only when the module is present.
jax.config.update("jax_ragged_dot_use_gpu_pallas_triton_lowering", ragged_dot_pallas_triton_available())


def _ragged_all_to_all(
    operand: jax.Array,
    output: jax.Array,
    input_offsets: jax.Array,
    send_sizes: jax.Array,
    output_offsets: jax.Array,
    recv_sizes: jax.Array,
    *,
    axis_name: str,
    axis_size: int,
) -> jax.Array:
    """Use native ragged transport, with a CPU-only semantic test lowering."""

    if jax.default_backend() != "cpu":
        return jax.lax.ragged_all_to_all(
            operand,
            output,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name=axis_name,
        )

    # XLA:CPU cannot lower ragged-all-to-all in JAX 0.10. Keep fake-device
    # correctness coverage by expressing the same slices with fixed buffers.
    # This branch is never selected by a GPU process.
    slices_per_device = input_offsets.shape[0] // axis_size
    buffer_size = operand.shape[0]
    positions = jnp.arange(buffer_size, dtype=jnp.int32)
    block_sizes = send_sizes.reshape(axis_size, slices_per_device)
    block_offsets = (jnp.cumsum(block_sizes, axis=1, dtype=jnp.int32) - block_sizes).reshape(-1)
    sent = jnp.zeros((axis_size, buffer_size, *operand.shape[1:]), dtype=operand.dtype)
    for slice_index in range(input_offsets.shape[0]):
        destination = slice_index // slices_per_device
        operand_indices = jnp.clip(input_offsets[slice_index] + positions, 0, operand.shape[0] - 1)
        sent_indices = jnp.clip(block_offsets[slice_index] + positions, 0, buffer_size - 1)
        mask = positions < send_sizes[slice_index]
        mask = mask.reshape((buffer_size,) + (1,) * (operand.ndim - 1))
        sent = sent.at[destination, sent_indices].add(jnp.where(mask, operand[operand_indices], 0))
    received = jax.lax.all_to_all(sent, axis_name, split_axis=0, concat_axis=0, tiled=True)
    received_output_offsets = jax.lax.all_to_all(
        output_offsets, axis_name, split_axis=0, concat_axis=0, tiled=True
    )
    received_block_sizes = recv_sizes.reshape(axis_size, slices_per_device)
    received_block_offsets = (
        jnp.cumsum(received_block_sizes, axis=1, dtype=jnp.int32) - received_block_sizes
    ).reshape(-1)
    result = output
    for slice_index in range(input_offsets.shape[0]):
        source = slice_index // slices_per_device
        received_indices = jnp.clip(received_block_offsets[slice_index] + positions, 0, buffer_size - 1)
        result_indices = jnp.clip(
            received_output_offsets[slice_index] + positions, 0, output.shape[0] - 1
        )
        mask = positions < recv_sizes[slice_index]
        mask = mask.reshape((buffer_size,) + (1,) * (operand.ndim - 1))
        result = result.at[result_indices].add(jnp.where(mask, received[source, received_indices], 0))
    return result


@dataclass(frozen=True, slots=True)
class RouterOutput:
    """Selected expert ids and normalized routing weights."""

    expert_ids: jax.Array
    weights: jax.Array
    scores: jax.Array


class SigmoidTopKRouter(nnx.Module):
    """Deterministic sigmoid top-k router."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        dtype: Any,
        param_dtype: Any,
        rngs: nnx.Rngs,
        route_scale: float = 1.0,
        kernel_init: Any | None = None,
    ):
        if top_k <= 0 or num_experts <= 0 or top_k > num_experts:
            raise ContractError("router top_k must be positive and <= num_experts")
        if route_scale <= 0.0:
            raise ContractError(f"router route_scale must be positive, got {route_scale}")
        linear_kwargs = {} if kernel_init is None else {"kernel_init": kernel_init}
        self.proj = nnx.Linear(
            hidden_size,
            num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            **linear_kwargs,
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.route_scale = float(route_scale)

    def __call__(self, x: jax.Array, expert_bias: jax.Array | None = None) -> RouterOutput:
        logits = jnp.asarray(self.proj(x), dtype=jnp.float32)
        scores = jax.nn.sigmoid(logits)
        if expert_bias is None:
            expert_bias = jnp.zeros((self.num_experts,), dtype=jnp.float32)
        expert_bias = jnp.asarray(expert_bias, dtype=jnp.float32)
        if expert_bias.shape != (self.num_experts,):
            raise ContractError(f"expert_bias must have shape [{self.num_experts}], got {expert_bias.shape}")
        _biased_scores, expert_ids = jax.lax.top_k(scores + expert_bias, self.top_k)
        top_scores = jnp.take_along_axis(scores, expert_ids, axis=-1)
        denominator = jnp.sum(top_scores, axis=-1, keepdims=True) + jnp.asarray(1e-20, dtype=jnp.float32)
        weights = (top_scores / denominator) * jnp.asarray(self.route_scale, dtype=jnp.float32)
        return RouterOutput(
            expert_ids=jnp.asarray(expert_ids, dtype=jnp.int32),
            weights=jnp.asarray(weights, dtype=x.dtype),
            scores=scores,
        )


class ExpertSwiGLU(nnx.Module):
    """Selected-expert SwiGLU feed-forward block."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        *,
        dtype: Any,
        param_dtype: Any,
        rngs: nnx.Rngs,
        kernel_init: Any | None = None,
    ):
        initializer = nnx.initializers.lecun_normal() if kernel_init is None else kernel_init
        self.gate = nnx.Param(initializer(rngs.params(), (num_experts, hidden_size, intermediate_size), param_dtype))
        self.up = nnx.Param(initializer(rngs.params(), (num_experts, hidden_size, intermediate_size), param_dtype))
        self.down = nnx.Param(initializer(rngs.params(), (num_experts, intermediate_size, hidden_size), param_dtype))
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.dtype = dtype
        self.param_dtype = param_dtype

    def local_forward(self, x: jax.Array, expert_ids: jax.Array) -> jax.Array:
        """Run selected complete expert matrices without applying route weights."""

        x = jnp.asarray(x, dtype=self.dtype)
        expert_ids = jnp.asarray(expert_ids, dtype=jnp.int32)
        if expert_ids.shape[:-1] != x.shape[:-1]:
            raise ContractError("expert routing shape must match input batch and sequence dimensions")

        gate = jnp.asarray(self.gate[...], dtype=self.dtype)[expert_ids]
        up = jnp.asarray(self.up[...], dtype=self.dtype)[expert_ids]
        down = jnp.asarray(self.down[...], dtype=self.dtype)[expert_ids]
        gate_x = jnp.einsum("...h,...khi->...ki", x, gate)
        up_x = jnp.einsum("...h,...khi->...ki", x, up)
        expert_hidden = jax.nn.silu(gate_x) * up_x
        return jnp.einsum("...ki,...kih->...kh", expert_hidden, down)

    def __call__(self, x: jax.Array, expert_ids: jax.Array, weights: jax.Array) -> jax.Array:
        weights = jnp.asarray(weights, dtype=self.dtype)
        if expert_ids.shape != weights.shape:
            raise ContractError(f"expert_ids shape {expert_ids.shape} must equal weights shape {weights.shape}")
        expert_output = self.local_forward(x, expert_ids)
        return jnp.sum(expert_output * weights[..., None], axis=-2)


class LocalExpertDispatcher:
    """Reference expert dispatcher for local or replicated routed experts."""

    def __call__(self, experts: ExpertSwiGLU, x: jax.Array, expert_ids: jax.Array, weights: jax.Array) -> jax.Array:
        return experts(x, expert_ids, weights)


class ExpertParallelDispatcher:
    """Exact expert-axis dispatcher for EP-sharded expert matrices.

    This is a correctness-first collective path. Tokens remain logically
    replicated over the expert axis, each shard computes only the selected
    experts it owns, and a psum combines the partial outputs.
    """

    def __init__(
        self,
        mesh: Any,
        *,
        axis_name: str = "ep",
        expert_fsdp_axis_name: str | None = None,
        context_parallel_axis_name: str | None = None,
    ):
        if axis_name not in mesh.axis_names:
            raise ContractError(f"expert parallel dispatcher requires mesh axis {axis_name!r}")
        if expert_fsdp_axis_name is not None and expert_fsdp_axis_name not in mesh.axis_names:
            raise ContractError(f"expert FSDP dispatcher requires mesh axis {expert_fsdp_axis_name!r}")
        if context_parallel_axis_name is not None and context_parallel_axis_name not in mesh.axis_names:
            raise ContractError(f"expert parallel dispatcher requires context axis {context_parallel_axis_name!r}")
        self.mesh = mesh
        self.axis_name = axis_name
        self.expert_fsdp_axis_name = expert_fsdp_axis_name
        self.context_parallel_axis_name = context_parallel_axis_name

    def __call__(self, experts: ExpertSwiGLU, x: jax.Array, expert_ids: jax.Array, weights: jax.Array) -> jax.Array:
        gate = jnp.asarray(experts.gate[...], dtype=experts.dtype)
        up = jnp.asarray(experts.up[...], dtype=experts.dtype)
        down = jnp.asarray(experts.down[...], dtype=experts.dtype)
        return _expert_parallel_swiglu(
            x=jnp.asarray(x, dtype=experts.dtype),
            expert_ids=jnp.asarray(expert_ids, dtype=jnp.int32),
            weights=jnp.asarray(weights, dtype=experts.dtype),
            gate=gate,
            up=up,
            down=down,
            mesh=self.mesh,
            axis_name=self.axis_name,
            expert_fsdp_axis_name=self.expert_fsdp_axis_name,
            context_parallel_axis_name=self.context_parallel_axis_name,
        )


class AllToAllExpertDispatcher:
    """Dropless source-sharded EP dispatcher with expert-major execution."""

    def __init__(
        self,
        mesh: Any,
        *,
        axis_name: str = "ep",
        expert_fsdp_axis_name: str | None = None,
        context_parallel_axis_name: str | None = None,
    ):
        if axis_name not in mesh.axis_names:
            raise ContractError(f"all-to-all expert dispatcher requires mesh axis {axis_name!r}")
        if expert_fsdp_axis_name is not None and expert_fsdp_axis_name not in mesh.axis_names:
            raise ContractError(f"all-to-all expert FSDP dispatcher requires mesh axis {expert_fsdp_axis_name!r}")
        if context_parallel_axis_name is not None and context_parallel_axis_name not in mesh.axis_names:
            raise ContractError(f"all-to-all expert dispatcher requires context axis {context_parallel_axis_name!r}")
        self.mesh = mesh
        self.axis_name = axis_name
        self.expert_fsdp_axis_name = expert_fsdp_axis_name
        self.context_parallel_axis_name = context_parallel_axis_name

    def __call__(self, experts: ExpertSwiGLU, x: jax.Array, expert_ids: jax.Array, weights: jax.Array) -> jax.Array:
        gate = jnp.asarray(experts.gate[...], dtype=experts.dtype)
        up = jnp.asarray(experts.up[...], dtype=experts.dtype)
        down = jnp.asarray(experts.down[...], dtype=experts.dtype)
        return _all_to_all_expert_swiglu(
            x=jnp.asarray(x, dtype=experts.dtype),
            expert_ids=jnp.asarray(expert_ids, dtype=jnp.int32),
            weights=jnp.asarray(weights, dtype=experts.dtype),
            gate=gate,
            up=up,
            down=down,
            mesh=self.mesh,
            axis_name=self.axis_name,
            expert_fsdp_axis_name=self.expert_fsdp_axis_name,
            context_parallel_axis_name=self.context_parallel_axis_name,
        )


class RdepStaticExpertDispatcher:
    """Semantic RDEP dispatcher that pools route rows across the data axis."""

    def __init__(self, mesh: Any, *, axis_name: str = "data", context_parallel_axis_name: str | None = None):
        if axis_name not in mesh.axis_names:
            raise ContractError(f"RDEP dispatcher requires mesh axis {axis_name!r}")
        if context_parallel_axis_name is not None and context_parallel_axis_name not in mesh.axis_names:
            raise ContractError(f"RDEP dispatcher requires context axis {context_parallel_axis_name!r}")
        self.mesh = mesh
        self.axis_name = axis_name
        self.context_parallel_axis_name = context_parallel_axis_name

    def __call__(self, experts: ExpertSwiGLU, x: jax.Array, expert_ids: jax.Array, weights: jax.Array) -> jax.Array:
        gate = jnp.asarray(experts.gate[...], dtype=experts.dtype)
        up = jnp.asarray(experts.up[...], dtype=experts.dtype)
        down = jnp.asarray(experts.down[...], dtype=experts.dtype)
        return _rdep_static_expert_swiglu(
            x=jnp.asarray(x, dtype=experts.dtype),
            expert_ids=jnp.asarray(expert_ids, dtype=jnp.int32),
            weights=jnp.asarray(weights, dtype=experts.dtype),
            gate=gate,
            up=up,
            down=down,
            mesh=self.mesh,
            axis_name=self.axis_name,
            context_parallel_axis_name=self.context_parallel_axis_name,
        )


class SharedSwiGLU(nnx.Module):
    """Always-on shared expert path with ordinary dense FFN matrices."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        dtype: Any,
        param_dtype: Any,
        rngs: nnx.Rngs,
        kernel_init: Any | None = None,
    ):
        linear_kwargs = {} if kernel_init is None else {"kernel_init": kernel_init}
        self.gate = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            **linear_kwargs,
        )
        self.up = nnx.Linear(
            hidden_size,
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            **linear_kwargs,
        )
        self.down = nnx.Linear(
            intermediate_size,
            hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            **linear_kwargs,
        )

    def __call__(self, x: jax.Array, execution: ModelExecutionContext | None = None) -> jax.Array:
        gate = column_parallel_linear(self.gate, x, execution)
        up = column_parallel_linear(self.up, x, execution)
        return row_parallel_linear(self.down, jax.nn.silu(gate) * up, execution)


class SparseMoE(nnx.Module):
    """Sparse selected-expert feed-forward module."""

    def __init__(self, spec: ModelSpec, moe: TrinityMoeSpec, rngs: nnx.Rngs, *, kernel_init: Any | None = None):
        if moe.expert_intermediate_size is None:
            raise ContractError("model.trinity.moe.expert_intermediate_size must be resolved before model construction")
        dtype = dtype_from_name(spec.compute_dtype)
        param_dtype = dtype_from_name(spec.param_dtype)
        self.router = SigmoidTopKRouter(
            spec.hidden_size,
            moe.num_experts,
            moe.top_k,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            route_scale=moe.route_scale,
            kernel_init=kernel_init,
        )
        self.expert_bias = nnx.Param(jnp.zeros((moe.num_experts,), dtype=jnp.float32))
        self.balance_name = moe.balance.name
        self.sequence_aux_loss_weight = float(moe.balance.sequence_aux_loss_weight)
        self.experts = ExpertSwiGLU(
            spec.hidden_size,
            moe.expert_intermediate_size,
            moe.num_experts,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            kernel_init=kernel_init,
        )
        self.dispatcher = LocalExpertDispatcher()
        if moe.num_shared_experts > 0:
            self.shared_experts = SharedSwiGLU(
                spec.hidden_size,
                moe.expert_intermediate_size * moe.num_shared_experts,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
                kernel_init=kernel_init,
            )

    def route(self, x: jax.Array) -> RouterOutput:
        return self.router(x, expert_bias=jax.lax.stop_gradient(self.expert_bias[...]))

    def __call__(self, x: jax.Array, execution: ModelExecutionContext | None = None) -> jax.Array:
        route = self.route(x)
        output = self._dispatcher(execution)(self.experts, x, route.expert_ids, route.weights)
        if hasattr(self, "shared_experts"):
            output = output + self.shared_experts(x, execution=execution)
        return output

    def forward_with_output(
        self,
        x: jax.Array,
        *,
        name: str,
        layer_index: int,
        execution: ModelExecutionContext | None = None,
    ) -> tuple[jax.Array, tuple[AuxLoss, ...], tuple[RouterStats, ...]]:
        route = self.route(x)
        output = self._dispatcher(execution)(self.experts, x, route.expert_ids, route.weights)
        if hasattr(self, "shared_experts"):
            output = output + self.shared_experts(x, execution=execution)
        stats = _router_stats(
            name=name,
            layer_index=layer_index,
            expert_ids=route.expert_ids,
            weights=route.weights,
            num_experts=self.experts.num_experts,
        )
        aux_losses = ()
        if self.balance_name == "smebu" and self.sequence_aux_loss_weight > 0.0:
            aux_losses = (
                AuxLoss(
                    name="moe_sequence_balance",
                    value=_sequence_balance_loss(route.expert_ids, route.scores, top_k=self.router.top_k),
                    weight=jnp.asarray(self.sequence_aux_loss_weight, dtype=jnp.float32),
                ),
            )
        return output, aux_losses, (stats,)

    def _dispatcher(self, execution: ModelExecutionContext | None) -> LocalExpertDispatcher | ExpertParallelDispatcher:
        if execution is not None and execution.expert_parallel_enabled:
            if execution.expert_parallel_dispatcher == "all_to_all":
                return AllToAllExpertDispatcher(
                    execution.expert_parallel_mesh,
                    axis_name=execution.expert_parallel_axis_name,
                    expert_fsdp_axis_name=execution.expert_fsdp_axis_name,
                    context_parallel_axis_name=execution.context_parallel_axis_name
                    if execution.context_parallel_enabled
                    else None,
                )
            if execution.expert_parallel_dispatcher == "rdep_static":
                if execution.expert_fsdp_axis_name is not None:
                    raise ContractError("RDEP dispatcher does not support expert_fsdp in this slice")
                return RdepStaticExpertDispatcher(
                    execution.expert_parallel_mesh,
                    axis_name=execution.expert_parallel_axis_name,
                    context_parallel_axis_name=execution.context_parallel_axis_name
                    if execution.context_parallel_enabled
                    else None,
                )
            if execution.expert_parallel_dispatcher == "psum":
                return ExpertParallelDispatcher(
                    execution.expert_parallel_mesh,
                    axis_name=execution.expert_parallel_axis_name,
                    expert_fsdp_axis_name=execution.expert_fsdp_axis_name,
                    context_parallel_axis_name=execution.context_parallel_axis_name
                    if execution.context_parallel_enabled
                    else None,
                )
            raise ContractError(f"unsupported expert parallel dispatcher {execution.expert_parallel_dispatcher!r}")
        return self.dispatcher


def _expert_parallel_swiglu(
    *,
    x: jax.Array,
    expert_ids: jax.Array,
    weights: jax.Array,
    gate: jax.Array,
    up: jax.Array,
    down: jax.Array,
    mesh: Any,
    axis_name: str,
    expert_fsdp_axis_name: str | None,
    context_parallel_axis_name: str | None,
) -> jax.Array:
    if x.ndim != 3:
        raise ContractError(f"expert parallel dispatcher requires x shape [batch, seq, hidden], got {x.shape}")
    if expert_ids.shape != weights.shape or expert_ids.ndim != 3:
        raise ContractError("expert parallel route ids and weights must have shape [batch, seq, top_k]")
    for name, value in (("gate", gate), ("up", up), ("down", down)):
        if value.ndim != 3:
            raise ContractError(f"expert parallel {name} matrix must be rank 3, got {value.shape}")

    def local_dispatch(local_x, local_expert_ids, local_weights, local_gate, local_up, local_down):
        local_expert_count = local_gate.shape[0]
        local_start = jax.lax.axis_index(axis_name) * local_expert_count
        local_ids = local_expert_ids - local_start
        selected_here = (local_ids >= 0) & (local_ids < local_expert_count)
        clipped_ids = jnp.clip(local_ids, 0, local_expert_count - 1)
        selected_gate = local_gate[clipped_ids]
        selected_up = local_up[clipped_ids]
        selected_down = local_down[clipped_ids]
        gate_x = jnp.einsum("...h,...khi->...ki", local_x, selected_gate)
        up_x = jnp.einsum("...h,...khi->...ki", local_x, selected_up)
        expert_hidden = jax.nn.silu(gate_x) * up_x
        expert_output = jnp.einsum("...ki,...kih->...kh", expert_hidden, selected_down)
        if expert_fsdp_axis_name is not None:
            expert_output = jax.lax.psum(expert_output, expert_fsdp_axis_name)
        weighted = expert_output * local_weights[..., None]
        local_output = jnp.sum(jnp.where(selected_here[..., None], weighted, 0), axis=-2)
        return jax.lax.psum(local_output, axis_name)

    mapped = jax.shard_map(
        local_dispatch,
        mesh=mesh,
        in_specs=(
            P("data", context_parallel_axis_name, None),
            P("data", context_parallel_axis_name, None),
            P("data", context_parallel_axis_name, None),
            P(axis_name, None, expert_fsdp_axis_name),
            P(axis_name, None, expert_fsdp_axis_name),
            P(axis_name, expert_fsdp_axis_name, None),
        ),
        out_specs=P("data", context_parallel_axis_name, None),
    )
    return mapped(x, expert_ids, weights, gate, up, down)


def _all_to_all_expert_swiglu(
    *,
    x: jax.Array,
    expert_ids: jax.Array,
    weights: jax.Array,
    gate: jax.Array,
    up: jax.Array,
    down: jax.Array,
    mesh: Any,
    axis_name: str,
    expert_fsdp_axis_name: str | None,
    context_parallel_axis_name: str | None,
) -> jax.Array:
    if x.ndim != 3:
        raise ContractError(f"all-to-all expert dispatcher requires x shape [batch, seq, hidden], got {x.shape}")
    if expert_ids.shape != weights.shape or expert_ids.ndim != 3:
        raise ContractError("all-to-all route ids and weights must have shape [batch, seq, top_k]")
    for name, value in (("gate", gate), ("up", up), ("down", down)):
        if value.ndim != 3:
            raise ContractError(f"all-to-all expert {name} matrix must be rank 3, got {value.shape}")
    ep_size = int(mesh.shape[axis_name])
    if gate.shape[0] <= 0 or gate.shape[0] % ep_size != 0:
        raise ContractError(
            f"expert count {gate.shape[0]} must be positive and divisible by ep axis size {ep_size}"
        )
    context_size = 1 if context_parallel_axis_name is None else int(mesh.shape[context_parallel_axis_name])
    sequence_partition_size = context_size * ep_size
    sequence_size = x.shape[1]
    padded_sequence_size = (
        (sequence_size + sequence_partition_size - 1) // sequence_partition_size
    ) * sequence_partition_size
    sequence_padding = padded_sequence_size - sequence_size
    if sequence_padding:
        x = jnp.pad(x, ((0, 0), (0, sequence_padding), (0, 0)))
        expert_ids = jnp.pad(expert_ids, ((0, 0), (0, sequence_padding), (0, 0)))
        weights = jnp.pad(weights, ((0, 0), (0, sequence_padding), (0, 0)))
    token_valid = jnp.arange(padded_sequence_size, dtype=jnp.int32) < sequence_size
    token_valid = jnp.broadcast_to(token_valid[None, :], x.shape[:2])

    def local_dispatch(local_x, local_expert_ids, local_weights, local_token_valid, local_gate, local_up, local_down):
        local_expert_count = local_gate.shape[0]
        expert_count = ep_size * local_expert_count
        token_count = local_x.shape[0] * local_x.shape[1]
        top_k = local_expert_ids.shape[-1]
        assignment_count = token_count * top_k
        flat_x = jnp.repeat(jnp.reshape(local_x, (token_count, local_x.shape[-1])), top_k, axis=0)
        flat_expert_ids = jnp.reshape(local_expert_ids, (assignment_count,))
        flat_weights = jnp.reshape(local_weights, (assignment_count,))
        flat_valid = jnp.repeat(jnp.reshape(local_token_valid, (token_count,)), top_k, axis=0)

        # Sort the source assignments globally by destination expert. Invalid
        # padding remains at the tail and is never transferred.
        source_order = jnp.argsort(
            jnp.where(flat_valid, flat_expert_ids, expert_count),
            stable=True,
        )
        inverse_source_order = jnp.argsort(source_order, stable=True)
        sorted_x = flat_x[source_order]
        expert_indices = jnp.arange(expert_count, dtype=jnp.int32)
        send_sizes = jnp.sum(
            flat_valid[None, :] & (flat_expert_ids[None, :] == expert_indices[:, None]),
            axis=1,
            dtype=jnp.int32,
        )
        input_offsets = jnp.cumsum(send_sizes, dtype=jnp.int32) - send_sizes

        # Exchange only metadata with fixed all-to-all. The activation payload
        # uses ragged all-to-all and lands directly in expert-major order.
        all_send_sizes = jax.lax.all_gather(send_sizes, axis_name)
        recv_sizes = jax.lax.all_to_all(send_sizes, axis_name, split_axis=0, concat_axis=0, tiled=True)
        source_rank = jax.lax.axis_index(axis_name)
        source_prefix = jnp.sum(
            jnp.where(
                jnp.arange(ep_size, dtype=jnp.int32)[:, None] < source_rank,
                all_send_sizes,
                0,
            ),
            axis=0,
            dtype=jnp.int32,
        )
        expert_totals = jnp.sum(all_send_sizes, axis=0, dtype=jnp.int32).reshape(
            ep_size, local_expert_count
        )
        expert_bases = (
            jnp.cumsum(expert_totals, axis=1, dtype=jnp.int32) - expert_totals
        ).reshape((expert_count,))
        output_offsets = expert_bases + source_prefix
        receive_capacity = ep_size * assignment_count
        packed_x = _ragged_all_to_all(
            sorted_x,
            jnp.zeros((receive_capacity, local_x.shape[-1]), dtype=local_x.dtype),
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name=axis_name,
            axis_size=ep_size,
        )
        group_sizes = jnp.sum(recv_sizes.reshape(ep_size, local_expert_count), axis=0, dtype=jnp.int32)

        gate_x = jax.lax.ragged_dot(packed_x, local_gate, group_sizes)
        up_x = jax.lax.ragged_dot(packed_x, local_up, group_sizes)
        expert_hidden = jax.nn.silu(gate_x) * up_x
        packed_output = jax.lax.ragged_dot(expert_hidden, local_down, group_sizes)
        if expert_fsdp_axis_name is not None:
            packed_output = jax.lax.psum(packed_output, expert_fsdp_axis_name)

        # Reverse the same ragged exchange. The expert rank sends source-major
        # chunks; each source receives them back in its expert-sorted order.
        received_offsets = jax.lax.all_to_all(
            output_offsets, axis_name, split_axis=0, concat_axis=0, tiled=True
        )
        returned_offsets = jax.lax.all_to_all(
            input_offsets, axis_name, split_axis=0, concat_axis=0, tiled=True
        )
        returned_sorted = _ragged_all_to_all(
            packed_output,
            jnp.zeros((assignment_count, local_x.shape[-1]), dtype=local_x.dtype),
            received_offsets,
            recv_sizes,
            returned_offsets,
            send_sizes,
            axis_name=axis_name,
            axis_size=ep_size,
        )
        assignment_output = returned_sorted[inverse_source_order]
        assignment_output = jnp.where(
            flat_valid[:, None], assignment_output * flat_weights[:, None], 0
        )
        token_output = jnp.sum(
            assignment_output.reshape(token_count, top_k, local_x.shape[-1]),
            axis=1,
        )
        return jnp.reshape(token_output, local_x.shape)

    sequence_axes = tuple(
        name for name in (context_parallel_axis_name, axis_name) if name is not None
    )

    mapped = jax.shard_map(
        local_dispatch,
        mesh=mesh,
        in_specs=(
            P("data", sequence_axes, None),
            P("data", sequence_axes, None),
            P("data", sequence_axes, None),
            P("data", sequence_axes),
            P(axis_name, None, expert_fsdp_axis_name),
            P(axis_name, None, expert_fsdp_axis_name),
            P(axis_name, expert_fsdp_axis_name, None),
        ),
        out_specs=P("data", sequence_axes, None),
    )
    output = mapped(x, expert_ids, weights, token_valid, gate, up, down)
    output = jax.lax.with_sharding_constraint(
        output,
        NamedSharding(mesh, P("data", context_parallel_axis_name, None)),
    )
    return output[:, :sequence_size, :]


def _rdep_static_expert_swiglu(
    *,
    x: jax.Array,
    expert_ids: jax.Array,
    weights: jax.Array,
    gate: jax.Array,
    up: jax.Array,
    down: jax.Array,
    mesh: Any,
    axis_name: str,
    context_parallel_axis_name: str | None,
) -> jax.Array:
    if x.ndim != 3:
        raise ContractError(f"RDEP expert dispatcher requires x shape [batch, seq, hidden], got {x.shape}")
    if expert_ids.shape != weights.shape or expert_ids.ndim != 3:
        raise ContractError("RDEP route ids and weights must have shape [batch, seq, top_k]")
    for name, value in (("gate", gate), ("up", up), ("down", down)):
        if value.ndim != 3:
            raise ContractError(f"RDEP expert {name} matrix must be rank 3, got {value.shape}")
    axis_size = int(mesh.shape[axis_name])
    if gate.shape[0] <= 0 or gate.shape[0] % axis_size != 0:
        raise ContractError(
            f"expert count {gate.shape[0]} must be positive and divisible by RDEP axis size {axis_size}"
        )

    def local_dispatch(local_x, local_expert_ids, local_weights, local_gate, local_up, local_down):
        local_expert_count = local_gate.shape[0]
        token_count = local_x.shape[0] * local_x.shape[1]
        top_k = local_expert_ids.shape[-1]
        assignment_count = token_count * top_k
        source_capacity = assignment_count
        flat_x = jnp.repeat(jnp.reshape(local_x, (token_count, local_x.shape[-1])), top_k, axis=0)
        flat_expert_ids = jnp.reshape(local_expert_ids, (assignment_count,))
        flat_weights = jnp.reshape(local_weights, (assignment_count,))
        flat_assignment_ids = jnp.arange(assignment_count, dtype=jnp.int32)
        owner_rank = flat_expert_ids // local_expert_count
        token_ids = flat_assignment_ids // top_k

        def bucket_for_owner(owner):
            mask = owner_rank == owner
            order_key = jnp.where(mask, flat_assignment_ids, assignment_count + flat_assignment_ids)
            order = jnp.argsort(order_key)[:source_capacity]
            valid = mask[order]
            local_ids = flat_expert_ids[order] - owner * local_expert_count
            return (
                flat_x[order],
                jnp.clip(local_ids, 0, local_expert_count - 1).astype(jnp.int32),
                flat_weights[order],
                token_ids[order],
                valid,
            )

        owners = jnp.arange(axis_size, dtype=jnp.int32)
        send_x, send_local_ids, send_weights, send_token_ids, send_valid = jax.vmap(bucket_for_owner)(owners)
        recv_x = jax.lax.all_to_all(send_x, axis_name, split_axis=0, concat_axis=0, tiled=True)
        recv_local_ids = jax.lax.all_to_all(send_local_ids, axis_name, split_axis=0, concat_axis=0, tiled=True)
        recv_weights = jax.lax.all_to_all(send_weights, axis_name, split_axis=0, concat_axis=0, tiled=True)
        recv_token_ids = jax.lax.all_to_all(send_token_ids, axis_name, split_axis=0, concat_axis=0, tiled=True)
        recv_valid = jax.lax.all_to_all(send_valid, axis_name, split_axis=0, concat_axis=0, tiled=True)

        selected_gate = local_gate[recv_local_ids]
        selected_up = local_up[recv_local_ids]
        selected_down = local_down[recv_local_ids]
        gate_x = jnp.einsum("...h,...hi->...i", recv_x, selected_gate)
        up_x = jnp.einsum("...h,...hi->...i", recv_x, selected_up)
        expert_hidden = jax.nn.silu(gate_x) * up_x
        recv_output = jnp.einsum("...i,...ih->...h", expert_hidden, selected_down)
        recv_output = jnp.where(recv_valid[..., None], recv_output * recv_weights[..., None], 0)

        return_by_source = jnp.reshape(recv_output, (axis_size, source_capacity, local_x.shape[-1]))
        token_by_source = jnp.reshape(recv_token_ids, (axis_size, source_capacity))
        valid_by_source = jnp.reshape(recv_valid, (axis_size, source_capacity))
        returned_output = jax.lax.all_to_all(return_by_source, axis_name, split_axis=0, concat_axis=0, tiled=True)
        returned_token_ids = jax.lax.all_to_all(token_by_source, axis_name, split_axis=0, concat_axis=0, tiled=True)
        returned_valid = jax.lax.all_to_all(valid_by_source, axis_name, split_axis=0, concat_axis=0, tiled=True)
        flat_output = jnp.zeros((token_count, local_x.shape[-1]), dtype=local_x.dtype)
        flat_output = flat_output.at[returned_token_ids].add(jnp.where(returned_valid[..., None], returned_output, 0))
        return jnp.reshape(flat_output, local_x.shape)

    mapped = jax.shard_map(
        local_dispatch,
        mesh=mesh,
        in_specs=(
            P(axis_name, context_parallel_axis_name, None),
            P(axis_name, context_parallel_axis_name, None),
            P(axis_name, context_parallel_axis_name, None),
            P(axis_name, None, None),
            P(axis_name, None, None),
            P(axis_name, None, None),
        ),
        out_specs=P(axis_name, context_parallel_axis_name, None),
    )
    return mapped(x, expert_ids, weights, gate, up, down)


def _router_stats(
    *,
    name: str,
    layer_index: int,
    expert_ids: jax.Array,
    weights: jax.Array,
    num_experts: int,
) -> RouterStats:
    selected = jax.nn.one_hot(expert_ids, num_experts, dtype=jnp.float32)
    reduce_axes = tuple(range(expert_ids.ndim))
    counts = jnp.sum(selected, axis=reduce_axes)
    importance = jnp.sum(selected * jnp.asarray(weights, dtype=jnp.float32)[..., None], axis=reduce_axes)
    total = jnp.sum(counts)
    mean = total / jnp.asarray(num_experts, dtype=jnp.float32)
    load_min = jnp.min(counts)
    load_max = jnp.max(counts)
    max_vio = (load_max - mean) / jnp.maximum(mean, jnp.asarray(1e-6, dtype=jnp.float32))
    probabilities = counts / jnp.maximum(total, jnp.asarray(1e-6, dtype=jnp.float32))
    safe_probabilities = jnp.where(probabilities > 0, probabilities, 1.0)
    entropy = -jnp.sum(jnp.where(probabilities > 0, probabilities * jnp.log(safe_probabilities), 0.0))
    return RouterStats(
        name=name,
        layer_index=layer_index,
        expert_counts=counts,
        importance=importance,
        total_assignments=total,
        load_min=load_min,
        load_max=load_max,
        load_mean=mean,
        max_vio=max_vio,
        load_entropy=entropy,
    )


def _sequence_balance_loss(expert_ids: jax.Array, scores: jax.Array, *, top_k: int) -> jax.Array:
    if scores.ndim != 3:
        raise ContractError(f"router scores must have shape [batch, seq, experts], got {scores.shape}")
    batch, seq_len, num_experts = scores.shape
    selected = jnp.sum(jax.nn.one_hot(expert_ids, num_experts, dtype=jnp.float32), axis=-2)
    selected_per_sequence = jnp.sum(selected, axis=1)
    f_i = (num_experts / (top_k * seq_len)) * selected_per_sequence
    normalized_scores = scores / jnp.maximum(
        jnp.sum(scores, axis=-1, keepdims=True),
        jnp.asarray(1e-20, dtype=jnp.float32),
    )
    p_i = jnp.mean(normalized_scores, axis=1)
    return jnp.sum(f_i * p_i) / jnp.asarray(batch, dtype=jnp.float32)
