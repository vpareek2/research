"""Mesh specs."""

from dataclasses import dataclass

from jaxtitan.errors import ContractError


@dataclass(frozen=True, slots=True)
class MeshSpec:
    """Static named-axis mesh contract.

    This does not create a JAX Mesh in the contracts-only slice.
    """

    axis_names: tuple[str, ...] = ("data",)
    axis_sizes: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_names", tuple(self.axis_names))
        object.__setattr__(self, "axis_sizes", tuple(self.axis_sizes))
        if len(self.axis_names) != len(self.axis_sizes):
            raise ContractError("mesh.axis_names and mesh.axis_sizes must have the same length")
        if not self.axis_names:
            raise ContractError("mesh must define at least one axis")
        if len(set(self.axis_names)) != len(self.axis_names):
            raise ContractError(f"mesh axis names must be unique, got {self.axis_names}")
        for axis_name in self.axis_names:
            if not axis_name:
                raise ContractError("mesh axis names must be non-empty")
        for axis_size in self.axis_sizes:
            if axis_size <= 0:
                raise ContractError(f"mesh axis sizes must be positive, got {self.axis_sizes}")
