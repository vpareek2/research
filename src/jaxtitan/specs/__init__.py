"""Static specs for Jaxtitan runs."""

from __future__ import annotations

from jaxtitan.specs.data import DataSpec, DatasetManifest, ShardInfo
from jaxtitan.specs.eval import EvalSpec
from jaxtitan.specs.generation import GenerationSpec
from jaxtitan.specs.mesh import MeshSpec
from jaxtitan.specs.model import ModelSpec
from jaxtitan.specs.optimizer import OptimizerSpec, ParamRouteRule, ScheduleSpec
from jaxtitan.specs.run import ArtifactSpec, RunDirs, RunManifest, RunSpec, TrainingSpec

__all__ = [
    "ArtifactSpec",
    "DataSpec",
    "DatasetManifest",
    "EvalSpec",
    "GenerationSpec",
    "MeshSpec",
    "ModelSpec",
    "OptimizerSpec",
    "ParamRouteRule",
    "RunDirs",
    "RunManifest",
    "RunSpec",
    "ScheduleSpec",
    "ShardInfo",
    "TrainingSpec",
]
