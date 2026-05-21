"""Model runtimes and build contracts."""

from jaxtitan.models.decoder import (
    DecoderModel,
    ModelBuildResult,
    ParamMetadata,
    ParamLayout,
    apply_model,
    apply_model_output,
    build_model,
    count_parameters,
    decode_model,
    dtype_from_name,
    prefill_model,
)
from jaxtitan.models.output import AuxLoss, AuxMetric, ModelOutput, RouterStats
from jaxtitan.models.trinity import TrinityModel

__all__ = [
    "AuxLoss",
    "AuxMetric",
    "DecoderModel",
    "ModelOutput",
    "ModelBuildResult",
    "ParamMetadata",
    "ParamLayout",
    "RouterStats",
    "apply_model",
    "apply_model_output",
    "build_model",
    "count_parameters",
    "decode_model",
    "dtype_from_name",
    "prefill_model",
    "TrinityModel",
]
