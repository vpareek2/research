"""Config loading and validation."""

from __future__ import annotations

from jaxtitan.config.load import load_config, run_spec_from_mapping
from jaxtitan.config.validate import validate_run_spec

__all__ = ["load_config", "run_spec_from_mapping", "validate_run_spec"]
