"""Prepared-data contracts."""

from __future__ import annotations

from jaxtitan.data.manifest import (
    PreparedDatasetManifest,
    TokenShard,
    TokenSplit,
    dataset_manifest_sha256,
    dataset_manifest_summary,
    load_dataset_manifest,
    prepared_dataset_manifest_to_dict,
    prepared_dataset_manifest_to_json,
    validate_dataset_manifest,
)
from jaxtitan.data.service import BatchProvenance, PreparedDataService, read_token_range

__all__ = [
    "BatchProvenance",
    "PreparedDatasetManifest",
    "PreparedDataService",
    "TokenShard",
    "TokenSplit",
    "dataset_manifest_sha256",
    "dataset_manifest_summary",
    "load_dataset_manifest",
    "prepared_dataset_manifest_to_dict",
    "prepared_dataset_manifest_to_json",
    "read_token_range",
    "validate_dataset_manifest",
]
