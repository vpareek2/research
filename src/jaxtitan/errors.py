"""Jaxtitan exception types."""

from __future__ import annotations


class JaxtitanError(Exception):
    """Base class for Jaxtitan errors."""


class ConfigError(JaxtitanError):
    """Raised when TOML configuration cannot be resolved into a valid spec."""


class ContractError(JaxtitanError):
    """Raised when a contract object is internally inconsistent."""
