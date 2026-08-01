"""Pluggable trace importers with format auto-detection."""

from .registry import detect_format, load

__all__ = ["detect_format", "load"]
