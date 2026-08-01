"""Bundled demo data for zero-setup trial (`--demo`)."""

from pathlib import Path

DEMO_TRACES_PATH = Path(__file__).with_name("sample_traces.jsonl")


def demo_path() -> Path:
    """Absolute path to the bundled demo traces file."""
    return DEMO_TRACES_PATH
