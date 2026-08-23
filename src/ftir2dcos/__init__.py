"""FTIR preprocessing, 2D-COS, and apparent peak-response ordering."""

from __future__ import annotations

from .peak_order import PeakOrderResult, PeakRequest

__version__ = "0.4.0"


def run_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Import the pipeline lazily so importing package metadata stays lightweight."""
    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


def run_multi_range_pipeline(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Import the multi-range pipeline lazily."""
    from .pipeline import run_multi_range_pipeline as _run_multi_range_pipeline

    return _run_multi_range_pipeline(*args, **kwargs)


def analyze_multi_range_peak_order(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Import the post-hoc multi-range peak-order analyzer lazily."""
    from .pipeline import analyze_multi_range_peak_order as _analyze

    return _analyze(*args, **kwargs)


__all__ = [
    "PeakOrderResult",
    "PeakRequest",
    "__version__",
    "analyze_multi_range_peak_order",
    "run_multi_range_pipeline",
    "run_pipeline",
]
