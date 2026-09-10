"""Build independent singleton recipes without changing scientific defaults."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ftir_baseline.config import CoarseBaselineConfig, FineBaselineConfig, PipelineConfig
from ftir_baseline.ranges import select_array_range

from .models import BatchError, PreparationConfig, SpectrumRecord


def disabled_fine() -> FineBaselineConfig:
    """One canonical disabled recipe, independent of every fine draft field."""

    return FineBaselineConfig(enabled=False, method="none")


def require_independent_config(config: PipelineConfig | Mapping[str, Any]) -> PipelineConfig:
    checked = PipelineConfig(**(config.to_dict() if isinstance(config, PipelineConfig) else config))
    if checked.series_mode != "independent_locked":
        raise BatchError("INCOMPATIBLE_RECIPE", "ordinary spectra require independent_locked")
    if checked.normalization.method != "none":
        raise BatchError("INCOMPATIBLE_RECIPE", "ordinary spectra require normalization=none")
    return checked


def coarse_config(preparation: PreparationConfig, coarse: CoarseBaselineConfig) -> PipelineConfig:
    return PipelineConfig(
        **preparation.to_dict(),
        coarse_baseline=CoarseBaselineConfig(**coarse.to_dict()),
        fine_baseline=disabled_fine(),
        normalization={"method": "none"},
        series_mode="independent_locked",
    )


def fine_config(parent_config: PipelineConfig, fine: FineBaselineConfig) -> PipelineConfig:
    parent = require_independent_config(parent_config)
    payload = parent.to_dict()
    payload["fine_baseline"] = fine.to_dict()
    return PipelineConfig(**payload)


def preparation_from_config(config: PipelineConfig) -> PreparationConfig:
    return PreparationConfig(**{key: config.to_dict()[key] for key in (
        "input_unit", "wavenumber_range", "baseline_smoothing", "transmittance_floor"
    )})


def validate_preparation(record: SpectrumRecord, preparation: PreparationConfig) -> np.ndarray:
    try:
        x, _ = select_array_range(
            record.wavenumber, record.raw_intensity, preparation.wavenumber_range,
            strict_bounds=True,
        )
    except ValueError as exc:
        raise BatchError("RANGE_INVALID", str(exc)) from exc
    smooth = preparation.baseline_smoothing
    if smooth.enabled and smooth.window_length > x.size:
        raise BatchError("RANGE_INVALID", "smoothing window exceeds selected point count")
    return x


def validate_fine(record: SpectrumRecord, preparation: PreparationConfig,
                  fine: FineBaselineConfig) -> None:
    x = validate_preparation(record, preparation)
    if not fine.enabled or fine.method in {"none", "endpoint_window_linear"}:
        return
    anchors = [anchor for anchor in fine.anchors if anchor.enabled]
    required = fine.polynomial_order + 1 if fine.method == "polynomial" else 2
    if len(anchors) < required:
        raise BatchError("ANCHOR_OUT_OF_RANGE", f"method requires at least {required} anchors")
    centers = np.array([(anchor.start + anchor.end) / 2 for anchor in anchors])
    if not (np.all(np.diff(centers) > 0) or np.all(np.diff(centers) < 0)):
        raise BatchError("ANCHOR_OUT_OF_RANGE", "anchor centers must be strictly ordered")
    low_x, high_x = float(np.min(x)), float(np.max(x))
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(low_x), abs(high_x)) * 16
    if centers.min() > low_x + tolerance or centers.max() < high_x - tolerance:
        raise BatchError("ANCHOR_OUT_OF_RANGE", "anchor centers must cover the selected domain")
    ordered = sorted((min(a.start, a.end), max(a.start, a.end)) for a in anchors)
    for i, (low, high) in enumerate(ordered):
        if not np.any((x >= low - tolerance) & (x <= high + tolerance)):
            raise BatchError("ANCHOR_OUT_OF_RANGE", f"anchor {i + 1} contains no selected points")
        if i and ordered[i - 1][1] >= low:
            raise BatchError("ANCHOR_OUT_OF_RANGE", "anchor windows may not overlap or touch")

