"""Shared-shape baseline model for related in-situ spectra."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..models import BaselineResult
from .anchors import _anchor_values, _prepare_windows
from .automatic import estimate_coarse
from .protocol import ascending_view, restore_axis, restore_dimensionality, validate_xy

ReferenceStatistic = Literal["median", "mean"]


def shared_shape_baseline(
    x: np.ndarray,
    spectra: np.ndarray,
    reference_method: str = "arpls",
    *,
    anchors: Sequence[Any],
    reference: ReferenceStatistic | str = "median",
    anchor_statistic: str = "median",
    reference_params: dict[str, Any] | None = None,
    **method_params: Any,
) -> BaselineResult:
    """Fit one common curve plus only an offset and slope per spectrum.

    ``B_ref`` is estimated from the median or mean spectrum.  For every input
    row, the residual anchor values are then projected onto the fixed two-column
    design ``[1, x]``.  Consequently, per-spectrum freedom is exactly two
    coefficients and cannot bend independently around peaks.
    """

    x_array, spectra_2d, was_1d = validate_xy(x, spectra)
    normalized_reference = str(reference).strip().lower()
    if normalized_reference == "median":
        reference_spectrum = np.median(spectra_2d, axis=0)
    elif normalized_reference == "mean":
        reference_spectrum = np.mean(spectra_2d, axis=0)
    else:
        raise ValueError("shared-shape reference must be 'median' or 'mean'")

    if reference_params is not None and method_params:
        duplicate = set(reference_params).intersection(method_params)
        if duplicate:
            raise ValueError(f"duplicate reference baseline parameters: {sorted(duplicate)}")
    combined_reference_params = dict(reference_params or {})
    combined_reference_params.update(method_params)
    reference_result = estimate_coarse(
        x_array,
        reference_spectrum,
        reference_method,
        **combined_reference_params,
    )
    reference_baseline = np.asarray(reference_result.total_baseline, dtype=np.float64)
    if reference_baseline.shape != x_array.shape:
        raise RuntimeError(
            "reference baseline estimator returned an unexpected shape: "
            f"{reference_baseline.shape} != {x_array.shape}"
        )

    residuals = spectra_2d - reference_baseline[None, :]
    x_ascending, residuals_ascending, reverse = ascending_view(x_array, residuals)
    active, centers, bounds = _prepare_windows(anchors, anchor_statistic)
    anchor_values, point_counts = _anchor_values(x_ascending, residuals_ascending, active, bounds)

    center = float(np.mean(centers))
    scale = float(np.ptp(centers) / 2.0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("shared-shape anchors must define at least two distinct locations")
    design_anchors = np.column_stack((np.ones_like(centers), (centers - center) / scale))
    normalized_x = (x_ascending - center) / scale
    design_full = np.column_stack((np.ones_like(normalized_x), normalized_x))
    normalized_coefficients = np.linalg.lstsq(design_anchors, anchor_values.T, rcond=None)[0]
    adjustment_ascending = (design_full @ normalized_coefficients).T
    adjustment = restore_axis(adjustment_ascending, reverse)
    common = np.broadcast_to(reference_baseline[None, :], spectra_2d.shape).copy()
    total = common + adjustment

    slopes = normalized_coefficients[1] / scale
    offsets = normalized_coefficients[0] - slopes * center
    windows_recipe = [
        {
            "start": float(window.start),
            "end": float(window.end),
            "statistic": str(window.statistic),
            "enabled": True,
        }
        for window in active
    ]
    return BaselineResult(
        coarse_baseline=restore_dimensionality(common, was_1d),
        fine_baseline=restore_dimensionality(adjustment, was_1d),
        total_baseline=restore_dimensionality(total, was_1d),
        corrected=restore_dimensionality(spectra_2d - total, was_1d),
        params={
            "method": "shared_shape",
            "reference": normalized_reference,
            "reference_method": reference_method,
            "reference_recipe": dict(reference_result.params),
            "anchors": windows_recipe,
            "anchor_centers": centers.tolist(),
            "anchor_point_counts": point_counts,
            "degrees_of_freedom_per_spectrum": 2,
            "allowed_per_spectrum_terms": ["constant", "linear_slope"],
            "fitted_offsets": offsets.tolist(),
            "fitted_slopes": slopes.tolist(),
            "series_recipe_locked": True,
        },
        metrics={
            "reference_metrics": dict(reference_result.metrics),
            "residual_anchor_values": anchor_values.tolist(),
        },
        warnings=tuple(reference_result.warnings),
    )


shared_shape = shared_shape_baseline


@dataclass(frozen=True)
class SharedShapeEstimator:
    """Reusable shared-shape series recipe."""

    anchors: tuple[Any, ...]
    reference_method: str = "arpls"
    reference: ReferenceStatistic = "median"
    anchor_statistic: str = "median"
    reference_params: dict[str, Any] = field(default_factory=dict)

    def fit_transform(self, x: np.ndarray, spectra: np.ndarray) -> BaselineResult:
        return shared_shape_baseline(
            x,
            spectra,
            self.reference_method,
            anchors=self.anchors,
            reference=self.reference,
            anchor_statistic=self.anchor_statistic,
            reference_params=self.reference_params,
        )
