"""Two-endpoint baselines for an already selected spectral interval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..models import BaselineResult
from .protocol import make_result, validate_xy

EndpointStatistic = Literal["median", "mean"]


def _statistic(values: np.ndarray, name: str, *, context: str) -> float:
    normalized = name.strip().lower()
    if normalized == "median":
        return float(np.median(values))
    if normalized == "mean":
        return float(np.mean(values))
    raise ValueError(f"unsupported {context} statistic {name!r}; use 'median' or 'mean'")


def _endpoint_window_baseline_2d(
    x: np.ndarray,
    spectra_2d: np.ndarray,
    *,
    window_width_cm1: float,
    statistic: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    width = float(window_width_cm1)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("endpoint window width must be a finite positive number")

    low = float(np.min(x))
    high = float(np.max(x))
    half_width = width / 2.0
    # The selected interval truncates the nominal +/- half-width windows at its
    # boundaries.  Include a tiny tolerance so a boundary sample is never lost
    # to binary floating-point rounding.
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(low), abs(high), width) * 8
    low_mask = np.abs(x - low) <= half_width + tolerance
    high_mask = np.abs(x - high) <= half_width + tolerance
    if not np.any(low_mask):
        raise ValueError(f"lower endpoint window around {low:g} cm^-1 contains no data points")
    if not np.any(high_mask):
        raise ValueError(f"upper endpoint window around {high:g} cm^-1 contains no data points")

    # The intensity statistic represents the samples in a window, so its x
    # coordinate must represent those same samples.  This makes a truly linear
    # spectrum exactly reconstructable even on an irregular wavenumber grid.
    x_low = _statistic(x[low_mask], statistic, context="endpoint")
    x_high = _statistic(x[high_mask], statistic, context="endpoint")
    if not x_high > x_low:
        raise ValueError(
            "endpoint windows overlap too strongly to define two distinct representative points"
        )

    low_values = np.asarray(
        [_statistic(row[low_mask], statistic, context="endpoint") for row in spectra_2d],
        dtype=np.float64,
    )
    high_values = np.asarray(
        [_statistic(row[high_mask], statistic, context="endpoint") for row in spectra_2d],
        dtype=np.float64,
    )
    fraction = (x - x_low) / (x_high - x_low)
    baselines = low_values[:, None] + (high_values - low_values)[:, None] * fraction
    fitted = {
        "representative_wavenumbers": [x_low, x_high],
        "lower_values": low_values.tolist(),
        "upper_values": high_values.tolist(),
        "lower_point_count": int(np.count_nonzero(low_mask)),
        "upper_point_count": int(np.count_nonzero(high_mask)),
    }
    return np.asarray(baselines, dtype=np.float64), fitted


def endpoint_window_baseline(
    x: np.ndarray,
    spectra: np.ndarray,
    window_width_cm1: float = 8.0,
    statistic: EndpointStatistic | str = "median",
    *,
    endpoint_window_width_cm1: float | None = None,
) -> BaselineResult:
    """Fit a robust line through statistics of the two endpoint windows.

    ``endpoint_window_width_cm1`` is accepted as a config-facing alias for
    ``window_width_cm1``.  The input axis direction and dimensionality are
    preserved in every result array.
    """

    x_array, spectra_2d, was_1d = validate_xy(x, spectra)
    width = (
        float(endpoint_window_width_cm1)
        if endpoint_window_width_cm1 is not None
        else float(window_width_cm1)
    )
    normalized_statistic = str(statistic).strip().lower()
    baselines, fitted = _endpoint_window_baseline_2d(
        x_array,
        spectra_2d,
        window_width_cm1=width,
        statistic=normalized_statistic,
    )
    return make_result(
        spectra_2d,
        baselines,
        component="fine",
        was_1d=was_1d,
        params={
            "method": "endpoint_window_linear",
            "endpoint_window_width_cm1": width,
            "statistic": normalized_statistic,
            "strict_endpoint": False,
            "series_recipe_locked": True,
            "fitted": fitted,
        },
    )


def strict_endpoint_baseline(x: np.ndarray, spectra: np.ndarray) -> BaselineResult:
    """Fit a line through the two exact numeric boundary samples.

    This deliberately noise-sensitive method is exposed separately so callers
    have to opt into strict endpoint behaviour.
    """

    x_array, spectra_2d, was_1d = validate_xy(x, spectra)
    low_index = int(np.argmin(x_array))
    high_index = int(np.argmax(x_array))
    x_low = float(x_array[low_index])
    x_high = float(x_array[high_index])
    y_low = spectra_2d[:, low_index]
    y_high = spectra_2d[:, high_index]
    fraction = (x_array - x_low) / (x_high - x_low)
    baselines = y_low[:, None] + (y_high - y_low)[:, None] * fraction
    return make_result(
        spectra_2d,
        baselines,
        component="fine",
        was_1d=was_1d,
        params={
            "method": "strict_endpoint",
            "strict_endpoint": True,
            "series_recipe_locked": True,
            "endpoint_wavenumbers": [x_low, x_high],
            "endpoint_values": {
                "lower": y_low.tolist(),
                "upper": y_high.tolist(),
            },
        },
        warnings=(
            "strict endpoint mode is sensitive to endpoint noise and real absorption at a boundary",
        ),
    )


# Clear, discoverable aliases used by older examples and external callers.
estimate_endpoint_window = endpoint_window_baseline
estimate_strict_endpoint = strict_endpoint_baseline
endpoint_window = endpoint_window_baseline
strict_endpoint = strict_endpoint_baseline


@dataclass(frozen=True)
class EndpointWindowEstimator:
    """Object-oriented wrapper implementing :class:`BaselineEstimator`."""

    window_width_cm1: float = 8.0
    statistic: EndpointStatistic = "median"
    strict: bool = False

    def fit_transform(self, x: np.ndarray, spectra: np.ndarray) -> BaselineResult:
        if self.strict:
            return strict_endpoint_baseline(x, spectra)
        return endpoint_window_baseline(
            x,
            spectra,
            window_width_cm1=self.window_width_cm1,
            statistic=self.statistic,
        )
