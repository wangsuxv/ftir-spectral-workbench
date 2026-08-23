"""Estimate-only smoothing for baseline fitting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_filter

from .config import SmoothingConfig
from .models import immutable_float64
from .validation import SpectrumValidationError, as_float64_array, require_finite

FloatArray = NDArray[np.float64]


def validate_savgol_parameters(
    n_points: int,
    *,
    window_length: int,
    polyorder: int,
) -> None:
    """Validate all SG constraints before calling SciPy."""

    if isinstance(window_length, bool) or not isinstance(window_length, (int, np.integer)):
        raise ValueError("Savitzky-Golay window_length must be an integer")
    if isinstance(polyorder, bool) or not isinstance(polyorder, (int, np.integer)):
        raise ValueError("Savitzky-Golay polyorder must be an integer")
    if window_length < 3:
        raise ValueError("Savitzky-Golay window_length must be at least 3")
    if window_length % 2 == 0:
        raise ValueError("Savitzky-Golay window_length must be odd")
    if polyorder < 0:
        raise ValueError("Savitzky-Golay polyorder must be non-negative")
    if window_length <= polyorder:
        raise ValueError("Savitzky-Golay window_length must be greater than polyorder")
    if window_length > n_points:
        raise ValueError(
            f"Savitzky-Golay window_length ({window_length}) exceeds spectral point "
            f"count ({n_points})"
        )


def savgol_estimate_only(
    spectra: ArrayLike,
    *,
    window_length: int = 7,
    polyorder: int = 2,
) -> FloatArray:
    """Return an SG-smoothed float64 copy along the final (wavenumber) axis."""

    values = as_float64_array(spectra, name="spectra", ndim=(1, 2), copy=True)
    require_finite(values, name="spectra")
    validate_savgol_parameters(values.shape[-1], window_length=window_length, polyorder=polyorder)
    smoothed = savgol_filter(
        values,
        window_length=window_length,
        polyorder=polyorder,
        axis=-1,
        mode="interp",
    )
    require_finite(smoothed, name="smoothed baseline-estimation spectra")
    return immutable_float64(smoothed, name="smoothed baseline-estimation spectra")


@dataclass(frozen=True, slots=True)
class BaselineEstimationChannels:
    """Separate immutable raw and baseline-estimation channels."""

    raw: FloatArray
    for_baseline: FloatArray
    settings: Mapping[str, Any]

    def __post_init__(self) -> None:
        raw = as_float64_array(self.raw, name="raw", ndim=(1, 2), copy=True)
        estimate = as_float64_array(self.for_baseline, name="for_baseline", ndim=(1, 2), copy=True)
        if raw.shape != estimate.shape:
            raise SpectrumValidationError(
                f"raw and for_baseline shapes differ: {raw.shape} vs {estimate.shape}"
            )
        require_finite(raw, name="raw")
        require_finite(estimate, name="for_baseline")
        object.__setattr__(self, "raw", immutable_float64(raw, name="raw"))
        object.__setattr__(self, "for_baseline", immutable_float64(estimate, name="for_baseline"))
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))


def prepare_baseline_channels(
    spectra: ArrayLike,
    config: SmoothingConfig | None = None,
    *,
    enabled: bool | None = None,
    window_length: int = 7,
    polyorder: int = 2,
    estimate_only: bool = True,
) -> BaselineEstimationChannels:
    """Prepare raw and optional SG baseline-estimation channels.

    Passing a ``SmoothingConfig`` is preferred.  Explicit keyword arguments are
    retained for lightweight programmatic callers.  The raw channel is always a
    detached copy and is never replaced by the smoothed channel.
    """

    raw = as_float64_array(spectra, name="spectra", ndim=(1, 2), copy=True)
    require_finite(raw, name="spectra")
    if config is not None:
        if enabled is not None:
            raise ValueError("do not pass enabled together with SmoothingConfig")
        active = config.enabled
        window_length = config.window_length
        polyorder = config.polyorder
        estimate_only = config.estimate_only
        method = config.method
    else:
        active = bool(enabled) if enabled is not None else False
        method = "savgol"
    if active and not estimate_only:
        raise ValueError("smoothing may only be used to estimate the baseline")

    if active:
        estimate = savgol_estimate_only(raw, window_length=window_length, polyorder=polyorder)
    else:
        estimate = raw.copy()
    settings = {
        "enabled": active,
        "method": method,
        "window_length": int(window_length),
        "polyorder": int(polyorder),
        "estimate_only": True,
        "axis": "wavenumber",
    }
    return BaselineEstimationChannels(raw=raw, for_baseline=estimate, settings=settings)


def smooth_for_baseline(
    spectra: ArrayLike,
    config: SmoothingConfig,
) -> FloatArray:
    """Return only the estimation channel for pipeline adapters."""

    return prepare_baseline_channels(spectra, config).for_baseline


apply_baseline_smoothing = smooth_for_baseline


__all__ = [
    "BaselineEstimationChannels",
    "apply_baseline_smoothing",
    "prepare_baseline_channels",
    "savgol_estimate_only",
    "smooth_for_baseline",
    "validate_savgol_parameters",
]
