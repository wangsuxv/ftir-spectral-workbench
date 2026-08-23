"""Transparent diagnostic metrics for comparing candidate baselines.

The scores in this module are diagnostics, not evidence that a candidate is
the unknown physical baseline.  Each component remains available separately
so callers can display the trade-offs instead of hiding them in one number.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_2d(values: ArrayLike, name: str) -> tuple[FloatArray, bool]:
    array = np.asarray(values, dtype=np.float64)
    was_1d = array.ndim == 1
    if was_1d:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one wavenumber point")
    if not np.all(np.isfinite(array)):
        index = np.argwhere(~np.isfinite(array))[0]
        raise ValueError(
            f"{name} contains NaN or Inf at spectrum {int(index[0])}, point {int(index[1])}"
        )
    return array, was_1d


def _axis(wavenumber: ArrayLike, n_points: int) -> FloatArray:
    x = np.asarray(wavenumber, dtype=np.float64)
    if x.ndim != 1 or x.size != n_points:
        raise ValueError(f"wavenumber must have shape ({n_points},)")
    if not np.all(np.isfinite(x)):
        raise ValueError("wavenumber contains NaN or Inf")
    if x.size > 1 and not (np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)):
        raise ValueError("wavenumber must be strictly monotonic")
    return x


def _restore(values: FloatArray, was_1d: bool) -> FloatArray | np.float64:
    return np.float64(values[0]) if was_1d else values


def estimate_noise_sigma(spectra: ArrayLike) -> FloatArray | np.float64:
    """Estimate Gaussian noise using the MAD of first differences.

    Differencing removes constants and most slow baseline drift.  The factor
    ``sqrt(2)`` accounts for differencing two independent noisy samples.
    """

    values, was_1d = _as_2d(spectra, "spectra")
    if values.shape[1] < 2:
        sigma = np.zeros(values.shape[0], dtype=np.float64)
    else:
        differences = np.diff(values, axis=1)
        centres = np.median(differences, axis=1, keepdims=True)
        mad = np.median(np.abs(differences - centres), axis=1)
        sigma = mad / (0.6744897501960817 * np.sqrt(2.0))
    return _restore(sigma, was_1d)


def _window_bounds(window: Any) -> tuple[float, float] | None:
    if isinstance(window, Mapping):
        if not bool(window.get("enabled", window.get("Enabled", True))):
            return None
        start = window.get("start", window.get("Start"))
        end = window.get("end", window.get("End"))
        if start is None or end is None:
            raise ValueError("anchor window mappings require start and end")
    elif hasattr(window, "start") and hasattr(window, "end"):
        if hasattr(window, "enabled") and not bool(window.enabled):
            return None
        start, end = window.start, window.end
    else:
        if len(window) != 2:
            raise ValueError("anchor windows must contain two bounds")
        start, end = window
    lo, hi = sorted((float(start), float(end)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        raise ValueError("anchor window bounds must be finite and distinct")
    return lo, hi


def anchor_mask(
    wavenumber: ArrayLike,
    anchor_windows: Sequence[Any],
) -> NDArray[np.bool_]:
    """Return the union of enabled fixed anchor windows."""

    x = np.asarray(wavenumber, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("wavenumber must be one-dimensional")
    mask = np.zeros(x.size, dtype=bool)
    enabled = 0
    for window in anchor_windows:
        bounds = _window_bounds(window)
        if bounds is None:
            continue
        enabled += 1
        lo, hi = bounds
        current = (x >= lo) & (x <= hi)
        if not np.any(current):
            raise ValueError(f"anchor window [{lo:g}, {hi:g}] contains no data points")
        mask |= current
    if enabled == 0:
        raise ValueError("at least one enabled anchor window is required")
    return mask


def anchor_residual_error(
    wavenumber: ArrayLike,
    corrected: ArrayLike,
    anchor_windows: Sequence[Any],
) -> FloatArray | np.float64:
    """Median absolute corrected absorbance inside fixed anchor windows."""

    values, was_1d = _as_2d(corrected, "corrected")
    x = _axis(wavenumber, values.shape[1])
    mask = anchor_mask(x, anchor_windows)
    result = np.median(np.abs(values[:, mask]), axis=1)
    return _restore(result, was_1d)


def negative_residual_fraction(
    corrected: ArrayLike,
    noise_sigma: ArrayLike | None = None,
    *,
    k: float = 3.0,
) -> FloatArray | np.float64:
    """Fraction of points below ``-k * noise_sigma`` for each spectrum."""

    values, was_1d = _as_2d(corrected, "corrected")
    k = float(k)
    if not np.isfinite(k) or k < 0:
        raise ValueError("k must be finite and non-negative")
    if noise_sigma is None:
        sigma = np.atleast_1d(estimate_noise_sigma(values)).astype(np.float64)
    else:
        sigma = np.asarray(noise_sigma, dtype=np.float64)
        if sigma.ndim == 0:
            sigma = np.full(values.shape[0], float(sigma), dtype=np.float64)
        if sigma.shape != (values.shape[0],):
            raise ValueError(f"noise_sigma must be scalar or have shape ({values.shape[0]},)")
        if np.any(~np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError("noise_sigma must be finite and non-negative")
    result = np.mean(values < (-k * sigma[:, np.newaxis]), axis=1, dtype=np.float64)
    return _restore(result, was_1d)


negative_fraction = negative_residual_fraction


def baseline_roughness(baseline: ArrayLike) -> FloatArray | np.float64:
    """Mean squared second difference along the wavenumber axis."""

    values, was_1d = _as_2d(baseline, "baseline")
    if values.shape[1] < 3:
        result = np.zeros(values.shape[0], dtype=np.float64)
    else:
        result = np.mean(np.diff(values, n=2, axis=1) ** 2, axis=1)
    return _restore(np.asarray(result, dtype=np.float64), was_1d)


def temporal_roughness(baselines: ArrayLike) -> float:
    """Mean squared second difference between adjacent sequence baselines."""

    values, _ = _as_2d(baselines, "baselines")
    if values.shape[0] < 3:
        return 0.0
    return float(np.mean(np.diff(values, n=2, axis=0) ** 2))


time_roughness = temporal_roughness


def reconstruction_error(
    raw_absorbance: ArrayLike,
    total_baseline: ArrayLike,
    corrected_absorbance: ArrayLike,
) -> FloatArray | np.float64:
    """Maximum absolute ``raw - baseline - corrected`` error per spectrum."""

    raw, was_1d = _as_2d(raw_absorbance, "raw_absorbance")
    baseline, _ = _as_2d(total_baseline, "total_baseline")
    corrected, _ = _as_2d(corrected_absorbance, "corrected_absorbance")
    if baseline.shape != raw.shape or corrected.shape != raw.shape:
        raise ValueError("raw_absorbance, total_baseline, and corrected_absorbance must match")
    result = np.max(np.abs(raw - baseline - corrected), axis=1)
    return _restore(result, was_1d)


def reconstruction_check(
    raw_absorbance: ArrayLike,
    total_baseline: ArrayLike,
    corrected_absorbance: ArrayLike,
    *,
    atol: float | None = None,
    rtol: float = 1e-12,
) -> bool:
    """Return whether reconstruction agrees to floating-point precision."""

    raw = np.asarray(raw_absorbance, dtype=np.float64)
    baseline = np.asarray(total_baseline, dtype=np.float64)
    corrected = np.asarray(corrected_absorbance, dtype=np.float64)
    if raw.shape != baseline.shape or raw.shape != corrected.shape:
        raise ValueError("raw_absorbance, total_baseline, and corrected_absorbance must match")
    if atol is None:
        scale = max(1.0, float(np.max(np.abs(raw))) if raw.size else 1.0)
        atol = 32.0 * np.finfo(np.float64).eps * scale
    return bool(np.allclose(raw, baseline + corrected, atol=float(atol), rtol=float(rtol)))


def _region_masks(
    x: FloatArray,
    peak_regions: Sequence[Any] | None,
) -> list[NDArray[np.bool_]]:
    if peak_regions is None:
        return [np.ones(x.size, dtype=bool)]
    masks: list[NDArray[np.bool_]] = []
    for region in peak_regions:
        bounds = _window_bounds(region)
        if bounds is None:
            continue
        lo, hi = bounds
        mask = (x >= lo) & (x <= hi)
        if np.count_nonzero(mask) < 3:
            raise ValueError(f"peak region [{lo:g}, {hi:g}] contains fewer than three points")
        masks.append(mask)
    if not masks:
        raise ValueError("at least one enabled peak region is required")
    return masks


def _correlation(left: FloatArray, right: FloatArray) -> float:
    left_c = left - np.mean(left)
    right_c = right - np.mean(right)
    denominator = float(np.linalg.norm(left_c) * np.linalg.norm(right_c))
    if denominator <= np.finfo(np.float64).tiny:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.clip(np.dot(left_c, right_c) / denominator, -1.0, 1.0))


def peak_preservation(
    wavenumber: ArrayLike,
    raw_absorbance: ArrayLike,
    corrected_absorbance: ArrayLike,
    peak_regions: Sequence[Any] | None = None,
) -> dict[str, FloatArray | np.float64]:
    """Compare derivative shape and peak locations before/after correction.

    With explicit peak regions, the position of the maximum signal in every
    region is compared.  Without regions, the strongest negative second-
    derivative feature is used as a reproducible proxy for the dominant peak.
    """

    raw, was_1d = _as_2d(raw_absorbance, "raw_absorbance")
    corrected, _ = _as_2d(corrected_absorbance, "corrected_absorbance")
    if raw.shape != corrected.shape:
        raise ValueError("raw_absorbance and corrected_absorbance must have matching shapes")
    x = _axis(wavenumber, raw.shape[1])
    order = np.argsort(x)
    xs = x[order]
    raw_s = raw[:, order]
    corrected_s = corrected[:, order]
    masks = _region_masks(xs, peak_regions)

    correlations = np.empty(raw.shape[0], dtype=np.float64)
    shifts = np.empty(raw.shape[0], dtype=np.float64)
    height_bias = np.empty(raw.shape[0], dtype=np.float64)
    for row in range(raw.shape[0]):
        if xs.size < 2:
            correlations[row] = 1.0
        else:
            raw_derivative = np.gradient(raw_s[row], xs)
            corrected_derivative = np.gradient(corrected_s[row], xs)
            correlations[row] = _correlation(raw_derivative, corrected_derivative)

        region_shifts: list[float] = []
        region_height_biases: list[float] = []
        for mask in masks:
            indices = np.flatnonzero(mask)
            if peak_regions is None and indices.size >= 3:
                raw_curvature = -np.gradient(np.gradient(raw_s[row], xs), xs)
                corrected_curvature = -np.gradient(np.gradient(corrected_s[row], xs), xs)
                raw_index = indices[int(np.argmax(raw_curvature[indices]))]
                corrected_index = indices[int(np.argmax(corrected_curvature[indices]))]
            else:
                raw_index = indices[int(np.argmax(raw_s[row, indices]))]
                corrected_index = indices[int(np.argmax(corrected_s[row, indices]))]
            region_shifts.append(abs(float(xs[raw_index] - xs[corrected_index])))
            raw_height = float(raw_s[row, raw_index])
            corrected_height = float(corrected_s[row, corrected_index])
            scale = max(abs(raw_height), np.finfo(np.float64).eps)
            region_height_biases.append((corrected_height - raw_height) / scale)
        shifts[row] = float(np.median(region_shifts))
        height_bias[row] = float(np.median(region_height_biases))

    correlation_out = _restore(correlations, was_1d)
    shift_out = _restore(shifts, was_1d)
    height_out = _restore(height_bias, was_1d)
    penalties = 1.0 - np.clip(correlations, -1.0, 1.0)
    return {
        "derivative_correlation": correlation_out,
        "peak_position_shift": shift_out,
        "peak_height_relative_change": height_out,
        "peak_change_penalty": _restore(penalties, was_1d),
    }


DEFAULT_DIAGNOSTIC_WEIGHTS: dict[str, float] = {
    "anchor_error": 1.0,
    "negative_penalty": 1.0,
    "negative_fraction": 1.0,
    "baseline_roughness": 1.0,
    "time_roughness": 1.0,
    "peak_change_penalty": 1.0,
}


def diagnostic_score(
    metrics: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Compute an optional transparent weighted candidate score.

    Missing and non-finite components are omitted.  Callers should present
    component metrics beside this value and label it as a ranking heuristic.
    """

    effective = dict(DEFAULT_DIAGNOSTIC_WEIGHTS if weights is None else weights)
    total = 0.0
    used = 0
    for name, weight in effective.items():
        if name not in metrics:
            continue
        values = np.asarray(metrics[name], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        coefficient = float(weight)
        if not np.isfinite(coefficient) or coefficient < 0:
            raise ValueError(f"weight for {name!r} must be finite and non-negative")
        total += coefficient * float(np.mean(finite))
        used += 1
    if used == 0:
        return float("nan")
    return total


@dataclass(frozen=True)
class RankedCandidate:
    name: str
    score: float
    metrics: Mapping[str, Any]


def rank_candidates(
    candidates: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float] | None = None,
) -> tuple[RankedCandidate, ...]:
    """Rank candidates while preserving every diagnostic component."""

    ranked = [
        RankedCandidate(name=name, score=diagnostic_score(metrics, weights), metrics=metrics)
        for name, metrics in candidates.items()
    ]
    ranked.sort(key=lambda item: (not np.isfinite(item.score), item.score, item.name))
    return tuple(ranked)


score_candidate = diagnostic_score


__all__ = [
    "DEFAULT_DIAGNOSTIC_WEIGHTS",
    "RankedCandidate",
    "anchor_mask",
    "anchor_residual_error",
    "baseline_roughness",
    "diagnostic_score",
    "estimate_noise_sigma",
    "negative_fraction",
    "negative_residual_fraction",
    "peak_preservation",
    "rank_candidates",
    "reconstruction_check",
    "reconstruction_error",
    "score_candidate",
    "temporal_roughness",
    "time_roughness",
]
