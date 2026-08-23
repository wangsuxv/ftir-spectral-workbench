"""Shared contracts and validation helpers for baseline estimators.

The public estimators accept either a single spectrum (``n_points,``) or a
series (``n_spectra, n_points``).  The pipeline normally uses the latter, but
supporting a single spectrum makes the scientific core convenient to test and
reuse without changing its numerical behaviour.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from ..models import BaselineResult

Component = Literal["coarse", "fine"]


@runtime_checkable
class BaselineEstimator(Protocol):
    """Structural interface implemented by estimator wrapper classes."""

    def fit_transform(self, x: np.ndarray, spectra: np.ndarray) -> BaselineResult:
        """Estimate a baseline and return all reconstruction components."""
        ...


def validate_xy(x: np.ndarray, spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Validate baseline input and return ``(x, spectra_2d, was_1d)``.

    A monotonic axis is mandatory.  Algorithms are allowed to sort a descending
    axis internally, but every public result is restored to the input direction.
    """

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(spectra, dtype=np.float64)
    if x_array.ndim != 1:
        raise ValueError("wavenumber x must be a one-dimensional array")
    if x_array.size < 2:
        raise ValueError("at least two wavenumber points are required")
    if not np.all(np.isfinite(x_array)):
        bad_x_indices = np.flatnonzero(~np.isfinite(x_array)).tolist()
        raise ValueError(f"wavenumber x contains NaN or Inf at indices {bad_x_indices}")

    differences = np.diff(x_array)
    if np.any(differences == 0):
        duplicates = np.flatnonzero(differences == 0).tolist()
        raise ValueError(f"wavenumber x contains duplicate adjacent values at indices {duplicates}")
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError("wavenumber x must be strictly monotonic")

    was_1d = y_array.ndim == 1
    if was_1d:
        y_array = y_array[np.newaxis, :]
    elif y_array.ndim != 2:
        raise ValueError("spectra must have shape (n_points,) or (n_spectra, n_points)")
    if y_array.shape[0] == 0:
        raise ValueError("spectra must contain at least one spectrum")
    if y_array.shape[1] != x_array.size:
        raise ValueError(
            f"spectra point count does not match x: {y_array.shape[1]} != {x_array.size}"
        )
    if not np.all(np.isfinite(y_array)):
        bad_spectrum_indices = np.argwhere(~np.isfinite(y_array)).tolist()
        raise ValueError(f"spectra contain NaN or Inf at indices {bad_spectrum_indices}")

    # Explicit copies prevent estimators or third-party libraries from mutating
    # caller-owned arrays and guarantee float64 throughout the scientific core.
    return x_array.copy(), y_array.copy(), was_1d


def ascending_view(x: np.ndarray, spectra_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return ascending-axis views plus whether results need reversing."""

    reverse = bool(x[0] > x[-1])
    if reverse:
        return x[::-1].copy(), spectra_2d[:, ::-1].copy(), True
    return x.copy(), spectra_2d.copy(), False


def restore_axis(values: np.ndarray, reverse: bool) -> np.ndarray:
    """Restore an internally ascending array to the caller's axis direction."""

    array = np.asarray(values, dtype=np.float64)
    return array[..., ::-1].copy() if reverse else array.copy()


def restore_dimensionality(values_2d: np.ndarray, was_1d: bool) -> np.ndarray:
    """Restore the input's one- versus two-dimensional convention."""

    array = np.asarray(values_2d, dtype=np.float64)
    return array[0].copy() if was_1d else array.copy()


def make_result(
    spectra_2d: np.ndarray,
    baseline_2d: np.ndarray,
    *,
    component: Component,
    was_1d: bool,
    params: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    warnings: tuple[str, ...] | list[str] = (),
) -> BaselineResult:
    """Construct a :class:`BaselineResult` for one baseline component."""

    spectra_array = np.asarray(spectra_2d, dtype=np.float64)
    baseline_array = np.asarray(baseline_2d, dtype=np.float64)
    if baseline_array.shape != spectra_array.shape:
        raise ValueError(
            "estimated baseline shape does not match spectra: "
            f"{baseline_array.shape} != {spectra_array.shape}"
        )
    if not np.all(np.isfinite(baseline_array)):
        bad = np.argwhere(~np.isfinite(baseline_array)).tolist()
        raise ValueError(f"estimated baseline contains NaN or Inf at indices {bad}")

    zeros = np.zeros_like(baseline_array, dtype=np.float64)
    if component == "coarse":
        coarse, fine = baseline_array, zeros
    elif component == "fine":
        coarse, fine = zeros, baseline_array
    else:  # pragma: no cover - protected by the Literal type
        raise ValueError(f"unsupported component: {component!r}")

    return BaselineResult(
        coarse_baseline=restore_dimensionality(coarse, was_1d),
        fine_baseline=restore_dimensionality(fine, was_1d),
        total_baseline=restore_dimensionality(baseline_array, was_1d),
        corrected=restore_dimensionality(spectra_array - baseline_array, was_1d),
        params=dict(params or {}),
        metrics=dict(metrics or {}),
        warnings=tuple(dict.fromkeys(str(item) for item in warnings)),
    )


def serializable_value(value: Any) -> Any:
    """Convert pybaselines metadata to JSON-friendly Python values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): serializable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable_value(item) for item in value]
    return value
