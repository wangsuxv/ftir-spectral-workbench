"""Strict validation helpers for FTIR spectral data.

The workbench deliberately rejects ambiguous axes and non-finite measurements.
Repairing such data belongs in an explicit, separately recorded operation rather
than in a reader or numerical algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

AxisDirection = Literal["ascending", "descending"]
FloatArray = NDArray[np.float64]


class SpectrumValidationError(ValueError):
    """Raised when spectral input violates the workbench data contract."""


@dataclass(frozen=True)
class ValidatedSpectrumArrays:
    """Validated float64 arrays and the detected wavenumber direction."""

    wavenumber: FloatArray
    spectra: FloatArray
    axis_direction: AxisDirection


def _positions(mask: NDArray[np.bool_], *, limit: int = 12) -> str:
    locations = np.argwhere(mask)
    shown = [tuple(int(part) for part in row) for row in locations[:limit]]
    suffix = "" if len(locations) <= limit else f" (first {limit} of {len(locations)})"
    return f"{shown}{suffix}"


def as_float64_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | tuple[int, ...] | None = None,
    copy: bool = True,
) -> FloatArray:
    """Return a C-contiguous float64 array or raise a contextual error."""

    try:
        array = np.array(values, dtype=np.float64, order="C", copy=copy)
    except (TypeError, ValueError) as exc:
        raise SpectrumValidationError(f"{name} must contain only numeric values") from exc

    allowed_ndim = (ndim,) if isinstance(ndim, int) else ndim
    if allowed_ndim is not None and array.ndim not in allowed_ndim:
        expected = " or ".join(str(value) for value in allowed_ndim)
        raise SpectrumValidationError(
            f"{name} must have {expected} dimension(s); got shape {array.shape}"
        )
    return array


def require_finite(values: ArrayLike, *, name: str) -> None:
    """Reject NaN and either sign of infinity, reporting array positions."""

    array = np.asarray(values)
    invalid = ~np.isfinite(array)
    if invalid.any():
        nan_count = int(np.isnan(array).sum())
        inf_count = int(np.isinf(array).sum())
        raise SpectrumValidationError(
            f"{name} contains non-finite values (NaN={nan_count}, Inf={inf_count}) "
            f"at positions {_positions(invalid)}"
        )


def validate_wavenumber(wavenumber: ArrayLike, *, minimum_points: int = 2) -> AxisDirection:
    """Validate a finite, unique, strictly monotonic one-dimensional axis."""

    x = as_float64_array(wavenumber, name="wavenumber", ndim=1)
    if x.size < minimum_points:
        raise SpectrumValidationError(
            f"wavenumber must contain at least {minimum_points} points; got {x.size}"
        )
    require_finite(x, name="wavenumber")

    unique_values, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    repeated_groups = np.flatnonzero(counts > 1)
    if repeated_groups.size:
        details: list[dict[str, object]] = []
        for group in repeated_groups[:12]:
            indices = np.flatnonzero(inverse == group)
            details.append(
                {
                    "value": float(unique_values[group]),
                    "indices": [int(index) for index in indices],
                }
            )
        suffix = "" if repeated_groups.size <= 12 else " (first 12 values shown)"
        raise SpectrumValidationError(f"wavenumber contains duplicate values: {details}{suffix}")

    differences = np.diff(x)
    if np.all(differences > 0.0):
        return "ascending"
    if np.all(differences < 0.0):
        return "descending"

    reversals = np.flatnonzero(np.sign(differences[1:]) != np.sign(differences[:-1])) + 1
    shown = [int(value) for value in reversals[:12]]
    raise SpectrumValidationError(
        f"wavenumber must be strictly monotonic; direction changes near indices {shown}"
    )


def validate_spectrum_arrays(
    wavenumber: ArrayLike,
    spectra: ArrayLike,
    *,
    allow_1d_spectra: bool = True,
) -> ValidatedSpectrumArrays:
    """Validate the central numerical contract and normalize spectra to 2-D."""

    x = as_float64_array(wavenumber, name="wavenumber", ndim=1)
    direction = validate_wavenumber(x)
    allowed = (1, 2) if allow_1d_spectra else 2
    y = as_float64_array(spectra, name="spectra", ndim=allowed)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    if y.shape[0] == 0:
        raise SpectrumValidationError("spectra must contain at least one spectrum")
    if y.shape[1] != x.size:
        raise SpectrumValidationError(
            "spectra point count must match wavenumber: "
            f"spectra.shape={y.shape}, wavenumber.shape={x.shape}"
        )
    require_finite(y, name="spectra")
    return ValidatedSpectrumArrays(x, y, direction)


def validate_perturbation(perturbation: ArrayLike, *, n_spectra: int) -> FloatArray:
    """Validate the perturbation coordinate without sorting it."""

    values = as_float64_array(perturbation, name="perturbation", ndim=1)
    if values.shape != (n_spectra,):
        raise SpectrumValidationError(
            f"perturbation must have shape ({n_spectra},); got {values.shape}"
        )
    require_finite(values, name="perturbation")
    return values


def validate_matching_axes(
    reference: ArrayLike,
    candidate: ArrayLike,
    *,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
) -> None:
    """Require point-for-point identical axes; never interpolate implicitly."""

    left = as_float64_array(reference, name=reference_name, ndim=1)
    right = as_float64_array(candidate, name=candidate_name, ndim=1)
    if left.shape != right.shape:
        raise SpectrumValidationError(
            f"wavenumber axes differ in length: {reference_name}={left.size}, "
            f"{candidate_name}={right.size}"
        )
    unequal = left != right
    if unequal.any():
        first = int(np.flatnonzero(unequal)[0])
        raise SpectrumValidationError(
            "wavenumber axes must match point-for-point; first mismatch at index "
            f"{first}: {reference_name}={left[first]!r}, {candidate_name}={right[first]!r}. "
            "Explicit resampling is required before combining these spectra."
        )


__all__ = [
    "AxisDirection",
    "FloatArray",
    "SpectrumValidationError",
    "ValidatedSpectrumArrays",
    "as_float64_array",
    "require_finite",
    "validate_matching_axes",
    "validate_perturbation",
    "validate_spectrum_arrays",
    "validate_wavenumber",
]
