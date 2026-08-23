"""Wavenumber range selection and explicit axis-orientation handling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import SpectrumSet
from .validation import SpectrumValidationError, as_float64_array, validate_spectrum_arrays

FloatArray = NDArray[np.float64]


def normalize_range(bounds: Sequence[float]) -> tuple[float, float]:
    """Return finite ``(low, high)`` bounds while accepting either user order."""

    if len(bounds) != 2:
        raise ValueError(f"wavenumber range must contain exactly two bounds; got {len(bounds)}")
    first, second = float(bounds[0]), float(bounds[1])
    if not np.isfinite([first, second]).all():
        raise ValueError("wavenumber range bounds must be finite")
    if first == second:
        raise ValueError("wavenumber range bounds must differ")
    return min(first, second), max(first, second)


def range_mask(wavenumber: ArrayLike, bounds: Sequence[float]) -> NDArray[np.bool_]:
    """Build an inclusive mask without reordering the wavenumber axis."""

    x = as_float64_array(wavenumber, name="wavenumber", ndim=1)
    low, high = normalize_range(bounds)
    return (x >= low) & (x <= high)


def select_array_range(
    wavenumber: ArrayLike,
    spectra: ArrayLike,
    bounds: Sequence[float],
    *,
    strict_bounds: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Select a common inclusive range from arrays, preserving direction/order."""

    validated = validate_spectrum_arrays(wavenumber, spectra)
    low, high = normalize_range(bounds)
    available_low = float(np.min(validated.wavenumber))
    available_high = float(np.max(validated.wavenumber))
    if strict_bounds and (low < available_low or high > available_high):
        raise SpectrumValidationError(
            f"requested range [{low}, {high}] cm^-1 lies outside available "
            f"[{available_low}, {available_high}] cm^-1"
        )
    mask = range_mask(validated.wavenumber, (low, high))
    selected_count = int(mask.sum())
    if selected_count < 2:
        raise SpectrumValidationError(
            f"requested range [{low}, {high}] cm^-1 selects {selected_count} point(s); "
            "at least two are required"
        )
    return validated.wavenumber[mask].copy(), validated.spectra[:, mask].copy()


def crop_spectrum_set(
    data: SpectrumSet,
    bounds: Sequence[float],
    *,
    strict_bounds: bool = False,
) -> SpectrumSet:
    """Crop every spectrum identically and append an auditable range record."""

    low, high = normalize_range(bounds)
    wavenumber, spectra = select_array_range(
        data.wavenumber, data.spectra, (low, high), strict_bounds=strict_bounds
    )
    metadata = data.mutable_metadata()
    history = list(metadata.get("range_selection_history", []))
    history.append(
        {
            "requested_bounds": [float(bounds[0]), float(bounds[1])],
            "normalized_bounds": [low, high],
            "actual_bounds": [float(np.min(wavenumber)), float(np.max(wavenumber))],
            "selected_point_count": int(wavenumber.size),
            "axis_direction_preserved": True,
        }
    )
    metadata["range_selection_history"] = history
    return SpectrumSet(
        wavenumber=wavenumber,
        perturbation=data.perturbation,
        perturbation_labels=data.perturbation_labels,
        spectra=spectra,
        intensity_unit=data.intensity_unit,
        source_name=data.source_name,
        metadata=metadata,
    )


def orient_spectrum_set(
    data: SpectrumSet,
    direction: Literal["ascending", "descending"] = "ascending",
) -> SpectrumSet:
    """Return an explicitly oriented copy, recording whether reversal occurred."""

    if direction not in {"ascending", "descending"}:
        raise ValueError("direction must be 'ascending' or 'descending'")
    if data.axis_direction == direction:
        return data
    metadata = data.mutable_metadata()
    history = list(metadata.get("axis_orientation_history", []))
    history.append(
        {
            "from": data.axis_direction,
            "to": direction,
            "reversed": True,
        }
    )
    metadata["axis_orientation_history"] = history
    metadata["axis_reversed_for_processing"] = direction == "ascending"
    return SpectrumSet(
        wavenumber=data.wavenumber[::-1],
        perturbation=data.perturbation,
        perturbation_labels=data.perturbation_labels,
        spectra=data.spectra[:, ::-1],
        intensity_unit=data.intensity_unit,
        source_name=data.source_name,
        metadata=metadata,
    )


def restore_original_axis(data: SpectrumSet) -> SpectrumSet:
    """Restore the direction captured at initial import, if needed."""

    original = data.metadata.get("original_axis_direction")
    if original not in {"ascending", "descending"}:
        raise SpectrumValidationError("original axis direction metadata is missing or invalid")
    return orient_spectrum_set(data, direction=original)


# Naming aliases for pipeline/UI callers.
select_wavenumber_range = crop_spectrum_set
crop_wavenumber_range = crop_spectrum_set


__all__ = [
    "crop_spectrum_set",
    "crop_wavenumber_range",
    "normalize_range",
    "orient_spectrum_set",
    "range_mask",
    "restore_original_axis",
    "select_array_range",
    "select_wavenumber_range",
]
