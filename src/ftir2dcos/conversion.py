"""Explicit FTIR intensity-unit conversion utilities."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import SpectralDataset

INTENSITY_UNITS = (
    "absorbance",
    "percent_transmittance",
    "fraction_transmittance",
    "unknown",
)


def canonical_intensity_unit(unit: str) -> str:
    """Normalize accepted display aliases to the data-contract unit names."""

    normalized = str(unit).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "a": "absorbance",
        "abs": "absorbance",
        "%t": "percent_transmittance",
        "percent_t": "percent_transmittance",
        "transmittance_percent": "percent_transmittance",
        "t": "fraction_transmittance",
        "fraction_t": "fraction_transmittance",
        "transmittance_fraction": "fraction_transmittance",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in INTENSITY_UNITS:
        raise ValueError(f"Unsupported intensity unit {unit!r}; expected one of {INTENSITY_UNITS}")
    return normalized


def transmittance_to_absorbance(
    values: ArrayLike,
    unit: str,
) -> NDArray[np.float64]:
    """Convert explicitly identified transmittance data to absorbance.

    No unit is inferred from numerical range.  All input values must be finite
    and strictly positive, otherwise logarithmic conversion is undefined.
    """

    canonical = canonical_intensity_unit(unit)
    if canonical not in {"percent_transmittance", "fraction_transmittance"}:
        raise ValueError(
            "transmittance_to_absorbance requires 'percent_transmittance' or "
            "'fraction_transmittance'"
        )
    array = np.array(values, dtype=np.float64, copy=True)
    nonfinite = np.argwhere(~np.isfinite(array))
    if nonfinite.size:
        raise ValueError(
            f"Transmittance contains NaN or Inf at positions {nonfinite[:10].tolist()}"
        )
    nonpositive = np.argwhere(array <= 0)
    if nonpositive.size:
        raise ValueError(
            "Transmittance must be strictly greater than zero; invalid positions "
            f"{nonpositive[:10].tolist()} were not converted."
        )
    output = -np.log10(array / 100.0) if canonical == "percent_transmittance" else -np.log10(array)
    return np.asarray(output, dtype=np.float64)


def to_absorbance(values: ArrayLike, input_unit: str) -> NDArray[np.float64]:
    """Return an owned float64 absorbance array for an explicitly given unit."""

    unit = canonical_intensity_unit(input_unit)
    if unit == "unknown":
        raise ValueError(
            "Input intensity unit is 'unknown'. Select a unit explicitly before conversion."
        )
    if unit == "absorbance":
        output = np.array(values, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(output)):
            locations = np.argwhere(~np.isfinite(output))
            raise ValueError(
                f"Absorbance contains NaN or Inf at positions {locations[:10].tolist()}"
            )
        return output
    return transmittance_to_absorbance(values, unit)


def convert_to_absorbance(
    dataset: SpectralDataset,
    input_unit: str | None = None,
) -> SpectralDataset:
    """Convert a dataset without modifying or overwriting its original arrays."""

    unit = canonical_intensity_unit(dataset.intensity_unit if input_unit is None else input_unit)
    converted = to_absorbance(dataset.spectra, unit)
    if unit == "percent_transmittance":
        formula = "A = -log10(T_percent / 100)"
        changed = True
    elif unit == "fraction_transmittance":
        formula = "A = -log10(T_fraction)"
        changed = True
    else:
        formula = "A unchanged (input explicitly identified as absorbance)"
        changed = False

    metadata = deepcopy(dict(dataset.metadata))
    history = list(metadata.get("processing_history", []))
    history.append(
        {
            "operation": "convert_to_absorbance",
            "input_unit": unit,
            "output_unit": "absorbance",
            "formula": formula,
            "values_changed": changed,
        }
    )
    metadata.update(
        {
            "input_intensity_unit": unit,
            "intensity_unit": "absorbance",
            "intensity_conversion_applied": changed,
            "intensity_conversion_formula": formula,
            "processing_history": history,
        }
    )
    return dataset.with_updates(
        spectra=converted,
        intensity_unit="absorbance",
        metadata=metadata,
    )


def suggest_intensity_unit(values: ArrayLike) -> dict[str, str | None]:
    """Return a non-operative display suggestion; it never converts data."""

    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    suggestion: str | None = None
    rationale = "No automatic unit decision was made."
    if finite.size:
        if np.all((finite > 0) & (finite <= 1.5)):
            suggestion = "fraction_transmittance_or_absorbance"
            rationale = (
                "Values fit both fractional transmittance and common absorbance ranges; "
                "the user must select the unit."
            )
        elif np.all((finite > 0) & (finite <= 110)):
            suggestion = "percent_transmittance_possible"
            rationale = "The numerical range permits percent transmittance but is not proof."
    return {"suggestion": suggestion, "rationale": rationale, "action_taken": "none"}


convert_intensity_to_absorbance = convert_to_absorbance


__all__ = [
    "INTENSITY_UNITS",
    "canonical_intensity_unit",
    "convert_intensity_to_absorbance",
    "convert_to_absorbance",
    "suggest_intensity_unit",
    "to_absorbance",
    "transmittance_to_absorbance",
]
