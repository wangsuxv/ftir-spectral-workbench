"""Non-scientific intensity representations for display and derived export.

The workbench's scientific contract remains baseline-corrected absorbance.  The
conversions in this module create detached display values only; they do not
construct or modify a prepared dataset, fingerprint, recipe, or pipeline result.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

DisplayIntensityUnit = Literal[
    "absorbance",
    "fraction_transmittance",
    "percent_transmittance",
]
FloatArray = NDArray[np.float64]

_DISPLAY_UNITS = frozenset(
    {
        "absorbance",
        "fraction_transmittance",
        "percent_transmittance",
    }
)
_NEGATIVE_ABSORBANCE_WARNING = (
    "Negative absorbance produces transmittance above its conventional maximum; "
    "values were not clipped."
)
_DERIVED_DISPLAY_NOTICE = (
    "This is a mathematical representation derived from baseline-corrected absorbance."
)
_NOT_INSTRUMENT_NOTICE = "It is not the original instrument transmittance."


def _owned_finite_float64(values: ArrayLike, *, name: str) -> FloatArray:
    """Return a detached real float64 array after strict validation."""

    try:
        source = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if np.iscomplexobj(source):
        raise TypeError(f"{name} must contain real values; complex values are unsupported")
    try:
        owned = np.array(source, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(owned)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return owned


def _immutable_float64(values: ArrayLike, *, name: str) -> FloatArray:
    """Detach values and return an array backed by immutable bytes."""

    owned = _owned_finite_float64(values, name=name)
    return np.frombuffer(owned.tobytes(order="C"), dtype=np.float64).reshape(owned.shape)


@dataclass(frozen=True, slots=True)
class DisplayConversionResult:
    """One explicitly labelled, immutable absorbance display conversion."""

    values: FloatArray
    input_unit: Literal["absorbance"]
    output_unit: DisplayIntensityUnit
    formula: str
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.input_unit != "absorbance":
            raise ValueError("display conversion input_unit must be 'absorbance'")
        if self.output_unit not in _DISPLAY_UNITS:
            raise ValueError(f"unsupported display intensity unit: {self.output_unit!r}")
        if not isinstance(self.formula, str) or not self.formula.strip():
            raise ValueError("display conversion formula must be a non-empty string")
        object.__setattr__(
            self,
            "values",
            _immutable_float64(self.values, name="display intensity"),
        )
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))


def _transmittance_warnings(absorbance: FloatArray) -> tuple[str, ...]:
    if np.any(absorbance < 0.0):
        return (_NEGATIVE_ABSORBANCE_WARNING,)
    return ()


def absorbance_to_fraction_transmittance(
    values: ArrayLike,
) -> DisplayConversionResult:
    """Return the display representation ``T = 10 ** (-A)`` without clipping."""

    absorbance = _owned_finite_float64(values, name="absorbance")
    with np.errstate(over="raise", invalid="raise"):
        converted = np.power(10.0, -absorbance)
    return DisplayConversionResult(
        values=converted,
        input_unit="absorbance",
        output_unit="fraction_transmittance",
        formula="T = 10 ** (-A)",
        warnings=_transmittance_warnings(absorbance),
    )


def absorbance_to_percent_transmittance(
    values: ArrayLike,
) -> DisplayConversionResult:
    """Return the display representation ``%T = 100 * 10 ** (-A)`` without clipping."""

    absorbance = _owned_finite_float64(values, name="absorbance")
    with np.errstate(over="raise", invalid="raise"):
        converted = np.multiply(100.0, np.power(10.0, -absorbance))
    return DisplayConversionResult(
        values=converted,
        input_unit="absorbance",
        output_unit="percent_transmittance",
        formula="%T = 100 * 10 ** (-A)",
        warnings=_transmittance_warnings(absorbance),
    )


def convert_absorbance_for_display(
    values: ArrayLike,
    output_unit: DisplayIntensityUnit,
) -> DisplayConversionResult:
    """Convert absorbance to an explicitly selected display-only representation."""

    if output_unit == "fraction_transmittance":
        return absorbance_to_fraction_transmittance(values)
    if output_unit == "percent_transmittance":
        return absorbance_to_percent_transmittance(values)
    if output_unit != "absorbance":
        raise ValueError(f"unsupported display intensity unit: {output_unit!r}")
    absorbance = _owned_finite_float64(values, name="absorbance")
    return DisplayConversionResult(
        values=absorbance,
        input_unit="absorbance",
        output_unit="absorbance",
        formula="A = absorbance (identity copy)",
        warnings=(),
    )


def derived_transmittance_filename(
    output_unit: Literal["fraction_transmittance", "percent_transmittance"],
) -> str:
    """Return the mandatory filename for one independent derived download."""

    if output_unit == "fraction_transmittance":
        return "derived_fraction_transmittance_from_corrected_absorbance.csv"
    if output_unit == "percent_transmittance":
        return "derived_percent_transmittance_from_corrected_absorbance.csv"
    raise ValueError("derived export unit must be fraction_transmittance or percent_transmittance")


def derived_transmittance_csv_bytes(
    wavenumber: ArrayLike,
    corrected_absorbance: ArrayLike,
    perturbation_labels: Sequence[str],
    *,
    output_unit: Literal["fraction_transmittance", "percent_transmittance"],
) -> bytes:
    """Serialize a labelled derived T/%T CSV with explicit non-instrument metadata."""

    axis = _owned_finite_float64(wavenumber, name="wavenumber")
    if axis.ndim != 1:
        raise ValueError("wavenumber must be one-dimensional")
    absorbance = _owned_finite_float64(
        corrected_absorbance,
        name="baseline-corrected absorbance",
    )
    labels = tuple(str(item) for item in perturbation_labels)
    if absorbance.shape != (len(labels), axis.size):
        raise ValueError(
            "baseline-corrected absorbance must have shape "
            "(number of labels, number of wavenumbers)"
        )
    converted = convert_absorbance_for_display(absorbance, output_unit)
    stream = io.StringIO(newline="")
    stream.write(f"# {_DERIVED_DISPLAY_NOTICE}\n")
    stream.write(f"# {_NOT_INSTRUMENT_NOTICE}\n")
    stream.write(f"# output_unit={converted.output_unit}\n")
    stream.write(f"# formula={converted.formula}\n")
    for warning in converted.warnings:
        stream.write(f"# warning={warning}\n")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["Wavenumber", *labels])
    for coordinate, intensities in zip(axis, converted.values.T, strict=True):
        writer.writerow(
            [format(float(coordinate), ".17g")]
            + [format(float(value), ".17g") for value in intensities]
        )
    return stream.getvalue().encode("utf-8")


__all__ = [
    "DisplayConversionResult",
    "DisplayIntensityUnit",
    "absorbance_to_fraction_transmittance",
    "absorbance_to_percent_transmittance",
    "convert_absorbance_for_display",
    "derived_transmittance_csv_bytes",
    "derived_transmittance_filename",
]
