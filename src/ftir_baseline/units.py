"""Explicit FTIR intensity-unit conversion with auditable repairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import immutable_float64
from .validation import SpectrumValidationError, as_float64_array, require_finite

IntensityUnit = Literal[
    "absorbance",
    "percent_transmittance",
    "fraction_transmittance",
]
FloatArray = NDArray[np.float64]


class InvalidTransmittanceError(SpectrumValidationError):
    """Raised for zero or negative transmittance without explicit repair."""


@dataclass(frozen=True, slots=True)
class UnitConversionRecord:
    input_unit: IntensityUnit
    output_unit: Literal["absorbance"]
    formula: str
    transmittance_floor: float | None
    repaired_count: int
    repaired_indices: tuple[tuple[int, ...], ...]

    @property
    def repaired(self) -> bool:
        return self.repaired_count > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "input_unit": self.input_unit,
            "output_unit": self.output_unit,
            "formula": self.formula,
            "transmittance_floor": self.transmittance_floor,
            "repaired_count": self.repaired_count,
            "repaired_indices": [list(index) for index in self.repaired_indices],
        }


@dataclass(frozen=True, slots=True)
class UnitConversionResult:
    absorbance: FloatArray
    record: UnitConversionRecord

    def __post_init__(self) -> None:
        require_finite(self.absorbance, name="converted absorbance")
        object.__setattr__(
            self,
            "absorbance",
            immutable_float64(self.absorbance, name="converted absorbance"),
        )


def _invalid_positions(mask: NDArray[np.bool_]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(part) for part in row) for row in np.argwhere(mask))


def _formula(input_unit: IntensityUnit) -> str:
    if input_unit == "percent_transmittance":
        return "A = -log10(percent_transmittance / 100)"
    if input_unit == "fraction_transmittance":
        return "A = -log10(fraction_transmittance)"
    return "A = absorbance (identity copy)"


def convert_to_absorbance(
    values: ArrayLike,
    input_unit: IntensityUnit,
    *,
    transmittance_floor: float | None = None,
) -> UnitConversionResult:
    """Convert an explicitly identified intensity unit to absorbance.

    NaN and Inf always stop processing.  Zero/negative transmittance also stops
    unless a positive floor is explicitly supplied; in that case every clipped
    position and the floor are returned in the mandatory conversion record.
    Caller-owned input is never modified.
    """

    if input_unit not in {
        "absorbance",
        "percent_transmittance",
        "fraction_transmittance",
    }:
        raise ValueError(
            "input_unit must be 'absorbance', 'percent_transmittance', or "
            "'fraction_transmittance'; unit inference is intentionally not performed"
        )
    array = as_float64_array(values, name="intensity", copy=True)
    require_finite(array, name="intensity")

    if input_unit == "absorbance":
        if transmittance_floor is not None:
            raise ValueError("transmittance_floor is not applicable to absorbance input")
        converted = array.copy()
        repaired_indices: tuple[tuple[int, ...], ...] = ()
    else:
        if transmittance_floor is not None:
            try:
                floor = float(transmittance_floor)
            except (TypeError, ValueError) as exc:
                raise ValueError("transmittance_floor must be a finite positive number") from exc
            if not np.isfinite(floor) or floor <= 0.0:
                raise ValueError("transmittance_floor must be a finite positive number")
            to_repair = array < floor
            repaired_indices = _invalid_positions(to_repair)
            if to_repair.any():
                array[to_repair] = floor
        else:
            invalid = array <= 0.0
            if invalid.any():
                positions = _invalid_positions(invalid)
                preview = positions[:12]
                suffix = "" if len(positions) <= 12 else f" (first 12 of {len(positions)})"
                raise InvalidTransmittanceError(
                    "transmittance must be strictly positive; found "
                    f"{len(positions)} value(s) <= 0 at positions {preview}{suffix}. "
                    "Provide an explicit positive transmittance_floor to repair and record them."
                )
            repaired_indices = ()

        with np.errstate(divide="raise", invalid="raise"):
            if input_unit == "percent_transmittance":
                # ``2 - log10(%T)`` is algebraically identical to dividing by
                # 100 first, but it avoids underflow for tiny positive float64
                # values and is the reference formula in the specification.
                converted = 2.0 - np.log10(array)
            else:
                converted = -np.log10(array)

    record = UnitConversionRecord(
        input_unit=input_unit,
        output_unit="absorbance",
        formula=_formula(input_unit),
        transmittance_floor=(
            float(transmittance_floor) if transmittance_floor is not None else None
        ),
        repaired_count=len(repaired_indices),
        repaired_indices=repaired_indices,
    )
    return UnitConversionResult(converted, record)


@overload
def to_absorbance(
    values: ArrayLike,
    input_unit: IntensityUnit,
    *,
    transmittance_floor: None = None,
    return_record: Literal[False] = False,
) -> FloatArray: ...


@overload
def to_absorbance(
    values: ArrayLike,
    input_unit: IntensityUnit,
    *,
    transmittance_floor: float | None = None,
    return_record: Literal[True],
) -> tuple[FloatArray, UnitConversionRecord]: ...


def to_absorbance(
    values: ArrayLike,
    input_unit: IntensityUnit,
    *,
    transmittance_floor: float | None = None,
    return_record: bool = False,
) -> FloatArray | tuple[FloatArray, UnitConversionRecord]:
    """Convenience conversion API.

    Repairs cannot be requested while discarding their audit record.  Use
    ``return_record=True`` whenever ``transmittance_floor`` is supplied.
    """

    if transmittance_floor is not None and not return_record:
        raise ValueError(
            "return_record=True is required with transmittance_floor so repairs are not silent"
        )
    result = convert_to_absorbance(values, input_unit, transmittance_floor=transmittance_floor)
    if return_record:
        return result.absorbance, result.record
    return result.absorbance


def percent_transmittance_to_absorbance(values: ArrayLike) -> FloatArray:
    """Strictly convert percent transmittance; no floor repair is performed."""

    return convert_to_absorbance(values, "percent_transmittance").absorbance


def fraction_transmittance_to_absorbance(values: ArrayLike) -> FloatArray:
    """Strictly convert 0--1 fractional transmittance to absorbance."""

    return convert_to_absorbance(values, "fraction_transmittance").absorbance


__all__ = [
    "FloatArray",
    "IntensityUnit",
    "InvalidTransmittanceError",
    "UnitConversionRecord",
    "UnitConversionResult",
    "convert_to_absorbance",
    "fraction_transmittance_to_absorbance",
    "percent_transmittance_to_absorbance",
    "to_absorbance",
]
