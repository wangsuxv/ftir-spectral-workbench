"""Validation helpers for the cross-package prepared-data contract.

The coordination layer validates data at its boundary instead of relying on
either legacy pipeline to repair, transpose, sort, or otherwise reinterpret it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:  # pragma: no cover - imports used only by static checkers
    from .models import PreparedSpectralDataset

AxisDirection = Literal["ascending", "descending"]
FloatArray = NDArray[np.float64]


class PreparedDatasetValidationError(ValueError):
    """Raised when prepared spectra violate the workbench data contract."""


@dataclass(frozen=True, slots=True)
class ValidatedPreparedArrays:
    """Detached ``float64`` arrays plus their validated axis direction."""

    wavenumber: FloatArray
    perturbation: FloatArray
    spectra: FloatArray
    perturbation_labels: tuple[str, ...]
    axis_direction: AxisDirection


def _float64_array(values: ArrayLike, *, name: str, ndim: int) -> FloatArray:
    if np.iscomplexobj(values):
        raise PreparedDatasetValidationError(f"{name} must contain real numeric values")
    try:
        array = np.array(values, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise PreparedDatasetValidationError(
            f"{name} must contain only real numeric values"
        ) from exc
    if array.ndim != ndim:
        raise PreparedDatasetValidationError(
            f"{name} must be {ndim}-dimensional; got shape {array.shape}"
        )
    invalid = ~np.isfinite(array)
    if invalid.any():
        locations = np.argwhere(invalid)[:12].tolist()
        raise PreparedDatasetValidationError(
            f"{name} contains NaN or infinite values at positions {locations}"
        )
    return array


def wavenumber_direction(wavenumber: ArrayLike) -> AxisDirection:
    """Return the strict direction of a finite, unique wavenumber axis."""

    axis = _float64_array(wavenumber, name="wavenumber", ndim=1)
    if axis.size < 2:
        raise PreparedDatasetValidationError(
            f"wavenumber must contain at least 2 points; got {axis.size}"
        )
    differences = np.diff(axis)
    if np.all(differences > 0.0):
        return "ascending"
    if np.all(differences < 0.0):
        return "descending"
    duplicate_indices = np.flatnonzero(differences == 0.0)
    if duplicate_indices.size:
        shown = duplicate_indices[:12].astype(int).tolist()
        raise PreparedDatasetValidationError(
            "wavenumber must be strictly monotonic and unique; "
            f"duplicate adjacent values start at indices {shown}"
        )
    reversals = np.flatnonzero(np.sign(differences[1:]) != np.sign(differences[:-1])) + 1
    raise PreparedDatasetValidationError(
        "wavenumber must be strictly monotonic; "
        f"direction changes near indices {reversals[:12].astype(int).tolist()}"
    )


def validate_perturbation_labels(
    labels: object,
    *,
    n_spectra: int,
) -> tuple[str, ...]:
    """Validate label count and reject blank labels without changing their order."""

    if isinstance(labels, (str, bytes)):
        raise PreparedDatasetValidationError(
            "perturbation_labels must be a sequence containing one label per spectrum"
        )
    if not isinstance(labels, Iterable):
        raise PreparedDatasetValidationError(
            "perturbation_labels must be an iterable of strings"
        )
    raw_labels: tuple[object, ...] = tuple(labels)
    if len(raw_labels) != n_spectra:
        raise PreparedDatasetValidationError(
            "perturbation_labels length must match number of spectra: "
            f"got {len(raw_labels)} labels for {n_spectra} spectra"
        )
    normalized: list[str] = []
    for index, value in enumerate(raw_labels):
        if not isinstance(value, str):
            raise PreparedDatasetValidationError(
                f"perturbation_labels[{index}] must be a string; got {type(value).__name__}"
            )
        if not value.strip():
            raise PreparedDatasetValidationError(
                f"perturbation_labels[{index}] must not be empty or whitespace-only"
            )
        normalized.append(value)
    return tuple(normalized)


def validate_prepared_arrays(
    wavenumber: ArrayLike,
    perturbation: ArrayLike,
    perturbation_labels: object,
    spectra: ArrayLike,
) -> ValidatedPreparedArrays:
    """Validate the fixed ``(n_spectra, n_wavenumbers)`` prepared orientation.

    No 1-D promotion, transpose, sorting, interpolation, or direction reversal is
    performed here.  Such operations must be explicit upstream operations.
    """

    axis = _float64_array(wavenumber, name="wavenumber", ndim=1)
    direction = wavenumber_direction(axis)
    matrix = _float64_array(spectra, name="spectra", ndim=2)
    if matrix.shape[0] < 1:
        raise PreparedDatasetValidationError("spectra must contain at least one spectrum")
    if matrix.shape[1] != axis.size:
        raise PreparedDatasetValidationError(
            "spectra must have shape (n_spectra, n_wavenumbers): "
            f"got spectra.shape={matrix.shape} and wavenumber.shape={axis.shape}"
        )
    perturbation_array = _float64_array(
        perturbation,
        name="perturbation",
        ndim=1,
    )
    if perturbation_array.shape != (matrix.shape[0],):
        raise PreparedDatasetValidationError(
            f"perturbation must have shape ({matrix.shape[0]},); "
            f"got {perturbation_array.shape}"
        )
    labels = validate_perturbation_labels(
        perturbation_labels,
        n_spectra=matrix.shape[0],
    )
    return ValidatedPreparedArrays(
        wavenumber=axis,
        perturbation=perturbation_array,
        spectra=matrix,
        perturbation_labels=labels,
        axis_direction=direction,
    )


def validate_prepared_dataset(dataset: PreparedSpectralDataset) -> None:
    """Revalidate an existing dataset, including its recorded direction/hash."""

    validated = validate_prepared_arrays(
        dataset.wavenumber,
        dataset.perturbation,
        dataset.perturbation_labels,
        dataset.spectra,
    )
    if dataset.current_axis_direction != validated.axis_direction:
        raise PreparedDatasetValidationError(
            "current_axis_direction does not match the wavenumber array: "
            f"recorded={dataset.current_axis_direction!r}, "
            f"detected={validated.axis_direction!r}"
        )
    from .fingerprints import prepared_data_sha256

    actual_hash = prepared_data_sha256(
        dataset.wavenumber,
        dataset.perturbation,
        dataset.perturbation_labels,
        dataset.spectra,
        normalization_state=dataset.normalization_state,
    )
    if dataset.prepared_data_sha256 != actual_hash:
        raise PreparedDatasetValidationError(
            "prepared_data_sha256 does not match the prepared arrays and labels"
        )


def validate_cross_prepared_compatibility(
    left: PreparedSpectralDataset,
    right: PreparedSpectralDataset,
) -> tuple[str, ...]:
    """Require identical perturbation values, labels, and order for cross 2D-COS.

    A different source is scientifically notable but not a numerical mismatch, so
    it is returned as a warning for the caller to surface and confirm.
    """

    if left.spectra.shape[0] != right.spectra.shape[0]:
        raise PreparedDatasetValidationError(
            "cross-range prepared blocks must contain the same number of spectra"
        )
    if not np.array_equal(left.perturbation, right.perturbation):
        raise PreparedDatasetValidationError(
            "cross-range prepared blocks must have point-for-point identical "
            "perturbation values and order"
        )
    if left.perturbation_labels != right.perturbation_labels:
        raise PreparedDatasetValidationError(
            "cross-range prepared blocks must have identical perturbation labels and order"
        )
    warnings: list[str] = []
    if (
        left.source_sha256 != right.source_sha256
        or left.source_name != right.source_name
    ):
        warnings.append(
            "Cross-range blocks have different recorded sources; explicit user confirmation "
            "is required before interpreting their correlation."
        )
    return tuple(warnings)


__all__ = [
    "AxisDirection",
    "FloatArray",
    "PreparedDatasetValidationError",
    "ValidatedPreparedArrays",
    "validate_cross_prepared_compatibility",
    "validate_perturbation_labels",
    "validate_prepared_arrays",
    "validate_prepared_dataset",
    "wavenumber_direction",
]
