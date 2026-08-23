"""Core immutable data contracts used by the FTIR processing pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _readonly_float64(value: ArrayLike, *, ndim: int, name: str) -> FloatArray:
    """Return an owned, C-contiguous, read-only ``float64`` array."""

    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got shape {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class SpectralDataset:
    """A collection of spectra sharing one wavenumber axis.

    The internal orientation is always ``(n_spectra, n_wavenumbers)``.  Array
    inputs are defensively copied, converted to ``float64``, and marked
    read-only.  This makes accidental in-place preprocessing of raw input much
    harder while still allowing efficient NumPy reads.

    Numerical validity (finite values, monotonicity, minimum sizes, and so on)
    is deliberately handled by :func:`ftir2dcos.validation.validate_dataset`.
    The model only enforces structural consistency so a malformed import can be
    represented and reported with all validation issues at once.
    """

    wavenumber: FloatArray
    perturbation: FloatArray
    perturbation_labels: tuple[str, ...]
    spectra: FloatArray
    intensity_unit: str = "unknown"
    source_name: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wavenumber = _readonly_float64(self.wavenumber, ndim=1, name="wavenumber")
        perturbation = _readonly_float64(self.perturbation, ndim=1, name="perturbation")
        spectra = _readonly_float64(self.spectra, ndim=2, name="spectra")
        labels = tuple(str(label) for label in self.perturbation_labels)

        n_spectra, n_wavenumbers = spectra.shape
        if wavenumber.shape != (n_wavenumbers,):
            raise ValueError(
                "wavenumber length must match spectra.shape[1]; "
                f"got {wavenumber.shape} and {spectra.shape}"
            )
        if perturbation.shape != (n_spectra,):
            raise ValueError(
                "perturbation length must match spectra.shape[0]; "
                f"got {perturbation.shape} and {spectra.shape}"
            )
        if len(labels) != n_spectra:
            raise ValueError(
                "perturbation_labels length must match spectra.shape[0]; "
                f"got {len(labels)} and {spectra.shape}"
            )

        object.__setattr__(self, "wavenumber", wavenumber)
        object.__setattr__(self, "perturbation", perturbation)
        object.__setattr__(self, "spectra", spectra)
        object.__setattr__(self, "perturbation_labels", labels)
        object.__setattr__(self, "intensity_unit", str(self.intensity_unit).lower())
        object.__setattr__(self, "source_name", str(self.source_name))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    @property
    def n_spectra(self) -> int:
        """Number of spectra (the perturbation dimension)."""

        return self.spectra.shape[0]

    @property
    def n_wavenumbers(self) -> int:
        """Number of points on the shared wavenumber axis."""

        return self.spectra.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        """The internal ``(n_spectra, n_wavenumbers)`` shape."""

        return self.spectra.shape

    def with_updates(self, **changes: Any) -> SpectralDataset:
        """Create another protected dataset without mutating this one."""

        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Result for a single baseline-corrected spectrum."""

    baseline: FloatArray
    corrected: FloatArray
    method: str
    parameters: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        baseline = _readonly_float64(self.baseline, ndim=1, name="baseline")
        corrected = _readonly_float64(self.corrected, ndim=1, name="corrected")
        if baseline.shape != corrected.shape:
            raise ValueError(
                f"baseline and corrected must have equal shapes; got {baseline.shape} and "
                f"{corrected.shape}"
            )
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "corrected", corrected)
        object.__setattr__(self, "method", str(self.method).lower())
        object.__setattr__(self, "parameters", deepcopy(dict(self.parameters)))
        object.__setattr__(self, "diagnostics", deepcopy(dict(self.diagnostics)))


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Structured output from dataset validation."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Whether validation found no blocking errors."""

        return not self.errors

    @property
    def valid(self) -> bool:
        """Alias for :attr:`is_valid` for ergonomic API use."""

        return self.is_valid

    def raise_for_errors(self) -> None:
        """Raise one readable error containing every blocking issue."""

        if self.errors:
            details = "\n- ".join(self.errors)
            raise ValueError(f"Dataset validation failed:\n- {details}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metrics": deepcopy(self.metrics),
        }


__all__ = [
    "BaselineResult",
    "FloatArray",
    "SpectralDataset",
    "ValidationReport",
]
