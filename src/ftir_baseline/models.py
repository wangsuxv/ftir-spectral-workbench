"""Immutable data models shared by the scientific core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import (
    SpectrumValidationError,
    require_finite,
    validate_perturbation,
    validate_spectrum_arrays,
)

FloatArray = NDArray[np.float64]


def immutable_float64(values: ArrayLike, *, name: str) -> FloatArray:
    """Create an ndarray backed by immutable bytes.

    Merely setting ``writeable=False`` is insufficient for an owning NumPy array:
    callers can set that flag back to true.  An immutable ``bytes`` owner prevents
    both direct writes and re-enabling the flag, while also severing all aliases to
    caller-owned input memory.
    """

    try:
        source = np.asarray(values, dtype=np.float64, order="C")
    except (TypeError, ValueError) as exc:
        raise SpectrumValidationError(f"{name} must contain only numeric values") from exc
    shape = source.shape
    frozen = np.frombuffer(source.tobytes(order="C"), dtype=np.float64).reshape(shape)
    return frozen


def _freeze(value: Any, *, path: str = "metadata") -> Any:
    """Recursively freeze JSON-like metadata and common scientific scalars."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(item, path=f"{path}.{key}") for key, item in value.items()}
        )
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            return immutable_float64(value, name=path)
        return tuple(_freeze(item, path=path) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, path=path) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item, path=path) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    raise TypeError(
        f"{path} contains unsupported mutable/non-serializable value of type {type(value).__name__}"
    )


def freeze_value(value: Any, *, path: str = "value") -> Any:
    """Recursively detach and freeze JSON-like scientific metadata."""

    return _freeze(value, path=path)


def thaw_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly mutable copy of a recursively frozen mapping."""

    def thaw(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: thaw(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [thaw(item) for item in value]
        if isinstance(value, frozenset):
            return [thaw(item) for item in sorted(value, key=repr)]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    return {key: thaw(value) for key, value in mapping.items()}


@dataclass(frozen=True, slots=True)
class SpectrumSet:
    """A validated, deeply immutable collection of spectra.

    All numerical arrays are copied to native float64 and backed by immutable
    memory.  Input spectrum order and axis direction are preserved.
    """

    wavenumber: FloatArray
    perturbation: FloatArray
    perturbation_labels: tuple[str, ...]
    spectra: FloatArray
    intensity_unit: str
    source_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validated = validate_spectrum_arrays(self.wavenumber, self.spectra)
        perturbation = validate_perturbation(
            self.perturbation, n_spectra=validated.spectra.shape[0]
        )
        labels = tuple(str(value) for value in self.perturbation_labels)
        if len(labels) != validated.spectra.shape[0]:
            raise SpectrumValidationError(
                "perturbation_labels length must match number of spectra: "
                f"got {len(labels)} labels for {validated.spectra.shape[0]} spectra"
            )
        if not self.intensity_unit or not isinstance(self.intensity_unit, str):
            raise SpectrumValidationError("intensity_unit must be a non-empty string")
        if not isinstance(self.source_name, str):
            raise SpectrumValidationError("source_name must be a string")

        metadata = dict(self.metadata)
        recorded_direction = metadata.get("original_axis_direction")
        if recorded_direction is not None and recorded_direction not in {"ascending", "descending"}:
            raise SpectrumValidationError(
                "metadata.original_axis_direction must be 'ascending' or 'descending'"
            )
        metadata.setdefault("original_axis_direction", validated.axis_direction)
        metadata["axis_direction"] = validated.axis_direction

        object.__setattr__(
            self, "wavenumber", immutable_float64(validated.wavenumber, name="wavenumber")
        )
        object.__setattr__(
            self, "perturbation", immutable_float64(perturbation, name="perturbation")
        )
        object.__setattr__(self, "spectra", immutable_float64(validated.spectra, name="spectra"))
        object.__setattr__(self, "perturbation_labels", labels)
        object.__setattr__(self, "metadata", _freeze(metadata))

    @property
    def n_spectra(self) -> int:
        return int(self.spectra.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.wavenumber.size)

    @property
    def axis_direction(self) -> str:
        return str(self.metadata["axis_direction"])

    def mutable_metadata(self) -> dict[str, Any]:
        """Return a detached JSON-friendly metadata copy."""

        return thaw_mapping(self.metadata)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Immutable baseline decomposition for one spectrum or a series."""

    coarse_baseline: FloatArray
    fine_baseline: FloatArray
    total_baseline: FloatArray
    corrected: FloatArray
    params: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = ("coarse_baseline", "fine_baseline", "total_baseline", "corrected")
        arrays = [np.asarray(getattr(self, name), dtype=np.float64) for name in names]
        if arrays[0].ndim not in (1, 2):
            raise SpectrumValidationError(
                f"baseline result arrays must be 1-D or 2-D; got {arrays[0].shape}"
            )
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            shapes = {name: array.shape for name, array in zip(names, arrays, strict=True)}
            raise SpectrumValidationError(
                f"baseline result arrays must have one shape; got {shapes}"
            )
        for name, array in zip(names, arrays, strict=True):
            require_finite(array, name=name)
        if not np.allclose(arrays[2], arrays[0] + arrays[1], rtol=1e-12, atol=1e-14):
            maximum_error = float(np.max(np.abs(arrays[2] - arrays[0] - arrays[1])))
            raise SpectrumValidationError(
                "total_baseline must equal coarse_baseline + fine_baseline; "
                f"maximum absolute error is {maximum_error:.6g}"
            )
        for name, array in zip(names, arrays, strict=True):
            object.__setattr__(self, name, immutable_float64(array, name=name))
        object.__setattr__(self, "params", _freeze(dict(self.params), path="params"))
        object.__setattr__(self, "metrics", _freeze(dict(self.metrics), path="metrics"))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))


__all__ = [
    "BaselineResult",
    "FloatArray",
    "SpectrumSet",
    "freeze_value",
    "immutable_float64",
    "thaw_mapping",
]
