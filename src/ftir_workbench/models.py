"""Immutable cross-package data contracts for FTIR Spectral Workbench."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ftir_baseline.models import freeze_value, thaw_mapping

from .fingerprints import prepared_data_sha256 as compute_prepared_data_sha256
from .validation import PreparedDatasetValidationError, validate_prepared_arrays

FloatArray = NDArray[np.float64]
AxisDirection = Literal["ascending", "descending"]
NormalizationState = Literal["none", "display_only", "scientific_explicit"]

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_PROVENANCE_SENTINELS = frozenset({"unknown", "unavailable"})


def _immutable_float64(values: ArrayLike) -> FloatArray:
    """Detach an array and back it by immutable bytes."""

    source = np.asarray(values, dtype=np.float64, order="C")
    return np.frombuffer(source.tobytes(order="C"), dtype=np.float64).reshape(source.shape)


def _validated_hash_or_sentinel(
    value: object,
    *,
    field_name: str,
    allow_provenance_sentinel: bool,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreparedDatasetValidationError(f"{field_name} must be a non-empty string")
    normalized = value.strip().lower()
    if allow_provenance_sentinel and normalized in _PROVENANCE_SENTINELS:
        return normalized
    if not _SHA256_PATTERN.fullmatch(normalized):
        suffix = " or an explicit provenance sentinel" if allow_provenance_sentinel else ""
        raise PreparedDatasetValidationError(
            f"{field_name} must be a 64-character hexadecimal SHA-256{suffix}"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class PreparedSpectralDataset:
    """A validated, immutable baseline-to-2D-COS handoff.

    Numerical orientation is always ``(n_spectra, n_wavenumbers)``.  Construction
    never transposes, sorts, reverses, interpolates, or fills values.  Arrays are
    copied to finite ``float64`` and backed by immutable bytes so callers cannot
    re-enable NumPy's write flag.
    """

    wavenumber: FloatArray
    perturbation: FloatArray
    perturbation_labels: tuple[str, ...]
    spectra: FloatArray

    intensity_unit: Literal["absorbance"]
    source_name: str
    source_sha256: str
    baseline_run_id: str
    baseline_fingerprint: str
    prepared_data_sha256: str

    original_axis_direction: AxisDirection
    current_axis_direction: AxisDirection
    perturbation_order_policy: str

    baseline_recipe: Mapping[str, Any]
    baseline_qc: Mapping[str, Any]
    warnings: tuple[str, ...]

    normalization_state: NormalizationState = "none"

    def __post_init__(self) -> None:
        validated = validate_prepared_arrays(
            self.wavenumber,
            self.perturbation,
            self.perturbation_labels,
            self.spectra,
        )
        if self.intensity_unit != "absorbance":
            raise PreparedDatasetValidationError(
                "PreparedSpectralDataset.intensity_unit must be exactly 'absorbance'"
            )
        if not isinstance(self.source_name, str) or not self.source_name.strip():
            raise PreparedDatasetValidationError("source_name must be a non-empty string")
        if not isinstance(self.baseline_run_id, str) or not self.baseline_run_id.strip():
            raise PreparedDatasetValidationError("baseline_run_id must be a non-empty string")
        if self.original_axis_direction not in {"ascending", "descending"}:
            raise PreparedDatasetValidationError(
                "original_axis_direction must be 'ascending' or 'descending'"
            )
        if self.current_axis_direction not in {"ascending", "descending"}:
            raise PreparedDatasetValidationError(
                "current_axis_direction must be 'ascending' or 'descending'"
            )
        if self.current_axis_direction != validated.axis_direction:
            raise PreparedDatasetValidationError(
                "current_axis_direction does not match the wavenumber array: "
                f"recorded={self.current_axis_direction!r}, "
                f"detected={validated.axis_direction!r}"
            )
        if (
            not isinstance(self.perturbation_order_policy, str)
            or not self.perturbation_order_policy.strip()
        ):
            raise PreparedDatasetValidationError(
                "perturbation_order_policy must be a non-empty string"
            )
        if self.normalization_state not in {
            "none",
            "display_only",
            "scientific_explicit",
        }:
            raise PreparedDatasetValidationError(
                "normalization_state must be 'none', 'display_only', or "
                "'scientific_explicit'"
            )
        if not isinstance(self.baseline_recipe, Mapping):
            raise PreparedDatasetValidationError("baseline_recipe must be a mapping")
        if not isinstance(self.baseline_qc, Mapping):
            raise PreparedDatasetValidationError("baseline_qc must be a mapping")

        warnings = tuple(str(item) for item in self.warnings)
        source_hash = _validated_hash_or_sentinel(
            self.source_sha256,
            field_name="source_sha256",
            allow_provenance_sentinel=True,
        )
        baseline_hash = _validated_hash_or_sentinel(
            self.baseline_fingerprint,
            field_name="baseline_fingerprint",
            allow_provenance_sentinel=True,
        )
        prepared_hash = _validated_hash_or_sentinel(
            self.prepared_data_sha256,
            field_name="prepared_data_sha256",
            allow_provenance_sentinel=False,
        )
        run_id = self.baseline_run_id.strip()
        if (
            source_hash in _PROVENANCE_SENTINELS
            or baseline_hash in _PROVENANCE_SENTINELS
            or run_id.lower() in _PROVENANCE_SENTINELS
        ) and not warnings:
            raise PreparedDatasetValidationError(
                "incomplete source/baseline provenance must be recorded in warnings"
            )

        expected_hash = compute_prepared_data_sha256(
            validated.wavenumber,
            validated.perturbation,
            validated.perturbation_labels,
            validated.spectra,
            normalization_state=self.normalization_state,
        )
        if prepared_hash != expected_hash:
            raise PreparedDatasetValidationError(
                "prepared_data_sha256 does not match wavenumber, perturbation, labels, "
                "spectra, and scientific branch state"
            )

        object.__setattr__(self, "wavenumber", _immutable_float64(validated.wavenumber))
        object.__setattr__(self, "perturbation", _immutable_float64(validated.perturbation))
        object.__setattr__(self, "spectra", _immutable_float64(validated.spectra))
        object.__setattr__(self, "perturbation_labels", validated.perturbation_labels)
        object.__setattr__(self, "source_name", self.source_name.strip())
        object.__setattr__(self, "source_sha256", source_hash)
        object.__setattr__(self, "baseline_run_id", run_id)
        object.__setattr__(self, "baseline_fingerprint", baseline_hash)
        object.__setattr__(self, "prepared_data_sha256", prepared_hash)
        object.__setattr__(
            self,
            "perturbation_order_policy",
            self.perturbation_order_policy.strip(),
        )
        object.__setattr__(
            self,
            "baseline_recipe",
            freeze_value(dict(self.baseline_recipe), path="baseline_recipe"),
        )
        object.__setattr__(
            self,
            "baseline_qc",
            freeze_value(dict(self.baseline_qc), path="baseline_qc"),
        )
        object.__setattr__(self, "warnings", warnings)

    @property
    def n_spectra(self) -> int:
        return int(self.spectra.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.spectra.shape[1])

    @property
    def n_wavenumbers(self) -> int:
        return self.n_points

    @property
    def axis_direction(self) -> AxisDirection:
        return self.current_axis_direction

    def to_metadata_dict(self, *, schema_version: str = "1.0") -> dict[str, Any]:
        """Return detached JSON-ready metadata for prepared CSV sidecars."""

        if not isinstance(schema_version, str) or not schema_version.strip():
            raise ValueError("schema_version must be a non-empty string")
        return {
            "schema_version": schema_version,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "baseline_run_id": self.baseline_run_id,
            "baseline_fingerprint": self.baseline_fingerprint,
            "prepared_data_sha256": self.prepared_data_sha256,
            "unit": self.intensity_unit,
            "intensity_unit": self.intensity_unit,
            "axis_direction": self.current_axis_direction,
            "original_axis_direction": self.original_axis_direction,
            "current_axis_direction": self.current_axis_direction,
            "perturbation_order_policy": self.perturbation_order_policy,
            "perturbation_values": self.perturbation.tolist(),
            "perturbation_labels": list(self.perturbation_labels),
            "normalization_state": self.normalization_state,
            "baseline_recipe": thaw_mapping(self.baseline_recipe),
            "qc_summary": thaw_mapping(self.baseline_qc),
            "baseline_qc": thaw_mapping(self.baseline_qc),
            "warnings": list(self.warnings),
            "shape": [self.n_spectra, self.n_points],
        }


__all__ = [
    "AxisDirection",
    "FloatArray",
    "NormalizationState",
    "PreparedSpectralDataset",
]
