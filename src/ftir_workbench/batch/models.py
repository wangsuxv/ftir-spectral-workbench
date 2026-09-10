"""Workspace-owned independent records, drafts, and immutable stage snapshots."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from pydantic import Field

from ftir_baseline.config import (
    CoarseBaselineConfig,
    FineBaselineConfig,
    IntensityUnit,
    PipelineConfig,
    RecipeModel,
    SmoothingConfig,
)
from ftir_baseline.models import FloatArray, SpectrumSet, freeze_value, immutable_float64
from ftir_baseline.pipeline import PipelineResult

Stage = Literal["coarse", "fine"]
FineDecision = Literal["not_decided", "applied", "explicitly_skipped"]


class BatchError(ValueError):
    """A stable machine-readable error code and a useful human explanation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class PreparationConfig(RecipeModel):
    input_unit: IntensityUnit = "absorbance"
    wavenumber_range: tuple[float, float]
    baseline_smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    transmittance_floor: float | None = Field(default=None, gt=0.0)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.wavenumber_range[0] == self.wavenumber_range[1]:
            raise BatchError("RANGE_INVALID", "range bounds must differ")


def scientific_input_hash(x: FloatArray, y: FloatArray, unit: str) -> str:
    """Hash this column's numeric input and interpretation, independent of labels."""

    digest = hashlib.sha256(b"independent-spectrum-input-v1\0")
    for array in (x, y):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    digest.update(unit.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ImportedSource:
    source_id: str
    original_filename: str
    original_bytes_sha256: str
    original_bytes: bytes
    default_input_unit: IntensityUnit
    import_options: Mapping[str, Any]
    import_probe: Mapping[str, Any]

    def __post_init__(self) -> None:
        if hashlib.sha256(self.original_bytes).hexdigest() != self.original_bytes_sha256:
            raise BatchError("WORKSPACE_INTEGRITY_FAILED", "source bytes hash mismatch")
        object.__setattr__(self, "original_bytes", bytes(self.original_bytes))
        object.__setattr__(self, "import_options", freeze_value(dict(self.import_options)))
        object.__setattr__(self, "import_probe", freeze_value(dict(self.import_probe)))


@dataclass(frozen=True, slots=True)
class SpectrumRecord:
    spectrum_id: str
    source_id: str
    original_column_index: int
    original_column_label: str
    display_name: str
    wavenumber: FloatArray
    raw_intensity: FloatArray
    confirmed_input_unit: IntensityUnit
    original_axis_direction: str
    scientific_input_sha256: str = ""
    duplicate_candidate: bool = False
    excluded: bool = False

    def __post_init__(self) -> None:
        # Keep the core's strict axis/finite/shape contract, with a neutral placeholder.
        validated = SpectrumSet(
            wavenumber=self.wavenumber,
            spectra=np.asarray(self.raw_intensity)[np.newaxis, :],
            perturbation=np.array([0.0]),
            perturbation_labels=("independent",),
            intensity_unit=self.confirmed_input_unit,
            source_name="independent input validation",
        )
        if self.original_column_index < 1:
            raise ValueError("intensity column indexes start at 1; x is column 0")
        if self.original_axis_direction != validated.axis_direction:
            raise BatchError("WORKSPACE_INTEGRITY_FAILED", "original axis direction mismatch")
        actual = scientific_input_hash(
            validated.wavenumber, validated.spectra[0], self.confirmed_input_unit
        )
        if self.scientific_input_sha256 and actual != self.scientific_input_sha256:
            raise BatchError("WORKSPACE_INTEGRITY_FAILED", "scientific input hash mismatch")
        object.__setattr__(self, "wavenumber", validated.wavenumber)
        object.__setattr__(self, "raw_intensity", immutable_float64(validated.spectra[0], name="y"))
        object.__setattr__(self, "scientific_input_sha256", actual)


@dataclass(frozen=True, slots=True)
class StageSnapshot:
    stage: Stage
    spectrum_id: str
    input_sha256: str
    config: PipelineConfig
    fingerprint: str
    result: PipelineResult
    parent_coarse_fingerprint: str | None = None
    implementation_fingerprint: str = ""
    parent_coarse_implementation_fingerprint: str | None = None


@dataclass(slots=True)
class SpectrumProcessingState:
    preparation_draft: PreparationConfig
    preparation_committed: PreparationConfig
    coarse_draft: CoarseBaselineConfig = field(default_factory=CoarseBaselineConfig)
    coarse_committed: CoarseBaselineConfig | None = None
    fine_draft: FineBaselineConfig = field(default_factory=FineBaselineConfig)
    fine_committed: FineBaselineConfig | None = None
    coarse_preview: StageSnapshot | None = None
    coarse_snapshot: StageSnapshot | None = None
    fine_preview: StageSnapshot | None = None
    fine_snapshot: StageSnapshot | None = None
    fine_decision: FineDecision = "not_decided"
    fine_parent_coarse_fingerprint: str | None = None
    coarse_stale: bool = False
    fine_stale: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    display_preferences: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BatchWorkspace:
    schema_version: str = "1.0"
    workspace_id: str = field(default_factory=lambda: uuid4().hex)
    workflow_mode: Literal["independent_batch"] = "independent_batch"
    sources: dict[str, ImportedSource] = field(default_factory=dict)
    records: dict[str, SpectrumRecord] = field(default_factory=dict)
    states: dict[str, SpectrumProcessingState] = field(default_factory=dict)
    display_order: list[str] = field(default_factory=list)
    selected_spectrum_id: str | None = None
    import_issues: list[dict[str, Any]] = field(default_factory=list)
    export_history: list[dict[str, Any]] = field(default_factory=list)
    last_export_summary: dict[str, Any] | None = None

