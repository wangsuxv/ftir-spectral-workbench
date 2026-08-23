from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from ftir_baseline.config import PipelineConfig
from ftir_baseline.models import SpectrumSet
from ftir_baseline.pipeline import run_pipeline
from ftir_workbench.adapters import (
    prepared_from_baseline_result,
    prepared_scientific_branch_from_baseline_result,
)
from ftir_workbench.fingerprints import prepared_data_sha256
from ftir_workbench.models import PreparedSpectralDataset
from ftir_workbench.validation import (
    PreparedDatasetValidationError,
    validate_prepared_dataset,
)


def _prepared(
    *,
    ascending: bool = False,
    source_sha256: str = "1" * 64,
    baseline_fingerprint: str = "2" * 64,
    baseline_run_id: str = "run-1",
    warnings: tuple[str, ...] = (),
) -> PreparedSpectralDataset:
    axis = np.array([1800, 1700, 1600, 1500], dtype=np.int32)
    spectra = np.array(
        [[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12]],
        dtype=np.float32,
    )
    if ascending:
        axis = axis[::-1]
        spectra = spectra[:, ::-1]
    perturbation = np.array([0, 2, 5], dtype=np.int16)
    labels = ("0 分钟", "2 分钟", "5 分钟")
    digest = prepared_data_sha256(axis, perturbation, labels, spectra)
    return PreparedSpectralDataset(
        wavenumber=axis,
        perturbation=perturbation,
        perturbation_labels=labels,
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="中文 路径/sample.csv",
        source_sha256=source_sha256,
        baseline_run_id=baseline_run_id,
        baseline_fingerprint=baseline_fingerprint,
        prepared_data_sha256=digest,
        original_axis_direction="descending",
        current_axis_direction="ascending" if ascending else "descending",
        perturbation_order_policy="preserve_file_order",
        baseline_recipe={"nested": {"values": [1, 2]}},
        baseline_qc={"passed": True},
        warnings=warnings,
    )


def _baseline_result(normalization: str = "none"):
    axis = np.linspace(1800.0, 900.0, 51)
    peak = np.exp(-0.5 * ((axis - 1250.0) / 35.0) ** 2)
    baseline = 0.01 + (1800.0 - axis) * 1.0e-5
    spectra = np.vstack([baseline + scale * peak for scale in (0.2, 0.35, 0.5)])
    data = SpectrumSet(
        wavenumber=axis,
        perturbation=np.array([0.0, 2.0, 5.0]),
        perturbation_labels=("0 min", "2 min", "5 min"),
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="合成数据.csv",
        metadata={"perturbation_order_policy": "preserve_file_order"},
    )
    config = PipelineConfig(
        input_unit="absorbance",
        wavenumber_range=(1800.0, 900.0),
        series_mode="independent_locked",
        coarse_baseline={"method": "none"},
        fine_baseline={
            "enabled": True,
            "method": "endpoint_window_linear",
            "endpoint_window_width_cm1": 40.0,
        },
        normalization={"method": normalization},
    )
    return run_pipeline(data, config)


@pytest.mark.parametrize("ascending", [False, True])
def test_prepared_is_float64_deeply_immutable_and_preserves_axis(ascending: bool) -> None:
    prepared = _prepared(ascending=ascending)

    assert prepared.wavenumber.dtype == np.float64
    assert prepared.perturbation.dtype == np.float64
    assert prepared.spectra.dtype == np.float64
    assert prepared.current_axis_direction == ("ascending" if ascending else "descending")
    assert prepared.spectra.shape == (3, 4)
    assert not prepared.wavenumber.flags.writeable
    assert not prepared.perturbation.flags.writeable
    assert not prepared.spectra.flags.writeable

    with pytest.raises(ValueError):
        prepared.spectra.flags.writeable = True
    with pytest.raises(TypeError):
        prepared.baseline_recipe["new"] = 1
    with pytest.raises(TypeError):
        prepared.baseline_recipe["nested"]["new"] = 1
    with pytest.raises(FrozenInstanceError):
        prepared.source_name = "changed"  # type: ignore[misc]
    validate_prepared_dataset(prepared)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"spectra": [[1.0, 2.0]]}, "shape"),
        ({"spectra": np.full((3, 4), np.nan)}, "NaN"),
        ({"wavenumber": [1800, 1600, 1700, 1500]}, "monotonic"),
        ({"perturbation": [0, 1]}, "perturbation must have shape"),
        ({"perturbation_labels": ("0", "1")}, "labels length"),
        ({"perturbation_labels": ("0", "", "2")}, "must not be empty"),
        ({"current_axis_direction": "ascending"}, "does not match"),
        ({"intensity_unit": "percent_transmittance"}, "absorbance"),
        ({"prepared_data_sha256": "0" * 64}, "does not match"),
    ],
)
def test_prepared_rejects_contract_violations(
    updates: dict[str, object],
    match: str,
) -> None:
    valid = _prepared().to_metadata_dict()
    axis = np.array([1800.0, 1700.0, 1600.0, 1500.0])
    spectra = np.array([[1, 2, 3, 4], [2, 4, 6, 8], [3, 6, 9, 12]], dtype=float)
    values: dict[str, object] = {
        "wavenumber": axis,
        "perturbation": np.array([0.0, 2.0, 5.0]),
        "perturbation_labels": ("0 分钟", "2 分钟", "5 分钟"),
        "spectra": spectra,
        "intensity_unit": "absorbance",
        "source_name": valid["source_name"],
        "source_sha256": valid["source_sha256"],
        "baseline_run_id": valid["baseline_run_id"],
        "baseline_fingerprint": valid["baseline_fingerprint"],
        "prepared_data_sha256": prepared_data_sha256(
            axis,
            np.array([0.0, 2.0, 5.0]),
            ("0 分钟", "2 分钟", "5 分钟"),
            spectra,
        ),
        "original_axis_direction": "descending",
        "current_axis_direction": "descending",
        "perturbation_order_policy": "preserve_file_order",
        "baseline_recipe": {},
        "baseline_qc": {},
        "warnings": (),
    }
    values.update(updates)
    with pytest.raises(PreparedDatasetValidationError, match=match):
        PreparedSpectralDataset(**values)  # type: ignore[arg-type]


def test_incomplete_provenance_requires_and_preserves_warning() -> None:
    with pytest.raises(PreparedDatasetValidationError, match="provenance"):
        _prepared(
            source_sha256="unknown",
            baseline_fingerprint="unknown",
            baseline_run_id="unknown",
        )
    prepared = _prepared(
        source_sha256="unknown",
        baseline_fingerprint="unknown",
        baseline_run_id="unknown",
        warnings=("Provenance incomplete for bare corrected CSV.",),
    )
    assert prepared.source_sha256 == "unknown"


def test_primary_adapter_uses_analysis_data_and_never_display_data() -> None:
    result = _baseline_result("minmax_display")
    assert not np.array_equal(result.analysis_data, result.view_data)

    prepared = prepared_from_baseline_result(result, baseline_run_id="baseline-main")

    np.testing.assert_array_equal(prepared.spectra, result.analysis_data)
    assert not np.array_equal(prepared.spectra, result.view_data)
    assert prepared.normalization_state == "display_only"
    assert prepared.baseline_recipe["prepared_data_contract"]["source_channel"] == (
        "PipelineResult.analysis_data"
    )


def test_explicit_scientific_normalization_is_a_distinct_branch() -> None:
    result = _baseline_result("vector")
    main = prepared_from_baseline_result(result, baseline_run_id="baseline-1")
    branch = prepared_scientific_branch_from_baseline_result(
        result,
        baseline_run_id="baseline-1",
        branch_name="vector sensitivity",
    )

    assert result.normalization.optional_normalized is not None
    np.testing.assert_array_equal(branch.spectra, result.normalization.optional_normalized)
    assert branch.normalization_state == "scientific_explicit"
    assert branch.prepared_data_sha256 != main.prepared_data_sha256
    assert branch.baseline_fingerprint != main.baseline_fingerprint
    assert any("sensitivity branch" in warning for warning in branch.warnings)


def test_display_only_normalization_does_not_invalidate_primary_science() -> None:
    without_display = prepared_from_baseline_result(
        _baseline_result("none"),
        baseline_run_id="baseline-none",
    )
    with_display = prepared_from_baseline_result(
        _baseline_result("minmax_display"),
        baseline_run_id="baseline-display",
    )

    np.testing.assert_array_equal(without_display.spectra, with_display.spectra)
    assert without_display.prepared_data_sha256 == with_display.prepared_data_sha256
    assert without_display.baseline_fingerprint == with_display.baseline_fingerprint
