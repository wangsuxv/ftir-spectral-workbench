from __future__ import annotations

import numpy as np
import pytest

from ftir2dcos.config import BaselineConfig, PipelineConfig
from ftir2dcos.conversion import convert_to_absorbance, transmittance_to_absorbance
from ftir2dcos.models import SpectralDataset
from ftir2dcos.validation import (
    apply_perturbation_order,
    ensure_ascending_wavenumber,
    select_wavenumber_range,
    validate_dataset,
)


def _dataset(
    *,
    wavenumber: np.ndarray | None = None,
    perturbation: np.ndarray | None = None,
    spectra: np.ndarray | None = None,
    unit: str = "absorbance",
) -> SpectralDataset:
    x = np.linspace(1800.0, 900.0, 20) if wavenumber is None else wavenumber
    p = np.array([0.0, 1.0, 3.0, 6.0, 10.0]) if perturbation is None else perturbation
    if spectra is None:
        spectra = np.vstack([np.sin(x / 100 + index) for index in range(p.size)])
    return SpectralDataset(
        wavenumber=x,
        perturbation=p,
        perturbation_labels=tuple(str(value) for value in p),
        spectra=spectra,
        intensity_unit=unit,
        source_name="synthetic",
        metadata={},
    )


def test_model_defensively_copies_and_protects_float64_arrays() -> None:
    raw = np.arange(100, dtype=np.float32).reshape(5, 20)
    dataset = _dataset(spectra=raw)
    raw[0, 0] = -999

    assert dataset.spectra.dtype == np.float64
    assert dataset.spectra[0, 0] != -999
    with pytest.raises(ValueError):
        dataset.spectra[0, 0] = 2.0


def test_validation_reports_direction_and_nonuniform_grid_without_resampling() -> None:
    report = validate_dataset(_dataset())

    assert report.is_valid
    assert report.metrics["wavenumber_direction"] == "descending"
    assert report.metrics["perturbation_approximately_equally_spaced"] is False
    assert report.metrics["hilbert_grid_strategy"] == "index_order"
    assert any("non-uniform" in warning for warning in report.warnings)


@pytest.mark.parametrize("defect", ["duplicate", "non_monotonic", "nan"])
def test_invalid_wavenumbers_are_not_silently_fixed(defect: str) -> None:
    x = np.linspace(900.0, 1800.0, 20)
    if defect == "duplicate":
        x[4] = x[3]
    elif defect == "non_monotonic":
        x[4], x[5] = x[5], x[4]
    else:
        x[4] = np.nan

    report = validate_dataset(_dataset(wavenumber=x))

    assert not report.is_valid
    with pytest.raises(ValueError, match="validation failed"):
        report.raise_for_errors()


def test_nan_inf_and_missing_perturbation_are_blocking_errors() -> None:
    spectra = np.ones((5, 20))
    spectra[1, 2] = np.nan
    spectra[2, 4] = np.inf
    perturbation = np.array([0.0, 1.0, np.nan, 3.0, 4.0])

    report = validate_dataset(_dataset(spectra=spectra, perturbation=perturbation))

    assert not report.is_valid
    assert any("NaN" in error for error in report.errors)
    assert any("Inf" in error for error in report.errors)
    assert any("Perturbation" in error for error in report.errors)


def test_order_and_wavenumber_reversal_are_explicit_and_recorded() -> None:
    dataset = _dataset(perturbation=np.array([10.0, 0.0, 5.0, 20.0, 15.0]))
    unchanged = apply_perturbation_order(dataset)
    ordered = apply_perturbation_order(dataset, "sort_by_perturbation")
    ascending = ensure_ascending_wavenumber(ordered)

    np.testing.assert_array_equal(unchanged.perturbation, dataset.perturbation)
    np.testing.assert_array_equal(ordered.perturbation, [0.0, 5.0, 10.0, 15.0, 20.0])
    assert ordered.metadata["perturbation_order_indices"] == [1, 2, 0, 4, 3]
    assert ascending.wavenumber[0] < ascending.wavenumber[-1]
    assert ascending.metadata["original_wavenumber_direction"] == "descending"
    assert ascending.metadata["wavenumber_order_changed"] is True
    np.testing.assert_array_equal(dataset.perturbation, [10.0, 0.0, 5.0, 20.0, 15.0])


def test_range_accepts_either_input_order_and_enforces_minimum_points() -> None:
    dataset = _dataset(wavenumber=np.linspace(1800.0, 900.0, 101))
    selected = select_wavenumber_range(dataset, 1050.0, 1650.0)
    reverse_input = select_wavenumber_range(dataset, 1650.0, 1050.0)

    np.testing.assert_array_equal(selected.wavenumber, reverse_input.wavenumber)
    assert selected.wavenumber[0] >= 1050.0
    assert selected.wavenumber[-1] <= 1650.0
    assert selected.metadata["selected_wavenumber_range"] == [1050.0, 1650.0]
    with pytest.raises(ValueError, match="at least 10"):
        select_wavenumber_range(dataset, 1000.0, 1020.0)
    with pytest.raises(ValueError, match="outside the available range"):
        select_wavenumber_range(dataset, 800.0, 1200.0)


def test_transmittance_conversion_is_explicit_correct_and_non_mutating() -> None:
    percent = np.array([[100.0] * 20, [10.0] * 20, [1.0] * 20])
    dataset = _dataset(
        perturbation=np.array([0.0, 1.0, 2.0]),
        spectra=percent,
        unit="percent_transmittance",
    )
    converted = convert_to_absorbance(dataset)

    np.testing.assert_allclose(converted.spectra[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(dataset.spectra, percent)
    assert converted.intensity_unit == "absorbance"
    assert converted.metadata["intensity_conversion_applied"] is True
    with pytest.raises(ValueError, match="strictly greater than zero"):
        transmittance_to_absorbance([50.0, 0.0], "percent_transmittance")


def test_pipeline_config_json_roundtrip_and_range_alias() -> None:
    config = PipelineConfig(
        low_wavenumber=1736,
        high_wavenumber=1509,
        perturbation_order="sort_by_perturbation",
        baseline=BaselineConfig(method="asls", asls_lam=2e6),
    )

    restored = PipelineConfig.from_json(config.to_json())
    alias = PipelineConfig.from_dict(
        {"wavenumber_range": [1509, 1736], "input_intensity_unit": "absorbance"}
    )

    assert restored == config
    assert config.wavenumber_range == (1509.0, 1736.0)
    assert alias.wavenumber_range == (1509.0, 1736.0)
