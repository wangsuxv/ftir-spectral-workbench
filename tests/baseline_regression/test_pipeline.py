from __future__ import annotations

import json

import numpy as np
import pytest

from ftir_baseline.config import PipelineConfig
from ftir_baseline.models import SpectrumSet
from ftir_baseline.pipeline import pipeline_result_fingerprint, run_pipeline


def make_series(unit: str = "absorbance") -> SpectrumSet:
    x = np.linspace(1800.0, 900.0, 451)
    baseline = 0.02 + 3e-5 * (1800.0 - x)
    peak = 0.2 * np.exp(-0.5 * ((x - 1250.0) / 25.0) ** 2)
    absorbance = np.vstack([baseline + scale * peak for scale in (0.8, 1.0, 1.3)])
    spectra = 100.0 * 10.0 ** (-absorbance) if unit == "percent_transmittance" else absorbance
    return SpectrumSet(
        wavenumber=x,
        perturbation=np.array([0.0, 1.0, 2.0]),
        perturbation_labels=("0 min", "1 min", "2 min"),
        spectra=spectra,
        intensity_unit=unit,
        source_name="synthetic",
        metadata={},
    )


def endpoint_config(unit: str = "absorbance", *, smoothing: bool = False) -> PipelineConfig:
    return PipelineConfig(
        input_unit=unit,
        wavenumber_range=(1800.0, 900.0),
        series_mode="independent_locked",
        baseline_smoothing={
            "enabled": smoothing,
            "window_length": 7,
            "polyorder": 2,
            "estimate_only": True,
        },
        coarse_baseline={"method": "none"},
        fine_baseline={
            "enabled": True,
            "method": "endpoint_window_linear",
            "endpoint_window_width_cm1": 8.0,
        },
        normalization={"method": "none"},
    )


def test_pipeline_order_reconstruction_and_branches() -> None:
    data = make_series()
    original = data.spectra.copy()
    result = run_pipeline(data, endpoint_config())

    assert np.array_equal(data.spectra, original)
    assert np.array_equal(result.normalization.analysis_data, result.baseline.corrected)
    assert np.array_equal(result.normalization.view_data, result.baseline.corrected)
    assert np.allclose(
        result.absorbance_selected.spectra,
        result.baseline.total_baseline + result.baseline.corrected,
        atol=1e-14,
    )
    assert result.qc.summary["reconstruction_passed"] is True
    assert tuple(result.recipe["processing_order"])[0:4] == (
        "raw",
        "unit_confirmation",
        "transmittance_to_absorbance",
        "range_selection",
    )
    assert set(result.sensitivity_branches) == {
        "uncorrected",
        "coarse_only",
        "coarse_plus_fine",
    }


def test_percent_transmittance_is_converted_before_baseline() -> None:
    transmittance = make_series("percent_transmittance")
    absorbance = make_series("absorbance")
    result = run_pipeline(transmittance, endpoint_config("percent_transmittance"))
    assert np.allclose(result.absorbance_full.spectra, absorbance.spectra)
    assert result.unit_conversion.formula == "A = -log10(percent_transmittance / 100)"


def test_smoothing_is_estimate_only() -> None:
    data = make_series()
    noisy = np.array(data.spectra, copy=True)
    noisy[:, 100] += 0.03
    noisy_data = SpectrumSet(
        wavenumber=data.wavenumber,
        perturbation=data.perturbation,
        perturbation_labels=data.perturbation_labels,
        spectra=noisy,
        intensity_unit="absorbance",
        source_name="noisy",
        metadata={},
    )
    result = run_pipeline(noisy_data, endpoint_config(smoothing=True))
    assert not np.array_equal(result.baseline_estimation_spectra, noisy)
    assert np.array_equal(result.sensitivity_branches["uncorrected"], noisy)
    assert np.array_equal(
        result.baseline.corrected,
        noisy - result.baseline.total_baseline,
    )


def test_recipe_reproducibility_is_elementwise() -> None:
    data = make_series()
    config = endpoint_config()
    first = run_pipeline(data, config)
    second = run_pipeline(data, config.to_json())
    replayed_from_full_recipe = run_pipeline(data, json.dumps(first.recipe_dict()))
    assert np.array_equal(first.baseline.corrected, second.baseline.corrected)
    assert np.array_equal(first.baseline.corrected, replayed_from_full_recipe.baseline.corrected)
    assert pipeline_result_fingerprint(first) == pipeline_result_fingerprint(second)
    assert first.recipe["software"]["version"]
    assert first.recipe["input_sha256"]


def test_pipeline_recipe_and_output_branches_are_deeply_immutable() -> None:
    result = run_pipeline(make_series(), endpoint_config())

    with pytest.raises(TypeError):
        result.recipe["config"]["input_unit"] = "percent_transmittance"
    with pytest.raises(ValueError):
        result.analysis_data[0, 0] = 99.0
    with pytest.raises(ValueError):
        result.normalization.factors[0] = 99.0
    with pytest.raises(ValueError):
        result.qc.per_spectrum["anchor_error"][0] = 99.0

    detached = result.recipe_dict()
    detached["config"]["input_unit"] = "percent_transmittance"
    assert result.recipe["config"]["input_unit"] == "absorbance"


def test_shared_shape_rejects_incompatible_fine_semantics() -> None:
    data = make_series()
    disabled = PipelineConfig(
        input_unit="absorbance",
        series_mode="shared_shape",
        coarse_baseline={"method": "arpls"},
        fine_baseline={"enabled": False, "method": "none"},
    )
    with pytest.raises(ValueError, match="intrinsically requires"):
        run_pipeline(data, disabled)

    incompatible = PipelineConfig(
        input_unit="absorbance",
        series_mode="shared_shape",
        coarse_baseline={"method": "arpls"},
        fine_baseline={
            "enabled": True,
            "method": "pchip",
            "anchors": [
                {"start": 1796.0, "end": 1804.0},
                {"start": 896.0, "end": 904.0},
            ],
        },
    )
    with pytest.raises(ValueError, match="not piecewise/PCHIP/polynomial"):
        run_pipeline(data, incompatible)


def test_input_unit_mismatch_stops_processing() -> None:
    with pytest.raises(ValueError, match="input_unit does not match"):
        run_pipeline(make_series("absorbance"), endpoint_config("percent_transmittance"))


def test_shared_shape_allows_only_offset_and_slope() -> None:
    data = make_series()
    config = PipelineConfig(
        input_unit="absorbance",
        wavenumber_range=(1800.0, 900.0),
        series_mode="shared_shape",
        coarse_baseline={"method": "arpls", "lambda": 1e5},
        fine_baseline={
            "enabled": True,
            "method": "endpoint_window_linear",
            "endpoint_window_width_cm1": 8.0,
        },
    )
    result = run_pipeline(data, config)
    assert result.baseline.params["degrees_of_freedom_per_spectrum"] == 2
    assert result.baseline.params["allowed_per_spectrum_terms"] == (
        "constant",
        "linear_slope",
    )
    assert result.qc.summary["reconstruction_passed"] is True


def test_shared_shape_honours_endpoint_window_statistic() -> None:
    source = make_series()
    spectra = np.array(source.spectra, copy=True)
    spectra[:, 0] += np.array([0.3, 0.4, 0.5])
    data = SpectrumSet(
        wavenumber=source.wavenumber,
        perturbation=source.perturbation,
        perturbation_labels=source.perturbation_labels,
        spectra=spectra,
        intensity_unit=source.intensity_unit,
        source_name="endpoint outlier",
        metadata={},
    )

    def shared_config(statistic: str) -> PipelineConfig:
        return PipelineConfig(
            input_unit="absorbance",
            wavenumber_range=(1800.0, 900.0),
            series_mode="shared_shape",
            coarse_baseline={"method": "none"},
            fine_baseline={
                "enabled": True,
                "method": "endpoint_window_linear",
                "endpoint_window_width_cm1": 8.0,
                "statistic": statistic,
            },
        )

    median_result = run_pipeline(data, shared_config("median"))
    mean_result = run_pipeline(data, shared_config("mean"))

    assert {window["statistic"] for window in mean_result.baseline.params["anchors"]} == {"mean"}
    assert {window["statistic"] for window in median_result.baseline.params["anchors"]} == {
        "median"
    }
    assert not np.array_equal(
        median_result.baseline.fine_baseline,
        mean_result.baseline.fine_baseline,
    )
