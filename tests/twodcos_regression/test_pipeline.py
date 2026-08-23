from __future__ import annotations

import json
import zipfile

import numpy as np

from ftir2dcos.config import BaselineConfig, PipelineConfig
from ftir2dcos.models import SpectralDataset
from ftir2dcos.pipeline import preprocess_dataset, run_pipeline


def make_dataset() -> SpectralDataset:
    wavenumber = np.linspace(1800.0, 1500.0, 31, dtype=np.float64)
    perturbation = np.array([10.0, 0.0, 5.0, 20.0, 30.0])
    labels = ("10 min", "0 min", "5 min", "20 min", "30 min")
    peak = np.exp(-0.5 * ((wavenumber - 1630.0) / 28.0) ** 2)
    shoulder = np.exp(-0.5 * ((wavenumber - 1550.0) / 20.0) ** 2)
    spectra = np.vstack(
        [
            0.04
            + 0.00005 * (wavenumber - 1500.0)
            + (0.4 + value / 100.0) * peak
            - (value / 300.0) * shoulder
            for value in perturbation
        ]
    )
    return SpectralDataset(
        wavenumber=wavenumber,
        perturbation=perturbation,
        perturbation_labels=labels,
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="synthetic.csv",
        metadata={"original_wavenumber_direction": "descending"},
    )


def test_preprocess_is_explicit_and_does_not_mutate_input() -> None:
    dataset = make_dataset()
    original_axis = dataset.wavenumber.copy()
    original_spectra = dataset.spectra.copy()
    config = PipelineConfig(
        low_wavenumber=1530,
        high_wavenumber=1760,
        perturbation_order="sort_by_perturbation",
        baseline=BaselineConfig(method="none"),
    )

    result = preprocess_dataset(dataset, config)

    np.testing.assert_array_equal(dataset.wavenumber, original_axis)
    np.testing.assert_array_equal(dataset.spectra, original_spectra)
    np.testing.assert_array_equal(result.processed.perturbation, [0, 5, 10, 20, 30])
    assert np.all(np.diff(result.processed.wavenumber) > 0)
    assert result.selected_raw.wavenumber[0] >= 1530
    assert result.selected_raw.wavenumber[-1] <= 1760
    np.testing.assert_array_equal(result.baselines, np.zeros_like(result.baselines))
    np.testing.assert_array_equal(result.baseline_corrected.spectra, result.processed.spectra)
    assert result.processed.metadata["smoothing_applied"] is False
    assert result.processed.metadata["normalization_applied"] is False


def test_pipeline_is_deterministic_and_qc_passes() -> None:
    dataset = make_dataset()
    config = PipelineConfig(
        low_wavenumber=1500,
        high_wavenumber=1800,
        perturbation_order="sort_by_perturbation",
        convention="2dpy_compatible",
    )

    first = run_pipeline(dataset, config)
    second = run_pipeline(dataset, PipelineConfig.from_dict(config.to_dict()))

    np.testing.assert_array_equal(first.processed.spectra, second.processed.spectra)
    np.testing.assert_allclose(first.twodcos.synchronous, second.twodcos.synchronous)
    np.testing.assert_allclose(first.twodcos.asynchronous, second.twodcos.asynchronous)
    assert first.qc_metrics["all_checks_passed"] is True
    assert first.twodcos.row_variable == "nu2"
    assert first.twodcos.column_variable == "nu1"


def test_canonical_and_2dpy_async_sign_relation() -> None:
    dataset = make_dataset()
    canonical = run_pipeline(dataset, PipelineConfig(convention="canonical"))
    compatible = run_pipeline(dataset, PipelineConfig(convention="2dpy_compatible"))

    np.testing.assert_allclose(
        compatible.twodcos.asynchronous,
        canonical.twodcos.asynchronous.T,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compatible.twodcos.asynchronous,
        -canonical.twodcos.asynchronous,
        rtol=1e-12,
        atol=1e-12,
    )


def test_pipeline_full_export(tmp_path) -> None:
    config = PipelineConfig(
        low_wavenumber=1500,
        high_wavenumber=1800,
        perturbation_order="sort_by_perturbation",
        display_percentile=100,
    )
    result = run_pipeline(make_dataset(), config, output_root=tmp_path)

    assert result.output_directory is not None
    assert result.bundle_path is not None and result.bundle_path.is_file()
    manifest = json.loads((result.output_directory / "manifest.json").read_text())
    assert manifest["final_data_shape"] == [5, 31]
    assert manifest["convention"] == "2dpy_compatible"
    assert manifest["qc_metrics"]["all_checks_passed"] is True
    assert manifest["baseline"]["method"] == "none"
    assert manifest["perturbation_final_values"] == [0.0, 5.0, 10.0, 20.0, 30.0]
    assert manifest["perturbation_final_intervals"] == [5.0, 5.0, 10.0, 10.0]
    assert manifest["perturbation_approximately_equally_spaced"] is False
    assert "Final processing-order" in manifest["nonuniform_perturbation_warning"]
    with zipfile.ZipFile(result.bundle_path) as archive:
        names = set(archive.namelist())
    assert "data/09_synchronous_matrix.csv" in names
    assert "data/10_asynchronous_matrix.csv" in names
    assert "figures/baseline_qc.png" in names
    assert "manifest.json" in names
