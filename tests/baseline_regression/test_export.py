from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from types import SimpleNamespace

import numpy as np

from ftir_baseline.export import (
    build_2dcos_sensitivity_report,
    build_export_zip,
    export_result,
    verify_export_manifest,
)
from ftir_baseline.normalization import normalize_spectra
from ftir_baseline.qc import run_quality_control


def _pipeline_result(
    *, normalized: bool = True, include_sensitivity: bool = True
) -> SimpleNamespace:
    x = np.array([900.0, 1000.0, 1100.0, 1200.0])
    perturbation = np.array([0.0, 5.0])
    labels = ("0 min", "5 min")
    corrected = np.array([[0.0, 1.0, 0.5, 0.0], [0.0, 2.0, 1.0, 0.0]])
    coarse = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]])
    fine = np.full_like(coarse, 0.01)
    total = coarse + fine
    selected = corrected + total

    def spectrum_set(values: np.ndarray, unit: str) -> SimpleNamespace:
        return SimpleNamespace(
            wavenumber=x,
            spectra=values,
            perturbation=perturbation,
            perturbation_labels=labels,
            intensity_unit=unit,
            source_name="example input.csv",
        )

    raw = spectrum_set(100.0 * 10.0 ** (-selected), "percent_transmittance")
    full = spectrum_set(selected, "absorbance")
    chosen = spectrum_set(selected, "absorbance")
    baseline = SimpleNamespace(
        coarse_baseline=coarse,
        fine_baseline=fine,
        total_baseline=total,
        corrected=corrected,
        params={"method": "test"},
        metrics={"fit_iterations": np.array([2, 3])},
        warnings=(),
    )
    normalization = normalize_spectra(
        x,
        corrected,
        "vector" if normalized else "none",
    )
    qc = run_quality_control(
        x,
        selected,
        total,
        corrected,
        anchor_windows=[(900.0, 1000.0), (1100.0, 1200.0)],
        perturbation=perturbation,
    )
    result = SimpleNamespace(
        raw_input=raw,
        absorbance_full=full,
        absorbance_selected=chosen,
        baseline=baseline,
        normalization=normalization,
        qc=qc,
        config={
            "restore_descending_axis_on_export": True,
            "normalization": {"method": normalization.method},
        },
        recipe={"user_note": "<keep escaped>"},
        input_sha256=hashlib.sha256(b"known input").hexdigest(),
        software_version="test-version",
        warnings=("test warning",),
    )
    if include_sensitivity:
        result.sensitivity_branches = {
            "uncorrected": selected,
            "coarse_only": selected - coarse,
            "coarse_plus_fine": corrected,
        }
    return result


def test_complete_zip_and_manifest_hashes() -> None:
    bundle = build_export_zip(_pipeline_result(), qc_figures={"overview.png": b"png-data"})
    assert verify_export_manifest(bundle)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        names = set(archive.namelist())
        required = {
            "01_raw_input.csv",
            "02_absorbance_full.csv",
            "03_absorbance_selected.csv",
            "04_coarse_baseline.csv",
            "05_fine_baseline.csv",
            "06_total_baseline.csv",
            "07_corrected_absorbance.csv",
            "08_normalized_optional.csv",
            "09_baseline_metrics.csv",
            "10_processing_recipe.json",
            "11_processing_report.html",
            "12_qc_figures/",
            "12_qc_figures/overview.png",
            "13_2dcos_sensitivity_report.json",
            "corrected_absorbance_for_2dcos.csv",
            "normalized_optional_for_sensitivity_analysis.csv",
            "manifest.json",
        }
        assert required <= names

        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["input_sha256"] == hashlib.sha256(b"known input").hexdigest()
        assert len(manifest["manifest_sha256"]) == 64
        listed = {entry["path"]: entry for entry in manifest["files"]}
        for name, entry in listed.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == entry["sha256"]


def test_2dcos_layout_preserves_perturbation_order_and_restores_descending_axis() -> None:
    bundle = build_export_zip(_pipeline_result(), qc_figures={})
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        rows = list(
            csv.reader(io.StringIO(archive.read("corrected_absorbance_for_2dcos.csv").decode()))
        )
    assert rows[0] == ["Wavenumber", "0 min", "5 min"]
    assert [float(row[0]) for row in rows[1:]] == [1200.0, 1100.0, 1000.0, 900.0]
    # Spectrum columns remain in original perturbation order.
    assert [float(rows[1][1]), float(rows[1][2])] == [0.0, 0.0]
    assert [float(rows[2][1]), float(rows[2][2])] == [0.5, 1.0]


def test_disabled_normalization_has_empty_standard_csv_and_no_optional_2dcos() -> None:
    bundle = build_export_zip(_pipeline_result(normalized=False), qc_figures={})
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        names = archive.namelist()
        assert "08_normalized_optional.csv" in names
        assert "normalized_optional_for_sensitivity_analysis.csv" not in names
        header = archive.read("08_normalized_optional.csv").decode().splitlines()[0]
        assert header == "Wavenumber"


def test_recipe_and_html_are_self_contained_and_escape_user_text() -> None:
    bundle = build_export_zip(_pipeline_result(), qc_figures={})
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        recipe = json.loads(archive.read("10_processing_recipe.json"))
        report = archive.read("11_processing_report.html").decode()
    assert recipe["normalization"]["analysis_branch_overwritten"] is False
    assert recipe["processing_order"][0] == "raw"
    assert "Candidate-ranking heuristic" in report
    assert "&lt;keep escaped&gt;" in report
    assert "2D-COS preprocessing sensitivity" in report


def test_export_preserves_pipeline_supplied_processing_order() -> None:
    result = _pipeline_result()
    expected = ["raw", "custom_scientific_step", "quality_control_and_export"]
    result.recipe = {**result.recipe, "processing_order": expected}

    bundle = build_export_zip(result, qc_figures={})
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        recipe = json.loads(archive.read("10_processing_recipe.json"))

    assert recipe["processing_order"] == expected


def test_2dcos_sensitivity_report_mean_centers_over_perturbation_axis() -> None:
    x = np.array([1000.0, 1100.0])
    uncorrected = np.array([[1.0, 3.0], [3.0, 7.0]])
    corrected = np.array([[0.0, 1.0], [2.0, 5.0]])
    report = build_2dcos_sensitivity_report(
        x,
        {"uncorrected": uncorrected, "coarse_plus_fine": corrected},
    )

    expected_dynamic = corrected - np.mean(corrected, axis=0, keepdims=True)
    expected_rms = float(np.sqrt(np.mean(expected_dynamic**2)))
    assert report["mean_centering"]["axis"] == 0
    assert report["perturbation_order_preserved"] is True
    assert (
        report["branch_statistics"]["coarse_plus_fine"]["mean_centered_dynamic_rms"] == expected_rms
    )
    pair = report["pairwise_branch_differences"]["uncorrected__vs__coarse_plus_fine"]
    # Both branches differ only by a time-invariant mean spectrum, so their
    # dynamic spectra are identical after perturbation-axis mean subtraction.
    assert pair["mean_centered_dynamic_difference_rms"] == 0.0
    assert pair["dynamic_flattened_correlation"] == 1.0
    assert "do not prove" in report["scientific_disclaimer"]


def test_sensitivity_report_is_optional_for_older_pipeline_results() -> None:
    bundle = build_export_zip(_pipeline_result(include_sensitivity=False), qc_figures={})
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert "13_2dcos_sensitivity_report.json" not in archive.namelist()
        assert verify_export_manifest(bundle)


def test_export_result_writes_verified_zip(tmp_path) -> None:
    path = export_result(_pipeline_result(), tmp_path, qc_figures={})
    assert path.is_absolute()
    assert path.name == "example_input_processed.zip"
    assert verify_export_manifest(path)
