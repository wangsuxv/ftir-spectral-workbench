from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from ftir_baseline.config import PipelineConfig
from ftir_baseline.io import read_spectrum_file
from ftir_baseline.pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_ROOT = PROJECT_ROOT / "ui"
sys.path.insert(0, str(UI_ROOT))

from components.baseline_preview import (  # noqa: E402
    add_anchor_overlays,
    anchor_diagnostics,
    anchor_diagnostics_table,
    baseline_preview_payload,
    coarse_preview_figure,
    fine_decomposition_figure,
    fine_residual_figure,
    representative_options,
    resolve_representative,
)
from components.series_qc import (  # noqa: E402
    REQUIRED_QC_FIELDS,
    complete_qc_table,
    drill_down_figure,
    drill_down_payload,
    filter_qc_table,
    five_heatmap_figures,
    qc_table_csv,
    qc_trend_figures,
)


@pytest.fixture(scope="module")
def result():  # type: ignore[no-untyped-def]
    data = read_spectrum_file(
        PROJECT_ROOT / "examples" / "baseline" / "example_absorbance.csv",
        input_unit="absorbance",
    )
    return run_pipeline(
        data,
        PipelineConfig(
            input_unit="absorbance",
            wavenumber_range=(1800.0, 900.0),
            series_mode="independent_locked",
            coarse_baseline={"method": "none"},
            fine_baseline={
                "enabled": True,
                "method": "endpoint_window_linear",
                "endpoint_window_width_cm1": 250.0,
            },
            normalization={"method": "none"},
        ),
    )


def test_representative_options_resolve_actual_rows_and_aggregates(result) -> None:  # type: ignore[no-untyped-def]
    labels = result.absorbance_selected.perturbation_labels
    options = representative_options(labels)

    assert [item.key for item in options[:5]] == [
        "first",
        "middle",
        "last",
        "mean",
        "median",
    ]
    assert [item.key for item in options[5:]] == ["spectrum:0", "spectrum:1", "spectrum:2"]
    assert resolve_representative(labels, labels[1]).spectrum_index == 1

    actual = baseline_preview_payload(result, "spectrum:1")
    mean = baseline_preview_payload(result, "mean")
    np.testing.assert_array_equal(actual.raw_absorbance, result.absorbance_selected.spectra[1])
    np.testing.assert_array_equal(actual.corrected, result.analysis_data[1])
    np.testing.assert_allclose(
        mean.raw_absorbance,
        np.mean(result.absorbance_selected.spectra, axis=0),
    )
    assert not actual.raw_absorbance.flags.writeable


def test_preview_figures_bind_pipeline_arrays_and_only_authorized_residual(result) -> None:  # type: ignore[no-untyped-def]
    index = 1
    payload = baseline_preview_payload(result, index)
    coarse = coarse_preview_figure(result, index)
    decomposition = fine_decomposition_figure(result, index)
    residual = fine_residual_figure(result, index)

    assert [trace.name for trace in coarse.data] == [
        "Raw selected absorbance",
        "Baseline-estimation channel",
        "Coarse baseline",
        "Residual after coarse baseline",
    ]
    np.testing.assert_array_equal(coarse.data[0].y, payload.raw_absorbance)
    np.testing.assert_array_equal(coarse.data[1].y, payload.baseline_estimation)
    np.testing.assert_array_equal(coarse.data[2].y, payload.coarse_baseline)
    np.testing.assert_array_equal(
        coarse.data[3].y,
        result.absorbance_selected.spectra[index] - result.baseline.coarse_baseline[index],
    )
    assert [trace.name for trace in decomposition.data] == [
        "A_raw",
        "A_for_baseline",
        "B_coarse",
        "B_fine",
        "B_total",
        "Corrected",
    ]
    assert [trace.name for trace in residual.data] == [
        "Residual after coarse",
        "B_fine",
        "Final corrected",
    ]
    np.testing.assert_array_equal(residual.data[2].y, result.analysis_data[index])


def test_anchor_diagnostics_use_fitted_endpoint_values_without_refitting(result) -> None:  # type: ignore[no-untyped-def]
    diagnostics = anchor_diagnostics(result, 1)
    table = anchor_diagnostics_table(result, 1)
    fitted = result.baseline.params["fine"]["fitted"]

    assert len(diagnostics) == 2
    np.testing.assert_array_equal(
        [item.representative_wavenumber for item in diagnostics],
        fitted["representative_wavenumbers"],
    )
    np.testing.assert_array_equal(
        [item.representative_value for item in diagnostics],
        [fitted["lower_values"][1], fitted["upper_values"][1]],
    )
    assert list(table.columns) == [
        "Anchor",
        "Start",
        "End",
        "Statistic",
        "Representative wavenumber",
        "Representative value (B_fine)",
    ]
    base = fine_decomposition_figure(result, 1)
    rendered = add_anchor_overlays(base, diagnostics)
    assert len(rendered.layout.shapes) == 2
    assert rendered.data[-1].name == "Anchor statistic on B_fine"


def test_five_heatmaps_bind_exact_arrays_and_center_corrected_at_zero(result) -> None:  # type: ignore[no-untyped-def]
    figures = five_heatmap_figures(result)
    expected = {
        "Raw absorbance": result.absorbance_selected.spectra,
        "Coarse baseline": result.baseline.coarse_baseline,
        "Fine baseline": result.baseline.fine_baseline,
        "Total baseline": result.baseline.total_baseline,
        "Corrected absorbance": result.analysis_data,
    }

    assert tuple(figures) == tuple(expected)
    for name, matrix in expected.items():
        np.testing.assert_array_equal(figures[name].data[0].z, matrix)
    corrected = figures["Corrected absorbance"].data[0]
    scale = float(np.max(np.abs(result.analysis_data)))
    assert corrected.zmid == 0.0
    assert corrected.zmin == -scale
    assert corrected.zmax == scale
    assert figures["Raw absorbance"].data[0].zmid is None


def test_complete_qc_table_preserves_every_field_and_label_alignment(result) -> None:  # type: ignore[no-untyped-def]
    table = complete_qc_table(result)

    assert set(REQUIRED_QC_FIELDS).issubset(table.columns)
    assert table["perturbation_label"].tolist() == list(
        result.absorbance_selected.perturbation_labels
    )
    for field, values in result.qc.per_spectrum.items():
        np.testing.assert_array_equal(table[field].to_numpy(), values)

    filtered = filter_qc_table(table, query=result.absorbance_selected.perturbation_labels[1])
    assert filtered["spectrum_index"].astype(int).tolist() == [1]
    assert qc_table_csv(table).startswith(b"spectrum_index,perturbation,perturbation_label")


def test_qc_trends_and_drill_down_reuse_existing_rows(result) -> None:  # type: ignore[no-untyped-def]
    figures = qc_trend_figures(result)
    assert tuple(figures) == (
        "Baseline continuity",
        "Residual diagnostics",
        "Peak preservation",
    )
    np.testing.assert_array_equal(
        figures["Baseline continuity"].data[0].y,
        result.qc.per_spectrum["baseline_area"],
    )
    np.testing.assert_array_equal(
        figures["Residual diagnostics"].data[1].y,
        result.qc.per_spectrum["negative_fraction"],
    )
    np.testing.assert_array_equal(
        figures["Peak preservation"].data[2].y,
        result.qc.per_spectrum["peak_height_relative_change"],
    )

    payload, row = drill_down_payload(result, "middle")
    assert payload.selection.spectrum_index == 1
    assert int(row["spectrum_index"]) == 1
    assert row["perturbation_label"] == result.absorbance_selected.perturbation_labels[1]
    figure = drill_down_figure(result, 1)
    assert [trace.name for trace in figure.data] == [
        "A_raw",
        "B_coarse",
        "B_fine",
        "B_total",
        "Corrected",
    ]
    with pytest.raises(ValueError, match="actual spectrum row"):
        drill_down_payload(result, "median")
