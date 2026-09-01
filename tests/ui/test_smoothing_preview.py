from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

import ftir_workbench.post_baseline_smoothing as smoothing_core_module
from ftir_workbench import PostBaselineSmoothingConfig, apply_post_baseline_smoothing
from tests.smoothing._helpers import make_prepared

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_ROOT = PROJECT_ROOT / "ui"
sys.path.insert(0, str(UI_ROOT))

from components.smoothing_preview import (  # noqa: E402
    resolve_smoothing_representative,
    smoothing_overlay_figure,
    smoothing_preview_payload,
    smoothing_qc_table,
    smoothing_removed_component_figure,
    smoothing_representative_options,
    smoothing_summary_payload,
)


@pytest.fixture
def prepared_and_result():  # type: ignore[no-untyped-def]
    parent = make_prepared()
    result = apply_post_baseline_smoothing(
        parent,
        PostBaselineSmoothingConfig(
            enabled=True,
            method="gaussian",
            gaussian_sigma_points=1.25,
            gaussian_truncate=3.0,
        ),
    )
    return parent, result


def test_representative_choices_and_resolution_are_stable(prepared_and_result) -> None:  # type: ignore[no-untyped-def]
    parent, result = prepared_and_result
    options = smoothing_representative_options(parent)

    assert [item.key for item in options[:5]] == [
        "first",
        "middle",
        "last",
        "mean",
        "median",
    ]
    assert [item.key for item in options[5:]] == [
        "spectrum:0",
        "spectrum:1",
        "spectrum:2",
    ]
    assert resolve_smoothing_representative(parent, "first").spectrum_index == 0
    assert resolve_smoothing_representative(parent, "middle").spectrum_index == 1
    assert resolve_smoothing_representative(parent, "last").spectrum_index == 2
    assert resolve_smoothing_representative(parent, 1).key == "spectrum:1"

    actual = smoothing_preview_payload(parent, result, 1)
    mean = smoothing_preview_payload(parent, result, "mean")
    median = smoothing_preview_payload(parent, result, "median")

    np.testing.assert_array_equal(actual.unsmoothed, parent.spectra[1])
    np.testing.assert_array_equal(actual.smoothed, result.smoothed_spectra[1])
    np.testing.assert_array_equal(actual.removed_component, result.removed_component[1])
    np.testing.assert_allclose(mean.unsmoothed, np.mean(parent.spectra, axis=0))
    np.testing.assert_allclose(mean.smoothed, np.mean(result.smoothed_spectra, axis=0))
    np.testing.assert_allclose(
        mean.removed_component,
        np.mean(result.removed_component, axis=0),
    )
    np.testing.assert_allclose(median.unsmoothed, np.median(parent.spectra, axis=0))
    np.testing.assert_allclose(
        median.smoothed,
        np.median(result.smoothed_spectra, axis=0),
    )
    np.testing.assert_allclose(
        median.removed_component,
        np.median(result.removed_component, axis=0),
    )
    assert not actual.unsmoothed.flags.writeable
    assert not actual.smoothed.flags.writeable
    assert not actual.removed_component.flags.writeable


def test_figures_bind_only_existing_preview_curves(prepared_and_result) -> None:  # type: ignore[no-untyped-def]
    parent, result = prepared_and_result
    payload = smoothing_preview_payload(parent, result, "spectrum:1")

    overlay = smoothing_overlay_figure(parent, result, "spectrum:1")
    removed = smoothing_removed_component_figure(parent, result, "spectrum:1")

    assert isinstance(overlay, Figure)
    assert isinstance(removed, Figure)
    assert len(overlay.axes) == 1
    assert len(removed.axes) == 1
    assert [line.get_label() for line in overlay.axes[0].lines] == [
        "Unsmoothed corrected absorbance",
        "Smoothed corrected absorbance",
    ]
    np.testing.assert_array_equal(overlay.axes[0].lines[0].get_xdata(), payload.wavenumber)
    np.testing.assert_array_equal(overlay.axes[0].lines[0].get_ydata(), payload.unsmoothed)
    np.testing.assert_array_equal(overlay.axes[0].lines[1].get_ydata(), payload.smoothed)
    assert [line.get_label() for line in removed.axes[0].lines] == [
        "Removed component",
        "Zero reference",
    ]
    np.testing.assert_array_equal(
        removed.axes[0].lines[0].get_ydata(),
        payload.removed_component,
    )
    assert overlay.axes[0].xaxis_inverted()
    assert removed.axes[0].xaxis_inverted()


def test_qc_and_summary_payloads_are_detached_stored_values(prepared_and_result) -> None:  # type: ignore[no-untyped-def]
    parent, result = prepared_and_result

    table = smoothing_qc_table(parent, result)
    summary = smoothing_summary_payload(parent, result)

    assert table["spectrum_index"].tolist() == [0, 1, 2]
    assert table["perturbation_label"].tolist() == list(parent.perturbation_labels)
    np.testing.assert_array_equal(table["perturbation"].to_numpy(), parent.perturbation)
    for name, values in result.per_spectrum_metrics.items():
        np.testing.assert_array_equal(table[name].to_numpy(), values)
    assert summary["parent_prepared_data_sha256"] == parent.prepared_data_sha256
    assert summary["smoothing_fingerprint"] == result.smoothing_fingerprint
    assert summary["method"] == "gaussian"
    assert summary["effective_parameters"] == {
        "sigma_points": 1.25,
        "truncate": 3.0,
        "mode": "reflect",
    }
    assert summary["summary_metrics"] == dict(result.summary_metrics)
    assert summary["warnings"] == list(result.warnings)

    table.loc[0, "rms_removed"] = -999.0
    assert result.per_spectrum_metrics["rms_removed"][0] >= 0.0
    summary_metrics = summary["summary_metrics"]
    assert isinstance(summary_metrics, dict)
    summary_metrics["mean_relative_rms_removed"] = -999.0
    assert result.summary_metrics["mean_relative_rms_removed"] >= 0.0


def test_helpers_do_not_mutate_inputs_or_call_smoothing_core(
    prepared_and_result,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    parent, result = prepared_and_result
    parent_axis = parent.wavenumber.copy()
    parent_spectra = parent.spectra.copy()
    smoothed = result.smoothed_spectra.copy()
    removed = result.removed_component.copy()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("display helpers must not call the smoothing core")

    monkeypatch.setattr(smoothing_core_module, "apply_post_baseline_smoothing", forbidden)

    smoothing_representative_options(parent)
    smoothing_preview_payload(parent, result, "median")
    smoothing_overlay_figure(parent, result, "first")
    smoothing_removed_component_figure(parent, result, "last")
    smoothing_qc_table(parent, result)
    smoothing_summary_payload(parent, result)

    np.testing.assert_array_equal(parent.wavenumber, parent_axis)
    np.testing.assert_array_equal(parent.spectra, parent_spectra)
    np.testing.assert_array_equal(result.smoothed_spectra, smoothed)
    np.testing.assert_array_equal(result.removed_component, removed)
    assert not parent.spectra.flags.writeable
    assert not result.smoothed_spectra.flags.writeable
    assert not result.removed_component.flags.writeable
