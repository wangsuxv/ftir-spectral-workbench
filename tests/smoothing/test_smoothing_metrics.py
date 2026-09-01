from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import trapezoid

from ftir_workbench.post_baseline_smoothing import (
    PostBaselineSmoothingConfig,
    apply_post_baseline_smoothing,
)

from ._helpers import make_prepared


def test_qc_metrics_match_direct_definitions_and_removed_identity() -> None:
    prepared = make_prepared()
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="moving_average",
        moving_average_window_length=5,
    )

    result = apply_post_baseline_smoothing(prepared, config)
    before = prepared.spectra
    after = result.smoothed_spectra
    removed = before - after
    metrics = result.per_spectrum_metrics

    np.testing.assert_array_equal(result.removed_component, removed)
    np.testing.assert_allclose(
        metrics["rms_removed"],
        np.sqrt(np.mean(removed**2, axis=1)),
    )
    np.testing.assert_array_equal(
        metrics["max_abs_removed"],
        np.max(np.abs(removed), axis=1),
    )
    roughness_before = np.mean(np.diff(before, n=2, axis=1) ** 2, axis=1)
    roughness_after = np.mean(np.diff(after, n=2, axis=1) ** 2, axis=1)
    np.testing.assert_allclose(metrics["roughness_before"], roughness_before)
    np.testing.assert_allclose(metrics["roughness_after"], roughness_after)
    np.testing.assert_allclose(
        metrics["roughness_ratio"],
        roughness_after / roughness_before,
    )
    expected_correlations = np.array(
        [
            np.corrcoef(np.diff(left), np.diff(right))[0, 1]
            for left, right in zip(before, after, strict=True)
        ]
    )
    np.testing.assert_allclose(
        metrics["first_derivative_correlation"],
        expected_correlations,
    )


def test_area_metrics_are_orientation_independent_and_use_ascending_integration() -> None:
    descending = make_prepared()
    ascending = make_prepared(
        wavenumber=descending.wavenumber[::-1],
        spectra=descending.spectra[:, ::-1],
    )
    config = PostBaselineSmoothingConfig(enabled=True, method="gaussian")

    descending_result = apply_post_baseline_smoothing(descending, config)
    ascending_result = apply_post_baseline_smoothing(ascending, config)
    expected_signed = trapezoid(
        ascending.spectra,
        x=ascending.wavenumber,
        axis=1,
    )
    expected_absolute = trapezoid(
        np.abs(ascending.spectra),
        x=ascending.wavenumber,
        axis=1,
    )

    np.testing.assert_allclose(
        descending_result.per_spectrum_metrics["signed_area_before"],
        expected_signed,
    )
    np.testing.assert_allclose(
        descending_result.per_spectrum_metrics["absolute_area_before"],
        expected_absolute,
    )
    for name in (
        "signed_area_before",
        "signed_area_after",
        "relative_signed_area_change",
        "absolute_area_before",
        "absolute_area_after",
        "relative_absolute_area_change",
    ):
        np.testing.assert_allclose(
            descending_result.per_spectrum_metrics[name],
            ascending_result.per_spectrum_metrics[name],
            rtol=1e-12,
            atol=1e-12,
        )


def test_edge_metrics_use_the_effective_window_radius() -> None:
    prepared = make_prepared()
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="moving_average",
        moving_average_window_length=5,
    )

    result = apply_post_baseline_smoothing(prepared, config)
    removed = result.removed_component
    edge = np.concatenate((removed[:, :2], removed[:, -2:]), axis=1)
    center = removed[:, 2:-2]
    expected_edge = np.sqrt(np.mean(edge**2, axis=1))
    expected_center = np.sqrt(np.mean(center**2, axis=1))

    np.testing.assert_allclose(result.per_spectrum_metrics["edge_rms_removed"], expected_edge)
    np.testing.assert_allclose(
        result.per_spectrum_metrics["center_rms_removed"],
        expected_center,
    )
    np.testing.assert_allclose(
        result.per_spectrum_metrics["edge_effect_ratio"],
        expected_edge / expected_center,
    )


def test_zero_and_constant_spectra_have_explicit_finite_neutral_ratios() -> None:
    axis = np.linspace(1800.0, 1600.0, 21)
    spectra = np.vstack(
        (
            np.zeros(axis.size),
            np.full(axis.size, 2.0),
            np.full(axis.size, -3.0),
        )
    )
    prepared = make_prepared(wavenumber=axis, spectra=spectra)
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="moving_average",
        moving_average_window_length=5,
    )

    result = apply_post_baseline_smoothing(prepared, config)
    metrics = result.per_spectrum_metrics

    assert all(np.all(np.isfinite(values)) for values in metrics.values())
    assert all(np.isfinite(value) for value in result.summary_metrics.values())
    np.testing.assert_array_equal(metrics["relative_rms_removed"], 0.0)
    np.testing.assert_array_equal(metrics["roughness_ratio"], 1.0)
    np.testing.assert_array_equal(metrics["first_derivative_correlation"], 1.0)
    np.testing.assert_array_equal(metrics["relative_signed_area_change"], 0.0)
    np.testing.assert_array_equal(metrics["relative_absolute_area_change"], 0.0)
    np.testing.assert_array_equal(metrics["edge_effect_ratio"], 1.0)


def test_diagnostic_thresholds_only_add_warnings_and_do_not_block_result() -> None:
    axis = np.linspace(1800.0, 1600.0, 101)
    spectra = np.full((3, axis.size), 0.01)
    spectra[0, 50] = 4.0
    spectra[1, 1] = 6.0
    spectra[2, 30:33] = (-3.0, 5.0, -2.0)
    prepared = make_prepared(wavenumber=axis, spectra=spectra)
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="moving_average",
        moving_average_window_length=21,
    )

    result = apply_post_baseline_smoothing(prepared, config)

    assert result.config == config
    assert result.smoothed_spectra.shape == prepared.spectra.shape
    assert any("Relative RMS removed" in warning for warning in result.warnings)
    assert any("First-derivative correlation" in warning for warning in result.warnings)
    assert result.summary_metrics["max_relative_rms_removed"] > 0.10
    assert result.summary_metrics["min_first_derivative_correlation"] < 0.95


def test_metric_arrays_and_mappings_are_deeply_immutable() -> None:
    result = apply_post_baseline_smoothing(
        make_prepared(),
        PostBaselineSmoothingConfig(enabled=True, method="gaussian"),
    )

    with pytest.raises(TypeError):
        result.per_spectrum_metrics["new"] = np.zeros(3)  # type: ignore[index]
    with pytest.raises(TypeError):
        result.summary_metrics["new"] = 0.0  # type: ignore[index]
    metric = result.per_spectrum_metrics["rms_removed"]
    assert not metric.flags.writeable
    with pytest.raises(ValueError):
        metric.flags.writeable = True
