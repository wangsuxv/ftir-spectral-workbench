from __future__ import annotations

import numpy as np
import pytest

from ftir_workbench.post_baseline_smoothing import (
    PostBaselineSmoothingConfig,
    apply_post_baseline_smoothing,
)

from ._helpers import make_prepared


def test_uniform_axis_diagnostics_and_savgol_physical_span() -> None:
    axis = 1800.0 - 2.0 * np.arange(41, dtype=np.float64)
    prepared = make_prepared(wavenumber=axis)
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="savgol",
        savgol_window_length=7,
    )

    result = apply_post_baseline_smoothing(prepared, config)

    assert result.median_wavenumber_spacing == 2.0
    assert result.spacing_relative_max_deviation == 0.0
    assert result.approximate_physical_width == {"span_cm1": 12.0}


def test_tiny_floating_axis_jitter_within_tolerance_passes() -> None:
    axis = 1800.0 - 2.0 * np.arange(41, dtype=np.float64)
    axis[20] += 1e-4
    prepared = make_prepared(wavenumber=axis)

    result = apply_post_baseline_smoothing(
        prepared,
        PostBaselineSmoothingConfig(enabled=True, method="gaussian"),
    )

    assert result.spacing_relative_max_deviation == pytest.approx(5e-5)
    assert not any("expert override" in warning for warning in result.warnings)


def test_nonuniform_axis_errors_by_default_without_resampling() -> None:
    axis = 1800.0 - 2.0 * np.arange(41, dtype=np.float64)
    axis[20] += 0.25
    prepared = make_prepared(wavenumber=axis)
    original_axis = prepared.wavenumber.copy()
    original_spectra = prepared.spectra.copy()

    with pytest.raises(ValueError, match="does not automatically resample"):
        apply_post_baseline_smoothing(
            prepared,
            PostBaselineSmoothingConfig(enabled=True, method="gaussian"),
        )

    np.testing.assert_array_equal(prepared.wavenumber, original_axis)
    np.testing.assert_array_equal(prepared.spectra, original_spectra)


def test_nonuniform_axis_explicit_override_warns_and_preserves_grid() -> None:
    axis = 1800.0 - 2.0 * np.arange(41, dtype=np.float64)
    axis[20] += 0.25
    prepared = make_prepared(wavenumber=axis)
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="gaussian",
        nonuniform_axis_policy="allow_index_space_with_warning",
    )

    result = apply_post_baseline_smoothing(prepared, config)

    assert result.smoothed_spectra.shape == prepared.spectra.shape
    assert result.parent_prepared is prepared
    np.testing.assert_array_equal(result.parent_prepared.wavenumber, axis)
    assert any("expert override" in warning for warning in result.warnings)
    assert any("not resampled" in warning for warning in result.warnings)


def test_disabled_identity_diagnoses_but_does_not_block_a_nonuniform_axis() -> None:
    axis = np.array([1800.0, 1790.0, 1775.0, 1760.0, 1740.0])
    spectra = np.arange(15.0).reshape(3, 5)
    prepared = make_prepared(wavenumber=axis, spectra=spectra)

    result = apply_post_baseline_smoothing(prepared, PostBaselineSmoothingConfig())

    np.testing.assert_array_equal(result.smoothed_spectra, spectra)
    assert result.spacing_relative_max_deviation > 0.0
    assert result.warnings == ()


@pytest.mark.parametrize(
    ("config", "expected"),
    (
        (
            PostBaselineSmoothingConfig(
                enabled=True,
                method="gaussian",
                gaussian_sigma_points=1.5,
            ),
            {"sigma_cm1": 3.0, "fwhm_cm1": 7.06446},
        ),
        (
            PostBaselineSmoothingConfig(
                enabled=True,
                method="moving_average",
                moving_average_window_length=5,
            ),
            {"span_cm1": 8.0},
        ),
        (
            PostBaselineSmoothingConfig(
                enabled=True,
                method="median",
                median_window_length=5,
            ),
            {"span_cm1": 8.0},
        ),
    ),
)
def test_method_physical_widths(
    config: PostBaselineSmoothingConfig,
    expected: dict[str, float],
) -> None:
    axis = 1800.0 - 2.0 * np.arange(41, dtype=np.float64)
    result = apply_post_baseline_smoothing(make_prepared(wavenumber=axis), config)

    assert dict(result.approximate_physical_width) == pytest.approx(expected)


def test_only_the_active_window_is_limited_by_axis_length() -> None:
    axis = np.linspace(1800.0, 1600.0, 5)
    spectra = np.arange(15.0).reshape(3, 5)
    prepared = make_prepared(wavenumber=axis, spectra=spectra)

    gaussian = apply_post_baseline_smoothing(
        prepared,
        PostBaselineSmoothingConfig(enabled=True, method="gaussian"),
    )
    assert gaussian.smoothed_spectra.shape == spectra.shape
    with pytest.raises(ValueError, match="must not exceed n_wavenumbers"):
        apply_post_baseline_smoothing(
            prepared,
            PostBaselineSmoothingConfig(
                enabled=True,
                method="savgol",
                savgol_window_length=7,
            ),
        )


@pytest.mark.parametrize(
    "config",
    (
        PostBaselineSmoothingConfig(enabled=True, method="savgol"),
        PostBaselineSmoothingConfig(enabled=True, method="gaussian"),
        PostBaselineSmoothingConfig(enabled=True, method="moving_average"),
        PostBaselineSmoothingConfig(enabled=True, method="median"),
    ),
)
def test_ascending_and_descending_axis_reversal_are_equivalent(
    config: PostBaselineSmoothingConfig,
) -> None:
    descending = make_prepared()
    ascending = make_prepared(
        wavenumber=descending.wavenumber[::-1],
        spectra=descending.spectra[:, ::-1],
    )

    descending_result = apply_post_baseline_smoothing(descending, config)
    ascending_result = apply_post_baseline_smoothing(ascending, config)

    np.testing.assert_allclose(
        ascending_result.smoothed_spectra,
        descending_result.smoothed_spectra[:, ::-1],
        rtol=1e-14,
        atol=1e-14,
    )
    for name in descending_result.per_spectrum_metrics:
        np.testing.assert_allclose(
            ascending_result.per_spectrum_metrics[name],
            descending_result.per_spectrum_metrics[name],
            rtol=1e-12,
            atol=1e-12,
        )
