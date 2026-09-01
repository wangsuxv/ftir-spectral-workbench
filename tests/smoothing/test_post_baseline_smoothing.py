from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import savgol_filter

from ftir_workbench import (
    PostBaselineSmoothingConfig,
    PostBaselineSmoothingResult,
    apply_post_baseline_smoothing,
    post_baseline_smoothing_fingerprint,
)

from ._helpers import make_prepared


def test_disabled_is_a_detached_immutable_identity_result() -> None:
    prepared = make_prepared()
    input_snapshot = prepared.spectra.copy()

    result = apply_post_baseline_smoothing(prepared, PostBaselineSmoothingConfig())

    assert isinstance(result, PostBaselineSmoothingResult)
    np.testing.assert_array_equal(result.smoothed_spectra, prepared.spectra)
    np.testing.assert_array_equal(result.removed_component, 0.0)
    np.testing.assert_array_equal(prepared.spectra, input_snapshot)
    assert result.smoothed_spectra.dtype == np.float64
    assert not np.shares_memory(result.smoothed_spectra, prepared.spectra)
    assert not result.smoothed_spectra.flags.writeable
    assert not result.removed_component.flags.writeable
    with pytest.raises(ValueError):
        result.smoothed_spectra.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        result.smoothing_fingerprint = "0" * 64  # type: ignore[misc]


@pytest.mark.parametrize(
    ("smoothed_factory", "removed_factory", "match"),
    (
        (
            lambda spectra: spectra + 1.0,
            lambda spectra: np.full(spectra.shape, -1.0),
            "element-for-element identical",
        ),
        (
            lambda spectra: spectra.copy(),
            lambda spectra: np.ones(spectra.shape),
            "all-zero removed_component",
        ),
    ),
)
def test_manually_constructed_disabled_result_rejects_non_identity_payloads(
    smoothed_factory: Callable[[np.ndarray], np.ndarray],
    removed_factory: Callable[[np.ndarray], np.ndarray],
    match: str,
) -> None:
    prepared = make_prepared()
    valid = apply_post_baseline_smoothing(prepared, PostBaselineSmoothingConfig())
    smoothed = smoothed_factory(prepared.spectra)
    removed = removed_factory(prepared.spectra)

    with pytest.raises(ValueError, match=match):
        PostBaselineSmoothingResult(
            parent_prepared=prepared,
            config=valid.config,
            smoothed_spectra=smoothed,
            removed_component=removed,
            per_spectrum_metrics=valid.per_spectrum_metrics,
            summary_metrics=valid.summary_metrics,
            median_wavenumber_spacing=valid.median_wavenumber_spacing,
            spacing_relative_max_deviation=valid.spacing_relative_max_deviation,
            approximate_physical_width=valid.approximate_physical_width,
            smoothing_fingerprint=valid.smoothing_fingerprint,
            warnings=valid.warnings,
        )


def test_savgol_matches_direct_scipy_axis_1_call() -> None:
    prepared = make_prepared()
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="savgol",
        savgol_window_length=7,
        savgol_polyorder=2,
        savgol_mode="mirror",
    )

    result = apply_post_baseline_smoothing(prepared, config)
    expected = savgol_filter(
        prepared.spectra,
        window_length=7,
        polyorder=2,
        deriv=0,
        axis=1,
        mode="mirror",
    )

    np.testing.assert_array_equal(result.smoothed_spectra, expected)
    np.testing.assert_array_equal(result.removed_component, prepared.spectra - expected)


def test_gaussian_matches_direct_scipy_axis_1_call_and_impulse_is_symmetric() -> None:
    axis = np.linspace(1800.0, 1600.0, 31)
    spectra = np.zeros((3, axis.size), dtype=np.float64)
    spectra[:, axis.size // 2] = (1.0, 3.0, 9.0)
    prepared = make_prepared(wavenumber=axis, spectra=spectra)
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="gaussian",
        gaussian_sigma_points=1.5,
        gaussian_truncate=3.0,
        convolution_mode="nearest",
    )

    result = apply_post_baseline_smoothing(prepared, config)
    expected = gaussian_filter1d(
        spectra,
        sigma=1.5,
        axis=1,
        mode="nearest",
        truncate=3.0,
    )

    np.testing.assert_array_equal(result.smoothed_spectra, expected)
    np.testing.assert_allclose(result.smoothed_spectra[0], result.smoothed_spectra[0, ::-1])
    assert result.smoothed_spectra.shape == spectra.shape


def test_moving_average_matches_direct_scipy_and_preserves_constants() -> None:
    axis = np.linspace(1800.0, 1600.0, 21)
    spectra = np.vstack(
        (
            np.full(axis.size, 1.0),
            np.full(axis.size, 10.0),
            np.full(axis.size, -2.0),
        )
    )
    prepared = make_prepared(wavenumber=axis, spectra=spectra)
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="moving_average",
        moving_average_window_length=5,
        convolution_mode="reflect",
    )

    result = apply_post_baseline_smoothing(prepared, config)
    expected = uniform_filter1d(spectra, size=5, axis=1, mode="reflect")

    np.testing.assert_array_equal(result.smoothed_spectra, expected)
    np.testing.assert_array_equal(result.smoothed_spectra, spectra)


def test_median_matches_direct_scipy_removes_spikes_without_cross_row_mixing() -> None:
    axis = np.linspace(1800.0, 1600.0, 21)
    spectra = np.vstack(
        (
            np.zeros(axis.size),
            np.full(axis.size, 10.0),
            np.full(axis.size, -5.0),
        )
    )
    spectra[0, 10] = 100.0
    spectra[1, 7] = -100.0
    prepared = make_prepared(wavenumber=axis, spectra=spectra)
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="median",
        median_window_length=3,
        convolution_mode="reflect",
    )

    result = apply_post_baseline_smoothing(prepared, config)
    expected = median_filter(spectra, size=(1, 3), mode="reflect")

    np.testing.assert_array_equal(result.smoothed_spectra, expected)
    assert result.smoothed_spectra[0, 10] == 0.0
    assert result.smoothed_spectra[1, 7] == 10.0
    assert any("nonlinear" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    "config",
    (
        PostBaselineSmoothingConfig(enabled=True, method="savgol"),
        PostBaselineSmoothingConfig(enabled=True, method="gaussian"),
        PostBaselineSmoothingConfig(enabled=True, method="moving_average"),
        PostBaselineSmoothingConfig(enabled=True, method="median"),
    ),
)
def test_each_row_is_independent_and_runs_are_deterministic(
    config: PostBaselineSmoothingConfig,
) -> None:
    prepared = make_prepared()
    changed = prepared.spectra.copy()
    changed[0] += np.linspace(-100.0, 100.0, prepared.n_points)
    changed_prepared = make_prepared(wavenumber=prepared.wavenumber, spectra=changed)

    first = apply_post_baseline_smoothing(prepared, config)
    repeated = apply_post_baseline_smoothing(prepared, config)
    changed_result = apply_post_baseline_smoothing(changed_prepared, config)

    np.testing.assert_array_equal(first.smoothed_spectra, repeated.smoothed_spectra)
    np.testing.assert_array_equal(
        first.smoothed_spectra[1:],
        changed_result.smoothed_spectra[1:],
    )
    assert first.smoothing_fingerprint == repeated.smoothing_fingerprint


@pytest.mark.parametrize(
    ("first_config", "inactive_changed"),
    (
        (
            PostBaselineSmoothingConfig(enabled=True, method="savgol"),
            PostBaselineSmoothingConfig(
                enabled=True,
                method="savgol",
                gaussian_sigma_points=2.5,
                gaussian_truncate=3.0,
                moving_average_window_length=9,
                median_window_length=11,
                convolution_mode="nearest",
            ),
        ),
        (
            PostBaselineSmoothingConfig(enabled=True, method="gaussian"),
            PostBaselineSmoothingConfig(
                enabled=True,
                method="gaussian",
                savgol_window_length=13,
                savgol_polyorder=4,
                savgol_mode="mirror",
                moving_average_window_length=9,
                median_window_length=11,
            ),
        ),
        (
            PostBaselineSmoothingConfig(
                enabled=True,
                method="moving_average",
                moving_average_window_length=5,
            ),
            PostBaselineSmoothingConfig(
                enabled=True,
                method="moving_average",
                savgol_window_length=13,
                savgol_polyorder=4,
                savgol_mode="nearest",
                gaussian_sigma_points=2.5,
                gaussian_truncate=3.0,
                moving_average_window_length=5,
                median_window_length=11,
            ),
        ),
        (
            PostBaselineSmoothingConfig(
                enabled=True,
                method="median",
                median_window_length=5,
            ),
            PostBaselineSmoothingConfig(
                enabled=True,
                method="median",
                savgol_window_length=13,
                savgol_polyorder=4,
                savgol_mode="nearest",
                gaussian_sigma_points=2.5,
                gaussian_truncate=3.0,
                moving_average_window_length=9,
                median_window_length=5,
            ),
        ),
    ),
)
def test_inactive_parameters_do_not_change_scientific_fingerprint_for_any_method(
    first_config: PostBaselineSmoothingConfig,
    inactive_changed: PostBaselineSmoothingConfig,
) -> None:
    prepared = make_prepared()

    first = apply_post_baseline_smoothing(prepared, first_config)
    second = apply_post_baseline_smoothing(prepared, inactive_changed)

    assert first_config.scientific_dict() == inactive_changed.scientific_dict()
    np.testing.assert_array_equal(first.smoothed_spectra, second.smoothed_spectra)
    assert first.smoothing_fingerprint == second.smoothing_fingerprint
    assert first.smoothing_fingerprint == post_baseline_smoothing_fingerprint(
        prepared,
        first_config,
        first.smoothed_spectra,
    )


def test_fingerprint_canonicalizes_c_and_fortran_memory_order() -> None:
    prepared = make_prepared()
    config = PostBaselineSmoothingConfig(enabled=True, method="gaussian")
    smoothed = apply_post_baseline_smoothing(prepared, config).smoothed_spectra
    c_order = np.array(smoothed, dtype=np.float64, order="C", copy=True)
    fortran_order = np.array(smoothed, dtype=np.float64, order="F", copy=True)

    assert c_order.flags.c_contiguous
    assert fortran_order.flags.f_contiguous
    assert post_baseline_smoothing_fingerprint(
        prepared,
        config,
        c_order,
    ) == post_baseline_smoothing_fingerprint(prepared, config, fortran_order)


def test_fingerprint_canonicalizes_big_endian_array_and_nested_list() -> None:
    prepared = make_prepared()
    config = PostBaselineSmoothingConfig(enabled=True, method="moving_average")
    smoothed = apply_post_baseline_smoothing(prepared, config).smoothed_spectra
    big_endian = np.asarray(smoothed, dtype=np.dtype(">f8"))
    nested_list = smoothed.tolist()

    assert big_endian.dtype.byteorder == ">"
    expected = post_baseline_smoothing_fingerprint(prepared, config, smoothed)
    assert post_baseline_smoothing_fingerprint(prepared, config, big_endian) == expected
    assert post_baseline_smoothing_fingerprint(prepared, config, nested_list) == expected


def test_scientific_normalization_and_chained_smoothing_are_rejected() -> None:
    scientific = make_prepared(normalization_state="scientific_explicit")
    top_level_chained = make_prepared(
        baseline_recipe={"post_baseline_smoothing": {"config": {}}}
    )
    nested_chained = make_prepared(
        baseline_recipe={
            "prepared_data_contract": {"branch_kind": "post_baseline_smoothing"}
        }
    )
    config = PostBaselineSmoothingConfig(enabled=True)

    with pytest.raises(ValueError, match="does not combine scientific normalization"):
        apply_post_baseline_smoothing(scientific, config)
    with pytest.raises(ValueError, match="Chained smoothing is disabled"):
        apply_post_baseline_smoothing(top_level_chained, config)
    with pytest.raises(ValueError, match="Chained smoothing is disabled"):
        apply_post_baseline_smoothing(nested_chained, config)
