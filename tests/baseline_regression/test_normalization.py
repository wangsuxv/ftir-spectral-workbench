from __future__ import annotations

import numpy as np
import pytest

from ftir_baseline.normalization import apply_normalization, normalize_spectra


def _integral_rows(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.sum(
        0.5 * (values[:, :-1] + values[:, 1:]) * np.diff(x)[np.newaxis, :],
        axis=1,
    )


def test_none_is_exact_and_does_not_modify_input() -> None:
    x = np.linspace(900.0, 1800.0, 9)
    spectra = np.arange(18.0).reshape(2, 9)
    before = spectra.copy()

    result = normalize_spectra(x, spectra, "none")

    np.testing.assert_array_equal(spectra, before)
    np.testing.assert_array_equal(result.analysis_data, before)
    np.testing.assert_array_equal(result.view_data, before)
    assert result.optional_normalized is None
    np.testing.assert_array_equal(result.factors, np.ones(2))


def test_internal_peak_height_factors_and_optional_branch() -> None:
    x = np.array([900.0, 1000.0, 1100.0, 1200.0])
    spectra = np.array([[0.0, 2.0, 4.0, 0.0], [0.0, 1.0, 2.0, 0.0]])

    result = normalize_spectra(
        x,
        spectra,
        "internal_peak_height",
        reference_interval=(950.0, 1150.0),
        target=1.0,
    )

    np.testing.assert_allclose(result.factors, [0.25, 0.5])
    assert result.optional_normalized is not None
    np.testing.assert_allclose(np.max(result.optional_normalized[:, 1:3], axis=1), 1.0)
    # The quantitative main branch remains baseline-corrected absorbance.
    np.testing.assert_array_equal(result.analysis_data, spectra)


def test_internal_peak_area_is_axis_direction_independent() -> None:
    x = np.linspace(1000.0, 1100.0, 11)
    spectra = np.stack([np.ones(11), np.full(11, 2.0)])

    ascending = normalize_spectra(x, spectra, "internal_peak_area", interval=(1000.0, 1100.0))
    descending = normalize_spectra(
        x[::-1], spectra[:, ::-1], "internal_peak_area", interval=(1100.0, 1000.0)
    )

    np.testing.assert_allclose(ascending.factors, [0.01, 0.005])
    np.testing.assert_allclose(ascending.factors, descending.factors)
    np.testing.assert_allclose(
        ascending.optional_normalized, descending.optional_normalized[:, ::-1]
    )


def test_zero_internal_area_has_clear_error() -> None:
    x = np.linspace(900.0, 1000.0, 7)
    with pytest.raises(ValueError, match="internal reference peak area is zero"):
        normalize_spectra(
            x,
            np.zeros((2, x.size)),
            "internal_peak_area",
            interval=(900.0, 1000.0),
        )


def test_vector_and_area_normalization_have_unit_target() -> None:
    x = np.linspace(0.0, 4.0, 5)
    spectra = np.array([[3.0, 4.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    vector = normalize_spectra(x, spectra, "vector")
    assert vector.optional_normalized is not None
    np.testing.assert_allclose(np.linalg.norm(vector.optional_normalized, axis=1), 1.0)

    area = normalize_spectra(x, spectra, "area", interval=(0.0, 4.0))
    assert area.optional_normalized is not None
    np.testing.assert_allclose(_integral_rows(x, np.abs(area.optional_normalized)), 1.0)


def test_minmax_is_strictly_display_only() -> None:
    x = np.arange(5.0)
    spectra = np.array([[10.0, 11.0, 14.0, 12.0, 13.0]])

    result = normalize_spectra(x, spectra, "minmax_display")

    np.testing.assert_array_equal(result.analysis_data, spectra)
    assert result.optional_normalized is None
    assert np.min(result.view_data) == pytest.approx(0.0)
    assert np.max(result.view_data) == pytest.approx(1.0)
    assert any("display-only" in warning for warning in result.warnings)


def test_config_wrapper_accepts_recipe_field_names() -> None:
    x = np.linspace(1000.0, 1100.0, 5)
    spectra = np.ones((2, 5))
    result = apply_normalization(
        x,
        spectra,
        {
            "method": "internal_peak_area",
            "internal_reference_range": (1000.0, 1100.0),
            "absolute": True,
            "target_value": 2.0,
        },
    )
    assert result.optional_normalized is not None
    np.testing.assert_allclose(_integral_rows(x, result.optional_normalized), 2.0)


def test_non_finite_input_and_empty_interval_are_rejected() -> None:
    x = np.arange(4.0)
    with pytest.raises(ValueError, match="NaN or Inf"):
        normalize_spectra(x, [1.0, np.nan, 2.0, 3.0], "none")
    with pytest.raises(ValueError, match="fewer than two"):
        normalize_spectra(x, np.ones(4), "area", interval=(2.0, 2.1))
