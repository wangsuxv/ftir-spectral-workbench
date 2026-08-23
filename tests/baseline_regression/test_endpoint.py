from __future__ import annotations

import numpy as np
import pytest

from ftir_baseline.baseline.endpoint import (
    endpoint_window_baseline,
    strict_endpoint_baseline,
)


def test_endpoint_window_removes_a_pure_linear_baseline() -> None:
    x = np.linspace(900.0, 1800.0, 901)
    spectra = np.vstack((0.2 + 2e-4 * x, -0.4 + 8e-4 * x))

    result = endpoint_window_baseline(x, spectra, window_width_cm1=8.0)

    np.testing.assert_allclose(result.corrected, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.total_baseline, spectra, atol=1e-14)
    assert result.params["statistic"] == "median"
    assert result.params["endpoint_window_width_cm1"] == 8.0


def test_window_median_is_robust_to_noisy_boundary_points() -> None:
    x = np.arange(11.0)
    expected = 1.0 + 0.25 * x
    noisy = expected.copy()
    noisy[0] += 100.0
    noisy[-1] -= 100.0

    robust = endpoint_window_baseline(x, noisy, window_width_cm1=4.0)
    strict = strict_endpoint_baseline(x, noisy)

    # A median window is not an oracle: on a sloping signal an extreme sample
    # can move the median to a neighbouring clean sample.  It should still make
    # the boundary noise orders of magnitude less influential than strict mode.
    robust_error = np.max(np.abs(robust.total_baseline - expected))
    strict_error = np.max(np.abs(strict.total_baseline - expected))
    assert robust_error < 0.01 * strict_error
    assert strict_error > 50


def test_strict_mode_zeros_the_two_concrete_boundary_samples() -> None:
    x = np.array([1800.0, 1600.0, 1200.0, 900.0])
    y = np.array([4.0, 10.0, -3.0, 2.0])

    result = strict_endpoint_baseline(x, y)

    assert result.corrected[0] == pytest.approx(0.0, abs=1e-15)
    assert result.corrected[-1] == pytest.approx(0.0, abs=1e-15)
    assert result.warnings


@pytest.mark.parametrize("function", [endpoint_window_baseline, strict_endpoint_baseline])
def test_ascending_and_descending_axes_are_equivalent(function: object) -> None:
    x = np.linspace(900.0, 1800.0, 181)
    y = np.vstack((np.sin(x / 200), 0.5 * np.cos(x / 170)))

    ascending = function(x, y)  # type: ignore[operator]
    descending = function(x[::-1], y[:, ::-1])  # type: ignore[operator]

    np.testing.assert_allclose(
        ascending.total_baseline, descending.total_baseline[:, ::-1], atol=1e-14
    )
    np.testing.assert_allclose(ascending.corrected, descending.corrected[:, ::-1], atol=1e-14)


@pytest.mark.parametrize("width", [0.0, -1.0, np.nan, np.inf])
def test_invalid_or_effectively_empty_endpoint_window_fails_clearly(width: float) -> None:
    with pytest.raises(ValueError, match="window width"):
        endpoint_window_baseline(np.arange(5.0), np.arange(5.0), width)


def test_overlapping_endpoint_windows_cannot_define_a_line() -> None:
    with pytest.raises(ValueError, match="overlap"):
        endpoint_window_baseline(np.arange(5.0), np.arange(5.0), 20.0)
