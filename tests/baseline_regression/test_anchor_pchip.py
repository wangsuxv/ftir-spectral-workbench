from __future__ import annotations

import numpy as np
import pytest

from ftir_baseline.baseline.anchors import (
    multipoint_linear_baseline,
    pchip_baseline,
    polynomial_baseline,
)


def test_pchip_passes_through_fixed_window_representative_values() -> None:
    x = np.arange(11.0)
    y = np.array([1, 2, 3, 4, 5, 3, 2, 2.5, 4, 5, 6], dtype=float)
    anchors = [(0, 0), (5, 5), (10, 10)]

    result = pchip_baseline(x, y, anchors)

    np.testing.assert_allclose(result.total_baseline[[0, 5, 10]], y[[0, 5, 10]])
    assert result.params["anchor_centers"] == (0.0, 5.0, 10.0)
    assert result.params["anchor_values"][0] == (1.0, 3.0, 6.0)


def test_two_anchor_pchip_degenerates_exactly_to_linear_interpolation() -> None:
    x = np.linspace(0, 10, 51)
    y = 4.0 - 0.3 * x
    anchors = [(0, 0), (10, 10)]

    pchip = pchip_baseline(x, y, anchors)
    linear = multipoint_linear_baseline(x, y, anchors)

    np.testing.assert_array_equal(pchip.total_baseline, linear.total_baseline)
    np.testing.assert_allclose(pchip.corrected, 0.0, atol=1e-15)


def test_anchor_interpolation_refuses_silent_extrapolation() -> None:
    x = np.arange(11.0)
    anchors = [(1, 1), (9, 9)]

    with pytest.raises(ValueError, match="extrapolation is forbidden"):
        pchip_baseline(x, x**2, anchors)


@pytest.mark.parametrize(
    "anchors",
    [
        [(0, 3), (2, 5), (10, 10)],
        [(0, 0), (10, 10), (5, 5)],
    ],
)
def test_overlapping_or_out_of_order_anchors_fail(anchors: list[tuple[int, int]]) -> None:
    with pytest.raises(ValueError, match=r"overlap|order"):
        pchip_baseline(np.arange(11.0), np.arange(11.0), anchors)


def test_empty_anchor_window_fails_with_its_index() -> None:
    with pytest.raises(ValueError, match=r"anchor window 1 .* contains no data"):
        pchip_baseline(
            np.arange(11.0),
            np.arange(11.0),
            [(0, 0), (4.25, 4.5), (10, 10)],
        )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_polynomial_orders_one_to_three(order: int) -> None:
    x = np.linspace(-1, 1, 41)
    y = 2 + x - 0.5 * x**2 + 0.25 * x**3
    anchors = [(-1, -1), (-0.5, -0.5), (0, 0), (0.5, 0.5), (1, 1)]

    result = polynomial_baseline(x, y, anchors, order=order)

    assert result.params["polynomial_order"] == order
    if order == 3:
        np.testing.assert_allclose(result.corrected, 0.0, atol=1e-14)


def test_anchor_baseline_is_axis_direction_equivalent() -> None:
    x = np.linspace(0, 10, 101)
    spectra = np.vstack((1 + x / 10, 0.2 + (x - 5) ** 2 / 30))
    anchors = [(0, 0), (5, 5), (10, 10)]

    ascending = pchip_baseline(x, spectra, anchors)
    descending = pchip_baseline(x[::-1], spectra[:, ::-1], anchors[::-1])

    np.testing.assert_allclose(
        ascending.total_baseline, descending.total_baseline[:, ::-1], atol=1e-14
    )
