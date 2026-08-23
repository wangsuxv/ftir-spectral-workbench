from __future__ import annotations

import numpy as np
import pytest

from ftir_baseline.models import SpectrumSet
from ftir_baseline.ranges import (
    crop_spectrum_set,
    orient_spectrum_set,
    range_mask,
    restore_original_axis,
    select_array_range,
)
from ftir_baseline.validation import SpectrumValidationError


def _data(descending: bool = True) -> SpectrumSet:
    x = np.arange(900.0, 1801.0, 1.0)
    y = np.vstack((x / 1000.0, x / 500.0))
    if descending:
        x = x[::-1]
        y = y[:, ::-1]
    return SpectrumSet(
        wavenumber=x,
        perturbation=np.array([0.0, 1.0]),
        perturbation_labels=("0", "1"),
        spectra=y,
        intensity_unit="absorbance",
        source_name="synthetic",
        metadata={"experiment": "range test"},
    )


def test_crop_accepts_either_bound_order_and_preserves_axis_direction() -> None:
    descending = crop_spectrum_set(_data(True), (1600, 1000))
    ascending = crop_spectrum_set(_data(False), (1000, 1600))

    assert descending.axis_direction == "descending"
    assert ascending.axis_direction == "ascending"
    np.testing.assert_array_equal(descending.wavenumber, ascending.wavenumber[::-1])
    np.testing.assert_array_equal(descending.spectra, ascending.spectra[:, ::-1])
    assert descending.metadata["original_axis_direction"] == "descending"
    record = descending.metadata["range_selection_history"][-1]
    assert record["requested_bounds"] == (1600.0, 1000.0)
    assert record["selected_point_count"] == 601
    assert record["axis_direction_preserved"] is True


def test_range_is_inclusive() -> None:
    mask = range_mask([5, 4, 3, 2, 1], (2, 4))
    np.testing.assert_array_equal(mask, [False, True, True, True, False])


def test_select_array_range_normalizes_1d_spectrum_to_series_shape() -> None:
    x, y = select_array_range([1, 2, 3, 4], [10, 20, 30, 40], (2, 3))
    np.testing.assert_array_equal(x, [2, 3])
    np.testing.assert_array_equal(y, [[20, 30]])


def test_empty_or_one_point_range_fails_clearly() -> None:
    with pytest.raises(SpectrumValidationError, match="at least two"):
        crop_spectrum_set(_data(), (1000.0, 1000.1))


def test_strict_bounds_rejects_outside_request() -> None:
    with pytest.raises(SpectrumValidationError, match="outside available"):
        crop_spectrum_set(_data(), (2000, 1000), strict_bounds=True)


@pytest.mark.parametrize("bounds", [(1000, 1000), (np.nan, 1000), (np.inf, 1000)])
def test_invalid_bounds_fail(bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        crop_spectrum_set(_data(), bounds)


def test_explicit_orientation_and_restore_are_lossless() -> None:
    original = _data(descending=True)
    ascending = orient_spectrum_set(original, "ascending")
    restored = restore_original_axis(ascending)

    assert ascending.axis_direction == "ascending"
    assert ascending.metadata["original_axis_direction"] == "descending"
    assert ascending.metadata["axis_reversed_for_processing"] is True
    np.testing.assert_array_equal(restored.wavenumber, original.wavenumber)
    np.testing.assert_array_equal(restored.spectra, original.spectra)
    assert restored.axis_direction == "descending"


def test_noop_orientation_does_not_copy_or_change_order() -> None:
    original = _data(descending=False)
    assert orient_spectrum_set(original, "ascending") is original


def test_savgol_helper_is_estimate_only() -> None:
    pytest.importorskip("pydantic")
    from ftir_baseline.config import SmoothingConfig
    from ftir_baseline.smoothing import prepare_baseline_channels

    n_points = 9
    raw = np.zeros((1, n_points), dtype=np.float64)
    raw[0, 4] = 1.0
    channels = prepare_baseline_channels(
        raw, SmoothingConfig(enabled=True, window_length=5, polyorder=2)
    )

    assert raw.shape[-1] == n_points
    assert channels.settings["estimate_only"] is True
    np.testing.assert_array_equal(channels.raw, raw)
    assert not np.array_equal(channels.for_baseline, raw)
    np.testing.assert_array_equal(raw, np.eye(1, 9, 4))


@pytest.mark.parametrize(
    ("window", "order", "message"),
    [(4, 2, "odd"), (5, 5, "greater than"), (11, 2, "exceeds")],
)
def test_savgol_parameter_validation(window: int, order: int, message: str) -> None:
    pytest.importorskip("pydantic")
    from ftir_baseline.smoothing import savgol_estimate_only

    with pytest.raises(ValueError, match=message):
        savgol_estimate_only(np.arange(9.0), window_length=window, polyorder=order)
