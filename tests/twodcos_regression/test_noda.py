"""Tests for the index-order Hilbert--Noda transformation matrix."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ftir2dcos.twodcos import hilbert_noda_matrix


def _loop_reference(count: int) -> np.ndarray:
    reference = np.zeros((count, count), dtype=np.float64)
    for row in range(count):
        for column in range(count):
            if row != column:
                reference[row, column] = 1.0 / math.pi / (column - row)
    return reference


@pytest.mark.parametrize("count", [3, 4, 10])
def test_hilbert_noda_matrix_matches_double_loop(count: int) -> None:
    actual = hilbert_noda_matrix(count)

    np.testing.assert_allclose(actual, _loop_reference(count), rtol=0.0, atol=0.0)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(np.diag(actual), np.zeros(count, dtype=np.float64))
    np.testing.assert_allclose(actual.T, -actual, rtol=0.0, atol=0.0)


def test_hilbert_noda_single_spectrum_is_well_defined() -> None:
    actual = hilbert_noda_matrix(1)

    np.testing.assert_array_equal(actual, np.zeros((1, 1), dtype=np.float64))


@pytest.mark.parametrize("invalid", [0, -1])
def test_hilbert_noda_rejects_nonpositive_counts(invalid: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        hilbert_noda_matrix(invalid)


@pytest.mark.parametrize("invalid", [True, 3.0, "3"])
def test_hilbert_noda_rejects_non_integer_counts(invalid: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        hilbert_noda_matrix(invalid)  # type: ignore[arg-type]
