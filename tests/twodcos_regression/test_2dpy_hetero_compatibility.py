"""Independent two-input oracle for the official 2Dpy hetero statements.

The oracle deliberately does not call production centring, Noda, cross, or
matrix-convention helpers.  It reproduces the official script's CSV transpose,
mean subtraction, nested-loop Noda construction, matrix multiplication, and
final transpose one statement at a time.
"""

from __future__ import annotations

import math
from io import StringIO

import numpy as np
import pandas as pd

from ftir2dcos.twodcos import compute_cross_2dcos


def _official_hetero_statements(
    first_wide: pd.DataFrame,
    second_wide: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    first_stream = StringIO()
    second_stream = StringIO()
    first_wide.to_csv(first_stream, index=False)
    second_wide.to_csv(second_stream, index=False)
    imported_first = pd.read_csv(
        StringIO(first_stream.getvalue()),
        header=0,
        index_col=0,
    ).T
    imported_second = pd.read_csv(
        StringIO(second_stream.getvalue()),
        header=0,
        index_col=0,
    ).T
    if len(imported_first) != len(imported_second):
        raise ValueError("official hetero inputs require the same perturbation count")

    dynamic_first = imported_first - imported_first.mean()
    dynamic_second = imported_second - imported_second.mean()
    count = len(dynamic_first)
    noda = np.zeros((count, count), dtype=np.float64)
    for row in range(count):
        for column in range(count):
            if row != column:
                noda[row, column] = 1.0 / math.pi / (column - row)

    synchronous = pd.DataFrame(
        dynamic_first.values.T @ dynamic_second.values / (count - 1),
        index=dynamic_first.columns,
        columns=dynamic_second.columns,
    ).T
    asynchronous = pd.DataFrame(
        dynamic_first.values.T @ noda @ dynamic_second.values / (count - 1),
        index=dynamic_first.columns,
        columns=dynamic_second.columns,
    ).T
    return dynamic_first, dynamic_second, noda, synchronous, asynchronous


def _hetero_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    first = pd.DataFrame(
        {
            "Wavenumber": [1736.0, 1650.0, 1540.0],
            "0 min": [0.1, 0.4, -0.2],
            "2 min": [0.3, 0.1, 0.0],
            "7 min": [0.9, -0.2, 0.5],
            "15 min": [1.4, 0.3, 1.1],
        }
    )
    second = pd.DataFrame(
        {
            "Wavenumber": [1250.0, 1190.0, 1140.0, 1080.0],
            "0 min": [0.2, -0.1, 0.8, 0.0],
            "2 min": [0.5, 0.0, 0.4, -0.3],
            "7 min": [0.7, 0.6, -0.2, 0.2],
            "15 min": [1.5, 1.0, 0.3, 0.9],
        }
    )
    return first, second


def test_two_distinct_inputs_match_independent_2dpy_hetero_oracle() -> None:
    first, second = _hetero_inputs()
    expected_first, expected_second, expected_noda, expected_sync, expected_async = (
        _official_hetero_statements(first, second)
    )
    axis_first = first.iloc[:, 0].to_numpy(dtype=np.float64)
    axis_second = second.iloc[:, 0].to_numpy(dtype=np.float64)
    spectra_first = first.iloc[:, 1:].to_numpy(dtype=np.float64).T
    spectra_second = second.iloc[:, 1:].to_numpy(dtype=np.float64).T

    actual = compute_cross_2dcos(
        spectra_first,
        spectra_second,
        axis_first,
        axis_second,
        convention="2dpy_compatible",
    )

    np.testing.assert_allclose(
        actual.dynamic1,
        expected_first.values,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        actual.dynamic2,
        expected_second.values,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        actual.noda,
        expected_noda,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        actual.synchronous,
        expected_sync.values,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        actual.asynchronous,
        expected_async.values,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(
        actual.row_wavenumber,
        expected_sync.index.to_numpy(dtype=np.float64),
    )
    np.testing.assert_array_equal(
        actual.column_wavenumber,
        expected_sync.columns.to_numpy(dtype=np.float64),
    )
    assert actual.row_variable == "nu2"
    assert actual.column_variable == "nu1"
    assert "hetero/two-input" in actual.convention_metadata["compatibility_notes"]


def test_hetero_reverse_maps_are_views_not_a_second_scientific_result() -> None:
    first, second = _hetero_inputs()
    actual = compute_cross_2dcos(
        first.iloc[:, 1:].to_numpy(dtype=np.float64).T,
        second.iloc[:, 1:].to_numpy(dtype=np.float64).T,
        first.iloc[:, 0].to_numpy(dtype=np.float64),
        second.iloc[:, 0].to_numpy(dtype=np.float64),
        convention="2dpy_compatible",
    )

    np.testing.assert_array_equal(actual.reverse_synchronous, actual.synchronous.T)
    np.testing.assert_array_equal(actual.reverse_asynchronous, -actual.asynchronous.T)
    np.testing.assert_array_equal(actual.reverse_row_wavenumber, actual.column_wavenumber)
    np.testing.assert_array_equal(actual.reverse_column_wavenumber, actual.row_wavenumber)
