"""Independent compatibility oracle for Shigeaki Morita's official 2Dpy.

Primary source inspected for these tests:
https://github.com/shigemorita/2Dpy/blob/master/2Dpy.py#L17-L64

The oracle below intentionally does not call any production Noda, centring,
correlation, or convention helper.  Its statements follow the official script:
``read_csv(..., index_col=0).T``; DataFrame mean subtraction; nested-loop Noda;
matrix multiplication; label assignment; and the final transpose of both maps.
"""

from __future__ import annotations

import math
from io import StringIO

import numpy as np
import pandas as pd

from ftir2dcos.twodcos import (
    TWODPY_SOURCE_URL,
    MatrixConvention,
    compute_2dcos,
)
from scripts.compare_external_2dpy_outputs import MatrixCsv, compare_matrices


def reference_2dpy(
    wide_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Reproduce the official homo 2Dpy statements as an independent oracle."""

    stream = StringIO()
    wide_dataframe.to_csv(stream, index=False)
    csv_text = stream.getvalue()

    # Official 2Dpy reads the same homo input into spec1 and spec2 separately.
    imported_spec1 = pd.read_csv(StringIO(csv_text), header=0, index_col=0).T
    imported_spec2 = pd.read_csv(StringIO(csv_text), header=0, index_col=0).T
    spec1 = imported_spec1.copy()
    spec2 = imported_spec2.copy()
    spec1 = spec1 - spec1.mean()
    spec2 = spec2 - spec2.mean()

    synchronous = pd.DataFrame(spec1.values.T @ spec2.values / (len(spec1) - 1))
    synchronous.index = spec1.columns
    synchronous.columns = spec2.columns
    synchronous = synchronous.T

    noda = np.zeros((len(spec1), len(spec1)))
    for row in range(len(spec1)):
        for column in range(len(spec1)):
            if row != column:
                noda[row, column] = 1.0 / math.pi / (column - row)

    asynchronous = pd.DataFrame(spec1.values.T @ noda @ spec2.values / (len(spec1) - 1))
    asynchronous.index = spec1.columns
    asynchronous.columns = spec2.columns
    asynchronous = asynchronous.T
    return imported_spec1, spec1, noda, synchronous, asynchronous


def _deterministic_wide_dataframe() -> pd.DataFrame:
    """Small nonlinear data whose asynchronous map is safely non-zero."""

    return pd.DataFrame(
        {
            "Wavenumber": [1736.0, 1662.0, 1588.0, 1509.0],
            "0 min": [0.10, 1.10, -0.40, 2.00],
            "3 min": [0.35, 0.80, -0.10, 1.70],
            "9 min": [1.20, 0.15, 0.20, 1.10],
            "20 min": [2.05, -0.25, 1.05, 0.40],
            "45 min": [2.40, -0.10, 2.20, -0.35],
        }
    )


def test_2dpy_compatible_matches_independent_statement_by_statement_oracle() -> None:
    wide = _deterministic_wide_dataframe()
    imported, spec1, expected_noda, expected_sync, expected_async = reference_2dpy(wide)
    wavenumber = wide.iloc[:, 0].to_numpy(dtype=np.float64)
    internal_spectra = wide.iloc[:, 1:].to_numpy(dtype=np.float64).T

    # This assertion independently checks the project's import transpose contract.
    np.testing.assert_array_equal(internal_spectra, imported.values)

    actual = compute_2dcos(
        internal_spectra,
        wavenumber,
        convention=MatrixConvention.TWODPY_COMPATIBLE,
    )

    np.testing.assert_allclose(actual.dynamic, spec1.values, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual.noda, expected_noda, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual.synchronous, expected_sync.values, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        actual.asynchronous, expected_async.values, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_array_equal(
        actual.row_wavenumber, expected_async.index.to_numpy(dtype=np.float64)
    )
    np.testing.assert_array_equal(
        actual.column_wavenumber, expected_async.columns.to_numpy(dtype=np.float64)
    )
    assert actual.row_variable == "nu2"
    assert actual.column_variable == "nu1"


def test_2dpy_final_transpose_reverses_homo_async_sign() -> None:
    wide = _deterministic_wide_dataframe()
    spectra = wide.iloc[:, 1:].to_numpy(dtype=np.float64).T
    wavenumber = wide.iloc[:, 0].to_numpy(dtype=np.float64)

    canonical = compute_2dcos(spectra, wavenumber, convention="canonical")
    compatible = compute_2dcos(spectra, wavenumber, convention="2dpy_compatible")

    assert np.max(np.abs(canonical.asynchronous)) > 1.0e-4
    np.testing.assert_allclose(
        compatible.synchronous,
        canonical.synchronous.T,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        compatible.asynchronous,
        canonical.asynchronous.T,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        compatible.asynchronous,
        -canonical.asynchronous,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_2dpy_axis_order_and_origin_ready_csv_direction_are_preserved() -> None:
    wide = _deterministic_wide_dataframe()
    wavenumber = wide.iloc[:, 0].to_numpy(dtype=np.float64)
    result = compute_2dcos(
        wide.iloc[:, 1:].to_numpy(dtype=np.float64).T,
        wavenumber,
        convention="2dpy_compatible",
    )

    # 2Dpy's left_large changes plot limits only; values and CSV labels retain input order.
    np.testing.assert_array_equal(result.row_wavenumber, wavenumber)
    np.testing.assert_array_equal(result.column_wavenumber, wavenumber)

    frame = pd.DataFrame(
        result.asynchronous,
        index=result.row_wavenumber,
        columns=result.column_wavenumber,
    )
    frame.index.name = "wavenumber_cm-1"
    csv_buffer = StringIO()
    frame.to_csv(csv_buffer)
    reloaded = pd.read_csv(StringIO(csv_buffer.getvalue()), index_col=0)

    np.testing.assert_array_equal(reloaded.index.to_numpy(dtype=np.float64), wavenumber)
    np.testing.assert_array_equal(reloaded.columns.to_numpy(dtype=np.float64), wavenumber)
    np.testing.assert_allclose(
        reloaded.to_numpy(dtype=np.float64),
        result.asynchronous,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_2dpy_source_and_algorithm_are_recorded_in_metadata() -> None:
    wide = _deterministic_wide_dataframe()
    result = compute_2dcos(wide.iloc[:, 1:].to_numpy().T)

    assert result.convention == "2dpy_compatible"
    assert result.convention_metadata["source_url"] == TWODPY_SOURCE_URL
    assert result.convention_metadata["final_transpose"] is True
    assert "-Psi_canonical" in result.convention_metadata["compatibility_notes"]


def test_external_comparison_reports_homo_transpose_sign_ambiguity() -> None:
    wide = _deterministic_wide_dataframe()
    spectra = wide.iloc[:, 1:].to_numpy(dtype=np.float64).T
    wavenumber = wide.iloc[:, 0].to_numpy(dtype=np.float64)
    labels = tuple(str(value) for value in wavenumber)
    canonical = compute_2dcos(spectra, wavenumber, convention="canonical")
    compatible = compute_2dcos(spectra, wavenumber, convention="2dpy_compatible")

    report = compare_matrices(
        MatrixCsv(canonical.asynchronous, labels, labels),
        MatrixCsv(compatible.asynchronous, labels, labels),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    assert report["direct_matches_tolerance"] is False
    assert report["overall_transpose"] is True
    assert report["overall_sign_reversal"] is True
    assert report["orientation_sign_ambiguity"] is True
