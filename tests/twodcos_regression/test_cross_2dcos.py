"""Rectangular cross-region 2D-COS formulas, conventions, and validation."""

from __future__ import annotations

import numpy as np
import pytest

from ftir2dcos.twodcos import (
    CrossTwoDCOSResult,
    MatrixConvention,
    compute_cross_2dcos,
    hilbert_noda_matrix,
)


def _cross_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spectra1 = np.array(
        [
            [0.2, 1.1, -0.4],
            [0.5, 0.7, -0.1],
            [1.3, 0.0, 0.3],
            [2.0, -0.4, 1.2],
            [2.4, -0.1, 2.1],
        ],
        dtype=np.float64,
    )
    spectra2 = np.array(
        [
            [-0.3, 0.4],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.5, -0.8],
            [-0.2, -1.3],
        ],
        dtype=np.float64,
    )
    wavenumber1 = np.array([1736.0, 1620.0, 1509.0], dtype=np.float64)
    wavenumber2 = np.array([1250.0, 1140.0], dtype=np.float64)
    return spectra1, spectra2, wavenumber1, wavenumber2


def test_canonical_cross_formulas_are_rectangular_and_explicit() -> None:
    spectra1, spectra2, wavenumber1, wavenumber2 = _cross_inputs()
    result = compute_cross_2dcos(
        spectra1,
        spectra2,
        wavenumber1,
        wavenumber2,
        convention=MatrixConvention.CANONICAL,
    )

    reference1 = spectra1.mean(axis=0)
    reference2 = spectra2.mean(axis=0)
    dynamic1 = spectra1 - reference1
    dynamic2 = spectra2 - reference2
    noda = hilbert_noda_matrix(spectra1.shape[0])
    expected_sync = dynamic1.T @ dynamic2 / (spectra1.shape[0] - 1)
    expected_async = dynamic1.T @ noda @ dynamic2 / (spectra1.shape[0] - 1)

    assert isinstance(result, CrossTwoDCOSResult)
    np.testing.assert_allclose(result.reference1, reference1, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.reference2, reference2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.dynamic1, dynamic1, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.dynamic2, dynamic2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.noda, noda, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.canonical_synchronous, expected_sync, atol=1.0e-14)
    np.testing.assert_allclose(result.canonical_asynchronous, expected_async, atol=1.0e-14)
    np.testing.assert_allclose(result.synchronous, expected_sync, atol=1.0e-14)
    np.testing.assert_allclose(result.asynchronous, expected_async, atol=1.0e-14)
    assert result.synchronous.shape == (3, 2)
    np.testing.assert_array_equal(result.row_wavenumber, wavenumber1)
    np.testing.assert_array_equal(result.column_wavenumber, wavenumber2)
    np.testing.assert_array_equal(result.canonical_row_wavenumber, wavenumber1)
    np.testing.assert_array_equal(result.canonical_column_wavenumber, wavenumber2)
    assert result.canonical_row_variable == "nu1"
    assert result.canonical_column_variable == "nu2"
    assert result.row_variable == "nu1"
    assert result.column_variable == "nu2"


def test_reverse_maps_match_independent_reverse_calculation_and_sign() -> None:
    spectra1, spectra2, wavenumber1, wavenumber2 = _cross_inputs()

    for convention in MatrixConvention:
        forward = compute_cross_2dcos(
            spectra1,
            spectra2,
            wavenumber1,
            wavenumber2,
            convention=convention,
        )
        independently_reversed = compute_cross_2dcos(
            spectra2,
            spectra1,
            wavenumber2,
            wavenumber1,
            convention=convention,
        )

        np.testing.assert_allclose(
            forward.reverse_synchronous,
            independently_reversed.synchronous,
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            forward.reverse_asynchronous,
            independently_reversed.asynchronous,
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            forward.reverse_synchronous,
            forward.synchronous.T,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            forward.reverse_asynchronous,
            -forward.asynchronous.T,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            forward.reverse_row_wavenumber,
            forward.column_wavenumber,
        )
        np.testing.assert_array_equal(
            forward.reverse_column_wavenumber,
            forward.row_wavenumber,
        )
        assert forward.reverse_row_variable == forward.column_variable
        assert forward.reverse_column_variable == forward.row_variable


def test_2dpy_cross_convention_transposes_values_and_swaps_axes() -> None:
    spectra1, spectra2, wavenumber1, wavenumber2 = _cross_inputs()
    canonical = compute_cross_2dcos(
        spectra1,
        spectra2,
        wavenumber1,
        wavenumber2,
        convention="canonical",
    )
    compatible = compute_cross_2dcos(
        spectra1,
        spectra2,
        wavenumber1,
        wavenumber2,
        convention="2dpy_compatible",
    )

    np.testing.assert_allclose(
        compatible.synchronous,
        canonical.canonical_synchronous.T,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        compatible.asynchronous,
        canonical.canonical_asynchronous.T,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(compatible.row_wavenumber, wavenumber2)
    np.testing.assert_array_equal(compatible.column_wavenumber, wavenumber1)
    np.testing.assert_array_equal(compatible.wavenumber1, wavenumber1)
    np.testing.assert_array_equal(compatible.wavenumber2, wavenumber2)
    assert compatible.synchronous.shape == (2, 3)
    assert compatible.row_variable == "nu2"
    assert compatible.column_variable == "nu1"
    assert compatible.convention_metadata["final_transpose"] is True


def test_cross_qc_checks_centring_and_direct_bidirectional_identities() -> None:
    spectra1, spectra2, _, _ = _cross_inputs()
    result = compute_cross_2dcos(spectra1, spectra2, convention="canonical")
    qc = result.qc_metrics

    assert qc["dynamic1_mean_ok"] is True
    assert qc["dynamic2_mean_ok"] is True
    assert qc["sync_reverse_transpose_ok"] is True
    assert qc["async_reverse_negative_transpose_ok"] is True
    assert qc["all_checks_passed"] is True
    assert qc["dynamic1_mean_error"] <= qc["dynamic1_tolerance"]
    assert qc["dynamic2_mean_error"] <= qc["dynamic2_tolerance"]
    assert qc["sync_reverse_transpose_error"] <= qc["sync_tolerance"]
    assert qc["async_reverse_negative_transpose_error"] <= qc["async_tolerance"]


def test_cross_qc_tolerances_scale_with_matrix_magnitude() -> None:
    spectra1, spectra2, _, _ = _cross_inputs()
    atol = 1.0e-12
    rtol = 2.0e-7
    result = compute_cross_2dcos(
        spectra1 * 1.0e8,
        spectra2 * 2.0e7,
        absolute_tolerance=atol,
        relative_tolerance=rtol,
    )

    expected_sync_tolerance = atol + rtol * float(np.max(np.abs(result.synchronous)))
    expected_async_tolerance = atol + rtol * float(np.max(np.abs(result.asynchronous)))
    assert result.qc_metrics["sync_tolerance"] == pytest.approx(expected_sync_tolerance)
    assert result.qc_metrics["async_tolerance"] == pytest.approx(expected_async_tolerance)
    assert result.qc_metrics["all_checks_passed"] is True


def test_default_axes_metadata_and_float64_outputs() -> None:
    spectra1, spectra2, _, _ = _cross_inputs()
    result = compute_cross_2dcos(spectra1.astype(np.float32), spectra2.astype(np.int64))

    np.testing.assert_array_equal(result.wavenumber1, np.arange(3, dtype=np.float64))
    np.testing.assert_array_equal(result.wavenumber2, np.arange(2, dtype=np.float64))
    np.testing.assert_array_equal(result.row_wavenumber, np.arange(2, dtype=np.float64))
    np.testing.assert_array_equal(result.column_wavenumber, np.arange(3, dtype=np.float64))
    arrays = (
        result.reference1,
        result.reference2,
        result.dynamic1,
        result.dynamic2,
        result.noda,
        result.canonical_synchronous,
        result.canonical_asynchronous,
        result.synchronous,
        result.asynchronous,
    )
    assert all(array.dtype == np.float64 for array in arrays)
    assert result.metadata["analysis_type"] == "cross_region_2dcos"
    assert result.metadata["canonical_matrix_shape"] == [3, 2]
    assert result.metadata["matrix_shape"] == [2, 3]


def test_cross_result_owns_read_only_arrays_and_preserves_inputs() -> None:
    spectra1, spectra2, wavenumber1, wavenumber2 = _cross_inputs()
    original1 = spectra1.copy()
    original2 = spectra2.copy()
    original_axis1 = wavenumber1.copy()
    original_axis2 = wavenumber2.copy()
    result = compute_cross_2dcos(spectra1, spectra2, wavenumber1, wavenumber2)

    spectra1[0, 0] = -999.0
    spectra2[0, 0] = -999.0
    wavenumber1[0] = -999.0
    wavenumber2[0] = -999.0

    arrays = (
        result.reference1,
        result.reference2,
        result.dynamic1,
        result.dynamic2,
        result.noda,
        result.canonical_synchronous,
        result.canonical_asynchronous,
        result.synchronous,
        result.asynchronous,
        result.wavenumber1,
        result.wavenumber2,
        result.row_wavenumber,
        result.column_wavenumber,
        result.reverse_synchronous,
        result.reverse_asynchronous,
    )
    assert all(array.flags.writeable is False for array in arrays)
    np.testing.assert_allclose(result.reference1, original1.mean(axis=0))
    np.testing.assert_allclose(result.reference2, original2.mean(axis=0))
    np.testing.assert_array_equal(result.wavenumber1, original_axis1)
    np.testing.assert_array_equal(result.wavenumber2, original_axis2)
    with pytest.raises(ValueError, match="read-only"):
        result.synchronous[0, 0] = 1.0


@pytest.mark.parametrize(
    ("spectra1", "spectra2", "message"),
    [
        (np.ones(4), np.ones((3, 2)), "spectra1 must be 2-dimensional"),
        (np.ones((3, 2)), np.ones(4), "spectra2 must be 2-dimensional"),
        (np.ones((1, 2)), np.ones((1, 3)), "at least 2 spectra"),
        (np.empty((3, 0)), np.ones((3, 2)), "at least 1 wavenumber"),
        (np.ones((4, 2)), np.ones((3, 2)), "same number of spectra"),
        (
            np.array([[1.0, np.nan], [2.0, 3.0]]),
            np.ones((2, 2)),
            "NaN or infinite",
        ),
    ],
)
def test_invalid_cross_spectral_inputs_are_rejected(
    spectra1: np.ndarray,
    spectra2: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_cross_2dcos(spectra1, spectra2)


def test_invalid_cross_axes_convention_and_tolerances_are_rejected() -> None:
    spectra1, spectra2, _, _ = _cross_inputs()

    with pytest.raises(ValueError, match="wavenumber1 length"):
        compute_cross_2dcos(spectra1, spectra2, np.arange(2))
    with pytest.raises(ValueError, match="wavenumber2 length"):
        compute_cross_2dcos(spectra1, spectra2, wavenumber2=np.arange(3))
    with pytest.raises(ValueError, match="wavenumber1 must be 1-dimensional"):
        compute_cross_2dcos(spectra1, spectra2, np.ones((1, 3)))
    with pytest.raises(ValueError, match="Unknown 2D-COS convention"):
        compute_cross_2dcos(spectra1, spectra2, convention="invented")
    with pytest.raises(ValueError, match="absolute_tolerance must be finite and non-negative"):
        compute_cross_2dcos(spectra1, spectra2, absolute_tolerance=-1.0)
    with pytest.raises(ValueError, match="relative_tolerance must be finite and non-negative"):
        compute_cross_2dcos(spectra1, spectra2, relative_tolerance=np.inf)
    with pytest.raises(TypeError, match="spectra1 must contain real values"):
        compute_cross_2dcos(spectra1.astype(complex) + 1j, spectra2)
