"""Scientific identities and validation for homo 2D-COS."""

from __future__ import annotations

import numpy as np
import pytest

from ftir2dcos.twodcos import (
    MatrixConvention,
    compute_2dcos,
    compute_asynchronous,
    compute_dynamic_spectra,
    compute_qc_metrics,
    compute_synchronous,
    hilbert_noda_matrix,
)


def _random_spectra() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260821)
    spectra = rng.normal(loc=0.3, scale=1.7, size=(9, 7)).astype(np.float64)
    wavenumber = np.linspace(1736.0, 1509.0, spectra.shape[1], dtype=np.float64)
    return spectra, wavenumber


def test_dynamic_reference_and_canonical_formulas() -> None:
    spectra, wavenumber = _random_spectra()

    result = compute_2dcos(
        spectra,
        wavenumber,
        convention=MatrixConvention.CANONICAL,
    )
    expected_reference = spectra.mean(axis=0)
    expected_dynamic = spectra - expected_reference[None, :]
    expected_noda = hilbert_noda_matrix(spectra.shape[0])

    np.testing.assert_allclose(result.reference, expected_reference, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.dynamic, expected_dynamic, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result.synchronous,
        expected_dynamic.T @ expected_dynamic / (spectra.shape[0] - 1),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        result.asynchronous,
        expected_dynamic.T @ expected_noda @ expected_dynamic / (spectra.shape[0] - 1),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_array_equal(result.row_wavenumber, wavenumber)
    np.testing.assert_array_equal(result.column_wavenumber, wavenumber)
    assert result.row_variable == "nu1"
    assert result.column_variable == "nu2"


def test_sync_async_homo_numerical_identities() -> None:
    spectra, wavenumber = _random_spectra()
    result = compute_2dcos(spectra, wavenumber, convention="canonical")

    np.testing.assert_allclose(result.synchronous, result.synchronous.T, atol=1.0e-12)
    np.testing.assert_allclose(result.asynchronous, -result.asynchronous.T, atol=1.0e-12)
    np.testing.assert_allclose(np.diag(result.asynchronous), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        np.diag(result.synchronous),
        np.var(result.dynamic, axis=0, ddof=1),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert result.qc_metrics["sync_symmetry_ok"] is True
    assert result.qc_metrics["async_antisymmetry_ok"] is True
    assert result.qc_metrics["async_diagonal_ok"] is True
    assert result.qc_metrics["sync_diagonal_variance_ok"] is True
    assert result.qc_metrics["all_checks_passed"] is True


def test_constant_input_produces_zero_dynamic_and_matrices() -> None:
    spectra = np.full((6, 5), 8.25, dtype=np.float64)

    for convention in MatrixConvention:
        result = compute_2dcos(spectra, convention=convention)
        np.testing.assert_array_equal(result.dynamic, np.zeros_like(spectra))
        np.testing.assert_array_equal(result.synchronous, np.zeros((5, 5)))
        np.testing.assert_array_equal(result.asynchronous, np.zeros((5, 5)))


@pytest.mark.parametrize("convention", list(MatrixConvention))
@pytest.mark.parametrize("scale", [-3.5, 0.25, 7.0])
def test_spectral_scaling_scales_both_maps_quadratically(
    convention: MatrixConvention,
    scale: float,
) -> None:
    spectra, wavenumber = _random_spectra()
    original = compute_2dcos(spectra, wavenumber, convention=convention)
    scaled = compute_2dcos(scale * spectra, wavenumber, convention=convention)

    np.testing.assert_allclose(
        scaled.synchronous,
        scale**2 * original.synchronous,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        scaled.asynchronous,
        scale**2 * original.asynchronous,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_public_calculation_functions_always_return_float64() -> None:
    spectra = np.arange(24).reshape(6, 4)

    reference, dynamic = compute_dynamic_spectra(spectra)
    noda = hilbert_noda_matrix(dynamic.shape[0])
    synchronous = compute_synchronous(dynamic)
    asynchronous = compute_asynchronous(dynamic, noda)

    for array in (reference, dynamic, noda, synchronous, asynchronous):
        assert array.dtype == np.float64


def test_result_owns_read_only_arrays_and_does_not_mutate_input() -> None:
    spectra, wavenumber = _random_spectra()
    original_spectra = spectra.copy()
    original_wavenumber = wavenumber.copy()

    result = compute_2dcos(spectra, wavenumber, convention="canonical")
    spectra[0, 0] = -999.0
    wavenumber[0] = -999.0

    assert result.dynamic.flags.writeable is False
    assert result.synchronous.flags.writeable is False
    np.testing.assert_array_equal(result.row_wavenumber, original_wavenumber)
    np.testing.assert_array_equal(
        result.reference,
        original_spectra.mean(axis=0),
    )


def test_qc_uses_relative_as_well_as_absolute_tolerance() -> None:
    _, dynamic = compute_dynamic_spectra(np.arange(24, dtype=np.float64).reshape(6, 4))
    synchronous = compute_synchronous(dynamic)
    asynchronous = compute_asynchronous(dynamic)
    synchronous[0, 1] += 1.0e-5
    synchronous[1, 0] -= 1.0e-5

    strict = compute_qc_metrics(
        synchronous,
        asynchronous,
        dynamic,
        absolute_tolerance=1.0e-12,
        relative_tolerance=0.0,
    )
    scale_aware = compute_qc_metrics(
        synchronous * 1.0e8,
        asynchronous * 1.0e8,
        dynamic * 1.0e4,
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-6,
    )

    assert strict["sync_symmetry_ok"] is False
    assert scale_aware["sync_symmetry_ok"] is True


@pytest.mark.parametrize(
    ("spectra", "message"),
    [
        (np.ones(4), "2-dimensional"),
        (np.ones((1, 4)), "at least 2 spectra"),
        (np.empty((3, 0)), "at least 1 wavenumber"),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), "NaN or infinite"),
        (np.array([[1.0, np.inf], [2.0, 3.0]]), "NaN or infinite"),
    ],
)
def test_invalid_spectral_input_is_rejected(spectra: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compute_2dcos(spectra)


def test_invalid_axis_and_convention_are_rejected() -> None:
    spectra = np.ones((3, 4), dtype=np.float64)

    with pytest.raises(ValueError, match="length"):
        compute_2dcos(spectra, np.arange(3))
    with pytest.raises(ValueError, match="Unknown 2D-COS convention"):
        compute_2dcos(spectra, convention="invented")
