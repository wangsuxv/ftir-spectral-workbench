"""Numerically explicit homo and cross-region 2D-COS calculation engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import (
    MatrixConvention,
    apply_matrix_convention,
    get_convention_spec,
    normalize_convention,
)
from .noda import hilbert_noda_matrix

FloatArray = NDArray[np.float64]
QCMetrics = dict[str, float | bool]

DEFAULT_ABSOLUTE_TOLERANCE = 1.0e-10
DEFAULT_RELATIVE_TOLERANCE = 1.0e-8


def _numeric_array(values: ArrayLike, *, name: str, ndim: int) -> FloatArray:
    """Return an owned finite real ``float64`` array with a fixed dimension."""

    if np.iscomplexobj(values):
        raise TypeError(f"{name} must contain real values")
    try:
        array = np.array(values, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain numeric values") from error
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _spectral_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    matrix = _numeric_array(values, name=name, ndim=2)
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} must contain at least 2 spectra")
    if matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least 1 wavenumber")
    return matrix


def _correlation_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    matrix = _numeric_array(values, name=name, ndim=2)
    if matrix.shape[0] < 1 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a non-empty square matrix; got shape {matrix.shape}")
    return matrix


def _nonnegative_tolerance(value: float, *, name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _readonly(values: ArrayLike, *, ndim: int, name: str) -> FloatArray:
    array = _numeric_array(values, name=name, ndim=ndim)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class TwoDCOSResult:
    """Complete 2D-COS numerical result with explicit output axes."""

    reference: FloatArray
    dynamic: FloatArray
    noda: FloatArray
    canonical_synchronous: FloatArray
    canonical_asynchronous: FloatArray
    synchronous: FloatArray
    asynchronous: FloatArray
    row_wavenumber: FloatArray
    column_wavenumber: FloatArray
    convention: str
    row_variable: str
    column_variable: str
    convention_metadata: dict[str, Any]
    qc_metrics: QCMetrics

    def __post_init__(self) -> None:
        reference = _readonly(self.reference, ndim=1, name="reference")
        dynamic = _readonly(self.dynamic, ndim=2, name="dynamic")
        noda = _readonly(self.noda, ndim=2, name="noda")
        canonical_sync = _readonly(self.canonical_synchronous, ndim=2, name="canonical_synchronous")
        canonical_async = _readonly(
            self.canonical_asynchronous, ndim=2, name="canonical_asynchronous"
        )
        synchronous = _readonly(self.synchronous, ndim=2, name="synchronous")
        asynchronous = _readonly(self.asynchronous, ndim=2, name="asynchronous")
        row_wavenumber = _readonly(self.row_wavenumber, ndim=1, name="row_wavenumber")
        column_wavenumber = _readonly(self.column_wavenumber, ndim=1, name="column_wavenumber")

        n_spectra, n_wavenumbers = dynamic.shape
        if reference.shape != (n_wavenumbers,):
            raise ValueError("reference length must match dynamic.shape[1]")
        if noda.shape != (n_spectra, n_spectra):
            raise ValueError("noda shape must be (n_spectra, n_spectra)")
        expected_matrix_shape = (n_wavenumbers, n_wavenumbers)
        matrices = {
            "canonical_synchronous": canonical_sync,
            "canonical_asynchronous": canonical_async,
            "synchronous": synchronous,
            "asynchronous": asynchronous,
        }
        for matrix_name, matrix in matrices.items():
            if matrix.shape != expected_matrix_shape:
                raise ValueError(
                    f"{matrix_name} must have shape {expected_matrix_shape}; got {matrix.shape}"
                )
        if row_wavenumber.shape != (n_wavenumbers,):
            raise ValueError("row_wavenumber length must match correlation matrix rows")
        if column_wavenumber.shape != (n_wavenumbers,):
            raise ValueError("column_wavenumber length must match correlation matrix columns")

        normalized_convention = normalize_convention(self.convention)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "dynamic", dynamic)
        object.__setattr__(self, "noda", noda)
        object.__setattr__(self, "canonical_synchronous", canonical_sync)
        object.__setattr__(self, "canonical_asynchronous", canonical_async)
        object.__setattr__(self, "synchronous", synchronous)
        object.__setattr__(self, "asynchronous", asynchronous)
        object.__setattr__(self, "row_wavenumber", row_wavenumber)
        object.__setattr__(self, "column_wavenumber", column_wavenumber)
        object.__setattr__(self, "convention", normalized_convention.value)
        object.__setattr__(self, "row_variable", str(self.row_variable))
        object.__setattr__(self, "column_variable", str(self.column_variable))
        object.__setattr__(self, "convention_metadata", deepcopy(self.convention_metadata))
        object.__setattr__(self, "qc_metrics", dict(self.qc_metrics))

    @property
    def dynamic_spectra(self) -> FloatArray:
        """Alias useful to exporters and plotting clients."""

        return self.dynamic

    @property
    def synchronous_matrix(self) -> FloatArray:
        """Alias for the convention-oriented synchronous matrix."""

        return self.synchronous

    @property
    def asynchronous_matrix(self) -> FloatArray:
        """Alias for the convention-oriented asynchronous matrix."""

        return self.asynchronous

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a copy of convention and axis metadata."""

        return {
            **deepcopy(self.convention_metadata),
            "convention": self.convention,
            "row_variable": self.row_variable,
            "column_variable": self.column_variable,
            "row_wavenumber": self.row_wavenumber.tolist(),
            "column_wavenumber": self.column_wavenumber.tolist(),
        }


@dataclass(frozen=True, slots=True)
class CrossTwoDCOSResult:
    """Complete rectangular cross-region 2D-COS result.

    The canonical matrices correlate ``spectra1`` (rows, ``nu1``) with
    ``spectra2`` (columns, ``nu2``).  The convention-oriented matrices
    may transpose that layout; their matching axes are always exposed through
    ``row_wavenumber`` and ``column_wavenumber``.
    """

    reference1: FloatArray
    reference2: FloatArray
    dynamic1: FloatArray
    dynamic2: FloatArray
    noda: FloatArray
    canonical_synchronous: FloatArray
    canonical_asynchronous: FloatArray
    synchronous: FloatArray
    asynchronous: FloatArray
    wavenumber1: FloatArray
    wavenumber2: FloatArray
    row_wavenumber: FloatArray
    column_wavenumber: FloatArray
    convention: str
    row_variable: str
    column_variable: str
    convention_metadata: dict[str, Any]
    qc_metrics: QCMetrics

    def __post_init__(self) -> None:
        reference1 = _readonly(self.reference1, ndim=1, name="reference1")
        reference2 = _readonly(self.reference2, ndim=1, name="reference2")
        dynamic1 = _readonly(self.dynamic1, ndim=2, name="dynamic1")
        dynamic2 = _readonly(self.dynamic2, ndim=2, name="dynamic2")
        noda = _readonly(self.noda, ndim=2, name="noda")
        canonical_sync = _readonly(
            self.canonical_synchronous,
            ndim=2,
            name="canonical_synchronous",
        )
        canonical_async = _readonly(
            self.canonical_asynchronous,
            ndim=2,
            name="canonical_asynchronous",
        )
        synchronous = _readonly(self.synchronous, ndim=2, name="synchronous")
        asynchronous = _readonly(self.asynchronous, ndim=2, name="asynchronous")
        wavenumber1 = _readonly(self.wavenumber1, ndim=1, name="wavenumber1")
        wavenumber2 = _readonly(self.wavenumber2, ndim=1, name="wavenumber2")
        row_wavenumber = _readonly(self.row_wavenumber, ndim=1, name="row_wavenumber")
        column_wavenumber = _readonly(
            self.column_wavenumber,
            ndim=1,
            name="column_wavenumber",
        )

        n_spectra1, n_wavenumbers1 = dynamic1.shape
        n_spectra2, n_wavenumbers2 = dynamic2.shape
        if n_spectra1 != n_spectra2:
            raise ValueError("dynamic1 and dynamic2 must contain the same number of spectra")
        if reference1.shape != (n_wavenumbers1,):
            raise ValueError("reference1 length must match dynamic1.shape[1]")
        if reference2.shape != (n_wavenumbers2,):
            raise ValueError("reference2 length must match dynamic2.shape[1]")
        if noda.shape != (n_spectra1, n_spectra1):
            raise ValueError("noda shape must be (n_spectra, n_spectra)")
        canonical_shape = (n_wavenumbers1, n_wavenumbers2)
        for matrix_name, matrix in {
            "canonical_synchronous": canonical_sync,
            "canonical_asynchronous": canonical_async,
        }.items():
            if matrix.shape != canonical_shape:
                raise ValueError(
                    f"{matrix_name} must have shape {canonical_shape}; got {matrix.shape}"
                )

        normalized_convention = normalize_convention(self.convention)
        oriented_shape = (
            canonical_shape[::-1]
            if normalized_convention is MatrixConvention.TWODPY_COMPATIBLE
            else canonical_shape
        )
        for matrix_name, matrix in {
            "synchronous": synchronous,
            "asynchronous": asynchronous,
        }.items():
            if matrix.shape != oriented_shape:
                raise ValueError(
                    f"{matrix_name} must have shape {oriented_shape}; got {matrix.shape}"
                )
        if wavenumber1.shape != (n_wavenumbers1,):
            raise ValueError("wavenumber1 length must match dynamic1.shape[1]")
        if wavenumber2.shape != (n_wavenumbers2,):
            raise ValueError("wavenumber2 length must match dynamic2.shape[1]")
        if row_wavenumber.shape != (oriented_shape[0],):
            raise ValueError("row_wavenumber length must match correlation matrix rows")
        if column_wavenumber.shape != (oriented_shape[1],):
            raise ValueError("column_wavenumber length must match correlation matrix columns")

        object.__setattr__(self, "reference1", reference1)
        object.__setattr__(self, "reference2", reference2)
        object.__setattr__(self, "dynamic1", dynamic1)
        object.__setattr__(self, "dynamic2", dynamic2)
        object.__setattr__(self, "noda", noda)
        object.__setattr__(self, "canonical_synchronous", canonical_sync)
        object.__setattr__(self, "canonical_asynchronous", canonical_async)
        object.__setattr__(self, "synchronous", synchronous)
        object.__setattr__(self, "asynchronous", asynchronous)
        object.__setattr__(self, "wavenumber1", wavenumber1)
        object.__setattr__(self, "wavenumber2", wavenumber2)
        object.__setattr__(self, "row_wavenumber", row_wavenumber)
        object.__setattr__(self, "column_wavenumber", column_wavenumber)
        object.__setattr__(self, "convention", normalized_convention.value)
        object.__setattr__(self, "row_variable", str(self.row_variable))
        object.__setattr__(self, "column_variable", str(self.column_variable))
        object.__setattr__(self, "convention_metadata", deepcopy(self.convention_metadata))
        object.__setattr__(self, "qc_metrics", dict(self.qc_metrics))

    @property
    def dynamic_spectra1(self) -> FloatArray:
        """Alias for the first input's dynamic spectra."""

        return self.dynamic1

    @property
    def dynamic_spectra2(self) -> FloatArray:
        """Alias for the second input's dynamic spectra."""

        return self.dynamic2

    @property
    def canonical_row_wavenumber(self) -> FloatArray:
        """Wavenumber axis for canonical matrix rows (input 1)."""

        return self.wavenumber1

    @property
    def canonical_column_wavenumber(self) -> FloatArray:
        """Wavenumber axis for canonical matrix columns (input 2)."""

        return self.wavenumber2

    @property
    def canonical_row_variable(self) -> str:
        """Variable name for canonical matrix rows."""

        return "nu1"

    @property
    def canonical_column_variable(self) -> str:
        """Variable name for canonical matrix columns."""

        return "nu2"

    @property
    def synchronous_matrix(self) -> FloatArray:
        """Alias for the convention-oriented synchronous matrix."""

        return self.synchronous

    @property
    def asynchronous_matrix(self) -> FloatArray:
        """Alias for the convention-oriented asynchronous matrix."""

        return self.asynchronous

    @property
    def reverse_synchronous(self) -> FloatArray:
        """Return the read-only convention-oriented spectrum-2-to-1 map."""

        return self.synchronous.T

    @property
    def reverse_asynchronous(self) -> FloatArray:
        """Return the read-only reverse map, ``-asynchronous.T``."""

        matrix = np.asarray(-self.asynchronous.T, dtype=np.float64, order="C")
        matrix.setflags(write=False)
        return matrix

    @property
    def reverse_row_wavenumber(self) -> FloatArray:
        """Axis for rows of the reverse maps."""

        return self.column_wavenumber

    @property
    def reverse_column_wavenumber(self) -> FloatArray:
        """Axis for columns of the reverse maps."""

        return self.row_wavenumber

    @property
    def reverse_row_variable(self) -> str:
        """Variable name for rows of the reverse maps."""

        return self.column_variable

    @property
    def reverse_column_variable(self) -> str:
        """Variable name for columns of the reverse maps."""

        return self.row_variable

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a copy of convention, input, and oriented-axis metadata."""

        return {
            **deepcopy(self.convention_metadata),
            "convention": self.convention,
            "row_variable": self.row_variable,
            "column_variable": self.column_variable,
            "wavenumber1": self.wavenumber1.tolist(),
            "wavenumber2": self.wavenumber2.tolist(),
            "row_wavenumber": self.row_wavenumber.tolist(),
            "column_wavenumber": self.column_wavenumber.tolist(),
        }


def compute_dynamic_spectra(spectra: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return the mean reference and column-centred dynamic spectra.

    The required input orientation is ``(n_spectra, n_wavenumbers)``.  No
    sorting, interpolation, normalization, or other preprocessing is applied.
    """

    matrix = _spectral_matrix(spectra, name="spectra")
    reference = matrix.mean(axis=0, dtype=np.float64)
    dynamic = matrix - reference[None, :]
    return reference.astype(np.float64, copy=False), dynamic.astype(np.float64, copy=False)


def compute_synchronous(dynamic: ArrayLike) -> FloatArray:
    """Compute canonical ``Phi = D.T @ D / (m - 1)``."""

    matrix = _spectral_matrix(dynamic, name="dynamic")
    return np.asarray(matrix.T @ matrix / (matrix.shape[0] - 1), dtype=np.float64)


def compute_asynchronous(
    dynamic: ArrayLike,
    noda: ArrayLike | None = None,
) -> FloatArray:
    """Compute canonical ``Psi = D.T @ N @ D / (m - 1)``."""

    matrix = _spectral_matrix(dynamic, name="dynamic")
    if noda is None:
        noda_matrix = hilbert_noda_matrix(matrix.shape[0])
    else:
        noda_matrix = _correlation_matrix(noda, name="noda")
        expected_shape = (matrix.shape[0], matrix.shape[0])
        if noda_matrix.shape != expected_shape:
            raise ValueError(f"noda must have shape {expected_shape}; got {noda_matrix.shape}")
    return np.asarray(matrix.T @ noda_matrix @ matrix / (matrix.shape[0] - 1), dtype=np.float64)


def compute_qc_metrics(
    synchronous: ArrayLike,
    asynchronous: ArrayLike,
    dynamic: ArrayLike,
    *,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> QCMetrics:
    """Measure homo 2D-COS symmetry, antisymmetry, diagonal, and variance identities."""

    sync = _correlation_matrix(synchronous, name="synchronous")
    async_ = _correlation_matrix(asynchronous, name="asynchronous")
    dynamic_matrix = _spectral_matrix(dynamic, name="dynamic")
    expected_shape = (dynamic_matrix.shape[1], dynamic_matrix.shape[1])
    if sync.shape != expected_shape or async_.shape != expected_shape:
        raise ValueError(
            "correlation matrices must both have shape "
            f"{expected_shape}; got {sync.shape} and {async_.shape}"
        )

    atol = _nonnegative_tolerance(absolute_tolerance, name="absolute_tolerance")
    rtol = _nonnegative_tolerance(relative_tolerance, name="relative_tolerance")

    sync_symmetry_error = float(np.max(np.abs(sync - sync.T)))
    async_antisymmetry_error = float(np.max(np.abs(async_ + async_.T)))
    async_diagonal_error = float(np.max(np.abs(np.diag(async_))))
    sample_variance = np.var(dynamic_matrix, axis=0, ddof=1, dtype=np.float64)
    variance_error = float(np.max(np.abs(np.diag(sync) - sample_variance)))

    sync_scale = float(max(np.max(np.abs(sync)), np.max(np.abs(sample_variance))))
    async_scale = float(np.max(np.abs(async_)))
    sync_tolerance = atol + rtol * sync_scale
    async_tolerance = atol + rtol * async_scale

    sync_ok = sync_symmetry_error <= sync_tolerance
    async_ok = async_antisymmetry_error <= async_tolerance
    async_diagonal_ok = async_diagonal_error <= async_tolerance
    variance_ok = variance_error <= sync_tolerance

    return {
        "sync_symmetry_error": sync_symmetry_error,
        "async_antisymmetry_error": async_antisymmetry_error,
        "async_diagonal_error": async_diagonal_error,
        "sync_diagonal_variance_error": variance_error,
        "sync_diagonal_sample_variance_error": variance_error,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "sync_tolerance": sync_tolerance,
        "async_tolerance": async_tolerance,
        "sync_symmetry_ok": sync_ok,
        "async_antisymmetry_ok": async_ok,
        "async_diagonal_ok": async_diagonal_ok,
        "sync_diagonal_variance_ok": variance_ok,
        "all_checks_passed": sync_ok and async_ok and async_diagonal_ok and variance_ok,
    }


def _compute_cross_qc_metrics(
    canonical_synchronous: FloatArray,
    canonical_asynchronous: FloatArray,
    reverse_synchronous_direct: FloatArray,
    reverse_asynchronous_direct: FloatArray,
    dynamic1: FloatArray,
    dynamic2: FloatArray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> QCMetrics:
    """Check centring and bidirectional identities for a rectangular cross map."""

    atol = _nonnegative_tolerance(absolute_tolerance, name="absolute_tolerance")
    rtol = _nonnegative_tolerance(relative_tolerance, name="relative_tolerance")

    dynamic1_mean_error = float(np.max(np.abs(np.mean(dynamic1, axis=0, dtype=np.float64))))
    dynamic2_mean_error = float(np.max(np.abs(np.mean(dynamic2, axis=0, dtype=np.float64))))
    sync_reverse_transpose_error = float(
        np.max(np.abs(reverse_synchronous_direct - canonical_synchronous.T))
    )
    async_reverse_negative_transpose_error = float(
        np.max(np.abs(reverse_asynchronous_direct + canonical_asynchronous.T))
    )

    dynamic1_scale = float(np.max(np.abs(dynamic1)))
    dynamic2_scale = float(np.max(np.abs(dynamic2)))
    sync_scale = float(
        max(
            np.max(np.abs(canonical_synchronous)),
            np.max(np.abs(reverse_synchronous_direct)),
        )
    )
    async_scale = float(
        max(
            np.max(np.abs(canonical_asynchronous)),
            np.max(np.abs(reverse_asynchronous_direct)),
        )
    )
    dynamic1_tolerance = atol + rtol * dynamic1_scale
    dynamic2_tolerance = atol + rtol * dynamic2_scale
    sync_tolerance = atol + rtol * sync_scale
    async_tolerance = atol + rtol * async_scale

    dynamic1_mean_ok = dynamic1_mean_error <= dynamic1_tolerance
    dynamic2_mean_ok = dynamic2_mean_error <= dynamic2_tolerance
    sync_reverse_transpose_ok = sync_reverse_transpose_error <= sync_tolerance
    async_reverse_negative_transpose_ok = async_reverse_negative_transpose_error <= async_tolerance

    return {
        "dynamic1_mean_error": dynamic1_mean_error,
        "dynamic2_mean_error": dynamic2_mean_error,
        "sync_reverse_transpose_error": sync_reverse_transpose_error,
        "async_reverse_negative_transpose_error": async_reverse_negative_transpose_error,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "dynamic1_tolerance": dynamic1_tolerance,
        "dynamic2_tolerance": dynamic2_tolerance,
        "sync_tolerance": sync_tolerance,
        "async_tolerance": async_tolerance,
        "dynamic1_mean_ok": dynamic1_mean_ok,
        "dynamic2_mean_ok": dynamic2_mean_ok,
        "sync_reverse_transpose_ok": sync_reverse_transpose_ok,
        "async_reverse_negative_transpose_ok": async_reverse_negative_transpose_ok,
        "all_checks_passed": (
            dynamic1_mean_ok
            and dynamic2_mean_ok
            and sync_reverse_transpose_ok
            and async_reverse_negative_transpose_ok
        ),
    }


def compute_cross_2dcos(
    spectra1: ArrayLike,
    spectra2: ArrayLike,
    wavenumber1: ArrayLike | None = None,
    wavenumber2: ArrayLike | None = None,
    *,
    convention: MatrixConvention | str = MatrixConvention.TWODPY_COMPATIBLE,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> CrossTwoDCOSResult:
    """Calculate a rectangular cross-region 2D-COS result.

    Both spectral matrices must use orientation ``(n_spectra, n_wavenumbers)``
    and must describe the same spectra/perturbation sequence.  Their numbers of
    wavenumbers may differ.  Canonical matrices are ``D1.T @ D2 / (m - 1)``
    and ``D1.T @ N @ D2 / (m - 1)``.  ``2dpy_compatible`` extends official
    2Dpy's documented homo-spectrum final-transpose orientation to this
    rectangular result, yielding rows from input 2 and columns from input 1;
    it does not claim that 2Dpy itself exposes a two-input cross API.
    """

    matrix1 = _spectral_matrix(spectra1, name="spectra1")
    matrix2 = _spectral_matrix(spectra2, name="spectra2")
    if matrix1.shape[0] != matrix2.shape[0]:
        raise ValueError(
            "spectra1 and spectra2 must contain the same number of spectra; "
            f"got {matrix1.shape[0]} and {matrix2.shape[0]}"
        )

    if wavenumber1 is None:
        axis1 = np.arange(matrix1.shape[1], dtype=np.float64)
    else:
        axis1 = _numeric_array(wavenumber1, name="wavenumber1", ndim=1)
        if axis1.shape != (matrix1.shape[1],):
            raise ValueError(
                "wavenumber1 length must match spectra1.shape[1]; "
                f"got {axis1.shape} and {matrix1.shape}"
            )
    if wavenumber2 is None:
        axis2 = np.arange(matrix2.shape[1], dtype=np.float64)
    else:
        axis2 = _numeric_array(wavenumber2, name="wavenumber2", ndim=1)
        if axis2.shape != (matrix2.shape[1],):
            raise ValueError(
                "wavenumber2 length must match spectra2.shape[1]; "
                f"got {axis2.shape} and {matrix2.shape}"
            )

    normalized_convention = normalize_convention(convention)
    spec = get_convention_spec(normalized_convention)
    reference1, dynamic1 = compute_dynamic_spectra(matrix1)
    reference2, dynamic2 = compute_dynamic_spectra(matrix2)
    n_spectra = matrix1.shape[0]
    noda = hilbert_noda_matrix(n_spectra)
    denominator = n_spectra - 1
    canonical_sync = np.asarray(dynamic1.T @ dynamic2 / denominator, dtype=np.float64)
    canonical_async = np.asarray(dynamic1.T @ noda @ dynamic2 / denominator, dtype=np.float64)

    # Calculate the reverse direction independently for an auditable numerical
    # check rather than defining it from the expected transpose identities.
    reverse_sync_direct = np.asarray(dynamic2.T @ dynamic1 / denominator, dtype=np.float64)
    reverse_async_direct = np.asarray(
        dynamic2.T @ noda @ dynamic1 / denominator,
        dtype=np.float64,
    )
    qc_metrics = _compute_cross_qc_metrics(
        canonical_sync,
        canonical_async,
        reverse_sync_direct,
        reverse_async_direct,
        dynamic1,
        dynamic2,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )

    synchronous, asynchronous = apply_matrix_convention(
        canonical_sync,
        canonical_async,
        normalized_convention,
    )
    if spec.final_transpose:
        row_axis, column_axis = axis2, axis1
    else:
        row_axis, column_axis = axis1, axis2

    metadata = spec.to_dict()
    metadata.update(
        {
            "analysis_type": "cross_region_2dcos",
            "input_spectra1_shape": list(matrix1.shape),
            "input_spectra2_shape": list(matrix2.shape),
            "canonical_matrix_shape": list(canonical_sync.shape),
            "matrix_shape": list(synchronous.shape),
            "canonical_row_variable": "nu1",
            "canonical_column_variable": "nu2",
            "synchronous_formula": "Phi12 = D1.T @ D2 / (m - 1)",
            "asynchronous_formula": "Psi12 = D1.T @ N @ D2 / (m - 1)",
            "reverse_synchronous_identity": "Phi21 = Phi12.T",
            "reverse_asynchronous_identity": "Psi21 = -Psi12.T",
            "input_layout": (
                "D1.shape == (n_spectra, n_wavenumbers1); D2.shape == (n_spectra, n_wavenumbers2)"
            ),
            "dtype": "float64",
            "perturbation_grid_strategy": "shared_index_order",
            "mean_reference": "independent_arithmetic_mean_across_spectra_for_each_input",
        }
    )
    if normalized_convention is MatrixConvention.TWODPY_COMPATIBLE:
        metadata["compatibility_notes"] = (
            "The exported cross matrices are Phi12.T and Psi12.T, with rows from "
            "spectrum 2 and columns from spectrum 1.  The independently computed reverse "
            "asynchronous map remains -asynchronous.T.  This is an explicit extension of "
            "2Dpy's homo-spectrum final-transpose orientation; the referenced 2Dpy script "
            "does not provide a two-input cross-correlation API."
        )

    return CrossTwoDCOSResult(
        reference1=reference1,
        reference2=reference2,
        dynamic1=dynamic1,
        dynamic2=dynamic2,
        noda=noda,
        canonical_synchronous=canonical_sync,
        canonical_asynchronous=canonical_async,
        synchronous=synchronous,
        asynchronous=asynchronous,
        wavenumber1=axis1,
        wavenumber2=axis2,
        row_wavenumber=row_axis,
        column_wavenumber=column_axis,
        convention=normalized_convention.value,
        row_variable=spec.row_variable,
        column_variable=spec.column_variable,
        convention_metadata=metadata,
        qc_metrics=qc_metrics,
    )


def compute_2dcos(
    spectra: ArrayLike,
    wavenumber: ArrayLike | None = None,
    *,
    convention: MatrixConvention | str = MatrixConvention.TWODPY_COMPATIBLE,
    absolute_tolerance: float = DEFAULT_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> TwoDCOSResult:
    """Calculate a complete homo 2D-COS result in one auditable operation.

    ``spectra`` must use the project-wide internal orientation
    ``(n_spectra, n_wavenumbers)``.  The supplied wavenumber order is retained
    exactly.  ``2dpy_compatible`` models official 2Dpy's final transpose; it
    does not reverse an ascending or descending wavenumber axis.
    """

    matrix = _spectral_matrix(spectra, name="spectra")
    if wavenumber is None:
        axis = np.arange(matrix.shape[1], dtype=np.float64)
    else:
        axis = _numeric_array(wavenumber, name="wavenumber", ndim=1)
        if axis.shape != (matrix.shape[1],):
            raise ValueError(
                "wavenumber length must match spectra.shape[1]; "
                f"got {axis.shape} and {matrix.shape}"
            )

    normalized_convention = normalize_convention(convention)
    spec = get_convention_spec(normalized_convention)
    reference, dynamic = compute_dynamic_spectra(matrix)
    noda = hilbert_noda_matrix(matrix.shape[0])
    canonical_sync = compute_synchronous(dynamic)
    canonical_async = compute_asynchronous(dynamic, noda)
    synchronous, asynchronous = apply_matrix_convention(
        canonical_sync, canonical_async, normalized_convention
    )
    qc_metrics = compute_qc_metrics(
        synchronous,
        asynchronous,
        dynamic,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )

    metadata = spec.to_dict()
    metadata.update(
        {
            "input_spectra_shape": list(matrix.shape),
            "matrix_shape": list(synchronous.shape),
            "dtype": "float64",
            "perturbation_grid_strategy": "index_order",
            "mean_reference": "arithmetic_mean_across_spectra",
        }
    )

    return TwoDCOSResult(
        reference=reference,
        dynamic=dynamic,
        noda=noda,
        canonical_synchronous=canonical_sync,
        canonical_asynchronous=canonical_async,
        synchronous=synchronous,
        asynchronous=asynchronous,
        row_wavenumber=axis,
        column_wavenumber=axis,
        convention=normalized_convention.value,
        row_variable=spec.row_variable,
        column_variable=spec.column_variable,
        convention_metadata=metadata,
        qc_metrics=qc_metrics,
    )


# Readable aliases for integrations that prefer a verb other than ``compute``.
calculate_2dcos = compute_2dcos
compute_2d_correlation = compute_2dcos


__all__ = [
    "DEFAULT_ABSOLUTE_TOLERANCE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "CrossTwoDCOSResult",
    "QCMetrics",
    "TwoDCOSResult",
    "calculate_2dcos",
    "compute_2d_correlation",
    "compute_2dcos",
    "compute_asynchronous",
    "compute_cross_2dcos",
    "compute_dynamic_spectra",
    "compute_qc_metrics",
    "compute_synchronous",
]
