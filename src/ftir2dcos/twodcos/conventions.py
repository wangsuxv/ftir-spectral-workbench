"""Matrix orientation conventions for two-dimensional correlation spectra.

The compatibility convention is based on Shigeaki Morita's official 2Dpy
script.  The source reads a wide CSV and transposes it, mean-centres every
spectral-variable column, calculates the two correlation matrices, and then
transposes both matrices before plotting and writing them.  Keeping that final
transpose explicit is essential: for homo 2D-COS it leaves the symmetric map
unchanged, but reverses the sign of the antisymmetric map.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

TWODPY_REPOSITORY_URL = "https://github.com/shigemorita/2Dpy"
TWODPY_SOURCE_URL = "https://github.com/shigemorita/2Dpy/blob/master/2Dpy.py#L17-L64"
TWODPY_RAW_SOURCE_URL = "https://raw.githubusercontent.com/shigemorita/2Dpy/master/2Dpy.py"


class MatrixConvention(StrEnum):
    """Supported row/column orientation and sign conventions."""

    CANONICAL = "canonical"
    TWODPY_COMPATIBLE = "2dpy_compatible"


@dataclass(frozen=True, slots=True)
class ConventionSpec:
    """Auditable description of one matrix convention."""

    name: str
    row_variable: str
    column_variable: str
    final_transpose: bool
    synchronous_formula: str
    asynchronous_formula: str
    input_layout: str
    axis_order_behavior: str
    source_url: str | None = None
    compatibility_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible convention metadata."""

        return asdict(self)


_CANONICAL_SPEC = ConventionSpec(
    name=MatrixConvention.CANONICAL.value,
    row_variable="nu1",
    column_variable="nu2",
    final_transpose=False,
    synchronous_formula="Phi = D.T @ D / (m - 1)",
    asynchronous_formula="Psi = D.T @ N @ D / (m - 1)",
    input_layout="D.shape == (n_spectra, n_wavenumbers)",
    axis_order_behavior="Preserve the supplied wavenumber order; do not sort or reverse it.",
)

_TWODPY_SPEC = ConventionSpec(
    name=MatrixConvention.TWODPY_COMPATIBLE.value,
    row_variable="nu2",
    column_variable="nu1",
    final_transpose=True,
    synchronous_formula="Phi_2Dpy = (D.T @ D / (m - 1)).T",
    asynchronous_formula="Psi_2Dpy = (D.T @ N @ D / (m - 1)).T",
    input_layout=(
        "Official 2Dpy reads a wavenumber-by-spectrum CSV and applies .T; this library "
        "receives the resulting (n_spectra, n_wavenumbers) internal layout."
    ),
    axis_order_behavior=(
        "Preserve the CSV wavenumber order.  2Dpy's left_large option only reverses plot "
        "limits and does not reorder exported matrix labels or values."
    ),
    source_url=TWODPY_SOURCE_URL,
    compatibility_notes=(
        "2Dpy sets rows to spectrum-2 variables and columns to spectrum-1 variables after "
        "the final transpose.  For cross data this is the transpose of the canonical "
        "spectrum-1-by-spectrum-2 matrix.  For homo data, Psi_2Dpy = Psi_canonical.T, "
        "which is also -Psi_canonical up to floating-point roundoff."
    ),
)


def normalize_convention(convention: MatrixConvention | str) -> MatrixConvention:
    """Normalize a public convention value with a readable validation error."""

    if isinstance(convention, MatrixConvention):
        return convention
    try:
        return MatrixConvention(str(convention))
    except ValueError as error:
        allowed = ", ".join(item.value for item in MatrixConvention)
        raise ValueError(
            f"Unknown 2D-COS convention {convention!r}; choose one of: {allowed}"
        ) from error


def get_convention_spec(convention: MatrixConvention | str) -> ConventionSpec:
    """Return the documented behavior for ``convention``."""

    normalized = normalize_convention(convention)
    if normalized is MatrixConvention.CANONICAL:
        return _CANONICAL_SPEC
    return _TWODPY_SPEC


def _as_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    """Validate one non-empty finite correlation matrix."""

    if np.iscomplexobj(values):
        raise TypeError(f"{name} must contain real values")
    try:
        matrix = np.array(values, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric matrix") from error
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty matrix; got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


def apply_matrix_convention(
    synchronous: ArrayLike,
    asynchronous: ArrayLike,
    convention: MatrixConvention | str,
) -> tuple[FloatArray, FloatArray]:
    """Apply only the documented final orientation step.

    Both inputs must be canonical matrices.  Copies are always returned, so a
    caller cannot mutate an intermediate canonical result through an output
    view.
    """

    sync = _as_matrix(synchronous, name="synchronous")
    async_ = _as_matrix(asynchronous, name="asynchronous")
    if sync.shape != async_.shape:
        raise ValueError(
            "synchronous and asynchronous matrices must have equal shapes; "
            f"got {sync.shape} and {async_.shape}"
        )

    normalized = normalize_convention(convention)
    if normalized is MatrixConvention.TWODPY_COMPATIBLE:
        return sync.T.copy(order="C"), async_.T.copy(order="C")
    return sync, async_


__all__ = [
    "TWODPY_RAW_SOURCE_URL",
    "TWODPY_REPOSITORY_URL",
    "TWODPY_SOURCE_URL",
    "ConventionSpec",
    "MatrixConvention",
    "apply_matrix_convention",
    "get_convention_spec",
    "normalize_convention",
]
