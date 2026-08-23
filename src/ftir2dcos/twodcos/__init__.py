"""Public homo and cross-region 2D-COS scientific API."""

from .conventions import (
    TWODPY_RAW_SOURCE_URL,
    TWODPY_REPOSITORY_URL,
    TWODPY_SOURCE_URL,
    ConventionSpec,
    MatrixConvention,
    apply_matrix_convention,
    get_convention_spec,
    normalize_convention,
)
from .engine import (
    DEFAULT_ABSOLUTE_TOLERANCE,
    DEFAULT_RELATIVE_TOLERANCE,
    CrossTwoDCOSResult,
    TwoDCOSResult,
    calculate_2dcos,
    compute_2d_correlation,
    compute_2dcos,
    compute_asynchronous,
    compute_cross_2dcos,
    compute_dynamic_spectra,
    compute_qc_metrics,
    compute_synchronous,
)
from .noda import hilbert_noda_matrix, noda_matrix

__all__ = [
    "DEFAULT_ABSOLUTE_TOLERANCE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "TWODPY_RAW_SOURCE_URL",
    "TWODPY_REPOSITORY_URL",
    "TWODPY_SOURCE_URL",
    "ConventionSpec",
    "CrossTwoDCOSResult",
    "MatrixConvention",
    "TwoDCOSResult",
    "apply_matrix_convention",
    "calculate_2dcos",
    "compute_2d_correlation",
    "compute_2dcos",
    "compute_asynchronous",
    "compute_cross_2dcos",
    "compute_dynamic_spectra",
    "compute_qc_metrics",
    "compute_synchronous",
    "get_convention_spec",
    "hilbert_noda_matrix",
    "noda_matrix",
    "normalize_convention",
]
