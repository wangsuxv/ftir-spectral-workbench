"""Hilbert--Noda transformation matrix construction."""

from __future__ import annotations

import operator

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def hilbert_noda_matrix(n_spectra: int) -> FloatArray:
    """Build the classic index-order Hilbert--Noda matrix in ``float64``.

    The zero-based implementation is algebraically identical to the usual
    one-based definition::

        N[j, k] = 0                       if j == k
        N[j, k] = 1 / (pi * (k - j))      otherwise

    Perturbation values and their spacing deliberately do not enter this
    first-version calculation.  The caller is responsible for validating and
    recording any non-uniform perturbation grid.
    """

    if isinstance(n_spectra, (bool, np.bool_)):
        raise TypeError("n_spectra must be an integer, not bool")
    try:
        count = operator.index(n_spectra)
    except TypeError as error:
        raise TypeError("n_spectra must be an integer") from error
    if count < 1:
        raise ValueError("n_spectra must be at least 1")

    indices = np.arange(count, dtype=np.float64)
    delta = indices[None, :] - indices[:, None]
    noda = np.zeros((count, count), dtype=np.float64)
    off_diagonal = delta != 0.0

    # The order of operations mirrors official 2Dpy's ``1 / math.pi / (j-i)``.
    noda[off_diagonal] = (1.0 / np.pi) / delta[off_diagonal]
    return noda


# Concise alias retained for callers that already use the specification's name.
noda_matrix = hilbert_noda_matrix


__all__ = ["hilbert_noda_matrix", "noda_matrix"]
