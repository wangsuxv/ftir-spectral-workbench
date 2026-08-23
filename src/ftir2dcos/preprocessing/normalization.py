"""Explicit, default-off normalization methods."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..config import NormalizationConfig
from ..models import SpectralDataset


def normalize_spectra(
    spectra: ArrayLike,
    config: NormalizationConfig,
    *,
    wavenumber: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalize an ``(m, n)`` matrix and return the applied row factors."""

    if not isinstance(config, NormalizationConfig):
        config = NormalizationConfig.from_dict(config)
    array = np.array(spectra, dtype=np.float64, copy=True, order="C")
    if array.ndim != 2:
        raise ValueError(f"spectra must be two-dimensional; got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Spectra contain NaN or Inf; normalization was not performed")
    if config.method == "none":
        return array, np.ones(array.shape[0], dtype=np.float64)
    if config.method == "vector":
        factors = np.linalg.norm(array, axis=1)
    else:
        if wavenumber is None:
            raise ValueError("reference_peak normalization requires a wavenumber axis")
        x = np.asarray(wavenumber, dtype=np.float64)
        if x.shape != (array.shape[1],):
            raise ValueError(f"wavenumber shape must be {(array.shape[1],)}; got {x.shape}")
        assert config.reference_peak_range is not None
        lower, upper = sorted(config.reference_peak_range)
        mask = (x >= lower) & (x <= upper)
        if not np.any(mask):
            raise ValueError(
                f"reference_peak_range [{lower}, {upper}] contains no wavenumber points"
            )
        # Absolute peak height makes the operation defined for spectra with a
        # negative-going reference band while still preserving spectral sign.
        factors = np.max(np.abs(array[:, mask]), axis=1)
    invalid = np.flatnonzero(~np.isfinite(factors) | np.isclose(factors, 0.0, atol=0.0))
    if invalid.size:
        raise ValueError(
            f"Normalization factor is zero or non-finite for spectra {invalid.tolist()}"
        )
    return np.asarray(array / factors[:, None], dtype=np.float64), np.asarray(
        factors, dtype=np.float64
    )


def normalize_dataset(
    dataset: SpectralDataset,
    config: NormalizationConfig,
) -> SpectralDataset:
    """Apply an explicitly selected normalization without modifying input."""

    if not isinstance(config, NormalizationConfig):
        config = NormalizationConfig.from_dict(config)
    normalized, factors = normalize_spectra(
        dataset.spectra,
        config,
        wavenumber=dataset.wavenumber,
    )
    metadata = deepcopy(dict(dataset.metadata))
    history = list(metadata.get("processing_history", []))
    history.append(
        {
            "operation": "normalize_spectra",
            "method": config.method,
            "parameters": config.to_dict(),
            "factors": factors.astype(float).tolist(),
        }
    )
    metadata.update(
        {
            "normalization": config.to_dict(),
            "normalization_applied": config.method != "none",
            "normalization_factors": factors.astype(float).tolist(),
            "processing_history": history,
        }
    )
    return dataset.with_updates(spectra=normalized, metadata=metadata)


apply_normalization = normalize_dataset


__all__ = ["apply_normalization", "normalize_dataset", "normalize_spectra"]
