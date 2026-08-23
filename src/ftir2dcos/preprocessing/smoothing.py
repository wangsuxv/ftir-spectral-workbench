"""Optional Savitzky-Golay smoothing along the wavenumber dimension."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_filter

from ..config import SmoothingConfig
from ..models import SpectralDataset


def _validate_savgol(config: SmoothingConfig, n_points: int) -> None:
    if config.window_length <= 0:
        raise ValueError("Savitzky-Golay window_length must be positive")
    if config.window_length % 2 == 0:
        raise ValueError("Savitzky-Golay window_length must be odd")
    if config.polyorder < 0:
        raise ValueError("Savitzky-Golay polyorder must be non-negative")
    if config.window_length <= config.polyorder:
        raise ValueError("Savitzky-Golay window_length must be greater than polyorder")
    if config.window_length > int(n_points):
        raise ValueError(
            f"Savitzky-Golay window_length={config.window_length} exceeds n_wavenumbers={n_points}"
        )


def smooth_spectra(
    spectra: ArrayLike,
    config: SmoothingConfig,
) -> NDArray[np.float64]:
    """Smooth an ``(m, n)`` matrix along axis 1; disabled means a clean copy."""

    if not isinstance(config, SmoothingConfig):
        config = SmoothingConfig.from_dict(config)
    array = np.array(spectra, dtype=np.float64, copy=True, order="C")
    if array.ndim != 2:
        raise ValueError(f"spectra must be two-dimensional; got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        locations = np.argwhere(~np.isfinite(array))
        raise ValueError(
            f"Spectra contain NaN or Inf at positions {locations[:10].tolist()}; "
            "smoothing was not performed"
        )
    if not config.enabled:
        return array
    _validate_savgol(config, array.shape[1])
    output = savgol_filter(
        array,
        window_length=config.window_length,
        polyorder=config.polyorder,
        axis=1,
        mode=config.mode,
    )
    return np.asarray(output, dtype=np.float64)


def smooth_dataset(
    dataset: SpectralDataset,
    config: SmoothingConfig,
) -> SpectralDataset:
    """Return final processed spectra and record whether smoothing was enabled."""

    if not isinstance(config, SmoothingConfig):
        config = SmoothingConfig.from_dict(config)
    smoothed = smooth_spectra(dataset.spectra, config)
    metadata = deepcopy(dict(dataset.metadata))
    history = list(metadata.get("processing_history", []))
    history.append(
        {
            "operation": "savitzky_golay_smoothing",
            "enabled": config.enabled,
            "parameters": config.to_dict(),
            "axis": "wavenumber",
            "axis_index": 1,
        }
    )
    metadata.update(
        {
            "smoothing": config.to_dict(),
            "smoothing_applied": config.enabled,
            "smoothing_axis": "wavenumber",
            "processing_history": history,
        }
    )
    return dataset.with_updates(spectra=smoothed, metadata=metadata)


apply_smoothing = smooth_dataset


__all__ = ["apply_smoothing", "smooth_dataset", "smooth_spectra"]
