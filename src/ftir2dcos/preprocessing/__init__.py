"""Reproducible preprocessing operations for spectral datasets."""

from .baseline import correct_baseline, correct_spectrum_baseline, estimate_baseline
from .normalization import apply_normalization, normalize_dataset, normalize_spectra
from .smoothing import apply_smoothing, smooth_dataset, smooth_spectra

__all__ = [
    "apply_normalization",
    "apply_smoothing",
    "correct_baseline",
    "correct_spectrum_baseline",
    "estimate_baseline",
    "normalize_dataset",
    "normalize_spectra",
    "smooth_dataset",
    "smooth_spectra",
]
