from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import savgol_filter

from ftir2dcos.config import NormalizationConfig, SmoothingConfig
from ftir2dcos.models import SpectralDataset
from ftir2dcos.preprocessing.normalization import normalize_dataset
from ftir2dcos.preprocessing.smoothing import smooth_dataset, smooth_spectra


def _dataset() -> SpectralDataset:
    rng = np.random.default_rng(42)
    x = np.linspace(900.0, 1800.0, 31)
    spectra = rng.normal(size=(5, x.size))
    return SpectralDataset(
        wavenumber=x,
        perturbation=np.arange(5, dtype=float),
        perturbation_labels=("0", "1", "2", "3", "4"),
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="synthetic",
        metadata={},
    )


def test_smoothing_is_off_by_default_and_returns_independent_float64_data() -> None:
    dataset = _dataset()
    original = dataset.spectra.copy()

    output = smooth_dataset(dataset, SmoothingConfig())

    np.testing.assert_array_equal(output.spectra, original)
    np.testing.assert_array_equal(dataset.spectra, original)
    assert output.spectra.dtype == np.float64
    assert output.metadata["smoothing_applied"] is False


def test_savgol_runs_only_along_wavenumber_axis() -> None:
    dataset = _dataset()
    config = SmoothingConfig(enabled=True, window_length=7, polyorder=2, mode="interp")

    actual = smooth_spectra(dataset.spectra, config)
    expected = np.vstack([savgol_filter(row, 7, 2, mode="interp") for row in dataset.spectra])

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(dataset.spectra, _dataset().spectra)


@pytest.mark.parametrize(
    "config, message",
    [
        (SmoothingConfig(enabled=True, window_length=6), "must be odd"),
        (SmoothingConfig(enabled=True, window_length=3, polyorder=3), "greater than polyorder"),
        (SmoothingConfig(enabled=True, window_length=33), "exceeds n_wavenumbers"),
    ],
)
def test_savgol_parameter_validation(config: SmoothingConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        smooth_spectra(_dataset().spectra, config)


def test_normalization_defaults_to_none_and_vector_is_explicit() -> None:
    dataset = _dataset()
    unchanged = normalize_dataset(dataset, NormalizationConfig())
    normalized = normalize_dataset(dataset, NormalizationConfig(method="vector"))

    np.testing.assert_array_equal(unchanged.spectra, dataset.spectra)
    np.testing.assert_allclose(np.linalg.norm(normalized.spectra, axis=1), 1.0)
    assert unchanged.metadata["normalization_applied"] is False
    assert normalized.metadata["normalization_applied"] is True


def test_reference_peak_normalization_uses_same_configured_window() -> None:
    dataset = _dataset()
    config = NormalizationConfig(method="reference_peak", reference_peak_range=(1100, 1300))
    output = normalize_dataset(dataset, config)
    mask = (dataset.wavenumber >= 1100) & (dataset.wavenumber <= 1300)

    np.testing.assert_allclose(np.max(np.abs(output.spectra[:, mask]), axis=1), 1.0)
    assert output.metadata["normalization"]["reference_peak_range"] == [1100.0, 1300.0]
