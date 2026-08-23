from __future__ import annotations

import numpy as np
import pytest

from ftir2dcos.config import BaselineConfig
from ftir2dcos.models import SpectralDataset
from ftir2dcos.preprocessing.baseline import correct_baseline, estimate_baseline


def _synthetic_dataset() -> tuple[SpectralDataset, np.ndarray]:
    x = np.linspace(900.0, 1800.0, 301)
    known_baseline = 0.04 + 2e-5 * (x - 1300.0) + 2e-8 * (x - 1300.0) ** 2
    peak = 0.25 * np.exp(-0.5 * ((x - 1300.0) / 30.0) ** 2)
    spectra = np.vstack([known_baseline + scale * peak for scale in (0.8, 1.0, 1.2, 1.4, 1.6)])
    dataset = SpectralDataset(
        wavenumber=x,
        perturbation=np.arange(5, dtype=float),
        perturbation_labels=("0", "1", "2", "3", "4"),
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="synthetic",
        metadata={},
    )
    return dataset, known_baseline


def test_none_baseline_is_exact_copy_and_does_not_modify_input() -> None:
    dataset, _ = _synthetic_dataset()
    original = dataset.spectra.copy()

    corrected, baselines, diagnostics = correct_baseline(dataset, BaselineConfig())

    np.testing.assert_array_equal(corrected.spectra, original)
    np.testing.assert_array_equal(baselines, np.zeros_like(original))
    np.testing.assert_array_equal(dataset.spectra, original)
    assert corrected.spectra is not dataset.spectra
    assert len(diagnostics) == dataset.n_spectra


def test_offset_supports_minimum_and_window_median() -> None:
    dataset, _ = _synthetic_dataset()
    minimum_result = estimate_baseline(
        dataset.wavenumber, dataset.spectra[0], BaselineConfig(method="offset")
    )
    window_result = estimate_baseline(
        dataset.wavenumber,
        dataset.spectra[0],
        BaselineConfig(
            method="constant",
            offset_mode="window_median",
            offset_window=(900.0, 950.0),
        ),
    )

    assert np.ptp(minimum_result.baseline) == 0.0
    assert np.ptp(window_result.baseline) == 0.0
    assert window_result.diagnostics["offset_source"] == "window_median"
    with pytest.raises(ValueError, match="requires an explicit offset_window"):
        estimate_baseline(
            dataset.wavenumber,
            dataset.spectra[0],
            BaselineConfig(method="offset", offset_mode="window_median"),
        )


def test_anchor_polynomial_recovers_known_quadratic_baseline() -> None:
    dataset, known_baseline = _synthetic_dataset()
    config = BaselineConfig(
        method="anchor_polynomial",
        anchor_ranges=((900.0, 1050.0), (1550.0, 1800.0)),
        polynomial_order=2,
    )

    result = estimate_baseline(dataset.wavenumber, dataset.spectra[0], config)

    # Gaussian tails are negligible in the selected anchor regions.
    np.testing.assert_allclose(result.baseline, known_baseline, rtol=0, atol=2e-10)
    assert result.diagnostics["anchor_point_count"] > 3
    assert len(result.diagnostics["anchor_mask"]) == dataset.n_wavenumbers


@pytest.mark.parametrize(
    "config",
    [
        BaselineConfig(method="asls", asls_lam=1e6, asls_p=0.01),
        BaselineConfig(method="rubberband", rubberband_segments=2),
    ],
)
def test_pybaselines_methods_are_finite_and_stable(config: BaselineConfig) -> None:
    dataset, _ = _synthetic_dataset()
    original = dataset.spectra.copy()

    corrected, baselines, diagnostics = correct_baseline(dataset, config)

    assert corrected.shape == dataset.shape
    assert baselines.shape == dataset.shape
    assert baselines.dtype == np.float64
    assert np.all(np.isfinite(corrected.spectra))
    assert np.all(np.isfinite(baselines))
    np.testing.assert_array_equal(dataset.spectra, original)
    assert all(item["method"] == config.method for item in diagnostics)
    assert corrected.metadata["baseline_same_parameters_for_all_spectra"] is True


def test_baseline_rejects_nonfinite_and_bad_anchor_intervals_readably() -> None:
    dataset, _ = _synthetic_dataset()
    bad = dataset.spectra[0].copy()
    bad[10] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        estimate_baseline(dataset.wavenumber, bad, BaselineConfig(method="asls"))
    with pytest.raises(ValueError, match="contains no wavenumber points"):
        estimate_baseline(
            dataset.wavenumber,
            dataset.spectra[0],
            BaselineConfig(
                method="anchor_polynomial",
                anchor_ranges=((100.0, 200.0),),
                polynomial_order=1,
            ),
        )
    with pytest.raises(ValueError, match="asls_p"):
        BaselineConfig(method="asls", asls_p=1.0)
