from __future__ import annotations

import numpy as np
import pytest

from ftir_baseline.synthetic import (
    SYNTHETIC_SCENARIOS,
    calculate_benchmark_metrics,
    generate_synthetic_ftir,
    generate_synthetic_suite,
    run_synthetic_benchmarks,
)


def test_generator_is_reproducible_and_preserves_generation_equation() -> None:
    first = generate_synthetic_ftir("quadratic", n_points=128, n_spectra=5, seed=42)
    second = generate_synthetic_ftir("quadratic", n_points=128, n_spectra=5, seed=42)

    for name in ("wavenumber", "perturbation", "true_signal", "true_baseline", "noise", "observed"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    np.testing.assert_allclose(
        first.observed, first.true_signal + first.true_baseline + first.noise, atol=0.0
    )


def test_suite_covers_all_required_scenarios() -> None:
    suite = generate_synthetic_suite(seed=1, n_points=64, n_spectra=5)
    assert set(suite) == set(SYNTHETIC_SCENARIOS)
    for dataset in suite.values():
        assert dataset.observed.shape == (5, 64)
        assert dataset.observed.dtype == np.float64


def test_descending_and_ascending_are_equivalent_reversals() -> None:
    ascending = generate_synthetic_ftir(
        "ascending", n_points=96, n_spectra=3, seed=7, descending=False
    )
    descending = generate_synthetic_ftir("descending", n_points=96, n_spectra=3, seed=7)
    np.testing.assert_array_equal(ascending.wavenumber, descending.wavenumber[::-1])
    np.testing.assert_array_equal(ascending.observed, descending.observed[:, ::-1])
    assert np.all(np.diff(descending.wavenumber) < 0)


def test_broad_oh_really_occupies_a_large_fraction_of_interval() -> None:
    dataset = generate_synthetic_ftir(
        "broad_oh", n_points=1000, n_spectra=1, noise_sigma=0.0, impulse_fraction=0.0
    )
    signal = dataset.true_signal[0]
    occupied = np.mean(signal > 0.1 * np.max(signal))
    assert occupied > 0.30


def test_temporal_drift_and_real_jump_are_distinguishable() -> None:
    smooth = generate_synthetic_ftir("smooth_drift", n_points=128, n_spectra=12, noise_sigma=0.0)
    jump = generate_synthetic_ftir("abrupt_jump", n_points=128, n_spectra=12, noise_sigma=0.0)
    smooth_second = np.mean(np.diff(smooth.true_baseline, n=2, axis=0) ** 2)
    jump_second = np.mean(np.diff(jump.true_baseline, n=2, axis=0) ** 2)
    assert jump_second > 10.0 * smooth_second
    assert jump.metadata["expected_jump_index"] == 6


def test_oracle_baseline_benchmark_has_zero_baseline_error_and_noise_limited_correction() -> None:
    dataset = generate_synthetic_ftir(
        "exponential", n_points=256, n_spectra=4, noise_sigma=0.002, seed=11
    )
    metrics = calculate_benchmark_metrics(dataset, dataset.true_baseline)
    assert metrics["baseline_rmse"] == pytest.approx(0.0)
    assert metrics["corrected_spectrum_rmse"] == pytest.approx(
        float(np.sqrt(np.mean(dataset.noise**2)))
    )
    assert metrics["reconstruction_max_error"] < 1e-14
    for key in (
        "peak_height_bias",
        "peak_area_bias",
        "peak_position_shift",
        "series_temporal_roughness",
    ):
        assert key in metrics


def test_benchmark_runner_accepts_array_estimator() -> None:
    def zero_baseline(_x: np.ndarray, spectra: np.ndarray) -> np.ndarray:
        return np.zeros_like(spectra)

    records = run_synthetic_benchmarks(
        zero_baseline,
        scenarios=("constant", "linear"),
        seed=3,
        generator_kwargs={"n_points": 64, "n_spectra": 3},
    )
    assert [record["scenario"] for record in records] == ["constant", "linear"]
    assert all(record["baseline_rmse"] > 0 for record in records)
