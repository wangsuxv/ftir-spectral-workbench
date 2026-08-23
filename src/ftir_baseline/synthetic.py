"""Reproducible synthetic FTIR series and quantitative benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .scoring import temporal_roughness

FloatArray = NDArray[np.float64]


SYNTHETIC_SCENARIOS: tuple[str, ...] = (
    "constant",
    "linear",
    "quadratic",
    "exponential",
    "broad_narrow",
    "broad_oh",
    "smooth_drift",
    "abrupt_jump",
    "noise_levels",
    "descending",
)


_SCENARIO_ALIASES = {
    "constant_baseline": "constant",
    "constant": "constant",
    "linear_slope": "linear",
    "linear": "linear",
    "quadratic_curvature": "quadratic",
    "quadratic": "quadratic",
    "exponential_background": "exponential",
    "exponential": "exponential",
    "wide_and_narrow_peaks": "broad_narrow",
    "broad_and_narrow": "broad_narrow",
    "broad_plus_narrow": "broad_narrow",
    "broad_narrow": "broad_narrow",
    "wide_oh": "broad_oh",
    "broad_oh": "broad_oh",
    "smooth_temporal_drift": "smooth_drift",
    "smooth_drift": "smooth_drift",
    "real_jump": "abrupt_jump",
    "abrupt_jump": "abrupt_jump",
    "varying_noise": "noise_levels",
    "noise_levels": "noise_levels",
    "low_noise": "low_noise",
    "high_noise": "high_noise",
    "ascending": "ascending",
    "ascending_axis": "ascending",
    "descending_axis": "descending",
    "descending": "descending",
}


@dataclass(frozen=True)
class SyntheticDataset:
    """Known-truth FTIR sequence used for scientific regression tests."""

    wavenumber: FloatArray
    perturbation: FloatArray
    true_signal: FloatArray
    true_baseline: FloatArray
    noise: FloatArray
    observed: FloatArray
    scenario: str
    peak_centers: FloatArray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def spectra(self) -> FloatArray:
        """Alias matching :class:`SpectrumSet` terminology."""

        return self.observed

    @property
    def corrected_truth(self) -> FloatArray:
        return self.true_signal


def gaussian_peak(x: ArrayLike, center: float, width: float, amplitude: float = 1.0) -> FloatArray:
    x_values = np.asarray(x, dtype=np.float64)
    width = float(width)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Gaussian width must be finite and positive")
    return np.asarray(
        float(amplitude) * np.exp(-0.5 * ((x_values - float(center)) / width) ** 2),
        dtype=np.float64,
    )


def lorentzian_peak(
    x: ArrayLike, center: float, width: float, amplitude: float = 1.0
) -> FloatArray:
    x_values = np.asarray(x, dtype=np.float64)
    width = float(width)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Lorentzian width must be finite and positive")
    return np.asarray(
        float(amplitude) * width**2 / ((x_values - float(center)) ** 2 + width**2),
        dtype=np.float64,
    )


def pseudo_voigt_peak(
    x: ArrayLike,
    center: float,
    width: float,
    amplitude: float = 1.0,
    fraction: float = 0.5,
) -> FloatArray:
    """Dependency-free pseudo-Voigt peak with a common peak height."""

    fraction = float(fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("pseudo-Voigt fraction must be between zero and one")
    return np.asarray(
        float(amplitude)
        * (
            fraction * lorentzian_peak(x, center, width, 1.0)
            + (1.0 - fraction) * gaussian_peak(x, center, width, 1.0)
        ),
        dtype=np.float64,
    )


voigt_peak = pseudo_voigt_peak


def _canonical_scenario(scenario: str) -> str:
    key = str(scenario).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _SCENARIO_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown synthetic scenario {scenario!r}; choose one of {SYNTHETIC_SCENARIOS}"
        ) from exc


def _chemical_signal(
    x: FloatArray,
    perturbation: FloatArray,
    scenario: str,
) -> tuple[FloatArray, FloatArray]:
    peak_centers = np.array([1080.0, 1260.0, 1640.0, 2920.0], dtype=np.float64)
    base = (
        gaussian_peak(x, peak_centers[0], 24.0, 0.34)
        + lorentzian_peak(x, peak_centers[1], 32.0, 0.24)
        + pseudo_voigt_peak(x, peak_centers[2], 45.0, 0.42, 0.35)
        + gaussian_peak(x, peak_centers[3], 38.0, 0.15)
    )
    if scenario == "broad_narrow":
        base = base + gaussian_peak(x, 2150.0, 330.0, 0.38)
        peak_centers = np.append(peak_centers, 2150.0)
    elif scenario == "broad_oh":
        # The width makes the O-H-like band occupy most of the upper interval,
        # intentionally challenging automatic baselines.
        base = base + pseudo_voigt_peak(x, 3300.0, 560.0, 0.62, 0.25)
        peak_centers = np.append(peak_centers, 3300.0)

    phase = 2.0 * np.pi * perturbation
    amplitudes = 1.0 + 0.14 * np.sin(phase) + 0.08 * perturbation
    signal = amplitudes[:, np.newaxis] * base[np.newaxis, :]
    # One band has its own kinetics, providing a non-trivial spectral series.
    signal += (
        0.10 * perturbation[:, np.newaxis] * gaussian_peak(x, 1450.0, 28.0, 1.0)[np.newaxis, :]
    )
    return np.asarray(signal, dtype=np.float64), peak_centers


def _baseline_series(
    z: FloatArray,
    perturbation: FloatArray,
    scenario: str,
) -> tuple[FloatArray, dict[str, Any]]:
    n_spectra = perturbation.size
    metadata: dict[str, Any] = {}
    if scenario == "constant":
        reference = np.full(z.size, 0.18, dtype=np.float64)
    elif scenario == "linear":
        reference = 0.08 + 0.20 * z
    elif scenario == "quadratic":
        reference = 0.07 + 0.06 * z + 0.22 * (z - 0.45) ** 2
    elif scenario == "exponential":
        reference = 0.045 + 0.16 * np.exp(3.2 * (z - 1.0))
    else:
        reference = 0.07 + 0.05 * z + 0.11 * (z - 0.5) ** 2

    baseline = np.repeat(reference[np.newaxis, :], n_spectra, axis=0)
    if scenario in {"smooth_drift", "noise_levels"}:
        smooth_offset = 0.055 * np.sin(np.pi * perturbation)
        smooth_slope = 0.035 * np.cos(2.0 * np.pi * perturbation)
        baseline += smooth_offset[:, np.newaxis] + smooth_slope[:, np.newaxis] * z
    elif scenario == "abrupt_jump":
        smooth_offset = 0.025 * np.sin(np.pi * perturbation)
        baseline += smooth_offset[:, np.newaxis]
        jump_index = n_spectra // 2
        baseline[jump_index:] += 0.12 + 0.035 * z
        metadata["expected_jump_index"] = jump_index
    return np.asarray(baseline, dtype=np.float64), metadata


def generate_synthetic_ftir(
    scenario: str = "quadratic",
    *,
    n_points: int = 1200,
    n_spectra: int = 12,
    wavenumber_range: tuple[float, float] = (900.0, 3800.0),
    noise_sigma: float = 0.003,
    impulse_fraction: float = 0.001,
    impulse_scale: float = 8.0,
    seed: int = 0,
    descending: bool | None = None,
) -> SyntheticDataset:
    """Generate an observed series with fully known signal, baseline, and noise.

    The same seed and arguments give element-for-element identical arrays.
    ``scenario='noise_levels'`` varies Gaussian noise monotonically across the
    series; ``scenario='descending'`` reverses every spectral array together.
    """

    canonical = _canonical_scenario(scenario)
    n_points = int(n_points)
    n_spectra = int(n_spectra)
    if n_points < 8:
        raise ValueError("n_points must be at least 8")
    if n_spectra < 1:
        raise ValueError("n_spectra must be at least 1")
    lo, hi = sorted((float(wavenumber_range[0]), float(wavenumber_range[1])))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        raise ValueError("wavenumber_range bounds must be finite and distinct")
    sigma = float(noise_sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("noise_sigma must be finite and non-negative")
    impulse_fraction = float(impulse_fraction)
    if not 0.0 <= impulse_fraction <= 1.0:
        raise ValueError("impulse_fraction must be between zero and one")
    impulse_scale = float(impulse_scale)
    if not np.isfinite(impulse_scale) or impulse_scale < 0:
        raise ValueError("impulse_scale must be finite and non-negative")

    x = np.linspace(lo, hi, n_points, dtype=np.float64)
    perturbation = (
        np.linspace(0.0, 1.0, n_spectra, dtype=np.float64)
        if n_spectra > 1
        else np.array([0.0], dtype=np.float64)
    )
    z = (x - lo) / (hi - lo)
    signal_scenario = canonical if canonical in {"broad_narrow", "broad_oh"} else "standard"
    true_signal, peak_centers = _chemical_signal(x, perturbation, signal_scenario)
    true_baseline, baseline_metadata = _baseline_series(z, perturbation, canonical)

    generator = np.random.default_rng(int(seed))
    if canonical == "noise_levels":
        row_sigma = np.linspace(0.25 * sigma, 3.0 * sigma, n_spectra, dtype=np.float64)
    elif canonical == "low_noise":
        row_sigma = np.full(n_spectra, 0.25 * sigma, dtype=np.float64)
    elif canonical == "high_noise":
        row_sigma = np.full(n_spectra, 3.0 * sigma, dtype=np.float64)
    else:
        row_sigma = np.full(n_spectra, sigma, dtype=np.float64)
    noise = generator.normal(size=(n_spectra, n_points)) * row_sigma[:, np.newaxis]
    impulse_count = round(impulse_fraction * n_spectra * n_points)
    if impulse_count and sigma > 0 and impulse_scale > 0:
        flat_indices = generator.choice(n_spectra * n_points, size=impulse_count, replace=False)
        impulses = generator.normal(0.0, impulse_scale * sigma, size=impulse_count)
        noise.reshape(-1)[flat_indices] += impulses

    should_descend = canonical == "descending" if descending is None else bool(descending)
    observed = true_signal + true_baseline + noise
    if should_descend:
        x = x[::-1].copy()
        true_signal = true_signal[:, ::-1].copy()
        true_baseline = true_baseline[:, ::-1].copy()
        noise = noise[:, ::-1].copy()
        observed = observed[:, ::-1].copy()

    metadata: dict[str, Any] = {
        "seed": int(seed),
        "noise_sigma": sigma,
        "noise_sigma_by_spectrum": row_sigma.tolist(),
        "impulse_fraction": impulse_fraction,
        "axis_direction": "descending" if should_descend else "ascending",
        "generation_equation": "observed = true_signal + true_baseline + noise",
        **baseline_metadata,
    }
    return SyntheticDataset(
        wavenumber=np.asarray(x, dtype=np.float64),
        perturbation=perturbation,
        true_signal=np.asarray(true_signal, dtype=np.float64),
        true_baseline=np.asarray(true_baseline, dtype=np.float64),
        noise=np.asarray(noise, dtype=np.float64),
        observed=np.asarray(observed, dtype=np.float64),
        scenario=canonical,
        peak_centers=np.asarray(peak_centers, dtype=np.float64),
        metadata=metadata,
    )


generate_synthetic_series = generate_synthetic_ftir


def generate_synthetic_suite(
    *,
    seed: int = 0,
    scenarios: Sequence[str] = SYNTHETIC_SCENARIOS,
    **kwargs: Any,
) -> dict[str, SyntheticDataset]:
    """Generate all requested validation scenarios with stable independent seeds."""

    return {
        _canonical_scenario(name): generate_synthetic_ftir(name, seed=int(seed) + index, **kwargs)
        for index, name in enumerate(scenarios)
    }


def _trapezoid_rows(x: FloatArray, values: FloatArray) -> FloatArray:
    order = np.argsort(x)
    ordered = values[:, order]
    widths = np.diff(x[order])
    return np.asarray(
        np.sum(0.5 * (ordered[:, :-1] + ordered[:, 1:]) * widths, axis=1),
        dtype=np.float64,
    )


def calculate_benchmark_metrics(
    dataset: SyntheticDataset,
    estimated_baseline: ArrayLike,
    corrected: ArrayLike | None = None,
) -> dict[str, Any]:
    """Measure baseline recovery and chemical peak fidelity against known truth."""

    baseline = np.asarray(estimated_baseline, dtype=np.float64)
    if baseline.ndim == 1 and dataset.observed.shape[0] == 1:
        baseline = baseline[np.newaxis, :]
    if baseline.shape != dataset.true_baseline.shape:
        raise ValueError(
            "estimated_baseline must have shape "
            f"{dataset.true_baseline.shape}, got {baseline.shape}"
        )
    if not np.all(np.isfinite(baseline)):
        raise ValueError("estimated_baseline contains NaN or Inf")
    if corrected is None:
        corrected_values = dataset.observed - baseline
    else:
        corrected_values = np.asarray(corrected, dtype=np.float64)
        if corrected_values.ndim == 1 and dataset.observed.shape[0] == 1:
            corrected_values = corrected_values[np.newaxis, :]
        if corrected_values.shape != dataset.true_signal.shape:
            raise ValueError(
                f"corrected must have shape {dataset.true_signal.shape}, got {corrected_values.shape}"
            )
        if not np.all(np.isfinite(corrected_values)):
            raise ValueError("corrected contains NaN or Inf")

    baseline_rmse_rows = np.sqrt(np.mean((baseline - dataset.true_baseline) ** 2, axis=1))
    corrected_rmse_rows = np.sqrt(np.mean((corrected_values - dataset.true_signal) ** 2, axis=1))
    truth_height = np.max(dataset.true_signal, axis=1)
    corrected_height = np.max(corrected_values, axis=1)
    height_bias_rows = corrected_height - truth_height
    height_relative_rows = height_bias_rows / np.maximum(
        np.abs(truth_height), np.finfo(np.float64).eps
    )
    truth_area = _trapezoid_rows(dataset.wavenumber, dataset.true_signal)
    corrected_area = _trapezoid_rows(dataset.wavenumber, corrected_values)
    area_bias_rows = corrected_area - truth_area
    area_relative_rows = area_bias_rows / np.maximum(np.abs(truth_area), np.finfo(np.float64).eps)
    truth_positions = dataset.wavenumber[np.argmax(dataset.true_signal, axis=1)]
    corrected_positions = dataset.wavenumber[np.argmax(corrected_values, axis=1)]
    position_shift_rows = np.abs(corrected_positions - truth_positions)

    return {
        "scenario": dataset.scenario,
        "baseline_rmse": float(np.sqrt(np.mean((baseline - dataset.true_baseline) ** 2))),
        "corrected_spectrum_rmse": float(
            np.sqrt(np.mean((corrected_values - dataset.true_signal) ** 2))
        ),
        "peak_height_bias": float(np.mean(height_bias_rows)),
        "peak_height_relative_bias": float(np.mean(height_relative_rows)),
        "peak_area_bias": float(np.mean(area_bias_rows)),
        "peak_area_relative_bias": float(np.mean(area_relative_rows)),
        "peak_position_shift": float(np.mean(position_shift_rows)),
        "series_temporal_roughness": temporal_roughness(baseline),
        "true_series_temporal_roughness": temporal_roughness(dataset.true_baseline),
        "baseline_rmse_per_spectrum": baseline_rmse_rows,
        "corrected_spectrum_rmse_per_spectrum": corrected_rmse_rows,
        "peak_height_bias_per_spectrum": height_bias_rows,
        "peak_area_bias_per_spectrum": area_bias_rows,
        "peak_position_shift_per_spectrum": position_shift_rows,
        "reconstruction_max_error": float(
            np.max(np.abs(dataset.observed - baseline - corrected_values))
        ),
    }


benchmark_baseline = calculate_benchmark_metrics


def _extract_estimator_output(output: Any, observed: FloatArray) -> tuple[FloatArray, FloatArray]:
    if hasattr(output, "total_baseline"):
        baseline = np.asarray(output.total_baseline, dtype=np.float64)
        corrected = np.asarray(getattr(output, "corrected", observed - baseline), dtype=np.float64)
    elif isinstance(output, Mapping):
        baseline_value = output.get("total_baseline", output.get("baseline"))
        if baseline_value is None:
            raise ValueError("estimator mapping output must contain total_baseline or baseline")
        baseline = np.asarray(baseline_value, dtype=np.float64)
        corrected = np.asarray(output.get("corrected", observed - baseline), dtype=np.float64)
    elif isinstance(output, tuple) and len(output) == 2:
        baseline = np.asarray(output[0], dtype=np.float64)
        corrected = np.asarray(output[1], dtype=np.float64)
    else:
        baseline = np.asarray(output, dtype=np.float64)
        corrected = observed - baseline
    return baseline, corrected


def run_synthetic_benchmarks(
    estimator: Any,
    *,
    scenarios: Sequence[str] = SYNTHETIC_SCENARIOS,
    seed: int = 0,
    generator_kwargs: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Run an estimator over the reproducible scenario suite.

    The estimator may expose ``fit_transform(x, spectra)`` or be a callable
    with the same arguments.  Outputs may be a BaselineResult, mapping,
    ``(baseline, corrected)`` tuple, or the baseline array itself.
    """

    kwargs = dict(generator_kwargs or {})
    records: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        dataset = generate_synthetic_ftir(scenario, seed=int(seed) + index, **kwargs)
        if hasattr(estimator, "fit_transform"):
            output = estimator.fit_transform(dataset.wavenumber, dataset.observed)
        elif callable(estimator):
            output = estimator(dataset.wavenumber, dataset.observed)
        else:
            raise TypeError("estimator must be callable or expose fit_transform")
        baseline, corrected = _extract_estimator_output(output, dataset.observed)
        records.append(calculate_benchmark_metrics(dataset, baseline, corrected))
    return tuple(records)


run_benchmarks = run_synthetic_benchmarks


__all__ = [
    "SYNTHETIC_SCENARIOS",
    "SyntheticDataset",
    "benchmark_baseline",
    "calculate_benchmark_metrics",
    "gaussian_peak",
    "generate_synthetic_ftir",
    "generate_synthetic_series",
    "generate_synthetic_suite",
    "lorentzian_peak",
    "pseudo_voigt_peak",
    "run_benchmarks",
    "run_synthetic_benchmarks",
    "voigt_peak",
]
