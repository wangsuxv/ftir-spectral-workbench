"""Baseline estimation and correction with one configuration for all spectra."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..config import BaselineConfig
from ..models import BaselineResult, SpectralDataset


def _finite_1d(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    locations = np.flatnonzero(~np.isfinite(array))
    if locations.size:
        raise ValueError(
            f"{name} contains NaN or Inf at indices {locations[:10].tolist()}; "
            "baseline correction was not attempted"
        )
    return array


def _range_mask(x: np.ndarray, interval: tuple[float, float]) -> np.ndarray:
    lower, upper = min(interval), max(interval)
    return (x >= lower) & (x <= upper)


def _trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(x)))


def _common_diagnostics(
    baseline: np.ndarray,
    corrected: np.ndarray,
    x: np.ndarray,
) -> dict[str, Any]:
    warnings: list[str] = []
    corrected_minimum = float(np.min(corrected))
    if corrected_minimum < 0:
        warnings.append(
            "Corrected spectrum contains negative absorbance; values were retained and not clipped."
        )
    return {
        "baseline_peak_to_peak": float(np.ptp(baseline)),
        "baseline_area": _trapezoid_area(baseline, x),
        "corrected_minimum": corrected_minimum,
        "corrected_maximum": float(np.max(corrected)),
        "warnings": warnings,
    }


def estimate_baseline(
    wavenumber: ArrayLike,
    spectrum: ArrayLike,
    config: BaselineConfig,
) -> BaselineResult:
    """Estimate and subtract one baseline using the configured fixed algorithm."""

    if not isinstance(config, BaselineConfig):
        config = BaselineConfig.from_dict(config)
    x = _finite_1d(wavenumber, name="wavenumber")
    y = _finite_1d(spectrum, name="spectrum")
    if x.shape != y.shape:
        raise ValueError(f"wavenumber and spectrum shapes differ: {x.shape} versus {y.shape}")
    if x.size < 2:
        raise ValueError("At least two points are required for baseline correction")
    differences = np.diff(x)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError(
            "Wavenumber must be finite, unique, and strictly monotonic for baseline correction"
        )

    method_diagnostics: dict[str, Any] = {}
    if config.method == "none":
        baseline = np.zeros_like(y, dtype=np.float64)
    elif config.method == "offset":
        if config.offset_mode == "minimum":
            offset = float(np.min(y))
            method_diagnostics["offset_source"] = "full_range_minimum"
            method_diagnostics["minimum_index"] = int(np.argmin(y))
        else:
            if config.offset_window is None:
                raise ValueError("offset_mode='window_median' requires an explicit offset_window")
            mask = _range_mask(x, config.offset_window)
            if not np.any(mask):
                lower, upper = sorted(config.offset_window)
                raise ValueError(f"Offset window [{lower}, {upper}] contains no wavenumber points")
            offset = float(np.median(y[mask]))
            method_diagnostics.update(
                {
                    "offset_source": "window_median",
                    "offset_window": list(config.offset_window),
                    "offset_window_points": int(np.count_nonzero(mask)),
                }
            )
        baseline = np.full_like(y, offset, dtype=np.float64)
        method_diagnostics["offset_value"] = offset
    elif config.method == "anchor_polynomial":
        if not config.anchor_ranges:
            raise ValueError("anchor_polynomial requires at least one anchor range")
        anchor_mask = np.zeros(x.shape, dtype=bool)
        interval_counts: list[int] = []
        for interval in config.anchor_ranges:
            interval_mask = _range_mask(x, interval)
            count = int(np.count_nonzero(interval_mask))
            interval_counts.append(count)
            if count == 0:
                lower, upper = sorted(interval)
                raise ValueError(
                    f"Anchor interval [{lower}, {upper}] contains no wavenumber points"
                )
            anchor_mask |= interval_mask
        anchor_count = int(np.count_nonzero(anchor_mask))
        required = config.polynomial_order + 1
        if anchor_count < required:
            raise ValueError(
                f"Polynomial order {config.polynomial_order} needs at least {required} anchor "
                f"points; got {anchor_count}"
            )
        if np.unique(x[anchor_mask]).size < required:
            raise ValueError("Anchor points do not contain enough unique wavenumbers")
        try:
            fitted = np.polynomial.Polynomial.fit(
                x[anchor_mask], y[anchor_mask], deg=config.polynomial_order
            ).convert()
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            raise ValueError(f"Anchor polynomial fit failed: {exc}") from exc
        baseline = np.asarray(fitted(x), dtype=np.float64)
        method_diagnostics.update(
            {
                "anchor_ranges": [list(interval) for interval in config.anchor_ranges],
                "anchor_interval_point_counts": interval_counts,
                "anchor_point_count": anchor_count,
                "anchor_indices": np.flatnonzero(anchor_mask).tolist(),
                "anchor_mask": anchor_mask.tolist(),
                "polynomial_order": config.polynomial_order,
                "polynomial_coefficients_ascending_power": fitted.coef.astype(float).tolist(),
            }
        )
    elif config.method in {"asls", "rubberband"}:
        try:
            from pybaselines import Baseline
        except ImportError as exc:  # pragma: no cover - exercised only in partial installs
            raise RuntimeError(
                f"Baseline method {config.method!r} requires the 'pybaselines' package"
            ) from exc
        fitter = Baseline(
            x_data=x,
            check_finite=True,
            assume_sorted=False,
            output_dtype=np.float64,
        )
        if config.method == "asls":
            if x.size <= config.asls_diff_order:
                raise ValueError(
                    f"AsLS diff_order={config.asls_diff_order} needs more than {config.asls_diff_order} points"
                )
            try:
                baseline, parameters = fitter.asls(
                    y,
                    lam=config.asls_lam,
                    p=config.asls_p,
                    diff_order=config.asls_diff_order,
                    max_iter=config.asls_max_iter,
                    tol=config.asls_tol,
                )
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                raise ValueError(f"AsLS baseline failed: {exc}") from exc
            tolerance_history = np.asarray(parameters.get("tol_history", []), dtype=np.float64)
            final_tolerance = float(tolerance_history[-1]) if tolerance_history.size else None
            converged = final_tolerance is None or final_tolerance <= config.asls_tol
            method_diagnostics.update(
                {
                    "iterations": int(tolerance_history.size),
                    "final_tolerance": final_tolerance,
                    "converged": bool(converged),
                }
            )
            if not converged:
                method_diagnostics["warnings"] = [
                    f"AsLS did not converge within {config.asls_max_iter} iterations "
                    f"(final tolerance {final_tolerance}, target {config.asls_tol})."
                ]
        else:
            if x.size < 3:
                raise ValueError("Rubberband baseline needs at least three points")
            try:
                baseline, parameters = fitter.rubberband(
                    y,
                    segments=config.rubberband_segments,
                    lam=config.rubberband_lam,
                    diff_order=config.rubberband_diff_order,
                    smooth_half_window=config.rubberband_smooth_half_window,
                )
            except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
                raise ValueError(f"Rubberband baseline failed: {exc}") from exc
            rubberband_mask = np.asarray(parameters.get("mask", []), dtype=bool)
            method_diagnostics.update(
                {
                    "baseline_point_count": int(np.count_nonzero(rubberband_mask)),
                    "baseline_indices": np.flatnonzero(rubberband_mask).tolist(),
                    "rubberband_mask": rubberband_mask.tolist(),
                    "advisory": (
                        "Rubberband is intended for convex baselines and can perform poorly "
                        "for strongly concave backgrounds; inspect the baseline preview."
                    ),
                }
            )
    else:  # BaselineConfig prevents this; keep a defensive branch for callers.
        raise ValueError(f"Unsupported baseline method: {config.method!r}")

    baseline = np.asarray(baseline, dtype=np.float64)
    if baseline.shape != y.shape or not np.all(np.isfinite(baseline)):
        raise ValueError(
            f"{config.method} produced an invalid baseline with shape {baseline.shape}"
        )
    corrected = np.asarray(y - baseline, dtype=np.float64)
    diagnostics = _common_diagnostics(baseline, corrected, x)
    method_warnings = list(method_diagnostics.pop("warnings", []))
    diagnostics.update(method_diagnostics)
    diagnostics["warnings"] = list(diagnostics["warnings"]) + method_warnings
    return BaselineResult(
        baseline=baseline,
        corrected=corrected,
        method=config.method,
        parameters=config.to_dict(),
        diagnostics=diagnostics,
    )


def correct_baseline(
    dataset: SpectralDataset,
    config: BaselineConfig,
) -> tuple[SpectralDataset, NDArray[np.float64], list[dict[str, Any]]]:
    """Apply exactly one baseline config across all spectra without mutation."""

    if not isinstance(config, BaselineConfig):
        config = BaselineConfig.from_dict(config)
    if not np.all(np.isfinite(dataset.wavenumber)):
        raise ValueError("Wavenumber contains NaN or Inf; baseline correction was not run")
    if not np.all(np.isfinite(dataset.spectra)):
        raise ValueError("Spectra contain NaN or Inf; baseline correction was not run")

    baseline_matrix = np.empty(dataset.spectra.shape, dtype=np.float64)
    corrected_matrix = np.empty(dataset.spectra.shape, dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []
    for index, spectrum in enumerate(dataset.spectra):
        try:
            result = estimate_baseline(dataset.wavenumber, spectrum, config)
        except (ValueError, RuntimeError) as exc:
            label = dataset.perturbation_labels[index]
            raise ValueError(
                f"Baseline correction failed for spectrum {index} ({label!r}): {exc}"
            ) from exc
        baseline_matrix[index] = result.baseline
        corrected_matrix[index] = result.corrected
        item = deepcopy(result.diagnostics)
        item.update(
            {
                "spectrum_index": index,
                "perturbation_label": dataset.perturbation_labels[index],
                "perturbation_value": float(dataset.perturbation[index]),
                "method": config.method,
            }
        )
        diagnostics.append(item)

    all_warnings: list[str] = []
    for item in diagnostics:
        for warning in item.get("warnings", []):
            all_warnings.append(
                f"Spectrum {item['spectrum_index']} ({item['perturbation_label']}): {warning}"
            )
    metadata = deepcopy(dict(dataset.metadata))
    history = list(metadata.get("processing_history", []))
    history.append(
        {
            "operation": "correct_baseline",
            "method": config.method,
            "parameters": config.to_dict(),
            "same_parameters_applied_to_all_spectra": True,
        }
    )
    processing_warnings = list(metadata.get("processing_warnings", []))
    processing_warnings.extend(all_warnings)
    metadata.update(
        {
            "baseline_method": config.method,
            "baseline_parameters": config.to_dict(),
            "baseline_same_parameters_for_all_spectra": True,
            "baseline_peak_to_peak": [float(item["baseline_peak_to_peak"]) for item in diagnostics],
            "baseline_area": [float(item["baseline_area"]) for item in diagnostics],
            "baseline_corrected_minimum": [
                float(item["corrected_minimum"]) for item in diagnostics
            ],
            "baseline_warnings": all_warnings,
            "processing_warnings": processing_warnings,
            "processing_history": history,
        }
    )
    corrected_dataset = dataset.with_updates(spectra=corrected_matrix, metadata=metadata)
    return corrected_dataset, baseline_matrix, diagnostics


correct_spectrum_baseline = estimate_baseline


__all__ = ["correct_baseline", "correct_spectrum_baseline", "estimate_baseline"]
