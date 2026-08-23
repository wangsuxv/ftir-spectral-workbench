"""Quality-control orchestration for corrected FTIR series."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import freeze_value, immutable_float64
from .scoring import (
    anchor_residual_error,
    baseline_roughness,
    diagnostic_score,
    estimate_noise_sigma,
    negative_residual_fraction,
    peak_preservation,
    reconstruction_check,
    reconstruction_error,
    temporal_roughness,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class QCResult:
    """Structured QC result suitable for tables, reports, and JSON recipes."""

    per_spectrum: Mapping[str, FloatArray]
    summary: Mapping[str, Any]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        per_spectrum = {
            str(name): immutable_float64(values, name=f"qc.per_spectrum.{name}")
            for name, values in self.per_spectrum.items()
        }
        object.__setattr__(
            self,
            "per_spectrum",
            freeze_value(per_spectrum, path="qc.per_spectrum"),
        )
        object.__setattr__(self, "summary", freeze_value(dict(self.summary), path="qc.summary"))
        object.__setattr__(self, "warnings", tuple(map(str, self.warnings)))

    @property
    def metrics(self) -> dict[str, Any]:
        return {"per_spectrum": dict(self.per_spectrum), "summary": dict(self.summary)}

    def as_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return convert(value.tolist())
            if isinstance(value, np.generic):
                return convert(value.item())
            if isinstance(value, float) and not np.isfinite(value):
                return None
            if isinstance(value, Mapping):
                return {str(key): convert(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [convert(item) for item in value]
            return value

        return {
            "per_spectrum": convert(self.per_spectrum),
            "summary": convert(self.summary),
            "warnings": list(self.warnings),
        }


def _as_series(values: ArrayLike, name: str) -> tuple[FloatArray, bool]:
    result = np.asarray(values, dtype=np.float64)
    was_1d = result.ndim == 1
    if was_1d:
        result = result[np.newaxis, :]
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_spectra, n_points) or (n_points,)")
    if not np.all(np.isfinite(result)):
        where = np.argwhere(~np.isfinite(result))[0]
        raise ValueError(
            f"{name} contains NaN or Inf at spectrum {int(where[0])}, point {int(where[1])}"
        )
    return result, was_1d


def _as_axis(wavenumber: ArrayLike, n_points: int) -> FloatArray:
    x = np.asarray(wavenumber, dtype=np.float64)
    if x.ndim != 1 or x.shape != (n_points,):
        raise ValueError(f"wavenumber must have shape ({n_points},)")
    if not np.all(np.isfinite(x)):
        raise ValueError("wavenumber contains NaN or Inf")
    if x.size > 1 and not (np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)):
        raise ValueError("wavenumber must be strictly monotonic")
    return x


def _integral_rows(x: FloatArray, values: FloatArray) -> FloatArray:
    order = np.argsort(x)
    ordered = values[:, order]
    widths = np.diff(x[order])
    return np.asarray(
        np.sum(0.5 * (ordered[:, :-1] + ordered[:, 1:]) * widths, axis=1),
        dtype=np.float64,
    )


def _safe_mean(values: ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _safe_max(values: ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else float("nan")


def _temporal_spikes(baseline: FloatArray) -> tuple[int, ...]:
    """Flag unusually large second differences without deleting any spectra."""

    if baseline.shape[0] < 3:
        return ()
    magnitudes = np.sqrt(np.mean(np.diff(baseline, n=2, axis=0) ** 2, axis=1))
    centre = float(np.median(magnitudes))
    mad = float(np.median(np.abs(magnitudes - centre)))
    if mad <= np.finfo(np.float64).eps:
        threshold = max(10.0 * centre, 64.0 * np.finfo(np.float64).eps)
    else:
        threshold = centre + 6.0 * 1.4826 * mad
    # A second difference centred on original row j corresponds to index j+1.
    return tuple((np.flatnonzero(magnitudes > threshold) + 1).astype(int).tolist())


def run_quality_control(
    wavenumber: ArrayLike,
    raw_absorbance: ArrayLike,
    total_baseline: ArrayLike,
    corrected_absorbance: ArrayLike,
    *,
    anchor_windows: Sequence[Any] | None = None,
    peak_regions: Sequence[Any] | None = None,
    perturbation: ArrayLike | None = None,
    negative_k: float = 3.0,
    score_weights: Mapping[str, float] | None = None,
) -> QCResult:
    """Compute all required baseline and series diagnostics.

    The incoming spectrum order is retained exactly.  A temporal discontinuity
    is reported as a warning only; it may be a real experimental change.
    """

    raw, _ = _as_series(raw_absorbance, "raw_absorbance")
    baseline, _ = _as_series(total_baseline, "total_baseline")
    corrected, _ = _as_series(corrected_absorbance, "corrected_absorbance")
    if raw.shape != baseline.shape or raw.shape != corrected.shape:
        raise ValueError("raw_absorbance, total_baseline, and corrected_absorbance must match")
    x = _as_axis(wavenumber, raw.shape[1])
    n_spectra = raw.shape[0]

    if perturbation is None:
        perturbation_values = np.arange(n_spectra, dtype=np.float64)
    else:
        perturbation_values = np.asarray(perturbation, dtype=np.float64)
        if perturbation_values.shape != (n_spectra,):
            raise ValueError(f"perturbation must have shape ({n_spectra},)")
        if not np.all(np.isfinite(perturbation_values)):
            raise ValueError("perturbation contains NaN or Inf")

    warnings: list[str] = []
    noise = np.asarray(estimate_noise_sigma(corrected), dtype=np.float64).reshape(n_spectra)
    negative = np.asarray(
        negative_residual_fraction(corrected, noise, k=negative_k), dtype=np.float64
    ).reshape(n_spectra)
    roughness = np.asarray(baseline_roughness(baseline), dtype=np.float64).reshape(n_spectra)
    reconstruction = np.asarray(
        reconstruction_error(raw, baseline, corrected), dtype=np.float64
    ).reshape(n_spectra)
    preservation = peak_preservation(x, raw, corrected, peak_regions)
    derivative_correlation = np.asarray(
        preservation["derivative_correlation"], dtype=np.float64
    ).reshape(n_spectra)
    position_shift = np.asarray(preservation["peak_position_shift"], dtype=np.float64).reshape(
        n_spectra
    )
    peak_penalty = np.asarray(preservation["peak_change_penalty"], dtype=np.float64).reshape(
        n_spectra
    )
    peak_height_change = np.asarray(
        preservation["peak_height_relative_change"], dtype=np.float64
    ).reshape(n_spectra)

    if anchor_windows:
        anchors = np.asarray(
            anchor_residual_error(x, corrected, anchor_windows), dtype=np.float64
        ).reshape(n_spectra)
    else:
        anchors = np.full(n_spectra, np.nan, dtype=np.float64)
        warnings.append("Anchor residual was not computed because no anchor windows were supplied.")

    baseline_area = _integral_rows(x, baseline)
    if n_spectra > 1:
        adjacent_change = np.concatenate(
            (
                np.array([np.nan], dtype=np.float64),
                np.sqrt(np.mean(np.diff(baseline, axis=0) ** 2, axis=1)),
            )
        )
    else:
        adjacent_change = np.array([np.nan], dtype=np.float64)

    reconstruction_passed = reconstruction_check(raw, baseline, corrected)
    temporal = temporal_roughness(baseline)
    spikes = _temporal_spikes(baseline)

    score_components: dict[str, Any] = {
        "anchor_error": anchors,
        "negative_fraction": negative,
        "baseline_roughness": roughness,
        "time_roughness": temporal,
        "peak_change_penalty": peak_penalty,
    }
    ranking_score = diagnostic_score(score_components, score_weights)

    if not reconstruction_passed:
        warnings.append(
            "Reconstruction check failed: raw_absorbance is not equal to "
            "total_baseline + corrected_absorbance within floating-point tolerance."
        )
    high_negative = np.flatnonzero(negative > 0.05)
    if high_negative.size:
        warnings.append(
            "Significant negative residuals exceed 5% in spectrum indices "
            f"{high_negative.astype(int).tolist()}; inspect rather than clipping them."
        )
    low_correlation = np.flatnonzero(derivative_correlation < 0.95)
    if low_correlation.size:
        warnings.append(
            "Derivative peak-shape correlation is below 0.95 in spectrum indices "
            f"{low_correlation.astype(int).tolist()}."
        )
    if spikes:
        warnings.append(
            "Possible temporal baseline jump near spectrum indices "
            f"{list(spikes)}. This may reflect a real contact or instrument change; "
            "no spectra were removed or reordered."
        )

    per_spectrum: dict[str, FloatArray] = {
        "spectrum_index": np.arange(n_spectra, dtype=np.float64),
        "perturbation": perturbation_values.copy(),
        "noise_sigma": noise,
        "anchor_error": anchors,
        "anchor_residual_error": anchors.copy(),
        "negative_fraction": negative,
        "baseline_roughness": roughness,
        "baseline_area": baseline_area,
        "adjacent_baseline_rms": adjacent_change,
        "derivative_correlation": derivative_correlation,
        "peak_position_shift": position_shift,
        "peak_height_relative_change": peak_height_change,
        "peak_change_penalty": peak_penalty,
        "reconstruction_error": reconstruction,
    }
    summary: dict[str, Any] = {
        "n_spectra": n_spectra,
        "n_points": raw.shape[1],
        "mean_anchor_error": _safe_mean(anchors),
        "mean_negative_fraction": _safe_mean(negative),
        "mean_baseline_roughness": _safe_mean(roughness),
        "time_roughness": temporal,
        "mean_derivative_correlation": _safe_mean(derivative_correlation),
        "mean_peak_position_shift": _safe_mean(position_shift),
        "maximum_reconstruction_error": _safe_max(reconstruction),
        "reconstruction_passed": reconstruction_passed,
        "temporal_jump_indices": list(spikes),
        "diagnostic_score": ranking_score,
        "diagnostic_score_disclaimer": (
            "Candidate-ranking heuristic only; it is not proof of the true baseline."
        ),
        "time_order_preserved": True,
    }
    return QCResult(per_spectrum=per_spectrum, summary=summary, warnings=tuple(warnings))


compute_qc_metrics = run_quality_control
evaluate_qc = run_quality_control


def generate_qc_figures(
    wavenumber: ArrayLike,
    raw_absorbance: ArrayLike,
    total_baseline: ArrayLike,
    corrected_absorbance: ArrayLike,
    qc_result: QCResult | None = None,
    *,
    perturbation: ArrayLike | None = None,
    strict: bool = False,
) -> dict[str, bytes]:
    """Render self-contained PNG figures for the export bundle.

    Matplotlib is imported lazily.  If this optional reporting dependency is
    unavailable, the numerical pipeline continues and an empty mapping is
    returned unless ``strict=True``.
    """

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        if strict:
            raise RuntimeError(
                "QC figure generation requires matplotlib; install the reporting dependencies"
            ) from None
        return {}

    raw, _ = _as_series(raw_absorbance, "raw_absorbance")
    baseline, _ = _as_series(total_baseline, "total_baseline")
    corrected, _ = _as_series(corrected_absorbance, "corrected_absorbance")
    if raw.shape != baseline.shape or raw.shape != corrected.shape:
        raise ValueError("raw_absorbance, total_baseline, and corrected_absorbance must match")
    x = _as_axis(wavenumber, raw.shape[1])
    if perturbation is None:
        p = np.arange(raw.shape[0], dtype=np.float64)
    else:
        p = np.asarray(perturbation, dtype=np.float64)
        if p.shape != (raw.shape[0],):
            raise ValueError(f"perturbation must have shape ({raw.shape[0]},)")

    if qc_result is None:
        qc_result = run_quality_control(x, raw, baseline, corrected, perturbation=p)

    figures: dict[str, bytes] = {}

    def save(name: str, figure: Any) -> None:
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(figure)
        figures[name] = buffer.getvalue()

    representative = sorted({0, raw.shape[0] // 2, raw.shape[0] - 1})
    figure, axes = plt.subplots(len(representative), 1, figsize=(9, 2.8 * len(representative)))
    axes_array = np.atleast_1d(axes)
    for axis, row in zip(axes_array, representative, strict=True):
        axis.plot(x, raw[row], label="Raw absorbance", linewidth=1.0)
        axis.plot(x, baseline[row], label="Total baseline", linewidth=1.0)
        axis.plot(x, corrected[row], label="Corrected", linewidth=1.0)
        axis.set_ylabel("Absorbance")
        axis.set_title(f"Spectrum {row}")
        axis.invert_xaxis()
        axis.legend(loc="best", fontsize="small")
    axes_array[-1].set_xlabel("Wavenumber (cm⁻¹)")
    figure.suptitle("Representative spectra and reconstructed components")
    figure.tight_layout()
    save("representative_spectra.png", figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    extent = [float(np.max(x)), float(np.min(x)), float(p[0]), float(p[-1])]
    for axis, matrix, title in zip(
        axes,
        (raw, baseline, corrected),
        ("Raw absorbance", "Total baseline", "Corrected absorbance"),
        strict=True,
    ):
        display_matrix = matrix if x.size < 2 or x[0] > x[-1] else matrix[:, ::-1]
        image = axis.imshow(display_matrix, aspect="auto", origin="lower", extent=extent)
        axis.set_title(title)
        axis.set_xlabel("Wavenumber (cm⁻¹)")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Perturbation")
    figure.tight_layout()
    save("series_heatmaps.png", figure)

    metrics = qc_result.per_spectrum
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes[0, 0].plot(p, metrics["baseline_area"], marker="o", markersize=3)
    axes[0, 0].set_ylabel("Baseline area")
    axes[0, 1].plot(p, metrics["anchor_error"], marker="o", markersize=3)
    axes[0, 1].set_ylabel("Anchor residual")
    axes[1, 0].plot(p, metrics["adjacent_baseline_rms"], marker="o", markersize=3)
    axes[1, 0].set_ylabel("Adjacent baseline RMS")
    axes[1, 1].plot(p, metrics["negative_fraction"], marker="o", markersize=3)
    axes[1, 1].set_ylabel("Negative fraction")
    for axis in axes[-1, :]:
        axis.set_xlabel("Perturbation")
    figure.suptitle("Baseline QC trends (original time order)")
    figure.tight_layout()
    save("metric_trends.png", figure)
    return figures


create_qc_figures = generate_qc_figures


__all__ = [
    "QCResult",
    "compute_qc_metrics",
    "create_qc_figures",
    "evaluate_qc",
    "generate_qc_figures",
    "run_quality_control",
]
