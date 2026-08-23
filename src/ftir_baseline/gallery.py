"""Transparent candidate gallery and small parameter-grid scans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .baseline.automatic import estimate_coarse
from .baseline.composite import compose_baselines, estimate_fine
from .config import SmoothingConfig
from .models import BaselineResult
from .qc import QCResult, run_quality_control
from .scoring import RankedCandidate, rank_candidates
from .smoothing import prepare_baseline_channels


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One inspectable coarse/fine recipe to evaluate on a representative spectrum."""

    name: str
    coarse_method: str = "none"
    coarse_params: Mapping[str, Any] = field(default_factory=dict)
    fine_method: str = "none"
    fine_params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    name: str
    spec: CandidateSpec
    result: BaselineResult
    qc: QCResult

    @property
    def metrics(self) -> Mapping[str, Any]:
        return self.qc.summary


@dataclass(frozen=True, slots=True)
class CandidateGallery:
    representative_name: str
    representative_spectrum: np.ndarray
    evaluations: tuple[CandidateEvaluation, ...]
    ranking: tuple[RankedCandidate, ...]
    disclaimer: str = (
        "诊断排序仅用于比较候选，不是真实基线的证明；最终配方必须结合峰保真度、"
        "锚点合理性和实验知识确认。"
    )


def starter_pchip_anchor_windows(
    wavenumber: np.ndarray,
    *,
    endpoint_window_width_cm1: float = 8.0,
    statistic: str = "median",
) -> tuple[dict[str, Any], ...]:
    """Return an inspectable starter set that covers the selected interval.

    The internal 1490--1510 and 930--950 cm⁻¹ windows mirror the
    specification's UI examples when they fit inside the selected range.  They
    are only starting hypotheses: the gallery disclaimer and anchor editor make
    clear that users must confirm they are genuinely absorption-free.
    """

    x = np.asarray(wavenumber, dtype=np.float64)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("wavenumber must be a finite one-dimensional axis")
    width = float(endpoint_window_width_cm1)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("endpoint_window_width_cm1 must be finite and positive")
    normalized_statistic = str(statistic).strip().lower()
    if normalized_statistic not in {"median", "mean"}:
        raise ValueError("anchor statistic must be 'median' or 'mean'")
    low = float(np.min(x))
    high = float(np.max(x))
    if not high > low:
        raise ValueError("wavenumber axis must span at least two distinct values")
    # Starter endpoint windows must remain disjoint even for a very narrow
    # selected range.  This adapts only the generated PCHIP hypothesis; it does
    # not alter the user's endpoint-baseline width in the processing recipe.
    half = min(width / 2.0, (high - low) / 4.0)

    def window(start: float, end: float) -> dict[str, Any]:
        return {
            "enabled": True,
            "start": float(start),
            "end": float(end),
            "statistic": normalized_statistic,
        }

    windows = [window(high - half, high + half)]
    for start, end in ((1490.0, 1510.0), (930.0, 950.0)):
        if start > low + half and end < high - half:
            windows.append(window(start, end))
    windows.append(window(low - half, low + half))
    return tuple(windows)


def default_candidate_specs(
    *,
    anchor_windows: Sequence[Any] | None = None,
    arpls_log10_lambda: Sequence[float] = (3, 4, 5, 6, 7, 8, 9),
    asls_log10_lambda: Sequence[float] = (4, 5, 6, 7, 8, 9),
    asls_p: Sequence[float] = (0.001, 0.01, 0.05),
    airpls_log10_lambda: Sequence[float] = (4, 5, 6, 7, 8),
    endpoint_window_width_cm1: float = 8.0,
) -> tuple[CandidateSpec, ...]:
    """Build the specification's editable first-version parameter grid."""

    candidates: list[CandidateSpec] = [
        CandidateSpec(
            "Endpoint linear",
            fine_method="endpoint_window_linear",
            fine_params={
                "endpoint_window_width_cm1": float(endpoint_window_width_cm1),
                "statistic": "median",
            },
        )
    ]
    if anchor_windows:
        candidates.append(
            CandidateSpec(
                "Anchor PCHIP",
                fine_method="pchip",
                fine_params={"anchors": list(anchor_windows), "statistic": "median"},
            )
        )
    candidates.extend(
        CandidateSpec(
            f"arPLS λ=1e{float(log_lam):g}",
            coarse_method="arpls",
            coarse_params={"lam": 10.0 ** float(log_lam)},
        )
        for log_lam in arpls_log10_lambda
    )
    candidates.extend(
        CandidateSpec(
            f"AsLS λ=1e{float(log_lam):g}, p={float(p):g}",
            coarse_method="asls",
            coarse_params={"lam": 10.0 ** float(log_lam), "p": float(p)},
        )
        for log_lam in asls_log10_lambda
        for p in asls_p
    )
    candidates.extend(
        CandidateSpec(
            f"airPLS λ=1e{float(log_lam):g}",
            coarse_method="airpls",
            coarse_params={"lam": 10.0 ** float(log_lam)},
        )
        for log_lam in airpls_log10_lambda
    )
    candidates.append(CandidateSpec("Rubberband", coarse_method="rubberband"))
    return tuple(candidates)


def _representative(
    spectra: np.ndarray,
    representative: str | int,
) -> tuple[np.ndarray, str]:
    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim == 1:
        return values.copy(), "single"
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("spectra must have shape (n_points,) or (n_spectra, n_points)")
    if isinstance(representative, int):
        if not -values.shape[0] <= representative < values.shape[0]:
            raise IndexError("representative spectrum index is out of range")
        index = representative % values.shape[0]
        return values[index].copy(), f"spectrum {index}"
    normalized = representative.strip().lower()
    if normalized in {"first", "start"}:
        return values[0].copy(), "first spectrum"
    if normalized in {"last", "end"}:
        return values[-1].copy(), "last spectrum"
    if normalized in {"middle", "mid"}:
        index = values.shape[0] // 2
        return values[index].copy(), f"middle spectrum {index}"
    if normalized == "mean":
        return np.mean(values, axis=0), "mean spectrum"
    if normalized == "median":
        return np.median(values, axis=0), "median spectrum"
    raise ValueError("representative must be first, middle, last, mean, median, or an index")


def scan_baseline_candidates(
    wavenumber: np.ndarray,
    spectra: np.ndarray,
    candidates: Sequence[CandidateSpec] | None = None,
    *,
    representative: str | int = "median",
    smoothing: SmoothingConfig | None = None,
    anchor_windows: Sequence[Any] | None = None,
    peak_regions: Sequence[Any] | None = None,
    score_weights: Mapping[str, float] | None = None,
) -> CandidateGallery:
    """Evaluate candidates without claiming to identify the physical baseline."""

    x = np.asarray(wavenumber, dtype=np.float64)
    y, representative_name = _representative(spectra, representative)
    channels = prepare_baseline_channels(y, smoothing)
    y_for_baseline = np.asarray(channels.for_baseline)
    specs = tuple(
        default_candidate_specs(anchor_windows=anchor_windows) if candidates is None else candidates
    )
    if not specs:
        raise ValueError("candidate gallery requires at least one candidate")

    evaluations: list[CandidateEvaluation] = []
    ranking_input: dict[str, Mapping[str, Any]] = {}
    seen_names: set[str] = set()
    for spec in specs:
        if spec.name in seen_names:
            raise ValueError(f"candidate names must be unique; duplicate {spec.name!r}")
        seen_names.add(spec.name)
        coarse = estimate_coarse(x, y_for_baseline, spec.coarse_method, **spec.coarse_params)
        residual = y_for_baseline - np.asarray(coarse.total_baseline)
        fine = estimate_fine(x, residual, spec.fine_method, **spec.fine_params)
        result = compose_baselines(y, coarse, fine)
        qc = run_quality_control(
            x,
            y,
            result.total_baseline,
            result.corrected,
            anchor_windows=anchor_windows,
            peak_regions=peak_regions,
            score_weights=score_weights,
        )
        evaluation = CandidateEvaluation(spec.name, spec, result, qc)
        evaluations.append(evaluation)
        ranking_input[spec.name] = {
            "anchor_error": qc.per_spectrum["anchor_error"],
            "negative_fraction": qc.per_spectrum["negative_fraction"],
            "baseline_roughness": qc.per_spectrum["baseline_roughness"],
            "time_roughness": qc.summary["time_roughness"],
            "peak_change_penalty": qc.per_spectrum["peak_change_penalty"],
        }

    return CandidateGallery(
        representative_name=representative_name,
        representative_spectrum=y.copy(),
        evaluations=tuple(evaluations),
        ranking=rank_candidates(ranking_input, score_weights),
    )


__all__ = [
    "CandidateEvaluation",
    "CandidateGallery",
    "CandidateSpec",
    "default_candidate_specs",
    "scan_baseline_candidates",
    "starter_pchip_anchor_windows",
]
