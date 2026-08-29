"""Pure baseline-preview payload and figure helpers.

The helpers in this module only read arrays already produced by the frozen
``ftir_baseline`` pipeline.  The sole derived scientific-looking curve is the
coarse residual explicitly required by the v0.2 UI specification:
``A_raw - B_coarse``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray

from ftir_baseline.pipeline import endpoint_anchor_windows

FloatArray = NDArray[np.float64]
AggregateKind = Literal["mean", "median"]


def _readonly_vector(values: Any, *, name: str) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class RepresentativeSelection:
    """One actual spectrum row or one UI-only series aggregate."""

    key: str
    label: str
    spectrum_index: int | None = None
    aggregate: AggregateKind | None = None

    def __post_init__(self) -> None:
        if (self.spectrum_index is None) == (self.aggregate is None):
            raise ValueError("representative must select exactly one row or aggregate")
        if self.spectrum_index is not None and self.spectrum_index < 0:
            raise ValueError("spectrum_index must be non-negative")


def representative_options(labels: Sequence[str]) -> tuple[RepresentativeSelection, ...]:
    """Return first/middle/last/mean/median plus unambiguous actual-row choices."""

    normalized = tuple(str(item) for item in labels)
    if not normalized:
        raise ValueError("at least one spectrum label is required")
    last_index = len(normalized) - 1
    middle_index = len(normalized) // 2
    shortcuts = (
        RepresentativeSelection("first", f"first · {normalized[0]}", spectrum_index=0),
        RepresentativeSelection(
            "middle",
            f"middle · {normalized[middle_index]}",
            spectrum_index=middle_index,
        ),
        RepresentativeSelection(
            "last",
            f"last · {normalized[last_index]}",
            spectrum_index=last_index,
        ),
        RepresentativeSelection("mean", "mean spectrum", aggregate="mean"),
        RepresentativeSelection("median", "median spectrum", aggregate="median"),
    )
    actual = tuple(
        RepresentativeSelection(
            f"spectrum:{index}",
            f"spectrum {index} · {label}",
            spectrum_index=index,
        )
        for index, label in enumerate(normalized)
    )
    return shortcuts + actual


def resolve_representative(
    labels: Sequence[str],
    selection: RepresentativeSelection | str | int,
) -> RepresentativeSelection:
    """Resolve a UI token, perturbation label, or integer row to a stable selection."""

    normalized = tuple(str(item) for item in labels)
    if not normalized:
        raise ValueError("at least one spectrum label is required")
    if isinstance(selection, RepresentativeSelection):
        if selection.spectrum_index is not None:
            if selection.spectrum_index >= len(normalized):
                raise IndexError("representative spectrum index is out of range")
            return RepresentativeSelection(
                selection.key,
                selection.label,
                spectrum_index=selection.spectrum_index,
            )
        return RepresentativeSelection(
            selection.key,
            selection.label,
            aggregate=selection.aggregate,
        )
    if isinstance(selection, int):
        if not 0 <= selection < len(normalized):
            raise IndexError("representative spectrum index is out of range")
        return RepresentativeSelection(
            f"spectrum:{selection}",
            f"spectrum {selection} · {normalized[selection]}",
            spectrum_index=selection,
        )

    token = str(selection).strip()
    folded = token.casefold()
    shortcut_lookup = {item.key: item for item in representative_options(normalized)[:5]}
    if folded in shortcut_lookup:
        return shortcut_lookup[folded]
    if folded.startswith("spectrum:"):
        try:
            return resolve_representative(normalized, int(folded.partition(":")[2]))
        except ValueError as exc:
            raise ValueError("spectrum token must end with an integer index") from exc
    matching_indices = [index for index, label in enumerate(normalized) if label == token]
    if len(matching_indices) == 1:
        return resolve_representative(normalized, matching_indices[0])
    if len(matching_indices) > 1:
        raise ValueError("perturbation label is ambiguous; use a spectrum:<index> token")
    raise ValueError(
        "representative must be first, middle, last, mean, median, an index, "
        "a spectrum:<index> token, or an exact perturbation label"
    )


def representative_values(
    matrix: Any,
    selection: RepresentativeSelection,
    *,
    name: str,
) -> FloatArray:
    """Select or aggregate rows without mutating the authoritative matrix."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"{name} must have shape (n_spectra, n_points)")
    if selection.spectrum_index is not None:
        if selection.spectrum_index >= values.shape[0]:
            raise IndexError("representative spectrum index is out of range")
        representative = values[selection.spectrum_index]
    elif selection.aggregate == "mean":
        representative = np.mean(values, axis=0)
    elif selection.aggregate == "median":
        representative = np.median(values, axis=0)
    else:  # pragma: no cover - guarded by RepresentativeSelection
        raise ValueError("unsupported representative aggregate")
    return _readonly_vector(representative, name=name)


@dataclass(frozen=True, slots=True)
class BaselinePreviewPayload:
    """Selected display curves from one complete-series ``PipelineResult``."""

    selection: RepresentativeSelection
    wavenumber: FloatArray
    raw_absorbance: FloatArray
    baseline_estimation: FloatArray
    coarse_baseline: FloatArray
    fine_baseline: FloatArray
    total_baseline: FloatArray
    corrected: FloatArray

    def __post_init__(self) -> None:
        vectors = {
            "wavenumber": self.wavenumber,
            "raw_absorbance": self.raw_absorbance,
            "baseline_estimation": self.baseline_estimation,
            "coarse_baseline": self.coarse_baseline,
            "fine_baseline": self.fine_baseline,
            "total_baseline": self.total_baseline,
            "corrected": self.corrected,
        }
        copied = {name: _readonly_vector(value, name=name) for name, value in vectors.items()}
        size = copied["wavenumber"].size
        if any(value.size != size for value in copied.values()):
            raise ValueError("all baseline-preview curves must match the wavenumber axis")
        for name, value in copied.items():
            object.__setattr__(self, name, value)

    @property
    def residual_after_coarse(self) -> FloatArray:
        """Return the specification-authorized UI residual ``A_raw - B_coarse``."""

        return _readonly_vector(
            self.raw_absorbance - self.coarse_baseline,
            name="residual_after_coarse",
        )


def baseline_preview_payload(
    result: Any,
    selection: RepresentativeSelection | str | int,
) -> BaselinePreviewPayload:
    """Bind one representative view to immutable arrays from a full pipeline run."""

    labels = result.absorbance_selected.perturbation_labels
    resolved = resolve_representative(labels, selection)
    return BaselinePreviewPayload(
        selection=resolved,
        wavenumber=result.absorbance_selected.wavenumber,
        raw_absorbance=representative_values(
            result.absorbance_selected.spectra,
            resolved,
            name="raw_absorbance",
        ),
        baseline_estimation=representative_values(
            result.baseline_estimation_spectra,
            resolved,
            name="baseline_estimation",
        ),
        coarse_baseline=representative_values(
            result.baseline.coarse_baseline,
            resolved,
            name="coarse_baseline",
        ),
        fine_baseline=representative_values(
            result.baseline.fine_baseline,
            resolved,
            name="fine_baseline",
        ),
        total_baseline=representative_values(
            result.baseline.total_baseline,
            resolved,
            name="total_baseline",
        ),
        corrected=representative_values(
            result.analysis_data,
            resolved,
            name="corrected",
        ),
    )


def _curve_figure(
    payload: BaselinePreviewPayload,
    curves: Sequence[tuple[str, FloatArray]],
    *,
    title: str,
) -> go.Figure:
    figure = go.Figure()
    for label, values in curves:
        figure.add_trace(
            go.Scatter(
                x=payload.wavenumber,
                y=values,
                mode="lines",
                name=label,
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance",
        height=470,
        margin={"l": 45, "r": 20, "t": 55, "b": 40},
    )
    figure.update_xaxes(autorange="reversed")
    return figure


def coarse_preview_figure(
    result: Any,
    selection: RepresentativeSelection | str | int,
    *,
    title: str = "Current coarse-baseline recipe preview",
) -> go.Figure:
    """Plot raw, estimation channel, coarse baseline, and coarse residual."""

    payload = baseline_preview_payload(result, selection)
    return _curve_figure(
        payload,
        (
            ("Raw selected absorbance", payload.raw_absorbance),
            ("Baseline-estimation channel", payload.baseline_estimation),
            ("Coarse baseline", payload.coarse_baseline),
            ("Residual after coarse baseline", payload.residual_after_coarse),
        ),
        title=title,
    )


def fine_decomposition_figure(
    result: Any,
    selection: RepresentativeSelection | str | int,
    *,
    title: str = "Fine-baseline decomposition preview",
) -> go.Figure:
    """Plot the complete decomposition supplied by one preview result."""

    payload = baseline_preview_payload(result, selection)
    return _curve_figure(
        payload,
        (
            ("A_raw", payload.raw_absorbance),
            ("A_for_baseline", payload.baseline_estimation),
            ("B_coarse", payload.coarse_baseline),
            ("B_fine", payload.fine_baseline),
            ("B_total", payload.total_baseline),
            ("Corrected", payload.corrected),
        ),
        title=title,
    )


def fine_residual_figure(
    result: Any,
    selection: RepresentativeSelection | str | int,
    *,
    title: str = "Fine-baseline residual preview",
) -> go.Figure:
    """Plot coarse residual, fitted fine baseline, and final corrected spectrum."""

    payload = baseline_preview_payload(result, selection)
    return _curve_figure(
        payload,
        (
            ("Residual after coarse", payload.residual_after_coarse),
            ("B_fine", payload.fine_baseline),
            ("Final corrected", payload.corrected),
        ),
        title=title,
    )


@dataclass(frozen=True, slots=True)
class AnchorDiagnostic:
    """One fitted anchor window and the exact representative consumed by the fit."""

    anchor: str
    start: float
    end: float
    statistic: str
    representative_wavenumber: float
    representative_value: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "Anchor": self.anchor,
            "Start": self.start,
            "End": self.end,
            "Statistic": self.statistic,
            "Representative wavenumber": self.representative_wavenumber,
            "Representative value (B_fine)": self.representative_value,
        }


def _fine_params_and_metrics(result: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    params = result.baseline.params
    metrics = result.baseline.metrics
    if result.config.series_mode == "shared_shape":
        return params, metrics
    fine_params = params.get("fine", {}) if isinstance(params, Mapping) else {}
    fine_metrics = metrics.get("fine", {}) if isinstance(metrics, Mapping) else {}
    return (
        fine_params if isinstance(fine_params, Mapping) else {},
        fine_metrics if isinstance(fine_metrics, Mapping) else {},
    )


def anchor_diagnostics(
    result: Any,
    selection: RepresentativeSelection | str | int,
) -> tuple[AnchorDiagnostic, ...]:
    """Extract fitted anchor windows/representatives without refitting anything."""

    resolved = resolve_representative(
        result.absorbance_selected.perturbation_labels,
        selection,
    )
    params, metrics = _fine_params_and_metrics(result)
    method = str(params.get("method", "none"))
    windows: list[dict[str, Any]] = []
    centers = np.empty(0, dtype=np.float64)
    values_by_spectrum = np.empty((result.absorbance_selected.n_spectra, 0), dtype=np.float64)

    if method == "shared_shape":
        windows = [dict(item) for item in params.get("anchors", ())]
        centers = np.asarray(params.get("anchor_centers", ()), dtype=np.float64)
        values_by_spectrum = np.asarray(
            metrics.get("residual_anchor_values", ()),
            dtype=np.float64,
        )
    elif method in {"multipoint_linear", "piecewise_linear", "pchip", "polynomial"}:
        windows = [dict(item) for item in params.get("anchors", ())]
        centers = np.asarray(params.get("anchor_centers", ()), dtype=np.float64)
        values_by_spectrum = np.asarray(params.get("anchor_values", ()), dtype=np.float64)
    elif method == "endpoint_window_linear":
        width = float(params.get("endpoint_window_width_cm1", 8.0))
        windows = endpoint_anchor_windows(result.absorbance_selected.wavenumber, width)
        statistic = str(params.get("statistic", "median"))
        for window in windows:
            window["statistic"] = statistic
        fitted = params.get("fitted", {})
        if isinstance(fitted, Mapping):
            centers = np.asarray(fitted.get("representative_wavenumbers", ()), dtype=np.float64)
            lower = np.asarray(fitted.get("lower_values", ()), dtype=np.float64)
            upper = np.asarray(fitted.get("upper_values", ()), dtype=np.float64)
            if lower.ndim == 1 and upper.ndim == 1 and lower.shape == upper.shape:
                values_by_spectrum = np.column_stack((lower, upper))
    elif method == "strict_endpoint":
        centers = np.asarray(params.get("endpoint_wavenumbers", ()), dtype=np.float64)
        endpoint_values = params.get("endpoint_values", {})
        if isinstance(endpoint_values, Mapping):
            lower = np.asarray(endpoint_values.get("lower", ()), dtype=np.float64)
            upper = np.asarray(endpoint_values.get("upper", ()), dtype=np.float64)
            if lower.ndim == 1 and upper.ndim == 1 and lower.shape == upper.shape:
                values_by_spectrum = np.column_stack((lower, upper))
        windows = [
            {
                "start": center,
                "end": center,
                "statistic": "exact endpoint",
            }
            for center in centers
        ]
    else:
        return ()

    if centers.ndim != 1 or values_by_spectrum.ndim != 2:
        return ()
    if values_by_spectrum.shape[0] != result.absorbance_selected.n_spectra:
        return ()
    values = representative_values(values_by_spectrum, resolved, name="anchor_values")
    if len(windows) != centers.size or centers.shape != values.shape:
        return ()
    if not np.isfinite(centers).all():
        return ()
    return tuple(
        AnchorDiagnostic(
            anchor=f"A{index}",
            start=float(window["start"]),
            end=float(window["end"]),
            statistic=str(window.get("statistic", "median")),
            representative_wavenumber=float(center),
            representative_value=float(value),
        )
        for index, (window, center, value) in enumerate(
            zip(windows, centers, values, strict=True),
            start=1,
        )
    )


def anchor_diagnostics_table(
    result: Any,
    selection: RepresentativeSelection | str | int,
) -> pd.DataFrame:
    """Return a detached, display-ready table of fitted anchor diagnostics."""

    columns = (
        "Anchor",
        "Start",
        "End",
        "Statistic",
        "Representative wavenumber",
        "Representative value (B_fine)",
    )
    rows = [item.to_dict() for item in anchor_diagnostics(result, selection)]
    return pd.DataFrame(rows, columns=columns)


def add_anchor_overlays(
    figure: go.Figure,
    diagnostics: Sequence[AnchorDiagnostic],
) -> go.Figure:
    """Return a copied figure with fitted windows and representative points."""

    rendered = go.Figure(figure)
    items = tuple(diagnostics)
    for item in items:
        start, end = sorted((item.start, item.end))
        rendered.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(255, 165, 0, 0.16)",
            line_width=1,
            line_color="rgba(220, 120, 0, 0.55)",
            annotation_text=item.anchor,
            annotation_position="top left",
        )
    if items:
        rendered.add_trace(
            go.Scatter(
                x=[item.representative_wavenumber for item in items],
                y=[item.representative_value for item in items],
                mode="markers+text",
                text=[f"{item.representative_value:.4g}" for item in items],
                textposition="top center",
                marker={"size": 9, "symbol": "diamond", "color": "darkorange"},
                name="Anchor statistic on B_fine",
            )
        )
    return rendered


__all__ = [
    "AnchorDiagnostic",
    "BaselinePreviewPayload",
    "RepresentativeSelection",
    "add_anchor_overlays",
    "anchor_diagnostics",
    "anchor_diagnostics_table",
    "baseline_preview_payload",
    "coarse_preview_figure",
    "fine_decomposition_figure",
    "fine_residual_figure",
    "representative_options",
    "representative_values",
    "resolve_representative",
]
