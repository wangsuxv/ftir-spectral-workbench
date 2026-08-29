"""Pure Series Consistency & QC display helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .baseline_preview import (
    BaselinePreviewPayload,
    RepresentativeSelection,
    baseline_preview_payload,
    fine_decomposition_figure,
    resolve_representative,
)

REQUIRED_QC_FIELDS = (
    "spectrum_index",
    "perturbation",
    "noise_sigma",
    "anchor_error",
    "anchor_residual_error",
    "negative_fraction",
    "baseline_roughness",
    "baseline_area",
    "adjacent_baseline_rms",
    "derivative_correlation",
    "peak_position_shift",
    "peak_height_relative_change",
    "peak_change_penalty",
    "reconstruction_error",
)


def _finite_matrix(values: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_spectra, n_points)")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _crosses_zero(values: np.ndarray) -> bool:
    return bool(float(np.min(values)) < 0.0 < float(np.max(values)))


def series_heatmap(
    wavenumber: Any,
    perturbation: Any,
    matrix: Any,
    *,
    title: str,
    diverging: bool,
) -> go.Figure:
    """Render a supplied pipeline matrix without recalculation or reordering."""

    axis = np.asarray(wavenumber, dtype=np.float64)
    perturbation_axis = np.asarray(perturbation, dtype=np.float64)
    values = _finite_matrix(matrix, name=title)
    if axis.ndim != 1 or axis.size != values.shape[1]:
        raise ValueError("wavenumber must match matrix columns")
    if perturbation_axis.ndim != 1 or perturbation_axis.size != values.shape[0]:
        raise ValueError("perturbation must match matrix rows")
    settings: dict[str, Any] = {
        "x": axis,
        "y": perturbation_axis,
        "z": values,
        "colorscale": "RdBu_r" if diverging else "Viridis",
        "colorbar": {"title": "Intensity"},
    }
    if diverging:
        scale = float(np.max(np.abs(values))) or 1.0
        settings.update({"zmin": -scale, "zmax": scale, "zmid": 0.0})
    figure = go.Figure(go.Heatmap(**settings))
    figure.update_layout(
        title=title,
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Perturbation",
        height=390,
        margin={"l": 55, "r": 25, "t": 50, "b": 45},
    )
    figure.update_xaxes(autorange="reversed")
    return figure


def five_heatmap_figures(result: Any) -> dict[str, go.Figure]:
    """Return the five mandated heatmaps bound to exact ``PipelineResult`` arrays."""

    axis = result.absorbance_selected.wavenumber
    perturbation = result.absorbance_selected.perturbation
    sources = (
        ("Raw absorbance", result.absorbance_selected.spectra, False),
        (
            "Coarse baseline",
            result.baseline.coarse_baseline,
            _crosses_zero(np.asarray(result.baseline.coarse_baseline)),
        ),
        (
            "Fine baseline",
            result.baseline.fine_baseline,
            _crosses_zero(np.asarray(result.baseline.fine_baseline)),
        ),
        (
            "Total baseline",
            result.baseline.total_baseline,
            _crosses_zero(np.asarray(result.baseline.total_baseline)),
        ),
        ("Corrected absorbance", result.analysis_data, True),
    )
    return {
        title: series_heatmap(
            axis,
            perturbation,
            matrix,
            title=title,
            diverging=diverging,
        )
        for title, matrix, diverging in sources
    }


def complete_qc_table(
    source: Any,
    perturbation_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return every per-spectrum QC field plus its aligned perturbation label."""

    qc = getattr(source, "qc", source)
    per_spectrum = getattr(qc, "per_spectrum", None)
    if not isinstance(per_spectrum, Mapping):
        raise TypeError("source must be a PipelineResult or QCResult")
    missing = [name for name in REQUIRED_QC_FIELDS if name not in per_spectrum]
    if missing:
        raise ValueError("QC result is missing required fields: " + ", ".join(missing))
    columns: dict[str, np.ndarray] = {}
    row_count: int | None = None
    for name, raw_values in per_spectrum.items():
        values = np.asarray(raw_values)
        if values.ndim != 1:
            raise ValueError(f"QC field {name!r} must be one-dimensional")
        if row_count is None:
            row_count = values.size
        elif values.size != row_count:
            raise ValueError("all per-spectrum QC fields must have the same length")
        columns[str(name)] = values.copy()
    assert row_count is not None
    if perturbation_labels is None and hasattr(source, "absorbance_selected"):
        perturbation_labels = source.absorbance_selected.perturbation_labels
    if perturbation_labels is None:
        raise ValueError("perturbation_labels are required when source is a QCResult")
    labels = tuple(str(item) for item in perturbation_labels)
    if len(labels) != row_count:
        raise ValueError("perturbation labels must align with per-spectrum QC rows")
    frame = pd.DataFrame(columns)
    insert_at = list(frame.columns).index("perturbation") + 1
    frame.insert(insert_at, "perturbation_label", labels)
    return frame


def filter_qc_table(
    table: pd.DataFrame,
    *,
    query: str = "",
    spectrum_indices: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Filter a detached display table without deleting spectra from the result."""

    filtered = table.copy(deep=True)
    if spectrum_indices is not None:
        wanted = {int(item) for item in spectrum_indices}
        filtered = filtered[filtered["spectrum_index"].astype(int).isin(wanted)]
    normalized_query = str(query).strip().casefold()
    if normalized_query:
        label_match = filtered["perturbation_label"].astype(str).str.casefold().str.contains(
            normalized_query,
            regex=False,
        )
        index_match = filtered["spectrum_index"].astype(str).str.casefold().str.contains(
            normalized_query,
            regex=False,
        )
        filtered = filtered[label_match | index_match]
    return filtered.reset_index(drop=True)


def qc_table_csv(table: pd.DataFrame) -> bytes:
    """Serialize exactly the currently displayed QC table for download."""

    return table.to_csv(index=False).encode("utf-8")


def trend_figure(
    perturbation: Any,
    metrics: Mapping[str, Any],
    *,
    title: str,
) -> go.Figure:
    """Plot existing QC arrays in their preserved spectrum order."""

    axis = np.asarray(perturbation, dtype=np.float64)
    if axis.ndim != 1:
        raise ValueError("perturbation must be one-dimensional")
    figure = go.Figure()
    for label, raw_values in metrics.items():
        values = np.asarray(raw_values, dtype=np.float64)
        if values.ndim != 1 or values.shape != axis.shape:
            raise ValueError(f"trend {label!r} must align with perturbation")
        figure.add_trace(
            go.Scatter(
                x=axis,
                y=values,
                mode="lines+markers",
                name=label,
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="Perturbation (preserved order)",
        yaxis_title="Diagnostic value",
        height=330,
        margin={"l": 45, "r": 20, "t": 50, "b": 40},
    )
    return figure


def qc_trend_figures(source: Any) -> dict[str, go.Figure]:
    """Return the three dimension-separated groups required by the specification."""

    qc = getattr(source, "qc", source)
    metrics = qc.per_spectrum
    perturbation = metrics["perturbation"]
    groups = (
        (
            "Baseline continuity",
            {
                "Baseline area": metrics["baseline_area"],
                "Adjacent baseline RMS": metrics["adjacent_baseline_rms"],
            },
        ),
        (
            "Residual diagnostics",
            {
                "Anchor error": metrics["anchor_error"],
                "Negative fraction": metrics["negative_fraction"],
            },
        ),
        (
            "Peak preservation",
            {
                "Derivative correlation": metrics["derivative_correlation"],
                "Peak position shift": metrics["peak_position_shift"],
                "Peak height relative change": metrics["peak_height_relative_change"],
            },
        ),
    )
    return {
        title: trend_figure(perturbation, values, title=title)
        for title, values in groups
    }


def drill_down_payload(
    result: Any,
    selection: RepresentativeSelection | str | int,
) -> tuple[BaselinePreviewPayload, pd.Series]:
    """Return one actual spectrum's decomposition and exactly aligned QC row."""

    resolved = resolve_representative(
        result.absorbance_selected.perturbation_labels,
        selection,
    )
    if resolved.spectrum_index is None:
        raise ValueError("single-spectrum drill-down requires an actual spectrum row")
    payload = baseline_preview_payload(result, resolved)
    row = complete_qc_table(result).iloc[resolved.spectrum_index].copy()
    return payload, row


def drill_down_figure(
    result: Any,
    selection: RepresentativeSelection | str | int,
    *,
    title: str | None = None,
) -> go.Figure:
    """Plot Raw/Coarse/Fine/Total/Corrected for one actual spectrum row."""

    resolved = resolve_representative(
        result.absorbance_selected.perturbation_labels,
        selection,
    )
    if resolved.spectrum_index is None:
        raise ValueError("single-spectrum drill-down requires an actual spectrum row")
    figure = fine_decomposition_figure(
        result,
        resolved,
        title=title or f"Single-spectrum drill-down · {resolved.label}",
    )
    keep = {"A_raw", "B_coarse", "B_fine", "B_total", "Corrected"}
    figure.data = tuple(trace for trace in figure.data if trace.name in keep)
    return figure


__all__ = [
    "REQUIRED_QC_FIELDS",
    "complete_qc_table",
    "drill_down_figure",
    "drill_down_payload",
    "filter_qc_table",
    "five_heatmap_figures",
    "qc_table_csv",
    "qc_trend_figures",
    "series_heatmap",
    "trend_figure",
]
