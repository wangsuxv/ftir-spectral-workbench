"""Pure display helpers for an existing post-baseline smoothing result.

This module never runs the smoothing core.  It only selects or aggregates rows
that already exist in a ``PreparedSpectralDataset`` and its matching immutable
``PostBaselineSmoothingResult``, then creates detached UI payloads.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from ftir_workbench.models import PreparedSpectralDataset
from ftir_workbench.post_baseline_smoothing import PostBaselineSmoothingResult

from .baseline_preview import (
    RepresentativeSelection,
    representative_options,
    representative_values,
    resolve_representative,
)

FloatArray = NDArray[np.float64]
RepresentativeInput = RepresentativeSelection | str | int


def _readonly_vector(values: ArrayLike, *, name: str) -> FloatArray:
    try:
        source = np.asarray(values, dtype=np.float64, order="C")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if source.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(source)):
        raise ValueError(f"{name} must contain only finite values")
    return np.frombuffer(source.tobytes(order="C"), dtype=np.float64)


def _validate_parent_result(
    parent: PreparedSpectralDataset,
    result: PostBaselineSmoothingResult,
) -> None:
    if not isinstance(parent, PreparedSpectralDataset):
        raise TypeError("parent must be a PreparedSpectralDataset")
    if not isinstance(result, PostBaselineSmoothingResult):
        raise TypeError("result must be a PostBaselineSmoothingResult")
    result_parent = result.parent_prepared
    if (
        result_parent.prepared_data_sha256 != parent.prepared_data_sha256
        or result_parent.baseline_run_id != parent.baseline_run_id
        or result_parent.baseline_fingerprint != parent.baseline_fingerprint
    ):
        raise ValueError("smoothing result does not belong to the supplied parent Prepared")
    if result.smoothed_spectra.shape != parent.spectra.shape:
        raise ValueError("smoothing result shape does not match the supplied parent Prepared")


def smoothing_representative_options(
    parent: PreparedSpectralDataset,
) -> tuple[RepresentativeSelection, ...]:
    """Return stable first/middle/last/mean/median and concrete-row choices."""

    if not isinstance(parent, PreparedSpectralDataset):
        raise TypeError("parent must be a PreparedSpectralDataset")
    return representative_options(parent.perturbation_labels)


def resolve_smoothing_representative(
    parent: PreparedSpectralDataset,
    selection: RepresentativeInput,
) -> RepresentativeSelection:
    """Resolve a shortcut, exact label, ``spectrum:<index>``, or integer index."""

    if not isinstance(parent, PreparedSpectralDataset):
        raise TypeError("parent must be a PreparedSpectralDataset")
    return resolve_representative(parent.perturbation_labels, selection)


@dataclass(frozen=True, slots=True)
class SmoothingPreviewPayload:
    """Detached curves for one actual spectrum or one display-only aggregate."""

    selection: RepresentativeSelection
    wavenumber: FloatArray
    unsmoothed: FloatArray
    smoothed: FloatArray
    removed_component: FloatArray

    def __post_init__(self) -> None:
        vectors = {
            "wavenumber": self.wavenumber,
            "unsmoothed": self.unsmoothed,
            "smoothed": self.smoothed,
            "removed_component": self.removed_component,
        }
        copied = {name: _readonly_vector(value, name=name) for name, value in vectors.items()}
        size = copied["wavenumber"].size
        if any(value.size != size for value in copied.values()):
            raise ValueError("all smoothing-preview curves must match the wavenumber axis")
        for name, value in copied.items():
            object.__setattr__(self, name, value)


def smoothing_preview_payload(
    parent: PreparedSpectralDataset,
    result: PostBaselineSmoothingResult,
    selection: RepresentativeInput,
) -> SmoothingPreviewPayload:
    """Select existing parent/result rows, aggregating only along spectra axis 0."""

    _validate_parent_result(parent, result)
    resolved = resolve_smoothing_representative(parent, selection)
    return SmoothingPreviewPayload(
        selection=resolved,
        wavenumber=parent.wavenumber,
        unsmoothed=representative_values(
            parent.spectra,
            resolved,
            name="unsmoothed",
        ),
        smoothed=representative_values(
            result.smoothed_spectra,
            resolved,
            name="smoothed",
        ),
        removed_component=representative_values(
            result.removed_component,
            resolved,
            name="removed_component",
        ),
    )


def _style_spectral_axes(axes: Axes, *, title: str, ylabel: str) -> None:
    axes.set_title(title)
    axes.set_xlabel("Wavenumber (cm⁻¹)")
    axes.set_ylabel(ylabel)
    axes.grid(True, alpha=0.2)
    axes.invert_xaxis()


def smoothing_overlay_figure(
    parent: PreparedSpectralDataset,
    result: PostBaselineSmoothingResult,
    selection: RepresentativeInput,
    *,
    title: str = "Unsmoothed vs smoothed corrected absorbance",
) -> Figure:
    """Plot an existing unsmoothed/smoothed representative without recomputation."""

    payload = smoothing_preview_payload(parent, result, selection)
    figure = Figure(figsize=(8.4, 4.8), constrained_layout=True)
    axes = figure.subplots()
    axes.plot(
        payload.wavenumber,
        payload.unsmoothed,
        label="Unsmoothed corrected absorbance",
        linewidth=1.5,
    )
    axes.plot(
        payload.wavenumber,
        payload.smoothed,
        label="Smoothed corrected absorbance",
        linewidth=1.5,
    )
    _style_spectral_axes(
        axes,
        title=f"{title} · {payload.selection.label}",
        ylabel="Absorbance",
    )
    axes.legend()
    return figure


def smoothing_removed_component_figure(
    parent: PreparedSpectralDataset,
    result: PostBaselineSmoothingResult,
    selection: RepresentativeInput,
    *,
    title: str = "Removed component",
) -> Figure:
    """Plot the already stored ``unsmoothed - smoothed`` component and zero line."""

    payload = smoothing_preview_payload(parent, result, selection)
    figure = Figure(figsize=(8.4, 4.2), constrained_layout=True)
    axes = figure.subplots()
    axes.plot(
        payload.wavenumber,
        payload.removed_component,
        label="Removed component",
        linewidth=1.35,
    )
    axes.axhline(0.0, color="black", linestyle="--", linewidth=0.9, label="Zero reference")
    _style_spectral_axes(
        axes,
        title=f"{title} · {payload.selection.label}",
        ylabel="Absorbance difference",
    )
    axes.legend()
    return figure


def smoothing_qc_table(
    parent: PreparedSpectralDataset,
    result: PostBaselineSmoothingResult,
) -> pd.DataFrame:
    """Return a detached all-spectrum table using only stored smoothing metrics."""

    _validate_parent_result(parent, result)
    columns: dict[str, object] = {
        "spectrum_index": np.arange(parent.n_spectra, dtype=np.int64),
        "perturbation": np.array(parent.perturbation, dtype=np.float64, copy=True),
        "perturbation_label": list(parent.perturbation_labels),
    }
    for name, values in result.per_spectrum_metrics.items():
        columns[str(name)] = np.array(values, dtype=np.float64, copy=True)
    return pd.DataFrame(columns)


def smoothing_summary_payload(
    parent: PreparedSpectralDataset,
    result: PostBaselineSmoothingResult,
) -> dict[str, object]:
    """Return detached configuration, axis, lineage, QC-summary, and warning data."""

    _validate_parent_result(parent, result)
    scientific_config = result.config.scientific_dict()
    parameters = scientific_config.get("parameters", {})
    if not isinstance(parameters, Mapping):  # pragma: no cover - config guards this
        parameters = {}
    return {
        "parent_prepared_data_sha256": parent.prepared_data_sha256,
        "smoothing_fingerprint": result.smoothing_fingerprint,
        "enabled": result.config.enabled,
        "method": result.config.method,
        "effective_parameters": deepcopy(dict(parameters)),
        "scientific_config": deepcopy(scientific_config),
        "median_wavenumber_spacing": float(result.median_wavenumber_spacing),
        "spacing_relative_max_deviation": float(
            result.spacing_relative_max_deviation
        ),
        "approximate_physical_width": dict(result.approximate_physical_width),
        "summary_metrics": dict(result.summary_metrics),
        "warnings": list(result.warnings),
    }


__all__ = [
    "RepresentativeInput",
    "RepresentativeSelection",
    "SmoothingPreviewPayload",
    "resolve_smoothing_representative",
    "smoothing_overlay_figure",
    "smoothing_preview_payload",
    "smoothing_qc_table",
    "smoothing_removed_component_figure",
    "smoothing_representative_options",
    "smoothing_summary_payload",
]
