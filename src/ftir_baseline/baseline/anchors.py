"""Fixed-window multi-anchor baselines.

Anchor locations are defined once and applied unchanged to every spectrum in a
series.  Only the representative intensities are calculated per spectrum.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.interpolate import PchipInterpolator

from ..models import BaselineResult
from .protocol import ascending_view, make_result, restore_axis, validate_xy

AnchorStatistic = Literal["median", "mean"]


@dataclass(frozen=True)
class AnchorWindow:
    """A fixed no-absorption window used to estimate a baseline anchor."""

    start: float
    end: float
    statistic: AnchorStatistic = "median"
    enabled: bool = True


def _representative(values: np.ndarray, statistic: str) -> float:
    normalized = statistic.strip().lower()
    if normalized == "median":
        return float(np.median(values))
    if normalized == "mean":
        return float(np.mean(values))
    raise ValueError(f"unsupported anchor statistic {statistic!r}; use 'median' or 'mean'")


def _coerce_window(item: Any, default_statistic: str) -> AnchorWindow:
    if isinstance(item, AnchorWindow):
        return item
    if isinstance(item, Mapping):
        try:
            return AnchorWindow(
                start=float(item["start"]),
                end=float(item["end"]),
                statistic=str(item.get("statistic", default_statistic)).lower(),  # type: ignore[arg-type]
                enabled=bool(item.get("enabled", True)),
            )
        except KeyError as exc:
            raise ValueError("each anchor mapping requires 'start' and 'end'") from exc
    if hasattr(item, "start") and hasattr(item, "end"):
        return AnchorWindow(
            start=float(item.start),
            end=float(item.end),
            statistic=str(getattr(item, "statistic", default_statistic)).lower(),  # type: ignore[arg-type]
            enabled=bool(getattr(item, "enabled", True)),
        )
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
        if len(item) == 2:
            return AnchorWindow(float(item[0]), float(item[1]), default_statistic)  # type: ignore[arg-type]
        if len(item) == 3:
            return AnchorWindow(float(item[0]), float(item[1]), str(item[2]).lower())  # type: ignore[arg-type]
    raise TypeError(
        "anchors must be AnchorWindow objects, mappings, or (start, end[, statistic]) tuples"
    )


def _prepare_windows(
    windows: Sequence[Any], default_statistic: str
) -> tuple[list[AnchorWindow], np.ndarray, list[tuple[float, float]]]:
    coerced = [_coerce_window(item, default_statistic) for item in windows]
    active = [window for window in coerced if window.enabled]
    if len(active) < 2:
        raise ValueError("at least two enabled anchor windows are required")

    bounds: list[tuple[float, float]] = []
    centers: list[float] = []
    for index, window in enumerate(active):
        start = float(window.start)
        end = float(window.end)
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError(f"anchor window {index} has a non-finite boundary")
        low, high = sorted((start, end))
        bounds.append((low, high))
        centers.append((low + high) / 2.0)
        # Validate even before a data window is evaluated so typos fail early.
        _representative(np.asarray([0.0]), str(window.statistic))

    center_differences = np.diff(np.asarray(centers, dtype=np.float64))
    if not (np.all(center_differences > 0) or np.all(center_differences < 0)):
        raise ValueError(
            "anchor windows must be supplied in strictly ascending or descending order"
        )

    order = np.argsort(np.asarray(centers, dtype=np.float64))
    active = [active[int(index)] for index in order]
    bounds = [bounds[int(index)] for index in order]
    sorted_centers = np.asarray([centers[int(index)] for index in order], dtype=np.float64)
    for index in range(len(bounds) - 1):
        if bounds[index][1] >= bounds[index + 1][0]:
            raise ValueError(
                f"anchor windows {index} and {index + 1} overlap or touch; use disjoint windows"
            )
    return active, sorted_centers, bounds


def _anchor_values(
    x_ascending: np.ndarray,
    spectra_ascending: np.ndarray,
    windows: list[AnchorWindow],
    bounds: list[tuple[float, float]],
) -> tuple[np.ndarray, list[int]]:
    values = np.empty((spectra_ascending.shape[0], len(windows)), dtype=np.float64)
    point_counts: list[int] = []
    scale = max(1.0, float(np.max(np.abs(x_ascending))))
    tolerance = np.finfo(np.float64).eps * scale * 8
    for anchor_index, (window, (low, high)) in enumerate(zip(windows, bounds, strict=True)):
        mask = (x_ascending >= low - tolerance) & (x_ascending <= high + tolerance)
        count = int(np.count_nonzero(mask))
        if count == 0:
            raise ValueError(
                f"anchor window {anchor_index} [{low:g}, {high:g}] cm^-1 contains no data points"
            )
        point_counts.append(count)
        for spectrum_index, row in enumerate(spectra_ascending):
            values[spectrum_index, anchor_index] = _representative(row[mask], str(window.statistic))
    return values, point_counts


def _require_no_extrapolation(x_ascending: np.ndarray, centers: np.ndarray) -> None:
    scale = max(1.0, float(np.max(np.abs(x_ascending))), float(np.max(np.abs(centers))))
    tolerance = np.finfo(np.float64).eps * scale * 16
    if x_ascending[0] < centers[0] - tolerance or x_ascending[-1] > centers[-1] + tolerance:
        raise ValueError(
            "anchor representative coordinates do not cover the analysis interval; "
            f"data span [{x_ascending[0]:g}, {x_ascending[-1]:g}] but anchors span "
            f"[{centers[0]:g}, {centers[-1]:g}]. Baseline extrapolation is forbidden."
        )


def _fit_anchor_baseline(
    x: np.ndarray,
    spectra: np.ndarray,
    windows: Sequence[Any],
    *,
    method: str,
    statistic: str,
    polynomial_order: int,
) -> BaselineResult:
    x_array, spectra_2d, was_1d = validate_xy(x, spectra)
    x_ascending, spectra_ascending, reverse = ascending_view(x_array, spectra_2d)
    active, centers, bounds = _prepare_windows(windows, statistic)
    _require_no_extrapolation(x_ascending, centers)
    values, point_counts = _anchor_values(x_ascending, spectra_ascending, active, bounds)

    normalized_method = method.strip().lower().replace("-", "_")
    if normalized_method in {"multipoint_linear", "piecewise_linear", "linear"}:
        recipe_method = "multipoint_linear"
        baseline_ascending = np.vstack(
            [np.interp(x_ascending, centers, row_values) for row_values in values]
        )
        order_used: int | None = None
    elif normalized_method in {"pchip", "anchor_pchip"}:
        recipe_method = "pchip"
        if centers.size == 2:
            baseline_ascending = np.vstack(
                [np.interp(x_ascending, centers, row_values) for row_values in values]
            )
        else:
            baseline_ascending = np.vstack(
                [
                    np.asarray(
                        PchipInterpolator(centers, row_values, extrapolate=False)(x_ascending),
                        dtype=np.float64,
                    )
                    for row_values in values
                ]
            )
        order_used = None
    elif normalized_method in {"polynomial", "poly"}:
        order_used = int(polynomial_order)
        if order_used not in {1, 2, 3}:
            raise ValueError("polynomial anchor baseline order must be 1, 2, or 3")
        if centers.size < order_used + 1:
            raise ValueError(
                f"polynomial order {order_used} requires at least {order_used + 1} anchors"
            )
        recipe_method = "polynomial"
        # Centering and scaling improve conditioning for FTIR axes around
        # 1000--4000 cm^-1 without changing the requested polynomial degree.
        x_center = float(np.mean(centers))
        x_scale = float(np.ptp(centers) / 2.0)
        normalized_centers = (centers - x_center) / x_scale
        normalized_x = (x_ascending - x_center) / x_scale
        baseline_ascending = np.vstack(
            [
                np.polynomial.polynomial.polyval(
                    normalized_x,
                    np.polynomial.polynomial.polyfit(
                        normalized_centers, row_values, deg=order_used
                    ),
                )
                for row_values in values
            ]
        )
    else:
        raise ValueError(
            f"unsupported anchor baseline method {method!r}; use multipoint_linear, pchip, "
            "or polynomial"
        )

    baseline = restore_axis(np.asarray(baseline_ascending, dtype=np.float64), reverse)
    window_recipe = [
        {
            "start": float(window.start),
            "end": float(window.end),
            "statistic": str(window.statistic),
            "enabled": True,
        }
        for window in active
    ]
    return make_result(
        spectra_2d,
        baseline,
        component="fine",
        was_1d=was_1d,
        params={
            "method": recipe_method,
            "anchors": window_recipe,
            "anchor_centers": centers.tolist(),
            "anchor_point_counts": point_counts,
            "anchor_values": values.tolist(),
            "series_recipe_locked": True,
            **({"polynomial_order": order_used} if order_used is not None else {}),
        },
    )


def multipoint_linear_baseline(
    x: np.ndarray,
    spectra: np.ndarray,
    anchors: Sequence[Any],
    *,
    statistic: AnchorStatistic | str = "median",
) -> BaselineResult:
    """Interpolate linearly through fixed-window representative values."""

    return _fit_anchor_baseline(
        x,
        spectra,
        anchors,
        method="multipoint_linear",
        statistic=str(statistic),
        polynomial_order=1,
    )


def pchip_baseline(
    x: np.ndarray,
    spectra: np.ndarray,
    anchors: Sequence[Any],
    *,
    statistic: AnchorStatistic | str = "median",
) -> BaselineResult:
    """Fit a shape-preserving PCHIP baseline; two anchors use a straight line."""

    return _fit_anchor_baseline(
        x,
        spectra,
        anchors,
        method="pchip",
        statistic=str(statistic),
        polynomial_order=1,
    )


def polynomial_baseline(
    x: np.ndarray,
    spectra: np.ndarray,
    anchors: Sequence[Any],
    *,
    order: int = 1,
    statistic: AnchorStatistic | str = "median",
) -> BaselineResult:
    """Fit a fixed-window low-order polynomial baseline (orders 1--3 only)."""

    return _fit_anchor_baseline(
        x,
        spectra,
        anchors,
        method="polynomial",
        statistic=str(statistic),
        polynomial_order=order,
    )


# Compatibility aliases with terminology used in the specification and UI.
piecewise_linear_baseline = multipoint_linear_baseline
anchor_pchip_baseline = pchip_baseline
multipoint_linear = multipoint_linear_baseline
piecewise_linear = multipoint_linear_baseline
pchip = pchip_baseline
polynomial = polynomial_baseline


@dataclass(frozen=True)
class AnchorBaselineEstimator:
    """Object-oriented wrapper for fixed-window anchor methods."""

    anchors: tuple[Any, ...]
    method: str = "pchip"
    statistic: AnchorStatistic = "median"
    polynomial_order: int = 1

    def fit_transform(self, x: np.ndarray, spectra: np.ndarray) -> BaselineResult:
        return _fit_anchor_baseline(
            x,
            spectra,
            self.anchors,
            method=self.method,
            statistic=self.statistic,
            polynomial_order=self.polynomial_order,
        )
