"""Fine-baseline dispatch and coarse-plus-fine composition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..models import BaselineResult
from .anchors import multipoint_linear_baseline, pchip_baseline, polynomial_baseline
from .endpoint import endpoint_window_baseline, strict_endpoint_baseline
from .protocol import BaselineEstimator


def _none_fine_result(x: np.ndarray, residual_spectra: np.ndarray) -> BaselineResult:
    # Importing here avoids exposing internal result builders as public API.
    from .protocol import make_result, validate_xy

    _, spectra_2d, was_1d = validate_xy(x, residual_spectra)
    return make_result(
        spectra_2d,
        np.zeros_like(spectra_2d),
        component="fine",
        was_1d=was_1d,
        params={"method": "none", "enabled": False, "series_recipe_locked": True},
    )


def _pop_anchors(params: dict[str, Any]) -> Sequence[Any]:
    anchors = params.pop("anchors", params.pop("anchor_windows", None))
    if anchors is None:
        raise ValueError("fixed anchor windows are required for this fine baseline method")
    if isinstance(anchors, (str, bytes)) or not isinstance(anchors, Sequence):
        raise TypeError("anchors must be a sequence of fixed window definitions")
    return anchors


def estimate_fine(
    x: np.ndarray,
    residual_spectra: np.ndarray,
    method: str = "endpoint_window_linear",
    **params: Any,
) -> BaselineResult:
    """Uniform fine-baseline dispatch.

    Fine estimators always operate on the residual after coarse correction.
    Their returned ``fine_baseline`` and ``total_baseline`` therefore describe
    only the additional fine component, ready for :func:`compose_baselines`.
    """

    normalized = str(method).strip().lower().replace("-", "_")
    aliases = {
        "endpoint": "endpoint_window_linear",
        "endpoint_linear": "endpoint_window_linear",
        "linear_interpolation": "multipoint_linear",
        "piecewise_linear": "multipoint_linear",
        "anchor_pchip": "pchip",
        "poly": "polynomial",
        "off": "none",
        "disabled": "none",
    }
    normalized = aliases.get(normalized, normalized)
    options = dict(params)
    enabled = bool(options.pop("enabled", True))
    if not enabled or normalized == "none":
        return _none_fine_result(x, residual_spectra)

    if normalized == "endpoint_window_linear":
        strict = bool(options.pop("strict_endpoint", False))
        if strict:
            if options:
                # Endpoint width/statistic are irrelevant in strict mode; omit
                # the common config fields without pretending they were used.
                options.pop("endpoint_window_width_cm1", None)
                options.pop("window_width_cm1", None)
                options.pop("window_width", None)
                options.pop("statistic", None)
            if options:
                raise ValueError(f"unexpected strict endpoint parameters: {sorted(options)}")
            return strict_endpoint_baseline(x, residual_spectra)
        width = options.pop(
            "endpoint_window_width_cm1",
            options.pop("window_width_cm1", options.pop("window_width", 8.0)),
        )
        statistic = options.pop("statistic", "median")
        if options:
            raise ValueError(f"unexpected endpoint-window parameters: {sorted(options)}")
        return endpoint_window_baseline(
            x,
            residual_spectra,
            endpoint_window_width_cm1=float(width),
            statistic=str(statistic),
        )
    if normalized == "strict_endpoint":
        # Accept but ignore config-only fields that do not affect strict mode.
        for key in (
            "strict_endpoint",
            "endpoint_window_width_cm1",
            "window_width_cm1",
            "window_width",
            "statistic",
        ):
            options.pop(key, None)
        if options:
            raise ValueError(f"unexpected strict endpoint parameters: {sorted(options)}")
        return strict_endpoint_baseline(x, residual_spectra)
    if normalized == "multipoint_linear":
        anchors = _pop_anchors(options)
        statistic = options.pop("statistic", "median")
        if options:
            raise ValueError(f"unexpected multipoint-linear parameters: {sorted(options)}")
        return multipoint_linear_baseline(x, residual_spectra, anchors, statistic=str(statistic))
    if normalized == "pchip":
        anchors = _pop_anchors(options)
        statistic = options.pop("statistic", "median")
        if options:
            raise ValueError(f"unexpected PCHIP parameters: {sorted(options)}")
        return pchip_baseline(x, residual_spectra, anchors, statistic=str(statistic))

    order_from_name: int | None = None
    for prefix in ("poly", "polynomial"):
        suffix = normalized.removeprefix(prefix).lstrip("_")
        if normalized.startswith(prefix) and suffix in {"1", "2", "3"}:
            order_from_name = int(suffix)
            normalized = "polynomial"
            break
    if normalized == "polynomial":
        anchors = _pop_anchors(options)
        statistic = options.pop("statistic", "median")
        configured_order = options.pop("polynomial_order", options.pop("order", 1))
        order = order_from_name if order_from_name is not None else int(configured_order)
        if order_from_name is not None and int(configured_order) not in {1, order_from_name}:
            raise ValueError(
                f"polynomial order in method name ({order_from_name}) conflicts with configured "
                f"order ({configured_order})"
            )
        if options:
            raise ValueError(f"unexpected polynomial parameters: {sorted(options)}")
        return polynomial_baseline(
            x,
            residual_spectra,
            anchors,
            order=order,
            statistic=str(statistic),
        )

    supported = (
        "none, endpoint_window_linear, strict_endpoint, multipoint_linear, "
        "pchip, polynomial (orders 1-3)"
    )
    raise ValueError(f"unsupported fine baseline method {method!r}; supported methods: {supported}")


def compose_baselines(
    raw: np.ndarray,
    coarse_result: BaselineResult,
    fine_result: BaselineResult,
) -> BaselineResult:
    """Combine sequential coarse and fine results with exact reconstruction."""

    raw_array = np.asarray(raw, dtype=np.float64)
    if raw_array.ndim not in {1, 2}:
        raise ValueError("raw spectra must be one- or two-dimensional")
    if not np.all(np.isfinite(raw_array)):
        bad = np.argwhere(~np.isfinite(raw_array)).tolist()
        raise ValueError(f"raw spectra contain NaN or Inf at indices {bad}")

    coarse = np.asarray(coarse_result.total_baseline, dtype=np.float64)
    fine = np.asarray(fine_result.total_baseline, dtype=np.float64)
    if coarse.shape != raw_array.shape:
        raise ValueError(
            f"coarse baseline shape {coarse.shape} does not match raw shape {raw_array.shape}"
        )
    if fine.shape != raw_array.shape:
        raise ValueError(
            f"fine baseline shape {fine.shape} does not match raw shape {raw_array.shape}"
        )
    if not np.all(np.isfinite(coarse)) or not np.all(np.isfinite(fine)):
        raise ValueError("coarse and fine baselines must contain only finite values")

    total = coarse + fine
    corrected = raw_array - total
    reconstruction_error = float(np.max(np.abs(raw_array - (total + corrected)), initial=0.0))
    warnings = tuple(dict.fromkeys((*coarse_result.warnings, *fine_result.warnings)))
    return BaselineResult(
        coarse_baseline=coarse.copy(),
        fine_baseline=fine.copy(),
        total_baseline=total,
        corrected=corrected,
        params={
            "method": "composite",
            "coarse": dict(coarse_result.params),
            "fine": dict(fine_result.params),
        },
        metrics={
            "coarse": dict(coarse_result.metrics),
            "fine": dict(fine_result.metrics),
            "reconstruction_max_abs_error": reconstruction_error,
        },
        warnings=warnings,
    )


composite_baseline = compose_baselines


@dataclass(frozen=True)
class CompositeBaselineEstimator:
    """Run reusable coarse and fine estimators sequentially."""

    coarse_estimator: BaselineEstimator
    fine_estimator: BaselineEstimator

    def fit_transform(self, x: np.ndarray, spectra: np.ndarray) -> BaselineResult:
        coarse = self.coarse_estimator.fit_transform(x, spectra)
        fine = self.fine_estimator.fit_transform(x, coarse.corrected)
        return compose_baselines(spectra, coarse, fine)
