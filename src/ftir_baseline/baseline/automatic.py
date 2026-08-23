"""Coarse baseline estimators and the thin :mod:`pybaselines` adapter.

The iterative PLS algorithms in this module are delegated to pybaselines.  No
local reimplementation or silent fallback is used when that dependency is not
available.  Offset, least-squares linear detrending, and the geometric
rubberband baseline are intentionally small, transparent native estimators.
"""

from __future__ import annotations

import warnings as python_warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..models import BaselineResult
from .protocol import (
    ascending_view,
    make_result,
    restore_axis,
    serializable_value,
    validate_xy,
)

PYBASELINES_METHODS = frozenset({"arpls", "asls", "airpls", "pspline_arpls"})
COARSE_METHODS = frozenset({"none", "offset", "linear", "rubberband", *PYBASELINES_METHODS})


class PybaselinesUnavailableError(ImportError):
    """Raised when an explicitly requested automatic method cannot be loaded."""


def _load_baseline_class() -> type[Any]:
    try:
        from pybaselines import Baseline
    except ImportError as exc:  # pragma: no cover - environment-dependent branch
        raise PybaselinesUnavailableError(
            "automatic baseline method requires the optional runtime dependency "
            "'pybaselines>=1.2'; install the project dependencies (for example, "
            "`python -m pip install pybaselines>=1.2`) and retry"
        ) from exc
    return Baseline


def _normalized_method(method: str) -> str:
    normalized = str(method).strip().lower().replace("-", "_")
    aliases = {
        "linear_detrend": "linear",
        "detrend": "linear",
        "ar_pls": "arpls",
        "as_ls": "asls",
        "air_pls": "airpls",
        "pspline_ar_pls": "pspline_arpls",
        "off": "none",
        "disabled": "none",
    }
    return aliases.get(normalized, normalized)


def _normalize_pls_parameters(method: str, params: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(params)
    if "lambda" in kwargs:
        if "lam" in kwargs:
            raise ValueError("specify only one of 'lambda' and 'lam'")
        kwargs["lam"] = kwargs.pop("lambda")
    if "lambda_" in kwargs:
        if "lam" in kwargs:
            raise ValueError("specify only one lambda parameter")
        kwargs["lam"] = kwargs.pop("lambda_")

    defaults: dict[str, Any] = {
        "lam": 1_000_000.0,
        "diff_order": 2,
        "max_iter": 50,
        "tol": 1e-3,
    }
    if method == "asls":
        defaults["p"] = 0.01
    if method == "pspline_arpls":
        defaults.update({"num_knots": 100, "spline_degree": 3})
    for key, value in defaults.items():
        kwargs.setdefault(key, value)

    lam = float(kwargs["lam"])
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lambda/lam must be a finite positive number")
    kwargs["lam"] = lam
    max_iter = int(kwargs["max_iter"])
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    kwargs["max_iter"] = max_iter
    tol = float(kwargs["tol"])
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and non-negative")
    kwargs["tol"] = tol
    if method == "asls":
        p = float(kwargs["p"])
        if not np.isfinite(p) or not 0 < p < 1:
            raise ValueError("AsLS p must be strictly between 0 and 1")
        kwargs["p"] = p
    return kwargs


def _recipe_parameters(method: str, method_kwargs: dict[str, Any]) -> dict[str, Any]:
    recipe = {str(key): serializable_value(value) for key, value in method_kwargs.items()}
    # JSON recipes use the scientifically familiar spelling while retaining
    # ``lam`` for direct reproducibility against the pybaselines call.
    if "lam" in recipe:
        recipe["lambda"] = recipe["lam"]
    return {"method": method, **recipe}


def none_baseline(x: np.ndarray, spectra: np.ndarray) -> BaselineResult:
    """Return an explicit zero coarse baseline."""

    _, spectra_2d, was_1d = validate_xy(x, spectra)
    return make_result(
        spectra_2d,
        np.zeros_like(spectra_2d),
        component="coarse",
        was_1d=was_1d,
        params={"method": "none", "series_recipe_locked": True},
    )


def offset_baseline(
    x: np.ndarray,
    spectra: np.ndarray,
    *,
    statistic: str = "minimum",
    quantile: float = 0.0,
    offset: float | np.ndarray | None = None,
) -> BaselineResult:
    """Estimate one constant offset per spectrum.

    By default the minimum is used, matching the conventional transparent
    offset correction for non-negative absorbance peaks.  A caller may instead
    request ``median``, ``mean``, ``quantile``, or supply explicit offsets.
    """

    _, spectra_2d, was_1d = validate_xy(x, spectra)
    normalized_statistic = str(statistic).strip().lower()
    if offset is not None:
        offsets = np.asarray(offset, dtype=np.float64)
        if offsets.ndim == 0:
            offsets = np.repeat(offsets[None], spectra_2d.shape[0])
        if offsets.shape != (spectra_2d.shape[0],):
            raise ValueError("explicit offset must be scalar or have one value per spectrum")
        if not np.all(np.isfinite(offsets)):
            raise ValueError("explicit offsets must all be finite")
        source = "explicit"
    elif normalized_statistic in {"minimum", "min"}:
        offsets = np.min(spectra_2d, axis=1)
        source = "minimum"
    elif normalized_statistic == "median":
        offsets = np.median(spectra_2d, axis=1)
        source = "median"
    elif normalized_statistic == "mean":
        offsets = np.mean(spectra_2d, axis=1)
        source = "mean"
    elif normalized_statistic == "quantile":
        q = float(quantile)
        if not np.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("offset quantile must lie in [0, 1]")
        offsets = np.quantile(spectra_2d, q, axis=1)
        source = "quantile"
    else:
        raise ValueError("offset statistic must be minimum, median, mean, or quantile")
    baseline = np.broadcast_to(offsets[:, None], spectra_2d.shape).copy()
    return make_result(
        spectra_2d,
        baseline,
        component="coarse",
        was_1d=was_1d,
        params={
            "method": "offset",
            "statistic": source,
            **({"quantile": float(quantile)} if source == "quantile" else {}),
            "fitted_offsets": offsets.tolist(),
            "series_recipe_locked": True,
        },
    )


def linear_detrend_baseline(x: np.ndarray, spectra: np.ndarray) -> BaselineResult:
    """Fit an ordinary least-squares constant plus slope to each spectrum."""

    x_array, spectra_2d, was_1d = validate_xy(x, spectra)
    x_center = float(np.mean(x_array))
    scale = float(np.ptp(x_array) / 2.0)
    normalized_x = (x_array - x_center) / scale
    design = np.column_stack((np.ones_like(normalized_x), normalized_x))
    coefficients = np.linalg.lstsq(design, spectra_2d.T, rcond=None)[0]
    baseline = (design @ coefficients).T
    slopes = coefficients[1] / scale
    intercepts = coefficients[0] - slopes * x_center
    return make_result(
        spectra_2d,
        np.asarray(baseline, dtype=np.float64),
        component="coarse",
        was_1d=was_1d,
        params={
            "method": "linear",
            "fit": "ordinary_least_squares",
            "fitted_intercepts": intercepts.tolist(),
            "fitted_slopes": slopes.tolist(),
            "series_recipe_locked": True,
        },
    )


def _lower_hull_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    hull: list[int] = []
    for index in range(x.size):
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            cross = (x[second] - x[first]) * (y[index] - y[first]) - (y[second] - y[first]) * (
                x[index] - x[first]
            )
            if cross > 0:
                break
            hull.pop()
        hull.append(index)
    return np.asarray(hull, dtype=int)


def rubberband_baseline(x: np.ndarray, spectra: np.ndarray) -> BaselineResult:
    """Interpolate the lower convex hull (the geometric rubberband baseline)."""

    x_array, spectra_2d, was_1d = validate_xy(x, spectra)
    x_ascending, spectra_ascending, reverse = ascending_view(x_array, spectra_2d)
    baseline_ascending = np.empty_like(spectra_ascending)
    hull_indices: list[list[int]] = []
    hull_wavenumbers: list[list[float]] = []
    for index, row in enumerate(spectra_ascending):
        hull = _lower_hull_indices(x_ascending, row)
        baseline_ascending[index] = np.interp(x_ascending, x_ascending[hull], row[hull])
        hull_indices.append(hull.tolist())
        hull_wavenumbers.append(x_ascending[hull].tolist())
    baseline = restore_axis(baseline_ascending, reverse)
    return make_result(
        spectra_2d,
        baseline,
        component="coarse",
        was_1d=was_1d,
        params={
            "method": "rubberband",
            "implementation": "lower_convex_hull",
            "fitted_hull_indices_ascending": hull_indices,
            "fitted_hull_wavenumbers": hull_wavenumbers,
            "series_recipe_locked": True,
        },
    )


def pybaselines_baseline(
    x: np.ndarray,
    spectra: np.ndarray,
    method: str = "arpls",
    **params: Any,
) -> BaselineResult:
    """Apply one pybaselines algorithm independently with a locked recipe."""

    normalized_method = _normalized_method(method)
    if normalized_method not in PYBASELINES_METHODS:
        supported = ", ".join(sorted(PYBASELINES_METHODS))
        raise ValueError(
            f"unsupported pybaselines method {method!r}; supported methods: {supported}"
        )
    x_array, spectra_2d, was_1d = validate_xy(x, spectra)
    x_ascending, spectra_ascending, reverse = ascending_view(x_array, spectra_2d)
    method_kwargs = _normalize_pls_parameters(normalized_method, params)
    Baseline = _load_baseline_class()
    fitter = Baseline(x_data=x_ascending, check_finite=True, assume_sorted=True)
    fit_method = getattr(fitter, normalized_method, None)
    if fit_method is None:
        raise RuntimeError(
            f"installed pybaselines does not provide Baseline.{normalized_method}(); "
            "upgrade to pybaselines>=1.2"
        )

    baseline_ascending = np.empty_like(spectra_ascending)
    fit_metadata: list[dict[str, Any]] = []
    result_warnings: list[str] = []
    for spectrum_index, row in enumerate(spectra_ascending):
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always")
            estimated, metadata = fit_method(row, **method_kwargs)
        estimated_array = np.asarray(estimated, dtype=np.float64)
        if estimated_array.shape != row.shape:
            raise RuntimeError(
                f"pybaselines {normalized_method} returned shape {estimated_array.shape}; "
                f"expected {row.shape}"
            )
        if not np.all(np.isfinite(estimated_array)):
            raise RuntimeError(
                f"pybaselines {normalized_method} returned a non-finite baseline for "
                f"spectrum {spectrum_index}"
            )
        baseline_ascending[spectrum_index] = estimated_array
        metadata_dict = dict(metadata or {})
        fit_metadata.append(serializable_value(metadata_dict))
        for item in caught:
            result_warnings.append(f"{normalized_method} spectrum {spectrum_index}: {item.message}")
        tolerance_history = np.asarray(metadata_dict.get("tol_history", []), dtype=np.float64)
        if tolerance_history.size and (
            not np.isfinite(tolerance_history[-1])
            or tolerance_history[-1] > float(method_kwargs["tol"])
        ):
            result_warnings.append(
                f"{normalized_method} spectrum {spectrum_index} did not converge within "
                f"max_iter={method_kwargs['max_iter']} (last tolerance "
                f"{tolerance_history[-1]:.6g})"
            )

    baseline = restore_axis(baseline_ascending, reverse)
    return make_result(
        spectra_2d,
        baseline,
        component="coarse",
        was_1d=was_1d,
        params={
            **_recipe_parameters(normalized_method, method_kwargs),
            "series_recipe_locked": True,
        },
        metrics={"pybaselines_fit_metadata": fit_metadata},
        warnings=result_warnings,
    )


def estimate_coarse(
    x: np.ndarray,
    spectra: np.ndarray,
    method: str = "arpls",
    **params: Any,
) -> BaselineResult:
    """Uniform coarse-baseline dispatch used by the pipeline and CLI."""

    normalized_method = _normalized_method(method)
    if normalized_method == "none":
        if params:
            raise ValueError(f"none baseline does not accept parameters: {sorted(params)}")
        return none_baseline(x, spectra)
    if normalized_method == "offset":
        return offset_baseline(x, spectra, **params)
    if normalized_method == "linear":
        if params:
            raise ValueError(f"linear detrend does not accept parameters: {sorted(params)}")
        return linear_detrend_baseline(x, spectra)
    if normalized_method == "rubberband":
        if params:
            raise ValueError(f"rubberband does not accept parameters: {sorted(params)}")
        return rubberband_baseline(x, spectra)
    if normalized_method in PYBASELINES_METHODS:
        return pybaselines_baseline(x, spectra, normalized_method, **params)
    supported = ", ".join(sorted(COARSE_METHODS))
    raise ValueError(
        f"unsupported coarse baseline method {method!r}; supported methods: {supported}"
    )


# Method-specific convenience functions keep notebooks concise.
def arpls(x: np.ndarray, spectra: np.ndarray, **params: Any) -> BaselineResult:
    return pybaselines_baseline(x, spectra, "arpls", **params)


def asls(x: np.ndarray, spectra: np.ndarray, **params: Any) -> BaselineResult:
    return pybaselines_baseline(x, spectra, "asls", **params)


def airpls(x: np.ndarray, spectra: np.ndarray, **params: Any) -> BaselineResult:
    return pybaselines_baseline(x, spectra, "airpls", **params)


def pspline_arpls(x: np.ndarray, spectra: np.ndarray, **params: Any) -> BaselineResult:
    return pybaselines_baseline(x, spectra, "pspline_arpls", **params)


# Concise aliases mirror the method strings accepted by ``estimate_coarse``.
offset = offset_baseline
linear_detrend = linear_detrend_baseline
rubberband = rubberband_baseline


@dataclass(frozen=True)
class AutomaticBaselineEstimator:
    """Object-oriented coarse estimator with a reusable locked recipe."""

    method: str = "arpls"
    params: dict[str, Any] = field(default_factory=dict)

    def fit_transform(self, x: np.ndarray, spectra: np.ndarray) -> BaselineResult:
        return estimate_coarse(x, spectra, self.method, **self.params)
