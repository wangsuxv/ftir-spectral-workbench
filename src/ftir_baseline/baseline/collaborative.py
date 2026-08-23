"""Collaborative penalized least-squares baseline adapter."""

from __future__ import annotations

import warnings as python_warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..models import BaselineResult
from .automatic import (
    PYBASELINES_METHODS,
    _load_baseline_class,
    _normalize_pls_parameters,
    _normalized_method,
    _recipe_parameters,
)
from .protocol import (
    ascending_view,
    make_result,
    restore_axis,
    serializable_value,
    validate_xy,
)


def _collaborative_convergence_warnings(
    metadata: dict[str, Any], *, method: str, tol: float, max_iter: int
) -> list[str]:
    result: list[str] = []
    method_params = metadata.get("method_params", {})
    if not isinstance(method_params, dict):
        return result
    histories = method_params.get("tol_history", [])
    if isinstance(histories, np.ndarray) and histories.ndim == 1:
        histories = [histories]
    for index, history in enumerate(histories):
        values = np.asarray(history, dtype=np.float64)
        # collab_pls deliberately performs each final shared-weight fit once
        # (effectively with an infinite stopping tolerance), so its returned
        # one-element history is not evidence of non-convergence.  Longer
        # histories retain the ordinary method semantics.
        if values.size > 1 and (not np.isfinite(values[-1]) or values[-1] > tol):
            result.append(
                f"collab_pls({method}) spectrum {index} did not converge within "
                f"max_iter={max_iter} (last tolerance {values[-1]:.6g})"
            )
    return result


def collaborative_pls_baseline(
    x: np.ndarray,
    spectra: np.ndarray,
    method: str = "arpls",
    *,
    average_dataset: bool = True,
    method_kwargs: dict[str, Any] | None = None,
    **params: Any,
) -> BaselineResult:
    """Fit a series with shared PLS weights using ``Baseline.collab_pls``.

    The row order is never changed; only a descending wavenumber axis is
    reversed temporarily for the third-party call.
    """

    normalized_method = _normalized_method(method)
    if normalized_method not in PYBASELINES_METHODS:
        supported = ", ".join(sorted(PYBASELINES_METHODS))
        raise ValueError(f"collab_pls underlying method must be one of {supported}; got {method!r}")
    if method_kwargs is not None and params:
        duplicate = set(method_kwargs).intersection(params)
        if duplicate:
            raise ValueError(f"duplicate collaborative method parameters: {sorted(duplicate)}")
    combined_kwargs = dict(method_kwargs or {})
    combined_kwargs.update(params)
    normalized_kwargs = _normalize_pls_parameters(normalized_method, combined_kwargs)

    x_array, spectra_2d, was_1d = validate_xy(x, spectra)
    x_ascending, spectra_ascending, reverse = ascending_view(x_array, spectra_2d)
    Baseline = _load_baseline_class()
    fitter = Baseline(x_data=x_ascending, check_finite=True, assume_sorted=True)
    collaborative_method = getattr(fitter, "collab_pls", None)
    if collaborative_method is None:
        raise RuntimeError(
            "installed pybaselines does not provide Baseline.collab_pls(); "
            "upgrade to pybaselines>=1.2"
        )

    with python_warnings.catch_warnings(record=True) as caught:
        python_warnings.simplefilter("always")
        estimated, metadata = collaborative_method(
            spectra_ascending,
            average_dataset=bool(average_dataset),
            method=normalized_method,
            method_kwargs=normalized_kwargs,
        )
    baseline_ascending = np.asarray(estimated, dtype=np.float64)
    if baseline_ascending.shape != spectra_ascending.shape:
        raise RuntimeError(
            f"pybaselines collab_pls returned shape {baseline_ascending.shape}; "
            f"expected {spectra_ascending.shape}"
        )
    if not np.all(np.isfinite(baseline_ascending)):
        raise RuntimeError("pybaselines collab_pls returned a non-finite baseline")

    metadata_dict = dict(metadata or {})
    result_warnings = [f"collab_pls({normalized_method}): {item.message}" for item in caught]
    result_warnings.extend(
        _collaborative_convergence_warnings(
            metadata_dict,
            method=normalized_method,
            tol=float(normalized_kwargs["tol"]),
            max_iter=int(normalized_kwargs["max_iter"]),
        )
    )
    baseline = restore_axis(baseline_ascending, reverse)
    flat_recipe = _recipe_parameters(normalized_method, normalized_kwargs)
    return make_result(
        spectra_2d,
        baseline,
        component="coarse",
        was_1d=was_1d,
        params={
            "method": "collab_pls",
            "underlying_method": normalized_method,
            "average_dataset": bool(average_dataset),
            "method_kwargs": flat_recipe,
            **{key: value for key, value in flat_recipe.items() if key != "method"},
            "series_recipe_locked": True,
        },
        metrics={"pybaselines_fit_metadata": serializable_value(metadata_dict)},
        warnings=result_warnings,
    )


# Short aliases align with pybaselines and the configuration vocabulary.
collab_pls = collaborative_pls_baseline


def _tag_series_result(result: BaselineResult, series_mode: str) -> BaselineResult:
    """Return an equivalent result with explicit series recipe metadata."""

    return BaselineResult(
        coarse_baseline=np.asarray(result.coarse_baseline, dtype=np.float64).copy(),
        fine_baseline=np.asarray(result.fine_baseline, dtype=np.float64).copy(),
        total_baseline=np.asarray(result.total_baseline, dtype=np.float64).copy(),
        corrected=np.asarray(result.corrected, dtype=np.float64).copy(),
        params={
            **dict(result.params),
            "series_mode": series_mode,
            "series_recipe_locked": True,
            "spectrum_order_preserved": True,
        },
        metrics=dict(result.metrics),
        warnings=tuple(result.warnings),
    )


def estimate_series_baseline(
    x: np.ndarray,
    spectra: np.ndarray,
    method: str = "arpls",
    series_mode: str = "independent_locked",
    **params: Any,
) -> BaselineResult:
    """Uniform dispatch for the three series-aware baseline modes."""

    normalized_mode = str(series_mode).strip().lower().replace("-", "_")
    mode_aliases = {
        "independent": "independent_locked",
        "locked": "independent_locked",
        "collaborative": "collaborative_pls",
        "collab_pls": "collaborative_pls",
        "shared": "shared_shape",
    }
    normalized_mode = mode_aliases.get(normalized_mode, normalized_mode)
    if normalized_mode == "independent_locked":
        from .automatic import estimate_coarse

        result = estimate_coarse(x, spectra, method, **params)
    elif normalized_mode == "collaborative_pls":
        underlying_method = params.pop("underlying_method", method)
        result = collaborative_pls_baseline(x, spectra, str(underlying_method), **params)
    elif normalized_mode == "shared_shape":
        from .shared_shape import shared_shape_baseline

        anchors = params.pop("anchors", params.pop("anchor_windows", None))
        if anchors is None:
            raise ValueError("shared_shape series mode requires fixed anchor windows")
        result = shared_shape_baseline(
            x,
            spectra,
            method,
            anchors=anchors,
            **params,
        )
    else:
        raise ValueError(
            f"unsupported series_mode {series_mode!r}; use independent_locked, "
            "collaborative_pls, or shared_shape"
        )
    return _tag_series_result(result, normalized_mode)


@dataclass(frozen=True)
class CollaborativePLSEstimator:
    """Reusable collaborative PLS recipe."""

    method: str = "arpls"
    average_dataset: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def fit_transform(self, x: np.ndarray, spectra: np.ndarray) -> BaselineResult:
        return collaborative_pls_baseline(
            x,
            spectra,
            self.method,
            average_dataset=self.average_dataset,
            **self.params,
        )
