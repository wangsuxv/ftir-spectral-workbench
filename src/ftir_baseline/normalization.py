"""Post-baseline normalization with separate analysis and display branches.

The main analytical branch is deliberately never overwritten.  An explicitly
requested scientific normalization is returned in ``optional_normalized``;
min-max scaling is display-only and is returned only in ``view_data``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import freeze_value, immutable_float64

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class NormalizationResult:
    """Outputs of the non-destructive normalization split.

    ``analysis_data`` is always the baseline-corrected absorbance supplied by
    the caller.  ``optional_normalized`` contains an explicitly requested
    internal-reference, vector, or area normalization.  ``view_data`` is the
    branch intended for plotting and may additionally contain display-only
    min-max scaling.
    """

    analysis_data: FloatArray
    view_data: FloatArray
    optional_normalized: FloatArray | None
    factors: FloatArray
    method: str
    warnings: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        analysis = immutable_float64(self.analysis_data, name="normalization.analysis_data")
        view = immutable_float64(self.view_data, name="normalization.view_data")
        factors = immutable_float64(self.factors, name="normalization.factors")
        optional = (
            None
            if self.optional_normalized is None
            else immutable_float64(
                self.optional_normalized,
                name="normalization.optional_normalized",
            )
        )
        if analysis.shape != view.shape or (
            optional is not None and optional.shape != analysis.shape
        ):
            raise ValueError("normalization data branches must have the same shape")
        object.__setattr__(self, "analysis_data", analysis)
        object.__setattr__(self, "view_data", view)
        object.__setattr__(self, "optional_normalized", optional)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "method", str(self.method))
        object.__setattr__(self, "warnings", tuple(map(str, self.warnings)))
        object.__setattr__(
            self, "params", freeze_value(dict(self.params), path="normalization.params")
        )


_METHOD_ALIASES = {
    "": "none",
    "off": "none",
    "none": "none",
    "internal_height": "internal_peak_height",
    "internal_reference_height": "internal_peak_height",
    "internal_reference_peak_height": "internal_peak_height",
    "peak_height": "internal_peak_height",
    "internal_peak_height": "internal_peak_height",
    "internal_area": "internal_peak_area",
    "internal_reference_area": "internal_peak_area",
    "internal_reference_peak_area": "internal_peak_area",
    "peak_area": "internal_peak_area",
    "internal_peak_area": "internal_peak_area",
    "l2": "vector",
    "vector": "vector",
    "total_area": "area",
    "area": "area",
    "min-max": "minmax_display",
    "min_max": "minmax_display",
    "minmax": "minmax_display",
    "minmax_display_only": "minmax_display",
    "minmax_display": "minmax_display",
}


def _as_2d_finite(spectra: ArrayLike) -> tuple[FloatArray, bool]:
    values = np.asarray(spectra, dtype=np.float64)
    was_1d = values.ndim == 1
    if was_1d:
        values = values[np.newaxis, :]
    if values.ndim != 2:
        raise ValueError("spectra must be a one- or two-dimensional numeric array")
    if values.shape[1] < 1:
        raise ValueError("spectra must contain at least one wavenumber point")
    invalid = np.argwhere(~np.isfinite(values))
    if invalid.size:
        row, column = invalid[0]
        raise ValueError(f"spectra contain NaN or Inf at spectrum {int(row)}, point {int(column)}")
    return values.copy(), was_1d


def _as_axis(wavenumber: ArrayLike, n_points: int) -> FloatArray:
    x = np.asarray(wavenumber, dtype=np.float64)
    if x.ndim != 1 or x.size != n_points:
        raise ValueError(f"wavenumber must have shape ({n_points},)")
    if not np.all(np.isfinite(x)):
        raise ValueError("wavenumber contains NaN or Inf")
    if x.size > 1 and not (np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)):
        raise ValueError("wavenumber must be strictly monotonic")
    return x


def _interval_mask(
    x: FloatArray,
    interval: tuple[float, float] | list[float] | None,
    *,
    required: bool,
) -> FloatArray | NDArray[np.bool_]:
    if interval is None:
        if required:
            raise ValueError("an integration/reference interval is required for this method")
        return np.ones(x.size, dtype=bool)
    if len(interval) != 2:
        raise ValueError("interval must contain exactly two wavenumber bounds")
    lo, hi = sorted((float(interval[0]), float(interval[1])))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        raise ValueError("interval bounds must be finite and distinct")
    mask = (x >= lo) & (x <= hi)
    if np.count_nonzero(mask) < 2:
        raise ValueError(f"interval [{lo:g}, {hi:g}] contains fewer than two wavenumber points")
    return mask


def _integral(x: FloatArray, values: FloatArray) -> FloatArray:
    """Integrate rows with a positive orientation for either axis direction."""

    order = np.argsort(x)
    x_ordered = x[order]
    y_ordered = values[:, order]
    widths = np.diff(x_ordered)
    return np.asarray(
        np.sum(0.5 * (y_ordered[:, :-1] + y_ordered[:, 1:]) * widths, axis=1),
        dtype=np.float64,
    )


def _restore_dimension(values: FloatArray, was_1d: bool) -> FloatArray:
    return values[0].copy() if was_1d else values.copy()


def _canonical_method(method: str) -> str:
    key = str(method).strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return _METHOD_ALIASES[key]
    except KeyError as exc:
        supported = sorted(set(_METHOD_ALIASES.values()))
        raise ValueError(
            f"unknown normalization method {method!r}; choose one of {supported}"
        ) from exc


def normalize_spectra(
    wavenumber: ArrayLike,
    spectra: ArrayLike,
    method: str = "none",
    *,
    interval: tuple[float, float] | list[float] | None = None,
    reference_interval: tuple[float, float] | list[float] | None = None,
    target: float = 1.0,
    use_absolute: bool = True,
    instability_cv_threshold: float = 0.10,
) -> NormalizationResult:
    """Create analysis, optional-normalized, and display branches.

    Parameters are intentionally explicit because internal reference choices
    are scientific assumptions that must be serializable in a recipe.
    Descending and ascending wavenumber axes produce equivalent areas.
    """

    values, was_1d = _as_2d_finite(spectra)
    x = _as_axis(wavenumber, values.shape[1])
    canonical = _canonical_method(method)
    target = float(target)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("normalization target must be finite and positive")
    threshold = float(instability_cv_threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("instability_cv_threshold must be finite and non-negative")

    analysis = values.copy()
    view = values.copy()
    normalized: FloatArray | None = None
    factors = np.ones(values.shape[0], dtype=np.float64)
    warnings: list[str] = []
    effective_interval = reference_interval if reference_interval is not None else interval
    params: dict[str, Any] = {
        "method": canonical,
        "target": target,
        "use_absolute": bool(use_absolute),
        "interval": None
        if effective_interval is None
        else [float(effective_interval[0]), float(effective_interval[1])],
    }

    if canonical == "none":
        pass
    elif canonical == "vector":
        references = np.linalg.norm(values, ord=2, axis=1)
        if np.any(references <= np.finfo(np.float64).tiny):
            row = int(np.flatnonzero(references <= np.finfo(np.float64).tiny)[0])
            raise ValueError(f"vector norm is zero for spectrum {row}")
        factors = target / references
        normalized = values * factors[:, np.newaxis]
        view = normalized.copy()
        warnings.append(
            "Vector normalization is a shape-focused optional branch and removes absolute scale."
        )
    elif canonical == "area":
        mask = _interval_mask(x, effective_interval, required=True)
        selected = np.abs(values[:, mask]) if use_absolute else values[:, mask]
        references = _integral(x[mask], selected)
        zero = np.isclose(references, 0.0, rtol=0.0, atol=np.finfo(np.float64).tiny)
        if np.any(zero):
            row = int(np.flatnonzero(zero)[0])
            raise ValueError(f"normalization area is zero for spectrum {row}")
        factors = target / references
        normalized = values * factors[:, np.newaxis]
        view = normalized.copy()
        warnings.append(
            "Area normalization removes changes in total integrated absorbance; do not use it "
            "when total absorption is the scientific outcome."
        )
    elif canonical in {"internal_peak_height", "internal_peak_area"}:
        mask = _interval_mask(x, effective_interval, required=True)
        selected = values[:, mask]
        if canonical == "internal_peak_height":
            references = (
                np.max(np.abs(selected), axis=1) if use_absolute else np.max(selected, axis=1)
            )
        else:
            area_values = np.abs(selected) if use_absolute else selected
            references = _integral(x[mask], area_values)
        zero = np.isclose(references, 0.0, rtol=0.0, atol=np.finfo(np.float64).tiny)
        if np.any(zero):
            row = int(np.flatnonzero(zero)[0])
            quantity = "height" if canonical.endswith("height") else "area"
            raise ValueError(f"internal reference peak {quantity} is zero for spectrum {row}")
        factors = target / references
        normalized = values * factors[:, np.newaxis]
        view = normalized.copy()

        reference_magnitudes = np.abs(references)
        mean_reference = float(np.mean(reference_magnitudes))
        cv = (
            float(np.std(reference_magnitudes, ddof=1) / mean_reference)
            if references.size > 1 and mean_reference > 0
            else 0.0
        )
        params["reference_values"] = references.tolist()
        params["reference_cv"] = cv
        warnings.append(
            "Internal-reference normalization is valid only if the selected peak is invariant "
            "throughout the perturbation series."
        )
        if cv > threshold:
            warnings.append(
                "Internal reference varies substantially across the series "
                f"(CV={cv:.3g}, threshold={threshold:.3g}); verify that it is stable."
            )
    elif canonical == "minmax_display":
        minima = np.min(values, axis=1)
        spans = np.ptp(values, axis=1)
        zero = np.isclose(spans, 0.0, rtol=0.0, atol=np.finfo(np.float64).tiny)
        if np.any(zero):
            row = int(np.flatnonzero(zero)[0])
            raise ValueError(f"min-max range is zero for spectrum {row}")
        factors = 1.0 / spans
        view = (values - minima[:, np.newaxis]) * factors[:, np.newaxis]
        warnings.append("Min-max normalization is display-only; analysis_data remains unchanged.")

    return NormalizationResult(
        analysis_data=_restore_dimension(analysis, was_1d),
        view_data=_restore_dimension(view, was_1d),
        optional_normalized=(
            None if normalized is None else _restore_dimension(normalized, was_1d)
        ),
        factors=factors.copy(),
        method=canonical,
        warnings=tuple(warnings),
        params=params,
    )


def apply_normalization(
    wavenumber: ArrayLike,
    spectra: ArrayLike,
    config: Any | None = None,
    **overrides: Any,
) -> NormalizationResult:
    """Config-friendly wrapper used by the pipeline and CLI.

    ``config`` may be a mapping, a Pydantic model, or any object exposing the
    expected attributes.  This keeps the numerical module independent of a
    particular Pydantic version.
    """

    data: dict[str, Any] = {}
    if config is not None:
        if isinstance(config, Mapping):
            data.update(config)
        elif hasattr(config, "model_dump"):
            data.update(config.model_dump(exclude_none=True))
        elif hasattr(config, "dict"):
            data.update(config.dict(exclude_none=True))
        else:
            for name in (
                "method",
                "interval",
                "reference_interval",
                "target",
                "use_absolute",
                "instability_cv_threshold",
            ):
                if hasattr(config, name):
                    data[name] = getattr(config, name)
    data.update(overrides)
    # Common configuration spellings are normalized here rather than leaking
    # UI names into the scientific core.
    aliases = {
        "internal_reference_range": "reference_interval",
        "internal_peak_range": "reference_interval",
        "integration_range": "interval",
        "absolute": "use_absolute",
        "target_value": "target",
    }
    for old, new in aliases.items():
        if old in data and new not in data:
            data[new] = data.pop(old)
    allowed = {
        "method",
        "interval",
        "reference_interval",
        "target",
        "use_absolute",
        "instability_cv_threshold",
    }
    filtered = {key: value for key, value in data.items() if key in allowed}
    return normalize_spectra(wavenumber, spectra, **filtered)


# Short, discoverable alias for callers that prefer a verb-only API.
normalize = normalize_spectra


__all__ = [
    "NormalizationResult",
    "apply_normalization",
    "normalize",
    "normalize_spectra",
]
