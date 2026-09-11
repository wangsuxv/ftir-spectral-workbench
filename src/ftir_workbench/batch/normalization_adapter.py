"""Strict ordinary-spectrum semantics around the unchanged normalization core.

All transformed output comes from the public ``apply_normalization`` function.
Reference calculations here provide positive/near-zero guards and provenance;
they do not replace the core's numerical output or modify its defaults.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid

from ftir_baseline.models import freeze_value
from ftir_baseline.normalization import apply_normalization
from ftir_workbench.post_baseline_smoothing import validate_spectral_arrays

from .models import BatchError

FloatArray = NDArray[np.float64]
NUMERICAL_GUARD_VERSION = "ordinary-normalization-reference-floor-v1"
NUMERICAL_FLOOR_EPS_MULTIPLIER = 64.0
_EPS = np.finfo(np.float64).eps
_METHODS = frozenset(
    {
        "none",
        "maximum",
        "internal_peak_height",
        "internal_peak_area",
        "area",
        "vector",
        "minmax_display",
    }
)
_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "method": "none",
    "reference_interval": None,
    "area_interval": None,
    "area_definition": "absolute",
    "target": 1.0,
    "minimum_reference": 0.0,
}


def _choice(value: Any, name: str, choices: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise BatchError("NORMALIZATION_CONFIG_INVALID", f"{name} must be a string")
    canonical = value.strip().lower().replace("-", "_")
    if canonical not in choices:
        raise BatchError("NORMALIZATION_CONFIG_INVALID", f"{name} must be one of {sorted(choices)}")
    return canonical


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise BatchError("NORMALIZATION_CONFIG_INVALID", f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BatchError("NORMALIZATION_CONFIG_INVALID", f"{name} must be a real number") from exc
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        condition = "positive" if positive else "nonnegative"
        raise BatchError("NORMALIZATION_CONFIG_INVALID", f"{name} must be finite and {condition}")
    return result


def _interval(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (Sequence, np.ndarray)) or isinstance(value, (str, bytes)):
        raise BatchError("NORMALIZATION_INTERVAL_INVALID", f"{name} must contain two finite bounds")
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise BatchError("NORMALIZATION_INTERVAL_INVALID", f"{name} must contain two scalar bounds")
    if len(value) != 2:
        raise BatchError("NORMALIZATION_INTERVAL_INVALID", f"{name} must contain two finite bounds")
    try:
        if any(isinstance(v, (bool, np.bool_)) or np.iscomplexobj(v) for v in value):
            raise ValueError("bounds must be real numbers")
        low, high = sorted(float(v) for v in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BatchError(
            "NORMALIZATION_INTERVAL_INVALID", f"{name} must contain two finite bounds"
        ) from exc
    if not math.isfinite(low) or not math.isfinite(high) or low == high:
        raise BatchError(
            "NORMALIZATION_INTERVAL_INVALID", f"{name} must contain distinct finite bounds"
        )
    return low, high


def _effective_values(raw: Mapping[str, Any]) -> dict[str, Any]:
    unknown = raw.keys() - _DEFAULTS.keys()
    if unknown:
        raise BatchError(
            "NORMALIZATION_CONFIG_INVALID", f"Unknown settings: {sorted(unknown, key=str)}"
        )
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise BatchError("NORMALIZATION_CONFIG_INVALID", "enabled must be a bool")
    values = dict(_DEFAULTS)
    if not enabled:
        return values
    method = _choice(raw.get("method", "none"), "method", _METHODS)
    if method == "none":
        return values
    values.update(enabled=True, method=method)
    values["minimum_reference"] = _number(raw.get("minimum_reference", 0.0), "minimum_reference")
    if method != "minmax_display":
        values["target"] = _number(raw.get("target", 1.0), "target", positive=True)
    if method in {"internal_peak_height", "internal_peak_area"}:
        values["reference_interval"] = _interval(
            raw.get("reference_interval"), "reference_interval"
        )
    if method == "area" and raw.get("area_interval") is not None:
        values["area_interval"] = _interval(raw["area_interval"], "area_interval")
    if method in {"internal_peak_area", "area"}:
        values["area_definition"] = _choice(
            raw.get("area_definition", "absolute"),
            "area_definition",
            frozenset({"absolute", "signed"}),
        )
    return values


@dataclass(frozen=True, slots=True)
class OrdinaryNormalizationConfig:
    """Canonical selected controls; raw drafts are retained by the workspace.

    ``area_interval=None`` explicitly means the whole parent domain. Internal
    reference methods require a range. Inactive controls become defaults, so an
    unfinished inactive editor cannot change or block the scientific operation.
    """

    enabled: bool = False
    method: str = "none"
    reference_interval: tuple[float, float] | None = None
    area_interval: tuple[float, float] | None = None
    area_definition: str = "absolute"
    target: float = 1.0
    minimum_reference: float = 0.0

    def __post_init__(self) -> None:
        values = _effective_values({key: getattr(self, key) for key in _DEFAULTS})
        for key, value in values.items():
            object.__setattr__(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize all canonical fields, not the separate original UI draft."""

        return {key: getattr(self, key) for key in _DEFAULTS}

    def scientific_dict(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        values: dict[str, Any] = {
            "enabled": True,
            "method": self.method,
            "minimum_reference": self.minimum_reference,
            "numerical_guard_version": NUMERICAL_GUARD_VERSION,
        }
        if self.method != "minmax_display":
            values["target"] = self.target
        if self.method in {"internal_peak_height", "internal_peak_area"}:
            values["reference_interval"] = self.reference_interval
        if self.method == "area":
            values["area_interval"] = self.area_interval
        if self.method in {"internal_peak_area", "area"}:
            values["area_definition"] = self.area_definition
        return values


def normalize_config(
    config: OrdinaryNormalizationConfig | Mapping[str, Any],
) -> OrdinaryNormalizationConfig:
    """Resolve only selected active settings, leaving the caller's raw draft intact."""

    if isinstance(config, OrdinaryNormalizationConfig):
        return config
    if not isinstance(config, Mapping):
        raise BatchError(
            "NORMALIZATION_CONFIG_INVALID", "config must be a configuration or mapping"
        )
    return OrdinaryNormalizationConfig(**_effective_values(config))


def _readonly(values: ArrayLike, name: str) -> FloatArray:
    if np.iscomplexobj(values):
        raise BatchError("NORMALIZATION_NUMERICAL_ERROR", f"{name} must be real")
    array = np.asarray(values, dtype=np.float64, order="C")
    if not np.all(np.isfinite(array)):
        raise BatchError("NORMALIZATION_NUMERICAL_ERROR", f"{name} is not finite")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class ArrayNormalizationResult:
    """Selected core output plus the complete affine and reference evidence."""

    wavenumber: FloatArray
    input_spectra: FloatArray
    normalized_spectra: FloatArray
    config: OrdinaryNormalizationConfig
    scale: FloatArray
    offset: FloatArray
    reference_details: Mapping[str, Any]
    quantity: str
    purpose: str
    warnings: tuple[str, ...]
    metrics: Mapping[str, FloatArray]

    def __post_init__(self) -> None:
        axis, source = validate_spectral_arrays(self.wavenumber, self.input_spectra)
        output = _readonly(self.normalized_spectra, "normalized_spectra")
        scale = _readonly(self.scale, "scale")
        offset = _readonly(self.offset, "offset")
        if (
            output.shape != source.shape
            or scale.shape != (source.shape[0],)
            or offset.shape != scale.shape
        ):
            raise BatchError(
                "NORMALIZATION_NUMERICAL_ERROR", "output/scale/offset shapes do not match input"
            )
        if np.any(scale <= 0):
            raise BatchError("NORMALIZATION_NUMERICAL_ERROR", "scale must be finite and positive")
        config = normalize_config(self.config)
        if not config.enabled and (
            not np.array_equal(output, source) or np.any(scale != 1) or np.any(offset != 0)
        ):
            raise BatchError(
                "NORMALIZATION_NUMERICAL_ERROR", "disabled normalization must be exact identity"
            )
        metrics: dict[str, FloatArray] = {}
        for key, raw in self.metrics.items():
            metric = _readonly(raw, f"metrics.{key}")
            if not isinstance(key, str) or metric.shape != scale.shape:
                raise BatchError(
                    "NORMALIZATION_NUMERICAL_ERROR", "metrics must contain one value per spectrum"
                )
            metrics[key] = metric
        object.__setattr__(self, "wavenumber", axis)
        object.__setattr__(self, "input_spectra", source)
        object.__setattr__(self, "normalized_spectra", output)
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "reference_details", freeze_value(dict(self.reference_details)))
        object.__setattr__(self, "warnings", tuple(map(str, self.warnings)))


def _selected_interval(
    axis: FloatArray,
    requested: tuple[float, float] | None,
) -> tuple[NDArray[np.bool_], dict[str, Any]]:
    domain = float(np.min(axis)), float(np.max(axis))
    bounds = domain if requested is None else requested
    if bounds[0] < domain[0] or bounds[1] > domain[1]:
        raise BatchError(
            "NORMALIZATION_INTERVAL_OUT_OF_BOUNDS",
            f"Requested bounds {bounds} must lie entirely within parent domain {domain}",
        )
    mask = (axis >= bounds[0]) & (axis <= bounds[1])
    count = int(np.count_nonzero(mask))
    if count < 2:
        raise BatchError(
            "NORMALIZATION_INSUFFICIENT_REFERENCE_POINTS",
            "Reference interval must include at least two actual samples",
        )
    selected = axis[mask]
    return mask, {
        "requested_bounds": list(bounds),
        "actual_bounds": [float(np.min(selected)), float(np.max(selected))],
        "point_count": count,
        "selection_rule": "inclusive_existing_samples_no_interpolation",
        "full_parent_domain": requested is None,
    }


def _area(axis: FloatArray, values: FloatArray) -> FloatArray:
    # Guard/metadata calculation only. Authoritative normalization is the core.
    if axis[0] > axis[-1]:
        axis, values = axis[::-1], values[:, ::-1]
    return np.asarray(trapezoid(values, x=axis, axis=1), dtype=np.float64)


def _stable_l2(values: FloatArray) -> FloatArray:
    magnitude = np.max(np.abs(values), axis=1)
    divided = np.divide(
        values, magnitude[:, None], out=np.zeros_like(values), where=magnitude[:, None] > 0
    )
    return np.asarray(magnitude * np.sqrt(np.sum(divided * divided, axis=1)), dtype=np.float64)


def _references(
    axis: FloatArray,
    values: FloatArray,
    config: OrdinaryNormalizationConfig,
) -> tuple[FloatArray, FloatArray, dict[str, Any], dict[str, Any]]:
    method = config.method
    core: dict[str, Any] = {"method": method, "target": config.target, "use_absolute": False}
    minima, maxima = np.min(values, axis=1), np.max(values, axis=1)
    scale = np.max(np.abs(values), axis=1)
    details: dict[str, Any] = {"minimum": minima.tolist(), "maximum": maxima.tolist()}
    if method in {"maximum", "internal_peak_height", "internal_peak_area", "area"}:
        interval = config.area_interval if method == "area" else config.reference_interval
        mask, interval_details = _selected_interval(axis, interval)
        details.update(interval_details)
        core["interval" if method == "area" else "reference_interval"] = details["requested_bounds"]
        core["method"] = "internal_peak_height" if method == "maximum" else method
        if method in {"area", "internal_peak_area"}:
            absolute = config.area_definition == "absolute"
            selected = values[:, mask]
            reference = _area(axis[mask], np.abs(selected) if absolute else selected)
            scale = _area(axis, np.abs(values))
            core["use_absolute"] = absolute
            details.update(
                area_definition=config.area_definition,
                integration="trapezoid_actual_x_ascending_copy",
            )
        else:
            reference = np.max(values[:, mask], axis=1)
            details["height_definition"] = "positive_maximum_not_absolute"
    elif method == "vector":
        reference = scale = _stable_l2(values)
        details.update(
            point_count=axis.size,
            selection_rule="all_parent_samples",
            norm="sampled_point_l2_without_x_weights",
        )
    else:
        reference = maxima - minima
        details.update(
            span=reference.tolist(), point_count=axis.size, selection_rule="all_parent_samples"
        )
    return np.asarray(reference), np.asarray(scale), details, core


def _reference_floor(
    reference: FloatArray,
    reference_scale: FloatArray,
    config: OrdinaryNormalizationConfig,
) -> FloatArray:
    floor = np.maximum(
        config.minimum_reference,
        NUMERICAL_FLOOR_EPS_MULTIPLIER * _EPS * reference_scale,
    )
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(reference_scale)):
        raise BatchError("NORMALIZATION_NUMERICAL_ERROR", "Reference calculation is not finite")
    invalid = reference <= floor
    if np.any(invalid):
        row = int(np.flatnonzero(invalid)[0])
        raise BatchError(
            "NORMALIZATION_REFERENCE_TOO_SMALL",
            f"Spectrum {row}: positive reference {reference[row]:.8g} must exceed "
            f"numerical floor {floor[row]:.8g}; check the range and negative/cancelling signal. "
            "This numerical threshold is not a detection limit.",
        )
    return np.asarray(floor, dtype=np.float64)


def validate_normalization_request(
    wavenumber: ArrayLike,
    spectra: ArrayLike,
    config: OrdinaryNormalizationConfig | Mapping[str, Any],
) -> None:
    """Check target arrays, range and reference guards without transforming output.

    This validation never calls the normalization core, smoothing or a baseline
    algorithm. A later explicit preview still validates the core's actual finite
    factors and output, including any implementation-specific overflow.
    """

    effective = normalize_config(config)
    try:
        axis, source = validate_spectral_arrays(wavenumber, spectra)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BatchError("NORMALIZATION_INPUT_INVALID", str(exc)) from exc
    if effective.enabled:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                reference, reference_scale, _, _ = _references(axis, source, effective)
                _reference_floor(reference, reference_scale, effective)
        except (FloatingPointError, OverflowError) as exc:
            raise BatchError(
                "NORMALIZATION_NUMERICAL_ERROR", "Finite reference values cannot be computed"
            ) from exc


def normalize_spectral_arrays(
    wavenumber: ArrayLike,
    spectra: ArrayLike,
    config: OrdinaryNormalizationConfig | Mapping[str, Any],
) -> ArrayNormalizationResult:
    """Normalize one or more independent rows from their exact selected parent.

    Positive maxima, signed areas, near-zero references, and full interval
    containment are ordinary-mode requirements beyond the frozen core defaults.
    Output preserves the complete input axis and every original signal sign
    except the explicitly requested affine Min-Max display transformation.
    """

    effective = normalize_config(config)
    try:
        axis, source = validate_spectral_arrays(wavenumber, spectra)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BatchError("NORMALIZATION_INPUT_INVALID", str(exc)) from exc
    rows = source.shape[0]
    warnings: list[str] = []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            if not effective.enabled:
                core = apply_normalization(axis, source, {"method": "none"})
                output = core.analysis_data
                scale, offset = core.factors, np.zeros(rows)
                details: dict[str, Any] = {"enabled": False, "core_method": "none"}
                metrics: dict[str, FloatArray] = {}
                quantity, purpose = "absorbance", "identity"
            else:
                reference, reference_scale, details, core_config = _references(
                    axis, source, effective
                )
                floor = _reference_floor(reference, reference_scale, effective)
                core = apply_normalization(axis, source, core_config)
                scale = core.factors
                if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                    raise BatchError(
                        "NORMALIZATION_NUMERICAL_ERROR",
                        "Core returned a nonpositive or nonfinite scale",
                    )
                if effective.method == "minmax_display":
                    output = core.view_data
                    offset = -np.min(source, axis=1) / reference
                    quantity, purpose = "minmax_scaled_intensity", "display_only"
                    details["authoritative_formula"] = "(y - minimum) * scale"
                else:
                    if core.optional_normalized is None:
                        raise BatchError(
                            "NORMALIZATION_NUMERICAL_ERROR", "Core scientific output is missing"
                        )
                    output = core.optional_normalized
                    offset = np.zeros(rows)
                    quantity, purpose = "normalized_intensity", "scientific"
                # For Min-Max the stable core view is authoritative. Affine
                # reconstruction may cancel, so record its rounding residual.
                reconstructed = scale[:, None] * source + offset[:, None]
                affine_error = np.max(np.abs(output - reconstructed), axis=1)
                rounding_scale = np.max(np.abs(scale[:, None] * source), axis=1) + np.abs(offset)
                if np.any(affine_error > 8 * _EPS * np.maximum(rounding_scale, 1.0)):
                    raise BatchError(
                        "NORMALIZATION_NUMERICAL_ERROR",
                        "Core output does not match the recorded affine transformation",
                    )
                details.update(
                    enabled=True,
                    core_method=core.method,
                    core_parameters=dict(core.params),
                    core_warnings=list(core.warnings),
                    reference_values=reference.tolist(),
                    reference_scale=reference_scale.tolist(),
                    numerical_floor=floor.tolist(),
                    numerical_guard_version=NUMERICAL_GUARD_VERSION,
                    numerical_floor_eps_multiplier=NUMERICAL_FLOOR_EPS_MULTIPLIER,
                    user_minimum_reference=effective.minimum_reference,
                    affine_formula="z = scale * y + offset",
                )
                metrics = {
                    "reference_values": reference,
                    "reference_scale": reference_scale,
                    "numerical_floor": floor,
                    "affine_reconstruction_max_abs_error": affine_error,
                }
                if effective.method in {"maximum", "internal_peak_height", "internal_peak_area"}:
                    warnings.append(
                        "A reference feature is a user assumption; processing an independent "
                        "spectrum cannot establish reference stability across samples. Core "
                        "reference_cv, including a single-spectrum value of zero, is not such evidence."
                    )
                else:
                    warnings.extend(core.warnings)
                if effective.method in {"area", "internal_peak_area"}:
                    warnings.append(
                        "Area-normalized intensity removes absolute scale and depends on the "
                        "wavenumber unit, selected range, and discrete integration; it is not original absorbance."
                    )
                elif effective.method == "vector":
                    warnings.append(
                        "L2 uses sampled-point squares without wavenumber weights; its scale "
                        "depends on sampling density and is not a common physical scale across grids."
                    )
                elif effective.method == "minmax_display":
                    warnings.append(
                        "Min-Max includes an offset; within-spectrum intensity ratios are not preserved."
                    )
                if effective.method != "minmax_display":
                    warnings.append(
                        "Normalization removes absolute intensity scale; the output is not absorbance or physical transmittance."
                    )
                else:
                    warnings.append(
                        "Display-scaled intensity is not absorbance or physical transmittance."
                    )
    except (FloatingPointError, OverflowError) as exc:
        raise BatchError(
            "NORMALIZATION_NUMERICAL_ERROR",
            "Finite reference, scale, offset and output cannot be computed",
        ) from exc
    except BatchError:
        raise
    except ValueError as exc:
        raise BatchError("NORMALIZATION_NUMERICAL_ERROR", str(exc)) from exc
    return ArrayNormalizationResult(
        wavenumber=axis,
        input_spectra=source,
        normalized_spectra=output,
        config=effective,
        scale=scale,
        offset=offset,
        reference_details=details,
        quantity=quantity,
        purpose=purpose,
        warnings=tuple(warnings),
        metrics=metrics,
    )


__all__ = [
    "NUMERICAL_FLOOR_EPS_MULTIPLIER",
    "NUMERICAL_GUARD_VERSION",
    "ArrayNormalizationResult",
    "OrdinaryNormalizationConfig",
    "normalize_config",
    "normalize_spectral_arrays",
    "validate_normalization_request",
]
