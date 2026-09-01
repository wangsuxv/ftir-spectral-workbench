"""Optional scientific smoothing after baseline correction.

This module creates a deterministic transformation result from an immutable
``PreparedSpectralDataset``.  It never mutates the parent dataset, resamples an
axis, or smooths along the perturbation dimension.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, SupportsFloat

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import savgol_filter

from .config import WorkbenchConfigMixin
from .fingerprints import canonical_json_bytes, update_array_hash
from .models import PreparedSpectralDataset
from .validation import validate_prepared_dataset

FloatArray = NDArray[np.float64]
SmoothingMethod = Literal["savgol", "gaussian", "moving_average", "median"]
SavgolMode = Literal["interp", "mirror", "nearest"]
ConvolutionMode = Literal["reflect", "mirror", "nearest"]
NonuniformAxisPolicy = Literal["error", "allow_index_space_with_warning"]

RELATIVE_RMS_WARNING_THRESHOLD = 0.10
DERIVATIVE_CORRELATION_WARNING_THRESHOLD = 0.95
RELATIVE_ABSOLUTE_AREA_CHANGE_WARNING_THRESHOLD = 0.02
EDGE_EFFECT_RATIO_WARNING_THRESHOLD = 2.0

_FWHM_PER_SIGMA = 2.35482
_FLOAT64_EPS = np.finfo(np.float64).eps
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_PER_SPECTRUM_METRICS = frozenset(
    {
        "rms_removed",
        "relative_rms_removed",
        "max_abs_removed",
        "roughness_before",
        "roughness_after",
        "roughness_ratio",
        "first_derivative_correlation",
        "signed_area_before",
        "signed_area_after",
        "relative_signed_area_change",
        "absolute_area_before",
        "absolute_area_after",
        "relative_absolute_area_change",
        "edge_rms_removed",
        "center_rms_removed",
        "edge_effect_ratio",
    }
)
_REQUIRED_SUMMARY_METRICS = frozenset(
    {
        "mean_relative_rms_removed",
        "max_relative_rms_removed",
        "mean_first_derivative_correlation",
        "min_first_derivative_correlation",
        "mean_relative_absolute_area_change",
        "max_relative_absolute_area_change",
        "mean_roughness_ratio",
        "mean_edge_effect_ratio",
        "max_edge_effect_ratio",
    }
)


def _strict_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer (bool is not accepted)")
    return int(value)


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number (bool is not accepted)")
    if not isinstance(value, (str, SupportsFloat)):
        raise TypeError(f"{name} must be a real number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _choice(value: object, *, name: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}")
    return normalized


def _validate_odd_window(value: object, *, name: str) -> int:
    window = _strict_integer(value, name=name)
    if window < 3:
        raise ValueError(f"{name} must be at least 3")
    if window % 2 == 0:
        raise ValueError(f"{name} must be odd")
    return window


@dataclass(frozen=True, slots=True)
class PostBaselineSmoothingConfig(WorkbenchConfigMixin):
    """Scientific controls for one optional smoothing transformation."""

    enabled: bool = False
    method: SmoothingMethod = "savgol"

    savgol_window_length: int = 7
    savgol_polyorder: int = 2
    savgol_mode: SavgolMode = "interp"

    gaussian_sigma_points: float = 1.0
    gaussian_truncate: float = 4.0

    moving_average_window_length: int = 3
    median_window_length: int = 3

    convolution_mode: ConvolutionMode = "reflect"

    uniformity_rtol: float = 1e-3
    nonuniform_axis_policy: NonuniformAxisPolicy = "error"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        method = _choice(
            self.method,
            name="method",
            allowed=frozenset({"savgol", "gaussian", "moving_average", "median"}),
        )
        savgol_window = _validate_odd_window(
            self.savgol_window_length,
            name="savgol_window_length",
        )
        savgol_polyorder = _strict_integer(
            self.savgol_polyorder,
            name="savgol_polyorder",
        )
        if savgol_polyorder < 0:
            raise ValueError("savgol_polyorder must be non-negative")
        if savgol_polyorder >= savgol_window:
            raise ValueError("savgol_polyorder must be less than savgol_window_length")
        savgol_mode = _choice(
            self.savgol_mode,
            name="savgol_mode",
            allowed=frozenset({"interp", "mirror", "nearest"}),
        )

        sigma = _finite_float(
            self.gaussian_sigma_points,
            name="gaussian_sigma_points",
        )
        if sigma <= 0.0:
            raise ValueError("gaussian_sigma_points must be positive")
        truncate = _finite_float(
            self.gaussian_truncate,
            name="gaussian_truncate",
        )
        if truncate <= 0.0:
            raise ValueError("gaussian_truncate must be positive")

        moving_window = _validate_odd_window(
            self.moving_average_window_length,
            name="moving_average_window_length",
        )
        median_window = _validate_odd_window(
            self.median_window_length,
            name="median_window_length",
        )
        convolution_mode = _choice(
            self.convolution_mode,
            name="convolution_mode",
            allowed=frozenset({"reflect", "mirror", "nearest"}),
        )
        uniformity_rtol = _finite_float(
            self.uniformity_rtol,
            name="uniformity_rtol",
        )
        if uniformity_rtol < 0.0:
            raise ValueError("uniformity_rtol must be non-negative")
        nonuniform_axis_policy = _choice(
            self.nonuniform_axis_policy,
            name="nonuniform_axis_policy",
            allowed=frozenset({"error", "allow_index_space_with_warning"}),
        )

        object.__setattr__(self, "method", method)
        object.__setattr__(self, "savgol_window_length", savgol_window)
        object.__setattr__(self, "savgol_polyorder", savgol_polyorder)
        object.__setattr__(self, "savgol_mode", savgol_mode)
        object.__setattr__(self, "gaussian_sigma_points", sigma)
        object.__setattr__(self, "gaussian_truncate", truncate)
        object.__setattr__(self, "moving_average_window_length", moving_window)
        object.__setattr__(self, "median_window_length", median_window)
        object.__setattr__(self, "convolution_mode", convolution_mode)
        object.__setattr__(self, "uniformity_rtol", uniformity_rtol)
        object.__setattr__(self, "nonuniform_axis_policy", nonuniform_axis_policy)

    def _active_parameters(self) -> dict[str, int | float | str]:
        if self.method == "savgol":
            return {
                "window_length": self.savgol_window_length,
                "polyorder": self.savgol_polyorder,
                "mode": self.savgol_mode,
            }
        if self.method == "gaussian":
            return {
                "sigma_points": self.gaussian_sigma_points,
                "truncate": self.gaussian_truncate,
                "mode": self.convolution_mode,
            }
        if self.method == "moving_average":
            return {
                "window_length": self.moving_average_window_length,
                "mode": self.convolution_mode,
            }
        return {
            "window_length": self.median_window_length,
            "mode": self.convolution_mode,
        }

    def scientific_dict(self) -> dict[str, Any]:
        """Return only settings that affect the selected scientific operation."""

        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "method": self.method,
            "parameters": self._active_parameters(),
            "axis": "wavenumber",
            "axis_index": 1,
            "uniformity_rtol": self.uniformity_rtol,
            "nonuniform_axis_policy": self.nonuniform_axis_policy,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete editable configuration, including inactive controls."""

        return {
            "enabled": self.enabled,
            "method": self.method,
            "savgol_window_length": self.savgol_window_length,
            "savgol_polyorder": self.savgol_polyorder,
            "savgol_mode": self.savgol_mode,
            "gaussian_sigma_points": self.gaussian_sigma_points,
            "gaussian_truncate": self.gaussian_truncate,
            "moving_average_window_length": self.moving_average_window_length,
            "median_window_length": self.median_window_length,
            "convolution_mode": self.convolution_mode,
            "uniformity_rtol": self.uniformity_rtol,
            "nonuniform_axis_policy": self.nonuniform_axis_policy,
        }


def _immutable_float64(values: ArrayLike, *, name: str) -> FloatArray:
    if np.iscomplexobj(values):
        raise TypeError(f"{name} must contain real values")
    try:
        source = np.asarray(values, dtype=np.float64, order="C")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(source)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.frombuffer(source.tobytes(order="C"), dtype=np.float64).reshape(source.shape)


def _immutable_metric_mapping(
    values: Mapping[str, ArrayLike],
    *,
    n_spectra: int,
) -> Mapping[str, FloatArray]:
    frozen: dict[str, FloatArray] = {}
    for key, raw in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("per_spectrum_metrics keys must be non-empty strings")
        array = _immutable_float64(raw, name=f"per_spectrum_metrics.{key}")
        if array.shape != (n_spectra,):
            raise ValueError(
                f"per_spectrum_metrics.{key} must have shape ({n_spectra},); "
                f"got {array.shape}"
            )
        frozen[key] = array
    missing = _REQUIRED_PER_SPECTRUM_METRICS.difference(frozen)
    if missing:
        raise ValueError(
            "per_spectrum_metrics is missing required metrics: " + ", ".join(sorted(missing))
        )
    return MappingProxyType(frozen)


def _immutable_float_mapping(
    values: Mapping[str, object],
    *,
    name: str,
    required: frozenset[str] = frozenset(),
) -> Mapping[str, float]:
    frozen: dict[str, float] = {}
    for key, raw in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        converted = _finite_float(raw, name=f"{name}.{key}")
        frozen[key] = converted
    missing = required.difference(frozen)
    if missing:
        raise ValueError(f"{name} is missing required metrics: " + ", ".join(sorted(missing)))
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class PostBaselineSmoothingResult:
    """Immutable result and neutral QC diagnostics for one smoothing operation."""

    parent_prepared: PreparedSpectralDataset
    config: PostBaselineSmoothingConfig
    smoothed_spectra: FloatArray
    removed_component: FloatArray
    per_spectrum_metrics: Mapping[str, FloatArray]
    summary_metrics: Mapping[str, float]
    median_wavenumber_spacing: float
    spacing_relative_max_deviation: float
    approximate_physical_width: Mapping[str, float]
    smoothing_fingerprint: str
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parent_prepared, PreparedSpectralDataset):
            raise TypeError("parent_prepared must be a PreparedSpectralDataset")
        if not isinstance(self.config, PostBaselineSmoothingConfig):
            raise TypeError("config must be a PostBaselineSmoothingConfig")
        validate_prepared_dataset(self.parent_prepared)

        smoothed = _immutable_float64(self.smoothed_spectra, name="smoothed_spectra")
        removed = _immutable_float64(self.removed_component, name="removed_component")
        expected_shape = self.parent_prepared.spectra.shape
        if smoothed.shape != expected_shape:
            raise ValueError(
                "smoothed_spectra must have the same shape as parent spectra: "
                f"expected {expected_shape}, got {smoothed.shape}"
            )
        if removed.shape != expected_shape:
            raise ValueError(
                "removed_component must have the same shape as parent spectra: "
                f"expected {expected_shape}, got {removed.shape}"
            )
        if not self.config.enabled:
            if not np.array_equal(smoothed, self.parent_prepared.spectra):
                raise ValueError(
                    "disabled post-baseline smoothing requires smoothed_spectra to be "
                    "element-for-element identical to parent_prepared.spectra"
                )
            if not np.array_equal(removed, np.zeros(expected_shape, dtype=np.float64)):
                raise ValueError(
                    "disabled post-baseline smoothing requires an all-zero removed_component"
                )
        expected_removed = self.parent_prepared.spectra - smoothed
        if not np.array_equal(removed, expected_removed):
            raise ValueError(
                "removed_component must equal parent_prepared.spectra - smoothed_spectra"
            )
        if not isinstance(self.per_spectrum_metrics, Mapping):
            raise TypeError("per_spectrum_metrics must be a mapping")
        per_spectrum = _immutable_metric_mapping(
            self.per_spectrum_metrics,
            n_spectra=self.parent_prepared.n_spectra,
        )
        if not isinstance(self.summary_metrics, Mapping):
            raise TypeError("summary_metrics must be a mapping")
        summary = _immutable_float_mapping(
            self.summary_metrics,
            name="summary_metrics",
            required=_REQUIRED_SUMMARY_METRICS,
        )
        median_spacing = _finite_float(
            self.median_wavenumber_spacing,
            name="median_wavenumber_spacing",
        )
        if median_spacing <= 0.0:
            raise ValueError("median_wavenumber_spacing must be positive")
        relative_deviation = _finite_float(
            self.spacing_relative_max_deviation,
            name="spacing_relative_max_deviation",
        )
        if relative_deviation < 0.0:
            raise ValueError("spacing_relative_max_deviation must be non-negative")
        if not isinstance(self.approximate_physical_width, Mapping):
            raise TypeError("approximate_physical_width must be a mapping")
        physical_width = _immutable_float_mapping(
            self.approximate_physical_width,
            name="approximate_physical_width",
        )
        if any(value < 0.0 for value in physical_width.values()):
            raise ValueError("approximate_physical_width values must be non-negative")
        if not isinstance(self.smoothing_fingerprint, str) or not _SHA256_PATTERN.fullmatch(
            self.smoothing_fingerprint
        ):
            raise ValueError("smoothing_fingerprint must be a lowercase SHA-256 string")
        expected_fingerprint = post_baseline_smoothing_fingerprint(
            self.parent_prepared,
            self.config,
            smoothed,
        )
        if self.smoothing_fingerprint != expected_fingerprint:
            raise ValueError(
                "smoothing_fingerprint does not match the parent, config, and smoothed spectra"
            )
        if isinstance(self.warnings, (str, bytes)):
            raise TypeError("warnings must be an iterable of messages, not a string")
        warnings = tuple(str(item) for item in self.warnings)

        object.__setattr__(self, "smoothed_spectra", smoothed)
        object.__setattr__(self, "removed_component", removed)
        object.__setattr__(self, "per_spectrum_metrics", per_spectrum)
        object.__setattr__(self, "summary_metrics", summary)
        object.__setattr__(self, "median_wavenumber_spacing", median_spacing)
        object.__setattr__(self, "spacing_relative_max_deviation", relative_deviation)
        object.__setattr__(self, "approximate_physical_width", physical_width)
        object.__setattr__(self, "warnings", warnings)


def _length_prefix(payload: bytes) -> bytes:
    return len(payload).to_bytes(8, byteorder="big", signed=False) + payload


def post_baseline_smoothing_fingerprint(
    parent_prepared: PreparedSpectralDataset,
    config: PostBaselineSmoothingConfig,
    smoothed_spectra: ArrayLike,
) -> str:
    """Hash the parent lineage, effective config, and canonical smoothed bytes."""

    if not isinstance(parent_prepared, PreparedSpectralDataset):
        raise TypeError("parent_prepared must be a PreparedSpectralDataset")
    if not isinstance(config, PostBaselineSmoothingConfig):
        raise TypeError("config must be a PostBaselineSmoothingConfig")
    validate_prepared_dataset(parent_prepared)
    smoothed = _immutable_float64(smoothed_spectra, name="smoothed_spectra")
    if smoothed.shape != parent_prepared.spectra.shape:
        raise ValueError(
            "smoothed_spectra must have the same shape as parent spectra: "
            f"expected {parent_prepared.spectra.shape}, got {smoothed.shape}"
        )
    digest = hashlib.sha256()
    payload = {
        "schema": "ftir-workbench-post-baseline-smoothing-v1",
        "parent_prepared_data_sha256": parent_prepared.prepared_data_sha256,
        "config": config.scientific_dict(),
    }
    digest.update(_length_prefix(canonical_json_bytes(payload)))
    update_array_hash(
        digest,
        smoothed,
        field_name="smoothed_corrected_absorbance",
    )
    return digest.hexdigest()


def _is_post_baseline_smoothing_branch(prepared: PreparedSpectralDataset) -> bool:
    if "post_baseline_smoothing" in prepared.baseline_recipe:
        return True
    contract = prepared.baseline_recipe.get("prepared_data_contract")
    return isinstance(contract, Mapping) and contract.get("branch_kind") == (
        "post_baseline_smoothing"
    )


def _axis_diagnostics(
    wavenumber: FloatArray,
    *,
    uniformity_rtol: float,
) -> tuple[float, float, bool]:
    spacing = np.abs(np.diff(wavenumber))
    median_spacing = float(np.median(spacing))
    if not math.isfinite(median_spacing) or median_spacing <= 0.0:
        raise ValueError("wavenumber spacing must have a finite positive median")
    relative_max_deviation = float(
        np.max(np.abs(spacing - median_spacing)) / median_spacing
    )
    approximately_uniform = bool(
        np.allclose(
            spacing,
            median_spacing,
            rtol=uniformity_rtol,
            atol=1e-8,
        )
    )
    return median_spacing, relative_max_deviation, approximately_uniform


def _validate_active_window(config: PostBaselineSmoothingConfig, *, n_points: int) -> None:
    window: int | None = None
    if config.method == "savgol":
        window = config.savgol_window_length
    elif config.method == "moving_average":
        window = config.moving_average_window_length
    elif config.method == "median":
        window = config.median_window_length
    if window is not None and window > n_points:
        raise ValueError(
            f"{config.method} window_length ({window}) must not exceed "
            f"n_wavenumbers ({n_points})"
        )


def _apply_filter(source: FloatArray, config: PostBaselineSmoothingConfig) -> FloatArray:
    if not config.enabled:
        return np.array(source, dtype=np.float64, order="C", copy=True)
    if config.method == "savgol":
        filtered = savgol_filter(
            source,
            window_length=config.savgol_window_length,
            polyorder=config.savgol_polyorder,
            deriv=0,
            axis=1,
            mode=config.savgol_mode,
        )
    elif config.method == "gaussian":
        filtered = gaussian_filter1d(
            source,
            sigma=config.gaussian_sigma_points,
            axis=1,
            mode=config.convolution_mode,
            truncate=config.gaussian_truncate,
        )
    elif config.method == "moving_average":
        filtered = uniform_filter1d(
            source,
            size=config.moving_average_window_length,
            axis=1,
            mode=config.convolution_mode,
        )
    else:
        filtered = median_filter(
            source,
            size=(1, config.median_window_length),
            mode=config.convolution_mode,
        )
    output = np.asarray(filtered, dtype=np.float64, order="C")
    if output.shape != source.shape:
        raise ValueError(
            f"smoothing changed spectra shape from {source.shape} to {output.shape}"
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("post-baseline smoothing produced NaN or infinite values")
    return output


def _approximate_physical_width(
    config: PostBaselineSmoothingConfig,
    *,
    median_spacing: float,
) -> dict[str, float]:
    if not config.enabled:
        return {}
    if config.method == "savgol":
        return {
            "span_cm1": float((config.savgol_window_length - 1) * median_spacing),
        }
    if config.method == "gaussian":
        sigma_cm1 = float(config.gaussian_sigma_points * median_spacing)
        return {
            "sigma_cm1": sigma_cm1,
            "fwhm_cm1": float(_FWHM_PER_SIGMA * sigma_cm1),
        }
    if config.method == "moving_average":
        return {
            "span_cm1": float(
                (config.moving_average_window_length - 1) * median_spacing
            ),
        }
    return {
        "span_cm1": float((config.median_window_length - 1) * median_spacing),
    }


def _row_rms(values: FloatArray) -> FloatArray:
    return np.asarray(
        np.linalg.norm(values, axis=1) / math.sqrt(values.shape[1]),
        dtype=np.float64,
    )


def _row_roughness(values: FloatArray) -> FloatArray:
    if values.shape[1] < 3:
        return np.zeros(values.shape[0], dtype=np.float64)
    second_difference = np.diff(values, n=2, axis=1)
    return np.asarray(np.mean(second_difference**2, axis=1), dtype=np.float64)


def _constant_tolerance(values: FloatArray) -> float:
    scale = max(1.0, float(np.max(np.abs(values))))
    return float(32.0 * _FLOAT64_EPS * scale * math.sqrt(max(1, values.size)))


def _derivative_correlation(before: FloatArray, after: FloatArray) -> float:
    before_centered = before - np.mean(before)
    after_centered = after - np.mean(after)
    before_norm = float(np.linalg.norm(before_centered))
    after_norm = float(np.linalg.norm(after_centered))
    before_constant = before_norm <= _constant_tolerance(before)
    after_constant = after_norm <= _constant_tolerance(after)
    if before_constant or after_constant:
        if not (before_constant and after_constant):
            return 0.0
        tolerance = max(_constant_tolerance(before), _constant_tolerance(after))
        return 1.0 if np.allclose(before, after, rtol=1e-12, atol=tolerance) else 0.0
    correlation = float(np.dot(before_centered, after_centered) / (before_norm * after_norm))
    return float(np.clip(correlation, -1.0, 1.0))


def _edge_width_points(config: PostBaselineSmoothingConfig, *, n_points: int) -> int:
    if not config.enabled:
        radius = 1
    elif config.method == "gaussian":
        radius = math.ceil(config.gaussian_truncate * config.gaussian_sigma_points)
    elif config.method == "savgol":
        radius = config.savgol_window_length // 2
    elif config.method == "moving_average":
        radius = config.moving_average_window_length // 2
    else:
        radius = config.median_window_length // 2
    return max(1, min(radius, n_points // 2))


def _ratio_with_equal_zero_as_one(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    output = numerator / np.maximum(np.abs(denominator), _FLOAT64_EPS)
    both_zero = (numerator == 0.0) & (denominator == 0.0)
    output[both_zero] = 1.0
    return np.asarray(output, dtype=np.float64)


def _compute_qc(
    before: FloatArray,
    after: FloatArray,
    wavenumber: FloatArray,
    config: PostBaselineSmoothingConfig,
) -> tuple[dict[str, FloatArray], dict[str, float], tuple[str, ...]]:
    removed = before - after
    rms_removed = _row_rms(removed)
    rms_before = _row_rms(before)
    relative_rms_removed = rms_removed / np.maximum(rms_before, _FLOAT64_EPS)
    max_abs_removed = np.asarray(np.max(np.abs(removed), axis=1), dtype=np.float64)

    roughness_before = _row_roughness(before)
    roughness_after = _row_roughness(after)
    roughness_ratio = _ratio_with_equal_zero_as_one(
        roughness_after,
        roughness_before,
    )

    derivative_before = np.diff(before, axis=1)
    derivative_after = np.diff(after, axis=1)
    first_derivative_correlation = np.asarray(
        [
            _derivative_correlation(left, right)
            for left, right in zip(derivative_before, derivative_after, strict=True)
        ],
        dtype=np.float64,
    )

    if wavenumber[0] > wavenumber[-1]:
        integration_axis = wavenumber[::-1]
        integration_before = before[:, ::-1]
        integration_after = after[:, ::-1]
    else:
        integration_axis = wavenumber
        integration_before = before
        integration_after = after
    signed_area_before = np.asarray(
        trapezoid(integration_before, x=integration_axis, axis=1),
        dtype=np.float64,
    )
    signed_area_after = np.asarray(
        trapezoid(integration_after, x=integration_axis, axis=1),
        dtype=np.float64,
    )
    relative_signed_area_change = (
        (signed_area_after - signed_area_before)
        / np.maximum(np.abs(signed_area_before), _FLOAT64_EPS)
    )
    absolute_area_before = np.asarray(
        trapezoid(np.abs(integration_before), x=integration_axis, axis=1),
        dtype=np.float64,
    )
    absolute_area_after = np.asarray(
        trapezoid(np.abs(integration_after), x=integration_axis, axis=1),
        dtype=np.float64,
    )
    relative_absolute_area_change = (
        np.abs(absolute_area_after - absolute_area_before)
        / np.maximum(absolute_area_before, _FLOAT64_EPS)
    )

    edge_width = _edge_width_points(config, n_points=before.shape[1])
    edge_values = np.concatenate(
        (removed[:, :edge_width], removed[:, -edge_width:]),
        axis=1,
    )
    center_values = removed[:, edge_width:-edge_width]
    if center_values.shape[1] == 0:
        # With no distinct center points, compare the edge metric with the full
        # residual so the diagnostic remains finite and neutral.
        center_values = removed
    edge_rms_removed = _row_rms(edge_values)
    center_rms_removed = _row_rms(center_values)
    edge_effect_ratio = _ratio_with_equal_zero_as_one(
        edge_rms_removed,
        center_rms_removed,
    )

    per_spectrum = {
        "rms_removed": rms_removed,
        "relative_rms_removed": relative_rms_removed,
        "max_abs_removed": max_abs_removed,
        "roughness_before": roughness_before,
        "roughness_after": roughness_after,
        "roughness_ratio": roughness_ratio,
        "first_derivative_correlation": first_derivative_correlation,
        "signed_area_before": signed_area_before,
        "signed_area_after": signed_area_after,
        "relative_signed_area_change": np.asarray(
            relative_signed_area_change,
            dtype=np.float64,
        ),
        "absolute_area_before": absolute_area_before,
        "absolute_area_after": absolute_area_after,
        "relative_absolute_area_change": np.asarray(
            relative_absolute_area_change,
            dtype=np.float64,
        ),
        "edge_rms_removed": edge_rms_removed,
        "center_rms_removed": center_rms_removed,
        "edge_effect_ratio": edge_effect_ratio,
    }
    if any(not np.all(np.isfinite(values)) for values in per_spectrum.values()):
        raise ValueError("post-baseline smoothing QC produced NaN or infinite values")

    summary = {
        "mean_relative_rms_removed": float(np.mean(relative_rms_removed)),
        "max_relative_rms_removed": float(np.max(relative_rms_removed)),
        "mean_first_derivative_correlation": float(
            np.mean(first_derivative_correlation)
        ),
        "min_first_derivative_correlation": float(
            np.min(first_derivative_correlation)
        ),
        "mean_relative_absolute_area_change": float(
            np.mean(relative_absolute_area_change)
        ),
        "max_relative_absolute_area_change": float(
            np.max(relative_absolute_area_change)
        ),
        "mean_roughness_ratio": float(np.mean(roughness_ratio)),
        "mean_edge_effect_ratio": float(np.mean(edge_effect_ratio)),
        "max_edge_effect_ratio": float(np.max(edge_effect_ratio)),
    }

    warnings: list[str] = []
    relative_rms_count = int(
        np.count_nonzero(relative_rms_removed > RELATIVE_RMS_WARNING_THRESHOLD)
    )
    if relative_rms_count:
        warnings.append(
            "Relative RMS removed exceeds the diagnostic threshold "
            f"({RELATIVE_RMS_WARNING_THRESHOLD:.2f}) for {relative_rms_count} spectrum/s; "
            f"maximum={np.max(relative_rms_removed):.6g}."
        )
    derivative_count = int(
        np.count_nonzero(
            first_derivative_correlation < DERIVATIVE_CORRELATION_WARNING_THRESHOLD
        )
    )
    if derivative_count:
        warnings.append(
            "First-derivative correlation is below the diagnostic threshold "
            f"({DERIVATIVE_CORRELATION_WARNING_THRESHOLD:.2f}) for "
            f"{derivative_count} spectrum/s; "
            f"minimum={np.min(first_derivative_correlation):.6g}."
        )
    area_count = int(
        np.count_nonzero(
            relative_absolute_area_change
            > RELATIVE_ABSOLUTE_AREA_CHANGE_WARNING_THRESHOLD
        )
    )
    if area_count:
        warnings.append(
            "Relative absolute-area change exceeds the diagnostic threshold "
            f"({RELATIVE_ABSOLUTE_AREA_CHANGE_WARNING_THRESHOLD:.2f}) for "
            f"{area_count} spectrum/s; "
            f"maximum={np.max(relative_absolute_area_change):.6g}."
        )
    edge_count = int(
        np.count_nonzero(edge_effect_ratio > EDGE_EFFECT_RATIO_WARNING_THRESHOLD)
    )
    if edge_count:
        warnings.append(
            "Edge RMS removal substantially exceeds center RMS removal "
            f"(ratio > {EDGE_EFFECT_RATIO_WARNING_THRESHOLD:.1f}) for "
            f"{edge_count} spectrum/s; maximum={np.max(edge_effect_ratio):.6g}."
        )
    return per_spectrum, summary, tuple(warnings)


def apply_post_baseline_smoothing(
    prepared: PreparedSpectralDataset,
    config: PostBaselineSmoothingConfig,
) -> PostBaselineSmoothingResult:
    """Apply one deterministic post-baseline filter along the wavenumber axis."""

    if not isinstance(prepared, PreparedSpectralDataset):
        raise TypeError("prepared must be a PreparedSpectralDataset")
    if not isinstance(config, PostBaselineSmoothingConfig):
        raise TypeError("config must be a PostBaselineSmoothingConfig")
    validate_prepared_dataset(prepared)
    if prepared.normalization_state == "scientific_explicit":
        raise ValueError(
            "v0.2.5 does not combine scientific normalization and post-baseline "
            "smoothing. Select the primary unnormalized Prepared branch."
        )
    if _is_post_baseline_smoothing_branch(prepared):
        raise ValueError(
            "The selected Prepared dataset is already a post-baseline smoothing branch. "
            "Chained smoothing is disabled in v0.2.5."
        )

    median_spacing, relative_deviation, approximately_uniform = _axis_diagnostics(
        prepared.wavenumber,
        uniformity_rtol=config.uniformity_rtol,
    )
    warnings: list[str] = []
    if config.enabled and not approximately_uniform:
        if config.nonuniform_axis_policy == "error":
            raise ValueError(
                "The wavenumber axis is not approximately uniformly spaced within "
                f"uniformity_rtol={config.uniformity_rtol:g}. v0.2.5 does not "
                "automatically resample. Export a regular grid or explicitly select "
                "nonuniform_axis_policy='allow_index_space_with_warning'."
            )
        warnings.append(
            "Wavenumber spacing exceeds the configured uniformity tolerance. "
            "Index-space smoothing was applied under the explicit expert override; "
            "the axis was not resampled."
        )

    if config.enabled:
        _validate_active_window(config, n_points=prepared.n_wavenumbers)
    source = np.array(prepared.spectra, dtype=np.float64, order="C", copy=True)
    smoothed = _apply_filter(source, config)
    removed = source - smoothed
    per_spectrum, summary, qc_warnings = _compute_qc(
        source,
        smoothed,
        prepared.wavenumber,
        config,
    )
    if config.enabled and config.method == "median":
        warnings.append(
            "Median / despike smoothing is nonlinear and may flatten genuine narrow peaks."
        )
    warnings.extend(qc_warnings)
    fingerprint = post_baseline_smoothing_fingerprint(prepared, config, smoothed)
    return PostBaselineSmoothingResult(
        parent_prepared=prepared,
        config=config,
        smoothed_spectra=smoothed,
        removed_component=removed,
        per_spectrum_metrics=per_spectrum,
        summary_metrics=summary,
        median_wavenumber_spacing=median_spacing,
        spacing_relative_max_deviation=relative_deviation,
        approximate_physical_width=_approximate_physical_width(
            config,
            median_spacing=median_spacing,
        ),
        smoothing_fingerprint=fingerprint,
        warnings=tuple(warnings),
    )


__all__ = [
    "PostBaselineSmoothingConfig",
    "PostBaselineSmoothingResult",
    "apply_post_baseline_smoothing",
    "post_baseline_smoothing_fingerprint",
]
