"""Validated, JSON-round-trippable configuration objects."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Self


def _normalise_method(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_compatible(item) for item in value]
    return value


class ConfigMixin:
    """Small serialization mixin shared by frozen configuration dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Return configuration as JSON-compatible primitives."""

        return _json_compatible(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize configuration to a deterministic JSON string."""

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        """Build a configuration from a mapping."""

        return cls(**dict(values))


@dataclass(frozen=True, slots=True)
class WavenumberRange(ConfigMixin):
    """One normalized analysis interval used by multi-range runs.

    Endpoints may be supplied in either order.  They are stored as explicit
    ``high_wavenumber`` and ``low_wavenumber`` values so configuration files
    and exported manifests remain unambiguous for the conventional FTIR
    high-to-low display direction.
    """

    high_wavenumber: float
    low_wavenumber: float
    label: str | None = None

    def __post_init__(self) -> None:
        first = float(self.high_wavenumber)
        second = float(self.low_wavenumber)
        if not math.isfinite(first) or not math.isfinite(second):
            raise ValueError("Wavenumber range endpoints must be finite")
        if first == second:
            raise ValueError("Wavenumber range endpoints must be distinct")

        label = None if self.label is None else str(self.label).strip()
        if self.label is not None and not label:
            raise ValueError("Wavenumber range label cannot be empty")

        object.__setattr__(self, "high_wavenumber", max(first, second))
        object.__setattr__(self, "low_wavenumber", min(first, second))
        object.__setattr__(self, "label", label)

    @property
    def bounds(self) -> tuple[float, float]:
        """Return the normalized ``(lower, upper)`` interval."""

        return self.low_wavenumber, self.high_wavenumber

    @property
    def display_name(self) -> str:
        """Return a concise human-readable interval name."""

        interval = f"{self.high_wavenumber:g}-{self.low_wavenumber:g} cm^-1"
        return interval if self.label is None else f"{self.label} ({interval})"

    @classmethod
    def from_value(cls, value: object) -> WavenumberRange:
        """Coerce a range object, mapping, or two/three-item sequence."""

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            data = dict(value)
            high = data.get("high_wavenumber", data.get("high", data.get("upper")))
            low = data.get("low_wavenumber", data.get("low", data.get("lower")))
            if high is None or low is None:
                raise ValueError(
                    "A wavenumber range mapping requires high_wavenumber and low_wavenumber"
                )
            return cls(high_wavenumber=high, low_wavenumber=low, label=data.get("label"))
        if isinstance(value, (list, tuple)):
            if len(value) not in {2, 3}:
                raise ValueError(
                    "A wavenumber range sequence must contain two endpoints and an optional label"
                )
            return cls(
                high_wavenumber=value[0],
                low_wavenumber=value[1],
                label=None if len(value) == 2 else value[2],
            )
        raise TypeError("Wavenumber ranges must be WavenumberRange objects, mappings, or sequences")


@dataclass(frozen=True, slots=True)
class BaselineConfig(ConfigMixin):
    """Parameters shared by all baseline correction methods."""

    method: str = "none"
    offset_mode: str = "minimum"
    offset_window: tuple[float, float] | None = None
    anchor_ranges: tuple[tuple[float, float], ...] = ()
    polynomial_order: int = 1
    asls_lam: float = 1e6
    asls_p: float = 0.01
    asls_diff_order: int = 2
    asls_max_iter: int = 50
    asls_tol: float = 1e-3
    rubberband_segments: int | tuple[int, ...] = 1
    rubberband_lam: float | None = None
    rubberband_diff_order: int = 2
    rubberband_smooth_half_window: int | None = None

    def __post_init__(self) -> None:
        method = _normalise_method(self.method)
        method = {
            "constant": "offset",
            "constant_offset": "offset",
            "anchor": "anchor_polynomial",
            "polynomial": "anchor_polynomial",
        }.get(method, method)
        if method not in {"none", "offset", "anchor_polynomial", "asls", "rubberband"}:
            raise ValueError(f"Unsupported baseline method: {self.method!r}")

        offset_mode = _normalise_method(self.offset_mode)
        offset_mode = {"min": "minimum", "window": "window_median"}.get(offset_mode, offset_mode)
        if offset_mode not in {"minimum", "window_median"}:
            raise ValueError("offset_mode must be 'minimum' or 'window_median'")

        offset_window = None
        if self.offset_window is not None:
            if len(self.offset_window) != 2:
                raise ValueError("offset_window must contain exactly two wavenumbers")
            offset_window = (float(self.offset_window[0]), float(self.offset_window[1]))

        anchor_ranges: list[tuple[float, float]] = []
        for interval in self.anchor_ranges:
            if len(interval) != 2:
                raise ValueError("Every anchor range must contain exactly two wavenumbers")
            anchor_ranges.append((float(interval[0]), float(interval[1])))

        if not 0 <= int(self.polynomial_order) <= 3:
            raise ValueError("polynomial_order must be one of 0, 1, 2, or 3")
        if float(self.asls_lam) <= 0:
            raise ValueError("asls_lam must be greater than 0")
        if not 0 < float(self.asls_p) < 1:
            raise ValueError("asls_p must be strictly between 0 and 1")
        if int(self.asls_diff_order) < 1:
            raise ValueError("asls_diff_order must be at least 1")
        if int(self.asls_max_iter) < 1:
            raise ValueError("asls_max_iter must be at least 1")
        if float(self.asls_tol) < 0:
            raise ValueError("asls_tol must be non-negative")

        segments: int | tuple[int, ...]
        if isinstance(self.rubberband_segments, int):
            segments = int(self.rubberband_segments)
            if segments < 1:
                raise ValueError("rubberband_segments must be at least 1")
        else:
            segments = tuple(int(value) for value in self.rubberband_segments)
            if not segments:
                raise ValueError("rubberband segment indices cannot be empty")
        if self.rubberband_lam is not None and float(self.rubberband_lam) < 0:
            raise ValueError("rubberband_lam must be non-negative or None")
        if int(self.rubberband_diff_order) < 1:
            raise ValueError("rubberband_diff_order must be at least 1")
        if (
            self.rubberband_smooth_half_window is not None
            and int(self.rubberband_smooth_half_window) < 0
        ):
            raise ValueError("rubberband_smooth_half_window must be non-negative or None")

        object.__setattr__(self, "method", method)
        object.__setattr__(self, "offset_mode", offset_mode)
        object.__setattr__(self, "offset_window", offset_window)
        object.__setattr__(self, "anchor_ranges", tuple(anchor_ranges))
        object.__setattr__(self, "polynomial_order", int(self.polynomial_order))
        object.__setattr__(self, "asls_lam", float(self.asls_lam))
        object.__setattr__(self, "asls_p", float(self.asls_p))
        object.__setattr__(self, "asls_diff_order", int(self.asls_diff_order))
        object.__setattr__(self, "asls_max_iter", int(self.asls_max_iter))
        object.__setattr__(self, "asls_tol", float(self.asls_tol))
        object.__setattr__(self, "rubberband_segments", segments)
        if self.rubberband_lam is not None:
            object.__setattr__(self, "rubberband_lam", float(self.rubberband_lam))
        object.__setattr__(self, "rubberband_diff_order", int(self.rubberband_diff_order))
        if self.rubberband_smooth_half_window is not None:
            object.__setattr__(
                self,
                "rubberband_smooth_half_window",
                int(self.rubberband_smooth_half_window),
            )


@dataclass(frozen=True, slots=True)
class SmoothingConfig(ConfigMixin):
    """Savitzky-Golay smoothing configuration; disabled by default."""

    enabled: bool = False
    window_length: int = 7
    polyorder: int = 2
    mode: str = "interp"

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"interp", "mirror", "constant", "nearest", "wrap"}:
            raise ValueError(f"Unsupported Savitzky-Golay mode: {self.mode!r}")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "window_length", int(self.window_length))
        object.__setattr__(self, "polyorder", int(self.polyorder))
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True, slots=True)
class NormalizationConfig(ConfigMixin):
    """Optional spectrum-wise normalization; disabled by default."""

    method: str = "none"
    reference_peak_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        method = _normalise_method(self.method)
        method = {"off": "none", "vector_normalization": "vector"}.get(method, method)
        if method not in {"none", "vector", "reference_peak"}:
            raise ValueError(f"Unsupported normalization method: {self.method!r}")
        peak_range = None
        if self.reference_peak_range is not None:
            if len(self.reference_peak_range) != 2:
                raise ValueError("reference_peak_range must contain exactly two wavenumbers")
            peak_range = (
                float(self.reference_peak_range[0]),
                float(self.reference_peak_range[1]),
            )
        if method == "reference_peak" and peak_range is None:
            raise ValueError("reference_peak normalization requires reference_peak_range")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "reference_peak_range", peak_range)


@dataclass(frozen=True, slots=True)
class PipelineConfig(ConfigMixin):
    """Complete reproducible pipeline configuration."""

    low_wavenumber: float | None = None
    high_wavenumber: float | None = None
    input_intensity_unit: str = "absorbance"
    perturbation_order: str = "preserve_file_order"
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    convention: str = "2dpy_compatible"
    grid_strategy: str = "index_order"
    contour_levels: int = 21
    display_percentile: float = 99.0

    def __post_init__(self) -> None:
        low = None if self.low_wavenumber is None else float(self.low_wavenumber)
        high = None if self.high_wavenumber is None else float(self.high_wavenumber)
        if (low is None) != (high is None):
            raise ValueError("low_wavenumber and high_wavenumber must be supplied together")

        unit = _normalise_method(self.input_intensity_unit)
        unit = {
            "a": "absorbance",
            "percent_t": "percent_transmittance",
            "fraction_t": "fraction_transmittance",
        }.get(unit, unit)
        if unit not in {
            "absorbance",
            "percent_transmittance",
            "fraction_transmittance",
            "unknown",
        }:
            raise ValueError(f"Unsupported input intensity unit: {self.input_intensity_unit!r}")

        order = _normalise_method(self.perturbation_order)
        if order not in {"preserve_file_order", "sort_by_perturbation"}:
            raise ValueError(
                "perturbation_order must be 'preserve_file_order' or 'sort_by_perturbation'"
            )
        convention = _normalise_method(self.convention)
        if convention not in {"canonical", "2dpy_compatible"}:
            raise ValueError("convention must be 'canonical' or '2dpy_compatible'")
        strategy = _normalise_method(self.grid_strategy)
        if strategy != "index_order":
            raise ValueError("The first release only supports grid_strategy='index_order'")
        if int(self.contour_levels) < 2:
            raise ValueError("contour_levels must be at least 2")
        if not 0 < float(self.display_percentile) <= 100:
            raise ValueError("display_percentile must be in the interval (0, 100]")

        baseline = (
            self.baseline
            if isinstance(self.baseline, BaselineConfig)
            else BaselineConfig.from_dict(self.baseline)
        )
        smoothing = (
            self.smoothing
            if isinstance(self.smoothing, SmoothingConfig)
            else SmoothingConfig.from_dict(self.smoothing)
        )
        normalization = (
            self.normalization
            if isinstance(self.normalization, NormalizationConfig)
            else NormalizationConfig.from_dict(self.normalization)
        )

        object.__setattr__(self, "low_wavenumber", low)
        object.__setattr__(self, "high_wavenumber", high)
        object.__setattr__(self, "input_intensity_unit", unit)
        object.__setattr__(self, "perturbation_order", order)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "smoothing", smoothing)
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "convention", convention)
        object.__setattr__(self, "grid_strategy", strategy)
        object.__setattr__(self, "contour_levels", int(self.contour_levels))
        object.__setattr__(self, "display_percentile", float(self.display_percentile))

    @property
    def wavenumber_range(self) -> tuple[float, float] | None:
        """Normalized ``(lower, upper)`` analysis range, if configured."""

        if self.low_wavenumber is None or self.high_wavenumber is None:
            return None
        return (
            min(self.low_wavenumber, self.high_wavenumber),
            max(self.low_wavenumber, self.high_wavenumber),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping including a convenient range field."""

        # Explicit dispatch is required for frozen, slotted dataclasses on
        # Python 3.13+; zero-argument super() can lose the class cell after the
        # dataclass transformation and raise TypeError.
        output = ConfigMixin.to_dict(self)
        output["wavenumber_range"] = (
            None if self.wavenumber_range is None else list(self.wavenumber_range)
        )
        return output

    def for_range(self, analysis_range: WavenumberRange | object) -> PipelineConfig:
        """Return this configuration with one explicit analysis interval."""

        normalized = WavenumberRange.from_value(analysis_range)
        return replace(
            self,
            low_wavenumber=normalized.low_wavenumber,
            high_wavenumber=normalized.high_wavenumber,
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> PipelineConfig:
        """Load a config, accepting either high/low fields or ``wavenumber_range``."""

        data = dict(values)
        range_value = data.pop("wavenumber_range", None)
        if range_value is not None:
            if len(range_value) != 2:
                raise ValueError("wavenumber_range must contain exactly two values")
            data.setdefault("low_wavenumber", float(range_value[0]))
            data.setdefault("high_wavenumber", float(range_value[1]))
        if isinstance(data.get("baseline"), Mapping):
            data["baseline"] = BaselineConfig.from_dict(data["baseline"])
        if isinstance(data.get("smoothing"), Mapping):
            data["smoothing"] = SmoothingConfig.from_dict(data["smoothing"])
        if isinstance(data.get("normalization"), Mapping):
            data["normalization"] = NormalizationConfig.from_dict(data["normalization"])
        return cls(**data)

    @classmethod
    def from_json(cls, source: str | Path) -> PipelineConfig:
        """Load configuration from JSON text or a JSON file path."""

        if isinstance(source, Path):
            payload = source.read_text(encoding="utf-8")
        else:
            stripped = source.lstrip()
            if stripped.startswith("{"):
                payload = source
            else:
                payload = Path(source).read_text(encoding="utf-8")
        parsed = json.loads(payload)
        if not isinstance(parsed, Mapping):
            raise ValueError("Pipeline configuration JSON must contain an object")
        return cls.from_dict(parsed)


def load_config(source: str | Path) -> PipelineConfig:
    """Load :class:`PipelineConfig` from a JSON file or JSON string."""

    return PipelineConfig.from_json(source)


def save_config(config: PipelineConfig, path: str | Path) -> Path:
    """Write a pipeline config as UTF-8 JSON and return the destination path."""

    destination = Path(path)
    destination.write_text(config.to_json() + "\n", encoding="utf-8")
    return destination


__all__ = [
    "BaselineConfig",
    "NormalizationConfig",
    "PipelineConfig",
    "SmoothingConfig",
    "WavenumberRange",
    "load_config",
    "save_config",
]
