"""Configuration models for orchestration without duplicate scientific knobs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np

from ftir_baseline.config import PipelineConfig

# Phase 1 deliberately reuses the authoritative baseline recipe rather than
# defining a second set of algorithm fields.
BaselineWorkflowConfig = PipelineConfig

IntensityUnit = Literal[
    "absorbance",
    "percent_transmittance",
    "fraction_transmittance",
]
PerturbationOrderPolicy = Literal["preserve_file_order", "sort_by_perturbation"]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    return value


class WorkbenchConfigMixin:
    """Deterministic JSON helpers shared by frozen coordination configs."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        return cls(**dict(values))

    @classmethod
    def from_json(cls, source: str | Path) -> Self:
        if isinstance(source, Path):
            payload = source.read_text(encoding="utf-8")
        else:
            stripped = source.lstrip()
            payload = source if stripped.startswith("{") else Path(source).read_text("utf-8")
        values = json.loads(payload)
        if not isinstance(values, Mapping):
            raise ValueError(f"{cls.__name__} JSON root must be an object")
        return cls.from_dict(values)


@dataclass(frozen=True, slots=True)
class ImportConfig(WorkbenchConfigMixin):
    """Import decisions that must be explicit before baseline processing."""

    input_unit: IntensityUnit = "percent_transmittance"
    perturbation_order_policy: PerturbationOrderPolicy = "preserve_file_order"

    def __post_init__(self) -> None:
        unit = str(self.input_unit).strip().lower().replace("%t", "percent_transmittance")
        unit = {
            "a": "absorbance",
            "percent_t": "percent_transmittance",
            "fraction_t": "fraction_transmittance",
        }.get(unit, unit)
        if unit not in {
            "absorbance",
            "percent_transmittance",
            "fraction_transmittance",
        }:
            raise ValueError(f"unsupported input_unit: {self.input_unit!r}")
        policy = str(self.perturbation_order_policy).strip().lower().replace("-", "_")
        policy = {
            "preserve": "preserve_file_order",
            "file_order": "preserve_file_order",
            "sort": "sort_by_perturbation",
        }.get(policy, policy)
        if policy not in {"preserve_file_order", "sort_by_perturbation"}:
            raise ValueError(
                "perturbation_order_policy must be 'preserve_file_order' or "
                "'sort_by_perturbation'"
            )
        object.__setattr__(self, "input_unit", unit)
        object.__setattr__(self, "perturbation_order_policy", policy)

    @property
    def input_intensity_unit(self) -> str:
        """Compatibility spelling used by the legacy 2D configuration."""

        return self.input_unit

    @property
    def perturbation_order(self) -> str:
        return self.perturbation_order_policy

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_unit": self.input_unit,
            "perturbation_order_policy": self.perturbation_order_policy,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ImportConfig:
        data = dict(values)
        if "input_unit" not in data and "input_intensity_unit" in data:
            data["input_unit"] = data.pop("input_intensity_unit")
        if "perturbation_order_policy" not in data and "perturbation_order" in data:
            data["perturbation_order_policy"] = data.pop("perturbation_order")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class TwoDCOSRange(WorkbenchConfigMixin):
    """One real 2D analysis interval; disconnected intervals stay separate."""

    high_wavenumber: float
    low_wavenumber: float
    label: str | None = None

    def __post_init__(self) -> None:
        high = float(self.high_wavenumber)
        low = float(self.low_wavenumber)
        if not math.isfinite(high) or not math.isfinite(low):
            raise ValueError("TwoDCOSRange endpoints must be finite")
        if high == low:
            raise ValueError("TwoDCOSRange endpoints must be distinct")
        label = None if self.label is None else str(self.label).strip()
        if self.label is not None and not label:
            raise ValueError("TwoDCOSRange label cannot be empty")
        object.__setattr__(self, "high_wavenumber", max(high, low))
        object.__setattr__(self, "low_wavenumber", min(high, low))
        object.__setattr__(self, "label", label)

    @property
    def bounds(self) -> tuple[float, float]:
        return self.low_wavenumber, self.high_wavenumber

    @property
    def display_name(self) -> str:
        interval = f"{self.high_wavenumber:g}-{self.low_wavenumber:g} cm^-1"
        return interval if self.label is None else f"{self.label} ({interval})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "high_wavenumber": self.high_wavenumber,
            "low_wavenumber": self.low_wavenumber,
            "label": self.label,
        }

    @classmethod
    def from_value(cls, value: object) -> TwoDCOSRange:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            data = dict(value)
            high = data.get("high_wavenumber", data.get("high", data.get("upper")))
            low = data.get("low_wavenumber", data.get("low", data.get("lower")))
            if high is None or low is None:
                raise ValueError(
                    "range mappings require high_wavenumber and low_wavenumber"
                )
            return cls(high_wavenumber=high, low_wavenumber=low, label=data.get("label"))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) not in {2, 3}:
                raise ValueError("range sequences require two endpoints and an optional label")
            return cls(
                high_wavenumber=value[0],
                low_wavenumber=value[1],
                label=None if len(value) == 2 else value[2],
            )
        raise TypeError("2D ranges must be TwoDCOSRange, mapping, or 2/3-item sequence")


@dataclass(frozen=True, slots=True)
class TwoDCOSDisplayConfig(WorkbenchConfigMixin):
    """Plot-only settings that never participate in scientific fingerprints."""

    contour_levels: int = 21
    display_percentile: float = 99.0

    def __post_init__(self) -> None:
        levels = int(self.contour_levels)
        percentile = float(self.display_percentile)
        if levels < 2:
            raise ValueError("contour_levels must be at least 2")
        if not math.isfinite(percentile) or not 0.0 < percentile <= 100.0:
            raise ValueError("display_percentile must be finite and in (0, 100]")
        object.__setattr__(self, "contour_levels", levels)
        object.__setattr__(self, "display_percentile", percentile)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contour_levels": self.contour_levels,
            "display_percentile": self.display_percentile,
        }


def _coerce_ranges(value: object) -> tuple[TwoDCOSRange, ...]:
    if isinstance(value, (TwoDCOSRange, Mapping)):
        return (TwoDCOSRange.from_value(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = tuple(value)
        if len(raw) in {2, 3} and raw and not isinstance(
            raw[0], (TwoDCOSRange, Mapping, tuple, list)
        ):
            return (TwoDCOSRange.from_value(raw),)
        return tuple(TwoDCOSRange.from_value(item) for item in raw)
    raise TypeError("ranges must contain one or more explicit 2D intervals")


@dataclass(frozen=True, slots=True)
class TwoDCOSConfig(WorkbenchConfigMixin):
    """Prepared-only 2D-COS settings; no preprocessing fields are permitted."""

    ranges: tuple[TwoDCOSRange, ...]
    convention: Literal["canonical", "2dpy_compatible"] = "2dpy_compatible"
    grid_strategy: Literal["index_order"] = "index_order"
    nonuniform_perturbation_policy: Literal["warn", "allow", "error"] = "warn"
    peak_matching_tolerance: float = 5.0
    cross_range_enabled: bool = True
    display: TwoDCOSDisplayConfig = field(default_factory=TwoDCOSDisplayConfig)

    def __post_init__(self) -> None:
        ranges = _coerce_ranges(self.ranges)
        if not ranges:
            raise ValueError("TwoDCOSConfig.ranges must contain at least one interval")
        convention = str(self.convention).strip().lower().replace("-", "_")
        if convention not in {"canonical", "2dpy_compatible"}:
            raise ValueError("convention must be 'canonical' or '2dpy_compatible'")
        strategy = str(self.grid_strategy).strip().lower().replace("-", "_")
        if strategy != "index_order":
            raise ValueError("only grid_strategy='index_order' is supported")
        policy = (
            str(self.nonuniform_perturbation_policy)
            .strip()
            .lower()
            .replace("-", "_")
        )
        policy = {
            "reject": "error",
            "raise": "error",
            "allow_index_order": "allow",
        }.get(policy, policy)
        if policy not in {"warn", "allow", "error"}:
            raise ValueError(
                "nonuniform_perturbation_policy must be 'warn', 'allow', or 'error'"
            )
        tolerance = float(self.peak_matching_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("peak_matching_tolerance must be finite and non-negative")
        if not isinstance(self.cross_range_enabled, bool):
            raise TypeError("cross_range_enabled must be a bool")
        display = (
            self.display
            if isinstance(self.display, TwoDCOSDisplayConfig)
            else TwoDCOSDisplayConfig.from_dict(self.display)
        )
        object.__setattr__(self, "ranges", ranges)
        object.__setattr__(self, "convention", convention)
        object.__setattr__(self, "grid_strategy", strategy)
        object.__setattr__(self, "nonuniform_perturbation_policy", policy)
        object.__setattr__(self, "peak_matching_tolerance", tolerance)
        object.__setattr__(self, "display", display)

    def scientific_dict(self) -> dict[str, Any]:
        """Return only settings whose changes invalidate numerical 2D results."""

        return {
            "ranges": [item.to_dict() for item in self.ranges],
            "convention": self.convention,
            "grid_strategy": self.grid_strategy,
            "nonuniform_perturbation_policy": self.nonuniform_perturbation_policy,
            "peak_matching_tolerance": self.peak_matching_tolerance,
            "cross_range_enabled": self.cross_range_enabled,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.scientific_dict(), "display": self.display.to_dict()}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TwoDCOSConfig:
        data = dict(values)
        display_payload = data.pop("display", None)
        # Accept the former flat display fields only while loading archived JSON.
        legacy_display = {
            name: data.pop(name)
            for name in ("contour_levels", "display_percentile")
            if name in data
        }
        if display_payload is not None and legacy_display:
            raise ValueError("display fields must be nested under 'display', not duplicated")
        if display_payload is None:
            display_payload = legacy_display
        data["display"] = (
            display_payload
            if isinstance(display_payload, TwoDCOSDisplayConfig)
            else TwoDCOSDisplayConfig.from_dict(display_payload or {})
        )
        return cls(**data)


def _baseline_from_mapping(value: object) -> PipelineConfig | None:
    if value is None or isinstance(value, PipelineConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("baseline must be PipelineConfig, mapping, or None")
    if hasattr(PipelineConfig, "model_validate"):
        return PipelineConfig.model_validate(value)
    return PipelineConfig.parse_obj(value)  # pragma: no cover - Pydantic 1


def _baseline_scientific_dict(value: PipelineConfig | None) -> dict[str, Any] | None:
    if value is None:
        return None
    output = dict(value.to_dict())
    output.pop("normalization", None)
    return _plain(output)


@dataclass(frozen=True, slots=True)
class WorkbenchProjectConfig(WorkbenchConfigMixin):
    """Top-level immutable workbench configuration."""

    import_config: ImportConfig = field(default_factory=ImportConfig)
    baseline: BaselineWorkflowConfig | None = None
    twodcos: TwoDCOSConfig | None = None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        import_config = (
            self.import_config
            if isinstance(self.import_config, ImportConfig)
            else ImportConfig.from_dict(self.import_config)
        )
        baseline = _baseline_from_mapping(self.baseline)
        twodcos = (
            self.twodcos
            if self.twodcos is None or isinstance(self.twodcos, TwoDCOSConfig)
            else TwoDCOSConfig.from_dict(self.twodcos)
        )
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string")
        object.__setattr__(self, "import_config", import_config)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "twodcos", twodcos)
        object.__setattr__(self, "schema_version", self.schema_version.strip())

    @property
    def twodcos_display(self) -> TwoDCOSDisplayConfig | None:
        return None if self.twodcos is None else self.twodcos.display

    def scientific_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "import_config": self.import_config.to_dict(),
            "baseline": _baseline_scientific_dict(self.baseline),
            "twodcos": None if self.twodcos is None else self.twodcos.scientific_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "import_config": self.import_config.to_dict(),
            "baseline": None if self.baseline is None else self.baseline.to_dict(),
            "twodcos": None if self.twodcos is None else self.twodcos.to_dict(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> WorkbenchProjectConfig:
        data = dict(values)
        if isinstance(data.get("import_config"), Mapping):
            data["import_config"] = ImportConfig.from_dict(data["import_config"])
        data["baseline"] = _baseline_from_mapping(data.get("baseline"))
        if isinstance(data.get("twodcos"), Mapping):
            data["twodcos"] = TwoDCOSConfig.from_dict(data["twodcos"])
        return cls(**data)


__all__ = [
    "BaselineWorkflowConfig",
    "ImportConfig",
    "IntensityUnit",
    "PerturbationOrderPolicy",
    "TwoDCOSConfig",
    "TwoDCOSDisplayConfig",
    "TwoDCOSRange",
    "WorkbenchConfigMixin",
    "WorkbenchProjectConfig",
]
