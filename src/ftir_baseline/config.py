"""Serializable processing recipe models.

The project targets Pydantic 2, while retaining a small compatibility surface for
Pydantic 1 environments commonly found in scientific Python installations.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

try:  # Pydantic 2
    from pydantic import ConfigDict

    _PYDANTIC_V2 = hasattr(BaseModel, "model_dump")
except ImportError:  # pragma: no cover - exercised only with Pydantic 1
    ConfigDict = None  # type: ignore[assignment,misc]
    _PYDANTIC_V2 = False

IntensityUnit = Literal[
    "absorbance",
    "percent_transmittance",
    "fraction_transmittance",
]
Statistic = Literal["median", "mean"]


class RecipeModel(BaseModel):
    """Base class with stable JSON aliases and v1/v2 serialization methods."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(
            populate_by_name=True,
            extra="forbid",
            frozen=True,
            validate_default=True,
            allow_inf_nan=False,
        )
    else:  # pragma: no cover - exercised only with Pydantic 1

        class Config:
            allow_population_by_field_name = True
            extra = "forbid"
            allow_mutation = False
            validate_all = True
            allow_inf_nan = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize with public JSON aliases (notably ``lambda``)."""

        if _PYDANTIC_V2:
            return self.model_dump(mode="json", by_alias=True)
        return json.loads(self.json(by_alias=True))

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to reproducible JSON using the same aliases as ``to_dict``."""

        if _PYDANTIC_V2:
            return self.model_dump_json(by_alias=True, indent=indent)
        return self.json(by_alias=True, indent=indent)

    @classmethod
    def from_json(cls, payload: str) -> RecipeModel:
        if _PYDANTIC_V2:
            return cls.model_validate_json(payload)
        return cls.parse_raw(payload)


class SmoothingConfig(RecipeModel):
    enabled: bool = False
    method: Literal["savgol"] = "savgol"
    window_length: int = Field(default=7, ge=3)
    polyorder: int = Field(default=2, ge=0)
    estimate_only: bool = True

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.window_length % 2 == 0:
            raise ValueError("baseline_smoothing.window_length must be odd")
        if self.window_length <= self.polyorder:
            raise ValueError("baseline_smoothing.window_length must be greater than polyorder")
        if self.enabled and not self.estimate_only:
            raise ValueError(
                "baseline smoothing is estimate-only and may not replace the raw spectrum"
            )


class CoarseBaselineConfig(RecipeModel):
    method: Literal[
        "none",
        "offset",
        "linear",
        "arpls",
        "asls",
        "airpls",
        "rubberband",
        "pspline_arpls",
    ] = "arpls"
    lam: float = Field(default=1_000_000.0, alias="lambda", gt=0.0)
    p: float = Field(default=0.01, gt=0.0, lt=0.5)
    max_iter: int = Field(default=50, ge=1)
    tol: float = Field(default=0.001, gt=0.0)


class AnchorWindowConfig(RecipeModel):
    enabled: bool = True
    start: float
    end: float
    statistic: Statistic = "median"

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.start == self.end:
            raise ValueError("anchor window start and end must differ")


class FineBaselineConfig(RecipeModel):
    enabled: bool = True
    method: Literal[
        "none",
        "endpoint_window_linear",
        "piecewise_linear",
        "pchip",
        "polynomial",
    ] = "endpoint_window_linear"
    endpoint_window_width_cm1: float = Field(default=8.0, gt=0.0)
    statistic: Statistic = "median"
    strict_endpoint: bool = False
    anchors: tuple[AnchorWindowConfig, ...] = ()
    polynomial_order: int = Field(default=1, ge=1, le=3)


class NormalizationConfig(RecipeModel):
    method: Literal[
        "none",
        "internal_peak_height",
        "internal_peak_area",
        "vector",
        "area",
        "minmax_display",
    ] = "none"
    internal_reference_range: tuple[float, float] | None = None
    integration_range: tuple[float, float] | None = None
    absolute: bool = True
    target_value: float = Field(default=1.0, gt=0.0)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.method.startswith("internal_peak") and self.internal_reference_range is None:
            raise ValueError(
                "normalization.internal_reference_range is required for internal peak methods"
            )
        if self.method == "area" and self.integration_range is None:
            raise ValueError("normalization.integration_range is required for area normalization")
        for name in ("internal_reference_range", "integration_range"):
            bounds = getattr(self, name)
            if bounds is not None and bounds[0] == bounds[1]:
                raise ValueError(f"normalization.{name} bounds must differ")


class PipelineConfig(RecipeModel):
    input_unit: IntensityUnit = "percent_transmittance"
    wavenumber_range: tuple[float, float] = (1800.0, 900.0)
    baseline_smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    coarse_baseline: CoarseBaselineConfig = Field(default_factory=CoarseBaselineConfig)
    fine_baseline: FineBaselineConfig = Field(default_factory=FineBaselineConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    series_mode: Literal[
        "independent_locked",
        "collaborative_pls",
        "shared_shape",
    ] = "collaborative_pls"
    restore_descending_axis_on_export: bool = True
    transmittance_floor: float | None = Field(default=None, gt=0.0)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.wavenumber_range[0] == self.wavenumber_range[1]:
            raise ValueError("wavenumber_range bounds must differ")


__all__ = [
    "AnchorWindowConfig",
    "CoarseBaselineConfig",
    "FineBaselineConfig",
    "IntensityUnit",
    "NormalizationConfig",
    "PipelineConfig",
    "RecipeModel",
    "SmoothingConfig",
    "Statistic",
]
