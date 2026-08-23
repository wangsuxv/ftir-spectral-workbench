"""End-to-end scientific pipeline shared by the CLI and Streamlit UI.

The implementation deliberately keeps the mandatory order visible.  Baseline
estimation may use a smoothed copy, but every reported corrected spectrum is
formed from the un-smoothed absorbance channel.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import __version__
from .baseline.automatic import PYBASELINES_METHODS, estimate_coarse
from .baseline.collaborative import collaborative_pls_baseline
from .baseline.composite import compose_baselines, estimate_fine
from .baseline.shared_shape import shared_shape_baseline
from .config import PipelineConfig
from .models import BaselineResult, SpectrumSet, freeze_value, immutable_float64, thaw_mapping
from .normalization import NormalizationResult, apply_normalization
from .qc import QCResult, run_quality_control
from .ranges import crop_spectrum_set
from .smoothing import BaselineEstimationChannels, prepare_baseline_channels
from .units import UnitConversionRecord, convert_to_absorbance

FloatArray = NDArray[np.float64]

PROCESSING_ORDER = (
    "raw",
    "unit_confirmation",
    "transmittance_to_absorbance",
    "range_selection",
    "optional_estimate_only_smoothing",
    "coarse_baseline",
    "fixed_anchor_fine_baseline",
    "corrected_absorbance",
    "optional_normalization",
    "quality_control_and_export",
)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """All immutable scientific outputs and their reproducibility record."""

    raw_input: SpectrumSet
    absorbance_full: SpectrumSet
    absorbance_selected: SpectrumSet
    baseline_estimation_spectra: FloatArray
    baseline: BaselineResult
    normalization: NormalizationResult
    unit_conversion: UnitConversionRecord
    qc: QCResult
    config: PipelineConfig
    recipe: Mapping[str, Any]
    input_sha256: str
    software_version: str = __version__
    warnings: tuple[str, ...] = ()
    sensitivity_branches: Mapping[str, FloatArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        estimate = immutable_float64(
            self.baseline_estimation_spectra,
            name="baseline_estimation_spectra",
        )
        branches = {
            str(name): immutable_float64(values, name=f"sensitivity_branches.{name}")
            for name, values in self.sensitivity_branches.items()
        }
        object.__setattr__(self, "baseline_estimation_spectra", estimate)
        object.__setattr__(self, "recipe", freeze_value(dict(self.recipe), path="recipe"))
        object.__setattr__(
            self,
            "sensitivity_branches",
            freeze_value(branches, path="sensitivity_branches"),
        )
        object.__setattr__(self, "warnings", tuple(dict.fromkeys(map(str, self.warnings))))

    @property
    def analysis_data(self) -> FloatArray:
        """Primary quantitative/2D-COS data (always unnormalized corrected absorbance)."""

        return self.normalization.analysis_data

    @property
    def view_data(self) -> FloatArray:
        return self.normalization.view_data

    def recipe_dict(self) -> dict[str, Any]:
        """Return a detached JSON-serializable recipe."""

        return _jsonable(self.recipe)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return value


def _coerce_config(config: PipelineConfig | Mapping[str, Any] | str) -> PipelineConfig:
    if isinstance(config, PipelineConfig):
        return config
    if isinstance(config, str):
        payload = json.loads(config)
        if not isinstance(payload, Mapping):
            raise ValueError("pipeline recipe JSON root must be an object")
        config = payload
    nested = config.get("config")
    if isinstance(nested, Mapping):
        config = nested
    if hasattr(PipelineConfig, "model_validate"):
        return PipelineConfig.model_validate(config)
    return PipelineConfig.parse_obj(config)  # pragma: no cover - Pydantic 1


def _input_hash(data: SpectrumSet) -> str:
    for key in ("combined_source_sha256", "source_sha256"):
        value = data.metadata.get(key)
        if isinstance(value, str) and value:
            return value
    digest = hashlib.sha256()
    digest.update(np.asarray(data.wavenumber, dtype="<f8").tobytes())
    digest.update(np.asarray(data.perturbation, dtype="<f8").tobytes())
    digest.update(np.asarray(data.spectra, dtype="<f8").tobytes())
    digest.update(data.intensity_unit.encode("utf-8"))
    digest.update(data.source_name.encode("utf-8"))
    return digest.hexdigest()


def _spectrum_like(
    source: SpectrumSet,
    spectra: np.ndarray,
    *,
    intensity_unit: str,
    metadata_updates: Mapping[str, Any] | None = None,
) -> SpectrumSet:
    metadata = source.mutable_metadata()
    metadata.update(dict(metadata_updates or {}))
    return SpectrumSet(
        wavenumber=source.wavenumber,
        perturbation=source.perturbation,
        perturbation_labels=source.perturbation_labels,
        spectra=spectra,
        intensity_unit=intensity_unit,
        source_name=source.source_name,
        metadata=metadata,
    )


def _coarse_parameters(config: PipelineConfig) -> dict[str, Any]:
    coarse = config.coarse_baseline
    if coarse.method not in PYBASELINES_METHODS:
        return {}
    params: dict[str, Any] = {
        "lam": float(coarse.lam),
        "max_iter": int(coarse.max_iter),
        "tol": float(coarse.tol),
    }
    if coarse.method == "asls":
        params["p"] = float(coarse.p)
    return params


def _fine_parameters(config: PipelineConfig) -> dict[str, Any]:
    fine = config.fine_baseline
    params: dict[str, Any] = {"enabled": bool(fine.enabled)}
    if fine.method == "endpoint_window_linear":
        params.update(
            {
                "endpoint_window_width_cm1": float(fine.endpoint_window_width_cm1),
                "statistic": fine.statistic,
                "strict_endpoint": bool(fine.strict_endpoint),
            }
        )
    elif fine.method in {"piecewise_linear", "pchip", "polynomial"}:
        params.update(
            {
                "anchors": [anchor.to_dict() for anchor in fine.anchors],
                "statistic": fine.statistic,
            }
        )
        if fine.method == "polynomial":
            params["polynomial_order"] = int(fine.polynomial_order)
    return params


def endpoint_anchor_windows(x: np.ndarray, width_cm1: float) -> list[dict[str, Any]]:
    """Return fixed endpoint windows suitable for QC and shared-shape fitting."""

    low = float(np.min(x))
    high = float(np.max(x))
    half = float(width_cm1) / 2.0
    return [
        {"start": low - half, "end": low + half, "statistic": "median", "enabled": True},
        {
            "start": high - half,
            "end": high + half,
            "statistic": "median",
            "enabled": True,
        },
    ]


def fixed_anchor_windows(config: PipelineConfig, x: np.ndarray) -> list[dict[str, Any]]:
    fine = config.fine_baseline
    if fine.method == "endpoint_window_linear" or not fine.anchors:
        windows = endpoint_anchor_windows(x, fine.endpoint_window_width_cm1)
        for window in windows:
            window["statistic"] = fine.statistic
        return windows
    return [anchor.to_dict() for anchor in fine.anchors if anchor.enabled]


def _shared_shape_anchor_windows(config: PipelineConfig, x: np.ndarray) -> list[dict[str, Any]]:
    """Resolve the locked windows used by both shared-shape fitting and QC."""

    explicit = [anchor.to_dict() for anchor in config.fine_baseline.anchors if anchor.enabled]
    return explicit or fixed_anchor_windows(config, x)


def _estimate_baseline(
    x: np.ndarray,
    channels: BaselineEstimationChannels,
    config: PipelineConfig,
) -> BaselineResult:
    estimate_data = np.asarray(channels.for_baseline, dtype=np.float64)
    raw_data = np.asarray(channels.raw, dtype=np.float64)
    method = config.coarse_baseline.method
    coarse_params = _coarse_parameters(config)

    if config.series_mode == "shared_shape":
        if not config.fine_baseline.enabled or config.fine_baseline.method == "none":
            raise ValueError(
                "series_mode='shared_shape' intrinsically requires its fixed-window affine "
                "adjustment; fine_baseline must be enabled"
            )
        if config.fine_baseline.method != "endpoint_window_linear":
            raise ValueError(
                "series_mode='shared_shape' uses a common curve plus an affine fixed-window "
                "adjustment, not piecewise/PCHIP/polynomial interpolation; set "
                "fine_baseline.method='endpoint_window_linear' and optionally provide "
                "multiple anchor windows"
            )
        if config.fine_baseline.strict_endpoint:
            raise ValueError(
                "series_mode='shared_shape' requires robust anchor windows; strict endpoint "
                "points are not supported"
            )
        anchors = _shared_shape_anchor_windows(config, x)
        shared = shared_shape_baseline(
            x,
            estimate_data,
            reference_method=method,
            anchors=anchors,
            reference_params=coarse_params,
        )
        # shared_shape already decomposes B_ref and each spectrum's allowed
        # constant+slope adjustment.  Applying another fine fit here would
        # repeat the same scientific correction.
        return BaselineResult(
            coarse_baseline=shared.coarse_baseline,
            fine_baseline=shared.fine_baseline,
            total_baseline=shared.total_baseline,
            corrected=raw_data - np.asarray(shared.total_baseline),
            params={
                **thaw_mapping(shared.params),
                "fine_recipe_role": "anchors constrain shared offset and slope",
            },
            metrics=thaw_mapping(shared.metrics),
            warnings=shared.warnings,
        )

    if config.series_mode == "collaborative_pls":
        if method not in PYBASELINES_METHODS:
            raise ValueError(
                "series_mode='collaborative_pls' requires arpls, asls, airpls, or "
                f"pspline_arpls; got coarse method {method!r}"
            )
        coarse_result = collaborative_pls_baseline(
            x,
            estimate_data,
            method=method,
            **coarse_params,
        )
    elif config.series_mode == "independent_locked":
        coarse_result = estimate_coarse(x, estimate_data, method, **coarse_params)
    else:  # protected by Pydantic, retained for callers bypassing validation
        raise ValueError(f"unsupported series mode: {config.series_mode!r}")

    residual_for_fine = estimate_data - np.asarray(coarse_result.total_baseline)
    fine_result = estimate_fine(
        x,
        residual_for_fine,
        config.fine_baseline.method,
        **_fine_parameters(config),
    )
    return compose_baselines(raw_data, coarse_result, fine_result)


def run_pipeline(
    data: SpectrumSet,
    config: PipelineConfig | Mapping[str, Any] | str,
    *,
    peak_regions: Sequence[Any] | None = None,
) -> PipelineResult:
    """Run the mandatory FTIR processing sequence on one immutable input set."""

    recipe_config = _coerce_config(config)
    if data.intensity_unit != recipe_config.input_unit:
        raise ValueError(
            "recipe input_unit does not match the explicitly confirmed input data unit: "
            f"{recipe_config.input_unit!r} != {data.intensity_unit!r}"
        )
    input_sha256 = _input_hash(data)

    conversion = convert_to_absorbance(
        data.spectra,
        recipe_config.input_unit,
        transmittance_floor=recipe_config.transmittance_floor,
    )
    absorbance_full = _spectrum_like(
        data,
        np.asarray(conversion.absorbance),
        intensity_unit="absorbance",
        metadata_updates={"unit_conversion": conversion.record.to_dict()},
    )
    absorbance_selected = crop_spectrum_set(
        absorbance_full,
        recipe_config.wavenumber_range,
        strict_bounds=True,
    )
    channels = prepare_baseline_channels(
        absorbance_selected.spectra,
        recipe_config.baseline_smoothing,
    )
    baseline = _estimate_baseline(
        np.asarray(absorbance_selected.wavenumber),
        channels,
        recipe_config,
    )
    if recipe_config.series_mode == "shared_shape":
        anchors = _shared_shape_anchor_windows(
            recipe_config,
            absorbance_selected.wavenumber,
        )
    else:
        anchors = fixed_anchor_windows(recipe_config, absorbance_selected.wavenumber)
    normalization = apply_normalization(
        absorbance_selected.wavenumber,
        baseline.corrected,
        recipe_config.normalization,
    )
    qc = run_quality_control(
        absorbance_selected.wavenumber,
        absorbance_selected.spectra,
        baseline.total_baseline,
        baseline.corrected,
        anchor_windows=anchors,
        peak_regions=peak_regions,
        perturbation=absorbance_selected.perturbation,
    )
    warnings: list[str] = []
    if conversion.record.repaired:
        warnings.append(
            f"Applied explicit transmittance floor {conversion.record.transmittance_floor:g} "
            f"to {conversion.record.repaired_count} point(s); positions are in the recipe."
        )
    warnings.extend(baseline.warnings)
    warnings.extend(normalization.warnings)
    warnings.extend(qc.warnings)

    config_payload = recipe_config.to_dict()
    recipe: dict[str, Any] = {
        "software": {"name": "ftir-baseline-workbench", "version": __version__},
        "input_sha256": input_sha256,
        "input_source_name": data.source_name,
        "input_metadata": data.mutable_metadata(),
        "processing_order": list(PROCESSING_ORDER),
        "config": config_payload,
        # Duplicating top-level config keys keeps the human-readable recipe
        # self-contained; run_pipeline also accepts this complete audit envelope.
        **config_payload,
        "unit_conversion_record": conversion.record.to_dict(),
        "baseline_estimation_channel": dict(channels.settings),
        "baseline_fit": {
            "params": thaw_mapping(baseline.params),
            "metrics": thaw_mapping(baseline.metrics),
        },
        "quality_control": qc.as_dict(),
        "normalization_result": {
            "method": normalization.method,
            "params": _jsonable(normalization.params),
            "factors": np.asarray(normalization.factors).tolist(),
            "warnings": list(normalization.warnings),
        },
        "warnings": list(dict.fromkeys(warnings)),
    }
    sensitivity = {
        "uncorrected": absorbance_selected.spectra,
        "coarse_only": absorbance_selected.spectra - np.asarray(baseline.coarse_baseline),
        "coarse_plus_fine": baseline.corrected,
    }
    return PipelineResult(
        raw_input=data,
        absorbance_full=absorbance_full,
        absorbance_selected=absorbance_selected,
        baseline_estimation_spectra=channels.for_baseline,
        baseline=baseline,
        normalization=normalization,
        unit_conversion=conversion.record,
        qc=qc,
        config=recipe_config,
        recipe=recipe,
        input_sha256=input_sha256,
        warnings=tuple(warnings),
        sensitivity_branches=sensitivity,
    )


def pipeline_result_fingerprint(result: PipelineResult) -> str:
    """Hash all scientific outputs for byte-level reproducibility tests."""

    digest = hashlib.sha256()
    for array in (
        result.absorbance_full.spectra,
        result.absorbance_selected.spectra,
        result.baseline.coarse_baseline,
        result.baseline.fine_baseline,
        result.baseline.total_baseline,
        result.baseline.corrected,
        result.normalization.analysis_data,
        result.normalization.view_data,
    ):
        digest.update(np.asarray(array, dtype="<f8").tobytes())
    if result.normalization.optional_normalized is not None:
        digest.update(np.asarray(result.normalization.optional_normalized, dtype="<f8").tobytes())
    # Fitted metadata may contain platform-sensitive iteration histories; the
    # scientific array fingerprint is therefore the reproducibility primitive.
    digest.update(
        json.dumps(result.config.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


__all__ = [
    "PROCESSING_ORDER",
    "PipelineResult",
    "endpoint_anchor_windows",
    "fixed_anchor_windows",
    "pipeline_result_fingerprint",
    "run_pipeline",
]
