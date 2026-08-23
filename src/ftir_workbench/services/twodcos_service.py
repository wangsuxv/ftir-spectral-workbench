"""Prepared-data-only orchestration for self and cross-range 2D-COS.

This module is intentionally a thin boundary around :mod:`ftir2dcos.twodcos`.
It validates and slices an already prepared absorbance matrix, then calls the
scientific core directly.  It must never import the legacy 2D pipeline,
conversion, baseline, smoothing, or normalization modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from numbers import Real

import numpy as np

from ftir2dcos.peak_order import (
    PeakOrderResult,
    PeakRequest,
    ResolvedPairValues,
    infer_peak_order,
)
from ftir2dcos.twodcos import (
    CrossTwoDCOSResult,
    TwoDCOSResult,
    compute_2dcos,
    compute_cross_2dcos,
)

from ..config import TwoDCOSConfig, TwoDCOSRange
from ..fingerprints import canonical_json_sha256, twodcos_fingerprint
from ..models import PreparedSpectralDataset
from ..validation import (
    PreparedDatasetValidationError,
    validate_cross_prepared_compatibility,
    validate_prepared_dataset,
)


class CrossPreparedConfirmationRequired(ValueError):
    """Raised before correlating two distinct prepared-data contracts."""


@dataclass(frozen=True, slots=True)
class HomoRangeResult:
    """One self-correlation block and its exact prepared-data parent."""

    analysis_range: TwoDCOSRange
    result: TwoDCOSResult
    parent_baseline_run_id: str
    parent_baseline_fingerprint: str
    parent_prepared_data_sha256: str

    @property
    def analysis(self) -> TwoDCOSResult:
        return self.result

    @property
    def synchronous(self) -> np.ndarray:
        return self.result.synchronous

    @property
    def asynchronous(self) -> np.ndarray:
        return self.result.asynchronous

    @property
    def dynamic(self) -> np.ndarray:
        return self.result.dynamic

    @property
    def qc_metrics(self) -> dict[str, float | bool]:
        return dict(self.result.qc_metrics)


@dataclass(frozen=True, slots=True)
class CrossRangeResult:
    """One rectangular cross-correlation block with both parent lineages."""

    first_range: TwoDCOSRange
    second_range: TwoDCOSRange
    result: CrossTwoDCOSResult
    first_parent_baseline_run_id: str
    second_parent_baseline_run_id: str
    first_parent_baseline_fingerprint: str
    second_parent_baseline_fingerprint: str
    first_parent_prepared_data_sha256: str
    second_parent_prepared_data_sha256: str
    first_parent_source_name: str
    second_parent_source_name: str
    first_parent_source_sha256: str
    second_parent_source_sha256: str
    different_prepared_blocks: bool = False
    different_prepared_blocks_confirmed: bool = False
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.different_prepared_blocks and not self.different_prepared_blocks_confirmed:
            raise ValueError(
                "a different-prepared-block cross result requires recorded confirmation"
            )
        if not self.different_prepared_blocks and self.different_prepared_blocks_confirmed:
            raise ValueError(
                "confirmation cannot be recorded when both ranges use the same prepared block"
            )
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))

    @property
    def analysis(self) -> CrossTwoDCOSResult:
        return self.result

    @property
    def synchronous(self) -> np.ndarray:
        return self.result.synchronous

    @property
    def asynchronous(self) -> np.ndarray:
        return self.result.asynchronous

    @property
    def qc_metrics(self) -> dict[str, float | bool]:
        return dict(self.result.qc_metrics)

    @property
    def parent_baseline_run_id(self) -> tuple[str, str]:
        return self.first_parent_baseline_run_id, self.second_parent_baseline_run_id

    @property
    def parent_baseline_fingerprint(self) -> tuple[str, str]:
        return (
            self.first_parent_baseline_fingerprint,
            self.second_parent_baseline_fingerprint,
        )

    @property
    def parent_prepared_data_sha256(self) -> tuple[str, str]:
        return (
            self.first_parent_prepared_data_sha256,
            self.second_parent_prepared_data_sha256,
        )

    @property
    def parent_source_sha256(self) -> tuple[str, str]:
        return self.first_parent_source_sha256, self.second_parent_source_sha256


@dataclass(frozen=True, slots=True)
class TwoDCOSAnalysisResult:
    """Complete multi-range 2D result rooted in one prepared dataset."""

    config: TwoDCOSConfig
    homo_results: tuple[HomoRangeResult, ...]
    cross_results: tuple[CrossRangeResult, ...]
    parent_baseline_run_id: str
    parent_baseline_fingerprint: str
    parent_prepared_data_sha256: str
    twodcos_fingerprint: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "homo_results", tuple(self.homo_results))
        object.__setattr__(self, "cross_results", tuple(self.cross_results))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        if not self.homo_results:
            raise ValueError("TwoDCOSAnalysisResult requires at least one homo result")
        for item in self.homo_results:
            if item.parent_prepared_data_sha256 != self.parent_prepared_data_sha256:
                raise ValueError("homo result does not match the parent prepared fingerprint")

    @property
    def self_results(self) -> tuple[HomoRangeResult, ...]:
        """Alias matching the common self/cross terminology."""

        return self.homo_results

    @property
    def all_checks_passed(self) -> bool:
        homo_ok = all(
            bool(item.result.qc_metrics.get("all_checks_passed", False))
            for item in self.homo_results
        )
        cross_ok = all(
            bool(item.result.qc_metrics.get("all_checks_passed", False))
            for item in self.cross_results
        )
        return homo_ok and cross_ok

    @property
    def result(self) -> TwoDCOSResult:
        """Expose the core result for the common one-range convenience case."""

        if len(self.homo_results) != 1:
            raise AttributeError("result is only available for a single homo range")
        return self.homo_results[0].result

    @property
    def analysis_range(self) -> TwoDCOSRange:
        if len(self.homo_results) != 1:
            raise AttributeError("analysis_range is only available for a single homo range")
        return self.homo_results[0].analysis_range

    @property
    def synchronous(self) -> np.ndarray:
        return self.result.synchronous

    @property
    def asynchronous(self) -> np.ndarray:
        return self.result.asynchronous

    @property
    def dynamic(self) -> np.ndarray:
        return self.result.dynamic

    @property
    def qc_metrics(self) -> dict[str, float | bool]:
        return dict(self.result.qc_metrics)


def _unique_warnings(*groups: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for group in groups:
        for raw_warning in group:
            warning = str(raw_warning)
            if warning not in seen:
                seen.add(warning)
                output.append(warning)
    return tuple(output)


def _validate_for_twodcos(prepared: PreparedSpectralDataset) -> None:
    if not isinstance(prepared, PreparedSpectralDataset):
        raise TypeError("2D-COS service accepts only PreparedSpectralDataset")
    validate_prepared_dataset(prepared)
    if prepared.intensity_unit != "absorbance":  # defensive: also enforced by the model
        raise PreparedDatasetValidationError("2D-COS prepared input must be absorbance")
    if prepared.n_spectra < 2:
        raise PreparedDatasetValidationError("2D-COS requires at least two spectra")


def _nonuniform_warnings(
    prepared: PreparedSpectralDataset,
    policy: str,
) -> tuple[str, ...]:
    perturbation = prepared.perturbation
    if perturbation.size < 3:
        return ()
    differences = np.diff(perturbation)
    scale = max(1.0, float(np.max(np.abs(differences))))
    uniform = bool(
        np.allclose(
            differences,
            differences[0],
            rtol=1.0e-9,
            atol=np.finfo(np.float64).eps * scale * 16.0,
        )
    )
    monotonic = bool(np.all(differences > 0.0) or np.all(differences < 0.0))
    if uniform and monotonic:
        return ()
    reasons: list[str] = []
    if not uniform:
        reasons.append("non-equally-spaced")
    if not monotonic:
        reasons.append("non-monotonic or repeated")
    warning = (
        "Perturbation values are "
        + " and ".join(reasons)
        + "; 2D-COS uses the preserved spectrum index order to construct the Noda matrix."
    )
    if policy == "error":
        raise PreparedDatasetValidationError(warning)
    return (warning,) if policy == "warn" else ()


def _validate_ranges(ranges: tuple[TwoDCOSRange, ...]) -> None:
    seen_bounds: set[tuple[float, float]] = set()
    seen_labels: set[str] = set()
    for analysis_range in ranges:
        bounds = analysis_range.bounds
        if bounds in seen_bounds:
            raise ValueError(f"duplicate 2D analysis interval: {bounds}")
        seen_bounds.add(bounds)
        if analysis_range.label is not None:
            folded = analysis_range.label.casefold()
            if folded in seen_labels:
                raise ValueError(
                    f"2D analysis range labels must be unique: {analysis_range.label!r}"
                )
            seen_labels.add(folded)


def _slice_prepared(
    prepared: PreparedSpectralDataset,
    analysis_range: TwoDCOSRange,
) -> tuple[np.ndarray, np.ndarray]:
    axis = prepared.wavenumber
    mask = (axis >= analysis_range.low_wavenumber) & (
        axis <= analysis_range.high_wavenumber
    )
    if not np.any(mask):
        raise ValueError(
            f"2D range {analysis_range.display_name} contains no sampled wavenumbers"
        )
    # Boolean indexing creates a detached block and preserves ascending or
    # descending axis order exactly.  No interpolation or concatenation occurs.
    return np.asarray(axis[mask], dtype=np.float64), np.asarray(
        prepared.spectra[:, mask], dtype=np.float64
    )


def _homo_range_result(
    prepared: PreparedSpectralDataset,
    analysis_range: TwoDCOSRange,
    config: TwoDCOSConfig,
) -> HomoRangeResult:
    axis, spectra = _slice_prepared(prepared, analysis_range)
    analysis = compute_2dcos(
        spectra,
        axis,
        convention=config.convention,
    )
    return HomoRangeResult(
        analysis_range=analysis_range,
        result=analysis,
        parent_baseline_run_id=prepared.baseline_run_id,
        parent_baseline_fingerprint=prepared.baseline_fingerprint,
        parent_prepared_data_sha256=prepared.prepared_data_sha256,
    )


def _cross_range_result(
    first: PreparedSpectralDataset,
    second: PreparedSpectralDataset,
    first_range: TwoDCOSRange,
    second_range: TwoDCOSRange,
    config: TwoDCOSConfig,
    *,
    confirm_different_prepared_blocks: bool = False,
) -> CrossRangeResult:
    if not isinstance(confirm_different_prepared_blocks, bool):
        raise TypeError("confirm_different_prepared_blocks must be a bool")
    compatibility_warnings = validate_cross_prepared_compatibility(first, second)
    different_prepared_blocks = first is not second
    if different_prepared_blocks and not confirm_different_prepared_blocks:
        raise CrossPreparedConfirmationRequired(
            "Cross-range calculation between two distinct PreparedSpectralDataset blocks "
            "requires explicit confirmation; pass "
            "confirm_different_prepared_blocks=True after reviewing both parents."
        )
    confirmation_warnings: tuple[str, ...] = ()
    if different_prepared_blocks:
        confirmation_warnings = (
            "Explicit confirmation recorded for cross-range calculation between two "
            "distinct prepared blocks: "
            f"first source_sha256={first.source_sha256}, "
            f"baseline_fingerprint={first.baseline_fingerprint}, "
            f"prepared_data_sha256={first.prepared_data_sha256}; "
            f"second source_sha256={second.source_sha256}, "
            f"baseline_fingerprint={second.baseline_fingerprint}, "
            f"prepared_data_sha256={second.prepared_data_sha256}.",
        )
    first_axis, first_spectra = _slice_prepared(first, first_range)
    second_axis, second_spectra = _slice_prepared(second, second_range)
    analysis = compute_cross_2dcos(
        first_spectra,
        second_spectra,
        first_axis,
        second_axis,
        convention=config.convention,
    )
    return CrossRangeResult(
        first_range=first_range,
        second_range=second_range,
        result=analysis,
        first_parent_baseline_run_id=first.baseline_run_id,
        second_parent_baseline_run_id=second.baseline_run_id,
        first_parent_baseline_fingerprint=first.baseline_fingerprint,
        second_parent_baseline_fingerprint=second.baseline_fingerprint,
        first_parent_prepared_data_sha256=first.prepared_data_sha256,
        second_parent_prepared_data_sha256=second.prepared_data_sha256,
        first_parent_source_name=first.source_name,
        second_parent_source_name=second.source_name,
        first_parent_source_sha256=first.source_sha256,
        second_parent_source_sha256=second.source_sha256,
        different_prepared_blocks=different_prepared_blocks,
        different_prepared_blocks_confirmed=(
            different_prepared_blocks and confirm_different_prepared_blocks
        ),
        warnings=_unique_warnings(compatibility_warnings, confirmation_warnings),
    )


def compute_homo_from_prepared(
    prepared: PreparedSpectralDataset,
    config: TwoDCOSConfig,
    analysis_range: TwoDCOSRange | None = None,
) -> TwoDCOSAnalysisResult:
    """Compute one or all configured self-correlation ranges without preprocessing."""

    _validate_for_twodcos(prepared)
    effective_config = config
    if analysis_range is not None:
        selected_range = TwoDCOSRange.from_value(analysis_range)
        effective_config = replace(
            config,
            ranges=(selected_range,),
            cross_range_enabled=False,
        )
    ranges = effective_config.ranges
    _validate_ranges(ranges)
    policy_warnings = _nonuniform_warnings(
        prepared,
        effective_config.nonuniform_perturbation_policy,
    )
    homo_results = tuple(
        _homo_range_result(prepared, item, effective_config) for item in ranges
    )
    return TwoDCOSAnalysisResult(
        config=effective_config,
        homo_results=homo_results,
        cross_results=(),
        parent_baseline_run_id=prepared.baseline_run_id,
        parent_baseline_fingerprint=prepared.baseline_fingerprint,
        parent_prepared_data_sha256=prepared.prepared_data_sha256,
        twodcos_fingerprint=twodcos_fingerprint(prepared, effective_config),
        warnings=_unique_warnings(prepared.warnings, policy_warnings),
    )


def compute_cross_from_prepared(
    first: PreparedSpectralDataset,
    second: PreparedSpectralDataset,
    config: TwoDCOSConfig,
    first_range: TwoDCOSRange | None = None,
    second_range: TwoDCOSRange | None = None,
    *,
    confirm_different_prepared_blocks: bool = False,
) -> CrossRangeResult:
    """Compute one rectangular block after validation and explicit consent.

    Passing two different prepared objects is blocked by default even when
    their perturbations and labels are exactly compatible.  The caller must
    explicitly confirm that the two baseline/prepared provenance chains were
    reviewed.  Passing the same prepared object for two spectral subranges does
    not require confirmation.
    """

    _validate_for_twodcos(first)
    _validate_for_twodcos(second)
    first_policy_warnings = _nonuniform_warnings(
        first, config.nonuniform_perturbation_policy
    )
    second_policy_warnings = _nonuniform_warnings(
        second, config.nonuniform_perturbation_policy
    )
    first_range = (
        config.ranges[0]
        if first_range is None
        else TwoDCOSRange.from_value(first_range)
    )
    if second_range is None:
        second_range = config.ranges[1] if len(config.ranges) > 1 else config.ranges[0]
    else:
        second_range = TwoDCOSRange.from_value(second_range)
    result = _cross_range_result(
        first,
        second,
        first_range,
        second_range,
        config,
        confirm_different_prepared_blocks=confirm_different_prepared_blocks,
    )
    return replace(
        result,
        warnings=_unique_warnings(
            first.warnings,
            second.warnings,
            first_policy_warnings,
            second_policy_warnings,
            result.warnings,
        ),
    )


class TwoDCOSWorkflowService:
    """Orchestrate all configured self blocks and unique same-parent cross blocks."""

    def compute(
        self,
        prepared: PreparedSpectralDataset,
        config: TwoDCOSConfig,
    ) -> TwoDCOSAnalysisResult:
        _validate_for_twodcos(prepared)
        _validate_ranges(config.ranges)
        policy_warnings = _nonuniform_warnings(
            prepared, config.nonuniform_perturbation_policy
        )

        homo_results = tuple(
            _homo_range_result(prepared, analysis_range, config)
            for analysis_range in config.ranges
        )
        cross_results: tuple[CrossRangeResult, ...] = ()
        if config.cross_range_enabled and len(config.ranges) > 1:
            cross_results = tuple(
                _cross_range_result(prepared, prepared, first_range, second_range, config)
                for first_range, second_range in combinations(config.ranges, 2)
            )
        warnings = _unique_warnings(
            prepared.warnings,
            policy_warnings,
            *(item.warnings for item in cross_results),
        )
        return TwoDCOSAnalysisResult(
            config=config,
            homo_results=homo_results,
            cross_results=cross_results,
            parent_baseline_run_id=prepared.baseline_run_id,
            parent_baseline_fingerprint=prepared.baseline_fingerprint,
            parent_prepared_data_sha256=prepared.prepared_data_sha256,
            twodcos_fingerprint=twodcos_fingerprint(prepared, config),
            warnings=warnings,
        )

    run = compute


def cross_result_fingerprint(
    result: CrossRangeResult,
    config: TwoDCOSConfig,
) -> str:
    """Fingerprint a potentially two-parent cross calculation for exporters."""

    return canonical_json_sha256(
        {
            "schema": "ftir-workbench-cross-2dcos-v1",
            "first_prepared_data_sha256": result.first_parent_prepared_data_sha256,
            "second_prepared_data_sha256": result.second_parent_prepared_data_sha256,
            "first_baseline_run_id": result.first_parent_baseline_run_id,
            "second_baseline_run_id": result.second_parent_baseline_run_id,
            "first_baseline_fingerprint": result.first_parent_baseline_fingerprint,
            "second_baseline_fingerprint": result.second_parent_baseline_fingerprint,
            "first_source_name": result.first_parent_source_name,
            "second_source_name": result.second_parent_source_name,
            "first_source_sha256": result.first_parent_source_sha256,
            "second_source_sha256": result.second_parent_source_sha256,
            "different_prepared_blocks": result.different_prepared_blocks,
            "different_prepared_blocks_confirmed": (
                result.different_prepared_blocks_confirmed
            ),
            "first_range": result.first_range.to_dict(),
            "second_range": result.second_range.to_dict(),
            "config": config.scientific_dict(),
        }
    )


@dataclass(frozen=True, slots=True)
class _MatchedPeak:
    request: PeakRequest
    range_index: int
    grid_index: int
    matched_wavenumber: float
    distance: float


def _coerce_peak_request(value: PeakRequest | Real | Mapping[str, object]) -> PeakRequest:
    if isinstance(value, PeakRequest):
        return value
    if isinstance(value, Real) and not isinstance(value, bool):
        return PeakRequest(float(value))
    if isinstance(value, Mapping):
        return PeakRequest(
            wavenumber=value["wavenumber"],  # type: ignore[arg-type]
            label=value.get("label"),  # type: ignore[arg-type]
            range_index=value.get("range_index"),  # type: ignore[arg-type]
        )
    raise TypeError("peaks must contain PeakRequest, real numbers, or mappings")


def _nearest_grid(axis: np.ndarray, requested: float) -> tuple[int, float, float]:
    numeric_axis = np.asarray(axis, dtype=np.float64)
    distances = np.abs(numeric_axis - requested)
    minimum = float(np.min(distances))
    magnitude = max(1.0, abs(requested), float(np.max(np.abs(numeric_axis))))
    tie_tolerance = 16.0 * float(np.spacing(magnitude))
    candidates = np.flatnonzero(
        np.isclose(distances, minimum, rtol=0.0, atol=tie_tolerance)
    )
    candidate_values = np.unique(numeric_axis[candidates])
    if candidate_values.size > 1:
        rendered = ", ".join(f"{value:g}" for value in candidate_values)
        raise ValueError(
            f"Requested peak {requested:g} cm^-1 is exactly equidistant from "
            f"sampled grid points {rendered}; choose a non-midpoint value"
        )
    index = int(candidates[0])
    return index, float(numeric_axis[index]), minimum


def _match_peak_requests(
    result: TwoDCOSAnalysisResult,
    peaks: Sequence[PeakRequest],
    *,
    tolerance: float,
) -> tuple[_MatchedPeak, ...]:
    matched: list[_MatchedPeak] = []
    for peak in peaks:
        if peak.range_index is None:
            candidate_indices = tuple(range(len(result.homo_results)))
        else:
            if peak.range_index >= len(result.homo_results):
                raise ValueError(
                    f"Peak {peak.display_label} specifies range {peak.range_index + 1}, "
                    f"but only {len(result.homo_results)} range(s) were analyzed"
                )
            candidate_indices = (peak.range_index,)

        candidates: list[_MatchedPeak] = []
        for range_index in candidate_indices:
            analysis = result.homo_results[range_index].result
            grid_index, matched_wavenumber, distance = _nearest_grid(
                analysis.row_wavenumber,
                peak.wavenumber,
            )
            if distance <= tolerance:
                candidates.append(
                    _MatchedPeak(
                        request=replace(peak, range_index=range_index),
                        range_index=range_index,
                        grid_index=grid_index,
                        matched_wavenumber=matched_wavenumber,
                        distance=distance,
                    )
                )
        if not candidates:
            raise ValueError(
                f"Peak {peak.display_label} has no sampled grid point within "
                f"{tolerance:g} cm^-1"
            )
        if len(candidates) > 1:
            rendered = ", ".join(
                f"range {item.range_index + 1} ({item.matched_wavenumber:g} cm^-1)"
                for item in candidates
            )
            raise ValueError(
                f"Peak {peak.display_label} is ambiguous across ranges: {rendered}; "
                "specify range_index"
            )
        matched.append(candidates[0])

    for first, second in combinations(matched, 2):
        same_grid_point = (
            first.range_index == second.range_index
            and first.grid_index == second.grid_index
        )
        same_physical_position = bool(
            np.isclose(
                first.matched_wavenumber,
                second.matched_wavenumber,
                rtol=1.0e-10,
                atol=1.0e-8,
            )
        )
        if same_grid_point or same_physical_position:
            raise ValueError(
                f"Peaks {first.request.display_label} and {second.request.display_label} "
                "resolve to the same physical spectral variable"
            )
    return tuple(matched)


def _cross_for_range_indices(
    result: TwoDCOSAnalysisResult,
    first_index: int,
    second_index: int,
) -> tuple[CrossRangeResult, int, int]:
    low_index, high_index = sorted((first_index, second_index))
    first_range = result.homo_results[low_index].analysis_range
    second_range = result.homo_results[high_index].analysis_range
    for cross in result.cross_results:
        if cross.first_range == first_range and cross.second_range == second_range:
            return cross, low_index, high_index
    raise ValueError(
        f"Missing cross-range matrices for analysis ranges {low_index + 1} and "
        f"{high_index + 1}; enable cross_range_enabled before peak-order analysis"
    )


def _matrix_scale(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _sample_peak_pair(
    result: TwoDCOSAnalysisResult,
    first: _MatchedPeak,
    second: _MatchedPeak,
    *,
    tolerance: float,
) -> ResolvedPairValues:
    # The Noda rule contract is always canonical[row=first,column=second],
    # independent of the configured plotting/export convention.
    if first.matched_wavenumber <= second.matched_wavenumber:
        raise ValueError("peak pair must be oriented from higher to lower wavenumber")

    if first.range_index == second.range_index:
        analysis = result.homo_results[first.range_index].result
        synchronous_matrix = analysis.canonical_synchronous
        asynchronous_matrix = analysis.canonical_asynchronous
        synchronous = float(synchronous_matrix[first.grid_index, second.grid_index])
        asynchronous = float(asynchronous_matrix[first.grid_index, second.grid_index])
        source = f"range_{first.range_index + 1}_canonical_self"
        orientation = "canonical[row=higher_peak,column=lower_peak]"
        matrix_ranges = [first.range_index + 1, first.range_index + 1]
    else:
        cross, stored_first_index, stored_second_index = _cross_for_range_indices(
            result,
            first.range_index,
            second.range_index,
        )
        synchronous_matrix = cross.result.canonical_synchronous
        asynchronous_matrix = cross.result.canonical_asynchronous
        matrix_ranges = [stored_first_index + 1, stored_second_index + 1]
        if first.range_index == stored_first_index:
            synchronous = float(synchronous_matrix[first.grid_index, second.grid_index])
            asynchronous = float(asynchronous_matrix[first.grid_index, second.grid_index])
            source = (
                f"cross_ranges_{stored_first_index + 1}_{stored_second_index + 1}_canonical"
            )
            orientation = "canonical[row=higher_peak,column=lower_peak]"
        else:
            synchronous = float(synchronous_matrix[second.grid_index, first.grid_index])
            asynchronous = -float(
                asynchronous_matrix[second.grid_index, first.grid_index]
            )
            source = (
                f"cross_ranges_{stored_first_index + 1}_{stored_second_index + 1}_"
                "canonical_reverse_identity"
            )
            orientation = "reverse canonical via Phi21=Phi12.T; Psi21=-Psi12.T"

    sync_scale = _matrix_scale(synchronous_matrix)
    async_scale = _matrix_scale(asynchronous_matrix)
    strength = None
    if sync_scale > 0.0 and async_scale > 0.0:
        strength = min(
            1.0,
            abs(synchronous) / sync_scale,
            abs(asynchronous) / async_scale,
        )
    return ResolvedPairValues(
        first=first.request,
        second=second.request,
        synchronous=synchronous,
        asynchronous=asynchronous,
        matched_first_wavenumber=first.matched_wavenumber,
        matched_second_wavenumber=second.matched_wavenumber,
        relative_signal_strength=strength,
        source=source,
        metadata={
            "pair_orientation": "higher_matched_wavenumber_first",
            "canonical_sampling": orientation,
            "matrix_range_indices_one_based": matrix_ranges,
            "first_grid_index_zero_based": first.grid_index,
            "second_grid_index_zero_based": second.grid_index,
            "first_match_distance_cm-1": first.distance,
            "second_match_distance_cm-1": second.distance,
            "match_tolerance_cm-1": tolerance,
            "sync_matrix_max_abs": sync_scale,
            "async_matrix_max_abs": async_scale,
            "parent_twodcos_fingerprint": result.twodcos_fingerprint,
        },
    )


def analyze_peak_order(
    result: TwoDCOSAnalysisResult,
    peaks: Sequence[PeakRequest | Real | Mapping[str, object]],
    *,
    tolerance: float | None = None,
) -> PeakOrderResult:
    """Infer apparent response order from existing prepared-only 2D blocks.

    This function performs only range selection, nearest-grid matching, and
    convention-independent canonical matrix sampling.  The Noda sign rule,
    ambiguity handling, and graph inference remain exclusively implemented by
    :func:`ftir2dcos.peak_order.infer_peak_order`.
    """

    if not isinstance(result, TwoDCOSAnalysisResult):
        raise TypeError("result must be a TwoDCOSAnalysisResult")
    requested = tuple(_coerce_peak_request(item) for item in peaks)
    if not requested:
        raise ValueError("peaks must contain at least one peak")
    effective_tolerance = (
        result.config.peak_matching_tolerance if tolerance is None else float(tolerance)
    )
    if not np.isfinite(effective_tolerance) or effective_tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    matched = _match_peak_requests(
        result,
        requested,
        tolerance=effective_tolerance,
    )
    pair_values: list[ResolvedPairValues] = []
    for left, right in combinations(matched, 2):
        first, second = (
            (left, right)
            if left.matched_wavenumber > right.matched_wavenumber
            else (right, left)
        )
        pair_values.append(
            _sample_peak_pair(
                result,
                first,
                second,
                tolerance=effective_tolerance,
            )
        )
    inferred = infer_peak_order(
        tuple(item.request for item in matched),
        tuple(pair_values),
    )
    return replace(
        inferred,
        warnings=_unique_warnings(inferred.warnings, result.warnings),
    )


# Friendly aliases used by callers and older implementation notes.
PreparedTwoDCOSService = TwoDCOSWorkflowService
TwoDCOSWorkflowResult = TwoDCOSAnalysisResult


__all__ = [
    "CrossPreparedConfirmationRequired",
    "CrossRangeResult",
    "HomoRangeResult",
    "PreparedTwoDCOSService",
    "TwoDCOSAnalysisResult",
    "TwoDCOSWorkflowResult",
    "TwoDCOSWorkflowService",
    "analyze_peak_order",
    "compute_cross_from_prepared",
    "compute_homo_from_prepared",
    "cross_result_fingerprint",
]
