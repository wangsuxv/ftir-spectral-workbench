"""Shared, auditable preprocessing and auto-/cross-range 2D-COS pipeline."""

from __future__ import annotations

import csv
import fnmatch
import json
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import combinations
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig, WavenumberRange
from .conversion import convert_to_absorbance
from .io import ReadableSource, load_input
from .models import SpectralDataset, ValidationReport
from .peak_order import (
    PeakOrderResult,
    PeakRequest,
    ResolvedPairValues,
    infer_peak_order,
)
from .preprocessing import correct_baseline, normalize_dataset, smooth_dataset
from .twodcos import (
    CrossTwoDCOSResult,
    TwoDCOSResult,
    compute_2dcos,
    compute_cross_2dcos,
)
from .validation import (
    apply_perturbation_order,
    ensure_ascending_wavenumber,
    select_wavenumber_range,
    validate_dataset,
)


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """All inspectable states produced before the 2D-COS calculation."""

    imported: SpectralDataset
    selected_raw: SpectralDataset
    baselines: np.ndarray
    baseline_corrected: SpectralDataset
    processed: SpectralDataset
    baseline_diagnostics: tuple[dict[str, Any], ...]
    input_validation: ValidationReport
    selected_validation: ValidationReport
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        baseline = np.array(self.baselines, dtype=np.float64, copy=True, order="C")
        if baseline.shape != self.selected_raw.spectra.shape:
            raise ValueError(
                "baselines must match selected_raw.spectra; "
                f"got {baseline.shape} and {self.selected_raw.spectra.shape}"
            )
        if not np.all(np.isfinite(baseline)):
            raise ValueError("baselines contain NaN or infinite values")
        baseline.setflags(write=False)
        object.__setattr__(self, "baselines", baseline)
        object.__setattr__(
            self,
            "baseline_diagnostics",
            tuple(dict(item) for item in self.baseline_diagnostics),
        )
        object.__setattr__(self, "warnings", tuple(dict.fromkeys(map(str, self.warnings))))


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Complete scientific result shared by the CLI, Streamlit, and exporters."""

    imported: SpectralDataset
    selected_raw: SpectralDataset
    baselines: np.ndarray
    baseline_corrected: SpectralDataset
    processed: SpectralDataset
    baseline_diagnostics: tuple[dict[str, Any], ...]
    twodcos: TwoDCOSResult
    warnings: tuple[str, ...]
    input_validation: ValidationReport
    selected_validation: ValidationReport
    output_directory: Path | None = None

    @property
    def qc_metrics(self) -> dict[str, float | bool]:
        """Return the numerical 2D-COS quality metrics."""

        return dict(self.twodcos.qc_metrics)

    @property
    def convention(self) -> str:
        """Return the active matrix orientation convention."""

        return self.twodcos.convention

    @property
    def bundle_path(self) -> Path | None:
        """Return the ZIP path when this run was exported."""

        if self.output_directory is None:
            return None
        return self.output_directory / "run_bundle.zip"


@dataclass(frozen=True, slots=True)
class RangePipelineResult:
    """One labelled interval and its complete independent pipeline result."""

    analysis_range: WavenumberRange
    result: PipelineResult

    @property
    def output_directory(self) -> Path | None:
        """Return this interval's export directory, when exported."""

        return self.result.output_directory

    @property
    def bundle_path(self) -> Path | None:
        """Return this interval's standalone ZIP, when exported."""

        return self.result.bundle_path


@dataclass(frozen=True, slots=True)
class CrossRangePipelineResult:
    """One unique rectangular cross-correlation between two analysis intervals."""

    first_index: int
    second_index: int
    first_range: WavenumberRange
    second_range: WavenumberRange
    twodcos: CrossTwoDCOSResult
    output_directory: Path | None = None

    def __post_init__(self) -> None:
        first_index = int(self.first_index)
        second_index = int(self.second_index)
        if first_index < 0 or second_index <= first_index:
            raise ValueError("Cross-range indices must satisfy 0 <= first_index < second_index")
        object.__setattr__(self, "first_index", first_index)
        object.__setattr__(self, "second_index", second_index)
        object.__setattr__(self, "first_range", WavenumberRange.from_value(self.first_range))
        object.__setattr__(self, "second_range", WavenumberRange.from_value(self.second_range))
        if self.output_directory is not None:
            object.__setattr__(self, "output_directory", Path(self.output_directory))

    @property
    def pair_label(self) -> str:
        """Return a readable first-range versus second-range label."""

        return f"{self.first_range.display_name} x {self.second_range.display_name}"

    @property
    def qc_metrics(self) -> dict[str, float | bool]:
        """Return numerical cross-correlation quality metrics."""

        return dict(self.twodcos.qc_metrics)

    @property
    def row_index(self) -> int:
        """Return the range index represented by exported matrix rows."""

        return self.first_index if self.twodcos.convention == "canonical" else self.second_index

    @property
    def column_index(self) -> int:
        """Return the range index represented by exported matrix columns."""

        return self.second_index if self.twodcos.convention == "canonical" else self.first_index

    @property
    def row_range(self) -> WavenumberRange:
        """Return the range represented by exported matrix rows."""

        return self.first_range if self.row_index == self.first_index else self.second_range

    @property
    def column_range(self) -> WavenumberRange:
        """Return the range represented by exported matrix columns."""

        return self.first_range if self.column_index == self.first_index else self.second_range


@dataclass(frozen=True, slots=True)
class MultiRangePipelineResult:
    """Complete result for an ordered collection of analysis intervals."""

    range_results: tuple[RangePipelineResult, ...]
    cross_results: tuple[CrossRangePipelineResult, ...] = ()
    warnings: tuple[str, ...] = ()
    output_directory: Path | None = None
    peak_order: PeakOrderResult | None = None

    def __post_init__(self) -> None:
        normalized = tuple(self.range_results)
        if not normalized:
            raise ValueError("A multi-range result must contain at least one interval")
        object.__setattr__(self, "range_results", normalized)
        object.__setattr__(self, "cross_results", tuple(self.cross_results))
        if self.peak_order is not None and not isinstance(self.peak_order, PeakOrderResult):
            raise TypeError("peak_order must be a PeakOrderResult or None")
        object.__setattr__(self, "warnings", tuple(dict.fromkeys(map(str, self.warnings))))
        if self.output_directory is not None:
            object.__setattr__(self, "output_directory", Path(self.output_directory))

    @property
    def results(self) -> tuple[PipelineResult, ...]:
        """Return interval results in the same order as the requested ranges."""

        return tuple(item.result for item in self.range_results)

    @property
    def ranges(self) -> tuple[WavenumberRange, ...]:
        """Return normalized requested ranges in processing order."""

        return tuple(item.analysis_range for item in self.range_results)

    @property
    def bundle_path(self) -> Path | None:
        """Return the aggregate multi-range ZIP, when exported."""

        if self.output_directory is None:
            return None
        return self.output_directory / "multi_range_bundle.zip"

    @property
    def cross_count(self) -> int:
        """Return the number of unique interval pairs that were cross-correlated."""

        return len(self.cross_results)

    @property
    def all_checks_passed(self) -> bool:
        """Whether every interval and every cross-range result passed numerical checks."""

        range_checks = all(
            bool(item.result.qc_metrics["all_checks_passed"]) for item in self.range_results
        )
        cross_checks = all(
            bool(item.twodcos.qc_metrics["all_checks_passed"]) for item in self.cross_results
        )
        return range_checks and cross_checks


def _coerce_config(config: PipelineConfig | dict[str, Any] | None) -> PipelineConfig:
    if config is None:
        return PipelineConfig()
    if isinstance(config, PipelineConfig):
        return config
    return PipelineConfig.from_dict(config)


def _load_source(
    source: SpectralDataset | ReadableSource | list[ReadableSource] | tuple[ReadableSource, ...],
    config: PipelineConfig,
    *,
    delimiter: str | None,
    perturbation: list[float] | tuple[float, ...] | np.ndarray | None,
    dpt_pattern: str,
) -> SpectralDataset:
    if isinstance(source, SpectralDataset):
        return source
    return load_input(
        source,
        intensity_unit=config.input_intensity_unit,
        delimiter=delimiter,
        perturbation=perturbation,
        perturbation_order="preserve_file_order",
        dpt_pattern=dpt_pattern,
    )


def _collect_warnings(
    *groups: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    flattened = [str(item) for group in groups for item in group if str(item).strip()]
    return tuple(dict.fromkeys(flattened))


def _order_context_warnings(
    warnings: tuple[str, ...],
    *,
    context: str,
) -> list[str]:
    contextualized: list[str] = []
    for warning in warnings:
        if warning.startswith("Perturbation"):
            contextualized.append(f"{context}: {warning}")
        else:
            contextualized.append(warning)
    return contextualized


def preprocess_dataset(
    source: SpectralDataset | ReadableSource | list[ReadableSource] | tuple[ReadableSource, ...],
    config: PipelineConfig | dict[str, Any] | None = None,
    *,
    delimiter: str | None = None,
    perturbation: list[float] | tuple[float, ...] | np.ndarray | None = None,
    dpt_pattern: str = "*MIN.dpt",
) -> PreprocessingResult:
    """Run the full explicit preprocessing path without calculating 2D-COS.

    This is also the baseline-preview entry point used by Streamlit.  The
    returned raw, baseline, and corrected states are exactly the states that
    :func:`run_pipeline` will use with the same configuration.
    """

    normalized_config = _coerce_config(config)
    imported = _load_source(
        source,
        normalized_config,
        delimiter=delimiter,
        perturbation=perturbation,
        dpt_pattern=dpt_pattern,
    )
    input_report = validate_dataset(imported, normalized_config.wavenumber_range)
    input_report.raise_for_errors()

    ordered = apply_perturbation_order(imported, normalized_config.perturbation_order)
    absorbance = convert_to_absorbance(ordered, normalized_config.input_intensity_unit)
    if normalized_config.wavenumber_range is None:
        selected_raw = ensure_ascending_wavenumber(absorbance)
    else:
        lower, upper = normalized_config.wavenumber_range
        selected_raw = select_wavenumber_range(absorbance, lower, upper)

    selected_report = validate_dataset(selected_raw)
    selected_report.raise_for_errors()
    baseline_corrected, baselines, diagnostics = correct_baseline(
        selected_raw, normalized_config.baseline
    )
    smoothed = smooth_dataset(baseline_corrected, normalized_config.smoothing)
    processed = normalize_dataset(smoothed, normalized_config.normalization)

    diagnostic_warnings = [
        f"Spectrum {item['spectrum_index']} ({item['perturbation_label']}): {warning}"
        for item in diagnostics
        for warning in item.get("warnings", [])
    ]
    metadata_warnings = [
        *map(str, processed.metadata.get("parse_warnings", [])),
        *map(str, processed.metadata.get("processing_warnings", [])),
    ]
    warnings = _collect_warnings(
        _order_context_warnings(input_report.warnings, context="Input file order"),
        _order_context_warnings(selected_report.warnings, context="Final processing order"),
        diagnostic_warnings,
        metadata_warnings,
    )
    return PreprocessingResult(
        imported=imported,
        selected_raw=selected_raw,
        baselines=baselines,
        baseline_corrected=baseline_corrected,
        processed=processed,
        baseline_diagnostics=tuple(diagnostics),
        input_validation=input_report,
        selected_validation=selected_report,
        warnings=warnings,
    )


def _infer_input_paths(
    source: object,
    *,
    dpt_pattern: str,
) -> tuple[Path, ...]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_file():
            return (path,)
        if path.is_dir():
            return tuple(
                sorted(
                    (
                        item
                        for item in path.iterdir()
                        if item.is_file()
                        and fnmatch.fnmatch(item.name.lower(), dpt_pattern.lower())
                    ),
                    key=lambda item: (item.name.casefold(), item.name),
                )
            )
    if isinstance(source, (list, tuple)):
        paths = tuple(Path(item) for item in source if isinstance(item, (str, Path)))
        if len(paths) == len(source) and all(path.is_file() for path in paths):
            return paths
    return ()


def run_pipeline(
    source: SpectralDataset | ReadableSource | list[ReadableSource] | tuple[ReadableSource, ...],
    config: PipelineConfig | dict[str, Any] | None = None,
    *,
    output_root: str | Path | None = None,
    input_paths: tuple[str | Path, ...] | list[str | Path] | None = None,
    delimiter: str | None = None,
    perturbation: list[float] | tuple[float, ...] | np.ndarray | None = None,
    dpt_pattern: str = "*MIN.dpt",
) -> PipelineResult:
    """Execute preprocessing, homo 2D-COS, QC, and optional complete export."""

    normalized_config = _coerce_config(config)
    preprocessing = preprocess_dataset(
        source,
        normalized_config,
        delimiter=delimiter,
        perturbation=perturbation,
        dpt_pattern=dpt_pattern,
    )
    correlation = compute_2dcos(
        preprocessing.processed.spectra,
        preprocessing.processed.wavenumber,
        convention=normalized_config.convention,
    )
    qc_warnings: list[str] = []
    if not bool(correlation.qc_metrics["all_checks_passed"]):
        qc_warnings.append(
            "One or more 2D-COS numerical property checks exceeded the configured tolerance; "
            "inspect qc_metrics.json."
        )
    result = PipelineResult(
        imported=preprocessing.imported,
        selected_raw=preprocessing.selected_raw,
        baselines=preprocessing.baselines,
        baseline_corrected=preprocessing.baseline_corrected,
        processed=preprocessing.processed,
        baseline_diagnostics=preprocessing.baseline_diagnostics,
        twodcos=correlation,
        warnings=_collect_warnings(list(preprocessing.warnings), qc_warnings),
        input_validation=preprocessing.input_validation,
        selected_validation=preprocessing.selected_validation,
    )

    if output_root is not None:
        # Import lazily so scientific-core and preview-only workflows do not
        # import Matplotlib or create its cache directory.
        from .export import export_run

        resolved_inputs = (
            tuple(Path(path) for path in input_paths)
            if input_paths is not None
            else _infer_input_paths(source, dpt_pattern=dpt_pattern)
        )
        output_directory = export_run(
            result,
            normalized_config,
            output_root,
            input_paths=resolved_inputs,
        )
        result = replace(result, output_directory=output_directory)
    return result


def _coerce_wavenumber_ranges(
    ranges: Sequence[WavenumberRange | object],
) -> tuple[WavenumberRange, ...]:
    normalized = tuple(WavenumberRange.from_value(item) for item in ranges)
    if not normalized:
        raise ValueError("At least one wavenumber range is required")
    seen: set[tuple[float, float]] = set()
    for item in normalized:
        if item.bounds in seen:
            raise ValueError(
                "Duplicate wavenumber range: "
                f"{item.high_wavenumber:g}-{item.low_wavenumber:g} cm^-1"
            )
        seen.add(item.bounds)
    return normalized


def _number_slug(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace("+", "").replace(".", "p")


def _range_directory_name(index: int, analysis_range: WavenumberRange) -> str:
    interval = (
        f"{_number_slug(analysis_range.high_wavenumber)}-"
        f"{_number_slug(analysis_range.low_wavenumber)}"
    )
    return f"{index:02d}_{interval}"


def _cross_directory_name(item: CrossRangePipelineResult) -> str:
    first = _range_directory_name(item.first_index + 1, item.first_range)
    second = _range_directory_name(item.second_index + 1, item.second_range)
    return f"{first}__{second}"


def _compute_cross_range_results(
    range_results: Sequence[RangePipelineResult],
    *,
    convention: str,
) -> tuple[CrossRangePipelineResult, ...]:
    """Calculate every unique interval pair without duplicating reverse blocks."""

    cross_results: list[CrossRangePipelineResult] = []
    for (first_index, first), (second_index, second) in combinations(enumerate(range_results), 2):
        first_dataset = first.result.processed
        second_dataset = second.result.processed
        if not np.array_equal(first_dataset.perturbation, second_dataset.perturbation):
            raise ValueError(
                f"Cross-correlation failed for {first.analysis_range.display_name} x "
                f"{second.analysis_range.display_name}: perturbation values are not aligned"
            )
        if first_dataset.perturbation_labels != second_dataset.perturbation_labels:
            raise ValueError(
                f"Cross-correlation failed for {first.analysis_range.display_name} x "
                f"{second.analysis_range.display_name}: perturbation labels are not aligned"
            )
        try:
            correlation = compute_cross_2dcos(
                first_dataset.spectra,
                second_dataset.spectra,
                first_dataset.wavenumber,
                second_dataset.wavenumber,
                convention=convention,
            )
        except Exception as exc:
            raise ValueError(
                f"Cross-correlation failed for {first.analysis_range.display_name} x "
                f"{second.analysis_range.display_name}: {exc}"
            ) from exc
        cross_results.append(
            CrossRangePipelineResult(
                first_index=first_index,
                second_index=second_index,
                first_range=first.analysis_range,
                second_range=second.analysis_range,
                twodcos=correlation,
            )
        )
    return tuple(cross_results)


@dataclass(frozen=True, slots=True)
class _MatchedPeak:
    request: PeakRequest
    range_index: int
    grid_index: int
    matched_wavenumber: float
    distance: float


def _finite_nonnegative(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _coerce_peak_requests(
    peaks: Sequence[PeakRequest | Real | Mapping[str, object]],
) -> tuple[PeakRequest, ...]:
    normalized: list[PeakRequest] = []
    for index, peak in enumerate(peaks, start=1):
        if isinstance(peak, PeakRequest):
            normalized.append(peak)
            continue
        if isinstance(peak, Real) and not isinstance(peak, bool):
            normalized.append(PeakRequest(float(peak)))
            continue
        if isinstance(peak, Mapping):
            expected = {"wavenumber", "label", "range_index"}
            unknown = set(peak) - expected
            if unknown:
                joined = ", ".join(sorted(map(str, unknown)))
                raise ValueError(f"peak #{index} has unknown field(s): {joined}")
            if "wavenumber" not in peak:
                raise ValueError(f"peak #{index} is missing required field 'wavenumber'")
            normalized.append(
                PeakRequest(
                    wavenumber=peak["wavenumber"],  # type: ignore[arg-type]
                    label=peak.get("label"),  # type: ignore[arg-type]
                    range_index=peak.get("range_index"),  # type: ignore[arg-type]
                )
            )
            continue
        raise TypeError(f"peak #{index} must be a PeakRequest, real wavenumber, or mapping")
    if len(normalized) < 2:
        raise ValueError("peak response-order analysis requires at least two peaks")
    return tuple(normalized)


def _nearest_grid_match(axis: np.ndarray, requested: float) -> tuple[int, float, float]:
    numeric_axis = np.asarray(axis, dtype=np.float64)
    distances = np.abs(numeric_axis - requested)
    minimum = float(np.min(distances))
    magnitude = max(1.0, abs(requested), float(np.max(np.abs(numeric_axis))))
    tie_tolerance = 16.0 * float(np.spacing(magnitude))
    candidate_indices = np.flatnonzero(np.isclose(distances, minimum, rtol=0.0, atol=tie_tolerance))
    candidate_values = np.unique(numeric_axis[candidate_indices])
    if candidate_values.size > 1:
        rendered = ", ".join(f"{value:g}" for value in candidate_values)
        raise ValueError(
            f"Requested peak {requested:g} cm^-1 is exactly equidistant from sampled "
            f"grid points {rendered} cm^-1; specify a non-midpoint peak position"
        )
    index = int(candidate_indices[0])
    return index, float(numeric_axis[index]), minimum


def _match_peaks_to_ranges(
    result: MultiRangePipelineResult,
    peaks: Sequence[PeakRequest],
    *,
    tolerance: float,
) -> tuple[_MatchedPeak, ...]:
    matched: list[_MatchedPeak] = []
    for peak in peaks:
        if peak.range_index is not None:
            if peak.range_index >= len(result.range_results):
                raise ValueError(
                    f"Peak {peak.display_label} specifies range {peak.range_index + 1}, but "
                    f"only {len(result.range_results)} range(s) were analyzed"
                )
            candidate_indices = (peak.range_index,)
        else:
            candidate_indices = tuple(range(len(result.range_results)))

        candidates: list[_MatchedPeak] = []
        for range_index in candidate_indices:
            axis = result.range_results[range_index].result.processed.wavenumber
            grid_index, matched_wavenumber, distance = _nearest_grid_match(
                axis,
                peak.wavenumber,
            )
            if distance <= tolerance:
                resolved_request = replace(peak, range_index=range_index)
                candidates.append(
                    _MatchedPeak(
                        request=resolved_request,
                        range_index=range_index,
                        grid_index=grid_index,
                        matched_wavenumber=matched_wavenumber,
                        distance=distance,
                    )
                )

        if not candidates:
            if peak.range_index is None:
                scope = "any analyzed range"
            else:
                scope = f"analysis range {peak.range_index + 1}"
            raise ValueError(
                f"Peak {peak.display_label} has no sampled grid point within "
                f"{tolerance:g} cm^-1 in {scope}"
            )
        if len(candidates) > 1:
            rendered = ", ".join(
                f"range {item.range_index + 1} ({item.matched_wavenumber:g} cm^-1)"
                for item in candidates
            )
            raise ValueError(
                f"Peak {peak.display_label} is ambiguous across overlapping ranges: {rendered}. "
                "Specify range_index (Python, zero-based) or @RANGE (CLI, one-based)."
            )
        matched.append(candidates[0])

    for first, second in combinations(matched, 2):
        same_requested_wavenumber = bool(
            np.isclose(
                first.request.wavenumber,
                second.request.wavenumber,
                rtol=1.0e-10,
                atol=1.0e-8,
            )
        )
        same_grid_point = (
            first.range_index == second.range_index and first.grid_index == second.grid_index
        )
        same_physical_wavenumber = bool(
            np.isclose(
                first.matched_wavenumber,
                second.matched_wavenumber,
                rtol=1.0e-10,
                atol=1.0e-8,
            )
        )
        if same_requested_wavenumber or same_grid_point or same_physical_wavenumber:
            raise ValueError(
                f"Peaks {first.request.display_label} and {second.request.display_label} "
                f"both resolve to the same physical grid position "
                f"({first.matched_wavenumber:g} cm^-1); duplicate spectral variables "
                "cannot be response-ordered"
            )
    return tuple(matched)


def _matrix_scale(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=np.float64)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _cross_result_for_ranges(
    result: MultiRangePipelineResult,
    first_range_index: int,
    second_range_index: int,
) -> CrossRangePipelineResult:
    low_index, high_index = sorted((first_range_index, second_range_index))
    try:
        return next(
            item
            for item in result.cross_results
            if item.first_index == low_index and item.second_index == high_index
        )
    except StopIteration as exc:
        raise ValueError(
            f"Missing cross-range matrices for analysis ranges {low_index + 1} and {high_index + 1}"
        ) from exc


def _sample_peak_pair(
    result: MultiRangePipelineResult,
    first: _MatchedPeak,
    second: _MatchedPeak,
    *,
    match_tolerance: float,
    synchronous_threshold: float,
    asynchronous_threshold: float,
    relative_threshold: float,
) -> ResolvedPairValues:
    """Sample canonical matrices with the higher matched wavenumber as ``first``."""

    if first.matched_wavenumber <= second.matched_wavenumber:
        raise ValueError("Internal peak-pair orientation must satisfy first wavenumber > second")

    if first.range_index == second.range_index:
        analysis = result.range_results[first.range_index].result.twodcos
        synchronous_matrix = analysis.canonical_synchronous
        asynchronous_matrix = analysis.canonical_asynchronous
        synchronous = float(synchronous_matrix[first.grid_index, second.grid_index])
        asynchronous = float(asynchronous_matrix[first.grid_index, second.grid_index])
        source = f"range_{first.range_index + 1}_canonical_self"
        orientation = "canonical[row=higher_peak,column=lower_peak]"
        matrix_range_indices = [first.range_index + 1, first.range_index + 1]
    else:
        cross = _cross_result_for_ranges(result, first.range_index, second.range_index)
        synchronous_matrix = cross.twodcos.canonical_synchronous
        asynchronous_matrix = cross.twodcos.canonical_asynchronous
        matrix_range_indices = [cross.first_index + 1, cross.second_index + 1]
        if first.range_index == cross.first_index:
            synchronous = float(synchronous_matrix[first.grid_index, second.grid_index])
            asynchronous = float(asynchronous_matrix[first.grid_index, second.grid_index])
            source = f"cross_ranges_{cross.first_index + 1}_{cross.second_index + 1}_canonical"
            orientation = "canonical[row=higher_peak,column=lower_peak]"
        else:
            # The stored canonical matrix is row=input1/first configured range,
            # column=input2/second configured range.  Reverse direction follows
            # Phi21=Phi12.T and Psi21=-Psi12.T.
            synchronous = float(synchronous_matrix[second.grid_index, first.grid_index])
            asynchronous = -float(asynchronous_matrix[second.grid_index, first.grid_index])
            source = (
                f"cross_ranges_{cross.first_index + 1}_{cross.second_index + 1}_"
                "canonical_reverse_identity"
            )
            orientation = "reverse canonical via Phi21=Phi12.T; Psi21=-Psi12.T"

    sync_scale = _matrix_scale(synchronous_matrix)
    async_scale = _matrix_scale(asynchronous_matrix)
    effective_sync = max(synchronous_threshold, relative_threshold * sync_scale)
    effective_async = max(asynchronous_threshold, relative_threshold * async_scale)
    relative_signal_strength = None
    if sync_scale > 0.0 and async_scale > 0.0:
        relative_signal_strength = min(
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
        sync_threshold=effective_sync,
        async_threshold=effective_async,
        relative_signal_strength=relative_signal_strength,
        source=source,
        metadata={
            "pair_orientation": "higher_matched_wavenumber_first",
            "canonical_sampling": orientation,
            "matrix_range_indices_one_based": matrix_range_indices,
            "first_range_index_one_based": first.range_index + 1,
            "second_range_index_one_based": second.range_index + 1,
            "first_grid_index_zero_based": first.grid_index,
            "second_grid_index_zero_based": second.grid_index,
            "first_match_distance_cm-1": first.distance,
            "second_match_distance_cm-1": second.distance,
            "match_tolerance_cm-1": match_tolerance,
            "sync_matrix_max_abs": sync_scale,
            "async_matrix_max_abs": async_scale,
            "absolute_synchronous_signal_cutoff": synchronous_threshold,
            "absolute_asynchronous_signal_cutoff": asynchronous_threshold,
            "relative_matrix_signal_cutoff": relative_threshold,
            "effective_synchronous_signal_cutoff": effective_sync,
            "effective_asynchronous_signal_cutoff": effective_async,
        },
    )


def _analysis_order_facts(
    result: MultiRangePipelineResult,
    *,
    additional_note: str | None,
) -> tuple[str, tuple[str, ...]]:
    perturbation = np.asarray(
        result.range_results[0].result.processed.perturbation,
        dtype=np.float64,
    )
    for range_result in result.range_results[1:]:
        other = np.asarray(range_result.result.processed.perturbation, dtype=np.float64)
        if not np.array_equal(perturbation, other):
            raise ValueError(
                "Peak response-order analysis requires identical processed perturbation "
                "sequences across all ranges"
            )

    differences = np.diff(perturbation)
    warnings: list[str] = []
    if np.all(differences > 0.0):
        direction = (
            "stored perturbation values are strictly increasing "
            f"from {perturbation[0]:g} to {perturbation[-1]:g}"
        )
    elif np.all(differences < 0.0):
        direction = (
            "stored perturbation values are strictly decreasing "
            f"from {perturbation[0]:g} to {perturbation[-1]:g}"
        )
    else:
        direction = "stored perturbation values are non-monotonic"
        warnings.append(
            "The processed perturbation values are non-monotonic; 'earlier' only describes "
            "the analyzed spectrum index order and may not represent a physical progression."
        )

    scale = max(1.0, float(np.max(np.abs(differences))))
    uniformly_spaced = bool(
        np.allclose(
            differences,
            differences[0],
            rtol=1.0e-6,
            atol=1.0e-10 * scale,
        )
    )
    spacing = "uniformly spaced" if uniformly_spaced else "non-uniformly spaced"
    if not uniformly_spaced:
        warnings.append(
            "Perturbation spacing is non-uniform; the Hilbert-Noda matrix uses analyzed "
            "spectrum index order rather than metric perturbation spacing."
        )

    note = (
        "Interpret 'earlier' only along the processed analysis sequence; "
        f"{direction}, and they are {spacing}. No physical direction such as increasing "
        "temperature is assumed."
    )
    if additional_note is not None:
        normalized_note = str(additional_note).strip()
        if normalized_note:
            note += f" Additional user context (does not override stored order): {normalized_note}"
    return note, tuple(warnings)


def analyze_multi_range_peak_order(
    result: MultiRangePipelineResult,
    peaks: Sequence[PeakRequest | Real | Mapping[str, object]],
    *,
    peak_match_tolerance: Real = 1.0,
    synchronous_threshold: Real = 0.0,
    asynchronous_threshold: Real = 0.0,
    relative_threshold: Real = 1.0e-6,
    analysis_order_note: str | None = None,
) -> PeakOrderResult:
    """Infer an auditable peak response order from a completed multi-range result.

    Peak positions are matched to the nearest sampled point.  An unqualified
    peak that matches more than one overlapping range is rejected rather than
    silently assigned.  Every pair is oriented from higher to lower matched
    wavenumber and sampled from convention-independent canonical matrices.
    """

    if not isinstance(result, MultiRangePipelineResult):
        raise TypeError("result must be a MultiRangePipelineResult")
    normalized_peaks = _coerce_peak_requests(peaks)
    tolerance = _finite_nonnegative(
        peak_match_tolerance,
        name="peak_match_tolerance",
    )
    absolute_sync = _finite_nonnegative(
        synchronous_threshold,
        name="synchronous_threshold",
    )
    absolute_async = _finite_nonnegative(
        asynchronous_threshold,
        name="asynchronous_threshold",
    )
    relative = _finite_nonnegative(relative_threshold, name="relative_threshold")
    if relative > 1.0:
        raise ValueError("relative_threshold must be between 0 and 1 inclusive")
    matched = _match_peaks_to_ranges(
        result,
        normalized_peaks,
        tolerance=tolerance,
    )

    pair_values: list[ResolvedPairValues] = []
    for left, right in combinations(matched, 2):
        if left.matched_wavenumber > right.matched_wavenumber:
            first, second = left, right
        else:
            first, second = right, left
        pair_values.append(
            _sample_peak_pair(
                result,
                first,
                second,
                match_tolerance=tolerance,
                synchronous_threshold=absolute_sync,
                asynchronous_threshold=absolute_async,
                relative_threshold=relative,
            )
        )

    order_note, order_warnings = _analysis_order_facts(
        result,
        additional_note=analysis_order_note,
    )
    inferred = infer_peak_order(
        tuple(item.request for item in matched),
        pair_values,
        analysis_order_note=order_note,
    )
    return replace(
        inferred,
        warnings=_collect_warnings(list(inferred.warnings), list(order_warnings)),
    )


def _create_unique_multi_range_directory(output_root: str | Path) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise NotADirectoryError(f"output root is not a directory: {root}")
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    for collision_index in range(10_000):
        suffix = "" if collision_index == 0 else f"_{collision_index:03d}"
        candidate = root / f"multi_range_{timestamp}{suffix}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"could not create a unique multi-range directory beneath {root}")


def _write_batch_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_peak_order_exports(
    batch_directory: Path,
    peak_order: PeakOrderResult,
) -> dict[str, object]:
    """Write the complete response-order audit JSON and a flat evidence CSV."""

    peak_directory = batch_directory / "peak_order"
    peak_directory.mkdir()
    _write_batch_json(peak_directory / "peak_order.json", peak_order.to_dict())

    csv_path = peak_directory / "pairwise_evidence.csv"
    fieldnames = [
        "pair_index_one_based",
        "first_label",
        "first_requested_wavenumber_cm-1",
        "first_matched_wavenumber_cm-1",
        "first_range_index_one_based",
        "second_label",
        "second_requested_wavenumber_cm-1",
        "second_matched_wavenumber_cm-1",
        "second_range_index_one_based",
        "synchronous",
        "asynchronous",
        "sync_threshold",
        "async_threshold",
        "value_product",
        "synchronous_to_cutoff_ratio",
        "asynchronous_to_cutoff_ratio",
        "minimum_cutoff_ratio",
        "relative_signal_strength",
        "relation",
        "earlier_label",
        "later_label",
        "sign_product",
        "source",
        "reason",
        "metadata_json",
    ]
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair_index, evidence in enumerate(peak_order.evidence, start=1):
            earlier = evidence.earlier
            later = evidence.later
            writer.writerow(
                {
                    "pair_index_one_based": pair_index,
                    "first_label": evidence.first.display_label,
                    "first_requested_wavenumber_cm-1": evidence.first.wavenumber,
                    "first_matched_wavenumber_cm-1": evidence.matched_first_wavenumber,
                    "first_range_index_one_based": (
                        None
                        if evidence.first.range_index is None
                        else evidence.first.range_index + 1
                    ),
                    "second_label": evidence.second.display_label,
                    "second_requested_wavenumber_cm-1": evidence.second.wavenumber,
                    "second_matched_wavenumber_cm-1": evidence.matched_second_wavenumber,
                    "second_range_index_one_based": (
                        None
                        if evidence.second.range_index is None
                        else evidence.second.range_index + 1
                    ),
                    "synchronous": evidence.synchronous,
                    "asynchronous": evidence.asynchronous,
                    "sync_threshold": evidence.sync_threshold,
                    "async_threshold": evidence.async_threshold,
                    "value_product": evidence.value_product,
                    "synchronous_to_cutoff_ratio": (evidence.synchronous_to_cutoff_ratio),
                    "asynchronous_to_cutoff_ratio": (evidence.asynchronous_to_cutoff_ratio),
                    "minimum_cutoff_ratio": evidence.minimum_cutoff_ratio,
                    "relative_signal_strength": evidence.relative_signal_strength,
                    "relation": evidence.relation.value,
                    "earlier_label": None if earlier is None else earlier.display_label,
                    "later_label": None if later is None else later.display_label,
                    "sign_product": evidence.sign_product,
                    "source": evidence.source,
                    "reason": evidence.reason,
                    "metadata_json": json.dumps(
                        evidence.metadata,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    ),
                }
            )

    unique_order = [peak.to_dict() for peak in peak_order.unique_order]
    first_metadata = peak_order.evidence[0].metadata if peak_order.evidence else {}
    return {
        "requested": True,
        "peak_count": len(peak_order.peaks),
        "requested_peaks": [peak.to_dict() for peak in peak_order.peaks],
        "pairwise_evidence_count": len(peak_order.evidence),
        "unresolved_pair_count": len(peak_order.unresolved_relations),
        "all_pairs_resolved": peak_order.all_pairs_resolved,
        "has_cycles": peak_order.has_cycles,
        "is_unique_total_order": peak_order.is_unique_total_order,
        "unique_order": unique_order,
        "analysis_order_note": peak_order.analysis_order_note,
        "rule_description": peak_order.rule_description,
        "matching_and_cutoff_settings": {
            "match_tolerance_cm-1": first_metadata.get("match_tolerance_cm-1"),
            "absolute_synchronous_signal_cutoff": first_metadata.get(
                "absolute_synchronous_signal_cutoff"
            ),
            "absolute_asynchronous_signal_cutoff": first_metadata.get(
                "absolute_asynchronous_signal_cutoff"
            ),
            "relative_matrix_signal_cutoff": first_metadata.get("relative_matrix_signal_cutoff"),
        },
        "pairwise_effective_signal_cutoffs": [
            {
                "first": evidence.first.to_dict(),
                "second": evidence.second.to_dict(),
                "sync_threshold": evidence.sync_threshold,
                "async_threshold": evidence.async_threshold,
                "relative_signal_strength": evidence.relative_signal_strength,
                "source": evidence.source,
            }
            for evidence in peak_order.evidence
        ],
        "pair_orientation": "higher matched wavenumber as canonical row/first peak",
        "canonical_matrix_contract": (
            "Self pairs use canonical homo matrices. Cross pairs use canonical input1-by-input2 "
            "matrices; reverse sampling applies Phi21=Phi12.T and Psi21=-Psi12.T."
        ),
        "signal_cutoff_contract": (
            "Each evidence row stores effective absolute synchronous/asynchronous signal "
            "cutoffs. These are numerical interpretation cutoffs, not confidence levels."
        ),
        "theory_references": [
            {
                "url": "https://doi.org/10.1366/0003702001950454",
                "scope": "generalized 2D-COS sign rule and sequential-order interpretation",
            },
            {
                "url": "https://doi.org/10.1038/s41467-024-45079-4",
                "scope": (
                    "literature response-sequence example in the main text and "
                    "Supplementary Table 2"
                ),
            },
        ],
        "output_files": [
            "peak_order/peak_order.json",
            "peak_order/pairwise_evidence.csv",
        ],
    }


def _create_multi_range_bundle(batch_directory: Path) -> Path:
    bundle = batch_directory / "multi_range_bundle.zip"
    with zipfile.ZipFile(
        bundle,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(batch_directory.rglob("*")):
            if not path.is_file() or path == bundle or path.name == "run_bundle.zip":
                continue
            archive.write(path, arcname=path.relative_to(batch_directory))
    return bundle


def _export_cross_range_results(
    cross_results: Sequence[CrossRangePipelineResult],
    range_results: Sequence[RangePipelineResult],
    *,
    batch_directory: Path,
    base_config: PipelineConfig,
) -> tuple[CrossRangePipelineResult, ...]:
    """Export every unique cross pair with explicit forward and reverse matrices."""

    if not cross_results:
        return ()

    from .export import write_matrix_csv
    from .plotting import plot_asynchronous_contour, plot_synchronous_contour

    cross_root = batch_directory / "cross_ranges"
    cross_root.mkdir()
    exported: list[CrossRangePipelineResult] = []
    plot_options = {
        "contour_levels": base_config.contour_levels,
        "display_percentile": base_config.display_percentile,
        "filled": True,
    }

    for item in cross_results:
        analysis = item.twodcos
        pair_directory = cross_root / _cross_directory_name(item)
        data_directory = pair_directory / "data"
        figure_directory = pair_directory / "figures"
        data_directory.mkdir(parents=True)
        figure_directory.mkdir()

        write_matrix_csv(
            data_directory / "01_synchronous_matrix.csv",
            analysis.synchronous,
            analysis.row_wavenumber,
            analysis.column_wavenumber,
        )
        write_matrix_csv(
            data_directory / "02_asynchronous_matrix.csv",
            analysis.asynchronous,
            analysis.row_wavenumber,
            analysis.column_wavenumber,
        )
        write_matrix_csv(
            data_directory / "03_reverse_synchronous_matrix.csv",
            analysis.reverse_synchronous,
            analysis.reverse_row_wavenumber,
            analysis.reverse_column_wavenumber,
        )
        write_matrix_csv(
            data_directory / "04_reverse_asynchronous_matrix.csv",
            analysis.reverse_asynchronous,
            analysis.reverse_row_wavenumber,
            analysis.reverse_column_wavenumber,
        )
        write_matrix_csv(
            data_directory / "05_canonical_synchronous_nu1_by_nu2.csv",
            analysis.canonical_synchronous,
            analysis.wavenumber1,
            analysis.wavenumber2,
        )
        write_matrix_csv(
            data_directory / "06_canonical_asynchronous_nu1_by_nu2.csv",
            analysis.canonical_asynchronous,
            analysis.wavenumber1,
            analysis.wavenumber2,
        )

        plot_synchronous_contour(
            analysis.row_wavenumber,
            analysis.synchronous,
            figure_directory / "synchronous_cross_2dcos",
            column_wavenumber=analysis.column_wavenumber,
            convention=analysis.convention,
            row_variable=analysis.row_variable,
            column_variable=analysis.column_variable,
            method="independently preprocessed cross-range",
            show_diagonal=False,
            **plot_options,
        )
        plot_asynchronous_contour(
            analysis.row_wavenumber,
            analysis.asynchronous,
            figure_directory / "asynchronous_cross_2dcos",
            column_wavenumber=analysis.column_wavenumber,
            convention=analysis.convention,
            row_variable=analysis.row_variable,
            column_variable=analysis.column_variable,
            method="independently preprocessed cross-range",
            show_diagonal=False,
            **plot_options,
        )

        first_shape = list(range_results[item.first_index].result.processed.shape)
        second_shape = list(range_results[item.second_index].result.processed.shape)
        manifest = {
            "analysis_type": "same-series_cross-region_2dcos",
            "pair_indices_one_based": [item.first_index + 1, item.second_index + 1],
            "pair_label": item.pair_label,
            "first_range": item.first_range.to_dict(),
            "second_range": item.second_range.to_dict(),
            "first_processed_shape": first_shape,
            "second_processed_shape": second_shape,
            "matrix_shape": list(analysis.synchronous.shape),
            "convention": analysis.convention,
            "matrix_axes": {
                "row_range_index_one_based": item.row_index + 1,
                "column_range_index_one_based": item.column_index + 1,
                "row_variable": analysis.row_variable,
                "column_variable": analysis.column_variable,
                "row_wavenumber_cm-1": analysis.row_wavenumber.tolist(),
                "column_wavenumber_cm-1": analysis.column_wavenumber.tolist(),
            },
            "formulas": {
                "canonical_synchronous": "Phi12 = D1.T @ D2 / (m - 1)",
                "canonical_asynchronous": "Psi12 = D1.T @ N @ D2 / (m - 1)",
                "reverse_synchronous": "Phi21 = Phi12.T",
                "reverse_asynchronous": "Psi21 = -Psi12.T",
            },
            "preprocessing_scope": (
                "Each interval was independently selected and preprocessed; both dynamic "
                "matrices use the same ordered perturbation observations."
            ),
            "terminology": (
                "Cross-region correlation within one FTIR series; this export does not claim "
                "a two-instrument heterospectral measurement."
            ),
            "theory_references": [
                "https://doi.org/10.1366/0003702001950454",
                "https://doi.org/10.1177/0003702818819880",
            ],
            "convention_metadata": analysis.convention_metadata,
            "qc_metrics": analysis.qc_metrics,
            "output_files": [
                "data/01_synchronous_matrix.csv",
                "data/02_asynchronous_matrix.csv",
                "data/03_reverse_synchronous_matrix.csv",
                "data/04_reverse_asynchronous_matrix.csv",
                "data/05_canonical_synchronous_nu1_by_nu2.csv",
                "data/06_canonical_asynchronous_nu1_by_nu2.csv",
                "figures/synchronous_cross_2dcos.png",
                "figures/synchronous_cross_2dcos.pdf",
                "figures/asynchronous_cross_2dcos.png",
                "figures/asynchronous_cross_2dcos.pdf",
                "qc_metrics.json",
                "manifest.json",
            ],
        }
        _write_batch_json(pair_directory / "qc_metrics.json", analysis.qc_metrics)
        _write_batch_json(pair_directory / "manifest.json", manifest)
        exported.append(replace(item, output_directory=pair_directory))
    return tuple(exported)


def _block_for_indices(
    range_results: Sequence[RangePipelineResult],
    cross_results: Sequence[CrossRangePipelineResult],
    *,
    row_index: int,
    column_index: int,
    kind: str,
) -> np.ndarray:
    if row_index == column_index:
        return np.asarray(getattr(range_results[row_index].result.twodcos, kind))
    first_index, second_index = sorted((row_index, column_index))
    pair = next(
        item
        for item in cross_results
        if item.first_index == first_index and item.second_index == second_index
    )
    if row_index == pair.row_index and column_index == pair.column_index:
        return np.asarray(getattr(pair.twodcos, kind))
    reverse_name = f"reverse_{kind}"
    return np.asarray(getattr(pair.twodcos, reverse_name))


def _export_multi_range_block_figures(
    range_results: Sequence[RangePipelineResult],
    cross_results: Sequence[CrossRangePipelineResult],
    *,
    batch_directory: Path,
    base_config: PipelineConfig,
) -> dict[str, object]:
    """Export literature-style full block maps with reversed row-range order."""

    from .plotting import (
        plot_multi_range_asynchronous_contour,
        plot_multi_range_synchronous_contour,
    )

    figure_directory = batch_directory / "figures"
    figure_directory.mkdir()
    column_indices = tuple(range(len(range_results)))
    row_indices = tuple(reversed(column_indices))
    row_axes = [range_results[index].result.twodcos.row_wavenumber for index in row_indices]
    column_axes = [
        range_results[index].result.twodcos.column_wavenumber for index in column_indices
    ]
    # Keep plot labels compact; complete user labels remain in the manifest and
    # Streamlit selectors.  The actual sampled interval is printed below each
    # compact range identifier by the plotting layer.
    row_labels = [f"Range {index + 1}" for index in row_indices]
    column_labels = [f"Range {index + 1}" for index in column_indices]
    diagonal_blocks = {
        (row_position, column_position)
        for row_position, range_index in enumerate(row_indices)
        for column_position, column_index in enumerate(column_indices)
        if range_index == column_index
    }
    synchronous_blocks = [
        [
            _block_for_indices(
                range_results,
                cross_results,
                row_index=row_index,
                column_index=column_index,
                kind="synchronous",
            )
            for column_index in column_indices
        ]
        for row_index in row_indices
    ]
    asynchronous_blocks = [
        [
            _block_for_indices(
                range_results,
                cross_results,
                row_index=row_index,
                column_index=column_index,
                kind="asynchronous",
            )
            for column_index in column_indices
        ]
        for row_index in row_indices
    ]
    options = {
        "row_labels": row_labels,
        "column_labels": column_labels,
        "convention": base_config.convention,
        "method": "auto + cross blocks",
        "contour_levels": base_config.contour_levels,
        "display_percentile": base_config.display_percentile,
        "filled": True,
        "diagonal_blocks": diagonal_blocks,
    }
    plot_multi_range_synchronous_contour(
        row_axes,
        column_axes,
        synchronous_blocks,
        figure_directory / "multi_range_synchronous_blocks",
        **options,
    )
    plot_multi_range_asynchronous_contour(
        row_axes,
        column_axes,
        asynchronous_blocks,
        figure_directory / "multi_range_asynchronous_blocks",
        **options,
    )
    return {
        "row_range_indices_one_based": [index + 1 for index in row_indices],
        "column_range_indices_one_based": [index + 1 for index in column_indices],
        "same_range_blocks": [list(pair) for pair in sorted(diagonal_blocks)],
        "shared_symmetric_color_scale_per_figure": True,
        "display_percentile": base_config.display_percentile,
        "files": [
            "figures/multi_range_synchronous_blocks.png",
            "figures/multi_range_synchronous_blocks.pdf",
            "figures/multi_range_asynchronous_blocks.png",
            "figures/multi_range_asynchronous_blocks.pdf",
        ],
    }


def _export_multi_range_result(
    range_results: Sequence[RangePipelineResult],
    cross_results: Sequence[CrossRangePipelineResult],
    *,
    peak_order: PeakOrderResult | None,
    base_config: PipelineConfig,
    output_root: str | Path,
    input_paths: tuple[Path, ...],
) -> MultiRangePipelineResult:
    from .export import export_run

    batch_directory = _create_unique_multi_range_directory(output_root)
    ranges_root = batch_directory / "ranges"
    ranges_root.mkdir()
    exported: list[RangePipelineResult] = []
    aggregate_warnings: list[str] = []

    for index, item in enumerate(range_results, start=1):
        range_root = ranges_root / _range_directory_name(index, item.analysis_range)
        range_config = base_config.for_range(item.analysis_range)
        output_directory = export_run(
            item.result,
            range_config,
            range_root,
            input_paths=input_paths,
        )
        exported_result = replace(item.result, output_directory=output_directory)
        exported.append(
            RangePipelineResult(
                analysis_range=item.analysis_range,
                result=exported_result,
            )
        )
        aggregate_warnings.extend(
            f"{item.analysis_range.display_name}: {warning}" for warning in item.result.warnings
        )

    exported_cross = _export_cross_range_results(
        cross_results,
        exported,
        batch_directory=batch_directory,
        base_config=base_config,
    )
    for item in exported_cross:
        if not bool(item.twodcos.qc_metrics["all_checks_passed"]):
            aggregate_warnings.append(
                f"{item.pair_label}: cross-range numerical checks did not all pass"
            )
    block_layout = _export_multi_range_block_figures(
        exported,
        exported_cross,
        batch_directory=batch_directory,
        base_config=base_config,
    )
    peak_order_manifest = (
        None if peak_order is None else _write_peak_order_exports(batch_directory, peak_order)
    )
    if peak_order is not None:
        aggregate_warnings.extend(peak_order.warnings)

    manifest_ranges: list[dict[str, object]] = []
    source_summary: dict[str, object] = {}
    for index, item in enumerate(exported, start=1):
        result = item.result
        processed_axis = result.processed.wavenumber
        assert result.output_directory is not None
        child_manifest = json.loads(
            (result.output_directory / "manifest.json").read_text(encoding="utf-8")
        )
        if not source_summary:
            source_summary = {
                "input_files": child_manifest.get("input_files", []),
                "original_data_shape": child_manifest.get("original_data_shape"),
                "original_internal_shape": child_manifest.get("original_internal_shape"),
                "original_wavenumber_direction": child_manifest.get(
                    "original_wavenumber_direction"
                ),
                "perturbation_original_order": child_manifest.get("perturbation_original_order"),
            }
        manifest_ranges.append(
            {
                "index": index,
                "label": item.analysis_range.label,
                "requested_range_cm-1": {
                    "high": item.analysis_range.high_wavenumber,
                    "low": item.analysis_range.low_wavenumber,
                },
                "actual_range_cm-1": {
                    "high": float(np.max(processed_axis)),
                    "low": float(np.min(processed_axis)),
                },
                "final_data_shape": list(result.processed.shape),
                "all_checks_passed": bool(result.qc_metrics["all_checks_passed"]),
                "qc_metrics": result.qc_metrics,
                "warning_count": len(result.warnings),
                "pipeline_config": base_config.for_range(item.analysis_range).to_dict(),
                "output_directory": str(result.output_directory.relative_to(batch_directory)),
                "standalone_bundle": {
                    "path": str(
                        result.bundle_path.relative_to(batch_directory)
                        if result.bundle_path is not None
                        else ""
                    ),
                    "included_in_aggregate_bundle": False,
                },
            }
        )

    all_checks_passed = all(
        bool(item.result.qc_metrics["all_checks_passed"]) for item in exported
    ) and all(bool(item.twodcos.qc_metrics["all_checks_passed"]) for item in exported_cross)
    manifest_cross_ranges = [
        {
            "first_range_index_one_based": item.first_index + 1,
            "second_range_index_one_based": item.second_index + 1,
            "pair_label": item.pair_label,
            "row_range_index_one_based": item.row_index + 1,
            "column_range_index_one_based": item.column_index + 1,
            "matrix_shape": list(item.twodcos.synchronous.shape),
            "all_checks_passed": bool(item.twodcos.qc_metrics["all_checks_passed"]),
            "qc_metrics": item.twodcos.qc_metrics,
            "output_directory": str(
                item.output_directory.relative_to(batch_directory)
                if item.output_directory is not None
                else ""
            ),
        }
        for item in exported_cross
    ]
    from . import __version__

    manifest = {
        "tool": "ftir2dcos",
        "tool_version": __version__,
        "mode": "multi_range",
        "run_time": datetime.now().astimezone().isoformat(),
        "range_count": len(exported),
        "cross_correlation_count": len(exported_cross),
        "all_checks_passed": all_checks_passed,
        **source_summary,
        "base_config": base_config.to_dict(),
        "ranges": manifest_ranges,
        "cross_correlations": manifest_cross_ranges,
        "block_figure_layout": block_layout,
        "cross_correlation_contract": {
            "scope": "same ordered FTIR perturbation series across distinct spectral regions",
            "canonical_synchronous": "Phi12 = D1.T @ D2 / (m - 1)",
            "canonical_asynchronous": "Psi12 = D1.T @ N @ D2 / (m - 1)",
            "reverse_synchronous": "Phi21 = Phi12.T",
            "reverse_asynchronous": "Psi21 = -Psi12.T",
            "unique_pairs_only": True,
            "pair_count_formula": "n * (n - 1) / 2",
            "theory_references": [
                "https://doi.org/10.1366/0003702001950454",
                "https://doi.org/10.1177/0003702818819880",
            ],
        },
        "aggregate_bundle": "multi_range_bundle.zip",
    }
    if peak_order_manifest is not None:
        manifest["peak_response_order"] = peak_order_manifest
    _write_batch_json(batch_directory / "multi_range_manifest.json", manifest)
    _write_batch_json(batch_directory / "base_config.json", base_config.to_dict())
    (batch_directory / "warnings.txt").write_text(
        "".join(f"{warning}\n" for warning in dict.fromkeys(aggregate_warnings)),
        encoding="utf-8",
    )
    _create_multi_range_bundle(batch_directory)
    return MultiRangePipelineResult(
        range_results=tuple(exported),
        cross_results=exported_cross,
        peak_order=peak_order,
        warnings=tuple(aggregate_warnings),
        output_directory=batch_directory,
    )


def run_multi_range_pipeline(
    source: SpectralDataset | ReadableSource | list[ReadableSource] | tuple[ReadableSource, ...],
    ranges: Sequence[WavenumberRange | object],
    config: PipelineConfig | dict[str, Any] | None = None,
    *,
    output_root: str | Path | None = None,
    input_paths: tuple[str | Path, ...] | list[str | Path] | None = None,
    delimiter: str | None = None,
    perturbation: list[float] | tuple[float, ...] | np.ndarray | None = None,
    dpt_pattern: str = "*MIN.dpt",
    peaks: Sequence[PeakRequest | Real | Mapping[str, object]] | None = None,
    peak_match_tolerance: Real = 1.0,
    synchronous_threshold: Real = 0.0,
    asynchronous_threshold: Real = 0.0,
    relative_threshold: Real = 1.0e-6,
    analysis_order_note: str | None = None,
) -> MultiRangePipelineResult:
    """Run auto- and cross-correlation 2D-COS analyses for any number of intervals.

    The source is parsed exactly once.  Each interval is then independently
    selected and preprocessed.  All unique interval pairs are additionally
    cross-correlated on the shared perturbation order.  Reverse display blocks
    are derived by transpose/sign identities instead of recalculation.  The
    existing :func:`run_pipeline` remains the single-range API.
    """

    normalized_ranges = _coerce_wavenumber_ranges(ranges)
    normalized_config = _coerce_config(config)
    base_config = replace(
        normalized_config,
        low_wavenumber=None,
        high_wavenumber=None,
    )
    imported = _load_source(
        source,
        base_config,
        delimiter=delimiter,
        perturbation=perturbation,
        dpt_pattern=dpt_pattern,
    )
    resolved_inputs = (
        tuple(Path(path) for path in input_paths)
        if input_paths is not None
        else _infer_input_paths(source, dpt_pattern=dpt_pattern)
    )

    computed: list[RangePipelineResult] = []
    for analysis_range in normalized_ranges:
        range_config = base_config.for_range(analysis_range)
        try:
            result = run_pipeline(imported, range_config)
        except Exception as exc:
            raise ValueError(f"Analysis failed for {analysis_range.display_name}: {exc}") from exc
        computed.append(RangePipelineResult(analysis_range=analysis_range, result=result))

    cross_results = _compute_cross_range_results(
        computed,
        convention=base_config.convention,
    )

    range_warnings = tuple(
        f"{item.analysis_range.display_name}: {warning}"
        for item in computed
        for warning in item.result.warnings
    )
    cross_warnings = tuple(
        f"{item.pair_label}: cross-range numerical checks did not all pass"
        for item in cross_results
        if not bool(item.twodcos.qc_metrics["all_checks_passed"])
    )
    warnings = _collect_warnings(list(range_warnings), list(cross_warnings))
    peak_order: PeakOrderResult | None = None
    if peaks is not None:
        unranked_result = MultiRangePipelineResult(
            range_results=tuple(computed),
            cross_results=cross_results,
            warnings=warnings,
        )
        peak_order = analyze_multi_range_peak_order(
            unranked_result,
            peaks,
            peak_match_tolerance=peak_match_tolerance,
            synchronous_threshold=synchronous_threshold,
            asynchronous_threshold=asynchronous_threshold,
            relative_threshold=relative_threshold,
            analysis_order_note=analysis_order_note,
        )
        warnings = _collect_warnings(list(warnings), list(peak_order.warnings))
    if output_root is None:
        return MultiRangePipelineResult(
            range_results=tuple(computed),
            cross_results=cross_results,
            peak_order=peak_order,
            warnings=warnings,
        )
    return _export_multi_range_result(
        computed,
        cross_results,
        peak_order=peak_order,
        base_config=base_config,
        output_root=output_root,
        input_paths=resolved_inputs,
    )


# Explicit alias used in UI copy and tests.
preview_preprocessing = preprocess_dataset


__all__ = [
    "CrossRangePipelineResult",
    "MultiRangePipelineResult",
    "PipelineResult",
    "PreprocessingResult",
    "RangePipelineResult",
    "analyze_multi_range_peak_order",
    "preprocess_dataset",
    "preview_preprocessing",
    "run_multi_range_pipeline",
    "run_pipeline",
]
