"""Single-page Streamlit workflow for auditable FTIR preprocessing and 2D-COS.

The page is intentionally a thin client: parsing, validation, preprocessing,
2D-COS, plotting, and export are delegated to :mod:`ftir2dcos`.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ftir2dcos.config import (
    BaselineConfig,
    NormalizationConfig,
    PipelineConfig,
    SmoothingConfig,
    WavenumberRange,
)
from ftir2dcos.io import load_input
from ftir2dcos.models import SpectralDataset, ValidationReport
from ftir2dcos.peak_order import PeakOrderResult, PeakRequest
from ftir2dcos.pipeline import (
    CrossRangePipelineResult,
    MultiRangePipelineResult,
    PipelineResult,
    PreprocessingResult,
    analyze_multi_range_peak_order,
    preview_preprocessing,
    run_multi_range_pipeline,
)
from ftir2dcos.plotting import (
    create_2d_contour,
    create_baseline_qc_representative,
    create_multi_range_2d_contour,
    create_spectra_overlay,
)
from ftir2dcos.validation import estimate_2d_matrix_memory, validate_dataset

WIDE_SUFFIXES = {".csv", ".txt", ".tsv"}
UPLOAD_SUFFIXES = WIDE_SUFFIXES | {".dpt"}
DISPLAY_CONFIG_FIELDS = {"contour_levels", "display_percentile"}
DEFAULT_LOW_WAVENUMBER = 1509.0
DEFAULT_HIGH_WAVENUMBER = 1736.0
DEFAULT_WAVENUMBER_RANGES = (
    WavenumberRange(DEFAULT_HIGH_WAVENUMBER, DEFAULT_LOW_WAVENUMBER),
    WavenumberRange(1250.0, 1140.0),
)
AUTO_PEAK_RANGE = "Auto (unique covering range)"
PEAK_WAVENUMBER_COLUMN = "Wavenumber (cm⁻¹)"
PEAK_LABEL_COLUMN = "Label (optional)"
PEAK_RANGE_COLUMN = "Range (optional)"


@dataclass(frozen=True, slots=True)
class CachedRun:
    """In-memory result and portable bundle retained across Streamlit reruns."""

    result: MultiRangePipelineResult
    bundle_bytes: bytes
    file_names: tuple[str, ...]
    config_json: str


def parse_anchor_ranges(text: str) -> tuple[tuple[float, float], ...]:
    """Parse semicolon/newline-separated positive wavenumber interval pairs."""

    stripped = str(text).strip()
    if not stripped:
        return ()
    intervals: list[tuple[float, float]] = []
    for chunk in re.split(r"[;\n]+", stripped):
        values = re.findall(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", chunk)
        if len(values) != 2:
            raise ValueError(
                "Each anchor interval must contain two wavenumbers; "
                "separate intervals with ';' or a new line."
            )
        first, second = (float(value) for value in values)
        if not np.isfinite(first) or not np.isfinite(second) or first == second:
            raise ValueError("Anchor interval endpoints must be distinct finite values.")
        intervals.append((first, second))
    return tuple(intervals)


def load_uploaded_dataset(
    uploaded_files: list[Any] | tuple[Any, ...],
    *,
    delimiter: str | None = None,
) -> SpectralDataset:
    """Dispatch one uploaded wide table or an ordered collection of DPT files."""

    files = tuple(uploaded_files)
    if not files:
        raise ValueError("Upload one wide table or at least three DPT spectra.")
    suffixes = tuple(Path(str(getattr(item, "name", ""))).suffix.lower() for item in files)
    unsupported = [suffix for suffix in suffixes if suffix not in UPLOAD_SUFFIXES]
    if unsupported:
        raise ValueError(f"Unsupported uploaded file extension(s): {unsupported}")
    if all(suffix == ".dpt" for suffix in suffixes):
        source: Any = files
    elif len(files) == 1 and suffixes[0] in WIDE_SUFFIXES:
        source = files[0]
    else:
        raise ValueError(
            "Upload either one CSV/TXT/TSV wide table or only DPT files; "
            "wide tables and DPT files cannot be mixed."
        )
    return load_input(
        source,
        intensity_unit="unknown",
        delimiter=delimiter,
        perturbation_order="preserve_file_order",
    )


def dataset_preview_frame(dataset: SpectralDataset, *, rows: int = 10) -> pd.DataFrame:
    """Return a display-only wide-table preview without changing the dataset."""

    count = max(0, min(int(rows), dataset.n_wavenumbers))
    values = np.column_stack((dataset.wavenumber[:count], dataset.spectra[:, :count].T))
    return pd.DataFrame(values, columns=["Wavenumber", *dataset.perturbation_labels])


def perturbation_editor_frame(dataset: SpectralDataset) -> pd.DataFrame:
    """Build the fixed-row editor used to supply numeric perturbations."""

    return pd.DataFrame(
        {
            "File order": np.arange(1, dataset.n_spectra + 1, dtype=np.int64),
            "Original label": dataset.perturbation_labels,
            "Perturbation": dataset.perturbation,
        }
    )


def default_analysis_range(wavenumber: np.ndarray) -> tuple[float, float]:
    """Return the first default range as ``(low, high)`` for compatibility."""

    first = default_wavenumber_ranges(wavenumber)[0]
    return first.bounds


def default_wavenumber_ranges(wavenumber: np.ndarray) -> tuple[WavenumberRange, ...]:
    """Return applicable default intervals, falling back to the full finite axis."""

    axis = np.asarray(wavenumber, dtype=np.float64)
    finite = axis[np.isfinite(axis)]
    if not finite.size:
        raise ValueError("Wavenumber axis has no finite values.")
    available_low = float(np.min(finite))
    available_high = float(np.max(finite))
    applicable = tuple(
        item
        for item in DEFAULT_WAVENUMBER_RANGES
        if available_low <= item.low_wavenumber < item.high_wavenumber <= available_high
    )
    return applicable or (
        WavenumberRange(
            high_wavenumber=available_high,
            low_wavenumber=available_low,
            label="full range",
        ),
    )


def range_editor_frame(wavenumber: np.ndarray) -> pd.DataFrame:
    """Build a typed editable table containing the axis-appropriate defaults."""

    ranges = default_wavenumber_ranges(wavenumber)
    return pd.DataFrame(
        {
            "Label": pd.Series([item.label or "" for item in ranges], dtype="string"),
            "High wavenumber": pd.Series(
                [item.high_wavenumber for item in ranges], dtype="float64"
            ),
            "Low wavenumber": pd.Series([item.low_wavenumber for item in ranges], dtype="float64"),
        }
    )


def validate_wavenumber_ranges(
    edited: pd.DataFrame,
    wavenumber: np.ndarray,
    *,
    minimum_points: int = 10,
) -> tuple[WavenumberRange, ...]:
    """Strictly validate dynamic editor rows against the uploaded axis."""

    required = {"Label", "High wavenumber", "Low wavenumber"}
    missing = required.difference(edited.columns)
    if missing:
        raise ValueError(f"Range table is missing columns: {sorted(missing)}")
    if edited.empty:
        raise ValueError("Add at least one wavenumber range.")

    axis = np.asarray(wavenumber, dtype=np.float64)
    finite_axis = axis[np.isfinite(axis)]
    if not finite_axis.size:
        raise ValueError("Wavenumber axis has no finite values.")
    available_low = float(np.min(finite_axis))
    available_high = float(np.max(finite_axis))
    if int(minimum_points) < 1:
        raise ValueError("minimum_points must be positive")

    validated: list[WavenumberRange] = []
    seen: set[tuple[float, float]] = set()
    for position, (_, row) in enumerate(edited.iterrows(), start=1):
        high_value = pd.to_numeric(row["High wavenumber"], errors="coerce")
        low_value = pd.to_numeric(row["Low wavenumber"], errors="coerce")
        if not np.isfinite(high_value) or not np.isfinite(low_value):
            raise ValueError(f"Range row {position} requires two finite endpoints.")
        label_value = row["Label"]
        label = (
            None
            if pd.isna(label_value) or not str(label_value).strip()
            else str(label_value).strip()
        )
        try:
            analysis_range = WavenumberRange(
                high_wavenumber=float(high_value),
                low_wavenumber=float(low_value),
                label=label,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid range row {position}: {exc}") from exc
        if (
            analysis_range.low_wavenumber < available_low
            or analysis_range.high_wavenumber > available_high
        ):
            raise ValueError(
                f"Range row {position} ({analysis_range.display_name}) is outside the "
                f"available axis {available_high:g}-{available_low:g} cm^-1."
            )
        if analysis_range.bounds in seen:
            raise ValueError(
                f"Range row {position} duplicates {analysis_range.high_wavenumber:g}-"
                f"{analysis_range.low_wavenumber:g} cm^-1."
            )
        selected_points = int(
            np.count_nonzero(
                (axis >= analysis_range.low_wavenumber) & (axis <= analysis_range.high_wavenumber)
            )
        )
        if selected_points < int(minimum_points):
            raise ValueError(
                f"Range row {position} contains {selected_points} measured points; at least "
                f"{int(minimum_points)} are required."
            )
        seen.add(analysis_range.bounds)
        validated.append(analysis_range)
    return tuple(validated)


def common_wavenumber_bounds(
    ranges: tuple[WavenumberRange, ...],
) -> tuple[float, float] | None:
    """Return the common ``(low, high)`` overlap, if every range overlaps."""

    if not ranges:
        return None
    low = max(item.low_wavenumber for item in ranges)
    high = min(item.high_wavenumber for item in ranges)
    return (low, high) if low < high else None


def matrix_memory_estimate_frame(
    wavenumber: np.ndarray,
    ranges: tuple[WavenumberRange, ...],
    *,
    convention: str,
) -> tuple[pd.DataFrame, int]:
    """Estimate two float64 matrices for every self and unique cross block."""

    if convention not in {"canonical", "2dpy_compatible"}:
        raise ValueError("convention must be 'canonical' or '2dpy_compatible'")
    axis = np.asarray(wavenumber, dtype=np.float64)
    point_counts = tuple(
        int(
            np.count_nonzero(
                (axis >= analysis_range.low_wavenumber) & (axis <= analysis_range.high_wavenumber)
            )
        )
        for analysis_range in ranges
    )
    rows: list[dict[str, object]] = []
    total_bytes = 0
    for analysis_range, points in zip(ranges, point_counts, strict=True):
        two_matrix_bytes = int(estimate_2d_matrix_memory(points)["two_matrix_bytes"])
        total_bytes += two_matrix_bytes
        rows.append(
            {
                "Type": "Self",
                "Range / pair": analysis_range.display_name,
                "Matrix shape": f"{points} x {points}",
                "Two matrices (MiB)": two_matrix_bytes / 1024**2,
            }
        )

    item_size = np.dtype(np.float64).itemsize
    for first_index, first_range in enumerate(ranges):
        for second_index in range(first_index + 1, len(ranges)):
            second_range = ranges[second_index]
            first_points = point_counts[first_index]
            second_points = point_counts[second_index]
            row_points, column_points = (
                (second_points, first_points)
                if convention == "2dpy_compatible"
                else (first_points, second_points)
            )
            two_matrix_bytes = int(2 * row_points * column_points * item_size)
            total_bytes += two_matrix_bytes
            rows.append(
                {
                    "Type": "Cross",
                    "Range / pair": (f"{first_range.display_name} x {second_range.display_name}"),
                    "Matrix shape": f"{row_points} x {column_points}",
                    "Two matrices (MiB)": two_matrix_bytes / 1024**2,
                }
            )
    return pd.DataFrame(rows), total_bytes


def dataset_fingerprint(dataset: SpectralDataset) -> str:
    """Hash imported values and user-edited perturbations for cache safety."""

    digest = hashlib.sha256()
    for array in (dataset.wavenumber, dataset.perturbation, dataset.spectra):
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    digest.update(json.dumps(dataset.perturbation_labels, ensure_ascii=False).encode("utf-8"))
    digest.update(dataset.source_name.encode("utf-8"))
    source_hash = dataset.metadata.get("source_sha256")
    digest.update(
        json.dumps(source_hash, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    )
    return digest.hexdigest()


def source_fingerprint(dataset: SpectralDataset) -> str:
    """Hash source/axis values while ignoring user-edited perturbations."""

    digest = hashlib.sha256()
    for array in (dataset.wavenumber, dataset.spectra):
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    digest.update(dataset.source_name.encode("utf-8"))
    digest.update(
        json.dumps(
            dataset.metadata.get("source_sha256"),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _range_payload(ranges: tuple[WavenumberRange, ...]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in ranges]


def config_fingerprint(
    dataset: SpectralDataset,
    config: PipelineConfig,
    *,
    ranges: tuple[WavenumberRange, ...] | None = None,
    preprocessing_only: bool = False,
) -> str:
    """Return a deterministic fingerprint, excluding display-only controls."""

    payload = config.to_dict()
    for name in DISPLAY_CONFIG_FIELDS:
        payload.pop(name, None)
    if preprocessing_only:
        payload.pop("convention", None)
        payload.pop("grid_strategy", None)
    normalized_ranges = ranges
    if normalized_ranges is None and config.wavenumber_range is not None:
        normalized_ranges = (
            WavenumberRange(
                high_wavenumber=config.high_wavenumber,
                low_wavenumber=config.low_wavenumber,
            ),
        )
    payload["analysis_ranges"] = _range_payload(normalized_ranges or ())
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256()
    digest.update(dataset_fingerprint(dataset).encode("ascii"))
    digest.update(encoded.encode("utf-8"))
    return digest.hexdigest()


def execute_pipeline_bundle(
    dataset: SpectralDataset,
    ranges: tuple[WavenumberRange, ...],
    config: PipelineConfig,
) -> CachedRun:
    """Run all ranges through the shared pipeline and retain the aggregate ZIP."""

    with tempfile.TemporaryDirectory(prefix="ftir2dcos_streamlit_") as output_root:
        result = run_multi_range_pipeline(
            dataset,
            ranges,
            config,
            output_root=output_root,
            input_paths=(),
        )
        bundle_path = result.bundle_path
        if bundle_path is None or not bundle_path.is_file():
            raise RuntimeError("The pipeline completed without creating multi_range_bundle.zip.")
        bundle_bytes = bundle_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
        file_names = tuple(sorted(archive.namelist()))
    return CachedRun(
        result=result,
        bundle_bytes=bundle_bytes,
        file_names=file_names,
        config_json=json.dumps(
            {
                "analysis_ranges": _range_payload(ranges),
                "pipeline_config": config.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    )


def _delimiter_summary(dataset: SpectralDataset) -> str:
    if dataset.metadata.get("input_format") == "wide_table":
        return str(dataset.metadata.get("delimiter_name", "unknown"))
    sources = dataset.metadata.get("source_files", ())
    names = [str(item.get("delimiter_name", "unknown")) for item in sources]
    return ", ".join(dict.fromkeys(names)) or "unknown"


def _show_report(report: ValidationReport, *, include_errors: bool = True) -> None:
    if include_errors:
        for message in report.errors:
            st.error(message)
    for message in report.warnings:
        st.warning(message)


def _show_figure(figure: Any) -> None:
    try:
        st.pyplot(figure, width="stretch")
    finally:
        plt.close(figure)


def _initialize_session_state() -> None:
    st.session_state.setdefault("preview_cache", {})
    st.session_state.setdefault("run_cache", {})
    st.session_state.setdefault("peak_order_cache", {})
    st.session_state.setdefault("confirmed_fingerprint", None)


def _remember_bounded(cache: dict[str, Any], key: str, value: Any, *, limit: int) -> None:
    cache[key] = value
    while len(cache) > limit:
        oldest = next(iter(cache))
        if oldest == key and len(cache) > 1:
            oldest = next(item for item in cache if item != key)
        cache.pop(oldest, None)


def _render_upload() -> SpectralDataset | None:
    st.header("1. Upload and parse")
    uploaded_files = st.file_uploader(
        "Upload one wide CSV/TXT/TSV table or multiple two-column DPT files",
        type=sorted(suffix.lstrip(".") for suffix in UPLOAD_SUFFIXES),
        accept_multiple_files=True,
        help="DPT files are kept in the upload order; no interpolation or numeric sorting occurs.",
    )
    if not uploaded_files:
        st.info(
            "Upload data to begin. When covered by the uploaded axis, the default ranges "
            "are 1736-1509 and 1250-1140 cm⁻¹."
        )
        return None

    delimiter_label = st.selectbox(
        "Delimiter",
        ("Auto-detect", "Comma", "Tab", "Semicolon"),
        help="Auto-detection accepts comma, tab, and semicolon only.",
    )
    delimiter_map = {"Auto-detect": None, "Comma": ",", "Tab": "\t", "Semicolon": ";"}
    try:
        dataset = load_uploaded_dataset(
            uploaded_files,
            delimiter=delimiter_map[delimiter_label],
        )
    except Exception as exc:
        st.error(f"The uploaded data could not be parsed: {exc}")
        return None

    finite_axis = dataset.wavenumber[np.isfinite(dataset.wavenumber)]
    columns = st.columns(4)
    columns[0].metric("Delimiter", _delimiter_summary(dataset))
    columns[1].metric("Spectra", dataset.n_spectra)
    columns[2].metric("Wavenumber points", dataset.n_wavenumbers)
    if finite_axis.size:
        range_text = f"{np.max(finite_axis):g}-{np.min(finite_axis):g} cm⁻¹"
    else:
        range_text = "Unavailable"
    columns[3].metric("Available range", range_text)
    st.caption(
        "Internal shape: "
        f"{dataset.shape} (spectra x wavenumbers); detected direction: "
        f"{dataset.metadata.get('original_wavenumber_direction', 'unknown')}."
    )
    st.dataframe(dataset_preview_frame(dataset), width="stretch", hide_index=True)
    for warning in dataset.metadata.get("parse_warnings", ()):  # parser-owned warnings
        st.warning(str(warning))
    if any(Path(file.name).name.casefold() == "baseline.dpt" for file in uploaded_files):
        st.warning(
            "BASELINE.dpt is included because it was explicitly uploaded. Remove it if it is "
            "an instrument background rather than a perturbation spectrum."
        )
    return dataset


def _render_perturbations(dataset: SpectralDataset) -> tuple[SpectralDataset, str] | None:
    st.header("2. Perturbation variable")
    input_key = dataset_fingerprint(dataset)[:16]
    edited = st.data_editor(
        perturbation_editor_frame(dataset),
        key=f"perturbation_editor_{input_key}",
        disabled=("File order", "Original label"),
        hide_index=True,
        num_rows="fixed",
        width="stretch",
        column_config={
            "Perturbation": st.column_config.NumberColumn(
                "Perturbation",
                help="Enter a finite numeric value for every spectrum.",
                format="%.8g",
            )
        },
    )
    numeric = pd.to_numeric(edited["Perturbation"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        st.error("Every perturbation value must be filled with a finite number.")
        return None
    edited_dataset = dataset.with_updates(perturbation=numeric)

    order_label = st.radio(
        "Spectrum order",
        ("Preserve file order", "Sort by perturbation"),
        horizontal=True,
        key=f"perturbation_order_{input_key}",
        help="Sorting is explicit and is recorded in exported metadata.",
    )
    order = (
        "preserve_file_order" if order_label == "Preserve file order" else "sort_by_perturbation"
    )
    report = validate_dataset(edited_dataset)
    intervals = report.metrics.get("perturbation_intervals")
    if intervals is not None:
        st.caption(f"Intervals in current file order: {intervals}")
    _show_report(report)
    if report.errors:
        return None
    return edited_dataset, order


def _render_wavenumber_ranges(
    dataset: SpectralDataset,
    *,
    key_prefix: str,
) -> tuple[WavenumberRange, ...] | None:
    st.subheader("Analysis ranges")
    st.caption(
        "Add or delete rows as needed. Each interval is preprocessed independently with the "
        "same settings; when two or more intervals are present, every unique pair is also "
        "cross-correlated within the same ordered FTIR sequence."
    )
    edited = st.data_editor(
        range_editor_frame(dataset.wavenumber),
        key=f"wavenumber_ranges_{key_prefix}",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "Label": st.column_config.TextColumn(
                "Label (optional)",
                help="A short name recorded in the aggregate manifest and result selector.",
            ),
            "High wavenumber": st.column_config.NumberColumn(
                "High wavenumber (cm⁻¹)",
                format="%.8g",
            ),
            "Low wavenumber": st.column_config.NumberColumn(
                "Low wavenumber (cm⁻¹)",
                format="%.8g",
            ),
        },
    )
    try:
        return validate_wavenumber_ranges(edited, dataset.wavenumber)
    except (TypeError, ValueError) as exc:
        st.error(f"Invalid analysis ranges: {exc}")
        return None


def _render_baseline_config(
    prefix: str,
    low: float,
    high: float,
    *,
    ranges: tuple[WavenumberRange, ...] = (),
) -> BaselineConfig:
    method = st.selectbox(
        "Baseline method",
        ("none", "offset", "anchor_polynomial", "asls", "rubberband"),
        key=f"baseline_method_{prefix}",
    )
    options: dict[str, Any] = {"method": method}
    if method == "offset":
        common_bounds = common_wavenumber_bounds(ranges)
        offset_options = ("minimum", "window_median") if common_bounds is not None else ("minimum",)
        if common_bounds is None and len(ranges) > 1:
            st.info(
                "Window-median offset requires one window shared by every range. The "
                "selected ranges do not overlap, so minimum offset is used instead."
            )
        offset_mode = st.selectbox(
            "Offset estimator",
            offset_options,
            index=1 if len(offset_options) == 2 else 0,
            key=f"offset_mode_{prefix}",
        )
        options["offset_mode"] = offset_mode
        if offset_mode == "window_median":
            window_low_limit, window_high_limit = common_bounds or (low, high)
            first, second = st.columns(2)
            window_low = first.number_input(
                "Offset window endpoint 1 (cm⁻¹)",
                value=float(window_low_limit),
                key=f"offset_low_{prefix}",
            )
            window_high = second.number_input(
                "Offset window endpoint 2 (cm⁻¹)",
                value=float(min(window_low_limit + 15.0, window_high_limit)),
                key=f"offset_high_{prefix}",
            )
            options["offset_window"] = (window_low, window_high)
    elif method == "anchor_polynomial":
        anchor_source = ranges or (WavenumberRange(high_wavenumber=high, low_wavenumber=low),)
        default_anchor = "; ".join(
            (
                f"{item.low_wavenumber:g}, "
                f"{min(item.low_wavenumber + 15.0, item.high_wavenumber):g}; "
                f"{max(item.high_wavenumber - 15.0, item.low_wavenumber):g}, "
                f"{item.high_wavenumber:g}"
            )
            for item in anchor_source
        )
        anchor_text = st.text_area(
            "Anchor intervals (cm⁻¹)",
            value=default_anchor,
            key=f"anchor_ranges_{prefix}",
            help="Use two endpoints per interval; separate intervals with ';' or new lines.",
        )
        options["anchor_ranges"] = parse_anchor_ranges(anchor_text)
        options["polynomial_order"] = st.selectbox(
            "Polynomial order",
            (0, 1, 2, 3),
            index=1,
            key=f"polynomial_order_{prefix}",
        )
    elif method == "asls":
        first, second = st.columns(2)
        options["asls_lam"] = first.number_input(
            "AsLS λ",
            min_value=1.0,
            value=1.0e6,
            format="%.4g",
            key=f"asls_lam_{prefix}",
        )
        options["asls_p"] = second.number_input(
            "AsLS p",
            min_value=0.000001,
            max_value=0.999999,
            value=0.01,
            format="%.6f",
            key=f"asls_p_{prefix}",
        )
        third, fourth, fifth = st.columns(3)
        options["asls_diff_order"] = third.number_input(
            "Difference order", min_value=1, value=2, key=f"asls_diff_{prefix}"
        )
        options["asls_max_iter"] = fourth.number_input(
            "Maximum iterations", min_value=1, value=50, key=f"asls_iter_{prefix}"
        )
        options["asls_tol"] = fifth.number_input(
            "Tolerance",
            min_value=0.0,
            value=1.0e-3,
            format="%.3g",
            key=f"asls_tol_{prefix}",
        )
    elif method == "rubberband":
        st.info(
            "Rubberband is most suitable for convex backgrounds and can perform poorly for "
            "strongly concave backgrounds. Inspect the representative preview."
        )
        first, second = st.columns(2)
        options["rubberband_segments"] = first.number_input(
            "Segments", min_value=1, value=1, key=f"rubber_segments_{prefix}"
        )
        options["rubberband_diff_order"] = second.number_input(
            "Difference order", min_value=1, value=2, key=f"rubber_diff_{prefix}"
        )
        use_lam = st.checkbox("Set rubberband λ", key=f"rubber_use_lam_{prefix}")
        if use_lam:
            options["rubberband_lam"] = st.number_input(
                "Rubberband λ",
                min_value=0.0,
                value=0.0,
                format="%.4g",
                key=f"rubber_lam_{prefix}",
            )
        use_smoothing = st.checkbox(
            "Set rubberband smoothing half-window",
            key=f"rubber_use_smooth_{prefix}",
        )
        if use_smoothing:
            options["rubberband_smooth_half_window"] = st.number_input(
                "Smoothing half-window",
                min_value=0,
                value=1,
                key=f"rubber_smooth_{prefix}",
            )
    return BaselineConfig(**options)


def _render_smoothing_config(prefix: str) -> SmoothingConfig:
    enabled = st.checkbox(
        "Enable Savitzky-Golay smoothing",
        value=False,
        key=f"smoothing_enabled_{prefix}",
        help="Disabled by default. The unsmoothed baseline-corrected spectra remain exported.",
    )
    if not enabled:
        return SmoothingConfig(enabled=False)
    first, second = st.columns(2)
    window_length = first.number_input(
        "SG window length", min_value=3, value=7, step=2, key=f"sg_window_{prefix}"
    )
    polyorder = second.number_input(
        "SG polynomial order", min_value=0, value=2, key=f"sg_polyorder_{prefix}"
    )
    mode = st.selectbox(
        "SG edge mode",
        ("interp", "mirror", "nearest", "constant", "wrap"),
        key=f"sg_mode_{prefix}",
    )
    return SmoothingConfig(
        enabled=True,
        window_length=window_length,
        polyorder=polyorder,
        mode=mode,
    )


def _render_normalization_config(
    prefix: str,
    low: float,
    high: float,
    *,
    ranges: tuple[WavenumberRange, ...] = (),
) -> NormalizationConfig:
    common_bounds = common_wavenumber_bounds(ranges)
    methods = (
        ("none", "vector", "reference_peak") if common_bounds is not None else ("none", "vector")
    )
    method = st.selectbox(
        "Normalization",
        methods,
        key=f"normalization_{prefix}",
        help="Normalization is disabled by default.",
    )
    if common_bounds is None and len(ranges) > 1:
        st.caption(
            "Reference-peak normalization requires one peak window present in every "
            "range, so it is unavailable for the selected non-overlapping ranges."
        )
    if method != "reference_peak":
        return NormalizationConfig(method=method)
    peak_default_low, peak_default_high = common_bounds or (low, high)
    first, second = st.columns(2)
    peak_low = first.number_input(
        "Reference peak endpoint 1 (cm⁻¹)",
        value=float(peak_default_low),
        key=f"norm_low_{prefix}",
    )
    peak_high = second.number_input(
        "Reference peak endpoint 2 (cm⁻¹)",
        value=float(peak_default_high),
        key=f"norm_high_{prefix}",
    )
    return NormalizationConfig(method="reference_peak", reference_peak_range=(peak_low, peak_high))


def _render_preprocessing(
    dataset: SpectralDataset,
    perturbation_order: str,
) -> tuple[PipelineConfig, tuple[WavenumberRange, ...]] | None:
    st.header("3. Preprocessing and 2D-COS settings")
    prefix = source_fingerprint(dataset)[:16]
    ranges = _render_wavenumber_ranges(dataset, key_prefix=prefix)
    if ranges is None:
        return None
    finite_axis = dataset.wavenumber[np.isfinite(dataset.wavenumber)]
    if not finite_axis.size:
        st.error("The uploaded wavenumber axis has no finite values.")
        return None
    available_low = float(np.min(finite_axis))
    available_high = float(np.max(finite_axis))
    baseline_low = min(item.low_wavenumber for item in ranges) if ranges else available_low
    baseline_high = max(item.high_wavenumber for item in ranges) if ranges else available_high
    intensity_unit = st.selectbox(
        "Input intensity unit",
        ("absorbance", "percent_transmittance", "fraction_transmittance", "unknown"),
        key=f"intensity_unit_{prefix}",
        help="Transmittance is converted only when explicitly selected.",
    )

    try:
        baseline = _render_baseline_config(
            prefix,
            baseline_low,
            baseline_high,
            ranges=ranges,
        )
        smoothing = _render_smoothing_config(prefix)
        normalization = _render_normalization_config(
            prefix,
            baseline_low,
            baseline_high,
            ranges=ranges,
        )
    except (TypeError, ValueError) as exc:
        st.error(f"Invalid preprocessing settings: {exc}")
        return None

    st.subheader("2D-COS and display")
    convention = st.selectbox(
        "Matrix convention",
        ("2dpy_compatible", "canonical"),
        key=f"convention_{prefix}",
        help="2dpy_compatible is the first-release default and records its final transpose.",
    )
    display_first, display_second = st.columns(2)
    contour_levels = display_first.number_input(
        "Contour levels",
        min_value=2,
        value=21,
        key=f"contour_levels_{prefix}",
    )
    display_percentile = display_second.slider(
        "Symmetric color display percentile",
        min_value=80.0,
        max_value=100.0,
        value=99.0,
        step=0.5,
        key=f"display_percentile_{prefix}",
        help="Display scaling never clips or changes exported matrix values.",
    )

    try:
        config = PipelineConfig(
            input_intensity_unit=intensity_unit,
            perturbation_order=perturbation_order,
            baseline=baseline,
            smoothing=smoothing,
            normalization=normalization,
            convention=convention,
            contour_levels=contour_levels,
            display_percentile=display_percentile,
        )
    except (TypeError, ValueError) as exc:
        st.error(f"Invalid pipeline configuration: {exc}")
        return None

    for analysis_range in ranges:
        selected_points = int(
            np.count_nonzero(
                (dataset.wavenumber >= analysis_range.low_wavenumber)
                & (dataset.wavenumber <= analysis_range.high_wavenumber)
            )
        )
        if selected_points > 2500:
            st.warning(
                f"{analysis_range.display_name} contains more than 2500 points; matrix "
                "CSV and figure exports may be large."
            )
    memory_frame, total_bytes = matrix_memory_estimate_frame(
        dataset.wavenumber,
        ranges,
        convention=config.convention,
    )
    st.dataframe(
        memory_frame,
        hide_index=True,
        width="stretch",
        column_config={"Two matrices (MiB)": st.column_config.NumberColumn(format="%.1f")},
    )
    st.caption(
        f"Approximate minimum for synchronous + asynchronous float64 matrices across "
        f"{len(ranges)} self block(s) and {len(ranges) * (len(ranges) - 1) // 2} unique "
        f"cross block(s): {total_bytes / 1024**2:.1f} MiB. Runtime canonical/convention "
        "arrays, dynamic spectra, figures, CSV files, and export overhead require more."
    )
    return config, ranges


def generate_range_previews(
    dataset: SpectralDataset,
    ranges: tuple[WavenumberRange, ...],
    config: PipelineConfig,
) -> tuple[PreprocessingResult, ...]:
    """Generate each inspectable preview through the shared preprocessing API."""

    generated: list[PreprocessingResult] = []
    for analysis_range in ranges:
        try:
            generated.append(preview_preprocessing(dataset, config.for_range(analysis_range)))
        except Exception as exc:
            raise ValueError(f"Preview failed for {analysis_range.display_name}: {exc}") from exc
    return tuple(generated)


def _render_preview(
    dataset: SpectralDataset,
    ranges: tuple[WavenumberRange, ...],
    config: PipelineConfig,
) -> bool:
    st.header("4. Inspect and confirm every baseline preview")
    preview_key = config_fingerprint(
        dataset,
        config,
        ranges=ranges,
        preprocessing_only=True,
    )
    preview_cache: dict[str, tuple[PreprocessingResult, ...]] = st.session_state["preview_cache"]
    if st.button("Generate / refresh representative preview", type="primary"):
        try:
            with st.spinner(f"Applying preprocessing to {len(ranges)} range(s)…"):
                previews = generate_range_previews(dataset, ranges, config)
            _remember_bounded(preview_cache, preview_key, previews, limit=5)
        except Exception as exc:
            st.error(f"The preprocessing preview could not be generated: {exc}")

    previews = preview_cache.get(preview_key)
    if previews is None:
        st.info(
            "Generate and confirm the first/middle/last raw-baseline-corrected preview for "
            "every analysis range before running 2D-COS."
        )
        return False

    selected_index = st.selectbox(
        "Range preview to inspect",
        options=tuple(range(len(ranges))),
        format_func=lambda index: ranges[index].display_name,
        key=f"preview_range_{preview_key}",
    )
    preview = previews[selected_index]
    selected_range = ranges[selected_index]
    st.subheader(selected_range.display_name)
    try:
        figure = create_baseline_qc_representative(
            preview.selected_raw.wavenumber,
            preview.selected_raw.spectra,
            preview.baselines,
            preview.baseline_corrected.spectra,
            labels=preview.selected_raw.perturbation_labels,
            intensity_label=preview.baseline_corrected.intensity_unit,
        )
        _show_figure(figure)
    except Exception as exc:
        st.error(f"The representative preview could not be drawn: {exc}")
        return False
    for warning in preview.warnings:
        st.warning(warning)
    with st.expander("Baseline diagnostics"):
        st.dataframe(pd.DataFrame(preview.baseline_diagnostics), width="stretch")

    run_key = config_fingerprint(dataset, config, ranges=ranges)
    st.caption(f"Current calculation fingerprint: `{run_key[:16]}`")
    confirmation_key = f"confirmed_{run_key}_{selected_index}"
    st.checkbox(
        f"I inspected and confirm the raw, baseline, and corrected curves for "
        f"{selected_range.display_name}.",
        key=confirmation_key,
        persist_state="session",
    )
    confirmation_keys = [f"confirmed_{run_key}_{index}" for index in range(len(ranges))]
    confirmed_values = [bool(st.session_state.get(key, False)) for key in confirmation_keys]
    st.dataframe(
        pd.DataFrame(
            {
                "Range": [item.display_name for item in ranges],
                "Preview confirmed": confirmed_values,
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={"Preview confirmed": st.column_config.CheckboxColumn(disabled=True)},
    )
    all_confirmed = all(confirmed_values)
    if all_confirmed:
        st.session_state["confirmed_fingerprint"] = run_key
    elif st.session_state.get("confirmed_fingerprint") == run_key:
        st.session_state["confirmed_fingerprint"] = None
    if not all_confirmed:
        st.info(
            f"Confirmed {sum(confirmed_values)} of {len(ranges)} range previews. Select and "
            "inspect each remaining range."
        )
    return bool(all_confirmed and st.session_state.get("confirmed_fingerprint") == run_key)


def _matrix_summary(matrix: np.ndarray) -> pd.DataFrame:
    values = np.asarray(matrix, dtype=np.float64)
    return pd.DataFrame(
        {
            "shape": [f"{values.shape[0]} x {values.shape[1]}"],
            "minimum": [float(np.min(values))],
            "maximum": [float(np.max(values))],
            "max |value|": [float(np.max(np.abs(values)))],
        }
    )


def axis_direction(wavenumber: np.ndarray) -> str:
    """Describe the stored order of one wavenumber coordinate array."""

    values = np.asarray(wavenumber, dtype=np.float64)
    if values.ndim != 1 or not values.size:
        raise ValueError("wavenumber must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        return "contains non-finite values"
    if values.size == 1:
        return "single point"
    differences = np.diff(values)
    if np.all(differences < 0.0):
        return "descending (high → low)"
    if np.all(differences > 0.0):
        return "ascending (low → high)"
    return "non-monotonic"


def cross_axis_frame(cross_result: CrossRangePipelineResult) -> pd.DataFrame:
    """Build explicit row/column metadata for one convention-oriented cross block."""

    analysis = cross_result.twodcos
    return pd.DataFrame(
        {
            "Matrix axis": ("Row (vertical)", "Column (horizontal)"),
            "Analysis range": (
                cross_result.row_range.display_name,
                cross_result.column_range.display_name,
            ),
            "Variable": (analysis.row_variable, analysis.column_variable),
            "Stored array order": (
                axis_direction(analysis.row_wavenumber),
                axis_direction(analysis.column_wavenumber),
            ),
            "Displayed contour direction": (
                "high → low cm⁻¹",
                "high → low cm⁻¹",
            ),
            "Points": (
                int(analysis.row_wavenumber.size),
                int(analysis.column_wavenumber.size),
            ),
        }
    )


def bundle_member_bytes(bundle_bytes: bytes, member_name: str) -> bytes | None:
    """Read one known aggregate-export member without extracting the ZIP."""

    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
        try:
            return archive.read(member_name)
        except KeyError:
            return None


def peak_range_options(result: MultiRangePipelineResult) -> tuple[str, ...]:
    """Return stable, one-based range choices for the peak editor."""

    return (
        AUTO_PEAK_RANGE,
        *(
            f"Range {index + 1}: {item.analysis_range.display_name}"
            for index, item in enumerate(result.range_results)
        ),
    )


def peak_editor_frame() -> pd.DataFrame:
    """Return an empty, explicitly typed frame for dynamic peak entry."""

    return pd.DataFrame(
        {
            PEAK_WAVENUMBER_COLUMN: pd.Series(dtype="float64"),
            PEAK_LABEL_COLUMN: pd.Series(dtype="string"),
            PEAK_RANGE_COLUMN: pd.Series(dtype="string"),
        }
    )


def validate_peak_requests(
    frame: pd.DataFrame,
    result: MultiRangePipelineResult,
) -> tuple[PeakRequest, ...]:
    """Validate editor rows and convert one-based UI ranges to core requests."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Peak input must be a table.")
    required = {PEAK_WAVENUMBER_COLUMN, PEAK_LABEL_COLUMN, PEAK_RANGE_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Peak table is missing column(s): {', '.join(sorted(missing))}")

    options = peak_range_options(result)
    range_indices = {label: index for index, label in enumerate(options[1:])}
    requests: list[PeakRequest] = []
    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        raw_wavenumber = row[PEAK_WAVENUMBER_COLUMN]
        raw_label = row[PEAK_LABEL_COLUMN]
        raw_range = row[PEAK_RANGE_COLUMN]
        label = "" if pd.isna(raw_label) else str(raw_label).strip()
        range_label = "" if pd.isna(raw_range) else str(raw_range).strip()
        if pd.isna(raw_wavenumber):
            if not label and (not range_label or range_label == AUTO_PEAK_RANGE):
                continue
            raise ValueError(f"Peak row {row_number} needs a finite wavenumber.")
        try:
            wavenumber = float(raw_wavenumber)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Peak row {row_number} has an invalid wavenumber.") from exc
        if not np.isfinite(wavenumber):
            raise ValueError(f"Peak row {row_number} needs a finite wavenumber.")

        if not range_label or range_label == AUTO_PEAK_RANGE:
            range_index = None
        else:
            try:
                range_index = range_indices[range_label]
            except KeyError as exc:
                raise ValueError(
                    f"Peak row {row_number} has an unknown analysis range: {range_label}"
                ) from exc
        requests.append(
            PeakRequest(
                wavenumber=wavenumber,
                label=label or None,
                range_index=range_index,
            )
        )

    if len(requests) < 2:
        raise ValueError("Enter at least two peak positions before analysis.")
    return tuple(requests)


def _peak_label(peak: PeakRequest) -> str:
    """Render a peak without hiding its requested position behind a custom label."""

    label = f"{peak.wavenumber:g} cm⁻¹"
    if peak.label:
        label = f"{peak.label} ({label})"
    if peak.range_index is not None:
        label += f" [R{peak.range_index + 1}]"
    return label


def peak_order_chain_text(result: PeakOrderResult) -> str:
    """Render a total or layered graph order without implying tied simultaneity."""

    if result.is_unique_total_order:
        return " → ".join(_peak_label(peak) for peak in result.unique_order)
    if result.topological_layers:
        layers = []
        for layer in result.topological_layers:
            labels = " ∥ ".join(_peak_label(peak) for peak in layer)
            layers.append(labels if len(layer) == 1 else f"{{{labels}}}")
        return " → ".join(layers)

    component_layers: list[str] = []
    for layer in result.component_layers:
        components: list[str] = []
        for component in layer:
            labels = " ↻ ".join(_peak_label(peak) for peak in component)
            components.append(labels if len(component) == 1 else f"cycle({labels})")
        component_layers.append(" ∥ ".join(components))
    return " → ".join(component_layers) or "No graph-supported order"


def _peak_range_name(
    peak: PeakRequest,
    result: MultiRangePipelineResult,
) -> str:
    if peak.range_index is None:
        return "Auto / unresolved"
    item = result.range_results[peak.range_index]
    return f"R{peak.range_index + 1}: {item.analysis_range.display_name}"


def peak_evidence_frame(
    order: PeakOrderResult,
    result: MultiRangePipelineResult,
) -> pd.DataFrame:
    """Build the user-facing and downloadable pairwise audit table."""

    rows: list[dict[str, object]] = []
    for evidence in order.evidence:
        if evidence.earlier is None or evidence.later is None:
            decision = evidence.relation.value
        else:
            decision = f"{_peak_label(evidence.earlier)} → {_peak_label(evidence.later)}"
        rows.append(
            {
                "First peak": _peak_label(evidence.first),
                "First requested (cm⁻¹)": evidence.first.wavenumber,
                "First matched (cm⁻¹)": evidence.matched_first_wavenumber,
                "First range": _peak_range_name(evidence.first, result),
                "Second peak": _peak_label(evidence.second),
                "Second requested (cm⁻¹)": evidence.second.wavenumber,
                "Second matched (cm⁻¹)": evidence.matched_second_wavenumber,
                "Second range": _peak_range_name(evidence.second, result),
                "Synchronous Φ": evidence.synchronous,
                "Asynchronous Ψ": evidence.asynchronous,
                "Phi * Psi": evidence.value_product,
                "Sign product": evidence.sign_product,
                "Decision": decision,
                "Relative signal strength": evidence.relative_signal_strength,
                "Minimum signal/cutoff ratio": evidence.minimum_cutoff_ratio,
                "Relation": evidence.relation.value,
                "Source block": evidence.source,
                "Reason": evidence.reason,
            }
        )
    return pd.DataFrame(rows)


def _peak_order_cache_key(
    fingerprint: str,
    peaks: tuple[PeakRequest, ...],
    *,
    tolerance: float,
    synchronous_threshold: float,
    asynchronous_threshold: float,
    relative_threshold: float,
) -> str:
    payload = {
        "run": fingerprint,
        "peaks": [peak.to_dict() for peak in peaks],
        "tolerance": tolerance,
        "synchronous_threshold": synchronous_threshold,
        "asynchronous_threshold": asynchronous_threshold,
        "relative_threshold": relative_threshold,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _render_matrix_result(
    result: PipelineResult,
    config: PipelineConfig,
    *,
    kind: str,
    fingerprint: str,
) -> None:
    analysis = result.twodcos
    matrix = analysis.synchronous if kind == "synchronous" else analysis.asynchronous
    figure = create_2d_contour(
        analysis.row_wavenumber,
        matrix,
        column_wavenumber=analysis.column_wavenumber,
        kind=kind,
        convention=analysis.convention,
        row_variable=analysis.row_variable,
        column_variable=analysis.column_variable,
        method=config.baseline.method,
        contour_levels=config.contour_levels,
        display_percentile=config.display_percentile,
    )
    _show_figure(figure)
    st.dataframe(_matrix_summary(matrix), width="stretch", hide_index=True)
    if st.checkbox(
        "Show a 30 x 30 numeric preview",
        key=f"matrix_preview_{kind}_{fingerprint}",
    ):
        row_count = min(30, matrix.shape[0])
        column_count = min(30, matrix.shape[1])
        frame = pd.DataFrame(
            matrix[:row_count, :column_count],
            index=analysis.row_wavenumber[:row_count],
            columns=analysis.column_wavenumber[:column_count],
        )
        st.dataframe(frame, width="stretch")


def _render_cross_matrix_result(
    cross_result: CrossRangePipelineResult,
    config: PipelineConfig,
    *,
    kind: str,
    fingerprint: str,
) -> None:
    analysis = cross_result.twodcos
    matrix = analysis.synchronous if kind == "synchronous" else analysis.asynchronous
    st.caption(
        "Rectangular cross-range block from two independently preprocessed windows in "
        "the same ordered FTIR sequence. No second instrument or second experiment is implied."
    )
    figure = create_2d_contour(
        analysis.row_wavenumber,
        matrix,
        column_wavenumber=analysis.column_wavenumber,
        kind=kind,
        convention=analysis.convention,
        row_variable=analysis.row_variable,
        column_variable=analysis.column_variable,
        method=f"{config.baseline.method}; same-series cross-range",
        contour_levels=config.contour_levels,
        display_percentile=config.display_percentile,
        show_diagonal=False,
    )
    _show_figure(figure)
    st.dataframe(_matrix_summary(matrix), width="stretch", hide_index=True)
    if st.checkbox(
        "Show a 30 x 30 numeric preview",
        key=f"cross_matrix_preview_{kind}_{fingerprint}",
    ):
        row_count = min(30, matrix.shape[0])
        column_count = min(30, matrix.shape[1])
        frame = pd.DataFrame(
            matrix[:row_count, :column_count],
            index=analysis.row_wavenumber[:row_count],
            columns=analysis.column_wavenumber[:column_count],
        )
        st.dataframe(frame, width="stretch")


def _render_files(cached: CachedRun, fingerprint: str, *, context: str) -> None:
    st.dataframe(
        pd.DataFrame({"Files in aggregate ZIP": cached.file_names}),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download complete multi-range ZIP",
        data=cached.bundle_bytes,
        file_name=f"ftir2dcos_multi_{fingerprint[:12]}.zip",
        mime="application/zip",
        type="primary",
        key=f"download_bundle_{context}_{fingerprint}",
    )
    with st.expander("Configuration recorded for this run"):
        st.code(cached.config_json, language="json")


def _render_independent_results(
    cached: CachedRun,
    config: PipelineConfig,
    fingerprint: str,
) -> None:
    """Render one square self-correlation result at a time."""

    range_results = cached.result.range_results
    st.subheader("Independent range self-correlation")
    st.caption(
        "Each square map correlates one independently preprocessed wavenumber window with itself."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Range": [item.analysis_range.display_name for item in range_results],
                "Processed shape": [
                    f"{item.result.processed.n_spectra} x {item.result.processed.n_wavenumbers}"
                    for item in range_results
                ],
                "QC passed": [
                    bool(item.result.qc_metrics["all_checks_passed"]) for item in range_results
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    selected_index = st.selectbox(
        "Result range",
        options=tuple(range(len(range_results))),
        format_func=lambda index: range_results[index].analysis_range.display_name,
        key=f"result_range_{fingerprint}",
    )
    selected = range_results[selected_index]
    result = selected.result
    st.subheader(selected.analysis_range.display_name)
    tabs = st.tabs(
        (
            "Processed spectra",
            "Dynamic spectra",
            "Synchronous",
            "Asynchronous",
            "QC",
            "Files",
        ),
        key=f"result_tabs_{fingerprint}_{selected_index}",
        on_change="rerun",
    )
    if tabs[0].open:
        with tabs[0]:
            _show_figure(
                create_spectra_overlay(
                    result.processed.wavenumber,
                    result.processed.spectra,
                    labels=result.processed.perturbation_labels,
                    title="Processed FTIR spectra",
                    intensity_label=result.processed.intensity_unit,
                )
            )
            st.dataframe(
                dataset_preview_frame(result.processed),
                width="stretch",
                hide_index=True,
            )
    if tabs[1].open:
        with tabs[1]:
            _show_figure(
                create_spectra_overlay(
                    result.processed.wavenumber,
                    result.twodcos.dynamic,
                    labels=result.processed.perturbation_labels,
                    title="Dynamic FTIR spectra",
                    intensity_label="Dynamic intensity",
                    palette="coolwarm",
                )
            )
    if tabs[2].open:
        with tabs[2]:
            _render_matrix_result(
                result,
                config,
                kind="synchronous",
                fingerprint=f"{fingerprint}_{selected_index}",
            )
    if tabs[3].open:
        with tabs[3]:
            _render_matrix_result(
                result,
                config,
                kind="asynchronous",
                fingerprint=f"{fingerprint}_{selected_index}",
            )
    if tabs[4].open:
        with tabs[4]:
            st.json(result.qc_metrics)
            if result.warnings:
                st.subheader("Warnings")
                for warning in result.warnings:
                    st.warning(warning)
            else:
                st.success("No pipeline warnings were reported for this range.")
    if tabs[5].open:
        with tabs[5]:
            _render_files(cached, fingerprint, context=f"independent_{selected_index}")


def _render_cross_pair_results(
    cached: CachedRun,
    config: PipelineConfig,
    fingerprint: str,
) -> None:
    """Render one rectangular same-series cross-range result at a time."""

    cross_results = cached.result.cross_results
    st.subheader("Cross-range pair")
    st.info(
        "These rectangular blocks compare two spectral windows from the same ordered "
        "FTIR sequence. They are distinct from the square self-correlation maps and do "
        "not represent measurements from different instruments."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Pair": [item.pair_label for item in cross_results],
                "Matrix shape": [
                    f"{item.twodcos.synchronous.shape[0]} x {item.twodcos.synchronous.shape[1]}"
                    for item in cross_results
                ],
                "Row range": [item.row_range.display_name for item in cross_results],
                "Column range": [item.column_range.display_name for item in cross_results],
                "Cross QC passed": [
                    bool(item.qc_metrics.get("all_checks_passed", False)) for item in cross_results
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    selected_index = st.selectbox(
        "Cross-range pair",
        options=tuple(range(len(cross_results))),
        format_func=lambda index: cross_results[index].pair_label,
        key=f"cross_pair_{fingerprint}",
    )
    selected = cross_results[selected_index]
    analysis = selected.twodcos
    st.caption(
        f"Convention: `{analysis.convention}`. The table follows the stored matrix "
        "orientation after that convention is applied."
    )
    st.dataframe(cross_axis_frame(selected), hide_index=True, width="stretch")

    pair_key = f"{selected.first_index}_{selected.second_index}"
    tabs = st.tabs(
        ("Synchronous block", "Asynchronous block", "Cross QC", "Files"),
        key=f"cross_tabs_{fingerprint}_{pair_key}",
        on_change="rerun",
    )
    if tabs[0].open:
        with tabs[0]:
            _render_cross_matrix_result(
                selected,
                config,
                kind="synchronous",
                fingerprint=f"{fingerprint}_{pair_key}",
            )
    if tabs[1].open:
        with tabs[1]:
            _render_cross_matrix_result(
                selected,
                config,
                kind="asynchronous",
                fingerprint=f"{fingerprint}_{pair_key}",
            )
    if tabs[2].open:
        with tabs[2]:
            if bool(selected.qc_metrics.get("all_checks_passed", False)):
                st.success("All numerical checks passed for this cross-range pair.")
            else:
                st.warning("One or more numerical checks failed for this cross-range pair.")
            st.json(selected.qc_metrics)
            st.caption(
                "Reverse-map transpose identities are included in these checks; the reverse "
                "asynchronous block carries the required sign change."
            )
    if tabs[3].open:
        with tabs[3]:
            _render_files(cached, fingerprint, context=f"cross_{pair_key}")


def _block_matrix(
    result: MultiRangePipelineResult,
    *,
    row_index: int,
    column_index: int,
    kind: str,
) -> np.ndarray:
    """Select one convention-oriented self/cross block for display only."""

    if kind not in {"synchronous", "asynchronous"}:
        raise ValueError("kind must be 'synchronous' or 'asynchronous'")
    if row_index == column_index:
        return np.asarray(getattr(result.range_results[row_index].result.twodcos, kind))
    first_index, second_index = sorted((row_index, column_index))
    cross_result = next(
        item
        for item in result.cross_results
        if item.first_index == first_index and item.second_index == second_index
    )
    if row_index == cross_result.row_index and column_index == cross_result.column_index:
        return np.asarray(getattr(cross_result.twodcos, kind))
    return np.asarray(getattr(cross_result.twodcos, f"reverse_{kind}"))


def _create_block_overview_figure(
    result: MultiRangePipelineResult,
    config: PipelineConfig,
    *,
    kind: str,
) -> Any:
    """Recreate an overview figure if an older aggregate ZIP lacks its PNG."""

    column_indices = tuple(range(len(result.range_results)))
    row_indices = tuple(reversed(column_indices))
    row_axes = [result.range_results[index].result.twodcos.row_wavenumber for index in row_indices]
    column_axes = [
        result.range_results[index].result.twodcos.column_wavenumber for index in column_indices
    ]
    block_matrices = [
        [
            _block_matrix(
                result,
                row_index=row_index,
                column_index=column_index,
                kind=kind,
            )
            for column_index in column_indices
        ]
        for row_index in row_indices
    ]
    same_range_blocks = {
        (row_position, column_position)
        for row_position, row_index in enumerate(row_indices)
        for column_position, column_index in enumerate(column_indices)
        if row_index == column_index
    }
    return create_multi_range_2d_contour(
        row_axes,
        column_axes,
        block_matrices,
        kind=kind,
        row_labels=[
            result.range_results[index].analysis_range.display_name for index in row_indices
        ],
        column_labels=[
            result.range_results[index].analysis_range.display_name for index in column_indices
        ],
        convention=config.convention,
        method="independently preprocessed auto- and same-series cross-range blocks",
        contour_levels=config.contour_levels,
        display_percentile=config.display_percentile,
        filled=True,
        diagonal_blocks=same_range_blocks,
    )


def _render_block_figure(cached: CachedRun, config: PipelineConfig, *, kind: str) -> None:
    member_name = f"figures/multi_range_{kind}_blocks.png"
    figure_bytes = bundle_member_bytes(cached.bundle_bytes, member_name)
    if figure_bytes is None:
        st.info(
            f"The aggregate export does not contain {member_name}; rendering the same "
            "overview from the cached matrices."
        )
        _show_figure(_create_block_overview_figure(cached.result, config, kind=kind))
    else:
        st.image(
            figure_bytes,
            caption=(
                f"{kind.capitalize()} full block map. Same-range self-correlation panels "
                "lie on the anti-diagonal in this literature-style layout; all remaining "
                f"panels are same-series cross-range blocks. Convention: {config.convention}."
            ),
            width="stretch",
        )
    st.caption(
        "Rows are arranged in reverse range order from top to bottom; columns follow the "
        "range-editor order from left to right. Wavenumber axes within every panel are "
        "displayed high → low cm⁻¹. All panels in this figure share one symmetric "
        "color scale."
    )


def _render_block_overview(
    cached: CachedRun,
    config: PipelineConfig,
    fingerprint: str,
) -> None:
    """Render the pipeline-exported complete auto/cross block figures."""

    st.subheader("Full block overview")
    st.caption(
        "Same-range square self-correlation panels appear on the anti-diagonal because "
        "row ranges are reversed. All remaining rectangular panels contain cross-range "
        "correlations from the same FTIR sequence."
    )
    tabs = st.tabs(
        ("Synchronous overview", "Asynchronous overview", "Cross QC", "Files"),
        key=f"block_tabs_{fingerprint}",
        on_change="rerun",
    )
    if tabs[0].open:
        with tabs[0]:
            _render_block_figure(cached, config, kind="synchronous")
    if tabs[1].open:
        with tabs[1]:
            _render_block_figure(cached, config, kind="asynchronous")
    if tabs[2].open:
        with tabs[2]:
            cross_results = cached.result.cross_results
            st.dataframe(
                pd.DataFrame(
                    {
                        "Pair": [item.pair_label for item in cross_results],
                        "Row range": [item.row_range.display_name for item in cross_results],
                        "Column range": [item.column_range.display_name for item in cross_results],
                        "Convention": [item.twodcos.convention for item in cross_results],
                        "Cross QC passed": [
                            bool(item.qc_metrics.get("all_checks_passed", False))
                            for item in cross_results
                        ],
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Select Cross-range pair above for the complete numerical QC dictionary "
                "and explicit row/column axis metadata for any pair."
            )
    if tabs[3].open:
        with tabs[3]:
            _render_files(cached, fingerprint, context="overview")


def _peak_intensity_frame(
    order: PeakOrderResult,
    result: MultiRangePipelineResult,
) -> pd.DataFrame:
    """Extract matched processed intensities in analysis-index order for review."""

    first_processed = result.range_results[0].result.processed
    frame = pd.DataFrame(
        {
            "Analysis index": np.arange(1, first_processed.n_spectra + 1, dtype=np.int64),
            "Perturbation": first_processed.perturbation,
        }
    )
    for peak_index, peak in enumerate(order.peaks, start=1):
        if peak.range_index is None:
            continue
        processed = result.range_results[peak.range_index].result.processed
        grid_index = int(np.argmin(np.abs(processed.wavenumber - peak.wavenumber)))
        matched = float(processed.wavenumber[grid_index])
        frame[f"P{peak_index}: {_peak_label(peak)}; matched {matched:g}"] = processed.spectra[
            :, grid_index
        ]
    return frame


def _render_peak_order_result(
    order: PeakOrderResult,
    result: MultiRangePipelineResult,
    *,
    fingerprint: str,
    result_key: str,
) -> None:
    """Render one auditable response-order result and portable downloads."""

    with st.container(border=True):
        st.markdown("**Apparent response chain along the current perturbation order**")
        st.code(peak_order_chain_text(order), language=None, wrap_lines=True)
        metrics = st.columns(4)
        metrics[0].metric("Matched peaks", len(order.peaks))
        metrics[1].metric("Pairwise decisions", len(order.evidence))
        metrics[2].metric("All pairs resolved", "Yes" if order.all_pairs_resolved else "No")
        metrics[3].metric(
            "Unique total order",
            "Yes" if order.is_unique_total_order else "No",
        )

    if order.has_cycles:
        cycle_text = "; ".join(
            " ↻ ".join(_peak_label(peak) for peak in group) for group in order.cyclic_groups
        )
        st.error(
            "The resolved pairwise signs contain a cycle, so no linear order is asserted. "
            f"Cyclic group(s): {cycle_text}"
        )
    elif not order.is_unique_total_order:
        st.warning(
            "The evidence supports a partial order only. Peaks joined by `∥` share an "
            "unresolved graph layer; this does not mean that they respond simultaneously."
        )
    else:
        st.success("Every requested peak is placed in one unique graph-supported total order.")

    st.info(order.analysis_order_note)
    for warning in order.warnings:
        if str(warning).strip() != order.analysis_order_note.strip():
            st.warning(warning)
    st.caption(f"Noda decision contract: {order.rule_description}")
    st.warning(
        "Interpret this as an apparent/integrated response order, not a reaction mechanism, "
        "local kinetic truth, or statistical-confidence ranking. Opposite-direction or "
        "non-monotonic intensity changes, rate differences, nonlinearity, peak shifts, "
        "near-zero signals, unresolved layers, and cycles require manual review."
    )

    evidence_frame = peak_evidence_frame(order, result)
    st.subheader("Pairwise Noda evidence")
    st.caption(
        "Relative signal strength is min(|Φᵢⱼ| / max|Φ block|, "
        "|Ψᵢⱼ| / max|Ψ block|). It is a scale diagnostic in [0, 1], "
        "not a probability, error bar, or statistical confidence."
    )
    st.dataframe(
        evidence_frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Relative signal strength": st.column_config.NumberColumn(format="%.4g"),
            "Minimum signal/cutoff ratio": st.column_config.NumberColumn(format="%.4g"),
            "Synchronous Φ": st.column_config.NumberColumn(format="%.8g"),
            "Asynchronous Ψ": st.column_config.NumberColumn(format="%.8g"),
            "Phi * Psi": st.column_config.NumberColumn(format="%.8g"),
        },
    )

    with st.expander("Inspect matched processed-intensity traces"):
        trace_frame = _peak_intensity_frame(order, result)
        intensity_columns = list(trace_frame.columns[2:])
        if intensity_columns:
            st.line_chart(trace_frame, x="Analysis index", y=intensity_columns)
        st.dataframe(trace_frame, hide_index=True, width="stretch")
        st.caption(
            "The x-axis preserves analyzed spectrum order. The Perturbation column shows "
            "the stored values separately so non-uniform or non-monotonic spacing remains visible."
        )

    json_bytes = (
        json.dumps(
            order.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    csv_bytes = evidence_frame.to_csv(index=False).encode("utf-8-sig")
    with st.container(horizontal=True):
        st.download_button(
            "Download order audit JSON",
            data=json_bytes,
            file_name=f"peak_order_{fingerprint[:12]}.json",
            mime="application/json",
            key=f"peak_order_json_{result_key}",
            on_click="ignore",
            icon=":material/download:",
        )
        st.download_button(
            "Download pairwise evidence CSV",
            data=csv_bytes,
            file_name=f"peak_order_evidence_{fingerprint[:12]}.csv",
            mime="text/csv",
            key=f"peak_order_csv_{result_key}",
            on_click="ignore",
            icon=":material/download:",
        )
    st.caption(
        "These post-hoc downloads are generated from the cached matrices and do not alter "
        "the run ZIP. Use repeated CLI --peak options when the peak-order files must be "
        "included in the aggregate export."
    )


def _render_peak_response_order(cached: CachedRun, fingerprint: str) -> None:
    """Collect peaks in a form and analyze only on explicit submission."""

    result = cached.result
    st.subheader("Peak response order")
    st.caption(
        "Enter measured peak positions after completing 2D-COS. Auto range matching is "
        "accepted only when exactly one analyzed range contains a grid point within tolerance; "
        "choose an explicit range when intervals overlap."
    )
    options = peak_range_options(result)
    with st.form(f"peak_order_form_{fingerprint}", border=True):
        edited = st.data_editor(
            peak_editor_frame(),
            key=f"peak_order_editor_{fingerprint}",
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            placeholder="Add at least two peak positions",
            column_config={
                PEAK_WAVENUMBER_COLUMN: st.column_config.NumberColumn(
                    "Wavenumber (cm⁻¹)",
                    help="Requested peak position; the nearest measured grid point is audited.",
                    format="%.8g",
                    required=True,
                ),
                PEAK_LABEL_COLUMN: st.column_config.TextColumn(
                    "Label (optional)",
                    help="Chemical assignments are user-supplied and are not inferred.",
                ),
                PEAK_RANGE_COLUMN: st.column_config.SelectboxColumn(
                    "Range (optional)",
                    options=options,
                    default=AUTO_PEAK_RANGE,
                    help="Ranges are one-based here; Auto rejects ambiguous overlaps.",
                ),
            },
        )
        threshold_columns = st.columns(4)
        tolerance = threshold_columns[0].number_input(
            "Peak match tolerance (cm⁻¹)",
            min_value=0.0,
            value=1.0,
            format="%.6g",
            key=f"peak_match_tolerance_{fingerprint}",
        )
        synchronous_threshold = threshold_columns[1].number_input(
            "Absolute |Φ| cutoff",
            min_value=0.0,
            value=0.0,
            format="%.6g",
            key=f"peak_sync_threshold_{fingerprint}",
            help="Numerical interpretation cutoff, not confidence.",
        )
        asynchronous_threshold = threshold_columns[2].number_input(
            "Absolute |Ψ| cutoff",
            min_value=0.0,
            value=0.0,
            format="%.6g",
            key=f"peak_async_threshold_{fingerprint}",
            help="Numerical interpretation cutoff, not confidence.",
        )
        relative_threshold = threshold_columns[3].number_input(
            "Relative block cutoff",
            min_value=0.0,
            max_value=1.0,
            value=1.0e-6,
            format="%.3e",
            key=f"peak_relative_threshold_{fingerprint}",
            help="Fraction of each source block's maximum absolute signal; not confidence.",
        )
        submitted = st.form_submit_button(
            "Analyze apparent response order",
            type="primary",
            icon=":material/account_tree:",
        )

    cache: dict[str, PeakOrderResult] = st.session_state["peak_order_cache"]
    active_state_key = f"active_peak_order_{fingerprint}"
    if submitted:
        try:
            peaks = validate_peak_requests(edited, result)
            result_key = _peak_order_cache_key(
                fingerprint,
                peaks,
                tolerance=float(tolerance),
                synchronous_threshold=float(synchronous_threshold),
                asynchronous_threshold=float(asynchronous_threshold),
                relative_threshold=float(relative_threshold),
            )
            order = analyze_multi_range_peak_order(
                result,
                peaks,
                peak_match_tolerance=float(tolerance),
                synchronous_threshold=float(synchronous_threshold),
                asynchronous_threshold=float(asynchronous_threshold),
                relative_threshold=float(relative_threshold),
            )
        except (TypeError, ValueError) as exc:
            st.error(f"Peak response order could not be analyzed: {exc}")
        else:
            _remember_bounded(cache, result_key, order, limit=12)
            st.session_state[active_state_key] = result_key

    active_result_key = st.session_state.get(active_state_key)
    active_order = cache.get(active_result_key) if active_result_key else None
    if active_order is None and result.peak_order is not None:
        active_order = result.peak_order
        active_result_key = f"pipeline_{fingerprint}"
    if active_order is None:
        st.info(
            "Add peak rows and submit the form. Editing this table does not rerun "
            "preprocessing or 2D-COS."
        )
        return
    _render_peak_order_result(
        active_order,
        result,
        fingerprint=fingerprint,
        result_key=str(active_result_key),
    )


def _render_results(cached: CachedRun, config: PipelineConfig, fingerprint: str) -> None:
    st.header("6. Results")
    multi_result = cached.result
    cross_results = tuple(getattr(multi_result, "cross_results", ()))
    cross_count = len(cross_results)
    st.caption(
        f"Completed {len(multi_result.range_results)} independent range analysis(es) and "
        f"{cross_count} unique same-series cross-range pair(s)."
    )
    overall_qc = getattr(multi_result, "all_checks_passed", None)
    if overall_qc is None:
        overall_qc = all(
            bool(item.result.qc_metrics.get("all_checks_passed", False))
            for item in multi_result.range_results
        )
    if overall_qc:
        st.success("All independent-range and cross-range numerical QC checks passed.")
    else:
        st.warning("At least one independent-range or cross-range numerical QC check failed.")

    supports_peak_order = isinstance(multi_result, MultiRangePipelineResult)
    if cross_count:
        view_options = (
            "Independent ranges",
            "Cross-range pair",
            "Full block overview",
            "Peak response order",
        )
    elif supports_peak_order:
        view_options = ("Independent ranges", "Peak response order")
    else:
        view_options = ()

    if view_options:
        view = st.segmented_control(
            "Result view",
            view_options,
            default="Independent ranges",
            required=True,
            key=f"result_view_{fingerprint}",
            persist_state="session",
            width="stretch",
        )
    else:
        view = "Independent ranges"

    if view == "Cross-range pair":
        _render_cross_pair_results(cached, config, fingerprint)
    elif view == "Full block overview":
        _render_block_overview(cached, config, fingerprint)
    elif view == "Peak response order":
        _render_peak_response_order(cached, fingerprint)
    else:
        _render_independent_results(cached, config, fingerprint)


def _render_run(
    dataset: SpectralDataset,
    ranges: tuple[WavenumberRange, ...],
    config: PipelineConfig,
    *,
    confirmed: bool,
) -> None:
    st.header("5. Run 2D-COS")
    fingerprint = config_fingerprint(dataset, config, ranges=ranges)
    cross_count = len(ranges) * (len(ranges) - 1) // 2
    run_cache: dict[str, CachedRun] = st.session_state["run_cache"]
    cached = run_cache.get(fingerprint)
    if cached is not None:
        st.success(
            "A cached result matches the current scientific configuration. Display controls "
            "can be changed without recalculating 2D-COS."
        )

    if st.button(
        f"Run 2D-COS for {len(ranges)} range(s)",
        type="primary",
        disabled=not confirmed,
        help=(
            None
            if confirmed
            else "Generate, inspect, and confirm the matching baseline preview first."
        ),
    ):
        if cached is None:
            progress = st.progress(
                10,
                text=(
                    f"Running {len(ranges)} independent self-correlation analysis(es) and "
                    f"{cross_count} same-series cross-range block(s)…"
                ),
            )
            try:
                cached = execute_pipeline_bundle(dataset, ranges, config)
                progress.progress(
                    100,
                    text="All self/cross calculations and aggregate ZIP export are complete.",
                )
                _remember_bounded(run_cache, fingerprint, cached, limit=3)
            except Exception as exc:
                progress.empty()
                st.error(f"2D-COS could not be completed: {exc}")
                return
        else:
            st.info(
                "Loaded the matching multi-range result from this session; no recalculation "
                "was performed."
            )

    cached = run_cache.get(fingerprint)
    if cached is not None:
        try:
            _render_results(cached, config, fingerprint)
        except Exception as exc:
            st.error(f"The result exists, but a result view could not be rendered: {exc}")
    elif not confirmed:
        st.info("Run remains locked until the current preview fingerprint is confirmed.")


def main() -> None:
    """Render the complete staged, single-page application."""

    st.set_page_config(
        page_title="FTIR preprocessing and 2D-COS",
        page_icon="📈",
        layout="wide",
    )
    _initialize_session_state()
    st.title("FTIR preprocessing and 2D-COS")
    st.caption(
        "Auditable local workflow: upload → edit perturbations → configure → inspect "
        "baseline → confirm → calculate → export. Raw uploads are never overwritten."
    )

    dataset = _render_upload()
    if dataset is None:
        return
    perturbation_settings = _render_perturbations(dataset)
    if perturbation_settings is None:
        return
    edited_dataset, perturbation_order = perturbation_settings
    preprocessing_settings = _render_preprocessing(edited_dataset, perturbation_order)
    if preprocessing_settings is None:
        return
    config, ranges = preprocessing_settings
    confirmed = _render_preview(edited_dataset, ranges, config)
    _render_run(edited_dataset, ranges, config, confirmed=confirmed)


if __name__ == "__main__":
    main()
