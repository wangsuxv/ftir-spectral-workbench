"""Reproducible CSV, metadata, figure and ZIP exports for FTIR 2D-COS runs."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .plotting import (
    DEFAULT_CONTOUR_LEVELS,
    DEFAULT_DISPLAY_PERCENTILE,
    plot_all_baselines_overlay,
    plot_asynchronous_contour,
    plot_baseline_qc_representative,
    plot_corrected_spectra_overlay,
    plot_dynamic_spectra_overlay,
    plot_raw_spectra_overlay,
    plot_synchronous_contour,
)

_NO_DEFAULT = object()
_MISSING = object()
_PACKAGE_NAMES = ("numpy", "pandas", "scipy", "pybaselines", "matplotlib", "streamlit")


def _get_field(value: object, name: str, default: object = _NO_DEFAULT) -> Any:
    if isinstance(value, Mapping) and name in value:
        return value[name]
    if hasattr(value, name):
        return getattr(value, name)
    if default is not _NO_DEFAULT:
        return default
    raise AttributeError(f"required field {name!r} is missing from {type(value).__name__}")


def _first_field(value: object, names: Sequence[str], default: object = _NO_DEFAULT) -> Any:
    for name in names:
        candidate = _get_field(value, name, _MISSING)
        if candidate is not _MISSING:
            return candidate
    if default is not _NO_DEFAULT:
        return default
    joined = ", ".join(repr(name) for name in names)
    raise AttributeError(f"none of the required fields ({joined}) are available")


def _json_ready(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("JSON metadata contains NaN or infinite values")
        return value
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, Enum):
        return _json_ready(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_ready(item) for item in value), key=str)
    return str(value)


def _write_json(path: Path, payload: object) -> None:
    serializable = _json_ready(payload)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            serializable, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        handle.write("\n")


def _format_cell(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("CSV data contains NaN or infinite values")
        return format(float(value), ".17g")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def write_spectra_csv(
    path: str | Path,
    wavenumber: np.ndarray | Sequence[float],
    spectra: np.ndarray | Sequence[Sequence[float]],
    labels: Sequence[str],
) -> Path:
    """Write an Origin-ready 1D wide table without changing axis order."""

    destination = Path(path)
    axis = np.asarray(wavenumber, dtype=np.float64)
    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if axis.ndim != 1:
        raise ValueError("wavenumber must be one-dimensional")
    if values.ndim != 2 or values.shape[1] != axis.size:
        raise ValueError(f"spectra must have shape (n_spectra, {axis.size}); got {values.shape}")
    normalized_labels = tuple(str(label) for label in labels)
    if len(normalized_labels) != values.shape[0]:
        raise ValueError(
            f"labels must contain {values.shape[0]} values; got {len(normalized_labels)}"
        )
    if not np.all(np.isfinite(axis)) or not np.all(np.isfinite(values)):
        raise ValueError("spectra CSV contains NaN or infinite values")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["wavenumber_cm-1", *normalized_labels])
        for point_index, axis_value in enumerate(axis):
            writer.writerow(
                [
                    _format_cell(axis_value),
                    *(_format_cell(value) for value in values[:, point_index]),
                ]
            )
    return destination


def write_matrix_csv(
    path: str | Path,
    matrix: np.ndarray | Sequence[Sequence[float]],
    row_axis: np.ndarray | Sequence[object],
    column_axis: np.ndarray | Sequence[object] | None = None,
    *,
    axis_name: str = "wavenumber_cm-1",
) -> Path:
    """Write an Origin-ready matrix with explicit row and column coordinates."""

    destination = Path(path)
    values = np.asarray(matrix, dtype=np.float64)
    rows = np.asarray(row_axis)
    columns = rows if column_axis is None else np.asarray(column_axis)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if rows.ndim != 1 or columns.ndim != 1:
        raise ValueError("matrix axes must be one-dimensional")
    if values.shape != (rows.size, columns.size):
        raise ValueError(
            "matrix shape must match row and column axes; "
            f"got {values.shape}, rows={rows.size}, columns={columns.size}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix CSV contains NaN or infinite values")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([axis_name, *(_format_cell(value) for value in columns)])
        for axis_value, row in zip(rows, values, strict=True):
            writer.writerow([_format_cell(axis_value), *(_format_cell(value) for value in row)])
    return destination


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for an input file without modifying it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def create_unique_run_directory(
    output_root: str | Path,
    *,
    now: datetime | None = None,
) -> Path:
    """Atomically create a timestamped run directory, adding a suffix on collision."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise NotADirectoryError(f"output root is not a directory: {root}")
    timestamp = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
    base_name = f"run_{timestamp}"
    for collision_index in range(10_000):
        suffix = "" if collision_index == 0 else f"_{collision_index:03d}"
        candidate = root / f"{base_name}{suffix}"
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"could not create a unique run directory beneath {root}")


def _dataset_parts(dataset: object) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    wavenumber = np.asarray(_get_field(dataset, "wavenumber"), dtype=np.float64)
    spectra = np.asarray(_get_field(dataset, "spectra"), dtype=np.float64)
    labels_value = _get_field(dataset, "perturbation_labels", None)
    if wavenumber.ndim != 1:
        raise ValueError("dataset wavenumber must be one-dimensional")
    if spectra.ndim != 2 or spectra.shape[1] != wavenumber.size:
        raise ValueError(
            "dataset spectra must follow (n_spectra, n_wavenumbers); "
            f"got {spectra.shape} for {wavenumber.size} wavenumbers"
        )
    if labels_value is None:
        labels = tuple(f"Spectrum {index + 1}" for index in range(spectra.shape[0]))
    else:
        labels = tuple(str(label) for label in labels_value)
    if len(labels) != spectra.shape[0]:
        raise ValueError("dataset perturbation_labels length does not match spectra")
    if not np.all(np.isfinite(wavenumber)) or not np.all(np.isfinite(spectra)):
        raise ValueError("dataset contains NaN or infinite values")
    return wavenumber, spectra, labels


def _axis_direction(axis: np.ndarray) -> str:
    differences = np.diff(np.asarray(axis, dtype=np.float64))
    if np.all(differences > 0):
        return "ascending"
    if np.all(differences < 0):
        return "descending"
    return "non_monotonic"


def _package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _warning_lines(warnings_value: object) -> list[str]:
    if warnings_value is None:
        return []
    if isinstance(warnings_value, str):
        return [warnings_value]
    if isinstance(warnings_value, Sequence):
        lines: list[str] = []
        for warning in warnings_value:
            if isinstance(warning, str):
                lines.append(warning)
            else:
                lines.append(json.dumps(_json_ready(warning), ensure_ascii=False, sort_keys=True))
        return lines
    return [str(warnings_value)]


def _nested_config(config_payload: Mapping[str, object], *names: str) -> Mapping[str, object]:
    for name in names:
        value = config_payload.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


def _figure_settings(config_payload: Mapping[str, object]) -> dict[str, object]:
    plotting = _nested_config(config_payload, "plotting", "plot", "figure")
    levels = plotting.get(
        "contour_levels", config_payload.get("contour_levels", DEFAULT_CONTOUR_LEVELS)
    )
    percentile = plotting.get(
        "display_percentile",
        config_payload.get("display_percentile", DEFAULT_DISPLAY_PERCENTILE),
    )
    filled = plotting.get("filled_contour", plotting.get("filled", True))
    return {
        "contour_levels": int(levels),
        "display_percentile": None if percentile is None else float(percentile),
        "filled": bool(filled),
    }


def _baseline_method(config_payload: Mapping[str, object]) -> str:
    baseline = _nested_config(config_payload, "baseline", "baseline_config")
    method = baseline.get("method", config_payload.get("baseline_method", "unspecified baseline"))
    if isinstance(method, Enum):
        method = method.value
    return str(method)


def _expected_relative_paths() -> list[str]:
    metadata_files = ["manifest.json", "config.json", "warnings.txt", "qc_metrics.json"]
    data_files = [
        "data/01_imported.csv",
        "data/02_selected_raw.csv",
        "data/03_baseline.csv",
        "data/04_baseline_corrected.csv",
        "data/05_processed.csv",
        "data/06_reference_spectrum.csv",
        "data/07_dynamic_spectra.csv",
        "data/08_hilbert_noda_matrix.csv",
        "data/09_synchronous_matrix.csv",
        "data/10_asynchronous_matrix.csv",
    ]
    figure_stems = [
        "raw_spectra",
        "baseline_qc",
        "all_baselines",
        "corrected_spectra",
        "dynamic_spectra",
        "synchronous_2dcos",
        "asynchronous_2dcos",
    ]
    figure_files = [
        f"figures/{stem}.{extension}" for stem in figure_stems for extension in ("png", "pdf")
    ]
    return [*metadata_files, *data_files, *figure_files, "run_bundle.zip"]


def _normalize_input_paths(input_paths: Sequence[str | Path] | str | Path) -> tuple[Path, ...]:
    if isinstance(input_paths, (str, Path)):
        return (Path(input_paths),)
    return tuple(Path(path) for path in input_paths)


def _create_bundle(run_directory: Path, relative_paths: Sequence[str]) -> Path:
    bundle_path = run_directory / "run_bundle.zip"
    with zipfile.ZipFile(
        bundle_path,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for relative_path in sorted(relative_paths):
            if relative_path == "run_bundle.zip":
                continue
            source = run_directory / relative_path
            if not source.is_file():
                raise FileNotFoundError(f"expected export file is missing: {relative_path}")
            archive.write(source, arcname=relative_path)
    return bundle_path


def export_run(
    result: object,
    config: object,
    output_root: str | Path,
    input_paths: Sequence[str | Path] | str | Path = (),
) -> Path:
    """Export one complete analysis run and return its unique directory.

    ``result`` and ``config`` intentionally use duck typing.  This keeps the export
    layer usable with frozen dataclasses as well as mappings while preserving the
    project's scientific model contract.
    """

    started_at = datetime.now().astimezone()
    run_directory = create_unique_run_directory(output_root, now=started_at)
    data_directory = run_directory / "data"
    figure_directory = run_directory / "figures"
    data_directory.mkdir()
    figure_directory.mkdir()

    imported = _get_field(result, "imported")
    selected_raw = _get_field(result, "selected_raw")
    baseline_corrected = _get_field(result, "baseline_corrected")
    processed = _get_field(result, "processed")
    baselines = np.asarray(
        _first_field(result, ("baselines", "baseline", "baseline_matrix")),
        dtype=np.float64,
    )

    imported_axis, imported_spectra, imported_labels = _dataset_parts(imported)
    selected_axis, selected_spectra, selected_labels = _dataset_parts(selected_raw)
    corrected_axis, corrected_spectra, corrected_labels = _dataset_parts(baseline_corrected)
    processed_axis, processed_spectra, processed_labels = _dataset_parts(processed)
    if not np.array_equal(selected_axis, corrected_axis) or not np.array_equal(
        selected_axis, processed_axis
    ):
        raise ValueError("selected, baseline-corrected and processed wavenumber axes differ")
    if selected_labels != corrected_labels or selected_labels != processed_labels:
        raise ValueError("selected, baseline-corrected and processed perturbation labels differ")
    if selected_spectra.shape != baselines.shape:
        raise ValueError(
            f"baselines must have shape {selected_spectra.shape}; got {baselines.shape}"
        )
    if not np.all(np.isfinite(baselines)):
        raise ValueError("baselines contain NaN or infinite values")

    analysis = _first_field(result, ("twodcos", "correlation", "analysis"), default=result)
    reference = np.asarray(_get_field(analysis, "reference"), dtype=np.float64)
    dynamic = np.asarray(_get_field(analysis, "dynamic"), dtype=np.float64)
    noda = np.asarray(_get_field(analysis, "noda"), dtype=np.float64)
    synchronous = np.asarray(_get_field(analysis, "synchronous"), dtype=np.float64)
    asynchronous = np.asarray(_get_field(analysis, "asynchronous"), dtype=np.float64)
    qc_metrics_value = _get_field(result, "qc_metrics", _MISSING)
    if qc_metrics_value is _MISSING:
        qc_metrics_value = _get_field(analysis, "qc_metrics", {})
    warnings_value = _get_field(result, "warnings", ())
    convention = str(
        _get_field(analysis, "convention", _get_field(result, "convention", "canonical"))
    )

    if reference.shape != (processed_axis.size,):
        raise ValueError(
            f"reference must have shape ({processed_axis.size},); got {reference.shape}"
        )
    if dynamic.shape != processed_spectra.shape:
        raise ValueError(f"dynamic must have shape {processed_spectra.shape}; got {dynamic.shape}")
    expected_correlation_shape = (processed_axis.size, processed_axis.size)
    if (
        synchronous.shape != expected_correlation_shape
        or asynchronous.shape != expected_correlation_shape
    ):
        raise ValueError(
            "synchronous and asynchronous matrices must both have shape "
            f"{expected_correlation_shape}"
        )

    perturbation = np.asarray(
        _get_field(processed, "perturbation", np.arange(processed_spectra.shape[0])),
        dtype=np.float64,
    )
    if perturbation.shape != (processed_spectra.shape[0],):
        raise ValueError("processed perturbation axis length does not match spectrum count")
    if noda.shape != (perturbation.size, perturbation.size):
        raise ValueError(
            f"noda must have shape ({perturbation.size}, {perturbation.size}); got {noda.shape}"
        )

    row_axis = np.asarray(_get_field(analysis, "row_wavenumber", processed_axis), dtype=np.float64)
    column_axis = np.asarray(
        _get_field(analysis, "column_wavenumber", processed_axis), dtype=np.float64
    )
    row_variable = str(_get_field(analysis, "row_variable", "nu1"))
    column_variable = str(_get_field(analysis, "column_variable", "nu2"))

    write_spectra_csv(
        data_directory / "01_imported.csv", imported_axis, imported_spectra, imported_labels
    )
    write_spectra_csv(
        data_directory / "02_selected_raw.csv",
        selected_axis,
        selected_spectra,
        selected_labels,
    )
    write_spectra_csv(data_directory / "03_baseline.csv", selected_axis, baselines, selected_labels)
    write_spectra_csv(
        data_directory / "04_baseline_corrected.csv",
        corrected_axis,
        corrected_spectra,
        corrected_labels,
    )
    write_spectra_csv(
        data_directory / "05_processed.csv",
        processed_axis,
        processed_spectra,
        processed_labels,
    )
    write_spectra_csv(
        data_directory / "06_reference_spectrum.csv",
        processed_axis,
        reference,
        ("mean_reference",),
    )
    write_spectra_csv(
        data_directory / "07_dynamic_spectra.csv",
        processed_axis,
        dynamic,
        processed_labels,
    )
    write_matrix_csv(
        data_directory / "08_hilbert_noda_matrix.csv",
        noda,
        perturbation,
        axis_name="perturbation",
    )
    write_matrix_csv(
        data_directory / "09_synchronous_matrix.csv",
        synchronous,
        row_axis,
        column_axis,
    )
    write_matrix_csv(
        data_directory / "10_asynchronous_matrix.csv",
        asynchronous,
        row_axis,
        column_axis,
    )

    config_payload_value = _json_ready(config)
    if not isinstance(config_payload_value, Mapping):
        config_payload: Mapping[str, object] = {"value": config_payload_value}
    else:
        config_payload = config_payload_value
    plot_settings = _figure_settings(config_payload)
    method = _baseline_method(config_payload)
    intensity_unit = str(_get_field(processed, "intensity_unit", "Intensity"))

    plot_raw_spectra_overlay(
        selected_axis,
        selected_spectra,
        figure_directory / "raw_spectra",
        labels=selected_labels,
        intensity_label=intensity_unit,
    )
    plot_baseline_qc_representative(
        selected_axis,
        selected_spectra,
        baselines,
        corrected_spectra,
        figure_directory / "baseline_qc",
        labels=selected_labels,
        intensity_label=intensity_unit,
    )
    plot_all_baselines_overlay(
        selected_axis,
        baselines,
        figure_directory / "all_baselines",
        labels=selected_labels,
        intensity_label=intensity_unit,
    )
    plot_corrected_spectra_overlay(
        corrected_axis,
        corrected_spectra,
        figure_directory / "corrected_spectra",
        labels=corrected_labels,
        intensity_label=intensity_unit,
    )
    plot_dynamic_spectra_overlay(
        processed_axis,
        dynamic,
        figure_directory / "dynamic_spectra",
        labels=processed_labels,
    )
    plot_synchronous_contour(
        row_axis,
        synchronous,
        figure_directory / "synchronous_2dcos",
        column_wavenumber=column_axis,
        convention=convention,
        row_variable=row_variable,
        column_variable=column_variable,
        method=method,
        **plot_settings,
    )
    plot_asynchronous_contour(
        row_axis,
        asynchronous,
        figure_directory / "asynchronous_2dcos",
        column_wavenumber=column_axis,
        convention=convention,
        row_variable=row_variable,
        column_variable=column_variable,
        method=method,
        **plot_settings,
    )

    warning_lines = _warning_lines(warnings_value)
    with (run_directory / "warnings.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for warning in warning_lines:
            handle.write(f"{warning}\n")
    _write_json(run_directory / "config.json", config_payload)
    _write_json(run_directory / "qc_metrics.json", qc_metrics_value)

    normalized_inputs = _normalize_input_paths(input_paths)
    input_records = [
        {
            "file_name": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in normalized_inputs
    ]
    metadata = _get_field(imported, "metadata", {})
    metadata_mapping = metadata if isinstance(metadata, Mapping) else {}
    processed_metadata = _get_field(processed, "metadata", {})
    processed_metadata_mapping = (
        processed_metadata if isinstance(processed_metadata, Mapping) else {}
    )
    source_name = str(_get_field(imported, "source_name", ""))
    if not input_records:
        metadata_sources = metadata_mapping.get("source_files", ())
        if isinstance(metadata_sources, Sequence) and not isinstance(
            metadata_sources, (str, bytes)
        ):
            for source in metadata_sources:
                if not isinstance(source, Mapping):
                    continue
                input_records.append(
                    {
                        "file_name": str(source.get("name", source.get("file_name", "unknown"))),
                        "sha256": source.get("sha256"),
                        "size_bytes": source.get("size_bytes"),
                    }
                )
        source_hashes = metadata_mapping.get("source_sha256")
        if not input_records and isinstance(source_hashes, Mapping):
            input_records.extend(
                {
                    "file_name": str(file_name),
                    "sha256": digest,
                    "size_bytes": None,
                }
                for file_name, digest in source_hashes.items()
            )
        elif not input_records and isinstance(source_hashes, str):
            input_records.append(
                {"file_name": source_name, "sha256": source_hashes, "size_bytes": None}
            )
        elif not input_records and source_name:
            input_records.append(
                {
                    "file_name": source_name,
                    "sha256": metadata_mapping.get("sha256"),
                    "size_bytes": metadata_mapping.get("size_bytes"),
                }
            )

    expected_paths = _expected_relative_paths()
    convention_metadata = _first_field(
        analysis,
        ("convention_metadata", "metadata", "formula"),
        default={},
    )
    perturbation_intervals = np.diff(perturbation).astype(float)
    perturbation_equally_spaced: bool | None = None
    nonuniform_warning: str | None = None
    if perturbation_intervals.size:
        absolute_intervals = np.abs(perturbation_intervals)
        reference_interval = float(np.median(absolute_intervals))
        interval_atol = max(reference_interval * 1e-12, np.finfo(np.float64).eps)
        perturbation_equally_spaced = bool(
            np.allclose(
                absolute_intervals,
                reference_interval,
                rtol=1e-5,
                atol=interval_atol,
            )
        )
        if not perturbation_equally_spaced:
            nonuniform_warning = (
                "Final processing-order perturbation intervals are non-uniform: "
                f"{perturbation_intervals.tolist()}. The Hilbert-Noda matrix was "
                "constructed from acquisition indices without time-weighting, interpolation, "
                "or resampling."
            )
    manifest = {
        "tool": "ftir2dcos",
        "tool_version": __version__,
        "git_commit": _git_commit(),
        "run_time": started_at.isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependency_versions": {name: _package_version(name) for name in _PACKAGE_NAMES},
        "input_file_name": input_records[0]["file_name"] if input_records else source_name,
        "input_sha256": input_records[0]["sha256"] if input_records else None,
        "input_files": input_records,
        "original_data_shape": metadata_mapping.get(
            "original_data_shape", list(imported_spectra.shape)
        ),
        "original_internal_shape": list(imported_spectra.shape),
        "final_data_shape": list(processed_spectra.shape),
        "wavenumber_range_cm-1": {
            "lower": float(np.min(processed_axis)),
            "upper": float(np.max(processed_axis)),
        },
        "original_wavenumber_direction": metadata_mapping.get(
            "original_wavenumber_direction", _axis_direction(imported_axis)
        ),
        "final_wavenumber_direction": _axis_direction(processed_axis),
        "perturbation_original_order": metadata_mapping.get(
            "perturbation_original_labels", list(imported_labels)
        ),
        "perturbation_final_order": processed_metadata_mapping.get(
            "perturbation_final_labels", list(processed_labels)
        ),
        "perturbation_order_changed": bool(
            processed_metadata_mapping.get(
                "perturbation_order_changed", imported_labels != processed_labels
            )
        ),
        "perturbation_original_values": np.asarray(
            _get_field(imported, "perturbation", ()), dtype=np.float64
        ).tolist(),
        "perturbation_final_values": perturbation.tolist(),
        "perturbation_final_intervals": perturbation_intervals.tolist(),
        "perturbation_approximately_equally_spaced": perturbation_equally_spaced,
        "perturbation_grid_strategy": "index_order",
        "baseline": _nested_config(config_payload, "baseline", "baseline_config"),
        "smoothing": _nested_config(config_payload, "smoothing", "smoothing_config"),
        "normalization": _nested_config(config_payload, "normalization"),
        "dynamic_reference_method": "mean_spectrum",
        "convention": convention,
        "matrix_axes": {
            "row_variable": row_variable,
            "column_variable": column_variable,
            "row_direction": _axis_direction(row_axis),
            "column_direction": _axis_direction(column_axis),
        },
        "convention_metadata": convention_metadata,
        "nonuniform_perturbation_warning": nonuniform_warning,
        "warnings": warning_lines,
        "qc_metrics": qc_metrics_value,
        "plot_display": plot_settings,
        "output_files": expected_paths,
        "outputs": expected_paths,
    }
    _write_json(run_directory / "manifest.json", manifest)
    _create_bundle(run_directory, expected_paths)
    return run_directory


__all__ = [
    "create_unique_run_directory",
    "export_run",
    "sha256_file",
    "write_matrix_csv",
    "write_spectra_csv",
]
