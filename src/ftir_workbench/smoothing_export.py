"""Deterministic export and verification for post-baseline smoothing runs.

The smoothing bundle is a self-contained scientific branch artifact.  It keeps
both the parent corrected absorbance and the committed smoothed Prepared child,
but never embeds the original raw spectrum or a baseline/project bundle.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from matplotlib.figure import Figure

from .adapters import prepared_from_smoothed_result
from .export import (
    _build_manifest_archive,
    _format_float,
    _json_bytes,
    _media_type,
    _prepared_from_archive_members,
    _read_bytes,
    _sha256,
    _wide_spectra_csv,
    serialize_prepared,
    verify_workbench_manifest,
)
from .models import PreparedSpectralDataset
from .post_baseline_smoothing import (
    DERIVATIVE_CORRELATION_WARNING_THRESHOLD,
    EDGE_EFFECT_RATIO_WARNING_THRESHOLD,
    RELATIVE_ABSOLUTE_AREA_CHANGE_WARNING_THRESHOLD,
    RELATIVE_RMS_WARNING_THRESHOLD,
    PostBaselineSmoothingConfig,
    PostBaselineSmoothingResult,
    apply_post_baseline_smoothing,
)

SMOOTHING_ARTIFACT_TYPE = "post_baseline_smoothing_run"
SMOOTHING_SCHEMA_VERSION = "1.0"

SOURCE_CSV = "source_corrected_absorbance.csv"
SOURCE_METADATA = "source_prepared_spectrum.meta.json"
CHILD_CSV = "smoothed_corrected_absorbance.csv"
CHILD_METADATA = "prepared_spectrum.meta.json"
REMOVED_CSV = "smoothing_removed_component.csv"
CONFIG_JSON = "smoothing_config.json"
METRICS_JSON = "smoothing_metrics.json"
METRICS_CSV = "smoothing_metrics.csv"
OVERLAY_PNG = "figures/selected_spectrum_overlay.png"
RESIDUAL_PNG = "figures/selected_spectrum_residual.png"
FIGURES_DIRECTORY = "figures/"
MANIFEST_JSON = "manifest.json"

SMOOTHING_PAYLOAD_MEMBERS = frozenset(
    {
        SOURCE_CSV,
        SOURCE_METADATA,
        CHILD_CSV,
        CHILD_METADATA,
        REMOVED_CSV,
        CONFIG_JSON,
        METRICS_JSON,
        METRICS_CSV,
        OVERLAY_PNG,
        RESIDUAL_PNG,
    }
)
SMOOTHING_BUNDLE_MEMBERS = frozenset(
    {*SMOOTHING_PAYLOAD_MEMBERS, FIGURES_DIRECTORY, MANIFEST_JSON}
)

SmoothingBundleSource = (
    bytes | bytearray | memoryview | str | Path | BinaryIO
)


def _bundle_payload(source: SmoothingBundleSource) -> bytes:
    payload, _, _ = _read_bytes(source, default_name="post_baseline_smoothing_run.zip")
    return payload


def _prepared_metadata_equal(
    first: PreparedSpectralDataset,
    second: PreparedSpectralDataset,
) -> bool:
    return _json_bytes(first.to_metadata_dict()) == _json_bytes(
        second.to_metadata_dict()
    )


def _assert_prepared_exact(
    actual: PreparedSpectralDataset,
    expected: PreparedSpectralDataset,
    *,
    context: str,
) -> None:
    if not np.array_equal(actual.wavenumber, expected.wavenumber):
        raise ValueError(f"{context} wavenumber differs")
    if not np.array_equal(actual.perturbation, expected.perturbation):
        raise ValueError(f"{context} perturbation differs")
    if actual.perturbation_labels != expected.perturbation_labels:
        raise ValueError(f"{context} perturbation labels differ")
    if not np.array_equal(actual.spectra, expected.spectra):
        raise ValueError(f"{context} spectra differ")
    if not _prepared_metadata_equal(actual, expected):
        raise ValueError(f"{context} metadata differs")


def _selected_spectrum_index(prepared: PreparedSpectralDataset) -> int:
    """Choose one deterministic actual spectrum for the required bundle figures."""

    return prepared.n_spectra // 2


def _figure_png(
    result: PostBaselineSmoothingResult,
    *,
    spectrum_index: int,
    residual: bool,
) -> bytes:
    parent = result.parent_prepared
    figure = Figure(figsize=(8.4, 4.8), constrained_layout=True)
    axes = figure.subplots()
    if residual:
        axes.plot(
            parent.wavenumber,
            result.removed_component[spectrum_index],
            label="Removed component",
            linewidth=1.35,
        )
        axes.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=0.9,
            label="Zero reference",
        )
        axes.set_title("Post-baseline smoothing removed component")
        axes.set_ylabel("Absorbance difference")
    else:
        axes.plot(
            parent.wavenumber,
            parent.spectra[spectrum_index],
            label="Unsmoothed corrected absorbance",
            linewidth=1.5,
        )
        axes.plot(
            parent.wavenumber,
            result.smoothed_spectra[spectrum_index],
            label="Smoothed corrected absorbance",
            linewidth=1.5,
        )
        axes.set_title("Unsmoothed vs smoothed corrected absorbance")
        axes.set_ylabel("Absorbance")
    axes.set_xlabel("Wavenumber (cm⁻¹)")
    axes.grid(True, alpha=0.2)
    axes.invert_xaxis()
    axes.legend()
    output = io.BytesIO()
    figure.savefig(
        output,
        format="png",
        dpi=150,
        metadata={"Software": "FTIR Spectral Workbench"},
    )
    return output.getvalue()


def _config_payload(result: PostBaselineSmoothingResult) -> dict[str, Any]:
    return result.config.to_dict()


def _metrics_payload(
    result: PostBaselineSmoothingResult,
    child: PreparedSpectralDataset,
    *,
    selected_spectrum_index: int,
) -> dict[str, Any]:
    parent = result.parent_prepared
    return {
        "schema_version": SMOOTHING_SCHEMA_VERSION,
        "parent_prepared_data_sha256": parent.prepared_data_sha256,
        "child_prepared_data_sha256": child.prepared_data_sha256,
        "smoothing_fingerprint": result.smoothing_fingerprint,
        "selected_spectrum_index": selected_spectrum_index,
        "axis_diagnostics": {
            "median_wavenumber_spacing": result.median_wavenumber_spacing,
            "spacing_relative_max_deviation": result.spacing_relative_max_deviation,
            "approximate_physical_width": dict(result.approximate_physical_width),
        },
        "diagnostic_thresholds": {
            "relative_rms_removed_warning_threshold": (
                RELATIVE_RMS_WARNING_THRESHOLD
            ),
            "first_derivative_correlation_warning_threshold": (
                DERIVATIVE_CORRELATION_WARNING_THRESHOLD
            ),
            "relative_absolute_area_change_warning_threshold": (
                RELATIVE_ABSOLUTE_AREA_CHANGE_WARNING_THRESHOLD
            ),
            "edge_effect_ratio_warning_threshold": (
                EDGE_EFFECT_RATIO_WARNING_THRESHOLD
            ),
        },
        "summary_metrics": dict(result.summary_metrics),
        "per_spectrum_metrics": {
            name: np.asarray(values, dtype=np.float64).tolist()
            for name, values in sorted(result.per_spectrum_metrics.items())
        },
        "warnings": list(result.warnings),
    }


def _metrics_csv(result: PostBaselineSmoothingResult) -> bytes:
    parent = result.parent_prepared
    metric_names = tuple(sorted(result.per_spectrum_metrics))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "spectrum_index",
            "perturbation",
            "perturbation_label",
            *metric_names,
        ]
    )
    for index, (perturbation, label) in enumerate(
        zip(parent.perturbation, parent.perturbation_labels, strict=True)
    ):
        writer.writerow(
            [
                index,
                _format_float(perturbation),
                label,
                *(
                    _format_float(result.per_spectrum_metrics[name][index])
                    for name in metric_names
                ),
            ]
        )
    return stream.getvalue().encode("utf-8")


def _manifest_base(
    result: PostBaselineSmoothingResult,
    child: PreparedSpectralDataset,
    *,
    selected_spectrum_index: int,
) -> dict[str, Any]:
    parent = result.parent_prepared
    scientific_config = result.config.scientific_dict()
    parameters = scientific_config.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("enabled smoothing config must contain effective parameters")
    return {
        "schema_version": SMOOTHING_SCHEMA_VERSION,
        "artifact_type": SMOOTHING_ARTIFACT_TYPE,
        "parent_lineage": {
            "baseline_run_id": parent.baseline_run_id,
            "baseline_fingerprint": parent.baseline_fingerprint,
            "parent_prepared_data_sha256": parent.prepared_data_sha256,
        },
        "source_prepared": {
            "csv_path": SOURCE_CSV,
            "metadata_path": SOURCE_METADATA,
            "prepared_data_sha256": parent.prepared_data_sha256,
        },
        "smoothing": {
            "smoothing_fingerprint": result.smoothing_fingerprint,
            "method": result.config.method,
            "parameters": dict(parameters),
            "config_path": CONFIG_JSON,
            "removed_component_path": REMOVED_CSV,
        },
        "child_prepared": {
            "csv_path": CHILD_CSV,
            "metadata_path": CHILD_METADATA,
            "prepared_data_sha256": child.prepared_data_sha256,
        },
        "metrics": {
            "json_path": METRICS_JSON,
            "csv_path": METRICS_CSV,
        },
        "figures": {
            "selected_spectrum_index": selected_spectrum_index,
            "overlay_path": OVERLAY_PNG,
            "residual_path": RESIDUAL_PNG,
        },
    }


def _scientific_files(
    result: PostBaselineSmoothingResult,
    child: PreparedSpectralDataset,
) -> tuple[dict[str, bytes], int]:
    parent = result.parent_prepared
    source_export = serialize_prepared(
        parent,
        csv_name=SOURCE_CSV,
        metadata_name=SOURCE_METADATA,
    )
    child_export = serialize_prepared(
        child,
        csv_name=CHILD_CSV,
        metadata_name=CHILD_METADATA,
    )
    selected_index = _selected_spectrum_index(parent)
    return {
        SOURCE_CSV: source_export.csv_bytes,
        SOURCE_METADATA: source_export.metadata_bytes,
        CHILD_CSV: child_export.csv_bytes,
        CHILD_METADATA: child_export.metadata_bytes,
        REMOVED_CSV: _wide_spectra_csv(
            parent.wavenumber,
            result.removed_component,
            parent.perturbation_labels,
        ),
        CONFIG_JSON: _json_bytes(_config_payload(result)),
        METRICS_JSON: _json_bytes(
            _metrics_payload(
                result,
                child,
                selected_spectrum_index=selected_index,
            )
        ),
        METRICS_CSV: _metrics_csv(result),
    }, selected_index


def _bundle_files(
    result: PostBaselineSmoothingResult,
    child: PreparedSpectralDataset,
) -> tuple[dict[str, bytes], int]:
    files, selected_index = _scientific_files(result, child)
    files.update(
        {
            OVERLAY_PNG: _figure_png(
                result,
                spectrum_index=selected_index,
                residual=False,
            ),
            RESIDUAL_PNG: _figure_png(
                result,
                spectrum_index=selected_index,
                residual=True,
            ),
        }
    )
    return files, selected_index


def build_smoothing_bundle(
    result: PostBaselineSmoothingResult,
    prepared: PreparedSpectralDataset,
) -> bytes:
    """Build one deterministic, self-contained smoothing scientific artifact."""

    if not isinstance(result, PostBaselineSmoothingResult):
        raise TypeError("result must be a PostBaselineSmoothingResult")
    if not isinstance(prepared, PreparedSpectralDataset):
        raise TypeError("prepared must be a PreparedSpectralDataset")
    if not result.config.enabled:
        raise ValueError("a smoothing bundle requires enabled=True")
    expected_child = prepared_from_smoothed_result(result)
    _assert_prepared_exact(
        prepared,
        expected_child,
        context="smoothing bundle child Prepared",
    )
    files, selected_index = _bundle_files(result, prepared)
    if set(files) != SMOOTHING_PAYLOAD_MEMBERS:  # pragma: no cover - defensive
        raise ValueError("smoothing bundle payload member contract is incomplete")
    bundle = _build_manifest_archive(
        files,
        directories=(FIGURES_DIRECTORY,),
        manifest_base=_manifest_base(
            result,
            prepared,
            selected_spectrum_index=selected_index,
        ),
    )
    if not verify_smoothing_bundle(bundle):  # pragma: no cover - defensive
        raise ValueError("built smoothing bundle failed verification")
    return bundle


def _expected_file_entries(
    archive: zipfile.ZipFile,
) -> list[dict[str, Any]]:
    return [
        {
            "path": name,
            "size_bytes": len(archive.read(name)),
            "sha256": _sha256(archive.read(name)),
            "media_type": _media_type(name),
        }
        for name in sorted(SMOOTHING_PAYLOAD_MEMBERS)
    ]


def _verify_smoothing_semantics(payload: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        if set(archive.namelist()) != SMOOTHING_BUNDLE_MEMBERS:
            raise ValueError("smoothing bundle member set is not exact")
        manifest_value = json.loads(archive.read(MANIFEST_JSON))
        if not isinstance(manifest_value, Mapping):
            raise ValueError("smoothing manifest root must be an object")
        manifest = dict(manifest_value)
        parent = _prepared_from_archive_members(
            archive,
            csv_member=SOURCE_CSV,
            metadata_member=SOURCE_METADATA,
            context="smoothing source Prepared",
        )
        child = _prepared_from_archive_members(
            archive,
            csv_member=CHILD_CSV,
            metadata_member=CHILD_METADATA,
            context="smoothing child Prepared",
        )
        config_value = json.loads(archive.read(CONFIG_JSON))
        if not isinstance(config_value, Mapping):
            raise ValueError("smoothing config root must be an object")
        config = PostBaselineSmoothingConfig.from_dict(config_value)
        if dict(config_value) != config.to_dict():
            raise ValueError("smoothing config schema is not exact")
        if not config.enabled:
            raise ValueError("smoothing bundle config must be enabled")
        recomputed = apply_post_baseline_smoothing(parent, config)
        expected_child = prepared_from_smoothed_result(recomputed)
        _assert_prepared_exact(
            child,
            expected_child,
            context="reloaded smoothing child Prepared",
        )
        selected_index = _selected_spectrum_index(parent)
        expected_files, expected_index = _scientific_files(
            recomputed,
            expected_child,
        )
        if expected_index != selected_index:  # pragma: no cover - deterministic helper
            raise ValueError("selected smoothing spectrum is not deterministic")
        for name in (
            SOURCE_CSV,
            SOURCE_METADATA,
            CHILD_CSV,
            CHILD_METADATA,
            REMOVED_CSV,
            CONFIG_JSON,
            METRICS_JSON,
            METRICS_CSV,
        ):
            if archive.read(name) != expected_files[name]:
                raise ValueError(f"smoothing bundle semantic member differs: {name}")
        for name in (OVERLAY_PNG, RESIDUAL_PNG):
            if not archive.read(name).startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"smoothing figure is not a PNG: {name}")
        expected_manifest = _manifest_base(
            recomputed,
            expected_child,
            selected_spectrum_index=selected_index,
        )
        expected_manifest.update(
            {
                "directories": [FIGURES_DIRECTORY],
                "hash_algorithm": "SHA-256",
                "files": _expected_file_entries(archive),
            }
        )
        expected_keys = {*expected_manifest, "manifest_sha256"}
        if set(manifest) != expected_keys:
            raise ValueError("smoothing manifest schema is not exact")
        for name, expected in expected_manifest.items():
            if manifest.get(name) != expected:
                raise ValueError(f"smoothing manifest semantic field differs: {name}")


def verify_smoothing_bundle(source: SmoothingBundleSource) -> bool:
    """Verify generic hashes plus the complete smoothing scientific contract."""

    try:
        payload = _bundle_payload(source)
        if not verify_workbench_manifest(payload):
            return False
        _verify_smoothing_semantics(payload)
    except (
        KeyError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ):
        return False
    return True


def load_smoothing_prepared(
    source: SmoothingBundleSource,
) -> PreparedSpectralDataset:
    """Strictly verify a smoothing ZIP and return its committed child Prepared."""

    payload = _bundle_payload(source)
    if not verify_smoothing_bundle(payload):
        raise ValueError("post-baseline smoothing ZIP verification failed")
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        return _prepared_from_archive_members(
            archive,
            csv_member=CHILD_CSV,
            metadata_member=CHILD_METADATA,
            context="verified smoothing child Prepared",
        )


__all__ = [
    "SMOOTHING_ARTIFACT_TYPE",
    "SMOOTHING_BUNDLE_MEMBERS",
    "SMOOTHING_PAYLOAD_MEMBERS",
    "SmoothingBundleSource",
    "build_smoothing_bundle",
    "load_smoothing_prepared",
    "verify_smoothing_bundle",
]
