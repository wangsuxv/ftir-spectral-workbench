from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import ftir_workbench.export as export_module
from ftir_baseline.models import thaw_mapping
from ftir_workbench import (
    SMOOTHING_BUNDLE_MEMBERS,
    SMOOTHING_PAYLOAD_MEMBERS,
    PostBaselineSmoothingConfig,
    PostBaselineSmoothingResult,
    PostBaselineSmoothingService,
    PreparedSpectralDataset,
    build_smoothing_bundle,
    verify_smoothing_bundle,
)
from ftir_workbench.export import load_prepared, verify_workbench_manifest
from ftir_workbench.smoothing_export import (
    CHILD_CSV,
    CHILD_METADATA,
    CONFIG_JSON,
    FIGURES_DIRECTORY,
    MANIFEST_JSON,
    METRICS_CSV,
    METRICS_JSON,
    OVERLAY_PNG,
    REMOVED_CSV,
    RESIDUAL_PNG,
    SOURCE_CSV,
    SOURCE_METADATA,
)

from ._helpers import make_prepared


def _assert_prepared_equal(
    expected: PreparedSpectralDataset,
    actual: PreparedSpectralDataset,
) -> None:
    np.testing.assert_array_equal(actual.wavenumber, expected.wavenumber)
    np.testing.assert_array_equal(actual.perturbation, expected.perturbation)
    np.testing.assert_array_equal(actual.spectra, expected.spectra)
    assert actual.perturbation_labels == expected.perturbation_labels
    assert actual.intensity_unit == expected.intensity_unit
    assert actual.source_name == expected.source_name
    assert actual.source_sha256 == expected.source_sha256
    assert actual.baseline_run_id == expected.baseline_run_id
    assert actual.baseline_fingerprint == expected.baseline_fingerprint
    assert actual.prepared_data_sha256 == expected.prepared_data_sha256
    assert actual.original_axis_direction == expected.original_axis_direction
    assert actual.current_axis_direction == expected.current_axis_direction
    assert actual.perturbation_order_policy == expected.perturbation_order_policy
    assert actual.normalization_state == expected.normalization_state
    assert thaw_mapping(actual.baseline_recipe) == thaw_mapping(expected.baseline_recipe)
    assert thaw_mapping(actual.baseline_qc) == thaw_mapping(expected.baseline_qc)
    assert actual.warnings == expected.warnings


def _run() -> tuple[
    PreparedSpectralDataset,
    PostBaselineSmoothingResult,
    PreparedSpectralDataset,
    PostBaselineSmoothingConfig,
]:
    parent = make_prepared(
        baseline_recipe={
            "prepared_data_contract": {
                "source_channel": "PipelineResult.analysis_data",
                "scientific_normalization": False,
            },
            "中文 history": {"values": [1, 2, 3]},
        }
    )
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="gaussian",
        gaussian_sigma_points=1.25,
        gaussian_truncate=3.0,
        convolution_mode="mirror",
    )
    result, child = PostBaselineSmoothingService().apply(parent, config)
    return parent, result, child, config


def _resign_bundle(
    bundle: bytes,
    *,
    file_updates: dict[str, bytes] | None = None,
    manifest_updates: dict[str, object] | None = None,
) -> bytes:
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        manifest = json.loads(archive.read(MANIFEST_JSON))
        files = {
            name: archive.read(name)
            for name in archive.namelist()
            if name != MANIFEST_JSON and not name.endswith("/")
        }
        directories = tuple(name for name in archive.namelist() if name.endswith("/"))
    files.update(file_updates or {})
    manifest.pop("manifest_sha256", None)
    manifest.pop("files", None)
    manifest.pop("directories", None)
    manifest.pop("hash_algorithm", None)
    manifest.update(manifest_updates or {})
    return export_module._build_manifest_archive(
        files,
        directories=directories,
        manifest_base=manifest,
    )


def test_bundle_is_deterministic_exact_and_round_trips_child() -> None:
    parent, result, child, config = _run()
    service = PostBaselineSmoothingService()

    first = build_smoothing_bundle(result, child)
    second = service.build_bundle(result, child)

    assert first == second
    assert verify_smoothing_bundle(first)
    assert verify_workbench_manifest(first)
    with zipfile.ZipFile(io.BytesIO(first), "r") as archive:
        assert set(archive.namelist()) == SMOOTHING_BUNDLE_MEMBERS
        assert set(archive.namelist()) == {
            *SMOOTHING_PAYLOAD_MEMBERS,
            FIGURES_DIRECTORY,
            MANIFEST_JSON,
        }
        manifest = json.loads(archive.read(MANIFEST_JSON))
        assert manifest["artifact_type"] == "post_baseline_smoothing_run"
        assert manifest["schema_version"] == "1.0"
        assert manifest["directories"] == [FIGURES_DIRECTORY]
        assert [entry["path"] for entry in manifest["files"]] == sorted(
            SMOOTHING_PAYLOAD_MEMBERS
        )
        assert manifest["parent_lineage"] == {
            "baseline_run_id": parent.baseline_run_id,
            "baseline_fingerprint": parent.baseline_fingerprint,
            "parent_prepared_data_sha256": parent.prepared_data_sha256,
        }
        assert manifest["smoothing"] == {
            "smoothing_fingerprint": result.smoothing_fingerprint,
            "method": "gaussian",
            "parameters": {
                "sigma_points": 1.25,
                "truncate": 3.0,
                "mode": "mirror",
            },
            "config_path": CONFIG_JSON,
            "removed_component_path": REMOVED_CSV,
        }
        assert manifest["child_prepared"]["prepared_data_sha256"] == (
            child.prepared_data_sha256
        )
        assert json.loads(archive.read(CONFIG_JSON)) == config.to_dict()
        metrics = json.loads(archive.read(METRICS_JSON))
        assert metrics["smoothing_fingerprint"] == result.smoothing_fingerprint
        assert metrics["summary_metrics"] == dict(result.summary_metrics)
        assert set(metrics["per_spectrum_metrics"]) == set(
            result.per_spectrum_metrics
        )
        assert metrics["diagnostic_thresholds"] == {
            "relative_rms_removed_warning_threshold": 0.10,
            "first_derivative_correlation_warning_threshold": 0.95,
            "relative_absolute_area_change_warning_threshold": 0.02,
            "edge_effect_ratio_warning_threshold": 2.0,
        }
        assert archive.read(OVERLAY_PNG).startswith(b"\x89PNG\r\n\x1a\n")
        assert archive.read(RESIDUAL_PNG).startswith(b"\x89PNG\r\n\x1a\n")

        source = load_prepared(
            (archive.read(SOURCE_CSV), archive.read(SOURCE_METADATA))
        )
        embedded_child = load_prepared(
            (archive.read(CHILD_CSV), archive.read(CHILD_METADATA))
        )
        removed_axis, removed_values, removed_labels = (
            export_module._parse_prepared_csv(archive.read(REMOVED_CSV))
        )
        metrics_header = archive.read(METRICS_CSV).decode("utf-8").splitlines()[0]

    _assert_prepared_equal(parent, source)
    _assert_prepared_equal(child, embedded_child)
    _assert_prepared_equal(child, load_prepared(first))
    np.testing.assert_array_equal(removed_axis, parent.wavenumber)
    np.testing.assert_array_equal(removed_values, result.removed_component)
    assert removed_labels == parent.perturbation_labels
    assert metrics_header.startswith(
        "spectrum_index,perturbation,perturbation_label,"
    )


@pytest.mark.parametrize(
    "semantic_target",
    ("manifest", "config", "metrics", "child", "removed"),
)
def test_resigned_semantic_tampering_is_rejected(semantic_target: str) -> None:
    _, result, child, _ = _run()
    bundle = build_smoothing_bundle(result, child)
    updates: dict[str, bytes] = {}
    manifest_updates: dict[str, object] = {}
    if semantic_target == "manifest":
        with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
            manifest = json.loads(archive.read(MANIFEST_JSON))
        smoothing = dict(manifest["smoothing"])
        smoothing["method"] = "median"
        manifest_updates["smoothing"] = smoothing
    elif semantic_target == "config":
        with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
            config = json.loads(archive.read(CONFIG_JSON))
        config["method"] = "median"
        updates[CONFIG_JSON] = export_module._json_bytes(config)
    elif semantic_target == "metrics":
        with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
            metrics = json.loads(archive.read(METRICS_JSON))
        metrics["summary_metrics"]["mean_relative_rms_removed"] = 999.0
        updates[METRICS_JSON] = export_module._json_bytes(metrics)
    else:
        member = CHILD_CSV if semantic_target == "child" else REMOVED_CSV
        with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
            rows = archive.read(member).decode("utf-8").splitlines()
        values = rows[1].split(",")
        values[1] = format(float(values[1]) + 0.125, ".17g")
        rows[1] = ",".join(values)
        updates[member] = ("\n".join(rows) + "\n").encode("utf-8")

    resigned = _resign_bundle(
        bundle,
        file_updates=updates,
        manifest_updates=manifest_updates,
    )

    assert verify_workbench_manifest(resigned)
    assert not verify_smoothing_bundle(resigned)
    with pytest.raises(ValueError, match="smoothing ZIP verification failed"):
        load_prepared(resigned)


def test_unresigned_payload_tamper_and_extra_raw_member_are_rejected() -> None:
    _, result, child, _ = _run()
    bundle = build_smoothing_bundle(result, child)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries[REMOVED_CSV] += b"0,0,0,0\n"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    assert not verify_smoothing_bundle(output.getvalue())

    resigned_with_raw = _resign_bundle(
        bundle,
        file_updates={"raw_original_input.dpt": b"private raw bytes"},
    )
    assert verify_workbench_manifest(resigned_with_raw)
    assert not verify_smoothing_bundle(resigned_with_raw)


def test_bundle_contains_corrected_source_but_no_raw_or_nested_run_payloads() -> None:
    _, result, child, _ = _run()
    bundle = build_smoothing_bundle(result, child)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        names = set(archive.namelist())
        assert SOURCE_CSV in names
        assert not any(
            Path(name).suffix.casefold()
            in {".dpt", ".spa", ".spg", ".spc", ".ftirw"}
            for name in names
        )
        assert not any(name.endswith(".zip") for name in names)
        assert not any("raw_" in Path(name).name.casefold() for name in names)


def test_constant_noop_child_with_same_data_hash_still_verifies() -> None:
    axis = np.linspace(1800.0, 1000.0, 41)
    parent = make_prepared(
        wavenumber=axis,
        spectra=np.vstack(
            (
                np.zeros(axis.size),
                np.full(axis.size, 1.5),
                np.full(axis.size, -2.0),
            )
        ),
    )
    result, child = PostBaselineSmoothingService().apply(
        parent,
        PostBaselineSmoothingConfig(
            enabled=True,
            method="moving_average",
            moving_average_window_length=5,
        ),
    )

    assert child.prepared_data_sha256 == parent.prepared_data_sha256
    bundle = build_smoothing_bundle(result, child)
    assert verify_smoothing_bundle(bundle)
    _assert_prepared_equal(child, load_prepared(bundle))


def test_unicode_and_space_path_verification_and_load(tmp_path: Path) -> None:
    _, result, child, _ = _run()
    bundle = build_smoothing_bundle(result, child)
    destination = tmp_path / "中文 路径" / "平滑 bundle.zip"
    destination.parent.mkdir()
    destination.write_bytes(bundle)

    assert verify_smoothing_bundle(destination)
    _assert_prepared_equal(child, load_prepared(destination))
