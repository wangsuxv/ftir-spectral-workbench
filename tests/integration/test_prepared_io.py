from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

import ftir_workbench.export as export_module
from ftir2dcos.twodcos import compute_2dcos
from ftir_workbench.config import TwoDCOSConfig, TwoDCOSRange
from ftir_workbench.export import (
    build_twodcos_bundle,
    export_prepared,
    load_prepared,
    serialize_prepared,
    verify_twodcos_bundle,
    verify_workbench_manifest,
)
from ftir_workbench.fingerprints import prepared_data_sha256
from ftir_workbench.models import PreparedSpectralDataset
from ftir_workbench.services.twodcos_service import TwoDCOSWorkflowService


def make_prepared() -> PreparedSpectralDataset:
    wavenumber = np.array(
        [
            np.nextafter(1736.0, np.inf),
            1662.1234567890123,
            1588.9876543210987,
            np.nextafter(1509.0, -np.inf),
        ],
        dtype=np.float64,
    )
    perturbation = np.array([0.0, 2.0, 5.0], dtype=np.float64)
    labels = ("0 分钟", "2 分钟", "5 分钟")
    spectra = np.array(
        [
            [0.1, np.nextafter(0.2, np.inf), -0.0, 0.4],
            [0.15, 0.27, 0.31, np.nextafter(0.46, -np.inf)],
            [0.22, 0.35, 0.49, 0.63],
        ],
        dtype=np.float64,
    )
    data_hash = prepared_data_sha256(wavenumber, perturbation, labels, spectra)
    return PreparedSpectralDataset(
        wavenumber=wavenumber,
        perturbation=perturbation,
        perturbation_labels=labels,
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="中文 原始数据.dpt",
        source_sha256="a" * 64,
        baseline_run_id="baseline-fixture",
        baseline_fingerprint="b" * 64,
        prepared_data_sha256=data_hash,
        original_axis_direction="descending",
        current_axis_direction="descending",
        perturbation_order_policy="sort_by_perturbation",
        baseline_recipe={"方法": "endpoint", "lambda": 1.0e6},
        baseline_qc={"reconstruction_passed": True},
        warnings=("保留轻微负吸光度",),
        normalization_state="none",
    )


def assert_same_prepared(
    expected: PreparedSpectralDataset,
    actual: PreparedSpectralDataset,
) -> None:
    np.testing.assert_array_equal(actual.wavenumber, expected.wavenumber)
    np.testing.assert_array_equal(actual.perturbation, expected.perturbation)
    np.testing.assert_array_equal(actual.spectra, expected.spectra)
    assert actual.perturbation_labels == expected.perturbation_labels
    assert actual.prepared_data_sha256 == expected.prepared_data_sha256
    assert actual.baseline_fingerprint == expected.baseline_fingerprint
    assert actual.baseline_run_id == expected.baseline_run_id
    assert actual.current_axis_direction == expected.current_axis_direction


def resign_bundle(
    bundle: bytes,
    *,
    manifest_updates: dict[str, object] | None = None,
    file_updates: dict[str, bytes] | None = None,
) -> bytes:
    """Rebuild a bundle with valid file hashes and a valid manifest signature."""

    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        files = {
            name: archive.read(name)
            for name in archive.namelist()
            if name != "manifest.json" and not name.endswith("/")
        }
        directories = tuple(name for name in archive.namelist() if name.endswith("/"))
    files.update(file_updates or {})
    manifest.pop("manifest_sha256", None)
    manifest.pop("files", None)
    manifest.pop("directories", None)
    manifest.update(manifest_updates or {})
    return export_module._build_manifest_archive(
        files,
        directories=directories,
        manifest_base=manifest,
    )


def test_prepared_csv_and_sidecar_round_trip_is_elementwise_and_utf8() -> None:
    prepared = make_prepared()

    artifact = serialize_prepared(prepared)
    reloaded = load_prepared(artifact)

    assert_same_prepared(prepared, reloaded)
    assert "中文" in artifact.metadata_bytes.decode("utf-8")
    assert artifact.csv_bytes.decode("utf-8").splitlines()[0] == (
        "Wavenumber,0 分钟,2 分钟,5 分钟"
    )


def test_prepared_files_can_load_from_csv_or_sidecar_path(tmp_path) -> None:
    prepared = make_prepared()
    paths = export_prepared(prepared, tmp_path)
    assert not isinstance(paths, bytes)

    from_csv = load_prepared(paths.csv_path)
    from_sidecar = load_prepared(paths.metadata_path)

    assert_same_prepared(prepared, from_csv)
    assert_same_prepared(prepared, from_sidecar)


def test_bare_csv_is_accepted_with_explicit_incomplete_provenance_warning() -> None:
    artifact = serialize_prepared(make_prepared())

    loaded = load_prepared(artifact.csv_bytes)

    assert loaded.source_sha256 == "unknown"
    assert loaded.baseline_fingerprint == "unknown"
    assert loaded.baseline_run_id == "unknown"
    np.testing.assert_array_equal(loaded.perturbation, [0.0, 2.0, 5.0])
    assert any("Provenance incomplete" in warning for warning in loaded.warnings)


def test_sidecar_detects_csv_tampering() -> None:
    artifact = serialize_prepared(make_prepared())
    tampered = artifact.csv_bytes.replace(b"0.10000000000000001", b"0.20000000000000001", 1)

    with pytest.raises(ValueError, match="SHA-256"):
        load_prepared(tampered, artifact.metadata_bytes)


def test_twodcos_bundle_contains_parent_linked_contract_and_verifies() -> None:
    prepared = make_prepared()
    analysis = compute_2dcos(
        prepared.spectra,
        prepared.wavenumber,
        convention="2dpy_compatible",
    )

    bundle = build_twodcos_bundle(
        prepared,
        analysis,
        {"ranges": [[1736.0, 1509.0]], "convention": "2dpy_compatible"},
    )

    assert verify_twodcos_bundle(bundle)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        names = set(archive.namelist())
        assert {
            "source_prepared_spectrum.csv",
            "source_prepared_spectrum.meta.json",
            "twodcos_config.json",
            "dynamic_spectra.csv",
            "synchronous_matrix.csv",
            "asynchronous_matrix.csv",
            "qc_metrics.json",
            "figures/",
            "peak_order/",
            "manifest.json",
        } <= names
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["parent_baseline_run_id"] == prepared.baseline_run_id
    assert manifest["parent_baseline_fingerprint"] == prepared.baseline_fingerprint
    assert manifest["parent_prepared_data_sha256"] == prepared.prepared_data_sha256


def test_twodcos_bundle_preserves_all_homo_and_cross_range_outputs() -> None:
    prepared = make_prepared()
    config = TwoDCOSConfig(
        ranges=(
            TwoDCOSRange(1737.0, 1600.0, label="upper"),
            TwoDCOSRange(1600.0, 1508.0, label="lower"),
        ),
        cross_range_enabled=True,
    )
    analysis = TwoDCOSWorkflowService().compute(prepared, config)

    bundle = build_twodcos_bundle(prepared, analysis)

    assert verify_twodcos_bundle(bundle)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
    assert {
        "ranges/range_01/range.json",
        "ranges/range_01/dynamic_spectra.csv",
        "ranges/range_01/synchronous_matrix.csv",
        "ranges/range_01/asynchronous_matrix.csv",
        "ranges/range_02/range.json",
        "ranges/range_02/dynamic_spectra.csv",
        "ranges/range_02/synchronous_matrix.csv",
        "ranges/range_02/asynchronous_matrix.csv",
        "cross_ranges/cross_01/ranges.json",
        "cross_ranges/cross_01/synchronous_matrix.csv",
        "cross_ranges/cross_01/asynchronous_matrix.csv",
    } <= names
    assert manifest["homo_result_count"] == 2
    assert manifest["cross_result_count"] == 1
    assert manifest["twodcos_fingerprint"] == analysis.twodcos_fingerprint


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    (
        ("parent_baseline_run_id", "another-baseline-run"),
        ("parent_baseline_fingerprint", "c" * 64),
        ("parent_prepared_data_sha256", "d" * 64),
    ),
)
def test_twodcos_bundle_rejects_resigned_parent_lineage_mismatch(
    field_name: str,
    forged_value: str,
) -> None:
    prepared = make_prepared()
    analysis = compute_2dcos(prepared.spectra, prepared.wavenumber)
    bundle = build_twodcos_bundle(prepared, analysis, {"ranges": [[1737.0, 1508.0]]})

    forged = resign_bundle(bundle, manifest_updates={field_name: forged_value})

    assert verify_workbench_manifest(forged)
    assert not verify_twodcos_bundle(forged)


def test_twodcos_bundle_rejects_resigned_forged_prepared_sidecar_hash() -> None:
    prepared = make_prepared()
    analysis = compute_2dcos(prepared.spectra, prepared.wavenumber)
    bundle = build_twodcos_bundle(prepared, analysis, {"ranges": [[1737.0, 1508.0]]})
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        sidecar = json.loads(archive.read("source_prepared_spectrum.meta.json"))
    sidecar["prepared_data_sha256"] = "e" * 64
    forged_sidecar = (json.dumps(sidecar, sort_keys=True) + "\n").encode("utf-8")

    forged = resign_bundle(
        bundle,
        file_updates={"source_prepared_spectrum.meta.json": forged_sidecar},
    )

    assert verify_workbench_manifest(forged)
    assert not verify_twodcos_bundle(forged)
