from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest

import ftir_workbench.export as export_module
from ftir2dcos.twodcos import compute_2dcos
from ftir_baseline.config import PipelineConfig
from ftir_baseline.export import build_export_zip, verify_export_manifest
from ftir_baseline.models import SpectrumSet
from ftir_workbench.export import (
    build_project_bundle,
    build_twodcos_bundle,
    load_prepared,
    verify_project_bundle,
    verify_twodcos_bundle,
    verify_workbench_manifest,
)
from ftir_workbench.fingerprints import prepared_data_sha256
from ftir_workbench.models import PreparedSpectralDataset
from ftir_workbench.services import BaselineWorkflowService


def make_series() -> SpectrumSet:
    wavenumber = np.linspace(1800.0, 900.0, 61)
    baseline = 0.02 + 2.0e-5 * (1800.0 - wavenumber)
    peak = np.exp(-0.5 * ((wavenumber - 1250.0) / 32.0) ** 2)
    spectra = np.vstack(
        [baseline + scale * 0.2 * peak for scale in (0.8, 1.0, 1.25)]
    )
    return SpectrumSet(
        wavenumber=wavenumber,
        perturbation=np.array([0.0, 2.0, 5.0]),
        perturbation_labels=("0MIN", "2MIN", "5MIN"),
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="synthetic-baseline-service",
        metadata={"order_policy": "numeric_perturbation_stable"},
    )


def baseline_config() -> PipelineConfig:
    return PipelineConfig(
        input_unit="absorbance",
        wavenumber_range=(1800.0, 900.0),
        series_mode="independent_locked",
        coarse_baseline={"method": "none"},
        fine_baseline={"enabled": False, "method": "none"},
        normalization={"method": "none"},
    )


def with_lineage_mismatch(
    prepared: PreparedSpectralDataset,
    field_name: str,
) -> PreparedSpectralDataset:
    if field_name == "parent_baseline_run_id":
        return replace(prepared, baseline_run_id="different-baseline-run")
    if field_name == "parent_baseline_fingerprint":
        return replace(prepared, baseline_fingerprint="c" * 64)
    if field_name == "parent_prepared_data_sha256":
        spectra = np.array(prepared.spectra, dtype=np.float64, copy=True)
        spectra[0, 0] += 1.0e-6
        data_hash = prepared_data_sha256(
            prepared.wavenumber,
            prepared.perturbation,
            prepared.perturbation_labels,
            spectra,
            normalization_state=prepared.normalization_state,
        )
        return replace(
            prepared,
            spectra=spectra,
            prepared_data_sha256=data_hash,
        )
    raise AssertionError(f"unsupported test lineage field {field_name!r}")


def signed_project_without_lineage_gate(
    baseline_bundle: bytes,
    twodcos_bundle: bytes,
) -> bytes:
    """Construct a correctly signed outer archive without using the guarded builder."""

    return export_module._build_manifest_archive(
        {
            "baseline_run.zip": baseline_bundle,
            "twodcos_run_01.zip": twodcos_bundle,
            "project_config.json": b"{}\n",
        },
        manifest_base={
            "schema_version": "1.0",
            "artifact_type": "ftir_workbench_project",
            "twodcos_run_count": 1,
        },
    )


def test_baseline_service_uses_analysis_data_and_never_calls_2d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ftir2dcos.twodcos as twodcos

    monkeypatch.setattr(
        twodcos,
        "compute_2dcos",
        lambda *args, **kwargs: pytest.fail("baseline-only workflow called 2D-COS"),
    )
    service = BaselineWorkflowService()

    result = service.run(make_series(), baseline_config())
    prepared = service.prepared(baseline_run_id="baseline-service-test")

    np.testing.assert_array_equal(prepared.spectra, result.analysis_data)
    assert prepared.normalization_state == "none"
    assert prepared.baseline_run_id == "baseline-service-test"


def test_baseline_only_bundle_adds_prepared_sidecar_and_remains_verifiable() -> None:
    service = BaselineWorkflowService()
    result = service.run(make_series(), baseline_config())
    prepared = service.prepared(baseline_run_id="baseline-service-export")

    bundle = service.export_baseline_only(
        result,
        prepared=prepared,
        qc_figures={},
    )

    assert isinstance(bundle, bytes)
    assert verify_export_manifest(bundle)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        names = set(archive.namelist())
        assert "corrected_absorbance_for_2dcos.csv" in names
        assert "prepared_spectrum.meta.json" in names
        sidecar = json.loads(archive.read("prepared_spectrum.meta.json"))
        manifest = json.loads(archive.read("manifest.json"))
        listed = {entry["path"] for entry in manifest["files"]}
    assert sidecar["baseline_run_id"] == "baseline-service-export"
    assert sidecar["prepared_data_sha256"] == prepared.prepared_data_sha256
    assert "prepared_spectrum.meta.json" in listed

    imported = load_prepared(bundle)
    np.testing.assert_array_equal(imported.spectra, prepared.spectra)
    np.testing.assert_array_equal(imported.wavenumber, prepared.wavenumber)
    assert imported.prepared_data_sha256 == prepared.prepared_data_sha256


def test_baseline_service_requires_a_completed_run() -> None:
    service = BaselineWorkflowService()

    with pytest.raises(RuntimeError, match="no completed baseline run"):
        service.prepared()
    with pytest.raises(RuntimeError, match="no completed baseline run"):
        service.export_baseline_only(qc_figures={})


def test_full_project_bundle_verifies_nested_baseline_and_twodcos_runs() -> None:
    service = BaselineWorkflowService()
    result = service.run(make_series(), baseline_config())
    prepared = service.prepared(baseline_run_id="baseline-project-test")
    baseline_bundle = service.export_baseline_only(
        result,
        prepared=prepared,
        qc_figures={},
    )
    analysis = compute_2dcos(prepared.spectra, prepared.wavenumber)
    twodcos_bundle = build_twodcos_bundle(
        prepared,
        analysis,
        {"ranges": [[1800.0, 900.0]], "convention": "2dpy_compatible"},
    )

    project = build_project_bundle(
        baseline_bundle,
        (twodcos_bundle,),
        {"project_name": "中文 FTIR 项目"},
    )

    assert verify_project_bundle(project)
    with zipfile.ZipFile(io.BytesIO(project), "r") as archive:
        assert {
            "baseline_run.zip",
            "twodcos_run_01.zip",
            "project_config.json",
            "manifest.json",
        } <= set(archive.namelist())


@pytest.mark.parametrize(
    "field_name",
    (
        "parent_baseline_run_id",
        "parent_baseline_fingerprint",
        "parent_prepared_data_sha256",
    ),
)
def test_project_rejects_resigned_cross_run_lineage_mismatch(field_name: str) -> None:
    service = BaselineWorkflowService()
    result = service.run(make_series(), baseline_config())
    prepared = service.prepared(baseline_run_id="baseline-project-parent")
    baseline_bundle = service.export_baseline_only(
        result,
        prepared=prepared,
        qc_figures={},
    )
    assert isinstance(baseline_bundle, bytes)
    mismatched = with_lineage_mismatch(prepared, field_name)
    analysis = compute_2dcos(mismatched.spectra, mismatched.wavenumber)
    twodcos_bundle = build_twodcos_bundle(
        mismatched,
        analysis,
        {"ranges": [[1800.0, 900.0]], "convention": "2dpy_compatible"},
    )
    assert verify_twodcos_bundle(twodcos_bundle)

    with pytest.raises(ValueError, match=field_name):
        build_project_bundle(baseline_bundle, (twodcos_bundle,))

    resigned_project = signed_project_without_lineage_gate(
        baseline_bundle,
        twodcos_bundle,
    )
    assert verify_workbench_manifest(resigned_project)
    assert not verify_project_bundle(resigned_project)


def test_baseline_only_project_still_accepts_legacy_bundle_without_sidecar() -> None:
    result = BaselineWorkflowService().run(make_series(), baseline_config())
    legacy_baseline = build_export_zip(result, qc_figures={})
    with zipfile.ZipFile(io.BytesIO(legacy_baseline), "r") as archive:
        assert "prepared_spectrum.meta.json" not in archive.namelist()

    project = build_project_bundle(legacy_baseline)

    assert verify_project_bundle(project)
