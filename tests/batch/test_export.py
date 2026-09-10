from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest

import ftir_workbench.batch.service as service
from ftir_baseline.config import CoarseBaselineConfig
from ftir_workbench.batch.export import build_batch_export, can_export_wide, verify_batch_export
from ftir_workbench.batch.importing import import_sources
from ftir_workbench.batch.models import BatchError
from ftir_workbench.batch.service import preview_coarse, preview_fine
from ftir_workbench.batch.state import confirm_coarse, confirm_fine, set_coarse_draft, skip_fine

from .test_importing import table_bytes
from .test_state_and_stages import workspace_pair


def ready_pair():  # type: ignore[no-untyped-def]
    ws, a, b = workspace_pair()
    for sid in (a, b):
        preview_coarse(ws, sid)
        confirm_coarse(ws, sid)
        preview_fine(ws, sid)
        confirm_fine(ws, sid)
    return ws, a, b


def test_e01_e02_e06_export_exact_confirmed_arrays_and_never_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    ws, a, b = ready_pair()
    snapshots = {sid: ws.states[sid].fine_snapshot for sid in (a, b)}

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("export must not fit")

    monkeypatch.setattr(service, "run_pipeline", forbidden)
    for stage in ("coarse", "fine"):
        artifact = build_batch_export(ws, [a, b], stage=stage, include_wide=True)
        assert verify_batch_export(artifact.zip_bytes)
        assert artifact.metadata["exported_spectrum_ids"] == [a, b]
        assert artifact.metadata["is_2d_ready"] is False
        with zipfile.ZipFile(io.BytesIO(artifact.zip_bytes)) as archive:
            assert not any("for_2dcos" in name or "prepared_spectrum" in name for name in archive.namelist())
            for entry in artifact.metadata["spectra"]:
                snapshot = getattr(ws.states[entry["spectrum_id"]], f"{stage}_snapshot")
                values = np.loadtxt(io.BytesIO(archive.read(entry["spectra_csv"])), delimiter=",", skiprows=1)
                np.testing.assert_array_equal(values[:, 0], snapshot.result.absorbance_selected.wavenumber)
                np.testing.assert_array_equal(values[:, 1], snapshot.result.analysis_data[0])
                components = np.loadtxt(io.BytesIO(archive.read(entry["components_csv"])), delimiter=",", skiprows=1)
                np.testing.assert_array_equal(components[:, 2], snapshot.result.baseline.coarse_baseline[0])
                np.testing.assert_array_equal(components[:, 3], snapshot.result.baseline.fine_baseline[0])
                assert entry["stage_fingerprint"] == snapshot.fingerprint
            wide = np.loadtxt(io.BytesIO(artifact.wide_csv_bytes), delimiter=",", skiprows=1)
            np.testing.assert_array_equal(wide[:, 1], getattr(ws.states[a], f"{stage}_snapshot").result.analysis_data[0])
            assert a in artifact.wide_csv_bytes.decode().splitlines()[0]
            assert b in artifact.wide_csv_bytes.decode().splitlines()[0]
            qc = archive.read("qc_summary.csv").decode()
            assert "cross_sample_time_continuity,N/A,not_applicable" in qc
            assert ",perturbation," not in qc
    assert ws.states[a].fine_snapshot is snapshots[a]
    assert ws.states[b].fine_snapshot is snapshots[b]
    assert ws.export_history == [] and ws.last_export_summary is None


@pytest.mark.parametrize("axis_kind", ["coordinate", "reverse", "short"])
def test_e03_actual_different_axes_reject_wide_but_export_zip(axis_kind: str) -> None:
    ws, a, _ = ready_pair()
    x = ws.records[a].wavenumber.copy()
    if axis_kind == "coordinate":
        x[10] += 0.01
    elif axis_kind == "reverse":
        x = x[::-1]
    else:
        x = x[:-1]
    b = import_sources(ws, [("unrelated.csv", table_bytes(x, 0.1 + x / 10000))])[0]
    preview_coarse(ws, b)
    confirm_coarse(ws, b)
    assert not can_export_wide(ws, [a, b], stage="coarse")[0]
    artifact = build_batch_export(ws, [a, b], stage="coarse")
    assert artifact.wide_csv_bytes is None
    assert artifact.wide_unavailable_reason
    with pytest.raises(BatchError, match="AXIS_MISMATCH_FOR_WIDE_EXPORT"):
        build_batch_export(ws, [a, b], stage="coarse", include_wide=True)


def test_e04_stale_fine_blocked_or_explicitly_reported_ready_only() -> None:
    ws, a, b = ready_pair()
    set_coarse_draft(ws, a, CoarseBaselineConfig(method="linear"))
    preview_coarse(ws, a)
    confirm_coarse(ws, a)
    with pytest.raises(BatchError, match="FINE_STALE"):
        build_batch_export(ws, [a, b], stage="fine")
    artifact = build_batch_export(ws, [a, b], stage="fine", ready_only=True)
    assert artifact.metadata["exported_spectrum_ids"] == [b]
    assert artifact.metadata["excluded_items"][0]["spectrum_id"] == a
    assert artifact.metadata["excluded_items"][0]["reason_code"] == "FINE_STALE"


def test_e04_unprocessed_failed_and_explicit_exclusion_report() -> None:
    ws, a, b = workspace_pair()
    preview_coarse(ws, a)
    confirm_coarse(ws, a)
    ws.states[b].errors[:] = ["PIPELINE_FAILED: synthetic failure"]
    ws.import_issues.append({"code": "IMPORT_FAILED", "original_filename": "broken.csv"})
    artifact = build_batch_export(ws, [a, b], stage="coarse", ready_only=True)
    assert artifact.report[1]["reason_code"] == "COARSE_REQUIRED"
    assert "synthetic failure" in artifact.report[1]["recorded_errors"]
    assert artifact.metadata["import_issues"][0]["code"] == "IMPORT_FAILED"
    ws.records[b] = replace(ws.records[b], excluded=True)
    assert build_batch_export(ws, [a, b], stage="coarse", ready_only=True).report[1]["reason_code"] == "EXCLUDED"


def test_e05_valid_or_invalid_uncommitted_editor_is_reported_but_never_applied() -> None:
    ws, a, _ = ready_pair()
    original = ws.states[a].coarse_snapshot
    set_coarse_draft(ws, a, CoarseBaselineConfig(method="offset"))
    ws.states[a].display_preferences["error_fine"] = "invalid unfinished anchors"
    artifact = build_batch_export(ws, [a], stage="coarse")
    assert artifact.report[0]["uncommitted_draft_not_used"]
    assert "unfinished anchors" in artifact.report[0]["invalid_editor_not_used"]
    assert artifact.metadata["spectra"][0]["stage_fingerprint"] == original.fingerprint


def test_fine_not_decided_and_explicit_skip_are_different() -> None:
    ws, a, _ = workspace_pair()
    preview_coarse(ws, a)
    confirm_coarse(ws, a)
    with pytest.raises(BatchError, match="FINE_REQUIRED"):
        build_batch_export(ws, [a], stage="fine")
    skip_fine(ws, a)
    artifact = build_batch_export(ws, [a], stage="fine")
    assert artifact.metadata["spectra"][0]["fine_decision"] == "explicitly_skipped"
    assert np.all(ws.states[a].fine_snapshot.result.baseline.fine_baseline == 0)


def test_e09_derived_transmittance_preserves_negative_absorbance_and_provenance() -> None:
    ws, a, _ = workspace_pair()
    set_coarse_draft(ws, a, CoarseBaselineConfig(method="linear"))
    preview_coarse(ws, a)
    confirm_coarse(ws, a)
    artifact = build_batch_export(ws, [a], stage="coarse", derived_units=("percent_transmittance",))
    entry = artifact.metadata["spectra"][0]
    with zipfile.ZipFile(io.BytesIO(artifact.zip_bytes)) as archive:
        derived = np.loadtxt(io.BytesIO(archive.read(entry["derived_outputs"][0]["filename"])),
                             delimiter=",", skiprows=2)
        y = ws.states[a].coarse_snapshot.result.analysis_data[0]
        np.testing.assert_array_equal(derived[:, 1], 100 * np.power(10.0, -y))
        assert np.any(y < 0) and np.any(derived[:, 1] > 100)
        recipe = json.loads(archive.read(f"recipes/{a}.json"))
        assert recipe["spectrum"]["source_id"] == ws.records[a].source_id
        assert recipe["qc_applicability"]["internal_adapter_placeholder"] == "not an experimental perturbation"


def test_duplicate_names_and_values_keep_distinct_export_provenance() -> None:
    ws, a, _ = ready_pair()
    source = ws.sources[ws.records[a].source_id]
    other = import_sources(ws, [(source.original_filename, source.original_bytes)])[0]
    preview_coarse(ws, other)
    confirm_coarse(ws, other)
    artifact = build_batch_export(ws, [a, other], stage="coarse", include_wide=True)
    entries = artifact.metadata["spectra"]
    assert entries[0]["source_id"] != entries[1]["source_id"]
    assert entries[0]["spectra_csv"] != entries[1]["spectra_csv"]
    assert entries[0]["scientific_input_sha256"] == entries[1]["scientific_input_sha256"]
    assert entries[0]["stage_fingerprint"] != entries[1]["stage_fingerprint"]


@pytest.mark.parametrize("ids", [[], ["missing"]])
def test_empty_and_unknown_export_selection_is_explicit(ids: list[str]) -> None:
    ws, _, _ = workspace_pair()
    with pytest.raises(BatchError, match="EXPORT_NOT_READY"):
        build_batch_export(ws, ids, stage="coarse")
    assert not can_export_wide(ws, ids, stage="coarse")[0]


@pytest.mark.parametrize("text", ["=1+1", "+SUM(1)", "-1+1", "@formula", "  =1+1", "\t=1+1", "\rvalue"])
def test_csv_formula_text_is_escaped_without_changing_metadata_or_numeric_values(text: str) -> None:
    import csv

    from ftir_workbench.batch.export import _table

    value = _table([text, "scientific"], [[text, -0.125]])
    parsed = list(csv.reader(io.StringIO(value.decode())))
    assert parsed[0][0].startswith("'")
    assert parsed[1][0] == "'" + text
    assert parsed[1][1] == "-0.125"
    ws, a, _ = ready_pair()
    ws.records[a] = replace(ws.records[a], display_name=text)
    artifact = build_batch_export(ws, [a], stage="coarse", include_wide=True)
    assert artifact.metadata["spectra"][0]["display_name"] == text
    with zipfile.ZipFile(io.BytesIO(artifact.zip_bytes)) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("processing_report.csv").decode())))
        assert rows[0]["display_name"] == "'" + text
