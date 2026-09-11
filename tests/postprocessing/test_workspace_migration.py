"""PP-001 and PP-061 to PP-063: exact legacy fixtures, lineage and persistence."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from ftir_baseline.config import CoarseBaselineConfig
from ftir_workbench.batch import postprocessing as postprocess
from ftir_workbench.batch.export import build_batch_export
from ftir_workbench.batch.models import BatchError, BatchWorkspace, PreparationConfig, StageSnapshot
from ftir_workbench.batch.service import preview_coarse
from ftir_workbench.batch.state import (
    confirm_coarse,
    confirm_preparation,
    set_coarse_draft,
    set_preparation_draft,
    skip_fine,
)
from ftir_workbench.batch.workspace import (
    load_batch_workspace,
    make_archive,
    read_verified_archive,
    save_batch_workspace,
)
from tests.postprocessing.helpers import confirmed_workspace
from tests.postprocessing.test_state_contracts import DERIVED, SMOOTH, apply_branch, graph

ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "9fc13c10ae34bd53cb220d0807edb18c9e2e41dc"
ARTIFACT_TYPE = "independent_baseline_workspace"


@pytest.fixture(scope="module")
def legacy_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = ROOT / "outputs/validation-private/v031/legacy"
    if not (directory / "legacy_generation.json").exists():
        directory = tmp_path_factory.mktemp("v031-exact-legacy")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/generate_v031_legacy_fixtures.py"),
             "--base-commit", BASE_COMMIT, "--output", str(directory)],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
    metadata = json.loads((directory / "legacy_generation.json").read_text())
    assert metadata["base_commit"] == BASE_COMMIT
    assert metadata["source_environment"] == "exact temporary Git archive"
    assert metadata["workbench_version"] == "0.3.0"
    assert metadata["experimental_data_read"] is False
    for name, details in metadata["artifacts"].items():
        payload = (directory / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == details["sha256"]
        assert len(payload) == details["size_bytes"]
    return directory


def forbid_calculations(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("workspace serialization/restoration must not run numerical processing")

    for target in (
        "ftir_baseline.pipeline.run_pipeline",
        "ftir_workbench.batch.service.run_pipeline",
        "ftir_workbench.services.baseline_service.BaselineWorkflowService.run",
        "ftir_workbench.post_baseline_smoothing.apply_post_baseline_smoothing",
        "ftir_workbench.post_baseline_smoothing.smooth_spectral_arrays",
        "ftir_workbench.batch.postprocessing.smooth_spectral_arrays",
        "ftir_workbench.batch.postprocessing.normalize_spectral_arrays",
        "ftir_workbench.batch.normalization_adapter.normalize_spectral_arrays",
        "ftir_baseline.normalization.apply_normalization",
        "ftir_workbench.models.PreparedSpectralDataset.__init__",
        "ftir_workbench.services.twodcos_service.TwoDCOSWorkflowService.compute",
    ):
        monkeypatch.setattr(target, forbidden)


def rewrite(payload: bytes, mutate: Callable[[dict, dict[str, bytes]], None]) -> bytes:
    members = read_verified_archive(payload, ARTIFACT_TYPE)
    data = json.loads(members["workspace.json"])
    mutate(data, members)
    members["workspace.json"] = json.dumps(data, allow_nan=False).encode()
    return make_archive(members, ARTIFACT_TYPE)


def assert_stage(actual: StageSnapshot, expected: StageSnapshot) -> None:
    for field in ("stage", "spectrum_id", "input_sha256", "fingerprint", "config",
                  "parent_coarse_fingerprint", "implementation_fingerprint",
                  "parent_coarse_implementation_fingerprint"):
        assert getattr(actual, field) == getattr(expected, field)
    assert actual.result.recipe_dict() == expected.result.recipe_dict()
    for field in ("coarse_baseline", "fine_baseline", "total_baseline", "corrected"):
        np.testing.assert_array_equal(getattr(actual.result.baseline, field),
                                      getattr(expected.result.baseline, field))
    np.testing.assert_array_equal(actual.result.absorbance_selected.wavenumber,
                                  expected.result.absorbance_selected.wavenumber)
    np.testing.assert_array_equal(actual.result.baseline_estimation_spectra,
                                  expected.result.baseline_estimation_spectra)


def assert_derived(actual: postprocess.PostprocessSnapshot,
                   expected: postprocess.PostprocessSnapshot) -> None:
    assert actual is not expected
    for field in ("workspace_id", "spectrum_id", "source_id", "input_sha256", "branch",
                  "parent_fingerprint", "implementation", "request_fingerprint", "fingerprint",
                  "quantity", "purpose", "warnings"):
        assert getattr(actual, field) == getattr(expected, field)
    for field in ("recipe", "effective_recipe", "versions", "reference_details", "qc"):
        assert postprocess.plain(getattr(actual, field)) == postprocess.plain(getattr(expected, field))
    for field in ("wavenumber", "spectra", "removed_component", "scale", "offset"):
        received, wanted = getattr(actual, field), getattr(expected, field)
        if wanted is None:
            assert received is None
        else:
            np.testing.assert_array_equal(received, wanted)
            assert not received.flags.writeable
            with pytest.raises(ValueError):
                received.setflags(write=True)
    assert_stage(actual.baseline, expected.baseline)
    if expected.parent_smoothed is None:
        assert actual.parent_smoothed is None
    else:
        assert actual.parent_smoothed is not None
        assert_derived(actual.parent_smoothed, expected.parent_smoothed)


def rich_workspace() -> tuple[BatchWorkspace, list[str]]:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="other-source.csv")
    _, c = confirmed_workspace(workspace=ws, name="preview-only.csv")
    _, d = confirmed_workspace(workspace=ws, name="coarse-only.csv", skip=False)
    graph(ws, a)
    graph(ws, b)
    # A retains a historical S parent inside stale N_S while current S is ready.
    apply_branch(ws, a, "smoothed", {**SMOOTH, "gaussian_sigma_points": 2.0})
    postprocess.update_draft(ws, a, "smoothed", {"gaussian_sigma_points": None})
    postprocess.update_draft(ws, a, "normalized_baseline", {
        "method": "internal_peak_area", "reference_interval": [None, 1650.0],
    })
    postprocess.update_draft(ws, a, "normalized_smoothed", {"target": None})
    state = postprocess.get_postprocessing_state(ws, a)
    state.normalization_source = "smoothed"
    state.export_choice = "normalized_smoothed"
    state.display_preferences = {"smoothing_unit": "percent_transmittance", "zoom": [1600.0, 1200.0]}
    state.normalized_baseline.errors[:] = ["unfinished reference window"]
    # B retains historical B in all descendants after a new range and B commit.
    prep = ws.states[b].preparation_committed.to_dict()
    prep["wavenumber_range"] = [1700.0, 1100.0]
    set_preparation_draft(ws, b, PreparationConfig(**prep))
    confirm_preparation(ws, b)
    set_coarse_draft(ws, b, CoarseBaselineConfig(method="offset"))
    preview_coarse(ws, b)
    confirm_coarse(ws, b)
    skip_fine(ws, b)
    postprocess.update_draft(ws, c, "smoothed", SMOOTH)
    postprocess.preview_branch(ws, c, "smoothed")
    ws.display_order = [d, c, a, b]
    ws.selected_spectrum_id = a
    ws.selected_spectrum_ids = [b, a]
    return ws, [a, b, c, d]


def test_pp001_pp062_exact_v030_workspace_preserves_all_baseline_arrays_and_progress(
    legacy_directory: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = json.loads((legacy_directory / "legacy_generation.json").read_text())
    forbid_calculations(monkeypatch)
    original = (legacy_directory / "legacy_workspace_v030.zip").read_bytes()
    ws = load_batch_workspace(original)
    assert ws.workspace_id == metadata["workspace_id"]
    assert ws.display_order == metadata["spectrum_ids"]
    assert ws.schema_version == "2.0"
    assert ws.selected_spectrum_ids == []
    with np.load(legacy_directory / "expected_arrays.npz", allow_pickle=False) as expected:
        for sid in ws.display_order:
            record, state = ws.records[sid], ws.states[sid]
            np.testing.assert_array_equal(record.wavenumber, expected[f"{sid}__raw_x"])
            np.testing.assert_array_equal(record.raw_intensity, expected[f"{sid}__raw_y"])
            assert state.fine_decision == metadata["states"][sid]["fine_decision"]
            assert state.fine_stale == metadata["states"][sid]["fine_stale"]
            for stage, details in metadata["states"][sid]["snapshots"].items():
                snapshot = getattr(state, stage + "_snapshot")
                assert snapshot is not None
                for key, value in details.items():
                    assert getattr(snapshot, key) == value
                np.testing.assert_array_equal(snapshot.result.absorbance_selected.wavenumber,
                                              expected[f"{sid}__{stage}__x"])
                np.testing.assert_array_equal(snapshot.result.baseline_estimation_spectra,
                                              expected[f"{sid}__{stage}__estimate"])
                for component in ("coarse_baseline", "fine_baseline", "total_baseline", "corrected"):
                    np.testing.assert_array_equal(getattr(snapshot.result.baseline, component),
                                                  expected[f"{sid}__{stage}__{component}"])
            derived = postprocess.get_postprocessing_state(ws, sid)
            assert derived.normalization_source == derived.export_choice == "baseline"
            for branch in DERIVED:
                part = getattr(derived, branch)
                assert not part.draft["enabled"]
                assert part.preview is part.committed is None
    # Migration emits the new inner schema but leaves the common archive contract.
    migrated = save_batch_workspace(ws)
    with zipfile.ZipFile(io.BytesIO(migrated)) as archive:
        assert json.loads(archive.read("manifest.json"))["schema_version"] == "1.0"
        assert json.loads(archive.read("workspace.json"))["schema_version"] == "2.0"
    restored = load_batch_workspace(migrated)
    for sid in ws.display_order:
        for stage in ("coarse", "fine"):
            before = getattr(ws.states[sid], stage + "_snapshot")
            if before is not None:
                assert_stage(getattr(restored.states[sid], stage + "_snapshot"), before)


@pytest.mark.parametrize("stage", ["coarse", "fine"])
def test_pp001_legacy_export_spectra_recipes_and_qc_values_remain_exact(
    legacy_directory: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    forbid_calculations(monkeypatch)
    ws = load_batch_workspace((legacy_directory / "legacy_workspace_v030.zip").read_bytes())
    actual = build_batch_export(ws, ws.display_order, stage=stage, ready_only=stage == "fine")
    expected = (legacy_directory / f"legacy_{stage}_export_v030.zip").read_bytes()
    received_members = read_verified_archive(actual.zip_bytes, "independent_baseline_batch")
    expected_members = read_verified_archive(expected, "independent_baseline_batch")
    assert set(received_members) == set(expected_members)
    for name, received in received_members.items():
        if name == "qc_summary.csv":
            # Existing workspace JSON serialization sorts metric mapping keys.
            # A restored export therefore changes only QC row order, not values.
            assert sorted(csv.reader(io.StringIO(received.decode()))) == sorted(
                csv.reader(io.StringIO(expected_members[name].decode()))
            )
        else:
            assert received == expected_members[name], name


def test_pp063_roundtrip_preserves_drafts_arrays_selected_ids_and_historical_parent_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, (a, b, c, d) = rich_workspace()
    statuses = {sid: {branch: postprocess.branch_status(ws, sid, branch)["status"]
                      for branch in DERIVED} for sid in (a, b, c, d)}
    forbid_calculations(monkeypatch)
    payload = save_batch_workspace(ws)
    data = json.loads(read_verified_archive(payload, ARTIFACT_TYPE)["workspace.json"])
    assert data["schema_version"] == "2.0"
    assert data["selected_spectrum_ids"] == [b, a]
    assert data["postprocessing"][c]["smoothed"]["preview"] is None
    restored = load_batch_workspace(payload)
    assert restored.schema_version == "2.0"
    assert restored.workspace_id == ws.workspace_id
    assert restored.display_order == ws.display_order
    assert restored.selected_spectrum_id == a and restored.selected_spectrum_ids == [b, a]
    for sid in ws.records:
        assert restored.records[sid].source_id == ws.records[sid].source_id
        source = ws.sources[ws.records[sid].source_id]
        assert restored.sources[source.source_id].original_bytes == source.original_bytes
        assert restored.records[sid].scientific_input_sha256 == ws.records[sid].scientific_input_sha256
        expected_state = postprocess.get_postprocessing_state(ws, sid)
        actual_state = postprocess.get_postprocessing_state(restored, sid)
        for field in ("normalization_source", "export_choice", "display_preferences"):
            assert getattr(actual_state, field) == getattr(expected_state, field)
        for branch in DERIVED:
            actual, expected = getattr(actual_state, branch), getattr(expected_state, branch)
            assert actual.draft == expected.draft
            assert actual.committed_draft == expected.committed_draft
            assert actual.errors == expected.errors
            assert actual.preview is None
            assert postprocess.branch_status(restored, sid, branch)["status"] == statuses[sid][branch]
            if expected.committed is None:
                assert actual.committed is None
            else:
                assert_derived(actual.committed, expected.committed)
            with pytest.raises(BatchError):
                postprocess.confirm_branch(restored, sid, branch)
    saved_a = postprocess.get_postprocessing_state(restored, a)
    assert saved_a.normalized_smoothed.committed.parent_fingerprint != saved_a.smoothed.committed.fingerprint
    assert saved_a.smoothed.draft["gaussian_sigma_points"] is None
    assert saved_a.normalized_baseline.draft["reference_interval"] == [None, 1650.0]
    saved_b = postprocess.get_postprocessing_state(restored, b)
    assert saved_b.smoothed.committed.baseline.fingerprint != restored.states[b].fine_snapshot.fingerprint
    assert restored.states[d].fine_decision == "not_decided"


@pytest.fixture
def complete_payload() -> tuple[bytes, str, str]:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="same-values-other-source.csv")
    graph(ws, a)
    graph(ws, b)
    ws.selected_spectrum_ids = [a, b]
    return save_batch_workspace(ws), a, b


@pytest.mark.parametrize("mutation", [
    "spectrum_id", "source_id", "workspace_id", "input_hash", "parent_hash", "effective_recipe",
    "purpose", "quantity", "output_member", "offset", "baseline_source", "smoothed_parent",
    "wrong_branch", "confirmable_preview", "unknown_branch", "unknown_record", "missing_state",
    "missing_postprocessing", "missing_selection", "duplicate_selection", "unknown_selection",
    "source_choice", "export_choice", "draft_array_codec", "errors_type", "committed_draft",
    "missing_committed_draft", "extra_field", "deep_parent",
])
def test_pp061_rehashed_derived_corruption_is_rejected_before_install(
    complete_payload: tuple[bytes, str, str], mutation: str,
) -> None:
    payload, a, b = complete_payload

    def corrupt(data: dict, members: dict[str, bytes]) -> None:
        state = data["postprocessing"][a]
        part = state["normalized_smoothed"]
        snapshot = part["committed"]
        choices = {
            "spectrum_id": (snapshot, "spectrum_id", b),
            "source_id": (snapshot, "source_id", data["records"][b]["source_id"]),
            "workspace_id": (snapshot, "workspace_id", "other-workspace"),
            "input_hash": (snapshot, "input_sha256", "0" * 64),
            "parent_hash": (snapshot, "parent_fingerprint", "0" * 64),
            "effective_recipe": (snapshot, "effective_recipe", {"enabled": True, "method": "vector", "target": 1.0}),
            "purpose": (snapshot, "purpose", "display_only"),
            "quantity": (snapshot, "quantity", "absorbance"),
            "offset": (snapshot, "offset", snapshot["scale"]),
            "baseline_source": (snapshot, "baseline", data["postprocessing"][b]["smoothed"]["committed"]["baseline"]),
            "smoothed_parent": (snapshot, "parent_smoothed", data["postprocessing"][b]["smoothed"]["committed"]),
            "wrong_branch": (snapshot, "branch", "normalized_baseline"),
            "confirmable_preview": (part, "preview", copy.deepcopy(snapshot)),
            "unknown_branch": (state, "uncontracted_branch", copy.deepcopy(part)),
            "unknown_record": (data["postprocessing"], "unknown-record", copy.deepcopy(state)),
            "duplicate_selection": (data, "selected_spectrum_ids", [a, a]),
            "unknown_selection": (data, "selected_spectrum_ids", [a, "unknown-record"]),
            "source_choice": (state, "normalization_source", "normalized_baseline"),
            "export_choice": (state, "export_choice", "unknown-branch"),
            "draft_array_codec": (part["draft"], "reference_interval", snapshot["wavenumber"]),
            "errors_type": (part, "errors", "must be a list"),
            "committed_draft": (part, "committed_draft", {"enabled": True, "method": "vector"}),
            "extra_field": (snapshot, "executable_object", "not supported"),
        }
        if mutation in choices:
            target, field, value = choices[mutation]
            target[field] = value
        elif mutation == "output_member":
            member = snapshot["spectra"]["__array__"]
            stream = io.BytesIO()
            np.save(stream, np.load(io.BytesIO(members[member]), allow_pickle=False) + 0.1,
                    allow_pickle=False)
            members[member] = stream.getvalue()
        elif mutation == "missing_state":
            del state["normalized_baseline"]
        elif mutation == "missing_postprocessing":
            del data["postprocessing"]
        elif mutation == "missing_selection":
            del data["selected_spectrum_ids"]
        elif mutation == "missing_committed_draft":
            part["committed_draft"] = None
        else:
            # A JSON tree cannot contain an actual reference cycle. Model an
            # invalid recursive S parent chain without executable deserialization.
            node = snapshot["parent_smoothed"]
            for _ in range(12):
                child = copy.deepcopy(node)
                node["parent_smoothed"] = child
                node = child
    with pytest.raises(BatchError, match="WORKSPACE_INTEGRITY_FAILED"):
        load_batch_workspace(rewrite(payload, corrupt))


@pytest.mark.parametrize("field", ["postprocessing", "selected_spectrum_ids"])
def test_pp061_legacy_schema_cannot_smuggle_new_business_fields(
    legacy_directory: Path, field: str,
) -> None:
    payload = (legacy_directory / "legacy_workspace_v030.zip").read_bytes()

    def corrupt(data: dict, members: dict[str, bytes]) -> None:
        data[field] = {} if field == "postprocessing" else []
    with pytest.raises(BatchError, match="WORKSPACE_INTEGRITY_FAILED"):
        load_batch_workspace(rewrite(payload, corrupt))


def test_pp061_derived_archive_retains_path_and_resource_boundaries(
    complete_payload: tuple[bytes, str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _, _ = complete_payload
    members = read_verified_archive(payload, ARTIFACT_TYPE)
    members["../escaped.py"] = b"raise AssertionError('must never execute')"
    with pytest.raises(BatchError, match="WORKSPACE_INTEGRITY_FAILED"):
        load_batch_workspace(make_archive(members, ARTIFACT_TYPE))
    monkeypatch.setattr("ftir_workbench.batch.workspace.MAX_MEMBERS", 2)
    with pytest.raises(BatchError, match="WORKSPACE_INTEGRITY_FAILED"):
        load_batch_workspace(payload)
