from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest

import ftir_workbench.batch.service as service
import ftir_workbench.batch.workspace as workspace_module
from ftir_baseline.config import CoarseBaselineConfig
from ftir_workbench.batch.export import build_batch_export
from ftir_workbench.batch.fingerprints import stage_fingerprint
from ftir_workbench.batch.models import BatchError, PreparationConfig
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

from .test_export import ready_pair
from .test_state_and_stages import workspace_pair


def rewrite(payload: bytes, callback):  # type: ignore[no-untyped-def]
    members = read_verified_archive(payload, "independent_baseline_workspace")
    data = json.loads(members["workspace.json"])
    callback(data, members)
    members["workspace.json"] = json.dumps(data).encode()
    return make_archive(members, "independent_baseline_workspace")


def test_e07_full_workspace_restores_arrays_identity_drafts_parent_and_no_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    ws, a, b = ready_pair()
    ws.display_order.reverse()
    ws.selected_spectrum_id = b
    set_coarse_draft(ws, a, CoarseBaselineConfig(method="linear"))
    ws.states[a].display_preferences = {"error_fine": "unfinished anchors", "editor_fine": {**ws.states[a].fine_draft.to_dict(), "anchors": [{"start": None, "end": float("nan")}]}}
    ws.export_history.append({"stage": "coarse", "exported_spectrum_ids": [a]})
    saved = save_batch_workspace(ws)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("workspace restore must not call pipeline")

    monkeypatch.setattr(service, "run_pipeline", forbidden)
    restored = load_batch_workspace(saved)
    assert restored.workspace_id == ws.workspace_id
    assert restored.display_order == [b, a]
    assert restored.selected_spectrum_id == b
    assert restored.export_history == ws.export_history
    assert set(restored.sources) == set(ws.sources)
    for source_id, source in restored.sources.items():
        assert source.original_bytes == ws.sources[source_id].original_bytes
        assert source.original_filename == ws.sources[source_id].original_filename
        assert source.original_bytes_sha256 == ws.sources[source_id].original_bytes_sha256
    for sid in (a, b):
        record, state = restored.records[sid], restored.states[sid]
        np.testing.assert_array_equal(record.wavenumber, ws.records[sid].wavenumber)
        np.testing.assert_array_equal(record.raw_intensity, ws.records[sid].raw_intensity)
        assert record.spectrum_id == sid and record.source_id == ws.records[sid].source_id
        assert state.coarse_draft == ws.states[sid].coarse_draft
        assert state.fine_draft == ws.states[sid].fine_draft
        assert state.coarse_preview is None and state.fine_preview is None
        for stage in ("coarse", "fine"):
            got, original = getattr(state, f"{stage}_snapshot"), getattr(ws.states[sid], f"{stage}_snapshot")
            assert got.fingerprint == original.fingerprint
            assert got.parent_coarse_fingerprint == original.parent_coarse_fingerprint
            assert got.result.recipe_dict() == original.result.recipe_dict()
            for name in ("coarse_baseline", "fine_baseline", "total_baseline", "corrected"):
                np.testing.assert_array_equal(getattr(got.result.baseline, name), getattr(original.result.baseline, name))
            np.testing.assert_array_equal(got.result.baseline_estimation_spectra, original.result.baseline_estimation_spectra)
            assert not got.result.analysis_data.flags.writeable
            for metric, values in got.result.qc.per_spectrum.items():
                np.testing.assert_array_equal(values, original.result.qc.per_spectrum[metric])
    assert restored.states[a].display_preferences["editor_fine"]["anchors"][0] == {"start": None, "end": None}
    assert restored.states[a].display_preferences["error_fine"] == "unfinished anchors"
    assert build_batch_export(restored, [a, b], stage="fine").metadata["exported_spectrum_ids"] == [a, b]


@pytest.mark.parametrize("change", ["coarse", "unit", "range", "smoothing"])
def test_stale_historical_results_survive_but_cannot_export(change: str) -> None:
    ws, a, b = ready_pair()
    old_fine = ws.states[a].fine_snapshot
    if change == "coarse":
        set_coarse_draft(ws, a, CoarseBaselineConfig(method="linear"))
        preview_coarse(ws, a)
        confirm_coarse(ws, a)
    else:
        data = ws.states[a].preparation_draft.to_dict()
        if change == "unit":
            data["input_unit"] = "fraction_transmittance"
        elif change == "range":
            data["wavenumber_range"] = [1700, 1000]
        else:
            data["baseline_smoothing"] = {"enabled": True}
        set_preparation_draft(ws, a, PreparationConfig(**data))
        confirm_preparation(ws, a)
    restored = load_batch_workspace(save_batch_workspace(ws))
    assert restored.states[a].fine_stale
    assert restored.states[a].fine_snapshot.fingerprint == old_fine.fingerprint
    with pytest.raises(BatchError, match="STALE"):
        build_batch_export(restored, [a, b], stage="fine")
    assert build_batch_export(restored, [a, b], stage="fine", ready_only=True).metadata["exported_spectrum_ids"] == [b]


def test_skip_and_undecided_survive_roundtrip() -> None:
    ws, a, b = workspace_pair()
    for sid in (a, b):
        preview_coarse(ws, sid)
        confirm_coarse(ws, sid)
    skip_fine(ws, a)
    restored = load_batch_workspace(save_batch_workspace(ws))
    assert restored.states[a].fine_decision == "explicitly_skipped"
    assert restored.states[b].fine_decision == "not_decided"
    assert build_batch_export(restored, [a, b], stage="fine", ready_only=True).metadata["exported_spectrum_ids"] == [a]


def test_old_implementation_is_verified_as_saved_but_clearly_marked() -> None:
    ws, a, _ = workspace_pair()
    preview_coarse(ws, a)
    snapshot = confirm_coarse(ws, a)
    old = "f" * 64
    ws.states[a].coarse_snapshot = replace(snapshot, implementation_fingerprint=old,
        fingerprint=stage_fingerprint(ws, a, snapshot.config, "coarse", implementation=old))
    restored = load_batch_workspace(save_batch_workspace(ws))
    assert restored.states[a].coarse_snapshot.implementation_fingerprint == old
    assert any("not recomputed" in warning for warning in restored.states[a].warnings)


@pytest.mark.parametrize("mutation", ["source_hash", "column", "raw", "parent", "recipe", "component", "order", "extra", "wrong_type"])
def test_e08_rehashed_but_inconsistent_contract_is_rejected(mutation: str) -> None:
    ws, a, _ = ready_pair()
    payload = save_batch_workspace(ws)

    def corrupt(data, members):  # type: ignore[no-untyped-def]
        if mutation == "source_hash":
            source = next(iter(data["sources"].values()))
            members[source["bytes_member"]] += b"\n"
        elif mutation == "column":
            data["records"][a]["original_column_index"] = 2
        elif mutation == "raw":
            data["records"][a]["raw_intensity"] = data["records"][a]["wavenumber"]
        elif mutation == "parent":
            data["states"][a]["fine_snapshot"]["parent_coarse_fingerprint"] = "0" * 64
        elif mutation == "recipe":
            data["states"][a]["fine_snapshot"]["result"]["recipe"]["config"]["coarse_baseline"]["method"] = "none"
        elif mutation == "component":
            baseline = data["states"][a]["fine_snapshot"]["result"]["baseline"]
            baseline["total_baseline"] = baseline["corrected"]
        elif mutation == "order":
            data["display_order"] = [a, a]
        elif mutation == "extra":
            members["unexpected.py"] = b"raise RuntimeError('must not execute')"
        else:
            data["artifact_type"] = "independent_baseline_batch"
    with pytest.raises(BatchError, match="WORKSPACE_INTEGRITY_FAILED"):
        load_batch_workspace(rewrite(payload, corrupt))


@pytest.mark.parametrize("name", ["../outside", "/absolute", "bad\\path", "C:outside", "a/./b"])
def test_e08_archive_path_traversal_rejected(name: str) -> None:
    payload = make_archive({"workspace.json": b"{}", name: b"no execution"}, "independent_baseline_workspace")
    with pytest.raises(BatchError, match="unsafe archive"):
        load_batch_workspace(payload)


def test_e08_duplicate_members_size_limits_and_hash_tampering(monkeypatch: pytest.MonkeyPatch) -> None:
    ws, _, _ = workspace_pair()
    payload = save_batch_workspace(ws)
    stream = io.BytesIO(payload)
    with zipfile.ZipFile(stream, "a") as archive, pytest.warns(UserWarning):
        archive.writestr("workspace.json", b"{}")
    with pytest.raises(BatchError, match="duplicate"):
        load_batch_workspace(stream.getvalue())
    monkeypatch.setattr(workspace_module, "MAX_TOTAL_BYTES", 1)
    with pytest.raises(BatchError, match="expanded size"):
        load_batch_workspace(payload)
    monkeypatch.setattr(workspace_module, "MAX_TOTAL_BYTES", 512 * 1024 * 1024)
    stream = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(payload)) as original, zipfile.ZipFile(stream, "w") as target:
        for name in original.namelist():
            target.writestr(name, original.read(name) + (b" " if name == "workspace.json" else b""))
    with pytest.raises(BatchError, match="SHA-256"):
        load_batch_workspace(stream.getvalue())


def test_e08_object_npy_pickle_is_rejected_without_execution() -> None:
    ws, a, _ = workspace_pair()
    payload = save_batch_workspace(ws)

    def corrupt(data, members):  # type: ignore[no-untyped-def]
        stream = io.BytesIO()
        np.save(stream, np.array([{"not": "numeric"}], dtype=object), allow_pickle=True)
        members[data["records"][a]["wavenumber"]["__array__"]] = stream.getvalue()
    with pytest.raises(BatchError, match="NPY dtype"):
        load_batch_workspace(rewrite(payload, corrupt))


@pytest.mark.parametrize("smoothing", [False, True])
def test_mixed_units_axes_full_source_and_cropped_results_roundtrip(smoothing: bool) -> None:
    from ftir_workbench.batch.importing import import_sources
    from ftir_workbench.batch.models import BatchWorkspace
    from ftir_workbench.batch.service import preview_fine
    from ftir_workbench.batch.state import confirm_fine

    from .test_importing import table_bytes

    ws = BatchWorkspace()
    ids = []
    for index, unit in enumerate(("absorbance", "fraction_transmittance", "percent_transmittance")):
        x = np.linspace(900, 1800, 181 + index * 10)
        if index == 1:
            x = x[::-1]
        absorbance = 0.05 + x / 10000 + 0.2 * np.exp(-((x - 1300) / 25) ** 2)
        y = absorbance if unit == "absorbance" else np.power(10.0, -absorbance)
        if unit == "percent_transmittance":
            y = y * 100
        sid = import_sources(ws, [("同名.csv", table_bytes(x, y))], input_unit=unit)[0]
        ids.append(sid)
        prep = ws.states[sid].preparation_draft.to_dict()
        prep.update(wavenumber_range=[1700, 1000], baseline_smoothing={"enabled": smoothing})
        set_preparation_draft(ws, sid, PreparationConfig(**prep))
        confirm_preparation(ws, sid)
        preview_coarse(ws, sid)
        confirm_coarse(ws, sid)
        preview_fine(ws, sid)
        confirm_fine(ws, sid)
    restored = load_batch_workspace(save_batch_workspace(ws))
    assert len(restored.sources) == 3
    assert len({item.original_filename for item in restored.sources.values()}) == 1
    artifact = build_batch_export(restored, ids, stage="fine")
    assert artifact.wide_unavailable_reason
    for sid in ids:
        np.testing.assert_array_equal(restored.records[sid].wavenumber, ws.records[sid].wavenumber)
        original, saved = ws.states[sid].fine_snapshot.result, restored.states[sid].fine_snapshot.result
        assert len(restored.records[sid].wavenumber) > saved.absorbance_selected.n_points
        np.testing.assert_array_equal(saved.analysis_data, original.analysis_data)
        np.testing.assert_array_equal(saved.baseline_estimation_spectra, original.baseline_estimation_spectra)
        assert saved.unit_conversion == original.unit_conversion


@pytest.mark.parametrize("mutation", ["probe", "empty_id", "singleton", "view", "shape"])
def test_additional_saved_contract_corruption_is_rejected(mutation: str) -> None:
    ws, a, _ = ready_pair()

    def corrupt(data, members):  # type: ignore[no-untyped-def]
        if mutation == "probe":
            source = next(iter(data["sources"].values()))
            source["import_probe"]["columns"] = 99
        elif mutation == "empty_id":
            data["workspace_id"] = ""
        elif mutation == "singleton":
            data["states"][a]["fine_snapshot"]["result"]["raw_input"]["perturbation_labels"] = ["experimental time"]
        elif mutation == "view":
            result = data["states"][a]["fine_snapshot"]["result"]
            result["normalization"]["view_data"] = result["absorbance_selected"]["spectra"]
        else:
            name = data["records"][a]["wavenumber"]["__array__"]
            stream = io.BytesIO()
            np.lib.format.write_array_header_1_0(stream, {"descr": "<f8", "fortran_order": False, "shape": (10**12,)})
            members[name] = stream.getvalue()
    with pytest.raises(BatchError, match="WORKSPACE_INTEGRITY_FAILED"):
        load_batch_workspace(rewrite(save_batch_workspace(ws), corrupt))


def test_symlink_and_deep_json_are_rejected_cleanly() -> None:
    import stat

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        info = zipfile.ZipInfo("symbolic.txt")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "outside")
    with pytest.raises(BatchError, match="unsafe archive"):
        load_batch_workspace(stream.getvalue())
    payload = make_archive({"workspace.json": b"[" * 10000 + b"]" * 10000}, "independent_baseline_workspace")
    with pytest.raises(BatchError, match="WORKSPACE_INTEGRITY_FAILED"):
        load_batch_workspace(payload)


def test_old_coarse_implementation_new_fine_then_replaced_coarse_retains_historical_parent() -> None:
    from ftir_workbench.batch.fingerprints import implementation_fingerprint
    from ftir_workbench.batch.service import preview_fine
    from ftir_workbench.batch.state import confirm_fine

    ws, a, _ = workspace_pair()
    preview_coarse(ws, a)
    original_coarse = confirm_coarse(ws, a)
    old = "b" * 64
    old_fingerprint = stage_fingerprint(ws, a, original_coarse.config, "coarse", implementation=old)
    ws.states[a].coarse_snapshot = replace(original_coarse,
        implementation_fingerprint=old, fingerprint=old_fingerprint)
    restored = load_batch_workspace(save_batch_workspace(ws))
    preview_fine(restored, a)
    fine = confirm_fine(restored, a)
    assert fine.implementation_fingerprint == implementation_fingerprint()
    assert fine.parent_coarse_implementation_fingerprint == old
    assert fine.parent_coarse_fingerprint == old_fingerprint
    current = load_batch_workspace(save_batch_workspace(restored))
    assert build_batch_export(current, [a], stage="fine").metadata["exported_spectrum_ids"] == [a]
    set_coarse_draft(current, a, CoarseBaselineConfig(method="linear"))
    preview_coarse(current, a)
    confirm_coarse(current, a)
    stale = load_batch_workspace(save_batch_workspace(current))
    assert stale.states[a].fine_stale
    assert stale.states[a].fine_snapshot.parent_coarse_implementation_fingerprint == old
    assert stale.states[a].fine_snapshot.parent_coarse_fingerprint == old_fingerprint
    with pytest.raises(BatchError, match="FINE_STALE"):
        build_batch_export(stale, [a], stage="fine")


@pytest.mark.parametrize("json_payload", [b'{"value":NaN}', b'{"value":1,"value":2}'])
def test_nonstandard_json_or_duplicate_keys_are_rejected(json_payload: bytes) -> None:
    payload = make_archive({"workspace.json": json_payload}, "independent_baseline_workspace")
    with pytest.raises(BatchError, match="WORKSPACE_INTEGRITY_FAILED"):
        load_batch_workspace(payload)


@pytest.mark.parametrize("mutation", [
    "sources_list", "records_list", "states_list", "display_order_string", "issues_none",
    "history_dict", "summary_list", "selected_int", "source_list", "source_name_list",
    "source_options_list", "record_list", "record_label_none", "record_column_bool",
    "excluded_string", "duplicate_int", "state_list", "prep_draft_none", "prep_committed_none",
    "coarse_draft_none", "fine_draft_none", "coarse_committed_list", "coarse_stale_string",
    "fine_stale_none", "errors_none", "warnings_none", "errors_object", "fine_decision_list",
    "preferences_none", "editor_none", "editor_missing_fields", "editor_wrong_scalar",
    "editor_choice_unknown", "editor_anchors_string", "editor_anchor_scalar",
    "editor_numeric_codec", "snapshot_impl_list", "result_version_none",
])
def test_rehashed_malformed_business_schema_is_rejected_before_workspace_install(mutation: str) -> None:
    ws, a, _ = ready_pair()

    def corrupt(data, members):  # type: ignore[no-untyped-def]
        source_key = next(iter(data["sources"]))
        state = data["states"][a]
        state["display_preferences"] = {"editor_fine": dict(state["fine_draft"])}
        options = {
            "sources_list": (data, "sources", []),
            "records_list": (data, "records", []),
            "states_list": (data, "states", []),
            "display_order_string": (data, "display_order", a),
            "issues_none": (data, "import_issues", None),
            "history_dict": (data, "export_history", {}),
            "summary_list": (data, "last_export_summary", []),
            "selected_int": (data, "selected_spectrum_id", 12),
            "source_list": (data["sources"], source_key, []),
            "source_name_list": (data["sources"][source_key], "original_filename", []),
            "source_options_list": (data["sources"][source_key], "import_options", []),
            "record_list": (data["records"], a, []),
            "record_label_none": (data["records"][a], "display_name", None),
            "record_column_bool": (data["records"][a], "original_column_index", True),
            "excluded_string": (data["records"][a], "excluded", "false"),
            "duplicate_int": (data["records"][a], "duplicate_candidate", 0),
            "state_list": (data["states"], a, []),
            "prep_draft_none": (state, "preparation_draft", None),
            "prep_committed_none": (state, "preparation_committed", None),
            "coarse_draft_none": (state, "coarse_draft", None),
            "fine_draft_none": (state, "fine_draft", None),
            "coarse_committed_list": (state, "coarse_committed", []),
            "coarse_stale_string": (state, "coarse_stale", "false"),
            "fine_stale_none": (state, "fine_stale", None),
            "errors_none": (state, "errors", None),
            "warnings_none": (state, "warnings", None),
            "errors_object": (state, "errors", [{}]),
            "fine_decision_list": (state, "fine_decision", []),
            "preferences_none": (state, "display_preferences", None),
            "editor_none": (state["display_preferences"], "editor_fine", None),
            "editor_missing_fields": (state["display_preferences"], "editor_fine", {}),
            "editor_wrong_scalar": (state["display_preferences"]["editor_fine"], "polynomial_order", {}),
            "editor_choice_unknown": (state["display_preferences"]["editor_fine"], "method", "not-a-method"),
            "editor_anchors_string": (state["display_preferences"]["editor_fine"], "anchors", "invalid"),
            "editor_anchor_scalar": (state["display_preferences"]["editor_fine"], "anchors", [{"start": []}]),
            "editor_numeric_codec": (state["display_preferences"], "unknown", data["records"][a]["wavenumber"]),
            "snapshot_impl_list": (state["fine_snapshot"], "implementation_fingerprint", []),
            "result_version_none": (state["fine_snapshot"]["result"], "software_version", None),
        }
        target, field, replacement = options[mutation]
        target[field] = replacement
    with pytest.raises(BatchError, match="WORKSPACE_INTEGRITY_FAILED"):
        load_batch_workspace(rewrite(save_batch_workspace(ws), corrupt))


def test_repeated_array_references_obey_decoded_memory_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    from ftir_workbench.batch.workspace import _Arrays

    stream = io.BytesIO()
    np.save(stream, np.ones(20, dtype=np.float64), allow_pickle=False)
    codec = _Arrays({"arrays/reused.npy": stream.getvalue()})
    monkeypatch.setattr(workspace_module, "MAX_TOTAL_BYTES", 200)
    with pytest.raises(BatchError, match="decoded array references exceed"):
        codec.decode([{"__array__": "arrays/reused.npy"}] * 2)
