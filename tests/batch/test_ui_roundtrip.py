"""AppTest download/upload round trips using Streamlit's real media storage."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass

import numpy as np
import pytest
from streamlit.runtime.media_file_storage import MediaFileKind
from streamlit.runtime.memory_media_file_storage import MemoryMediaFileStorage
from streamlit.testing.v1 import AppTest

from ftir_workbench.batch.models import BatchWorkspace
from tests.batch.test_ui import (
    assert_ok,
    batch_app,
    coarse_confirm,
    element,
    page,
    select_spectrum,
    uploaded_app,
    workspace,
)
from tests.batch.test_ui import pipeline_calls as pipeline_calls


@dataclass(frozen=True)
class DownloadedFile:
    filename: str
    content: bytes
    mimetype: str


@pytest.fixture
def downloaded_files(monkeypatch: pytest.MonkeyPatch) -> dict[str, DownloadedFile]:
    """Observe bytes registered by actual download widgets, without replacing exports.

    AppTest 1.62 creates a real MemoryMediaFileStorage on each rerun but discards
    its Runtime afterwards. Preserve the output of that real storage call so a
    download button's actual URL can be resolved and uploaded into a new app.
    """

    downloads: dict[str, DownloadedFile] = {}
    original = MemoryMediaFileStorage.load_and_get_id

    def observe(storage: MemoryMediaFileStorage, path_or_data: str | bytes,
                mimetype: str, kind: MediaFileKind, filename: str | None = None) -> str:
        file_id = original(storage, path_or_data, mimetype, kind, filename)
        actual_file = storage.get_file(file_id)
        if kind == MediaFileKind.DOWNLOADABLE:
            downloads[storage.get_url(file_id)] = DownloadedFile(
                filename=actual_file.filename or "download", content=actual_file.content,
                mimetype=actual_file.mimetype,
            )
        return file_id

    monkeypatch.setattr(MemoryMediaFileStorage, "load_and_get_id", observe)
    return downloads


def click_download(app: AppTest, label: str,
                   downloads: dict[str, DownloadedFile]) -> DownloadedFile:
    button = element(app.get("download_button"), label)
    actual_file = downloads[button.proto.url]
    button.click().run()
    assert_ok(app)
    assert actual_file.content
    return actual_file


def assert_restored(before: BatchWorkspace, after: BatchWorkspace) -> None:
    assert after is not before
    assert after.workspace_id == before.workspace_id
    assert after.workflow_mode == before.workflow_mode == "independent_batch"
    assert after.display_order == before.display_order
    assert after.selected_spectrum_id == before.selected_spectrum_id
    assert set(after.sources) == set(before.sources)
    assert set(after.records) == set(before.records)
    for source_id, source in before.sources.items():
        restored_source = after.sources[source_id]
        assert restored_source.original_bytes == source.original_bytes
        assert restored_source.original_bytes_sha256 == source.original_bytes_sha256
        assert restored_source.original_filename == source.original_filename
        assert restored_source.default_input_unit == source.default_input_unit
        assert restored_source.import_options == source.import_options
        assert restored_source.import_probe == source.import_probe
    for sid, record in before.records.items():
        restored_record = after.records[sid]
        for name in ("spectrum_id", "source_id", "original_column_index", "original_column_label",
                     "display_name", "confirmed_input_unit", "scientific_input_sha256",
                     "original_axis_direction", "excluded", "duplicate_candidate"):
            assert getattr(restored_record, name) == getattr(record, name)
        np.testing.assert_array_equal(restored_record.wavenumber, record.wavenumber)
        np.testing.assert_array_equal(restored_record.raw_intensity, record.raw_intensity)
        assert not restored_record.wavenumber.flags.writeable
        assert not restored_record.raw_intensity.flags.writeable
        state, restored_state = before.states[sid], after.states[sid]
        for name in ("preparation_draft", "preparation_committed", "coarse_draft", "coarse_committed",
                     "fine_draft", "fine_committed", "fine_decision", "coarse_stale", "fine_stale",
                     "fine_parent_coarse_fingerprint", "display_preferences"):
            assert getattr(restored_state, name) == getattr(state, name)
        for stage in ("coarse", "fine"):
            snapshot = getattr(state, stage + "_snapshot")
            restored_snapshot = getattr(restored_state, stage + "_snapshot")
            if snapshot is None:
                assert restored_snapshot is None
                continue
            assert restored_snapshot is not None
            for name in ("stage", "spectrum_id", "input_sha256", "config", "fingerprint",
                         "parent_coarse_fingerprint", "implementation_fingerprint"):
                assert getattr(restored_snapshot, name) == getattr(snapshot, name)
            for name in ("coarse_baseline", "fine_baseline", "total_baseline", "corrected"):
                actual = getattr(restored_snapshot.result.baseline, name)
                np.testing.assert_array_equal(actual, getattr(snapshot.result.baseline, name))
                assert not actual.flags.writeable
            np.testing.assert_array_equal(restored_snapshot.result.analysis_data,
                                          snapshot.result.analysis_data)
            np.testing.assert_array_equal(restored_snapshot.result.baseline_estimation_spectra,
                                          snapshot.result.baseline_estimation_spectra)
            assert restored_snapshot.result.raw_input.metadata == snapshot.result.raw_input.metadata


def test_confirm_and_next_visits_all_three_records_before_returning_to_a(
    pipeline_calls: list[str],
) -> None:
    app = uploaded_app()
    ws = workspace(app)
    a, b, c = ws.display_order
    page(app, 3)
    saved = {}
    for current, following, method in ((a, b, "linear"), (b, c, "offset"), (c, a, "none")):
        assert ws.selected_spectrum_id == current
        assert element(app.sidebar.selectbox, "当前光谱").value == current
        element(app.selectbox, "粗调方法").set_value(method).run()
        element(app.button, "计算粗调预览").click().run()
        assert_ok(app)
        preview = ws.states[current].coarse_preview
        assert preview is not None
        saved[current] = preview
        element(app.button, "确认粗调并下一条").click().run()
        assert_ok(app)
        assert ws.selected_spectrum_id == following
        assert element(app.sidebar.selectbox, "当前光谱").value == following
        assert ws.states[current].coarse_snapshot is preview
    assert pipeline_calls == [a, b, c]
    assert element(app.selectbox, "粗调方法").value == "linear"
    for sid, snapshot in saved.items():
        assert ws.states[sid].coarse_snapshot is snapshot
        np.testing.assert_array_equal(ws.states[sid].coarse_snapshot.result.analysis_data,
                                      snapshot.result.analysis_data)


def test_invalid_target_editor_is_reported_and_never_fits_its_last_valid_draft(
    pipeline_calls: list[str],
) -> None:
    app = uploaded_app()
    ws = workspace(app)
    a, b, _ = ws.display_order
    coarse_confirm(app)
    select_spectrum(app, b)
    coarse_confirm(app)
    select_spectrum(app, a)
    page(app, 4)
    element(app.selectbox, "细调方法").set_value("pchip").run()
    element(app.button, "预填覆盖当前范围的锚点").click().run()
    assert_ok(app)
    valid_draft = ws.states[a].fine_draft
    editor = next(item for item in app.get("dataframe")
                  if item.key and item.key.endswith("_fine_anchors"))
    # AppTest has no cell-edit method in 1.62. Feed the real data_editor session
    # delta schema (edited_rows/added_rows/deleted_rows), verified in installed
    # streamlit.elements.widgets.data_editor.DataEditorState, then rerun the UI.
    app.session_state[editor.key] = {
        "edited_rows": {0: {"start": float(editor.value.iloc[0]["end"])}},
        "added_rows": [], "deleted_rows": [],
    }
    app.run()
    assert not app.exception
    assert any("anchor window start and end must differ" in item.value for item in app.error)
    assert ws.states[a].fine_draft == valid_draft
    assert ws.states[a].display_preferences["error_fine"]
    select_spectrum(app, b)
    element(app.sidebar.multiselect, "批量勾选（独立于当前光谱）").set_value([a, b]).run()
    element(app.button, "为选中项计算各自细调预览").click().run()
    assert_ok(app)
    assert pipeline_calls == [a, b, b]
    assert ws.states[a].fine_preview is None
    assert ws.states[b].fine_preview is not None
    assert "草稿无效，未运行" in app.session_state["batch_operation_report"][a]
    assert app.session_state["batch_operation_report"][b] is None


def assert_export_arrays(payload: bytes, ws: BatchWorkspace, stage: str,
                         ids: list[str]) -> dict:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        metadata = json.loads(archive.read("batch_metadata.json"))
        assert metadata["workflow_mode"] == "independent_batch"
        assert metadata["is_2d_ready"] is False
        assert metadata["stage"] == stage
        assert metadata["exported_spectrum_ids"] == ids
        assert not any("for_2dcos" in name or "prepared" in name for name in archive.namelist())
        for entry in metadata["spectra"]:
            sid = entry["spectrum_id"]
            snapshot = getattr(ws.states[sid], stage + "_snapshot")
            values = np.loadtxt(io.BytesIO(archive.read(entry["spectra_csv"])), delimiter=",", skiprows=1)
            np.testing.assert_array_equal(values[:, 0], snapshot.result.absorbance_selected.wavenumber)
            np.testing.assert_array_equal(values[:, 1], snapshot.result.analysis_data[0])
            assert entry["stage_fingerprint"] == snapshot.fingerprint
            assert entry["source_id"] == ws.records[sid].source_id
            assert entry["source_column_index"] == ws.records[sid].original_column_index
        return metadata


def test_actual_upload_coarse_fine_download_workspace_restore_and_stale_export_loop(
    pipeline_calls: list[str], downloaded_files: dict[str, DownloadedFile],
) -> None:
    app = uploaded_app()
    ws = workspace(app)
    a, b, c = ws.display_order
    page(app, 2)
    element(app.checkbox, "仅平滑基线估计通道").set_value(True).run()
    element(app.button, "确认此谱单位与处理范围").click().run()
    assert_ok(app)
    for sid, method in ((a, "linear"), (b, "offset"), (c, "none")):
        select_spectrum(app, sid)
        coarse_confirm(app, method)
    select_spectrum(app, a)
    page(app, 4)
    element(app.button, "计算细调预览").click().run()
    assert_ok(app)
    element(app.button, "确认当前细调").click().run()
    assert_ok(app)
    select_spectrum(app, b)
    element(app.button, "确认不做细调").click().run()
    assert_ok(app)
    select_spectrum(app, a)
    # Store uncommitted edits alongside older valid formal snapshots.
    element(app.number_input, "端点窗口宽度 cm⁻¹").set_value(35.0).run()
    page(app, 3)
    element(app.selectbox, "粗调方法").set_value("offset").run()
    assert ws.states[a].fine_decision == "applied"
    assert not ws.states[a].fine_stale
    before_download_calls = list(pipeline_calls)
    assert before_download_calls == [a, b, c, a]

    page(app, 5)
    element(app.selectbox, "粗调导出范围").set_value("全部条目").run()
    wide_coarse = next(item for item in app.checkbox if item.key == "export_wide_coarse")
    assert wide_coarse.disabled
    element(app.button, "打包已确认粗调结果").click().run()
    assert_ok(app)
    coarse_download = click_download(app, "下载已确认粗调 ZIP", downloaded_files)
    coarse_metadata = assert_export_arrays(coarse_download.content, ws, "coarse", [a, b, c])
    assert coarse_metadata["wide_available"] is False

    element(app.selectbox, "最终导出范围").set_value("全部条目").run()
    next(item for item in app.checkbox if item.key == "export_ready_fine").set_value(True).run()
    element(app.button, "打包已确认最终结果").click().run()
    assert_ok(app)
    fine_download = click_download(app, "下载已确认最终 ZIP", downloaded_files)
    fine_metadata = assert_export_arrays(fine_download.content, ws, "fine", [a, b])
    assert fine_metadata["excluded_items"][0]["spectrum_id"] == c
    assert fine_metadata["excluded_items"][0]["reason_code"] == "FINE_REQUIRED"
    assert [entry["fine_decision"] for entry in fine_metadata["spectra"]] == [
        "applied", "explicitly_skipped"]
    with zipfile.ZipFile(io.BytesIO(fine_download.content)) as archive:
        assert "True" in archive.read("processing_report.csv").decode()

    element(app.button, "准备工作区下载").click().run()
    assert_ok(app)
    saved_workspace = click_download(app, "保存工作区 ZIP", downloaded_files)
    assert pipeline_calls == before_download_calls

    restored_app = batch_app()
    element(restored_app.get("file_uploader"), "恢复普通工作区 ZIP").set_value(
        (saved_workspace.filename, saved_workspace.content, saved_workspace.mimetype),
    ).run()
    element(restored_app.button, "恢复工作区").click().run()
    assert_ok(restored_app)
    restored = workspace(restored_app)
    assert_restored(ws, restored)
    assert pipeline_calls == before_download_calls
    page(restored_app, 3)
    assert element(restored_app.selectbox, "粗调方法").value == "offset"
    page(restored_app, 4)
    assert element(restored_app.number_input, "端点窗口宽度 cm⁻¹").value == 35.0
    assert pipeline_calls == before_download_calls

    # Confirming changed coarse makes only A's old fine stale. Export excludes A.
    coarse_confirm(restored_app, "offset")
    assert restored.states[a].fine_stale
    assert not restored.states[b].fine_stale
    after_new_coarse_calls = list(pipeline_calls)
    assert after_new_coarse_calls == [a, b, c, a, a]
    page(restored_app, 5)
    element(restored_app.selectbox, "最终导出范围").set_value("全部条目").run()
    next(item for item in restored_app.checkbox if item.key == "export_ready_fine").set_value(True).run()
    element(restored_app.button, "打包已确认最终结果").click().run()
    assert_ok(restored_app)
    stale_filtered = click_download(restored_app, "下载已确认最终 ZIP", downloaded_files)
    final_metadata = assert_export_arrays(stale_filtered.content, restored, "fine", [b])
    reasons = {row["spectrum_id"]: row["reason_code"] for row in final_metadata["excluded_items"]}
    assert reasons == {a: "FINE_STALE", c: "FINE_REQUIRED"}
    assert pipeline_calls == after_new_coarse_calls


def test_same_axis_wide_download_uses_confirmed_arrays_and_disables_for_heterogeneous_scope(
    pipeline_calls: list[str], downloaded_files: dict[str, DownloadedFile],
) -> None:
    app = uploaded_app()
    ws = workspace(app)
    a, b, c = ws.display_order
    page(app, 3)
    for sid in (a, b, c):
        select_spectrum(app, sid)
        coarse_confirm(app, "none")
    page(app, 5)
    element(app.sidebar.multiselect, "批量勾选（独立于当前光谱）").set_value([a, b]).run()
    element(app.selectbox, "粗调导出范围").set_value("勾选条目").run()
    wide = next(item for item in app.checkbox if item.key == "export_wide_coarse")
    assert not wide.disabled
    wide.set_value(True).run()
    element(app.button, "打包已确认粗调结果").click().run()
    assert_ok(app)
    downloaded = click_download(app, "下载粗调同轴宽表 CSV", downloaded_files)
    values = np.loadtxt(io.BytesIO(downloaded.content), delimiter=",", skiprows=1)
    assert values.shape == (181, 3)
    np.testing.assert_array_equal(values[:, 0], ws.states[a].coarse_snapshot.result.absorbance_selected.wavenumber)
    np.testing.assert_array_equal(values[:, 1], ws.states[a].coarse_snapshot.result.analysis_data[0])
    np.testing.assert_array_equal(values[:, 2], ws.states[b].coarse_snapshot.result.analysis_data[0])
    assert pipeline_calls == [a, b, c]
    element(app.selectbox, "粗调导出范围").set_value("全部条目").run()
    assert_ok(app)
    assert next(item for item in app.checkbox if item.key == "export_wide_coarse").disabled
    assert not any(item.label == "下载粗调同轴宽表 CSV" for item in app.get("download_button"))
    element(app.button, "打包已确认粗调结果").click().run()
    assert_ok(app)
    zipped = click_download(app, "下载已确认粗调 ZIP", downloaded_files)
    assert_export_arrays(zipped.content, ws, "coarse", [a, b, c])
    with zipfile.ZipFile(io.BytesIO(zipped.content)) as archive:
        assert not any(name.endswith("_wide.csv") for name in archive.namelist())
    assert pipeline_calls == [a, b, c]
