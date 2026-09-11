"""Real media downloads and workspace uploads for ordinary derived branches."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ftir_workbench.batch import postprocessing as pp
from tests.batch.test_ui import (
    APP_PATH,
    assert_ok,
    batch_app,
    coarse_confirm,
    element,
    page,
    select_spectrum,
    uploaded_app,
    workspace,
)
from tests.batch.test_ui_roundtrip import DownloadedFile, click_download
from tests.batch.test_ui_roundtrip import downloaded_files as downloaded_files
from tests.postprocessing.helpers import confirmed_workspace
from tests.postprocessing.test_ui_postprocessing import control, post_page
from tests.postprocessing.test_ui_postprocessing import science_calls as science_calls


def _smooth(app: AppTest, method: str = "savgol") -> None:
    element(app.checkbox, "启用普通谱平滑").set_value(True).run()
    element(app.selectbox, "普通谱平滑方法").set_value(method).run()
    element(app.button, "计算当前平滑预览").click().run()
    element(app.button, "确认当前平滑").click().run()
    assert_ok(app)


def _normalize(app: AppTest, method: str, target: float = 1.0) -> None:
    element(app.checkbox, "启用当前来源归一化").set_value(True).run()
    element(app.selectbox, "普通谱归一化方法").set_value(method).run()
    if method != "minmax_display":
        element(app.number_input, "归一化目标值").set_value(target).run()
    element(app.button, "计算当前归一化预览").click().run()
    element(app.button, "确认当前归一化").click().run()
    assert_ok(app)


def test_real_uploaded_workspace_restores_selection_sources_drafts_parents_and_no_previews(
    downloaded_files: dict[str, DownloadedFile], science_calls: dict[str, int],
) -> None:
    app = uploaded_app()
    ws = workspace(app)
    a, b, c = ws.display_order
    page(app, 3)
    for sid in (a, b, c):
        select_spectrum(app, sid)
        coarse_confirm(app, "none")
        page(app, 4)
        element(app.button, "确认不做细调").click().run()
        assert_ok(app)
    select_spectrum(app, a)
    post_page(app)
    _smooth(app)
    _normalize(app, "maximum")
    element(app.button, "使用已确认平滑谱 S").click().run()
    _normalize(app, "area", target=2.0)
    element(app.number_input, "归一化目标值").set_value(3.0).run()
    # A second confirmed S supersedes S without deleting the old N_S parent.
    control(app, "number_input", "smoothed", "savgol_window_length").set_value(9).run()
    element(app.button, "计算当前平滑预览").click().run()
    element(app.button, "确认当前平滑").click().run()
    assert_ok(app)
    assert pp.branch_status(ws, a, "normalized_smoothed")["status"] == "stale"
    select_spectrum(app, b)
    _smooth(app, "gaussian")
    _normalize(app, "minmax_display")
    select_spectrum(app, a)
    element(app.sidebar.multiselect, "批量勾选（独立于当前光谱）").set_value([a, b]).run()
    assert ws.selected_spectrum_ids == [a, b]
    before = dict(science_calls)
    element(app.selectbox, "后处理导出分支").set_value("normalized_baseline").run()
    element(app.button, "打包已确认后处理结果").click().run()
    assert_ok(app)
    normalized_download = click_download(app, "下载已确认后处理 ZIP", downloaded_files)
    metadata = _assert_bundle(normalized_download.content, ws)
    assert metadata["requested_branches"] == ["normalized_baseline"]
    page(app, 5)
    element(app.button, "准备工作区下载").click().run()
    assert_ok(app)
    saved = click_download(app, "保存工作区 ZIP", downloaded_files)
    with zipfile.ZipFile(io.BytesIO(saved.content)) as archive:
        assert json.loads(archive.read("workspace.json"))["schema_version"] == "2.0"
    restored_app = batch_app()
    element(restored_app.get("file_uploader"), "恢复普通工作区 ZIP").set_value(
        (saved.filename, saved.content, saved.mimetype),
    ).run()
    element(restored_app.button, "恢复工作区").click().run()
    assert_ok(restored_app)
    restored = workspace(restored_app)
    assert restored.workspace_id == ws.workspace_id
    assert restored.selected_spectrum_id == a
    assert restored.selected_spectrum_ids == [a, b]
    assert restored_app.session_state["batch_target_ids"] == [a, b]
    assert element(restored_app.sidebar.multiselect, "批量勾选（独立于当前光谱）").value == [a, b]
    for sid in (a, b, c):
        original_baseline = pp.get_branch(ws, sid)
        restored_baseline = pp.get_branch(restored, sid)
        assert original_baseline.fingerprint == restored_baseline.fingerprint
        np.testing.assert_array_equal(pp.branch_arrays(original_baseline)[1], pp.branch_arrays(restored_baseline)[1])
        original = pp.get_postprocessing_state(ws, sid)
        actual = pp.get_postprocessing_state(restored, sid)
        assert actual.normalization_source == original.normalization_source
        assert actual.export_choice == original.export_choice
        for branch in pp.DERIVED_BRANCHES:
            left, right = getattr(original, branch), getattr(actual, branch)
            assert right.draft == left.draft
            assert right.committed_draft == left.committed_draft
            assert right.preview is None
            if left.committed is not None:
                assert right.committed is not None
                assert right.committed.fingerprint == left.committed.fingerprint
                assert right.committed.parent_fingerprint == left.committed.parent_fingerprint
                np.testing.assert_array_equal(right.committed.spectra, left.committed.spectra)
    post_page(restored_app)
    assert element(restored_app.number_input, "归一化目标值").value == 3.0
    assert element(restored_app.button, "确认当前归一化").disabled
    assert pp.branch_status(restored, a, "normalized_smoothed")["status"] == "stale"
    assert any("父级已过期" in item.value for item in restored_app.warning)
    assert science_calls == before


@pytest.mark.parametrize("level,overflow", [(-0.2, False), (-400.0, True)])
def test_negative_absorbance_display_retains_values_and_reports_overflow(
    science_calls: dict[str, int], level: float, overflow: bool,
) -> None:
    x = np.linspace(1800.0, 900.0, 181)
    ws, sid = confirmed_workspace(x=x, y=level + 0.01 * np.sin(x / 13))
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.session_state["batch_workspace"] = ws
    app.session_state["workflow_mode"] = "普通光谱模式"
    app.run()
    post_page(app)
    _smooth(app)
    snapshot = pp.get_branch(ws, sid, "smoothed")
    assert isinstance(snapshot, pp.PostprocessSnapshot)
    original = snapshot.spectra.copy()
    before = dict(science_calls)
    element(app.selectbox, "B / S 显示单位").set_value("percent_transmittance").run()
    assert not app.exception
    assert any("显示值未裁剪" in item.value for item in app.warning)
    if overflow:
        assert any("超出有限数值范围" in item.value for item in app.error)
    else:
        assert_ok(app)
        assert np.all(pp.display_arrays(snapshot, "percent_transmittance")[1] > 100)
    element(app.selectbox, "B / S 显示单位").set_value("absorbance").run()
    assert_ok(app)
    np.testing.assert_array_equal(snapshot.spectra, original)
    assert science_calls == before


def _csv_arrays(payload: bytes) -> dict[str, tuple[list[str], np.ndarray]]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return {
            name: (list(table.columns), table.to_numpy())
            for name in archive.namelist() if name.startswith("spectra/") and name.endswith(".csv")
            for table in [pd.read_csv(io.BytesIO(archive.read(name)), float_precision="round_trip")]
        }


def _assert_bundle(payload: bytes, ws: Any) -> dict[str, Any]:
    arrays = _csv_arrays(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        metadata = json.loads(archive.read("batch_metadata.json"))
        index = json.loads(archive.read("branch_index.json"))
        assert not metadata["is_2d_ready"]
        assert not any("for_2dcos" in name for name in archive.namelist())
        for node in index["nodes"]:
            recipe = json.loads(archive.read(node["recipe_json"]))
            snapshot = pp.get_branch(ws, node["spectrum_id"], node["branch"])
            expected_x, expected_y = pp.branch_arrays(snapshot)
            header, values = arrays[node["spectra_csv"]]
            assert header == ["wavenumber_cm-1", recipe["column_label"]]
            np.testing.assert_array_equal(values[:, 0], expected_x)
            np.testing.assert_array_equal(values[:, 1], expected_y[0])
            assert node["fingerprint"] == snapshot.fingerprint
    return metadata


def _confirmed_app() -> tuple[AppTest, Any, list[str]]:
    ws, a = confirmed_workspace(name="same.csv")
    _, b = confirmed_workspace(workspace=ws, name="same.csv")
    x = np.linspace(2200.0, 500.0, 171)
    _, c = confirmed_workspace(workspace=ws, name="other.csv", x=x)
    for sid in (a, b, c):
        pp.update_draft(ws, sid, "smoothed", {"enabled": True})
        pp.preview_branch(ws, sid, "smoothed")
        pp.confirm_branch(ws, sid, "smoothed")
    for branch in ("normalized_baseline", "normalized_smoothed"):
        pp.update_draft(ws, a, branch, {"enabled": True, "method": "maximum"})
        pp.preview_branch(ws, a, branch)
        pp.confirm_branch(ws, a, branch)
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.session_state["batch_workspace"] = ws
    app.session_state["workflow_mode"] = "普通光谱模式"
    app.run()
    post_page(app)
    return app, ws, [a, b, c]


def test_explicit_exports_include_parents_report_missing_and_preserve_download_bytes(
    downloaded_files: dict[str, DownloadedFile], science_calls: dict[str, int],
) -> None:
    app, ws, ids = _confirmed_app()
    a, b, c = ids
    before = dict(science_calls)
    assert element(app.selectbox, "后处理导出分支").value == "baseline"
    element(app.selectbox, "后处理导出分支").set_value("normalized_smoothed").run()
    element(app.selectbox, "后处理导出范围").set_value("全部条目").run()
    assert element(app.button, "打包已确认后处理结果").disabled
    preflight = next(item.value for item in app.dataframe if "将被导出" in item.value.columns)
    assert preflight["将被导出"].tolist() == [True, False, False]
    element(app.checkbox, "只导出有效项（明确排除下表不可用项）").set_value(True).run()
    element(app.button, "打包已确认后处理结果").click().run()
    assert_ok(app)
    first = click_download(app, "下载已确认后处理 ZIP", downloaded_files)
    original_bytes = first.content
    metadata = _assert_bundle(original_bytes, ws)
    assert metadata["exported_items"][0]["spectrum_id"] == a
    assert {row["spectrum_id"] for row in metadata["excluded_items"]} == {b, c}
    with zipfile.ZipFile(io.BytesIO(original_bytes)) as archive:
        nodes = json.loads(archive.read("branch_index.json"))["nodes"]
        assert {node["branch"] for node in nodes} == {"baseline", "smoothed", "normalized_smoothed"}
    # Changing a raw draft invalidates only the UI download identity, not arrays.
    element(app.checkbox, "启用当前来源归一化").set_value(True).run()
    element(app.selectbox, "普通谱归一化方法").set_value("area").run()
    assert not any(item.label == "下载已确认后处理 ZIP" for item in app.get("download_button"))
    assert first.content == original_bytes
    element(app.button, "打包已确认后处理结果").click().run()
    assert_ok(app)
    second = click_download(app, "下载已确认后处理 ZIP", downloaded_files)
    _assert_bundle(second.content, ws)
    # A real rename changes file labels/provenance, and never triggers science.
    element(app.text_input, "展示名称").set_value("renamed A").run()
    assert_ok(app)
    assert not any(item.label == "下载已确认后处理 ZIP" for item in app.get("download_button"))
    element(app.button, "打包已确认后处理结果").click().run()
    renamed = click_download(app, "下载已确认后处理 ZIP", downloaded_files)
    assert renamed.content != second.content
    _assert_bundle(renamed.content, ws)
    assert first.content == original_bytes
    assert science_calls == before
    page(app, 5)
    element(app.button, "打包已确认最终结果").click().run()
    legacy = click_download(app, "下载已确认最终 ZIP", downloaded_files)
    with zipfile.ZipFile(io.BytesIO(legacy.content)) as archive:
        legacy_metadata = json.loads(archive.read("batch_metadata.json"))
        assert legacy_metadata["stage"] == "fine"
        assert not any("normalized_smoothed" in name for name in archive.namelist())
    assert science_calls == before


def test_heterogeneous_axes_disable_wide_but_selected_same_axes_download_exactly(
    downloaded_files: dict[str, DownloadedFile], science_calls: dict[str, int],
) -> None:
    app, ws, ids = _confirmed_app()
    a, b, _ = ids
    before = dict(science_calls)
    element(app.selectbox, "后处理导出分支").set_value("smoothed").run()
    element(app.selectbox, "后处理导出范围").set_value("全部条目").run()
    assert element(app.checkbox, "提供后处理同轴宽表").disabled
    element(app.button, "打包已确认后处理结果").click().run()
    independent = click_download(app, "下载已确认后处理 ZIP", downloaded_files)
    metadata = _assert_bundle(independent.content, ws)
    assert not metadata["wide_available"]
    assert len(metadata["exported_items"]) == 3
    element(app.sidebar.multiselect, "批量勾选（独立于当前光谱）").set_value([a, b]).run()
    element(app.selectbox, "后处理导出范围").set_value("勾选条目").run()
    assert not element(app.checkbox, "提供后处理同轴宽表").disabled
    element(app.checkbox, "提供后处理同轴宽表").set_value(True).run()
    element(app.button, "打包已确认后处理结果").click().run()
    aligned = click_download(app, "下载后处理同轴宽表 CSV", downloaded_files)
    actual = pd.read_csv(io.BytesIO(aligned.content), float_precision="round_trip").to_numpy()
    expected = [pp.branch_arrays(pp.get_branch(ws, sid, "smoothed")) for sid in (a, b)]
    np.testing.assert_array_equal(actual[:, 0], expected[0][0])
    np.testing.assert_array_equal(actual[:, 1], expected[0][1][0])
    np.testing.assert_array_equal(actual[:, 2], expected[1][1][0])
    assert science_calls == before


def test_all_branch_export_requires_explicit_valid_only_and_preserves_minmax_quantity(
    downloaded_files: dict[str, DownloadedFile], science_calls: dict[str, int],
) -> None:
    app, ws, ids = _confirmed_app()
    a, b, _ = ids
    pp.update_draft(ws, b, "normalized_baseline", {"enabled": True, "method": "minmax_display"})
    pp.preview_branch(ws, b, "normalized_baseline")
    pp.confirm_branch(ws, b, "normalized_baseline")
    before = dict(science_calls)
    element(app.selectbox, "后处理导出分支").set_value("all").run()
    element(app.selectbox, "后处理导出范围").set_value("全部条目").run()
    assert not element(app.checkbox, "只导出有效项（明确排除下表不可用项）").value
    assert element(app.button, "打包已确认后处理结果").disabled
    element(app.checkbox, "只导出有效项（明确排除下表不可用项）").set_value(True).run()
    assert not element(app.button, "打包已确认后处理结果").disabled
    element(app.button, "打包已确认后处理结果").click().run()
    downloaded = click_download(app, "下载已确认后处理 ZIP", downloaded_files)
    metadata = _assert_bundle(downloaded.content, ws)
    assert metadata["selection_policy"] == "valid_only"
    assert set(metadata["requested_branches"]) == set(pp.BRANCHES)
    assert metadata["excluded_items"]
    arrays = _csv_arrays(downloaded.content)
    minmax = [pair for name, pair in arrays.items() if b in name and "minmax" in name]
    assert len(minmax) == 1
    assert minmax[0][0][1] == "minmax_scaled_intensity_display_only"
    assert np.min(minmax[0][1][:, 1]) == 0 and np.max(minmax[0][1][:, 1]) == 1
    assert pp.get_postprocessing_state(ws, a).normalization_source == "baseline"
    assert science_calls == before
