"""Postprocessing export previews share preferences and serialize confirmed nodes."""

from __future__ import annotations

import io
import zipfile

import numpy as np

from ftir_workbench.batch import postprocessing as pp
from tests.batch.test_export_naming_ui import naming_preview, spectra_members
from tests.batch.test_ui import assert_ok, element, page
from tests.batch.test_ui_roundtrip import DownloadedFile, click_download
from tests.batch.test_ui_roundtrip import downloaded_files as downloaded_files
from tests.postprocessing.test_ui_postprocessing import post_page
from tests.postprocessing.test_ui_postprocessing import science_calls as science_calls
from tests.postprocessing.test_ui_roundtrip import _confirmed_app


def test_ns_preview_uses_confirmed_methods_parent_roles_and_real_download_arrays(
    downloaded_files: dict[str, DownloadedFile], science_calls: dict[str, int],
) -> None:
    app, ws, ids = _confirmed_app()
    sid = ids[0]
    calls = dict(science_calls)
    snapshots = {branch: pp.get_branch(ws, sid, branch) for branch in pp.BRANCHES}
    element(app.button, "使用已确认平滑谱 S").click().run()
    element(app.selectbox, "普通谱平滑方法").set_value("gaussian").run()
    element(app.selectbox, "普通谱归一化方法").set_value("area").run()
    element(app.selectbox, "后处理导出分支").set_value("normalized_smoothed").run()
    element(app.checkbox, "覆盖当前光谱的导出基础名").set_value(True).run()
    element(app.text_input, "当前光谱导出基础名").set_value("样品 α").run()
    assert_ok(app)
    table = naming_preview(app)
    paths = set(table["最终 CSV 路径"])
    assert paths == {
        "spectra/样品 α__baseline.csv",
        "spectra/样品 α__baseline__smooth-savgol.csv",
        "spectra/样品 α__baseline__smooth-savgol__norm-max.csv",
    }
    assert table["导出角色"].tolist().count("自动包含的父级") == 2
    assert table["导出角色"].tolist().count("请求结果") == 1
    element(app.button, "打包已确认后处理结果").click().run()
    assert_ok(app)
    actual = click_download(app, "下载已确认后处理 ZIP", downloaded_files)
    assert spectra_members(actual.content) == paths
    with zipfile.ZipFile(io.BytesIO(actual.content)) as archive:
        assert "export_name_map.csv" in archive.namelist()
        for branch, filename in (
            ("baseline", "样品 α__baseline.csv"),
            ("smoothed", "样品 α__baseline__smooth-savgol.csv"),
            ("normalized_smoothed", "样品 α__baseline__smooth-savgol__norm-max.csv"),
        ):
            output = np.loadtxt(io.BytesIO(archive.read("spectra/" + filename)), delimiter=",", skiprows=1)
            x, y = pp.branch_arrays(snapshots[branch])
            np.testing.assert_array_equal(output[:, 0], x)
            np.testing.assert_array_equal(output[:, 1], y[0])
    assert science_calls == calls
    assert ws.postprocessing[sid].normalization_source == "smoothed"
    for branch, snapshot in snapshots.items():
        assert pp.get_branch(ws, sid, branch) is snapshot
    page(app, 4)
    assert element(app.text_input, "当前光谱导出基础名").value == "样品 α"
    element(app.selectbox, "导出文件命名方式").set_value("display").run()
    post_page(app)
    assert element(app.selectbox, "导出文件命名方式").value == "display"
    assert ws.postprocessing[sid].normalization_source == "smoothed"
    assert science_calls == calls


def test_post_policy_peer_override_and_source_rename_refresh_download_without_compute(
    downloaded_files: dict[str, DownloadedFile], science_calls: dict[str, int],
) -> None:
    app, ws, ids = _confirmed_app()
    sid, peer = ids[:2]
    calls = dict(science_calls)
    element(app.button, "打包已确认后处理结果").click().run()
    original = click_download(app, "下载已确认后处理 ZIP", downloaded_files)
    cache_key = f"_batch_{ws.workspace_id}_{sid}_post_export_download"
    previous = app.session_state[cache_key]
    # Another spectrum's naming preference changes the workspace-wide collision plan.
    from ftir_workbench.batch.export_naming import set_name_override

    set_name_override(ws, peer, "peer only")
    app.run()
    assert_ok(app)
    assert not any(item.label == "下载已确认后处理 ZIP" for item in app.get("download_button"))
    assert app.session_state[cache_key][1].zip_bytes == original.content
    element(app.button, "打包已确认后处理结果").click().run()
    changed = click_download(app, "下载已确认后处理 ZIP", downloaded_files)
    assert spectra_members(changed.content) == {"spectra/same__baseline.csv"}
    assert app.session_state[cache_key][0] != previous[0]
    element(app.selectbox, "导出文件命名方式").set_value("display").run()
    assert not any(item.label == "下载已确认后处理 ZIP" for item in app.get("download_button"))
    element(app.text_input, "展示名称").set_value("显示新名").run()
    assert naming_preview(app)["最终 CSV 路径"].tolist() == ["spectra/显示新名__baseline.csv"]
    element(app.button, "打包已确认后处理结果").click().run()
    renamed = click_download(app, "下载已确认后处理 ZIP", downloaded_files)
    assert spectra_members(renamed.content) == {"spectra/显示新名__baseline.csv"}
    assert science_calls == calls


def test_post_legacy_preview_is_exact_and_missing_selection_remains_blocked(
    downloaded_files: dict[str, DownloadedFile], science_calls: dict[str, int],
) -> None:
    app, _ws, ids = _confirmed_app()
    calls = dict(science_calls)
    element(app.selectbox, "导出文件命名方式").set_value("legacy_id").run()
    element(app.selectbox, "后处理导出分支").set_value("normalized_smoothed").run()
    element(app.selectbox, "后处理导出范围").set_value("全部条目").run()
    assert element(app.button, "打包已确认后处理结果").disabled
    table = naming_preview(app)
    assert (table["最终 CSV 路径"] == "—").sum() == 2
    assert table["导出角色"].tolist().count("自动包含的父级") == 2
    element(app.checkbox, "只导出有效项（明确排除下表不可用项）").set_value(True).run()
    paths = {name for name in naming_preview(app)["最终 CSV 路径"] if name != "—"}
    assert all(ids[0] in name for name in paths)
    element(app.button, "打包已确认后处理结果").click().run()
    actual = click_download(app, "下载已确认后处理 ZIP", downloaded_files)
    assert spectra_members(actual.content) == paths
    with zipfile.ZipFile(io.BytesIO(actual.content)) as archive:
        assert "export_name_map.csv" not in archive.namelist()
    assert science_calls == calls
