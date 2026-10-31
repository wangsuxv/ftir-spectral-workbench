"""Readable names through real widgets, downloads, and workspace restoration."""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np
from streamlit.testing.v1 import AppTest

from ftir_workbench.batch.export_naming import get_naming_options
from tests.batch.test_importing import table_bytes
from tests.batch.test_ui import (
    assert_ok,
    batch_app,
    coarse_confirm,
    element,
    page,
    workspace,
)
from tests.batch.test_ui import pipeline_calls as pipeline_calls
from tests.batch.test_ui_roundtrip import DownloadedFile, click_download
from tests.batch.test_ui_roundtrip import downloaded_files as downloaded_files
from tests.postprocessing.test_ui_postprocessing import post_page


def upload_one(app: AppTest, name: str = "G2_90RH.csv") -> None:
    x = np.linspace(1800.0, 900.0, 81)
    y = 0.1 + np.exp(-((x - 1300.0) / 60.0) ** 2)
    element(app.get("file_uploader"), "上传普通光谱文本文件").set_value(
        [(name, table_bytes(x, y, labels=("Absorbance",)), "text/csv")],
    ).run()
    element(app.button, "确认单位并导入").click().run()
    assert_ok(app)


def naming_preview(app: AppTest) -> Any:
    return next(item.value for item in app.dataframe if "最终 CSV 路径" in item.value.columns)


def spectra_members(payload: bytes) -> set[str]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return {name for name in archive.namelist() if name.startswith("spectra/") and name.endswith(".csv")}


def test_shared_policy_override_download_and_workspace_roundtrip(
    downloaded_files: dict[str, DownloadedFile], pipeline_calls: list[str],
) -> None:
    app = batch_app()
    upload_one(app)
    coarse_confirm(app, "none")
    ws = workspace(app)
    sid = ws.selected_spectrum_id
    record = ws.records[sid]
    source = ws.sources[record.source_id]
    snapshot = ws.states[sid].coarse_snapshot
    assert naming_preview(app)["最终 CSV 路径"].tolist() == ["spectra/G2_90RH__baseline-coarse.csv"]
    calls = list(pipeline_calls)
    element(app.text_input, "展示名称").set_value("展示 α").run()
    element(app.selectbox, "导出文件命名方式").set_value("display").run()
    assert naming_preview(app)["最终 CSV 路径"].tolist() == ["spectra/展示 α__baseline-coarse.csv"]
    element(app.checkbox, "覆盖当前光谱的导出基础名").set_value(True).run()
    assert "fallback" in naming_preview(app)["命名说明"].iloc[0]
    element(app.text_input, "当前光谱导出基础名").set_value("样品 G2 90%RH").run()
    element(app.button, "打包已确认粗调结果").click().run()
    assert_ok(app)
    coarse = click_download(app, "下载已确认粗调 ZIP", downloaded_files)
    assert spectra_members(coarse.content) == set(naming_preview(app)["最终 CSV 路径"])
    with zipfile.ZipFile(io.BytesIO(coarse.content)) as archive:
        assert "export_name_map.csv" in archive.namelist()
    page(app, 4)
    element(app.button, "确认不做细调").click().run()
    assert_ok(app)
    assert element(app.selectbox, "导出文件命名方式").value == "display"
    assert element(app.text_input, "当前光谱导出基础名").value == "样品 G2 90%RH"
    assert naming_preview(app)["最终 CSV 路径"].tolist() == ["spectra/样品 G2 90%RH__baseline.csv"]
    post_page(app)
    assert element(app.selectbox, "导出文件命名方式").value == "display"
    assert element(app.text_input, "当前光谱导出基础名").value == "样品 G2 90%RH"
    assert ws.postprocessing[sid].normalization_source == "baseline"
    element(app.selectbox, "导出文件命名方式").set_value("original").run()
    page(app, 3)
    assert element(app.selectbox, "导出文件命名方式").value == "original"
    assert pipeline_calls == calls
    assert ws.states[sid].coarse_snapshot is snapshot
    assert ws.sources[record.source_id] is source
    assert ws.records[sid].original_column_label == record.original_column_label
    assert ws.records[sid].scientific_input_sha256 == record.scientific_input_sha256
    assert ws.records[sid].display_name == "展示 α"
    page(app, 5)
    element(app.button, "准备工作区下载").click().run()
    saved = click_download(app, "保存工作区 ZIP", downloaded_files)
    restored_app = batch_app()
    element(restored_app.get("file_uploader"), "恢复普通工作区 ZIP").set_value(
        (saved.filename, saved.content, saved.mimetype),
    ).run()
    element(restored_app.button, "恢复工作区").click().run()
    assert_ok(restored_app)
    restored = workspace(restored_app)
    assert get_naming_options(restored) == get_naming_options(ws)
    assert restored.display_order == ws.display_order
    assert restored.sources[record.source_id].original_bytes == source.original_bytes
    assert restored.states[sid].coarse_snapshot.fingerprint == snapshot.fingerprint
    np.testing.assert_array_equal(restored.states[sid].coarse_snapshot.result.analysis_data,
                                  snapshot.result.analysis_data)
    post_page(restored_app)
    assert element(restored_app.selectbox, "导出文件命名方式").value == "original"
    assert element(restored_app.text_input, "当前光谱导出基础名").value == "样品 G2 90%RH"
    assert pipeline_calls == calls


def test_unselected_collision_import_and_override_invalidate_frozen_download(
    downloaded_files: dict[str, DownloadedFile], pipeline_calls: list[str],
) -> None:
    app = batch_app()
    upload_one(app)
    coarse_confirm(app, "none")
    ws = workspace(app)
    sid = ws.selected_spectrum_id
    element(app.button, "打包已确认粗调结果").click().run()
    first = click_download(app, "下载已确认粗调 ZIP", downloaded_files)
    cached = app.session_state["batch_download_coarse"]
    before = list(pipeline_calls)
    page(app, 1)
    upload_one(app)
    assert len(ws.records) == 2
    assert ws.selected_spectrum_id == sid
    page(app, 3)
    assert not any(item.label == "下载已确认粗调 ZIP" for item in app.get("download_button"))
    assert app.session_state["batch_download_coarse"][1].zip_bytes == first.content
    changed = naming_preview(app)["最终 CSV 路径"].tolist()
    assert changed != ["spectra/G2_90RH__baseline-coarse.csv"]
    element(app.button, "打包已确认粗调结果").click().run()
    second = click_download(app, "下载已确认粗调 ZIP", downloaded_files)
    assert spectra_members(second.content) == set(changed)
    assert app.session_state["batch_download_coarse"][0] != cached[0]
    element(app.checkbox, "覆盖当前光谱的导出基础名").set_value(True).run()
    element(app.text_input, "当前光谱导出基础名").set_value("unique").run()
    assert not any(item.label == "下载已确认粗调 ZIP" for item in app.get("download_button"))
    assert app.session_state["batch_download_coarse"][1].zip_bytes == second.content
    assert pipeline_calls == before


def test_legacy_selection_retains_override_and_actual_legacy_structure(
    downloaded_files: dict[str, DownloadedFile], pipeline_calls: list[str],
) -> None:
    app = batch_app()
    upload_one(app)
    coarse_confirm(app, "none")
    ws = workspace(app)
    sid = ws.selected_spectrum_id
    calls = list(pipeline_calls)
    element(app.checkbox, "覆盖当前光谱的导出基础名").set_value(True).run()
    element(app.text_input, "当前光谱导出基础名").set_value("custom").run()
    element(app.selectbox, "导出文件命名方式").set_value("legacy_id").run()
    assert element(app.text_input, "当前光谱导出基础名").disabled
    assert element(app.checkbox, "覆盖当前光谱的导出基础名").disabled
    paths = set(naming_preview(app)["最终 CSV 路径"])
    assert all(sid in name for name in paths)
    element(app.button, "打包已确认粗调结果").click().run()
    actual = click_download(app, "下载已确认粗调 ZIP", downloaded_files)
    assert spectra_members(actual.content) == paths
    with zipfile.ZipFile(io.BytesIO(actual.content)) as archive:
        assert "export_name_map.csv" not in archive.namelist()
        assert json.loads(archive.read("batch_metadata.json"))["schema_version"] == "1.0"
    element(app.selectbox, "导出文件命名方式").set_value("original").run()
    assert element(app.text_input, "当前光谱导出基础名").value == "custom"
    assert naming_preview(app)["最终 CSV 路径"].tolist() == ["spectra/custom__baseline-coarse.csv"]
    assert pipeline_calls == calls


def test_derived_name_preview_and_new_import_inherit_shared_policy(
    downloaded_files: dict[str, DownloadedFile], pipeline_calls: list[str],
) -> None:
    app = batch_app()
    upload_one(app)
    coarse_confirm(app, "none")
    ws = workspace(app)
    calls = list(pipeline_calls)
    element(app.text_input, "展示名称").set_value("G2 展示").run()
    element(app.selectbox, "导出文件命名方式").set_value("display").run()
    element(app.multiselect, "附加派生透过率副本").set_value(
        ["fraction_transmittance", "percent_transmittance"],
    ).run()
    table = naming_preview(app)
    assert table["导出角色"].tolist().count("附加派生透过率") == 2
    element(app.button, "打包已确认粗调结果").click().run()
    actual = click_download(app, "下载已确认粗调 ZIP", downloaded_files)
    assert spectra_members(actual.content) == set(table["最终 CSV 路径"])
    page(app, 1)
    upload_one(app, "new.csv")
    assert get_naming_options(ws)["policy"] == "display"
    assert all(state.display_preferences["export_naming"]["policy"] == "display"
               for state in ws.states.values())
    assert pipeline_calls == calls
