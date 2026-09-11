"""Real AppTest uploads and interactions for the independent workspace UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from ftir_baseline.config import PipelineConfig
from ftir_baseline.models import SpectrumSet
from ftir_baseline.pipeline import PipelineResult
from ftir_workbench.batch import service as batch_service
from ftir_workbench.batch.models import BatchWorkspace
from tests.batch.test_importing import table_bytes

ROOT = Path(__file__).resolve().parents[2]
APP_PATH = ROOT / "ui" / "streamlit_app.py"
BATCH_PAGES = [
    "1. 普通光谱导入与检查",
    "2. 单位与处理范围",
    "3. 逐谱粗调",
    "4. 逐谱细调",
    "6. 批量检查与导出",
]


def element(elements: Any, label: str) -> Any:
    return next(item for item in elements if item.label == label)


def assert_ok(app: AppTest) -> None:
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]


def batch_app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert_ok(app)
    element(app.selectbox, "数据模式").set_value("普通光谱模式").run()
    assert_ok(app)
    return app


def synthetic_uploads() -> list[tuple[str, bytes, str]]:
    x = np.linspace(1800.0, 900.0, 181)
    a = 0.3 + 0.0001 * x + 0.5 * np.exp(-((x - 1450.0) / 30.0) ** 2)
    b = 0.4 + 0.0002 * x + 0.8 * np.exp(-((x - 1300.0) / 40.0) ** 2)
    other_x = np.linspace(2200.0, 500.0, 171)
    c = 0.2 + 0.00005 * other_x + 0.3 * np.exp(-((other_x - 1700.0) / 50.0) ** 2)
    return [("wide.csv", table_bytes(x, a, b, labels=("A", "B")), "text/csv"),
            ("other.csv", table_bytes(other_x, c, labels=("C",)), "text/csv")]


@pytest.fixture
def pipeline_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    authoritative = batch_service.run_pipeline

    def counted(data: SpectrumSet, config: PipelineConfig) -> PipelineResult:
        calls.append(str(data.metadata["spectrum_id"]))
        return authoritative(data, config)

    monkeypatch.setattr(batch_service, "run_pipeline", counted)
    return calls


def workspace(app: AppTest) -> BatchWorkspace:
    return app.session_state["batch_workspace"]


def page(app: AppTest, index: int) -> None:
    element(app.sidebar.radio, "工作流").set_value(BATCH_PAGES[index - 1]).run()
    assert_ok(app)


def uploaded_app() -> AppTest:
    app = batch_app()
    element(app.get("file_uploader"), "上传普通光谱文本文件").set_value(synthetic_uploads()).run()
    assert_ok(app)
    element(app.button, "确认单位并导入").click().run()
    assert_ok(app)
    assert len(workspace(app).records) == 3
    return app


def select_spectrum(app: AppTest, sid: str) -> None:
    element(app.sidebar.selectbox, "当前光谱").set_value(sid).run()
    assert_ok(app)
    assert workspace(app).selected_spectrum_id == sid


def coarse_confirm(app: AppTest, method: str = "linear") -> None:
    page(app, 3)
    element(app.selectbox, "粗调方法").set_value(method).run()
    element(app.button, "计算粗调预览").click().run()
    assert_ok(app)
    element(app.button, "确认当前粗调").click().run()
    assert_ok(app)


def test_modes_are_selectable_before_upload_with_separate_navigation(
    pipeline_calls: list[str],
) -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert_ok(app)
    mode = element(app.selectbox, "数据模式")
    assert mode.value == "原位序列模式"
    assert list(mode.options) == ["原位序列模式", "普通光谱模式"]
    old_pages = list(element(app.sidebar.radio, "工作流").options)
    assert len(old_pages) == 10
    mode.set_value("普通光谱模式").run()
    assert_ok(app)
    assert list(element(app.sidebar.radio, "工作流").options) == [
        *BATCH_PAGES[:4], "5. 普通光谱后处理", BATCH_PAGES[4]
    ]
    assert not workspace(app).records
    assert workspace(app).workflow_mode == "independent_batch"
    assert app.session_state["prepared"] is None
    assert app.session_state["twodcos_result"] is None
    assert pipeline_calls == []


def test_real_upload_splits_wide_columns_and_preserves_unaligned_file_axes(
    pipeline_calls: list[str],
) -> None:
    app = uploaded_app()
    ws = workspace(app)
    records = [ws.records[sid] for sid in ws.display_order]
    assert [record.original_column_label for record in records] == ["A", "B", "C"]
    assert [record.wavenumber.size for record in records] == [181, 181, 171]
    assert len(ws.sources) == 2
    assert len({record.scientific_input_sha256 for record in records}) == 3
    assert [ws.states[record.spectrum_id].preparation_committed.wavenumber_range
            for record in records] == [(1800.0, 900.0), (1800.0, 900.0), (2200.0, 500.0)]
    assert all(state.coarse_snapshot is None and state.fine_snapshot is None
               for state in ws.states.values())
    assert pipeline_calls == []
    assert app.session_state["raw_data"] is None
    assert app.session_state["prepared"] is None
    assert app.session_state["twodcos_result"] is None
    table = app.dataframe[0].value
    assert table["原始列序号（x=0）"].tolist() == [1, 2, 1]


def test_a_b_a_and_unrendered_widgets_restore_drafts_and_confirmed_arrays(
    pipeline_calls: list[str],
) -> None:
    app = uploaded_app()
    ws = workspace(app)
    a, b, _ = ws.display_order
    coarse_confirm(app)
    saved_a = ws.states[a].coarse_snapshot
    assert saved_a is not None
    original_arrays = saved_a.result.analysis_data.copy()
    element(app.selectbox, "粗调方法").set_value("offset").run()
    element(app.number_input, "Lambda").set_value(234567.0).run()
    a_method_key = element(app.selectbox, "粗调方法").key
    assert ws.states[a].coarse_draft.method == "offset"
    assert ws.states[a].coarse_draft.lam == 234567.0
    assert ws.states[a].coarse_snapshot is saved_a
    assert pipeline_calls == [a]

    select_spectrum(app, b)
    assert a_method_key not in app.session_state
    element(app.selectbox, "粗调方法").set_value("none").run()
    assert ws.states[b].coarse_draft.method == "none"
    select_spectrum(app, a)
    assert element(app.selectbox, "粗调方法").value == "offset"
    assert element(app.number_input, "Lambda").value == 234567.0
    assert ws.states[a].coarse_snapshot is saved_a
    np.testing.assert_array_equal(saved_a.result.analysis_data, original_arrays)

    page(app, 2)
    assert a_method_key not in app.session_state
    page(app, 3)
    assert element(app.selectbox, "粗调方法").value == "offset"
    element(app.sidebar.text_input, "搜索光谱").set_value("no matching spectrum").run()
    assert_ok(app)
    assert a_method_key not in app.session_state
    element(app.sidebar.text_input, "搜索光谱").set_value("").run()
    assert_ok(app)
    select_spectrum(app, a)
    assert element(app.selectbox, "粗调方法").value == "offset"
    assert element(app.number_input, "Lambda").value == 234567.0
    assert ws.states[a].coarse_snapshot is saved_a
    assert ws.states[b].coarse_draft.method == "none"
    assert pipeline_calls == [a]


def test_fine_draft_and_snapshot_restore_and_new_coarse_marks_only_a_fine_stale(
    pipeline_calls: list[str],
) -> None:
    app = uploaded_app()
    ws = workspace(app)
    a, b, _ = ws.display_order
    coarse_confirm(app)
    page(app, 4)
    assert ws.states[a].fine_decision == "not_decided"
    element(app.button, "计算细调预览").click().run()
    assert_ok(app)
    assert ws.states[a].fine_snapshot is None
    element(app.button, "确认当前细调").click().run()
    assert_ok(app)
    saved_fine = ws.states[a].fine_snapshot
    assert saved_fine is not None
    element(app.number_input, "端点窗口宽度 cm⁻¹").set_value(25.0).run()
    assert ws.states[a].fine_snapshot is saved_fine
    assert not ws.states[a].fine_stale
    select_spectrum(app, b)
    assert element(app.number_input, "端点窗口宽度 cm⁻¹").value == 8.0
    element(app.number_input, "端点窗口宽度 cm⁻¹").set_value(40.0).run()
    select_spectrum(app, a)
    assert element(app.number_input, "端点窗口宽度 cm⁻¹").value == 25.0
    assert ws.states[a].fine_snapshot is saved_fine
    assert pipeline_calls == [a, a]

    coarse_confirm(app, "offset")
    assert pipeline_calls == [a, a, a]
    assert ws.states[a].fine_stale
    assert ws.states[a].fine_snapshot is saved_fine
    assert ws.states[b].fine_draft.endpoint_window_width_cm1 == 40.0
    assert not ws.states[b].fine_stale
    page(app, 4)
    element(app.radio, "显示结果").set_value("已确认结果").run()
    assert_ok(app)
    assert any("历史快照已过期" in item.value for item in app.warning)
    assert not app.get("plotly_chart")


def test_view_changes_selection_filter_sort_and_confirmation_do_not_run_pipeline(
    pipeline_calls: list[str],
) -> None:
    app = uploaded_app()
    ws = workspace(app)
    a, b, c = ws.display_order
    coarse_confirm(app)
    snapshot = ws.states[a].coarse_snapshot
    assert snapshot is not None
    assert pipeline_calls == [a]
    element(app.selectbox, "校正谱派生显示单位").set_value("percent_transmittance").run()
    element(app.radio, "显示结果").set_value("已确认结果").run()
    element(app.text_input, "展示名称").set_value("Renamed A").run()
    element(app.sidebar.multiselect, "批量勾选（独立于当前光谱）").set_value([b, c]).run()
    select_spectrum(app, b)
    select_spectrum(app, a)
    assert app.session_state["batch_target_ids"] == [b, c]
    element(app.sidebar.button, "反转显示顺序").click().run()
    assert_ok(app)
    assert ws.display_order == [c, b, a]
    element(app.sidebar.selectbox, "状态筛选").set_value("粗调完成").run()
    assert_ok(app)
    assert element(app.sidebar.selectbox, "当前光谱").value == a
    element(app.sidebar.selectbox, "状态筛选").set_value("全部").run()
    select_spectrum(app, a)
    element(app.button, "确认当前粗调").click().run()
    assert_ok(app)
    assert ws.states[a].coarse_snapshot is snapshot
    assert pipeline_calls == [a]


def test_copy_parameters_changes_each_targets_draft_without_fitting_or_committing(
    pipeline_calls: list[str],
) -> None:
    app = uploaded_app()
    ws = workspace(app)
    a, b, c = ws.display_order
    coarse_confirm(app)
    a_snapshot = ws.states[a].coarse_snapshot
    select_spectrum(app, b)
    coarse_confirm(app, "none")
    b_snapshot = ws.states[b].coarse_snapshot
    select_spectrum(app, a)
    element(app.selectbox, "粗调方法").set_value("offset").run()
    element(app.number_input, "Lambda").set_value(777000.0).run()
    element(app.sidebar.multiselect, "批量勾选（独立于当前光谱）").set_value([b, c]).run()
    before = list(pipeline_calls)
    element(app.button, "复制粗调参数到选中项").click().run()
    assert_ok(app)
    assert pipeline_calls == before
    assert ws.states[a].coarse_snapshot is a_snapshot
    assert ws.states[b].coarse_snapshot is b_snapshot
    assert ws.states[c].coarse_snapshot is None
    assert ws.states[b].coarse_draft.method == ws.states[c].coarse_draft.method == "offset"
    assert ws.states[b].coarse_draft.lam == ws.states[c].coarse_draft.lam == 777000.0
    assert ws.states[b].coarse_draft is not ws.states[c].coarse_draft
    assert ws.states[c].preparation_draft.wavenumber_range == (2200.0, 500.0)
    select_spectrum(app, b)
    assert element(app.selectbox, "粗调方法").value == "offset"
    element(app.number_input, "Lambda").set_value(888000.0).run()
    assert ws.states[c].coarse_draft.lam == 777000.0
    assert pipeline_calls == before


def test_switching_modes_preserves_real_in_situ_input_and_batch_snapshot(
    pipeline_calls: list[str],
) -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    x = np.linspace(1800.0, 900.0, 181)
    data = table_bytes(x, 0.1 + x / 5000.0, 0.2 + x / 5000.0, labels=("0MIN", "1MIN"))
    app.get("file_uploader")[0].set_value([("series.csv", data, "text/csv")]).run()
    element(app.radio, "已确认输入单位").set_value("absorbance").run()
    element(app.button, "Load using these settings").click().run()
    assert_ok(app)
    legacy_input = app.session_state["raw_data"]
    legacy_config = dict(app.session_state["baseline_config"])
    assert legacy_input.n_spectra == 2

    element(app.selectbox, "数据模式").set_value("普通光谱模式").run()
    element(app.get("file_uploader"), "上传普通光谱文本文件").set_value(synthetic_uploads()).run()
    element(app.button, "确认单位并导入").click().run()
    assert_ok(app)
    ws = workspace(app)
    a = ws.display_order[0]
    coarse_confirm(app)
    snapshot = ws.states[a].coarse_snapshot
    element(app.selectbox, "粗调方法").set_value("offset").run()
    element(app.selectbox, "数据模式").set_value("原位序列模式").run()
    assert_ok(app)
    assert app.session_state["raw_data"] is legacy_input
    assert app.session_state["baseline_config"] == legacy_config
    element(app.selectbox, "数据模式").set_value("普通光谱模式").run()
    assert_ok(app)
    page(app, 3)
    assert workspace(app) is ws
    assert ws.states[a].coarse_snapshot is snapshot
    assert element(app.selectbox, "粗调方法").value == "offset"
    assert app.session_state["prepared"] is None
    assert app.session_state["twodcos_result"] is None
    assert pipeline_calls == [a]


def test_explicit_fine_skip_is_distinct_from_unprocessed_and_does_not_fit(
    pipeline_calls: list[str],
) -> None:
    app = uploaded_app()
    ws = workspace(app)
    a, b, _ = ws.display_order
    coarse_confirm(app)
    page(app, 4)
    assert ws.states[a].fine_decision == "not_decided"
    assert ws.states[a].fine_snapshot is None
    element(app.button, "确认不做细调").click().run()
    assert_ok(app)
    assert ws.states[a].fine_decision == "explicitly_skipped"
    assert ws.states[a].fine_snapshot is not None
    assert ws.states[a].fine_snapshot.result is ws.states[a].coarse_snapshot.result
    assert ws.states[b].fine_decision == "not_decided"
    assert ws.states[b].fine_snapshot is None
    assert pipeline_calls == [a]


def test_preparation_unit_range_and_smoothing_drafts_are_independent_until_confirmed(
    pipeline_calls: list[str],
) -> None:
    app = uploaded_app()
    ws = workspace(app)
    a, b, c = ws.display_order
    coarse_confirm(app)
    original_a = ws.states[a].coarse_snapshot
    page(app, 2)
    element(app.number_input, "处理范围上限 cm⁻¹").set_value(1750.0).run()
    element(app.number_input, "处理范围下限 cm⁻¹").set_value(1000.0).run()
    element(app.checkbox, "仅平滑基线估计通道").set_value(True).run()
    element(app.selectbox, "此谱强度单位").set_value("fraction_transmittance").run()
    assert_ok(app)
    assert ws.states[a].preparation_draft.wavenumber_range == (1750.0, 1000.0)
    assert ws.states[a].preparation_draft.baseline_smoothing.enabled
    assert ws.records[a].confirmed_input_unit == "absorbance"
    assert ws.states[a].coarse_snapshot is original_a
    assert not ws.states[a].coarse_stale
    select_spectrum(app, c)
    assert element(app.number_input, "处理范围上限 cm⁻¹").value == 2200.0
    assert element(app.number_input, "处理范围下限 cm⁻¹").value == 500.0
    assert not element(app.checkbox, "仅平滑基线估计通道").value
    select_spectrum(app, a)
    assert element(app.number_input, "处理范围上限 cm⁻¹").value == 1750.0
    assert element(app.number_input, "处理范围下限 cm⁻¹").value == 1000.0
    assert element(app.checkbox, "仅平滑基线估计通道").value
    assert element(app.selectbox, "此谱强度单位").value == "fraction_transmittance"
    element(app.button, "确认此谱单位与处理范围").click().run()
    assert_ok(app)
    assert ws.records[a].confirmed_input_unit == "fraction_transmittance"
    assert ws.states[a].coarse_stale
    assert not ws.states[b].coarse_stale
    assert ws.records[b].confirmed_input_unit == "absorbance"
    assert pipeline_calls == [a]
    coarse_confirm(app)
    assert pipeline_calls == [a, a]
    assert ws.states[a].coarse_snapshot.config.baseline_smoothing.enabled
    assert ws.states[a].coarse_snapshot.config.wavenumber_range == (1750.0, 1000.0)
    page(app, 4)
    element(app.button, "计算细调预览").click().run()
    assert_ok(app)
    assert any("fine 拟合使用估计残差" in item.value for item in app.info)
    assert pipeline_calls == [a, a, a]


def test_prefilled_fine_anchors_survive_selection_and_copy_as_separate_drafts(
    pipeline_calls: list[str],
) -> None:
    app = uploaded_app()
    ws = workspace(app)
    a, b, c = ws.display_order
    coarse_confirm(app)
    page(app, 4)
    element(app.selectbox, "细调方法").set_value("pchip").run()
    element(app.button, "预填覆盖当前范围的锚点").click().run()
    assert_ok(app)
    original_anchors = ws.states[a].fine_draft.anchors
    assert len(original_anchors) == 4
    select_spectrum(app, b)
    assert not ws.states[b].fine_draft.anchors
    select_spectrum(app, a)
    assert element(app.selectbox, "细调方法").value == "pchip"
    assert ws.states[a].fine_draft.anchors == original_anchors
    assert len(app.get("dataframe")[0].value) == 4
    element(app.button, "计算细调预览").click().run()
    assert_ok(app)
    element(app.button, "确认当前细调").click().run()
    assert_ok(app)
    saved_fine = ws.states[a].fine_snapshot
    element(app.sidebar.multiselect, "批量勾选（独立于当前光谱）").set_value([b, c]).run()
    element(app.button, "复制细调参数到选中项").click().run()
    assert_ok(app)
    assert ws.states[b].fine_draft.anchors == original_anchors
    assert ws.states[b].fine_draft is not ws.states[a].fine_draft
    assert ws.states[b].fine_snapshot is None
    assert ws.states[c].fine_draft.method == "endpoint_window_linear"
    assert app.session_state["batch_operation_report"][b] is None
    assert "ANCHOR_OUT_OF_RANGE" in app.session_state["batch_operation_report"][c]
    assert ws.states[a].fine_snapshot is saved_fine
    assert pipeline_calls == [a, a]


def test_explicit_batch_preview_uses_each_selected_records_own_recipe(
    pipeline_calls: list[str],
) -> None:
    app = uploaded_app()
    ws = workspace(app)
    a, b, c = ws.display_order
    page(app, 3)
    element(app.selectbox, "粗调方法").set_value("linear").run()
    select_spectrum(app, b)
    element(app.selectbox, "粗调方法").set_value("none").run()
    select_spectrum(app, c)
    element(app.selectbox, "粗调方法").set_value("offset").run()
    element(app.sidebar.multiselect, "批量勾选（独立于当前光谱）").set_value([a, b]).run()
    assert pipeline_calls == []
    element(app.button, "为选中项计算各自粗调预览").click().run()
    assert_ok(app)
    assert pipeline_calls == [a, b]
    assert ws.states[a].coarse_preview.config.coarse_baseline.method == "linear"
    assert ws.states[b].coarse_preview.config.coarse_baseline.method == "none"
    assert ws.states[c].coarse_preview is None
    assert all(state.coarse_snapshot is None for state in ws.states.values())
