"""Independent real AppTest checks for edge cases in ordinary postprocessing.

The baseline fixture uses the public import/preview/confirm workflow. Subsequent
steps use actual Streamlit controls, never synthetic widget or callback APIs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from ftir_workbench.batch import postprocessing as postprocess
from ftir_workbench.batch.models import BatchWorkspace
from tests.batch.test_ui import APP_PATH, assert_ok, element, select_spectrum
from tests.postprocessing.helpers import confirmed_workspace

POST_PAGE = "5. 普通光谱后处理"


def app_for_workspace(ws: BatchWorkspace) -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.session_state["batch_workspace"] = ws
    app.session_state["workflow_mode"] = "普通光谱模式"
    app.run()
    assert_ok(app)
    return app


def choose_page(app: AppTest, label: str) -> None:
    element(app.sidebar.radio, "工作流").set_value(label).run()
    assert_ok(app)


@pytest.fixture
def numerical_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    smooth = postprocess.smooth_spectral_arrays
    normalize = postprocess.normalize_spectral_arrays

    def counted_smoothing(*args: Any, **kwargs: Any) -> Any:
        calls.append("smooth")
        return smooth(*args, **kwargs)

    def counted_normalization(*args: Any, **kwargs: Any) -> Any:
        calls.append("normalize")
        return normalize(*args, **kwargs)

    monkeypatch.setattr(postprocess, "smooth_spectral_arrays", counted_smoothing)
    monkeypatch.setattr(postprocess, "normalize_spectral_arrays", counted_normalization)
    return calls


def confirm_smoothing(app: AppTest) -> None:
    element(app.button, "计算当前平滑预览").click().run()
    assert_ok(app)
    element(app.button, "确认当前平滑").click().run()
    assert_ok(app)


def test_inactive_invalid_sg_draft_survives_switching_records_and_unrendered_pages(
    numerical_calls: list[str],
) -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="B.csv")
    app = app_for_workspace(ws)
    choose_page(app, POST_PAGE)
    element(app.checkbox, "启用普通谱平滑").check().run()
    element(app.number_input, "SG 窗口点数（奇数）").set_value(4).run()
    assert element(app.button, "计算当前平滑预览").disabled
    element(app.selectbox, "普通谱平滑方法").set_value("gaussian").run()
    element(app.number_input, "Gaussian σ（点数）").set_value(1.3).run()
    confirm_smoothing(app)
    state_a = postprocess.get_postprocessing_state(ws, a)
    saved = state_a.smoothed.committed
    assert saved is not None and saved.recipe["savgol_window_length"] == 4
    key = element(app.number_input, "Gaussian σ（点数）").key
    select_spectrum(app, b)
    assert key not in app.session_state
    element(app.selectbox, "普通谱平滑方法").set_value("median").run()
    element(app.number_input, "平滑窗口点数（奇数）").set_value(9).run()
    select_spectrum(app, a)
    assert element(app.number_input, "Gaussian σ（点数）").value == 1.3
    choose_page(app, "2. 单位与处理范围")
    assert key not in app.session_state
    choose_page(app, POST_PAGE)
    element(app.sidebar.text_input, "搜索光谱").set_value("nothing matches this").run()
    assert_ok(app)
    assert key not in app.session_state
    element(app.sidebar.text_input, "搜索光谱").set_value("").run()
    select_spectrum(app, a)
    assert element(app.number_input, "Gaussian σ（点数）").value == 1.3
    element(app.selectbox, "普通谱平滑方法").set_value("savgol").run()
    assert element(app.number_input, "SG 窗口点数（奇数）").value == 4
    assert element(app.button, "计算当前平滑预览").disabled
    assert state_a.smoothed.committed is saved
    assert postprocess.get_postprocessing_state(ws, b).smoothed.draft["median_window_length"] == 9
    assert numerical_calls == ["smooth"]


def test_source_specific_normalization_drafts_and_export_choice_survive_new_s_and_switches(
    numerical_calls: list[str],
) -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="B.csv")
    app = app_for_workspace(ws)
    choose_page(app, POST_PAGE)
    element(app.checkbox, "启用普通谱平滑").check().run()
    element(app.selectbox, "普通谱平滑方法").set_value("gaussian").run()
    confirm_smoothing(app)
    element(app.checkbox, "启用当前来源归一化").check().run()
    element(app.selectbox, "普通谱归一化方法").set_value("maximum").run()
    element(app.number_input, "归一化目标值").set_value(2.0).run()
    element(app.button, "计算当前归一化预览").click().run()
    element(app.button, "确认当前归一化").click().run()
    assert_ok(app)
    state = postprocess.get_postprocessing_state(ws, a)
    normalized_b = state.normalized_baseline.committed
    assert state.export_choice == "baseline"
    element(app.button, "使用已确认平滑谱 S").click().run()
    element(app.checkbox, "启用当前来源归一化").check().run()
    element(app.selectbox, "普通谱归一化方法").set_value("vector").run()
    element(app.number_input, "归一化目标值").set_value(3.0).run()
    element(app.button, "计算当前归一化预览").click().run()
    element(app.button, "确认当前归一化").click().run()
    assert_ok(app)
    normalized_s = state.normalized_smoothed.committed
    element(app.number_input, "Gaussian σ（点数）").set_value(2.0).run()
    confirm_smoothing(app)
    assert state.normalization_source == "smoothed" and state.export_choice == "baseline"
    assert state.normalized_baseline.committed is normalized_b
    assert state.normalized_smoothed.committed is normalized_s
    assert postprocess.branch_status(ws, a, "normalized_smoothed")["status"] == "stale"
    assert any("父级已过期" in item.value for item in app.warning)
    assert element(app.button, "确认当前归一化").disabled
    select_spectrum(app, b)
    element(app.selectbox, "普通谱归一化方法").set_value("minmax_display").run()
    assert not any(item.label == "归一化目标值" for item in app.number_input)
    select_spectrum(app, a)
    assert state.normalization_source == "smoothed"
    assert element(app.selectbox, "普通谱归一化方法").value == "vector"
    assert element(app.number_input, "归一化目标值").value == 3.0
    element(app.button, "使用未平滑校正谱 B").click().run()
    assert element(app.selectbox, "普通谱归一化方法").value == "maximum"
    assert element(app.number_input, "归一化目标值").value == 2.0
    choose_page(app, "1. 普通光谱导入与检查")
    choose_page(app, POST_PAGE)
    assert element(app.number_input, "归一化目标值").value == 2.0
    assert numerical_calls == ["smooth", "normalize", "normalize", "smooth"]


def test_new_preview_is_shown_and_display_controls_do_not_compute(
    numerical_calls: list[str],
) -> None:
    ws, sid = confirmed_workspace()
    app = app_for_workspace(ws)
    choose_page(app, POST_PAGE)
    element(app.checkbox, "启用普通谱平滑").check().run()
    element(app.selectbox, "普通谱平滑方法").set_value("gaussian").run()
    confirm_smoothing(app)
    state = postprocess.get_postprocessing_state(ws, sid).smoothed
    committed = state.committed
    element(app.number_input, "Gaussian σ（点数）").set_value(2.0).run()
    element(app.button, "计算当前平滑预览").click().run()
    assert_ok(app)
    assert state.committed is committed and state.preview is not committed
    assert element(app.selectbox, "显示后处理结果").value == "当前预览"
    assert numerical_calls == ["smooth", "smooth"]
    for unit in ("percent_transmittance", "fraction_transmittance", "absorbance"):
        element(app.selectbox, "B / S 显示单位").set_value(unit).run()
        assert_ok(app)
    element(app.selectbox, "显示后处理结果").set_value("已确认结果").run()
    element(app.selectbox, "显示后处理结果").set_value("当前预览").run()
    assert_ok(app)
    assert numerical_calls == ["smooth", "smooth"]
    assert postprocess.get_branch(ws, sid, "smoothed") is committed


def test_batch_preview_rejects_short_target_without_losing_existing_result(
    numerical_calls: list[str],
) -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(np.arange(5.0), np.ones(5), workspace=ws, name="short.csv")
    postprocess.update_draft(ws, b, "smoothed", {"enabled": True, "method": "gaussian"})
    postprocess.preview_branch(ws, b, "smoothed")
    previous = postprocess.confirm_branch(ws, b, "smoothed")
    numerical_calls.clear()
    app = app_for_workspace(ws)
    choose_page(app, POST_PAGE)
    element(app.checkbox, "启用普通谱平滑").check().run()
    element(app.number_input, "SG 窗口点数（奇数）").set_value(7).run()
    element(app.sidebar.multiselect, "批量勾选（独立于当前光谱）").set_value([a, b]).run()
    element(app.button, "复制平滑参数到选中项").click().run()
    assert_ok(app)
    state = postprocess.get_postprocessing_state(ws, a)
    copied = state.display_preferences["smoothed_report"]["outcomes"]
    assert copied[a] is None and copied[b]
    assert numerical_calls == []
    element(app.button, "为选中项计算各自平滑预览").click().run()
    assert_ok(app)
    report = state.display_preferences["smoothed_report"]["outcomes"]
    assert report[a] is None and report[b]
    assert state.smoothed.committed is None
    element(app.button, "确认选中项有效平滑预览").click().run()
    assert_ok(app)
    report = state.display_preferences["smoothed_report"]["outcomes"]
    assert report[a] is None and report[b]
    assert state.smoothed.committed is not None
    assert postprocess.get_branch(ws, b, "smoothed") is previous
    # The second call is the explicit short-target attempt. It rejects before
    # filtering; neither copies nor confirmations call either numerical API.
    assert numerical_calls == ["smooth", "smooth"]


def test_normalization_copy_and_batch_confirmation_report_missing_s_and_window_per_target(
    numerical_calls: list[str],
) -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="no-smoothed-parent.csv")
    _, c = confirmed_workspace(np.linspace(1500.0, 1400.0, 21), np.ones(21),
                               workspace=ws, name="different-range.csv")
    for sid in (a, c):
        postprocess.update_draft(ws, sid, "smoothed", {"enabled": True, "method": "gaussian"})
        postprocess.preview_branch(ws, sid, "smoothed")
        postprocess.confirm_branch(ws, sid, "smoothed")
    numerical_calls.clear()
    app = app_for_workspace(ws)
    choose_page(app, POST_PAGE)
    element(app.button, "使用已确认平滑谱 S").click().run()
    element(app.checkbox, "启用当前来源归一化").check().run()
    element(app.selectbox, "普通谱归一化方法").set_value("internal_peak_height").run()
    element(app.number_input, "归一化区间下界（cm⁻¹）").set_value(1600.0).run()
    element(app.number_input, "归一化区间上界（cm⁻¹）").set_value(1700.0).run()
    element(app.sidebar.multiselect, "批量勾选（独立于当前光谱）").set_value([a, b, c]).run()
    element(app.button, "复制归一化参数到选中项").click().run()
    assert_ok(app)
    state = postprocess.get_postprocessing_state(ws, a)
    report = state.display_preferences["normalized_smoothed_report"]["outcomes"]
    assert report[a] is None and report[b] and report[c]
    assert numerical_calls == []
    element(app.button, "为选中项计算各自归一化预览").click().run()
    assert_ok(app)
    assert state.normalized_smoothed.committed is None
    report = state.display_preferences["normalized_smoothed_report"]["outcomes"]
    assert report[a] is None and report[b] and report[c]
    element(app.button, "确认选中项有效归一化预览").click().run()
    assert_ok(app)
    report = state.display_preferences["normalized_smoothed_report"]["outcomes"]
    assert report[a] is None and report[b] and report[c]
    assert state.normalized_smoothed.committed is not None
    assert state.normalized_smoothed.committed.parent_fingerprint == state.smoothed.committed.fingerprint
    for sid in (b, c):
        target = postprocess.get_postprocessing_state(ws, sid)
        assert target.normalized_smoothed.committed is None
        assert target.normalized_baseline.preview is None
        assert target.normalized_baseline.committed is None
    table = next(item.value for item in app.dataframe if "结果" in item.value.columns)
    assert set(table["光谱 ID"]) == {a, b, c}
    assert table.loc[table["光谱 ID"] == a, "结果"].iloc[0] == "已确认"
    assert numerical_calls == ["normalize", "normalize"]


def test_confirm_and_next_visits_three_postprocessing_records_and_preserves_each_branch(
    numerical_calls: list[str],
) -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="B.csv")
    _, c = confirmed_workspace(workspace=ws, name="C.csv")
    app = app_for_workspace(ws)
    choose_page(app, POST_PAGE)
    saved = {}
    for sid, next_sid, method in ((a, b, "maximum"), (b, c, "vector"), (c, a, "minmax_display")):
        assert ws.selected_spectrum_id == sid
        element(app.checkbox, "启用当前来源归一化").check().run()
        element(app.selectbox, "普通谱归一化方法").set_value(method).run()
        element(app.button, "计算当前归一化预览").click().run()
        assert_ok(app)
        state = postprocess.get_postprocessing_state(ws, sid)
        saved[sid] = state.normalized_baseline.preview
        element(app.button, "确认归一化并下一条").click().run()
        assert_ok(app)
        assert ws.selected_spectrum_id == next_sid
        assert element(app.sidebar.selectbox, "当前光谱").value == next_sid
        assert state.normalized_baseline.committed is saved[sid]
        assert state.export_choice == "baseline" and state.normalization_source == "baseline"
    assert element(app.selectbox, "普通谱归一化方法").value == "maximum"
    for sid, snapshot in saved.items():
        assert postprocess.get_branch(ws, sid, "normalized_baseline") is snapshot
    assert numerical_calls == ["normalize", "normalize", "normalize"]
