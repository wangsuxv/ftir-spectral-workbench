"""Actual AppTest controls for ordinary postprocessing; uploaded synthetic CSVs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from ftir_workbench.batch import postprocessing as pp
from tests.batch.test_ui import (
    assert_ok,
    coarse_confirm,
    element,
    page,
    uploaded_app,
    workspace,
)
from ui.batch_workflow import _workspace_context

POST_PAGE = "5. 普通光谱后处理"


def post_page(app: AppTest) -> None:
    element(app.sidebar.radio, "工作流").set_value(POST_PAGE).run()
    assert_ok(app)


def control(app: AppTest, kind: str, branch: str, field: str) -> Any:
    return next(item for item in getattr(app, kind)
                if str(item.key).endswith(f"_post_{branch}_{field}"))


@pytest.fixture
def ready_app() -> Iterator[AppTest]:
    app = uploaded_app()
    coarse_confirm(app)
    page(app, 4)
    element(app.button, "确认不做细调").click().run()
    assert_ok(app)
    post_page(app)
    yield app


@pytest.fixture
def science_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"smoothing": 0, "normalization": 0}
    smooth, normalize = pp.smooth_spectral_arrays, pp.normalize_spectral_arrays

    def counted_smooth(*args: Any, **kwargs: Any) -> Any:
        calls["smoothing"] += 1
        return smooth(*args, **kwargs)

    def counted_normalize(*args: Any, **kwargs: Any) -> Any:
        calls["normalization"] += 1
        return normalize(*args, **kwargs)

    monkeypatch.setattr(pp, "smooth_spectral_arrays", counted_smooth)
    monkeypatch.setattr(pp, "normalize_spectral_arrays", counted_normalize)
    return calls


def test_defaults_and_final_baseline_prerequisite(science_calls: dict[str, int]) -> None:
    app = uploaded_app()
    coarse_confirm(app)
    post_page(app)
    ws, sid = workspace(app), workspace(app).selected_spectrum_id
    assert sid is not None
    state = pp.get_postprocessing_state(ws, sid)
    assert state.normalization_source == state.export_choice == "baseline"
    assert all(not getattr(state, branch).draft["enabled"] for branch in pp.DERIVED_BRANCHES)
    assert element(app.button, "使用已确认平滑谱 S").disabled
    assert element(app.button, "计算当前平滑预览").disabled
    element(app.checkbox, "启用普通谱平滑").set_value(True).run()
    assert element(app.button, "计算当前平滑预览").disabled
    assert any("先完成细调确认" in item.value for item in app.info)
    assert science_calls == {"smoothing": 0, "normalization": 0}
    page(app, 4)
    element(app.button, "确认不做细调").click().run()
    post_page(app)
    assert not element(app.button, "计算当前平滑预览").disabled


@pytest.mark.parametrize("method,field,value", [
    ("savgol", "savgol_window_length", 9),
    ("gaussian", "gaussian_sigma_points", 1.8),
    ("moving_average", "moving_average_window_length", 5),
    ("median", "median_window_length", 5),
])
def test_all_smoothing_methods_preview_confirm_and_plots(
    ready_app: AppTest, science_calls: dict[str, int], method: str, field: str,
    value: int | float,
) -> None:
    app = ready_app
    ws, sid = workspace(app), workspace(app).selected_spectrum_id
    assert sid is not None
    original = pp.branch_arrays(pp.get_branch(ws, sid))[1].copy()
    element(app.checkbox, "启用普通谱平滑").set_value(True).run()
    element(app.selectbox, "普通谱平滑方法").set_value(method).run()
    control(app, "number_input", "smoothed", field).set_value(value).run()
    boundary = "SG 边界模式" if method == "savgol" else "卷积边界模式"
    element(app.selectbox, boundary).set_value("nearest").run()
    element(app.button, "计算当前平滑预览").click().run()
    assert_ok(app)
    state = pp.get_postprocessing_state(ws, sid)
    assert state.smoothed.preview is not None and state.smoothed.committed is None
    assert science_calls["smoothing"] == 1
    assert len(app.get("plotly_chart")) == 2
    element(app.button, "确认当前平滑").click().run()
    assert_ok(app)
    snapshot = pp.get_branch(ws, sid, "smoothed")
    assert snapshot is state.smoothed.committed
    assert science_calls["smoothing"] == 1
    assert state.normalization_source == state.export_choice == "baseline"
    np.testing.assert_array_equal(pp.branch_arrays(pp.get_branch(ws, sid))[1], original)
    assert any("窗口" in item.value or "Gaussian" in item.value for item in app.caption)
    assert element(app.button, "使用已确认平滑谱 S").disabled is False


@pytest.mark.parametrize("method", [
    "maximum", "internal_peak_height", "internal_peak_area", "area", "vector", "minmax_display",
])
def test_normalization_active_controls_and_selected_output(
    ready_app: AppTest, science_calls: dict[str, int], method: str,
) -> None:
    app = ready_app
    ws, sid = workspace(app), workspace(app).selected_spectrum_id
    assert sid is not None
    branch = "normalized_baseline"
    element(app.checkbox, "启用当前来源归一化").set_value(True).run()
    element(app.selectbox, "普通谱归一化方法").set_value(method).run()
    if method.startswith("internal_"):
        element(app.number_input, "归一化区间下界（cm⁻¹）").set_value(1350.0).run()
        element(app.number_input, "归一化区间上界（cm⁻¹）").set_value(1550.0).run()
    if method == "area":
        element(app.checkbox, "面积使用指定区间").set_value(True).run()
        element(app.number_input, "归一化区间下界（cm⁻¹）").set_value(1100.0).run()
        element(app.number_input, "归一化区间上界（cm⁻¹）").set_value(1700.0).run()
    if method == "minmax_display":
        assert not any(item.label == "归一化目标值" for item in app.number_input)
    else:
        element(app.number_input, "归一化目标值").set_value(2.0).run()
    if method in {"vector", "minmax_display"}:
        assert not any("归一化区间" in item.label for item in app.number_input)
    element(app.button, "计算当前归一化预览").click().run()
    assert_ok(app)
    state = pp.branch_state(ws, sid, branch)
    assert state.preview is not None and state.committed is None
    element(app.button, "确认当前归一化").click().run()
    assert_ok(app)
    snapshot = pp.get_branch(ws, sid, branch)
    assert isinstance(snapshot, pp.PostprocessSnapshot)
    assert science_calls == {"smoothing": 0, "normalization": 1}
    assert len(app.get("plotly_chart")) == 2
    plots = [json.loads(item.proto.spec) for item in app.get("plotly_chart")]
    ylabel = plots[-1]["layout"]["yaxis"]["title"]["text"]
    assert ylabel == ("Min-Max scaled intensity (0-1, display only)"
                      if method == "minmax_display" else "Normalized intensity")
    assert not any(item.label == "B / S 显示单位" for item in app.selectbox)
    if method == "maximum":
        assert np.max(snapshot.spectra) == pytest.approx(2.0)
    elif method == "vector":
        assert np.linalg.norm(snapshot.spectra) == pytest.approx(2.0)
    elif method == "minmax_display":
        assert snapshot.purpose == "display_only"
        assert np.min(snapshot.spectra) == 0.0 and np.max(snapshot.spectra) == 1.0


def test_committed_draft_restore_and_save_download_context(
    ready_app: AppTest, science_calls: dict[str, int],
) -> None:
    app = ready_app
    ws, sid = workspace(app), workspace(app).selected_spectrum_id
    assert sid is not None
    old_save = _workspace_context(ws, for_save=True)
    baseline_export = _workspace_context(ws)
    element(app.checkbox, "启用普通谱平滑").set_value(True).run()
    element(app.button, "计算当前平滑预览").click().run()
    element(app.button, "确认当前平滑").click().run()
    assert_ok(app)
    saved = pp.get_branch(ws, sid, "smoothed")
    changed_save = _workspace_context(ws, for_save=True)
    assert changed_save != old_save
    assert _workspace_context(ws) == baseline_export
    control(app, "number_input", "smoothed", "savgol_window_length").set_value(11).run()
    assert pp.get_branch(ws, sid, "smoothed") is saved
    assert any("存在未确认修改" in item.value for item in app.warning)
    assert element(app.button, "确认当前平滑").disabled
    assert _workspace_context(ws, for_save=True) != changed_save
    element(app.button, "恢复已确认平滑参数").click().run()
    assert_ok(app)
    assert control(app, "number_input", "smoothed", "savgol_window_length").value == 7
    assert science_calls == {"smoothing": 1, "normalization": 0}


def test_batch_explicit_preview_confirm_with_per_spectrum_widths(
    ready_app: AppTest, science_calls: dict[str, int],
) -> None:
    app = ready_app
    ws = workspace(app)
    ids = list(ws.display_order)
    from ftir_workbench.batch.service import preview_coarse
    from ftir_workbench.batch.state import confirm_coarse, skip_fine

    for sid in ids[1:]:
        preview_coarse(ws, sid)
        confirm_coarse(ws, sid)
        skip_fine(ws, sid)
    element(app.sidebar.multiselect, "批量勾选（独立于当前光谱）").set_value(ids).run()
    element(app.checkbox, "启用普通谱平滑").set_value(True).run()
    element(app.button, "复制平滑参数到选中项").click().run()
    assert_ok(app)
    assert science_calls["smoothing"] == 0
    assert all(pp.branch_state(ws, sid, "smoothed").committed is None for sid in ids)
    element(app.button, "为选中项计算各自平滑预览").click().run()
    assert_ok(app)
    assert science_calls["smoothing"] == 3
    assert all(pp.branch_state(ws, sid, "smoothed").committed is None for sid in ids)
    element(app.button, "确认选中项有效平滑预览").click().run()
    assert_ok(app)
    assert science_calls["smoothing"] == 3
    snapshots = [pp.get_branch(ws, sid, "smoothed") for sid in ids]
    assert all(isinstance(item, pp.PostprocessSnapshot) for item in snapshots)
    assert any("物理窗口（cm⁻¹）" in item.value.columns for item in app.dataframe)
    table = next(item.value for item in app.dataframe if "物理窗口（cm⁻¹）" in item.value.columns)
    assert table["物理窗口（cm⁻¹）"].nunique() == 2
