"""Independent batch UI. Widgets edit workspace drafts; buttons alone run science."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ftir_baseline.config import CoarseBaselineConfig, FineBaselineConfig
from ftir_baseline.io import TextImportOptions
from ftir_workbench.batch.importing import import_sources
from ftir_workbench.batch.models import BatchWorkspace, PreparationConfig
from ftir_workbench.batch.service import preview_coarse, preview_fine, preview_selected
from ftir_workbench.batch.state import (
    confirm_coarse,
    confirm_fine,
    confirm_preparation,
    copy_parameters,
    draft_changes,
    get_ready_snapshot,
    restore_committed_draft,
    set_coarse_draft,
    set_fine_draft,
    set_preparation_draft,
    skip_fine,
)
from ftir_workbench.display_units import convert_absorbance_for_display

try:
    from ui.components.baseline_preview import (
        anchor_diagnostics_table,
        coarse_preview_figure,
        fine_decomposition_figure,
        fine_residual_figure,
    )
except ModuleNotFoundError:
    from components.baseline_preview import (  # type: ignore[no-redef]
        anchor_diagnostics_table,
        coarse_preview_figure,
        fine_decomposition_figure,
        fine_residual_figure,
    )

PAGES = (
    "1. 普通光谱导入与检查",
    "2. 单位与处理范围",
    "3. 逐谱粗调",
    "4. 逐谱细调",
    "5. 批量检查与导出",
)
UNITS = ("absorbance", "percent_transmittance", "fraction_transmittance")
UNIT_LABELS = {
    "absorbance": "A（吸光度）",
    "percent_transmittance": "%T（百分比透过率）",
    "fraction_transmittance": "T（分数透过率）",
}
COARSE_METHODS = (
    "arpls",
    "asls",
    "airpls",
    "pspline_arpls",
    "rubberband",
    "offset",
    "linear",
    "none",
)
FINE_METHODS = ("endpoint_window_linear", "piecewise_linear", "pchip", "polynomial", "none")


def status_label(ws: BatchWorkspace, sid: str) -> str:
    record, state = ws.records[sid], ws.states[sid]
    if record.excluded:
        return "已排除"
    if state.coarse_stale:
        return "粗调过期"
    if state.fine_stale:
        return "细调过期"
    if state.errors:
        return "计算失败"
    if state.fine_snapshot is not None:
        return "已跳过细调" if state.fine_decision == "explicitly_skipped" else "细调完成"
    if state.coarse_snapshot is not None:
        return "粗调完成"
    return "待处理"


def spectrum_table(ws: BatchWorkspace) -> pd.DataFrame:
    rows = []
    for sid in ws.display_order:
        r, s = ws.records[sid], ws.states[sid]
        source = ws.sources[r.source_id]
        rows.append(
            {
                "光谱 ID": sid,
                "展示名称": r.display_name,
                "源文件": source.original_filename,
                "原始列名": r.original_column_label,
                "原始列序号（x=0）": r.original_column_index,
                "单位": UNIT_LABELS[r.confirmed_input_unit],
                "最小波数": float(r.wavenumber.min()),
                "最大波数": float(r.wavenumber.max()),
                "点数": r.wavenumber.size,
                "方向": r.original_axis_direction,
                "状态": status_label(ws, sid),
                "重复候选": r.duplicate_candidate,
                "编码": source.import_probe["selected_encoding"],
                "分隔符": source.import_probe["selected_delimiter"],
                "parser 警告": "; ".join(source.import_probe["warnings"]),
                "错误": "; ".join(s.errors),
            }
        )
    return pd.DataFrame(rows)


def _key(ws: BatchWorkspace, sid: str, stage: str, field: str) -> str:
    return f"_batch_{ws.workspace_id}_{sid}_{stage}_{field}"


def _values(ws: BatchWorkspace, sid: str, stage: str) -> dict[str, Any]:
    state = ws.states[sid]
    name = "editor_" + stage
    if name not in state.display_preferences:
        if stage == "preparation":
            p = state.preparation_draft
            high, low = max(p.wavenumber_range), min(p.wavenumber_range)
            value = {
                "input_unit": p.input_unit,
                "high": high,
                "low": low,
                "smoothing": p.baseline_smoothing.enabled,
                "window_length": p.baseline_smoothing.window_length,
                "polyorder": p.baseline_smoothing.polyorder,
                "use_floor": p.transmittance_floor is not None,
                "floor": p.transmittance_floor or 1e-6,
            }
        else:
            value = getattr(state, stage + "_draft").to_dict()
        state.display_preferences[name] = value
    return state.display_preferences[name]


def _sync(ws: BatchWorkspace, sid: str, stage: str) -> None:
    values = _values(ws, sid, stage)
    state = ws.states[sid]
    try:
        if stage == "preparation":
            p = PreparationConfig(
                input_unit=values["input_unit"],
                wavenumber_range=(values["high"], values["low"]),
                baseline_smoothing={
                    "enabled": values["smoothing"],
                    "estimate_only": True,
                    "window_length": values["window_length"],
                    "polyorder": values["polyorder"],
                },
                transmittance_floor=values["floor"] if values["use_floor"] else None,
            )
            set_preparation_draft(ws, sid, p)
        elif stage == "coarse":
            set_coarse_draft(ws, sid, CoarseBaselineConfig(**values))
        else:
            set_fine_draft(ws, sid, FineBaselineConfig(**values))
        state.display_preferences.pop("error_" + stage, None)
    except ValueError as exc:
        state.display_preferences["error_" + stage] = str(exc)


def _changed(ws: BatchWorkspace, sid: str, stage: str, field: str, key: str) -> None:
    _values(ws, sid, stage)[field] = st.session_state[key]
    _sync(ws, sid, stage)


def _widget(
    ws: BatchWorkspace, sid: str, stage: str, field: str, label: str, kind: str, **kwargs: Any
) -> Any:
    key = _key(ws, sid, stage, field)
    if key not in st.session_state:
        st.session_state[key] = _values(ws, sid, stage)[field]
    return getattr(st, kind)(
        label, key=key, on_change=_changed, args=(ws, sid, stage, field, key), **kwargs
    )


def _reset_editor(ws: BatchWorkspace, sid: str, stage: str) -> None:
    prefs = ws.states[sid].display_preferences
    prefs.pop("editor_" + stage, None)
    prefs.pop("error_" + stage, None)
    prefix = _key(ws, sid, stage, "")
    for key in list(st.session_state):
        if str(key).startswith(prefix):
            del st.session_state[key]


def _attempt(action: Any, *args: Any, **kwargs: Any) -> bool:
    try:
        action(*args, **kwargs)
        return True
    except (ValueError, KeyError) as exc:
        st.error(str(exc))
        return False


def _sidebar_selection(ws: BatchWorkspace) -> str | None:
    if not ws.records:
        return None
    query = st.sidebar.text_input("搜索光谱", key=f"batch_search_{ws.workspace_id}")
    status = st.sidebar.selectbox(
        "状态筛选",
        (
            "全部",
            "待处理",
            "粗调完成",
            "细调完成",
            "细调过期",
            "粗调过期",
            "计算失败",
            "已排除",
            "已跳过细调",
        ),
    )
    visible = [
        sid
        for sid in ws.display_order
        if (
            query.casefold()
            in (
                ws.records[sid].display_name
                + " "
                + ws.sources[ws.records[sid].source_id].original_filename
            ).casefold()
            and (status == "全部" or status_label(ws, sid) == status)
        )
    ]
    labels = {
        sid: f"{ws.records[sid].display_name} · {ws.sources[ws.records[sid].source_id].original_filename} · 列{ws.records[sid].original_column_index} · {sid[:6]}"
        for sid in ws.display_order
    }

    def label(sid: str) -> str:
        return labels[sid]

    if not visible:
        st.sidebar.info("没有符合筛选条件的条目。")
        return None
    current = ws.selected_spectrum_id
    index = visible.index(current) if current in visible else 0
    # Selection is a display operation; workspace owns every hidden record's drafts.
    chosen = st.sidebar.selectbox(
        "当前光谱",
        visible,
        index=index,
        format_func=label,
        key=f"batch_current_{ws.workspace_id}_{query}_{status}_{st.session_state.get('batch_selection_revision', 0)}",
    )
    ws.selected_spectrum_id = chosen
    targets_key = f"batch_targets_{ws.workspace_id}"
    saved = st.session_state.get("batch_target_ids", [])
    if targets_key not in st.session_state:
        st.session_state[targets_key] = [sid for sid in saved if sid in ws.records]
    selected = st.sidebar.multiselect(
        "批量勾选（独立于当前光谱）", ws.display_order, format_func=label, key=targets_key
    )
    st.session_state["batch_target_ids"] = list(selected)
    if st.sidebar.button("反转显示顺序"):
        ws.display_order.reverse()
        ws.last_export_summary = None
        st.rerun()
    return chosen


def _import_page(ws: BatchWorkspace) -> None:
    st.subheader(PAGES[0])
    st.write("每列独立导入，保留各自波数轴。导入确认的是强度单位，不需要时间、温度或扰动坐标。")
    unit = st.selectbox("新文件默认强度单位", UNITS, format_func=UNIT_LABELS.get)
    with st.expander("文本解析选项"):
        delimiter = st.selectbox("分隔符", ("auto", "comma", "tab", "semicolon", "whitespace"))
        decimal = st.selectbox("小数符号", ("auto", "dot", "comma"))
        encoding = st.selectbox(
            "编码", ("auto", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "gb18030", "cp1252")
        )
        header = st.selectbox("表头", ("auto", "present", "absent"))
        skip = st.number_input("跳过开头行数", min_value=0, value=0)
        trim = st.checkbox("去除可证明全空的边缘列", value=True)
    uploads = st.file_uploader(
        "上传普通光谱文本文件",
        type=["csv", "tsv", "tab", "txt", "dpt", "asc", "dat", "xy"],
        accept_multiple_files=True,
        key="batch_raw_uploads",
    )
    units = []
    for index, upload in enumerate(uploads or []):
        units.append(
            st.selectbox(
                f"文件单位 · {upload.name} [{index + 1}]",
                UNITS,
                index=UNITS.index(unit),
                format_func=UNIT_LABELS.get,
                key=f"batch_upload_unit_{upload.file_id}",
            )
        )
    if st.button("确认单位并导入", disabled=not uploads, type="primary"):
        try:
            options = TextImportOptions(
                delimiter=delimiter,
                decimal_mark=decimal,
                encoding=encoding,
                header_mode=header,
                skip_rows=int(skip),
                trim_empty_edge_columns=trim,
            )
            added = []
            for upload, source_unit in zip(uploads, units, strict=True):
                added.extend(
                    import_sources(
                        ws,
                        [(upload.name, upload.getvalue())],
                        input_unit=source_unit,
                        options=options,
                    )
                )
            st.success(f"已导入 {len(added)} 条独立光谱。")
        except ValueError as exc:
            st.error(str(exc))
    if ws.records:
        st.dataframe(spectrum_table(ws), hide_index=True, width="stretch")
        st.caption("重复上传只标记候选；可在任一处理页明确保留或排除。列序号以波数列为 0。")
    if ws.import_issues:
        st.error("以下文件导入失败，其他成功条目已保留。")
        st.dataframe(pd.DataFrame(ws.import_issues), hide_index=True)
    _workspace_restore(ws)


def _record_header(ws: BatchWorkspace, sid: str) -> None:
    record, state = ws.records[sid], ws.states[sid]
    st.subheader(f"当前：{record.display_name}")
    st.caption(
        f"{ws.sources[record.source_id].original_filename} · 第 {record.original_column_index} 强度列 · {UNIT_LABELS[record.confirmed_input_unit]} · {status_label(ws, sid)}"
    )
    name_key = _key(ws, sid, "display", "name")
    name = st.text_input("展示名称", value=record.display_name, key=name_key)
    excluded = st.checkbox(
        "从批量导出中排除此条目", value=record.excluded, key=_key(ws, sid, "display", "excluded")
    )
    if name != record.display_name or excluded != record.excluded:
        ws.records[sid] = replace(record, display_name=name, excluded=excluded)
        ws.last_export_summary = None
    if record.duplicate_candidate:
        st.info("重复上传候选：当前保留。若不需要，请明确勾选排除。")
    dirty = draft_changes(ws, sid)
    if dirty["preparation"]:
        st.warning("单位 / 范围 / 估计平滑有未确认草稿。预览和正式导出使用已确认的准备设置。")
    for warning in state.warnings:
        st.warning(warning)
    for error in state.errors:
        st.error(error)


def _preparation_page(ws: BatchWorkspace, sid: str) -> None:
    stage = "preparation"
    _widget(
        ws,
        sid,
        stage,
        "input_unit",
        "此谱强度单位",
        "selectbox",
        options=UNITS,
        format_func=UNIT_LABELS.get,
    )
    left, right = st.columns(2)
    with left:
        _widget(ws, sid, stage, "high", "处理范围上限 cm⁻¹", "number_input")
    with right:
        _widget(ws, sid, stage, "low", "处理范围下限 cm⁻¹", "number_input")
    _widget(ws, sid, stage, "smoothing", "仅平滑基线估计通道", "checkbox")
    _widget(
        ws,
        sid,
        stage,
        "window_length",
        "估计平滑窗口点数（奇数）",
        "number_input",
        min_value=3,
        step=2,
    )
    _widget(ws, sid, stage, "polyorder", "估计平滑多项式阶数", "number_input", min_value=0)
    _widget(ws, sid, stage, "use_floor", "显式启用透过率 floor 修复", "checkbox")
    _widget(
        ws,
        sid,
        stage,
        "floor",
        "透过率 floor（输入单位）",
        "number_input",
        min_value=1e-15,
        format="%.8g",
    )
    error = ws.states[sid].display_preferences.get("error_" + stage)
    if error:
        st.error(error)
    if st.button("确认此谱单位与处理范围", disabled=bool(error), type="primary") and _attempt(
        confirm_preparation, ws, sid
    ):
        st.success("准备设置已确认；受影响的旧结果已标为过期。")
    r = ws.records[sid]
    fig = go.Figure(go.Scatter(x=r.wavenumber, y=r.raw_intensity, name="原始强度"))
    fig.update_xaxes(title="Wavenumber / cm⁻¹", autorange="reversed")
    fig.update_yaxes(title=UNIT_LABELS[r.confirmed_input_unit])
    st.plotly_chart(fig, width="stretch")
    st.caption("图形缩放只改变显示，不改变处理范围。原始数值始终保存。")


def _coarse_editor(ws: BatchWorkspace, sid: str) -> None:
    _widget(ws, sid, "coarse", "method", "粗调方法", "selectbox", options=COARSE_METHODS)
    _widget(ws, sid, "coarse", "lambda", "Lambda", "number_input", min_value=1e-8, format="%.8g")
    _widget(
        ws,
        sid,
        "coarse",
        "p",
        "AsLS p",
        "number_input",
        min_value=1e-8,
        max_value=0.49999,
        format="%.6g",
    )
    _widget(ws, sid, "coarse", "max_iter", "最大迭代次数", "number_input", min_value=1)
    _widget(ws, sid, "coarse", "tol", "收敛容差", "number_input", min_value=1e-12, format="%.8g")


def _fine_editor(ws: BatchWorkspace, sid: str) -> None:
    method = _widget(ws, sid, "fine", "method", "细调方法", "selectbox", options=FINE_METHODS)
    _widget(ws, sid, "fine", "enabled", "启用细调拟合", "checkbox")
    _widget(
        ws,
        sid,
        "fine",
        "endpoint_window_width_cm1",
        "端点窗口宽度 cm⁻¹",
        "number_input",
        min_value=0.0001,
    )
    _widget(ws, sid, "fine", "statistic", "窗口统计量", "selectbox", options=("median", "mean"))
    _widget(ws, sid, "fine", "strict_endpoint", "使用严格端点（对噪声敏感）", "checkbox")
    _widget(
        ws,
        sid,
        "fine",
        "polynomial_order",
        "细调多项式阶数",
        "number_input",
        min_value=1,
        max_value=3,
    )
    if method in {"piecewise_linear", "pchip", "polynomial"}:
        if st.button("预填覆盖当前范围的锚点"):
            p = ws.states[sid].preparation_committed
            x = ws.records[sid].wavenumber
            x = x[(x >= min(p.wavenumber_range)) & (x <= max(p.wavenumber_range))]
            centers = x[np.linspace(0, x.size - 1, 4, dtype=int)]
            width = min(4.0, float(np.min(np.abs(np.diff(centers)))) / 4)
            _values(ws, sid, "fine")["anchors"] = [
                dict(start=float(c - width), end=float(c + width), enabled=True, statistic="median")
                for c in centers
            ]
            _sync(ws, sid, "fine")
            st.rerun()
        table = pd.DataFrame(
            _values(ws, sid, "fine")["anchors"], columns=["enabled", "start", "end", "statistic"]
        )
        edited = st.data_editor(
            table,
            num_rows="dynamic",
            hide_index=True,
            key=_key(ws, sid, "fine", "anchors"),
            column_config={
                "enabled": st.column_config.CheckboxColumn(default=True),
                "start": st.column_config.NumberColumn(required=True),
                "end": st.column_config.NumberColumn(required=True),
                "statistic": st.column_config.SelectboxColumn(
                    options=["median", "mean"], default="median"
                ),
            },
        )
        _values(ws, sid, "fine")["anchors"] = (
            edited.astype(object).where(edited.notna(), None).to_dict("records")
        )
        _sync(ws, sid, "fine")
        st.caption("锚点窗口必须有实际点，中心须覆盖处理域；允许端点窗口跨出域边界。窗口不得交叠。")


def _plot_snapshot(ws: BatchWorkspace, sid: str, stage: str) -> None:
    state = ws.states[sid]
    choice = st.radio(
        "显示结果",
        ("当前预览", "已确认结果"),
        horizontal=True,
        key=_key(ws, sid, stage, "result_view"),
    )
    snapshot = getattr(state, stage + ("_preview" if choice == "当前预览" else "_snapshot"))
    if snapshot is None:
        st.info("尚无此类结果。调整草稿后点击计算预览，再明确确认。")
        return
    if choice == "已确认结果":
        try:
            get_ready_snapshot(ws, sid, stage)
        except ValueError as exc:
            st.warning(f"历史快照已过期，不显示为当前结果：{exc}")
            return
    else:
        from ftir_workbench.batch.fingerprints import stage_fingerprint
        from ftir_workbench.batch.recipes import coarse_config, fine_config

        try:
            parent = get_ready_snapshot(ws, sid, "coarse") if stage == "fine" else None
            config = (
                coarse_config(state.preparation_committed, state.coarse_draft)
                if parent is None
                else fine_config(parent.config, state.fine_draft)
            )
            if snapshot.fingerprint != stage_fingerprint(
                ws, sid, config, stage, parent.fingerprint if parent else None
            ):
                st.warning("预览已过期，请重新计算后确认。")
                return
        except ValueError as exc:
            st.warning(str(exc))
            return
    result = snapshot.result
    st.caption(
        f"{choice} · {snapshot.config.coarse_baseline.method} / {snapshot.config.fine_baseline.method} · 范围 {snapshot.config.wavenumber_range} cm⁻¹"
    )
    if stage == "coarse":
        st.plotly_chart(coarse_preview_figure(result, 0), width="stretch")
    else:
        fig = fine_residual_figure(result, 0)
        fig.add_trace(
            go.Scatter(
                x=result.absorbance_selected.wavenumber,
                y=result.baseline_estimation_spectra[0] - result.baseline.coarse_baseline[0],
                name="拟合用估计残差（估计通道 − 粗基线）",
                line={"dash": "dash"},
            )
        )
        st.plotly_chart(fig, width="stretch")
        if result.config.baseline_smoothing.enabled:
            st.info("估计平滑已开启：fine 拟合使用估计残差；最终校正仍从未平滑吸光度扣除总基线。")
        st.plotly_chart(fine_decomposition_figure(result, 0), width="stretch")
        diagnostics = anchor_diagnostics_table(result, 0)
        fine_params = result.baseline.params.get("fine", {})
        counts = fine_params.get("anchor_point_counts", ())
        if fine_params.get("method") == "strict_endpoint":
            counts = [1, 1]
        if not counts:
            fitted = fine_params.get("fitted", {})
            counts = [fitted.get("lower_point_count"), fitted.get("upper_point_count")]
        if len(diagnostics) and len(counts) == len(diagnostics):
            diagnostics["实际点数"] = list(counts)
        if not diagnostics.empty:
            st.dataframe(diagnostics, hide_index=True)
    unit = st.selectbox(
        "校正谱派生显示单位",
        UNITS,
        format_func=UNIT_LABELS.get,
        key=_key(ws, sid, stage, "display_unit"),
    )
    try:
        derived = convert_absorbance_for_display(result.analysis_data, unit)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        st.error(f"派生显示失败，吸光度结果保留：{exc}")
        return
    figure = go.Figure(
        go.Scatter(
            x=result.absorbance_selected.wavenumber, y=derived.values[0], name=UNIT_LABELS[unit]
        )
    )
    figure.update_xaxes(autorange="reversed", title="Wavenumber / cm⁻¹")
    figure.update_yaxes(title=UNIT_LABELS[unit])
    st.plotly_chart(figure, width="stretch")
    st.caption("基线分解始终为吸光度；派生透过率不裁剪负吸光度产生的 >100%T。")
    for warning in derived.warnings:
        st.warning(warning)
    metrics = result.qc.as_dict()["per_spectrum"]
    skip = {"perturbation", "spectrum_index", "adjacent_baseline_rms"}
    st.dataframe(
        pd.DataFrame(
            [
                {
                    k: v[0] if v and v[0] is not None else "N/A"
                    for k, v in metrics.items()
                    if k not in skip
                }
            ]
        ),
        hide_index=True,
    )
    st.caption(
        "时间连续性、相邻谱变化和扰动趋势：不适用。无峰区指标显示 N/A，诊断警告不自动删除或裁负值。"
    )


def _processing_page(ws: BatchWorkspace, sid: str, stage: str) -> None:
    _coarse_editor(ws, sid) if stage == "coarse" else _fine_editor(ws, sid)
    state = ws.states[sid]
    error = state.display_preferences.get("error_" + stage)
    if error:
        st.error(error)
    title = "粗调" if stage == "coarse" else "细调"
    preview = preview_coarse if stage == "coarse" else preview_fine
    confirm = confirm_coarse if stage == "coarse" else confirm_fine
    disabled = bool(error) or (
        stage == "fine" and (not state.fine_draft.enabled or state.fine_draft.method == "none")
    )
    left, center, right = st.columns(3)
    if left.button(f"计算{title}预览", disabled=disabled, type="primary") and _attempt(
        preview, ws, sid
    ):
        st.session_state[_key(ws, sid, stage, "result_view")] = "当前预览"
    if center.button(f"确认当前{title}", disabled=bool(error)) and _attempt(confirm, ws, sid):
        st.session_state[_key(ws, sid, stage, "result_view")] = "已确认结果"
        st.success(f"当前{title}已确认保存。")
    if right.button(f"确认{title}并下一条", disabled=bool(error)) and _attempt(confirm, ws, sid):
        index = ws.display_order.index(sid)
        ws.selected_spectrum_id = ws.display_order[(index + 1) % len(ws.display_order)]
        st.session_state["batch_selection_revision"] = (
            st.session_state.get("batch_selection_revision", 0) + 1
        )
        for key in list(st.session_state):
            if str(key).startswith(f"batch_current_{ws.workspace_id}_"):
                del st.session_state[key]
        st.rerun()
    if stage == "fine" and st.button("确认不做细调") and _attempt(skip_fine, ws, sid):
        st.session_state[_key(ws, sid, stage, "result_view")] = "已确认结果"
        st.success("已确认跳过细调，最终结果使用当前已确认粗调数组。")
    if st.button(f"恢复已确认{title}参数") and _attempt(restore_committed_draft, ws, sid, stage):
        _reset_editor(ws, sid, stage)
        st.rerun()
    with st.expander("复制参数与批量预览"):
        st.caption("仅复制独立草稿。单位、原始数据、已确认结果不随参数复制。")
        copy_range = st.checkbox("同时复制处理范围草稿", key="copy_range_" + stage)
        copy_smooth = st.checkbox("同时复制估计平滑草稿", key="copy_smooth_" + stage)
        targets = st.session_state.get("batch_target_ids", [])
        if st.button(f"复制{title}参数到选中项", disabled=not targets or bool(error)):
            outcomes = copy_parameters(
                ws,
                sid,
                targets,
                stage=stage,
                include_range=copy_range,
                include_smoothing=copy_smooth,
            )
            st.session_state["batch_operation_report"] = outcomes
            for target, failure in outcomes.items():
                if failure is None:
                    _reset_editor(ws, target, stage)
                    if copy_range or copy_smooth:
                        _reset_editor(ws, target, "preparation")
            st.rerun()
        if st.button(f"为选中项计算各自{title}预览", disabled=not targets):
            with st.spinner("按各条自己的配方逐谱计算预览…"):
                errors = {
                    target: ws.states[target].display_preferences.get("error_" + stage)
                    for target in targets
                }
                ready = [target for target in targets if not errors[target]]
                outcomes = preview_selected(ws, ready, stage=stage)
                outcomes.update(
                    {
                        target: "草稿无效，未运行：" + error
                        for target, error in errors.items()
                        if error
                    }
                )
                st.session_state["batch_operation_report"] = outcomes
        if st.session_state.get("batch_operation_report"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "光谱": ws.records[item].display_name,
                            "结果": error or "成功（未自动确认）",
                        }
                        for item, error in st.session_state["batch_operation_report"].items()
                        if item in ws.records
                    ]
                ),
                hide_index=True,
            )
    _plot_snapshot(ws, sid, stage)
    _export_controls(ws, sid, stage)


def _workspace_context(
    ws: BatchWorkspace, *, for_save: bool = False, spectrum_ids: list[str] | None = None
) -> str:
    from ftir_workbench.batch.fingerprints import json_fingerprint

    payload: dict[str, Any] = {
        "workspace": ws.workspace_id,
        "order": ws.display_order if spectrum_ids is None else spectrum_ids,
        "import_issues": ws.import_issues,
        "records": {
            sid: {
                "input": ws.records[sid].scientific_input_sha256,
                "source": ws.records[sid].source_id,
                "name": ws.records[sid].display_name,
                "excluded": ws.records[sid].excluded,
                "drafts": [
                    getattr(state, field + "_draft").to_dict()
                    for field in ("preparation", "coarse", "fine")
                ],
                "snapshots": [
                    getattr(state, stage + "_snapshot").fingerprint
                    if getattr(state, stage + "_snapshot")
                    else None
                    for stage in ("coarse", "fine")
                ],
                "stale": [state.coarse_stale, state.fine_stale],
                "decision": state.fine_decision,
                "errors": state.errors,
                "warnings": state.warnings,
                "editor_errors": {
                    k: v for k, v in state.display_preferences.items() if k.startswith("error_")
                },
            }
            for sid, state in ws.states.items()
            if spectrum_ids is None or sid in spectrum_ids
        },
    }
    if for_save:
        payload.update(
            selected=ws.selected_spectrum_id,
            display={sid: state.display_preferences for sid, state in ws.states.items()},
            exports=ws.export_history,
            summary=ws.last_export_summary,
            import_issues=ws.import_issues,
        )
    return json_fingerprint(payload)


def _workspace_restore(ws: BatchWorkspace) -> None:
    from ftir_workbench.batch.workspace import load_batch_workspace

    with st.expander("恢复普通光谱工作区"):
        uploaded = st.file_uploader(
            "恢复普通工作区 ZIP", type=["zip"], key="batch_workspace_upload"
        )
        st.caption("此入口仅接收本模式的工作区包；恢复会替换当前普通工作区。原位结果独立保留。")
        if st.button("恢复工作区", disabled=uploaded is None):
            try:
                restored = load_batch_workspace(uploaded.getvalue())
                st.session_state.batch_workspace = restored
                st.session_state["batch_target_ids"] = []
                for key in (
                    "batch_save_download",
                    "batch_download_coarse",
                    "batch_download_fine",
                    "batch_operation_report",
                ):
                    st.session_state.pop(key, None)
                # Restoring the same workspace ID must still restore every saved draft widget.
                for key in list(st.session_state):
                    if str(key).startswith(("_batch_", "batch_current_", "batch_targets_")):
                        del st.session_state[key]
                st.session_state["batch_restore_message"] = (
                    f"已恢复工作区：{len(restored.records)} 条光谱；正式结果与草稿分别恢复，预览待重新请求。"
                )
                st.rerun()
            except (ValueError, TypeError, KeyError) as exc:
                st.error(str(exc))


def _export_controls(ws: BatchWorkspace, sid: str | None, stage: str) -> None:
    from ftir_workbench.batch.export import build_batch_export, can_export_wide

    title = "粗调" if stage == "coarse" else "最终"
    st.divider()
    st.subheader(f"已确认{title}结果导出")
    scope = st.selectbox(
        f"{title}导出范围", ("当前光谱", "勾选条目", "全部条目"), key="export_scope_" + stage
    )
    ids = (
        ([sid] if sid else [])
        if scope == "当前光谱"
        else (
            st.session_state.get("batch_target_ids", [])
            if scope == "勾选条目"
            else ws.display_order
        )
    )
    ready_only = st.checkbox("仅导出已就绪项（报告列出全部排除项）", key="export_ready_" + stage)
    wide_allowed, reason = can_export_wide(ws, ids, stage=stage, ready_only=ready_only)
    wide = st.checkbox("同时提供同轴宽表", disabled=not wide_allowed, key="export_wide_" + stage)
    if not wide_allowed:
        st.caption(f"宽表不可用：{reason}")
    derived = st.multiselect(
        "附加派生透过率副本", UNITS[1:], format_func=UNIT_LABELS.get, key="export_derived_" + stage
    )
    if any(
        any(draft_changes(ws, item).values())
        or any(k.startswith("error_") for k in ws.states[item].display_preferences)
        for item in ids
    ):
        st.info("存在未提交或无效草稿。此导出只使用各条已确认的正式数组，草稿不会被采用。")
    options = (tuple(ids), ready_only, bool(wide and wide_allowed), tuple(derived))
    context = (_workspace_context(ws, spectrum_ids=list(ids)), options)
    cache_key = "batch_download_" + stage
    if st.button(f"打包已确认{title}结果", disabled=not ids):
        try:
            artifact = build_batch_export(
                ws,
                ids,
                stage=stage,
                ready_only=ready_only,
                include_wide=bool(wide and wide_allowed),
                derived_units=derived,
            )
            st.session_state[cache_key] = (context, artifact)
            ws.last_export_summary = artifact.metadata
            ws.export_history.append(
                {
                    "stage": stage,
                    "requested": list(ids),
                    "exported": artifact.metadata["exported_spectrum_ids"],
                }
            )
        except (ValueError, TypeError, KeyError, FloatingPointError) as exc:
            st.session_state.pop(cache_key, None)
            st.error(str(exc))
    cached = st.session_state.get(cache_key)
    if cached and cached[0] == context:
        artifact = cached[1]
        st.download_button(
            f"下载已确认{title} ZIP",
            data=artifact.zip_bytes,
            file_name=artifact.filename,
            mime="application/zip",
            key="download_zip_" + stage,
        )
        if artifact.wide_csv_bytes is not None:
            st.download_button(
                f"下载{title}同轴宽表 CSV",
                data=artifact.wide_csv_bytes,
                file_name=f"independent_{stage}_wide.csv",
                mime="text/csv",
                key="download_wide_" + stage,
            )
        st.dataframe(pd.DataFrame(artifact.report), hide_index=True)
    elif cached:
        st.caption("上次导出包的条目、正式结果或报告信息已变化，请重新打包。")


def _workspace_save(ws: BatchWorkspace) -> None:
    from ftir_workbench.batch.workspace import save_batch_workspace

    st.divider()
    st.subheader("保存工作区")
    st.caption("保存原始文件、稳定身份、各条草稿及正式数组。保存工作区不等于确认草稿。")
    context = _workspace_context(ws, for_save=True)
    if st.button("准备工作区下载", disabled=not ws.records):
        try:
            payload = save_batch_workspace(ws)
            st.session_state["batch_save_download"] = (context, payload)
        except (ValueError, TypeError, KeyError) as exc:
            st.error(str(exc))
    cached = st.session_state.get("batch_save_download")
    if cached and cached[0] == context:
        st.download_button(
            "保存工作区 ZIP",
            data=cached[1],
            file_name=f"independent_workspace_{ws.workspace_id[:8]}.zip",
            mime="application/zip",
        )
    elif cached:
        st.info("工作区状态已变化，请重新准备工作区下载。")


def render_batch_workflow() -> None:
    if "batch_workspace" not in st.session_state:
        st.session_state.batch_workspace = BatchWorkspace()
    ws = st.session_state.batch_workspace
    if st.session_state.get("batch_restore_message"):
        st.success(st.session_state["batch_restore_message"])
    page = st.sidebar.radio("工作流", PAGES, key="batch_page")
    st.sidebar.caption("普通光谱 → 逐谱粗调 / 细调 → CSV / ZIP / 工作区")
    sid = _sidebar_selection(ws)
    if page == PAGES[0]:
        _import_page(ws)
    elif page == PAGES[4]:
        st.subheader(PAGES[4])
        st.dataframe(spectrum_table(ws), hide_index=True, width="stretch")
        _export_controls(ws, sid, "coarse")
        _export_controls(ws, sid, "fine")
        _workspace_save(ws)
        _workspace_restore(ws)
    elif sid is None:
        st.info("请先导入普通光谱，或调整搜索 / 状态筛选。")
    else:
        _record_header(ws, sid)
        if page == PAGES[1]:
            _preparation_page(ws, sid)
        else:
            _processing_page(ws, sid, "coarse" if page == PAGES[2] else "fine")
