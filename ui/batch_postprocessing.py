"""Ordinary postprocessing controls; only explicit preview buttons run science."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ftir_workbench.batch.fingerprints import json_fingerprint
from ftir_workbench.batch.models import BatchWorkspace
from ftir_workbench.batch.postprocessing import (
    BRANCHES,
    DERIVED_BRANCHES,
    PARENTS,
    PostprocessSnapshot,
    branch_arrays,
    branch_state,
    branch_status,
    confirm_branch,
    confirm_selected,
    copy_drafts,
    display_arrays,
    effective_config,
    get_branch,
    get_postprocessing_state,
    plain,
    preview_branch,
    preview_selected,
    update_draft,
)
from ftir_workbench.batch.state import draft_changes
from ftir_workbench.display_units import DisplayIntensityUnit

BRANCH_LABELS = {
    "baseline": "B · 最终基线校正谱",
    "smoothed": "S · 平滑校正谱",
    "normalized_baseline": "N_B · 基线后归一化",
    "normalized_smoothed": "N_S · 平滑后归一化",
}
METHOD_LABELS = {
    "none": "不做归一化",
    "maximum": "全范围最大正峰高",
    "internal_peak_height": "参考窗口正峰高",
    "internal_peak_area": "参考窗口面积",
    "area": "全范围 / 指定范围面积",
    "vector": "L2 向量归一化",
    "minmax_display": "Min–Max 0–1（仅显示缩放）",
}
STATUS_LABELS = {"missing": "未创建", "ready": "已确认", "stale": "父级过期"}
EXPORT_LABELS = {**BRANCH_LABELS, "all": "全部分支（B / S / N_B / N_S）"}


def _key(ws: BatchWorkspace, sid: str, branch: str, field: str) -> str:
    return f"_batch_{ws.workspace_id}_{sid}_post_{branch}_{field}"


def _changed(ws: BatchWorkspace, sid: str, branch: str, field: str, key: str,
             index: int | None) -> None:
    value = st.session_state[key]
    if index is not None:
        interval = list(branch_state(ws, sid, branch).draft.get(field) or [None, None])
        interval[index] = value
        value = interval
    update_draft(ws, sid, branch, {field: value})


def _control(ws: BatchWorkspace, sid: str, branch: str, field: str, label: str,
             kind: str, *, index: int | None = None, **kwargs: Any) -> Any:
    key = _key(ws, sid, branch, field + ("_" + str(index) if index is not None else ""))
    if key not in st.session_state:
        value = branch_state(ws, sid, branch).draft.get(field)
        st.session_state[key] = (value or [None, None])[index] if index is not None else value
    return getattr(st, kind)(label, key=key, on_change=_changed,
                             args=(ws, sid, branch, field, key, index), **kwargs)


def _preference_changed(ws: BatchWorkspace, sid: str, name: str, key: str) -> None:
    get_postprocessing_state(ws, sid).display_preferences[name] = st.session_state[key]


def _preference(ws: BatchWorkspace, sid: str, name: str, label: str,
                options: tuple[str, ...], default: str) -> str:
    prefs = get_postprocessing_state(ws, sid).display_preferences
    key = _key(ws, sid, "view", name)
    if key not in st.session_state:
        value = prefs.get(name, default)
        st.session_state[key] = value if value in options else default
    return str(st.selectbox(label, options, key=key, on_change=_preference_changed,
                            args=(ws, sid, name, key)))


def reset_postprocessing_editor(ws: BatchWorkspace, sid: str, branch: str) -> None:
    prefix = _key(ws, sid, branch, "")
    for key in list(st.session_state):
        if str(key).startswith(prefix):
            del st.session_state[key]


def postprocessing_save_context(ws: BatchWorkspace) -> dict[str, Any]:
    """JSON-only download identity; previews intentionally are not persisted."""
    return {
        sid: {
            "branches": {
                branch: {
                    "draft": getattr(state, branch).draft,
                    "committed": getattr(state, branch).committed.fingerprint
                    if getattr(state, branch).committed else None,
                    "committed_draft": getattr(state, branch).committed_draft,
                    "errors": getattr(state, branch).errors,
                } for branch in DERIVED_BRANCHES
            },
            "normalization_source": state.normalization_source,
            "export_choice": state.export_choice,
            "display_preferences": state.display_preferences,
        } for sid, state in ws.postprocessing.items()
    }


def _export_choice_changed(ws: BatchWorkspace, sid: str, key: str) -> None:
    get_postprocessing_state(ws, sid).export_choice = st.session_state[key]


def _export_preflight(ws: BatchWorkspace, ids: list[str], branches: tuple[str, ...]) -> list[dict[str, Any]]:
    """Expose every unavailable target before a user creates a download snapshot."""
    rows = []
    for target in ids:
        record = ws.records[target]
        for branch in branches:
            status = branch_status(ws, target, branch)
            error = "条目已明确排除" if record.excluded else status["reason"]
            ready = not record.excluded and status["status"] == "ready"
            recorded_errors = (ws.states[target].errors if branch == "baseline"
                               else branch_state(ws, target, branch).errors)
            snapshot = (ws.states[target].fine_snapshot if branch == "baseline"
                        else branch_state(ws, target, branch).committed)
            warnings = list(snapshot.warnings) if isinstance(snapshot, PostprocessSnapshot) else []
            rows.append({"光谱": record.display_name, "光谱 ID": target,
                         "分支": BRANCH_LABELS[branch], "将被导出": ready,
                         "状态 / 排除原因": "已确认可用" if ready else error,
                         "已记录错误 / 警告": "; ".join([*recorded_errors, *warnings]),
                         "存在未确认草稿": (any(draft_changes(ws, target).values())
                                         if branch == "baseline" else bool(status["draft_modified"]))})
    return rows


def _post_export_context(ws: BatchWorkspace, ids: list[str], branches: tuple[str, ...],
                         valid_only: bool, wide: bool) -> str:
    post = postprocessing_save_context(ws)
    return json_fingerprint({
        "workspace": ws.workspace_id, "ids": ids, "branches": branches,
        "valid_only": valid_only, "wide": wide,
        "items": {
            sid: {
                "name": ws.records[sid].display_name,
                "source_id": ws.records[sid].source_id,
                "source_filename": ws.sources[ws.records[sid].source_id].original_filename,
                "input": ws.records[sid].scientific_input_sha256,
                "excluded": ws.records[sid].excluded,
                "baseline": ws.states[sid].fine_snapshot.fingerprint if ws.states[sid].fine_snapshot else None,
                "baseline_status": branch_status(ws, sid, "baseline"),
                "baseline_drafts": {stage: getattr(ws.states[sid], stage + "_draft").to_dict()
                                    for stage in ("preparation", "coarse", "fine")},
                "baseline_editor_errors": {name: value for name, value in ws.states[sid].display_preferences.items()
                                           if name.startswith("error_")},
                "postprocessing": post.get(sid),
                "derived_status": {branch: branch_status(ws, sid, branch) for branch in DERIVED_BRANCHES},
            } for sid in ids
        },
    })


def _export_controls(ws: BatchWorkspace, sid: str) -> None:
    from ftir_workbench.batch.postprocessing_export import (
        build_postprocessing_export,
        can_export_postprocessing_wide,
    )

    state = get_postprocessing_state(ws, sid)
    key = _key(ws, sid, "export", "choice")
    if key not in st.session_state:
        st.session_state[key] = state.export_choice
    choice = st.selectbox("后处理导出分支", tuple(EXPORT_LABELS), format_func=EXPORT_LABELS.get,
                          key=key, on_change=_export_choice_changed, args=(ws, sid, key))
    scope = _preference(ws, sid, "post_export_scope", "后处理导出范围",
                        ("当前光谱", "勾选条目", "全部条目"), "当前光谱")
    ids = ([sid] if scope == "当前光谱" else list(ws.selected_spectrum_ids)
           if scope == "勾选条目" else list(ws.display_order))
    branches = tuple(BRANCHES) if choice == "all" else (choice,)
    valid_key = _key(ws, sid, "export", "valid_only")
    if valid_key not in st.session_state:
        st.session_state[valid_key] = bool(state.display_preferences.get("post_export_valid_only", False))
    checked = st.checkbox("只导出有效项（明确排除下表不可用项）", key=valid_key,
                          on_change=_preference_changed,
                          args=(ws, sid, "post_export_valid_only", valid_key))
    valid_only = bool(checked)
    if choice == "all":
        st.info("已选择全部分支；明确勾选只导出有效项后，才会跳过下表不可用项并保留完整报告。"
                if valid_only else "已选择全部分支：所有目标的 B / S / N_B / N_S 均须有效；存在不可用项时默认阻止打包。")
    report = _export_preflight(ws, ids, branches)
    st.dataframe(pd.DataFrame(report), hide_index=True, width="stretch")
    invalid = [row for row in report if not row["将被导出"]]
    if invalid:
        st.warning("上表包含不可用项。指定分支不会退回其他来源；需明确排除不可用项才能打包其余正式结果。")
    if any(row["存在未确认草稿"] for row in report):
        st.info("存在未确认草稿，导出只使用当前有效的已确认版本。")
    wide_allowed, reason = can_export_postprocessing_wide(ws, ids, branches=branches, valid_only=valid_only)
    wide_key = _key(ws, sid, "export", "wide")
    if wide_key not in st.session_state:
        st.session_state[wide_key] = bool(state.display_preferences.get("post_export_wide", False))
    wide = bool(st.checkbox("提供后处理同轴宽表", key=wide_key, disabled=not wide_allowed,
                            on_change=_preference_changed, args=(ws, sid, "post_export_wide", wide_key))
                and wide_allowed)
    if not wide_allowed:
        st.caption("后处理宽表不可用：" + reason)
    st.caption("ZIP 按谱保留原轴；N_B 包含 B，N_S 包含 B 与 S。归一化 / Min–Max 不导出为物理 T 或 %T。")
    context = _post_export_context(ws, ids, branches, valid_only, wide)
    cache_key = _key(ws, sid, "export", "download")
    disabled = not report or not any(row["将被导出"] for row in report) or (bool(invalid) and not valid_only)
    if st.button("打包已确认后处理结果", key=_key(ws, sid, "export", "build"), disabled=disabled):
        try:
            artifact = build_postprocessing_export(ws, ids, branches=branches, valid_only=valid_only,
                                                   include_wide=wide)
            st.session_state[cache_key] = (context, artifact)
            ws.last_export_summary = artifact.metadata
            ws.export_history.append({"artifact": "ordinary_postprocess", "requested": ids,
                                      "branches": list(branches), "valid_only": valid_only})
        except (ValueError, TypeError, KeyError, FloatingPointError) as exc:
            st.error(str(exc))
    cached = st.session_state.get(cache_key)
    if cached and cached[0] == context:
        artifact = cached[1]
        st.download_button("下载已确认后处理 ZIP", data=artifact.zip_bytes, file_name=artifact.filename,
                           mime="application/zip", key=_key(ws, sid, "export", "zip"))
        if artifact.wide_csv_bytes is not None:
            st.download_button("下载后处理同轴宽表 CSV", data=artifact.wide_csv_bytes,
                               file_name="ordinary_postprocessing_wide.csv", mime="text/csv",
                               key=_key(ws, sid, "export", "wide_csv"))
        st.dataframe(pd.DataFrame(artifact.report), hide_index=True, width="stretch")
        for warning in artifact.metadata.get("warnings", []):
            st.warning(str(warning))
    elif cached:
        st.info("条目、草稿、来源或显示名称已变化，请重新打包下载；已生成的文件内容保持不变。")


def _status_text(ws: BatchWorkspace, sid: str, branch: str) -> str:
    status = branch_status(ws, sid, branch)
    text = STATUS_LABELS[status["status"]]
    state = branch_state(ws, sid, branch)
    if status["preview_current"] and state.committed is None:
        text = "预览未确认"
    if state.errors:
        text += " · 计算失败"
    if status["draft_modified"] and state.committed is not None:
        text += " · 草稿与确认版不同"
    if not state.draft.get("enabled") and state.committed is None:
        text += " · 未启用"
    return text


def postprocessing_table(ws: BatchWorkspace) -> pd.DataFrame:
    rows = []
    for sid in ws.display_order:
        state = get_postprocessing_state(ws, sid)
        baseline = branch_status(ws, sid, "baseline")
        for branch in DERIVED_BRANCHES:
            part = branch_state(ws, sid, branch)
            status = branch_status(ws, sid, branch)
            snapshot = part.committed
            rows.append({
                "名称": ws.records[sid].display_name,
                "光谱 ID": sid,
                "B 状态": STATUS_LABELS[baseline["status"]],
                "分支": BRANCH_LABELS[branch],
                "状态": _status_text(ws, sid, branch),
                "草稿方法": part.draft.get("method"),
                "正式方法": snapshot.recipe.get("method") if snapshot else "—",
                "当前归一化来源": BRANCH_LABELS[state.normalization_source],
                "参考 / 面积区间": str(part.draft.get("reference_interval")
                                        or part.draft.get("area_interval") or "完整父谱范围"),
                "未确认草稿": bool(status["draft_modified"]),
                "warning / error": "; ".join([*part.errors, *(snapshot.warnings if snapshot else ()),
                                               status["reason"]]),
            })
    return pd.DataFrame(rows)


def _smoothing_editor(ws: BatchWorkspace, sid: str) -> None:
    branch = "smoothed"
    _control(ws, sid, branch, "enabled", "启用普通谱平滑", "checkbox")
    method = _control(ws, sid, branch, "method", "普通谱平滑方法", "selectbox",
                      options=("savgol", "gaussian", "moving_average", "median"))
    if method == "savgol":
        _control(ws, sid, branch, "savgol_window_length", "SG 窗口点数（奇数）", "number_input", step=2)
        _control(ws, sid, branch, "savgol_polyorder", "SG 多项式阶数", "number_input", step=1)
        _control(ws, sid, branch, "savgol_mode", "SG 边界模式", "selectbox",
                 options=("interp", "mirror", "nearest"))
    elif method == "gaussian":
        _control(ws, sid, branch, "gaussian_sigma_points", "Gaussian σ（点数）", "number_input", step=0.1)
        _control(ws, sid, branch, "gaussian_truncate", "Gaussian 截断倍数", "number_input", step=0.5)
    else:
        field = "moving_average_window_length" if method == "moving_average" else "median_window_length"
        _control(ws, sid, branch, field, "平滑窗口点数（奇数）", "number_input", step=2)
    if method != "savgol":
        _control(ws, sid, branch, "convolution_mode", "卷积边界模式", "selectbox",
                 options=("reflect", "mirror", "nearest"))
    with st.expander("采样轴专家选项"):
        _control(ws, sid, branch, "uniformity_rtol", "采样间距相对容差", "number_input", step=0.0001, format="%.6f")
        policy = _control(ws, sid, branch, "nonuniform_axis_policy", "非均匀轴处理", "selectbox",
                          options=("error", "allow_index_space_with_warning"),
                          format_func=lambda value: "默认拒绝非均匀轴" if value == "error" else "明确允许按索引处理（有警告）")
        if policy != "error":
            st.warning("专家覆盖按采样点索引滤波；不插值，也不代表恒定物理窗口。不支持跨不连续分段。")
    st.caption("仅沿当前 B 的完整波数轴处理。图形缩放不改变范围；窗口宽度不是仪器分辨率。")
    _width_hint(ws, sid)


def _width_hint(ws: BatchWorkspace, sid: str) -> None:
    try:
        x, _ = branch_arrays(get_branch(ws, sid, "baseline"))
        draft = branch_state(ws, sid, "smoothed").draft
        spacing = float(np.median(np.abs(np.diff(x))))
        if draft.get("method") == "gaussian":
            sigma = float(draft["gaussian_sigma_points"]) * spacing
            text = f"Gaussian σ ≈ {sigma:.6g} cm⁻¹；FWHM ≈ {2.35482 * sigma:.6g} cm⁻¹"
        else:
            field = {"savgol": "savgol_window_length", "moving_average": "moving_average_window_length",
                     "median": "median_window_length"}[draft["method"]]
            text = f"窗口跨度 ≈ {(float(draft[field]) - 1) * spacing:.6g} cm⁻¹"
        st.caption(f"当前谱中位采样间距 {spacing:.6g} cm⁻¹；{text}（按中位间距估算）。")
    except (ValueError, TypeError, KeyError):
        st.caption("获得有效最终 B 与窗口参数后显示当前谱物理窗口。")


def _area_scope_changed(ws: BatchWorkspace, sid: str, branch: str, key: str) -> None:
    state = get_postprocessing_state(ws, sid)
    draft = branch_state(ws, sid, branch).draft
    saved = branch + "_area_interval"
    if st.session_state[key]:
        value = state.display_preferences.get(saved, [None, None])
    else:
        state.display_preferences[saved] = draft.get("area_interval")
        value = None
    update_draft(ws, sid, branch, {"area_interval": value})


def _normalization_editor(ws: BatchWorkspace, sid: str, branch: str) -> None:
    _control(ws, sid, branch, "enabled", "启用当前来源归一化", "checkbox")
    method = _control(ws, sid, branch, "method", "普通谱归一化方法", "selectbox",
                      options=tuple(METHOD_LABELS), format_func=METHOD_LABELS.get)
    interval_field = None
    if method in {"internal_peak_height", "internal_peak_area"}:
        interval_field = "reference_interval"
        st.caption("参考窗口必须完整位于父谱内且至少覆盖两个实测点；不插值端点。")
    elif method == "area":
        key = _key(ws, sid, branch, "area_scope")
        if key not in st.session_state:
            st.session_state[key] = branch_state(ws, sid, branch).draft.get("area_interval") is not None
        if st.checkbox("面积使用指定区间", key=key, on_change=_area_scope_changed,
                       args=(ws, sid, branch, key)):
            interval_field = "area_interval"
    if interval_field:
        _control(ws, sid, branch, interval_field, "归一化区间下界（cm⁻¹）", "number_input", index=0, step=1.0)
        _control(ws, sid, branch, interval_field, "归一化区间上界（cm⁻¹）", "number_input", index=1, step=1.0)
    if method in {"internal_peak_area", "area"}:
        _control(ws, sid, branch, "area_definition", "积分面积定义", "selectbox",
                 options=("absolute", "signed"),
                 format_func=lambda value: "绝对面积 ∫|y|dx" if value == "absolute" else "有符号面积 ∫y dx（须为正且不严重抵消）")
    if method not in {"none", "minmax_display"}:
        _control(ws, sid, branch, "target", "归一化目标值", "number_input", step=0.1)
    if method != "none":
        _control(ws, sid, branch, "minimum_reference", "最小有效参考值（0 使用数值保护）", "number_input", step=0.001, format="%.6g")
    if method == "minmax_display":
        st.info("Min–Max 固定缩放至 0–1，仅用于显示比较；结果不再标为吸光度或物理透过率。")
    elif method == "vector":
        st.caption("L2 按完整父谱的采样点平方和计算，不含波数积分权重；不同网格不代表相同物理尺度。")
    if method in {"maximum", "internal_peak_height"}:
        st.caption("以最大正峰为参考，不以负谷的绝对值作分母；全负 / 全零输入会拒绝。")
    if method.startswith("internal_"):
        st.caption("参考峰需由实验知识选择。单谱内部参考 CV=0 不能验证跨样品内标稳定性。")


def _next(ws: BatchWorkspace, sid: str) -> None:
    ws.selected_spectrum_id = ws.display_order[(ws.display_order.index(sid) + 1) % len(ws.display_order)]
    st.session_state["batch_selection_revision"] = st.session_state.get("batch_selection_revision", 0) + 1
    for key in list(st.session_state):
        if str(key).startswith(f"batch_current_{ws.workspace_id}_"):
            del st.session_state[key]
    st.rerun()


def _actions(ws: BatchWorkspace, sid: str, branch: str) -> None:
    title = "平滑" if branch == "smoothed" else "归一化"
    state = branch_state(ws, sid, branch)
    status = branch_status(ws, sid, branch)
    problem = ""
    try:
        effective_config(branch, state.draft)
        get_branch(ws, sid, PARENTS[branch])
    except (ValueError, TypeError, KeyError) as exc:
        problem = str(exc)
    if problem:
        st.info("当前草稿 / 来源尚不可预览：" + problem)
    st.write(f"{BRANCH_LABELS[branch]}：{_status_text(ws, sid, branch)}")
    if status["draft_modified"] and state.committed is not None:
        st.warning("存在未确认修改；父级仍有效时，导出继续采用已确认版本。")
    if status["status"] == "stale":
        st.warning("历史确认结果的父级已过期，不能作为当前结果导出。" + status["reason"])
    left, center, right = st.columns(3)
    if left.button(f"计算当前{title}预览", key=_key(ws, sid, branch, "preview"), disabled=bool(problem), type="primary"):
        try:
            preview_branch(ws, sid, branch)
            get_postprocessing_state(ws, sid).display_preferences[branch + "_result"] = "当前预览"
            st.session_state.pop(_key(ws, sid, "view", branch + "_result"), None)
            st.rerun()
        except (ValueError, TypeError, KeyError) as exc:
            st.error(str(exc))
    if center.button(f"确认当前{title}", key=_key(ws, sid, branch, "confirm"), disabled=not status["preview_current"]):
        confirm_branch(ws, sid, branch)
        get_postprocessing_state(ws, sid).display_preferences[branch + "_result"] = "已确认结果"
        st.session_state.pop(_key(ws, sid, "view", branch + "_result"), None)
        st.rerun()
    if right.button(f"确认{title}并下一条", key=_key(ws, sid, branch, "next"), disabled=not status["preview_current"]):
        confirm_branch(ws, sid, branch)
        _next(ws, sid)
    if st.button(f"恢复已确认{title}参数", key=_key(ws, sid, branch, "restore"), disabled=state.committed_draft is None):
        assert state.committed_draft is not None
        update_draft(ws, sid, branch, state.committed_draft)
        reset_postprocessing_editor(ws, sid, branch)
        st.rerun()
    with st.expander(f"{title}参数复制与批量操作"):
        st.caption("复制仅写各条草稿；按同一语义来源逐目标检查，不复制父 hash、系数或结果。")
        targets = st.session_state.get("batch_target_ids", [])
        operations = (
            (f"复制{title}参数到选中项", "copy"),
            (f"为选中项计算各自{title}预览", "preview"),
            (f"确认选中项有效{title}预览", "confirm"),
        )
        for label, operation in operations:
            if st.button(label, key=_key(ws, sid, branch, "batch_" + operation), disabled=not targets):
                if operation == "copy":
                    report = copy_drafts(ws, sid, targets, branch=branch)
                    for target in targets:
                        reset_postprocessing_editor(ws, target, branch)
                else:
                    action = preview_selected if operation == "preview" else confirm_selected
                    report = action(ws, targets, branch=branch)
                get_postprocessing_state(ws, sid).display_preferences[branch + "_report"] = {
                    "operation": operation, "outcomes": report,
                }
                st.rerun()
        saved = get_postprocessing_state(ws, sid).display_preferences.get(branch + "_report")
        if saved:
            rows = []
            for target, error in saved["outcomes"].items():
                if target not in ws.records:
                    continue
                part = branch_state(ws, target, branch)
                snapshot = part.preview if saved["operation"] == "preview" else part.committed
                rows.append({"光谱": ws.records[target].display_name, "光谱 ID": target,
                             "结果": error or ("已确认" if saved["operation"] == "confirm" else "成功（未自动确认）"),
                             "当前状态": _status_text(ws, target, branch),
                             "物理窗口（cm⁻¹）": str(plain(snapshot.qc.get("approximate_physical_width", {}))) if snapshot else "—"})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
    for error in state.errors:
        st.warning("上次计算失败：" + error)


def _figure(x: np.ndarray, curves: list[tuple[str, np.ndarray]], title: str, ylabel: str) -> go.Figure:
    figure = go.Figure()
    for label, y in curves:
        figure.add_trace(go.Scatter(x=x, y=y, name=label, mode="lines"))
    figure.update_layout(title=title, xaxis_title="Wavenumber (cm⁻¹)", yaxis_title=ylabel,
                         height=340, legend={"orientation": "h"})
    figure.update_xaxes(autorange="reversed")
    return figure


def _plots(ws: BatchWorkspace, sid: str, branch: str) -> None:
    state = branch_state(ws, sid, branch)
    if state.preview is None and state.committed is None:
        return
    view = _preference(ws, sid, branch + "_result", "显示后处理结果", ("当前预览", "已确认结果"),
                       "当前预览" if state.preview is not None else "已确认结果")
    snapshot: PostprocessSnapshot | None = state.preview if view == "当前预览" else state.committed
    if snapshot is None:
        st.info("所选版本尚不存在，请先预览或确认。")
        return
    status = branch_status(ws, sid, branch)
    if view == "当前预览" and not status["preview_current"]:
        st.warning("正在查看历史预览；它不匹配当前草稿或父级，不能直接确认。")
    st.caption(f"{view} · {BRANCH_LABELS[branch]} · 版本 {snapshot.fingerprint[:12]} · 父级 {snapshot.parent_fingerprint[:12]}")
    parent = snapshot.parent_smoothed or snapshot.baseline
    if branch == "smoothed":
        unit = _preference(ws, sid, "smoothing_unit", "B / S 显示单位",
                           ("absorbance", "percent_transmittance", "fraction_transmittance"), "absorbance")
        x = snapshot.wavenumber
        try:
            _, source, ylabel = display_arrays(parent, cast(DisplayIntensityUnit, unit))
            _, output, _ = display_arrays(snapshot, cast(DisplayIntensityUnit, unit))
            st.plotly_chart(_figure(x, [("B", source[0]), ("S", output[0])], "B 与 S", ylabel),
                            key=_key(ws, sid, branch, "overlay"))
        except (ValueError, TypeError, FloatingPointError, OverflowError) as exc:
            st.error("所选透过率显示超出有限数值范围；请切回 absorbance 查看原结果。" + str(exc))
        assert snapshot.removed_component is not None
        st.plotly_chart(_figure(x, [("B − S", snapshot.removed_component[0])],
                               "平滑移除分量 / smoothing removed component", "Absorbance difference"),
                        key=_key(ws, sid, branch, "removed"))
        if unit != "absorbance":
            st.caption("此处 T / %T 为吸光度数学派生值，并非原始仪器透过率；移除分量仍以吸光度显示。")
            if np.any(branch_arrays(parent)[1] < 0) or np.any(snapshot.spectra < 0):
                st.warning("负吸光度会产生大于 1 的 T / 大于 100% 的 %T；显示值未裁剪。")
        st.caption("移除分量可能包含真实谱峰；不等于真实噪声，不自动证明 SNR 或处理正确性提升。")
    else:
        x, source = branch_arrays(parent)
        _, output, ylabel = display_arrays(snapshot)
        st.plotly_chart(_figure(x, [(BRANCH_LABELS[PARENTS[branch]], source[0])],
                               "归一化来源", "Absorbance"), key=_key(ws, sid, branch, "source_plot"))
        st.plotly_chart(_figure(x, [(BRANCH_LABELS[branch], output[0])], "归一化结果", ylabel),
                        key=_key(ws, sid, branch, "normalized_plot"))
        st.write({"scale": plain(snapshot.scale), "offset": plain(snapshot.offset),
                  "reference": plain(snapshot.reference_details), "quantity": snapshot.quantity,
                  "purpose": snapshot.purpose})
    with st.expander("当前版本诊断与配方"):
        st.json({"recipe": plain(snapshot.effective_recipe), "diagnostics": plain(snapshot.qc)})
        st.caption("诊断量用于审阅，不是通用实验合格标准。普通谱不计算跨样品时间连续性评分。")
    for warning in snapshot.warnings:
        st.warning(warning)


def render_postprocessing(ws: BatchWorkspace, sid: str) -> None:
    st.subheader("5. 普通光谱后处理")
    state = get_postprocessing_state(ws, sid)
    baseline = branch_status(ws, sid, "baseline")
    if baseline["status"] == "ready":
        st.caption(f"最终母谱 B 已确认 · 版本 {get_branch(ws, sid, 'baseline').fingerprint[:12]}")
    else:
        st.info("先完成细调确认，或在细调页明确确认不做细调，才能获得母谱 B。" + baseline["reason"])
    st.caption("平滑与归一化均为可选独立分支。确认 S 不改变归一化来源或导出选择；旧基线导出继续使用 B。")
    smooth_tab, normalize_tab, status_tab, export_tab = st.tabs(
        ["平滑 Smoothing", "归一化 Normalization", "分支状态", "结果导出"]
    )
    with smooth_tab:
        _smoothing_editor(ws, sid)
        _actions(ws, sid, "smoothed")
        _plots(ws, sid, "smoothed")
    with normalize_tab:
        left, right = st.columns(2)
        if left.button("使用未平滑校正谱 B", key=_key(ws, sid, "source", "baseline")):
            state.normalization_source = "baseline"
            st.rerun()
        smoothing_status = branch_status(ws, sid, "smoothed")
        if right.button("使用已确认平滑谱 S", key=_key(ws, sid, "source", "smoothed"),
                        disabled=smoothing_status["status"] != "ready"):
            state.normalization_source = "smoothed"
            st.rerun()
        st.write("当前归一化来源：" + BRANCH_LABELS[state.normalization_source])
        if state.normalization_source == "smoothed" and smoothing_status["status"] != "ready":
            st.warning("所选 S 来源缺失或过期。保留此来源及草稿，不自动退回 B；请确认新的 S 或明确选择 B。")
        branch = "normalized_" + state.normalization_source
        _normalization_editor(ws, sid, branch)
        _actions(ws, sid, branch)
        _plots(ws, sid, branch)
    with status_tab:
        st.caption("各条可保留不同配方；范围、参考窗口、面积定义或平滑参数不一致时，不能自动认定定量可比。")
        st.dataframe(postprocessing_table(ws), hide_index=True, width="stretch")
    with export_tab:
        _export_controls(ws, sid)
