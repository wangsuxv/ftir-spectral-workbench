"""Shared ordinary-export naming controls; preferences never alter scientific state."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ftir_workbench.batch.export_naming import (
    allocate_bases,
    get_naming_options,
    naming_plan_digest,
    set_name_override,
    set_naming_policy,
)
from ftir_workbench.batch.models import BatchWorkspace

POLICY_LABELS = {
    "original": "原始名称（默认）",
    "display": "展示名称",
    "legacy_id": "旧版 ID 命名（兼容）",
}


def naming_context(ws: BatchWorkspace) -> str:
    """Include unselected collision peers without evaluating any spectra."""
    return naming_plan_digest(allocate_bases(ws))


def _key(ws: BatchWorkspace, context: str, field: str) -> str:
    return f"_batch_{ws.workspace_id}_naming_{context}_{field}"


def _policy_changed(ws: BatchWorkspace, key: str) -> None:
    set_naming_policy(ws, str(st.session_state[key]))


def _override_changed(ws: BatchWorkspace, sid: str, enabled_key: str, text_key: str) -> None:
    set_name_override(ws, sid, str(st.session_state.get(text_key, ""))
                      if st.session_state[enabled_key] else None)


def render_naming_settings(ws: BatchWorkspace, sid: str | None, *, context: str) -> dict[str, Any]:
    """Restore both entry points from the shared workspace preference namespace."""
    options = get_naming_options(ws)
    with st.expander("导出文件命名设置", expanded=True):
        policy_key = _key(ws, context, "policy")
        st.session_state[policy_key] = options["policy"]
        st.selectbox("导出文件命名方式", tuple(POLICY_LABELS), format_func=POLICY_LABELS.get,
                     key=policy_key, on_change=_policy_changed, args=(ws, policy_key))
        if sid is not None and sid in ws.records:
            enabled_key = _key(ws, context, sid + "_override_enabled")
            text_key = _key(ws, context, sid + "_override")
            overrides = options["overrides"]
            st.session_state[enabled_key] = sid in overrides
            st.session_state[text_key] = overrides.get(sid, "")
            disabled = options["policy"] == "legacy_id"
            st.checkbox("覆盖当前光谱的导出基础名", key=enabled_key, disabled=disabled,
                        on_change=_override_changed, args=(ws, sid, enabled_key, text_key))
            st.text_input("当前光谱导出基础名", key=text_key,
                          disabled=disabled or sid not in overrides,
                          on_change=_override_changed, args=(ws, sid, enabled_key, text_key))
            st.caption("覆盖只影响此谱导出名称，不修改展示名或原始来源；空覆盖会使用安全回退名称。")
        if options["policy"] == "legacy_id":
            st.caption("兼容模式保留旧版 ID 路径和导出结构；已保存的基础名覆盖暂不应用。")
        else:
            st.caption("处理后缀取自已确认结果。文件名会安全清理并在整个工作区中稳定消歧。")
    return options


def render_naming_preview(ws: BatchWorkspace, preview: dict[str, Any]) -> None:
    """Show the same paths the exporter will serialize, including automatic parents."""
    rows = preview["name_map"]
    report = preview["report"]
    table: list[dict[str, Any]] = []
    represented = set()
    for row in rows:
        sid = row["spectrum_id"]
        branch = row.get("branch", "")
        represented.add((sid, branch))
        table.append({
            "原始文件": row.get("original_filename", ""),
            "原始列名": row.get("original_column_label", ""),
            "展示名称": row.get("display_name", ""),
            "处理状态": branch,
            "最终 CSV 路径": row.get("spectra_csv", ""),
            "导出角色": "自动包含的父级" if row.get("included_as") == "parent" else "请求结果",
            "就绪 / 排除原因": "已确认可用",
            "命名说明": "; ".join(row.get("notes", [])),
        })
        for derived_name, path in row.get("derived_outputs", {}).items():
            table.append({**table[-1], "处理状态": f"{branch} · {derived_name}",
                          "最终 CSV 路径": path, "导出角色": "附加派生透过率"})
    for row in report:
        sid = row.get("spectrum_id")
        branch = row.get("branch", row.get("stage", ""))
        if row.get("status") == "exported" or (sid, branch) in represented or sid not in ws.records:
            continue
        record = ws.records[sid]
        table.append({
            "原始文件": ws.sources[record.source_id].original_filename,
            "原始列名": record.original_column_label,
            "展示名称": record.display_name,
            "处理状态": branch,
            "最终 CSV 路径": "—",
            "导出角色": "请求结果",
            "就绪 / 排除原因": row.get("reason") or row.get("status", "不可用"),
            "命名说明": "",
        })
    st.caption("文件名预览（已确认版本；不会执行拟合）")
    if table:
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    else:
        st.caption("当前没有可预览的导出条目。")
    with st.expander("导出来源身份与路径明细"):
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if report:
            st.dataframe(pd.DataFrame(report), hide_index=True, width="stretch")
