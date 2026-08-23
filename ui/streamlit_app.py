"""Unified Streamlit interface for baseline-first FTIR and optional 2D-COS."""

from __future__ import annotations

import json
import tempfile
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ftir_baseline.config import NormalizationConfig, PipelineConfig
from ftir_baseline.io import (
    load_spectrum_directory,
    load_spectrum_files,
    read_spectrum_file,
)
from ftir_baseline.normalization import apply_normalization
from ftir_workbench.adapters import (
    prepared_scientific_branch_from_baseline_result,
)
from ftir_workbench.config import (
    TwoDCOSConfig,
    TwoDCOSDisplayConfig,
    TwoDCOSRange,
)
from ftir_workbench.export import (
    build_baseline_bundle,
    build_project_bundle,
    build_twodcos_bundle,
    load_prepared,
    serialize_prepared,
)
from ftir_workbench.services.baseline_service import BaselineWorkflowService
from ftir_workbench.services.twodcos_service import (
    TwoDCOSWorkflowService,
    analyze_peak_order,
)
from ftir_workbench.workflow import twodcos_science_changed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_DATA = PROJECT_ROOT / "data" / "original"

PAGES = (
    "1. Import & Perturbation",
    "2. Absorbance & Range",
    "3. Coarse Baseline",
    "4. Fine Baseline",
    "5. Series QC",
    "6. Normalization / Branches",
    "7. Baseline Result & Export",
    "8. Optional 2D-COS Setup",
    "9. 2D-COS Results",
)


def _has_local_dpt_series(directory: Path = BUNDLED_DATA) -> bool:
    """Return whether a local, non-control DPT spectrum is available."""

    return directory.is_dir() and any(
        path.is_file()
        and path.suffix.casefold() == ".dpt"
        and path.name.casefold() != "baseline.dpt"
        for path in directory.iterdir()
    )


def _initial_state() -> None:
    defaults: dict[str, Any] = {
        "raw_data": None,
        "baseline_config": PipelineConfig(input_unit="absorbance").to_dict(),
        "baseline_result": None,
        "prepared": None,
        "active_prepared": None,
        "prepared_source": None,
        "sensitivity_prepared": None,
        "twodcos_result": None,
        "twodcos_config": None,
        "baseline_bundle": None,
        "twodcos_bundle": None,
        "peak_order_result": None,
        "twodcos_status": None,
        "display_normalization": "none",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _pipeline_config(payload: dict[str, Any] | None = None) -> PipelineConfig:
    values = st.session_state.baseline_config if payload is None else payload
    if hasattr(PipelineConfig, "model_validate"):
        return PipelineConfig.model_validate(values)
    return PipelineConfig.parse_obj(values)  # pragma: no cover - Pydantic 1


def _invalidate_from_baseline() -> None:
    for key in (
        "baseline_result",
        "prepared",
        "active_prepared",
        "prepared_source",
        "sensitivity_prepared",
        "twodcos_result",
        "twodcos_config",
        "baseline_bundle",
        "twodcos_bundle",
        "peak_order_result",
        "twodcos_status",
    ):
        st.session_state[key] = None


def _set_baseline_config(**updates: Any) -> None:
    candidate = dict(st.session_state.baseline_config)
    candidate.update(updates)
    validated = _pipeline_config(candidate).to_dict()
    if validated != st.session_state.baseline_config:
        st.session_state.baseline_config = validated
        _invalidate_from_baseline()


def _set_config_section(section: str, **updates: Any) -> None:
    candidate = dict(st.session_state.baseline_config)
    nested = dict(candidate.get(section, {}))
    nested.update(updates)
    candidate[section] = nested
    validated = _pipeline_config(candidate).to_dict()
    if validated != st.session_state.baseline_config:
        st.session_state.baseline_config = validated
        _invalidate_from_baseline()


def _set_raw_data(data: Any) -> None:
    config = _pipeline_config()
    payload = config.to_dict()
    payload["input_unit"] = data.intensity_unit
    payload["wavenumber_range"] = [
        min(1800.0, float(np.max(data.wavenumber))),
        max(900.0, float(np.min(data.wavenumber))),
    ]
    st.session_state.raw_data = data
    st.session_state.baseline_config = _pipeline_config(payload).to_dict()
    _invalidate_from_baseline()


def _activate_prepared_for_twodcos(
    state: MutableMapping[str, Any],
    prepared: Any,
    *,
    source: str,
) -> None:
    """Activate one explicit Prepared branch and invalidate prior 2D descendants."""

    state["active_prepared"] = prepared
    state["prepared_source"] = source
    state["twodcos_config"] = None
    state["twodcos_result"] = None
    state["twodcos_bundle"] = None
    state["peak_order_result"] = None
    state["twodcos_status"] = (
        "PREPARED_FOR_2DCOS：校正谱分支已就绪，旧 2D 状态已失效。"
    )


def _uploaded_raw(
    uploads: list[Any],
    *,
    unit: str,
    sort_by_perturbation: bool,
) -> Any:
    names = [Path(item.name).name for item in uploads]
    folded = [name.casefold() for name in names]
    duplicates = sorted({name for name in names if folded.count(name.casefold()) > 1})
    if duplicates:
        raise ValueError("存在重复上传文件名：" + ", ".join(duplicates))
    with tempfile.TemporaryDirectory(prefix="ftir_workbench_upload_") as directory:
        paths: list[Path] = []
        for item in uploads:
            path = Path(directory) / Path(item.name).name
            path.write_bytes(item.getvalue())
            paths.append(path)
        if len(paths) == 1:
            return read_spectrum_file(
                paths[0],
                input_unit=unit,
                source_name=names[0],
                sort_by_perturbation=sort_by_perturbation,
            )
        return load_spectrum_files(
            paths,
            input_unit=unit,
            sort_by_perturbation=sort_by_perturbation,
            exclude_names=("BASELINE.dpt",),
            source_name=f"{len(paths)} uploaded FTIR spectra",
        )


def _line_figure(
    axis: np.ndarray,
    matrix: np.ndarray,
    labels: tuple[str, ...],
    *,
    title: str,
    y_title: str = "Absorbance",
    limit: int = 18,
) -> go.Figure:
    figure = go.Figure()
    indices = np.linspace(0, matrix.shape[0] - 1, min(limit, matrix.shape[0])).astype(int)
    for index in np.unique(indices):
        figure.add_trace(
            go.Scatter(
                x=axis,
                y=matrix[index],
                mode="lines",
                name=labels[int(index)],
                line={"width": 1.2},
                opacity=0.75,
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title=y_title,
        height=430,
        margin={"l": 40, "r": 20, "t": 55, "b": 40},
        showlegend=matrix.shape[0] <= 15,
    )
    figure.update_xaxes(autorange="reversed")
    return figure


def _heatmap(
    row_axis: np.ndarray,
    column_axis: np.ndarray,
    matrix: np.ndarray,
    *,
    title: str,
    percentile: float,
    contour_levels: int,
) -> go.Figure:
    scale = float(np.percentile(np.abs(matrix), percentile))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.max(np.abs(matrix))) or 1.0
    levels = max(2, int(contour_levels))
    contour_size = 2.0 * scale / (levels - 1)
    figure = go.Figure(
        go.Contour(
            x=column_axis,
            y=row_axis,
            z=matrix,
            zmin=-scale,
            zmax=scale,
            colorscale="RdBu_r",
            colorbar={"title": "Intensity"},
            contours={
                "start": -scale,
                "end": scale,
                "size": contour_size,
                "coloring": "heatmap",
                "showlines": True,
            },
            line={"width": 0.5},
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="Column wavenumber (cm⁻¹)",
        yaxis_title="Row wavenumber (cm⁻¹)",
        height=560,
        margin={"l": 60, "r": 30, "t": 55, "b": 55},
    )
    figure.update_xaxes(autorange="reversed")
    figure.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    return figure


def _show_raw_summary(data: Any) -> None:
    columns = st.columns(5)
    columns[0].metric("光谱数", data.n_spectra)
    columns[1].metric("每谱点数", data.n_points)
    columns[2].metric("波数上限", f"{np.max(data.wavenumber):.2f}")
    columns[3].metric("波数下限", f"{np.min(data.wavenumber):.2f}")
    columns[4].metric("轴方向", data.axis_direction)
    st.caption(
        f"来源：{data.source_name} · 单位：{data.intensity_unit} · "
        f"扰动：{data.perturbation[0]:g} → {data.perturbation[-1]:g}"
    )


def page_import() -> None:
    st.header("Import & Perturbation")
    st.caption("程序不猜测物理单位；单位和扰动排序都会写入可追溯配方。")
    raw_tab, corrected_tab = st.tabs(("从原始 FTIR 开始", "Start from corrected absorbance"))
    with raw_tab:
        uploads = st.file_uploader(
            "上传一个宽表 CSV/TXT，或多条二列 DPT",
            type=("dpt", "csv", "txt"),
            accept_multiple_files=True,
            key="raw_uploads",
        )
        unit = st.radio(
            "已确认输入单位",
            ("absorbance", "percent_transmittance", "fraction_transmittance"),
            horizontal=True,
            key="raw_unit",
        )
        sort_values = st.checkbox("按文件名中的扰动数值显式排序", value=True)
        left, right = st.columns(2)
        if left.button("载入上传数据", type="primary", disabled=not uploads):
            try:
                _set_raw_data(
                    _uploaded_raw(
                        uploads,
                        unit=unit,
                        sort_by_perturbation=sort_values,
                    )
                )
                st.success("原始数据已载入；尚未执行基线。")
            except Exception as exc:
                st.error(str(exc))
        if right.button(
            "载入本机 data/original 中的 DPT",
            disabled=not _has_local_dpt_series(),
        ):
            try:
                local_data = load_spectrum_directory(
                    BUNDLED_DATA,
                    input_unit="absorbance",
                    exclude_names=("BASELINE.dpt",),
                    sort_by_perturbation=True,
                    source_name="local DPT series",
                )
                _set_raw_data(local_data)
                st.success(
                    f"已数值排序 {local_data.n_spectra} 条本机光谱；"
                    "BASELINE.dpt 已按演示约定排除。"
                )
            except Exception as exc:
                st.error(str(exc))
        data = st.session_state.raw_data
        if data is not None:
            _show_raw_summary(data)
            st.plotly_chart(
                _line_figure(
                    data.wavenumber,
                    data.spectra,
                    data.perturbation_labels,
                    title="原始光谱预览",
                    y_title=data.intensity_unit,
                ),
                width="stretch",
            )
    with corrected_tab:
        prepared_file = st.file_uploader(
            "校正 CSV 或 baseline ZIP",
            type=("csv", "zip"),
            key="prepared_upload",
        )
        metadata_file = st.file_uploader(
            "可选 prepared_spectrum.meta.json",
            type=("json",),
            key="prepared_metadata_upload",
        )
        confirmed = st.checkbox(
            "我确认该校正光谱的强度单位是 absorbance",
            key="prepared_unit_confirmed",
        )
        if st.button(
            "载入校正谱并进入 2D-COS",
            type="primary",
            disabled=prepared_file is None or not confirmed,
        ):
            try:
                metadata = None if metadata_file is None else metadata_file.getvalue()
                prepared = load_prepared(prepared_file.getvalue(), metadata=metadata)
                _activate_prepared_for_twodcos(
                    st.session_state,
                    prepared,
                    source="reloaded corrected absorbance",
                )
                st.success("校正吸光度已载入；不会执行单位转换或基线校正。")
            except Exception as exc:
                st.error(str(exc))
        prepared = st.session_state.active_prepared
        if prepared is not None and st.session_state.prepared_source == "reloaded corrected absorbance":
            _show_prepared_info(prepared)


def page_range() -> None:
    st.header("Absorbance & Range")
    data = st.session_state.raw_data
    if data is None:
        st.info("请先在 Import & Perturbation 页载入原始数据。")
        return
    config = _pipeline_config()
    minimum, maximum = float(np.min(data.wavenumber)), float(np.max(data.wavenumber))
    current_high = min(max(config.wavenumber_range), maximum)
    current_low = max(min(config.wavenumber_range), minimum)
    left, right = st.columns(2)
    high = left.number_input("基线连续区间上限 (cm⁻¹)", minimum, maximum, current_high)
    low = right.number_input("基线连续区间下限 (cm⁻¹)", minimum, maximum, current_low)
    if high <= low:
        st.error("上限必须大于下限；不能把两个不连续区间拼成一个基线 block。")
        return
    _set_baseline_config(wavenumber_range=[high, low], input_unit=data.intensity_unit)
    mask = (data.wavenumber >= low) & (data.wavenumber <= high)
    st.caption(f"当前连续 block：{np.count_nonzero(mask)} 个实测点；不会插值或伪拼接。")
    st.plotly_chart(
        _line_figure(
            data.wavenumber[mask],
            data.spectra[:, mask],
            data.perturbation_labels,
            title="基线处理区间",
            y_title=data.intensity_unit,
        ),
        width="stretch",
    )


def page_coarse() -> None:
    st.header("Coarse Baseline")
    if st.session_state.raw_data is None:
        st.info("请先载入原始数据。")
        return
    config = _pipeline_config()
    coarse = config.coarse_baseline
    method = st.selectbox(
        "粗基线方法",
        ("none", "offset", "linear", "arpls", "asls", "airpls", "rubberband", "pspline_arpls"),
        index=("none", "offset", "linear", "arpls", "asls", "airpls", "rubberband", "pspline_arpls").index(coarse.method),
    )
    series_mode = st.selectbox(
        "序列模式",
        ("collaborative_pls", "independent_locked", "shared_shape"),
        index=("collaborative_pls", "independent_locked", "shared_shape").index(config.series_mode),
    )
    left, middle, right = st.columns(3)
    lam = left.number_input("λ", min_value=1.0, value=float(coarse.lam), format="%.6g")
    max_iter = middle.number_input("最大迭代", min_value=1, value=int(coarse.max_iter))
    tol = right.number_input("收敛容差", min_value=1e-12, value=float(coarse.tol), format="%.6g")
    smoothing = st.checkbox("启用 estimate-only Savitzky–Golay 平滑", value=config.baseline_smoothing.enabled)
    _set_baseline_config(series_mode=series_mode)
    _set_config_section(
        "coarse_baseline",
        method=method,
        **{"lambda": float(lam)},
        max_iter=int(max_iter),
        tol=float(tol),
    )
    _set_config_section("baseline_smoothing", enabled=smoothing, estimate_only=True)
    st.info("平滑副本只用于估计基线；最终校正谱始终由未平滑吸光度减去总基线。")


def _anchors_frame(config: PipelineConfig) -> pd.DataFrame:
    rows = [
        {
            "Enabled": item.enabled,
            "Start": item.start,
            "End": item.end,
            "Statistic": item.statistic,
        }
        for item in config.fine_baseline.anchors
    ]
    if not rows:
        high, low = max(config.wavenumber_range), min(config.wavenumber_range)
        rows = [
            {"Enabled": True, "Start": high - 8.0, "End": high, "Statistic": "median"},
            {"Enabled": True, "Start": low, "End": low + 8.0, "Statistic": "median"},
        ]
    return pd.DataFrame(rows)


def page_fine() -> None:
    st.header("Fine Baseline")
    if st.session_state.raw_data is None:
        st.info("请先载入原始数据。")
        return
    config = _pipeline_config()
    methods = ("none", "endpoint_window_linear", "piecewise_linear", "pchip", "polynomial")
    method = st.selectbox("精细基线方法", methods, index=methods.index(config.fine_baseline.method))
    width = st.number_input(
        "端点窗口宽度 (cm⁻¹)",
        min_value=0.1,
        value=float(config.fine_baseline.endpoint_window_width_cm1),
    )
    anchors: list[dict[str, Any]] = []
    if method in {"piecewise_linear", "pchip", "polynomial"}:
        edited = st.data_editor(
            _anchors_frame(config),
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
        )
        for row in edited.to_dict("records"):
            anchors.append(
                {
                    "enabled": bool(row["Enabled"]),
                    "start": float(row["Start"]),
                    "end": float(row["End"]),
                    "statistic": str(row["Statistic"]),
                }
            )
    _set_config_section(
        "fine_baseline",
        enabled=method != "none",
        method=method,
        endpoint_window_width_cm1=float(width),
        anchors=anchors,
    )
    st.caption("锚点窗口固定于整个序列，并写入 recipe 与 baseline fingerprint。")


def _run_baseline() -> None:
    data = st.session_state.raw_data
    if data is None:
        raise RuntimeError("没有已载入的原始数据")
    service = BaselineWorkflowService()
    result = service.run(data, _pipeline_config())
    prepared = service.prepared()
    st.session_state.baseline_result = result
    st.session_state.prepared = prepared
    st.session_state.active_prepared = None
    st.session_state.prepared_source = None
    st.session_state.twodcos_config = None
    st.session_state.twodcos_result = None
    st.session_state.baseline_bundle = None
    st.session_state.twodcos_bundle = None
    st.session_state.peak_order_result = None
    st.session_state.twodcos_status = None


def page_series_qc() -> None:
    st.header("Series QC")
    if st.session_state.raw_data is None:
        st.info("请先载入原始数据并配置基线。")
        return
    if st.button("运行基线、分解与 QC", type="primary"):
        try:
            with st.spinner("使用唯一 ftir_baseline 科学路径计算…"):
                _run_baseline()
            st.success("BASELINE_COMPLETED：可以独立导出并结束。")
        except Exception as exc:
            st.error(str(exc))
    result = st.session_state.baseline_result
    if result is None:
        st.info("尚无结果。此操作不会运行任何 2D-COS 代码。")
        return
    columns = st.columns(4)
    columns[0].metric("重建最大误差", f"{result.qc.summary.get('reconstruction_max_abs_error', 0):.3g}")
    columns[1].metric("负残差比例", f"{result.qc.summary.get('negative_residual_fraction_mean', 0):.3g}")
    columns[2].metric("光谱数", result.absorbance_selected.n_spectra)
    columns[3].metric("选区点数", result.absorbance_selected.n_points)
    for warning in result.warnings:
        st.warning(warning)
    tabs = st.tabs(("吸光度", "总基线", "校正谱"))
    for tab, matrix, title in zip(
        tabs,
        (
            result.absorbance_selected.spectra,
            result.baseline.total_baseline,
            result.analysis_data,
        ),
        ("选区吸光度", "总基线", "未归一化校正吸光度"),
        strict=True,
    ):
        with tab:
            st.plotly_chart(
                _line_figure(
                    result.absorbance_selected.wavenumber,
                    matrix,
                    result.absorbance_selected.perturbation_labels,
                    title=title,
                ),
                width="stretch",
            )


def page_normalization() -> None:
    st.header("Normalization / Branches")
    result = st.session_state.baseline_result
    if result is None:
        st.info("请先完成基线与 QC。")
        return
    method = st.radio(
        "显示分支",
        ("none", "minmax_display"),
        horizontal=True,
        index=0 if st.session_state.display_normalization == "none" else 1,
    )
    st.session_state.display_normalization = method
    view = apply_normalization(
        result.absorbance_selected.wavenumber,
        result.analysis_data,
        NormalizationConfig(method=method),
    ).view_data
    st.plotly_chart(
        _line_figure(
            result.absorbance_selected.wavenumber,
            view,
            result.absorbance_selected.perturbation_labels,
            title="显示分支（不进入主 2D-COS）",
        ),
        width="stretch",
    )
    st.info("显示归一化不会改变 prepared fingerprint，也不会使已计算的 2D 矩阵失效。")
    st.subheader("科学归一化敏感性分支")
    scientific_method = st.selectbox("方法", ("vector",), key="scientific_normalization_method")
    if st.button("创建独立敏感性分支"):
        try:
            normalization = apply_normalization(
                result.absorbance_selected.wavenumber,
                result.analysis_data,
                NormalizationConfig(method=scientific_method),
            )
            st.session_state.sensitivity_prepared = (
                prepared_scientific_branch_from_baseline_result(
                    result,
                    normalized_spectra=normalization.optional_normalized,
                    normalization_method=scientific_method,
                    branch_name=f"{scientific_method} sensitivity",
                )
            )
            st.success("已建立独立 fingerprint；未覆盖未归一化主分支。")
        except Exception as exc:
            st.error(str(exc))
    sensitivity = st.session_state.sensitivity_prepared
    if sensitivity is not None:
        st.warning(
            "这是显式科学归一化敏感性分支；它有独立 fingerprint，"
            "不会替换默认未归一化主分析。"
        )
        _show_prepared_info(sensitivity)
        artifact = serialize_prepared(
            sensitivity,
            csv_name="normalized_optional_for_sensitivity_analysis.csv",
            metadata_name="prepared_spectrum.sensitivity.meta.json",
        )
        download_csv, download_meta, continue_branch = st.columns(3)
        download_csv.download_button(
            "下载敏感性 CSV",
            data=artifact.csv_bytes,
            file_name=artifact.csv_name,
            mime="text/csv",
            width="stretch",
        )
        download_meta.download_button(
            "下载敏感性 metadata",
            data=artifact.metadata_bytes,
            file_name=artifact.metadata_name,
            mime="application/json",
            width="stretch",
        )
        if continue_branch.button(
            "用此分支继续 2D-COS",
            width="stretch",
        ):
            _activate_prepared_for_twodcos(
                st.session_state,
                sensitivity,
                source=f"scientific sensitivity branch ({scientific_method})",
            )
            st.success(
                "已切换到独立敏感性分支；默认未归一化 Prepared 仍完整保留。"
            )


def _show_prepared_info(prepared: Any) -> None:
    columns = st.columns(5)
    columns[0].metric("Unit", prepared.intensity_unit)
    columns[1].metric("Normalization", prepared.normalization_state)
    columns[2].metric("Spectra", prepared.n_spectra)
    columns[3].metric("Prepared points", prepared.n_points)
    columns[4].metric("Axis", prepared.current_axis_direction)
    st.code(
        f"Baseline run: {prepared.baseline_run_id}\n"
        f"Baseline fingerprint: {prepared.baseline_fingerprint}\n"
        f"Prepared SHA-256: {prepared.prepared_data_sha256}"
    )
    for warning in prepared.warnings:
        st.warning(warning)


def page_baseline_result() -> None:
    st.header("Baseline Result & Export")
    result = st.session_state.baseline_result
    prepared = st.session_state.prepared
    if result is None or prepared is None:
        st.info("请先在 Series QC 页成功完成基线。")
        return
    _show_prepared_info(prepared)
    if st.session_state.baseline_bundle is None:
        with st.spinner("生成 baseline bundle、sidecar 与 manifest…"):
            st.session_state.baseline_bundle = build_baseline_bundle(result, prepared=prepared)
    bundle = st.session_state.baseline_bundle
    left, right = st.columns(2)
    left.download_button(
        "导出校正谱并结束",
        data=bundle,
        file_name="baseline_run.zip",
        mime="application/zip",
        type="primary",
        width="stretch",
    )
    if right.button(
        "使用当前校正谱继续 2D-COS",
        type="primary",
        width="stretch",
    ):
        _activate_prepared_for_twodcos(
            st.session_state,
            prepared,
            source="current baseline run (in memory)",
        )
        st.success("PREPARED_FOR_2DCOS：已使用当前内存数据；没有写盘重读。")
    st.download_button(
        "保存 baseline recipe",
        data=json.dumps(result.recipe_dict(), ensure_ascii=False, indent=2),
        file_name="baseline_recipe.json",
        mime="application/json",
    )
    st.caption(
        "baseline-only 是完整成功状态。ZIP 包含原始输入、吸光度、粗/细/总基线、"
        "校正谱、可选归一化、QC、2D-COS-ready 校正谱、HTML 报告、图和 manifest。"
    )


def _default_twodcos_ranges(axis: np.ndarray) -> list[dict[str, Any]]:
    minimum, maximum = float(np.min(axis)), float(np.max(axis))
    candidates = (
        (1736.0, 1509.0, "amide_1736_1509"),
        (1250.0, 1140.0, "fingerprint_1250_1140"),
    )
    rows = [
        {"Enabled": True, "High": high, "Low": low, "Label": label}
        for high, low, label in candidates
        if minimum <= low < high <= maximum
    ]
    if not rows:
        rows.append({"Enabled": True, "High": maximum, "Low": minimum, "Label": "full_range"})
    return rows


def _twodcos_editor_rows(
    prepared: Any,
    config: TwoDCOSConfig | None,
) -> list[dict[str, Any]]:
    """Return editor rows without joining disconnected analysis intervals."""

    if config is None:
        return _default_twodcos_ranges(prepared.wavenumber)
    return [
        {
            "Enabled": True,
            "High": item.high_wavenumber,
            "Low": item.low_wavenumber,
            "Label": item.label or f"range_{index}",
        }
        for index, item in enumerate(config.ranges, start=1)
    ]


def _invalidate_twodcos_result(
    state: MutableMapping[str, Any],
    *,
    status: str,
) -> None:
    """Clear only numerical 2D descendants while preserving prepared data."""

    state["twodcos_result"] = None
    state["twodcos_bundle"] = None
    state["peak_order_result"] = None
    state["twodcos_status"] = status


def _reconcile_twodcos_config(
    state: MutableMapping[str, Any],
    candidate: TwoDCOSConfig,
) -> str:
    """Apply the project-service invalidation policy to Streamlit session state.

    Scientific changes invalidate matrices immediately. Display-only changes
    retain the exact result object and only invalidate rendered/export metadata.
    The helper never computes 2D, making rerun behavior explicit and testable.
    """

    current = state.get("twodcos_config")
    if current is None:
        state["twodcos_config"] = candidate
        if state.get("twodcos_result") is not None:
            _invalidate_twodcos_result(
                state,
                status=(
                    "TWODCOS_INVALIDATED：缺少旧配置，无法证明已有矩阵仍属于当前设置。"
                ),
            )
            return "science_invalidated"
        state["twodcos_status"] = "TWODCOS_CONFIGURED：参数已就绪，尚未计算矩阵。"
        return "configured"

    if twodcos_science_changed(current, candidate):
        state["twodcos_config"] = candidate
        _invalidate_twodcos_result(
            state,
            status=(
                "TWODCOS_INVALIDATED：2D 区间、矩阵约定或扰动策略已变化；"
                "旧矩阵和下游导出已清除。"
            ),
        )
        return "science_invalidated"

    if current != candidate:
        state["twodcos_config"] = candidate
        state["twodcos_bundle"] = None
        state["twodcos_status"] = (
            "DISPLAY_UPDATED：仅显示参数已更新；现有同步/异步矩阵保持不变。"
        )
        return "display_updated"
    return "unchanged"


def page_twodcos_setup() -> None:
    st.header("Optional 2D-COS Setup")
    prepared = st.session_state.active_prepared
    if prepared is None:
        st.info(
            "请在 Baseline Result 页选择“使用当前校正谱继续 2D-COS”，"
            "或在 Import 页载入以后保存的校正谱。"
        )
        return
    st.success("2D-COS 阶段不会重新执行单位转换、平滑、基线校正或默认归一化。")
    st.caption(f"Data source: {st.session_state.prepared_source}")
    _show_prepared_info(prepared)
    current_config = st.session_state.twodcos_config
    defaults_key = f"range_editor_{prepared.prepared_data_sha256[:12]}"
    frame = st.data_editor(
        pd.DataFrame(_twodcos_editor_rows(prepared, current_config)),
        key=defaults_key,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
    )
    left, middle, right = st.columns(3)
    conventions = ("2dpy_compatible", "canonical")
    policies = ("warn", "allow", "error")
    convention = left.selectbox(
        "矩阵约定",
        conventions,
        index=(
            0
            if current_config is None
            else conventions.index(current_config.convention)
        ),
    )
    nonuniform = middle.selectbox(
        "非等间隔扰动",
        policies,
        index=(
            0
            if current_config is None
            else policies.index(current_config.nonuniform_perturbation_policy)
        ),
    )
    cross_enabled = right.checkbox(
        "计算跨区间矩形矩阵",
        value=True if current_config is None else current_config.cross_range_enabled,
    )
    display_left, display_right = st.columns(2)
    contour_levels = display_left.number_input(
        "等高线数量（仅显示）",
        min_value=2,
        value=21 if current_config is None else current_config.display.contour_levels,
    )
    percentile = display_right.number_input(
        "色阶百分位（仅显示）",
        min_value=1.0,
        max_value=100.0,
        value=99.0 if current_config is None else current_config.display.display_percentile,
    )
    st.caption(
        "科学参数一旦变化会立即清除旧矩阵；等高线和色阶只刷新绘图，不触发科学重算。"
    )

    config: TwoDCOSConfig | None = None
    try:
        ranges = tuple(
            TwoDCOSRange(float(row["High"]), float(row["Low"]), str(row["Label"]))
            for row in frame.to_dict("records")
            if bool(row.get("Enabled", True))
        )
        config = TwoDCOSConfig(
            ranges=ranges,
            convention=convention,
            nonuniform_perturbation_policy=nonuniform,
            cross_range_enabled=cross_enabled,
            display=TwoDCOSDisplayConfig(
                contour_levels=int(contour_levels),
                display_percentile=float(percentile),
            ),
        )
        _reconcile_twodcos_config(st.session_state, config)
    except (KeyError, TypeError, ValueError) as exc:
        if st.session_state.twodcos_result is not None:
            _invalidate_twodcos_result(
                st.session_state,
                status=(
                    "TWODCOS_INVALIDATED：当前 2D 参数无效；旧矩阵和下游导出已清除。"
                ),
            )
        st.error(str(exc))

    status = st.session_state.twodcos_status
    if status:
        if str(status).startswith("TWODCOS_INVALIDATED"):
            st.warning(status)
        elif str(status).startswith("DISPLAY_UPDATED"):
            st.info(status)
        else:
            st.success(status)

    result_is_current = st.session_state.twodcos_result is not None
    button_label = (
        "科学结果已是最新"
        if result_is_current
        else "计算 self / cross 2D-COS"
    )
    if st.button(
        button_label,
        type="primary",
        disabled=config is None or result_is_current,
    ):
        try:
            assert config is not None
            with st.spinner("直接调用 ftir2dcos.twodcos 科学核心…"):
                result = TwoDCOSWorkflowService().compute(prepared, config)
            st.session_state.twodcos_config = config
            st.session_state.twodcos_result = result
            st.session_state.twodcos_bundle = None
            st.session_state.peak_order_result = None
            st.session_state.twodcos_status = (
                "TWODCOS_COMPLETED：所有区间均从同一 prepared block 截取。"
            )
            st.success(st.session_state.twodcos_status)
        except Exception as exc:
            st.error(str(exc))
    result = st.session_state.twodcos_result
    if result is not None:
        columns = st.columns(4)
        columns[0].metric("Self blocks", len(result.homo_results))
        columns[1].metric("Cross blocks", len(result.cross_results))
        columns[2].metric("QC", "PASS" if result.all_checks_passed else "FAIL")
        columns[3].metric("Convention", st.session_state.twodcos_config.convention)
        for warning in result.warnings:
            st.warning(warning)


def page_twodcos_results() -> None:
    st.header("2D-COS Results")
    prepared = st.session_state.active_prepared
    result = st.session_state.twodcos_result
    config = st.session_state.twodcos_config
    if prepared is None or result is None or config is None:
        status = st.session_state.twodcos_status
        if status and str(status).startswith("TWODCOS_INVALIDATED"):
            st.warning(status)
        else:
            st.info("请先在 Optional 2D-COS Setup 页完成计算。")
        return
    if st.session_state.twodcos_status and str(
        st.session_state.twodcos_status
    ).startswith("DISPLAY_UPDATED"):
        st.info(st.session_state.twodcos_status)
    st.caption(
        f"Parent baseline: {result.parent_baseline_run_id} · "
        f"Prepared SHA-256: {result.parent_prepared_data_sha256}"
    )
    percentile = config.display.display_percentile
    contour_levels = config.display.contour_levels
    for index, item in enumerate(result.homo_results, start=1):
        st.subheader(f"Self {index}: {item.analysis_range.display_name}")
        sync_tab, async_tab, dynamic_tab = st.tabs(("Synchronous", "Asynchronous", "Dynamic"))
        with sync_tab:
            st.plotly_chart(
                _heatmap(
                    item.result.row_wavenumber,
                    item.result.column_wavenumber,
                    item.result.synchronous,
                    title="Synchronous 2D-COS",
                    percentile=percentile,
                    contour_levels=contour_levels,
                ),
                width="stretch",
            )
        with async_tab:
            st.plotly_chart(
                _heatmap(
                    item.result.row_wavenumber,
                    item.result.column_wavenumber,
                    item.result.asynchronous,
                    title="Asynchronous 2D-COS",
                    percentile=percentile,
                    contour_levels=contour_levels,
                ),
                width="stretch",
            )
        with dynamic_tab:
            st.plotly_chart(
                _line_figure(
                    item.result.column_wavenumber,
                    item.result.dynamic,
                    prepared.perturbation_labels,
                    title="Dynamic spectra",
                ),
                width="stretch",
            )
        st.json(item.result.qc_metrics)
    for index, item in enumerate(result.cross_results, start=1):
        st.subheader(
            f"Cross {index}: {item.first_range.display_name} × {item.second_range.display_name}"
        )
        sync_tab, async_tab = st.tabs(("Synchronous", "Asynchronous"))
        with sync_tab:
            st.plotly_chart(
                _heatmap(
                    item.result.row_wavenumber,
                    item.result.column_wavenumber,
                    item.result.synchronous,
                    title="Cross synchronous 2D-COS",
                    percentile=percentile,
                    contour_levels=contour_levels,
                ),
                width="stretch",
            )
        with async_tab:
            st.plotly_chart(
                _heatmap(
                    item.result.row_wavenumber,
                    item.result.column_wavenumber,
                    item.result.asynchronous,
                    title="Cross asynchronous 2D-COS",
                    percentile=percentile,
                    contour_levels=contour_levels,
                ),
                width="stretch",
            )
        st.json(item.result.qc_metrics)
    st.subheader("Apparent peak-response order")
    peak_text = st.text_input(
        "峰位（cm⁻¹，以逗号分隔）",
        placeholder="例如：1650, 1540, 1220",
    )
    if st.button("分析表观响应顺序", disabled=not peak_text.strip()):
        try:
            peaks = tuple(float(value.strip()) for value in peak_text.split(",") if value.strip())
            st.session_state.peak_order_result = analyze_peak_order(result, peaks)
            st.session_state.twodcos_bundle = None
        except Exception as exc:
            st.error(str(exc))
    peak_order = st.session_state.peak_order_result
    if peak_order is not None:
        if peak_order.unique_order:
            st.success(" → ".join(item.display_label for item in peak_order.unique_order))
        elif peak_order.topological_layers:
            layers = [
                " = ".join(item.display_label for item in layer)
                for layer in peak_order.topological_layers
            ]
            st.info(" → ".join(layers))
        else:
            st.warning("当前证据不足或存在环，未强行给出唯一顺序。")
        st.dataframe(
            pd.DataFrame(peak_order.to_evidence_records()),
            hide_index=True,
            width="stretch",
        )
        for warning in peak_order.warnings:
            st.warning(warning)
    if st.session_state.twodcos_bundle is None:
        with st.spinner("生成 2D bundle 与父级哈希 manifest…"):
            peak_files = None
            if peak_order is not None:
                peak_files = {
                    "peak_order.json": json.dumps(
                        peak_order.to_dict(),
                        ensure_ascii=False,
                        indent=2,
                    ).encode("utf-8")
                }
            st.session_state.twodcos_bundle = build_twodcos_bundle(
                prepared,
                result,
                config,
                peak_order_files=peak_files,
            )
    bundle = st.session_state.twodcos_bundle
    left, right = st.columns(2)
    left.download_button(
        "下载 2D-COS bundle",
        data=bundle,
        file_name="twodcos_run.zip",
        mime="application/zip",
        type="primary",
        width="stretch",
    )
    baseline_bundle = st.session_state.baseline_bundle
    if baseline_bundle is not None:
        project_bundle = build_project_bundle(
            baseline_bundle,
            twodcos_bundles=(bundle,),
            project_config={
                "baseline": st.session_state.baseline_config,
                "twodcos": config.to_dict(),
            },
        )
        right.download_button(
            "下载完整 project.ftirw",
            data=project_bundle,
            file_name="project.ftirw",
            mime="application/zip",
            width="stretch",
        )


def main() -> None:
    st.set_page_config(
        page_title="FTIR Spectral Workbench",
        page_icon="〰️",
        layout="wide",
    )
    _initial_state()
    st.title("FTIR Spectral Workbench")
    st.caption("唯一基线路径 · 可独立结束 · prepared-only 2D-COS · SHA-256 lineage")
    page = st.sidebar.radio("工作流", PAGES)
    st.sidebar.divider()
    st.sidebar.code(
        "raw → absorbance → baseline → QC\n"
        "        ├─ export & stop\n"
        "        └─ prepared → 2D-COS"
    )
    handlers = {
        PAGES[0]: page_import,
        PAGES[1]: page_range,
        PAGES[2]: page_coarse,
        PAGES[3]: page_fine,
        PAGES[4]: page_series_qc,
        PAGES[5]: page_normalization,
        PAGES[6]: page_baseline_result,
        PAGES[7]: page_twodcos_setup,
        PAGES[8]: page_twodcos_results,
    }
    handlers[page]()


if __name__ == "__main__":
    main()
