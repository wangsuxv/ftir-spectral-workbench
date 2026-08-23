"""Seven-page Streamlit interface for FTIR Baseline Workbench.

All numerical work is delegated to :mod:`ftir_baseline`; this file contains UI
state and visualisation only.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ftir_baseline.config import FineBaselineConfig, NormalizationConfig, PipelineConfig
from ftir_baseline.export import build_export_zip
from ftir_baseline.gallery import (
    CandidateSpec,
    scan_baseline_candidates,
    starter_pchip_anchor_windows,
)
from ftir_baseline.io import load_spectrum_directory, load_spectrum_files, read_spectrum_file
from ftir_baseline.models import SpectrumSet
from ftir_baseline.pipeline import endpoint_anchor_windows, run_pipeline
from ftir_baseline.ranges import crop_spectrum_set
from ftir_baseline.units import IntensityUnit, InvalidTransmittanceError, convert_to_absorbance

PAGES = (
    "1 · Import & Units",
    "2 · Range & Raw QC",
    "3 · Coarse Baseline Gallery",
    "4 · Fine Anchors",
    "5 · Series Consistency",
    "6 · Normalization",
    "7 · Export",
)


def _initialise_state() -> None:
    if "config_payload" not in st.session_state:
        st.session_state.config_payload = PipelineConfig(input_unit="absorbance").to_dict()
    st.session_state.setdefault("dataset", None)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("gallery", None)
    st.session_state.setdefault("selected_candidate", None)


def _config() -> PipelineConfig:
    payload = st.session_state.config_payload
    if hasattr(PipelineConfig, "model_validate"):
        return PipelineConfig.model_validate(payload)
    return PipelineConfig.parse_obj(payload)


def _validated_config_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an edited recipe before atomically publishing it to UI state."""

    if hasattr(PipelineConfig, "model_validate"):
        config = PipelineConfig.model_validate(payload)
    else:  # pragma: no cover - Pydantic 1
        config = PipelineConfig.parse_obj(payload)
    return config.to_dict()


def _set_config_section(section: str, values: Mapping[str, Any]) -> None:
    payload = dict(st.session_state.config_payload)
    current = dict(payload.get(section, {}))
    current.update(dict(values))
    payload[section] = current
    validated = _validated_config_payload(payload)
    if validated == st.session_state.config_payload:
        return
    st.session_state.config_payload = validated
    st.session_state.result = None
    st.session_state.gallery = None
    st.session_state.selected_candidate = None


def _set_config(**values: Any) -> None:
    payload = dict(st.session_state.config_payload)
    payload.update(values)
    validated = _validated_config_payload(payload)
    if validated == st.session_state.config_payload:
        return
    st.session_state.config_payload = validated
    st.session_state.result = None
    st.session_state.gallery = None
    st.session_state.selected_candidate = None


def _set_dataset(data: SpectrumSet) -> None:
    payload = dict(st.session_state.config_payload)
    configured = payload.get("wavenumber_range", [1800.0, 900.0])
    configured_low, configured_high = sorted((float(configured[0]), float(configured[1])))
    available_low = float(np.min(data.wavenumber))
    available_high = float(np.max(data.wavenumber))
    intersection_low = max(configured_low, available_low)
    intersection_high = min(configured_high, available_high)
    if intersection_high <= intersection_low:
        intersection_low, intersection_high = available_low, available_high
    payload["wavenumber_range"] = [intersection_high, intersection_low]
    payload["input_unit"] = data.intensity_unit
    # A floor is an explicit repair authorization for one particular
    # transmittance dataset.  It must never leak into a subsequently loaded set.
    payload["transmittance_floor"] = None
    # Anchor windows and normalization intervals are dataset-specific scientific
    # assumptions.  New data must require a fresh, visible confirmation.
    payload["fine_baseline"] = FineBaselineConfig().to_dict()
    payload["normalization"] = NormalizationConfig().to_dict()
    validated = _validated_config_payload(payload)
    st.session_state.dataset = data
    st.session_state.config_payload = validated
    st.session_state.result = None
    st.session_state.gallery = None
    st.session_state.selected_candidate = None


def _require_dataset() -> SpectrumSet | None:
    data = st.session_state.dataset
    if data is None:
        st.info("请先在 “Import & Units” 页面导入数据并确认单位。")
        return None
    return data


def _selected_absorbance(data: SpectrumSet, config: PipelineConfig) -> SpectrumSet:
    conversion = convert_to_absorbance(
        data.spectra,
        config.input_unit,
        transmittance_floor=config.transmittance_floor,
    )
    converted = SpectrumSet(
        wavenumber=data.wavenumber,
        perturbation=data.perturbation,
        perturbation_labels=data.perturbation_labels,
        spectra=conversion.absorbance,
        intensity_unit="absorbance",
        source_name=data.source_name,
        metadata={**data.mutable_metadata(), "unit_conversion": conversion.record.to_dict()},
    )
    return crop_spectrum_set(converted, config.wavenumber_range, strict_bounds=True)


def _line_figure(
    x: np.ndarray,
    series: Iterable[tuple[str, np.ndarray]],
    *,
    title: str,
    y_title: str = "Absorbance",
) -> go.Figure:
    figure = go.Figure()
    for name, values in series:
        figure.add_trace(go.Scatter(x=x, y=values, mode="lines", name=name))
    figure.update_layout(
        title=title,
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title=y_title,
        legend_orientation="h",
        margin=dict(l=30, r=20, t=55, b=35),
    )
    figure.update_xaxes(autorange="reversed")
    return figure


def _overlay_figure(data: SpectrumSet, *, title: str) -> go.Figure:
    figure = go.Figure()
    indices = np.arange(data.n_spectra)
    if data.n_spectra > 30:
        indices = np.unique(np.linspace(0, data.n_spectra - 1, 30).astype(int))
    for index in indices:
        figure.add_trace(
            go.Scatter(
                x=data.wavenumber,
                y=data.spectra[index],
                mode="lines",
                name=data.perturbation_labels[index],
                opacity=0.55,
                line=dict(width=1),
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title=data.intensity_unit,
        showlegend=data.n_spectra <= 15,
        margin=dict(l=30, r=20, t=55, b=35),
    )
    figure.update_xaxes(autorange="reversed")
    return figure


def _load_uploaded(
    files: list[Any], unit: IntensityUnit, sort_by_perturbation: bool
) -> SpectrumSet:
    names = [Path(uploaded.name).name for uploaded in files]
    folded = [name.casefold() for name in names]
    duplicates = sorted({name for name in names if folded.count(name.casefold()) > 1})
    if duplicates:
        raise ValueError(
            "上传文件包含重复文件名，无法无歧义保留来源和标签：" + ", ".join(duplicates)
        )
    with tempfile.TemporaryDirectory(prefix="ftir_upload_") as directory:
        paths: list[Path] = []
        for uploaded in files:
            path = Path(directory) / Path(uploaded.name).name
            path.write_bytes(uploaded.getvalue())
            paths.append(path)
        if len(paths) == 1:
            return read_spectrum_file(
                paths[0],
                input_unit=unit,
                source_name="uploaded FTIR table",
                sort_by_perturbation=sort_by_perturbation,
            )
        return load_spectrum_files(
            paths,
            input_unit=unit,
            sort_by_perturbation=sort_by_perturbation,
            exclude_names=("BASELINE.dpt",),
            source_name=f"{len(paths)} uploaded FTIR file(s)",
        )


def page_import() -> None:
    st.header("Import & Units")
    st.caption("程序不会根据数值范围替你决定物理单位；确认是处理配方的一部分。")
    files = st.file_uploader(
        "导入 DPT / CSV / TXT（可多选单谱文件，或上传一个宽表）",
        type=["dpt", "csv", "txt"],
        accept_multiple_files=True,
    )
    unit: IntensityUnit = st.radio(
        "输入强度单位",
        ("absorbance", "percent_transmittance", "fraction_transmittance"),
        horizontal=True,
        format_func=lambda value: {
            "absorbance": "吸光度",
            "percent_transmittance": "百分透过率 (%T)",
            "fraction_transmittance": "小数透过率 (0–1)",
        }[value],
    )
    sort_values = st.checkbox("按文件名中的扰动数值显式排序", value=True)
    confirmed = st.checkbox("我已确认上述单位", value=False)
    col_load, col_demo = st.columns(2)
    if col_load.button("载入上传文件", type="primary", disabled=not files or not confirmed):
        try:
            uploaded_data = _load_uploaded(files, unit, sort_values)
            _set_dataset(uploaded_data)
            _set_config(input_unit=unit)
            st.session_state.result = None
            st.success(f"已载入 {uploaded_data.n_spectra} 条光谱。")
        except Exception as exc:
            st.error(str(exc))
    bundled = Path(__file__).resolve().parents[1] / "data" / "original"
    has_local_dpt = bundled.is_dir() and any(
        path.is_file()
        and path.suffix.casefold() == ".dpt"
        and path.name.casefold() != "baseline.dpt"
        for path in bundled.iterdir()
    )
    if col_demo.button("载入本机 data/original 中的 DPT", disabled=not has_local_dpt):
        try:
            demo_data = load_spectrum_directory(
                bundled,
                input_unit="absorbance",
                sort_by_perturbation=True,
                exclude_names=("BASELINE.dpt",),
                source_name="local DPT series",
            )
            _set_dataset(demo_data)
            _set_config(input_unit="absorbance")
            st.success(
                f"已载入 {demo_data.n_spectra} 条本机光谱；"
                "BASELINE.dpt 已按演示约定排除。"
            )
        except Exception as exc:
            st.error(str(exc))

    current_data = _require_dataset()
    if current_data is None:
        return
    columns = st.columns(5)
    columns[0].metric("光谱数", current_data.n_spectra)
    columns[1].metric("每谱点数", current_data.n_points)
    columns[2].metric("波数上限", f"{np.max(current_data.wavenumber):.2f}")
    columns[3].metric("波数下限", f"{np.min(current_data.wavenumber):.2f}")
    columns[4].metric("轴方向", current_data.axis_direction)
    minimum = float(np.min(current_data.spectra))
    maximum = float(np.max(current_data.spectra))
    st.write(
        f"强度范围：`{minimum:.6g}` 至 `{maximum:.6g}`；确认单位：`{current_data.intensity_unit}`"
    )
    invalid = np.argwhere(~np.isfinite(current_data.spectra))
    nonpositive = (
        np.argwhere(current_data.spectra <= 0)
        if current_data.intensity_unit != "absorbance"
        else np.empty((0, 2))
    )
    if invalid.size:
        st.error(f"存在 {len(invalid)} 个 NaN/Inf，处理已禁止。")
    if nonpositive.size:
        st.error(f"存在 {len(nonpositive)} 个 T≤0 点；必须停止或显式设置透过率下限。")
        floor = st.number_input("显式透过率下限", min_value=1e-12, value=1e-6, format="%.8g")
        if st.button("记录并使用该下限"):
            _set_config(transmittance_floor=float(floor))
    preview_indices = np.linspace(
        0, current_data.n_points - 1, min(12, current_data.n_points)
    ).astype(int)
    preview = {"Wavenumber": current_data.wavenumber[preview_indices]}
    for index in range(min(3, current_data.n_spectra)):
        preview[current_data.perturbation_labels[index]] = current_data.spectra[
            index, preview_indices
        ]
    st.dataframe(pd.DataFrame(preview), width="stretch", hide_index=True)


def page_range() -> None:
    st.header("Range & Raw QC")
    data = _require_dataset()
    if data is None:
        return
    config = _config()
    available_low = float(np.min(data.wavenumber))
    available_high = float(np.max(data.wavenumber))
    current_low, current_high = sorted(config.wavenumber_range)
    current_high = min(max(float(current_high), available_low), available_high)
    current_low = min(max(float(current_low), available_low), available_high)
    col_high, col_low = st.columns(2)
    high = col_high.number_input(
        "分析高波数端 (cm⁻¹)",
        min_value=available_low,
        max_value=available_high,
        value=float(current_high),
    )
    low = col_low.number_input(
        "分析低波数端 (cm⁻¹)",
        min_value=available_low,
        max_value=available_high,
        value=float(current_low),
    )
    if st.button("应用公共波数范围", type="primary"):
        try:
            _set_config(wavenumber_range=[float(high), float(low)])
            selected = _selected_absorbance(data, _config())
            st.success(f"所有光谱共同保留 {selected.n_points} 个点；未改变轴方向。")
        except Exception as exc:
            st.error(str(exc))
    try:
        selected = _selected_absorbance(data, _config())
    except (ValueError, InvalidTransmittanceError) as exc:
        st.error(str(exc))
        return
    st.plotly_chart(_overlay_figure(selected, title="Selected absorbance series"), width="stretch")
    representative = sorted({0, selected.n_spectra // 2, selected.n_spectra - 1})
    series = [
        (selected.perturbation_labels[index], selected.spectra[index]) for index in representative
    ]
    st.plotly_chart(
        _line_figure(selected.wavenumber, series, title="First / middle / last spectra"),
        width="stretch",
    )
    checks = pd.DataFrame(
        {
            "检查": ["NaN", "Inf", "重复波数", "严格单调", "全组同一波数轴"],
            "结果": [
                str(int(np.isnan(data.spectra).sum())),
                str(int(np.isinf(data.spectra).sum())),
                str(int(data.wavenumber.size - np.unique(data.wavenumber).size)),
                "通过"
                if np.all(np.diff(data.wavenumber) > 0) or np.all(np.diff(data.wavenumber) < 0)
                else "失败",
                "通过",
            ],
        }
    )
    st.dataframe(checks, width="stretch", hide_index=True)


def _gallery_specs(
    methods: list[str],
    log_lambdas: list[int],
    asls_p: float,
    endpoint_width_cm1: float,
    anchor_windows: list[dict[str, Any]],
) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    if "Endpoint linear" in methods:
        specs.append(
            CandidateSpec(
                "Endpoint linear",
                fine_method="endpoint_window_linear",
                fine_params={
                    "endpoint_window_width_cm1": float(endpoint_width_cm1),
                    "statistic": "median",
                },
            )
        )
    if "Anchor PCHIP" in methods:
        specs.append(
            CandidateSpec(
                "Anchor PCHIP",
                fine_method="pchip",
                fine_params={"anchors": anchor_windows, "statistic": "median"},
            )
        )
    if "Rubberband" in methods:
        specs.append(CandidateSpec("Rubberband", coarse_method="rubberband"))
    for log_lam in log_lambdas:
        lam = 10.0**log_lam
        if "arPLS" in methods:
            specs.append(CandidateSpec(f"arPLS λ=1e{log_lam}", "arpls", {"lam": lam}))
        if "AsLS" in methods:
            specs.append(
                CandidateSpec(
                    f"AsLS λ=1e{log_lam}, p={asls_p:g}", "asls", {"lam": lam, "p": asls_p}
                )
            )
        if "airPLS" in methods:
            specs.append(CandidateSpec(f"airPLS λ=1e{log_lam}", "airpls", {"lam": lam}))
    return specs


def _candidate_anchor_windows(config: PipelineConfig, x: np.ndarray) -> list[dict[str, Any]]:
    explicit = [anchor.to_dict() for anchor in config.fine_baseline.anchors if anchor.enabled]
    return explicit or list(
        starter_pchip_anchor_windows(
            x,
            endpoint_window_width_cm1=config.fine_baseline.endpoint_window_width_cm1,
            statistic=config.fine_baseline.statistic,
        )
    )


def page_gallery() -> None:
    st.header("Coarse Baseline Gallery")
    st.warning("诊断分数只是候选排序，不是真实基线的证明。请同时查看宽峰保留、负残差和锚点合理性。")
    data = _require_dataset()
    if data is None:
        return
    config = _config()
    try:
        selected = _selected_absorbance(data, config)
    except Exception as exc:
        st.error(str(exc))
        return
    representative = st.selectbox("代表谱", ("first", "median", "last"), index=1)
    methods = st.multiselect(
        "候选方法",
        ("Endpoint linear", "Anchor PCHIP", "arPLS", "AsLS", "airPLS", "Rubberband"),
        default=("Endpoint linear", "Anchor PCHIP", "arPLS", "AsLS", "Rubberband"),
    )
    log_lambdas = st.multiselect("log10(λ)", list(range(3, 10)), default=[4, 6, 8])
    asls_p = st.select_slider("AsLS p", options=[0.001, 0.005, 0.01, 0.05], value=0.01)
    smoothing_enabled = st.checkbox(
        "只在用于估计基线的副本上进行轻微 SG 平滑",
        value=config.baseline_smoothing.enabled,
    )
    _set_config_section("baseline_smoothing", {"enabled": smoothing_enabled, "estimate_only": True})
    if st.button("运行候选画廊", type="primary", disabled=not methods):
        try:
            anchors = _candidate_anchor_windows(config, selected.wavenumber)
            specs = _gallery_specs(
                methods,
                sorted(log_lambdas),
                float(asls_p),
                config.fine_baseline.endpoint_window_width_cm1,
                anchors,
            )
            st.session_state.gallery = scan_baseline_candidates(
                selected.wavenumber,
                selected.spectra,
                specs,
                representative=representative,
                smoothing=_config().baseline_smoothing,
                anchor_windows=anchors,
            )
        except Exception as exc:
            st.error(str(exc))
    gallery = st.session_state.gallery
    if gallery is None:
        return
    rows = []
    rank_lookup = {entry.name: (rank, entry.score) for rank, entry in enumerate(gallery.ranking, 1)}
    for evaluation in gallery.evaluations:
        rank, score = rank_lookup[evaluation.name]
        rows.append(
            {
                "Rank": rank,
                "Candidate": evaluation.name,
                "J (heuristic)": score,
                "Anchor error": evaluation.qc.summary["mean_anchor_error"],
                "Negative fraction": evaluation.qc.summary["mean_negative_fraction"],
                "Baseline roughness": evaluation.qc.summary["mean_baseline_roughness"],
                "Derivative correlation": evaluation.qc.summary["mean_derivative_correlation"],
            }
        )
    st.dataframe(pd.DataFrame(rows).sort_values("Rank"), width="stretch", hide_index=True)
    names = [evaluation.name for evaluation in gallery.evaluations]
    chosen = st.selectbox("查看/采用候选", names)
    evaluation = next(item for item in gallery.evaluations if item.name == chosen)
    st.plotly_chart(
        _line_figure(
            selected.wavenumber,
            (
                ("Raw representative", gallery.representative_spectrum),
                ("Estimated baseline", evaluation.result.total_baseline),
                ("Corrected", evaluation.result.corrected),
            ),
            title=chosen,
        ),
        width="stretch",
    )
    if st.button("采用该候选配方"):
        spec = evaluation.spec
        payload = dict(st.session_state.config_payload)
        coarse = dict(payload["coarse_baseline"])
        fine = dict(payload["fine_baseline"])
        coarse["method"] = spec.coarse_method
        if "lam" in spec.coarse_params:
            coarse["lambda"] = spec.coarse_params["lam"]
        if "p" in spec.coarse_params:
            coarse["p"] = spec.coarse_params["p"]
        if spec.fine_method != "none":
            fine.update(
                {
                    "enabled": True,
                    "method": spec.fine_method,
                    **dict(spec.fine_params),
                }
            )
        else:
            # The gallery preview contains this coarse candidate alone.  Keep
            # the adopted recipe identical; a fine baseline can be added on
            # the next page after inspecting the coarse residual.
            fine["enabled"] = False
        _set_config(
            coarse_baseline=coarse,
            fine_baseline=fine,
            series_mode=(
                "collaborative_pls"
                if spec.coarse_method in {"arpls", "asls", "airpls", "pspline_arpls"}
                else "independent_locked"
            ),
        )
        st.session_state.selected_candidate = chosen
        st.success(f"已记录候选：{chosen}")


def _default_anchor_table(config: PipelineConfig) -> pd.DataFrame:
    anchors = config.fine_baseline.anchors
    if anchors:
        return pd.DataFrame([anchor.to_dict() for anchor in anchors])
    rows = starter_pchip_anchor_windows(
        np.asarray(config.wavenumber_range, dtype=np.float64),
        endpoint_window_width_cm1=config.fine_baseline.endpoint_window_width_cm1,
        statistic=config.fine_baseline.statistic,
    )
    return pd.DataFrame(rows)


def _fine_anchor_visuals(
    result: Any, spectrum_index: int
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Extract the exact fitted windows and representatives for UI inspection."""

    params: Mapping[str, Any]
    metrics: Mapping[str, Any]
    if result.config.series_mode == "shared_shape":
        params = result.baseline.params
        metrics = result.baseline.metrics
    else:
        candidate = result.baseline.params.get("fine", {})
        params = candidate if isinstance(candidate, Mapping) else {}
        metrics = {}
    method = str(params.get("method", "none"))
    windows: list[dict[str, Any]] = []
    centers = np.empty(0, dtype=np.float64)
    values = np.empty(0, dtype=np.float64)

    if method == "shared_shape":
        windows = [dict(item) for item in params.get("anchors", ())]
        centers = np.asarray(params.get("anchor_centers", ()), dtype=np.float64)
        all_values = np.asarray(metrics.get("residual_anchor_values", ()), dtype=np.float64)
        if all_values.ndim == 2:
            values = all_values[spectrum_index]
    elif method in {"multipoint_linear", "pchip", "polynomial"}:
        windows = [dict(item) for item in params.get("anchors", ())]
        centers = np.asarray(params.get("anchor_centers", ()), dtype=np.float64)
        all_values = np.asarray(params.get("anchor_values", ()), dtype=np.float64)
        if all_values.ndim == 2:
            values = all_values[spectrum_index]
    elif method == "endpoint_window_linear":
        width = float(params.get("endpoint_window_width_cm1", 8.0))
        windows = endpoint_anchor_windows(result.absorbance_selected.wavenumber, width)
        statistic = str(params.get("statistic", "median"))
        for window in windows:
            window["statistic"] = statistic
        fitted = params.get("fitted", {})
        if isinstance(fitted, Mapping):
            centers = np.asarray(fitted.get("representative_wavenumbers", ()), dtype=np.float64)
            lower = np.asarray(fitted.get("lower_values", ()), dtype=np.float64)
            upper = np.asarray(fitted.get("upper_values", ()), dtype=np.float64)
            if lower.ndim == 1 and upper.ndim == 1:
                values = np.asarray(
                    [lower[spectrum_index], upper[spectrum_index]], dtype=np.float64
                )
    elif method == "strict_endpoint":
        centers = np.asarray(params.get("endpoint_wavenumbers", ()), dtype=np.float64)
        endpoint_values = params.get("endpoint_values", {})
        if isinstance(endpoint_values, Mapping):
            lower = np.asarray(endpoint_values.get("lower", ()), dtype=np.float64)
            upper = np.asarray(endpoint_values.get("upper", ()), dtype=np.float64)
            if lower.ndim == 1 and upper.ndim == 1:
                values = np.asarray(
                    [lower[spectrum_index], upper[spectrum_index]], dtype=np.float64
                )

    if (
        centers.shape != values.shape
        or not np.isfinite(centers).all()
        or not np.isfinite(values).all()
    ):
        centers = np.empty(0, dtype=np.float64)
        values = np.empty(0, dtype=np.float64)
    return windows, centers, values


def page_anchors() -> None:
    st.header("Fine Anchors")
    data = _require_dataset()
    if data is None:
        return
    config = _config()
    series_mode = st.selectbox(
        "序列模式",
        ("independent_locked", "collaborative_pls", "shared_shape"),
        index=("independent_locked", "collaborative_pls", "shared_shape").index(config.series_mode),
        help="所有模式都保留输入时间顺序，并锁定算法、参数和锚点位置。",
    )
    shared_shape_mode = series_mode == "shared_shape"
    if shared_shape_mode:
        st.info(
            "Shared-shape 模式由公共弯曲形状加每谱常数/一次斜率构成。"
            "下表窗口用于约束这两个自由度，不会使用 PCHIP、分段线性或多项式插值。"
        )
    enabled = st.checkbox(
        "启用局部细调",
        value=True if shared_shape_mode else config.fine_baseline.enabled,
        disabled=shared_shape_mode,
        help="Shared-shape 的 affine 调整是该模式固有部分，不能关闭。",
    )
    methods = (
        ("endpoint_window_linear",)
        if shared_shape_mode
        else ("endpoint_window_linear", "piecewise_linear", "pchip", "polynomial", "none")
    )
    current_method = (
        config.fine_baseline.method if config.fine_baseline.method in methods else methods[0]
    )
    method = st.selectbox(
        "细调方法",
        methods,
        index=methods.index(current_method),
        disabled=shared_shape_mode,
    )
    col_width, col_stat, col_strict = st.columns(3)
    width = col_width.number_input(
        "端点窗口宽度 (cm⁻¹)",
        min_value=0.1,
        value=float(config.fine_baseline.endpoint_window_width_cm1),
    )
    statistic = col_stat.selectbox(
        "窗口统计", ("median", "mean"), index=0 if config.fine_baseline.statistic == "median" else 1
    )
    strict = col_strict.checkbox(
        "Strict endpoint（高级）",
        value=False if shared_shape_mode else config.fine_baseline.strict_endpoint,
        disabled=shared_shape_mode,
    )
    if strict:
        st.warning("Strict endpoint 直接受两个边界噪声点和边界真实峰控制。")
    anchor_table = st.data_editor(
        _default_anchor_table(config),
        num_rows="dynamic",
        width="stretch",
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "start": st.column_config.NumberColumn("Start", format="%.2f"),
            "end": st.column_config.NumberColumn("End", format="%.2f"),
            "statistic": st.column_config.SelectboxColumn("Statistic", options=["median", "mean"]),
        },
    )
    order = st.slider(
        "多项式阶数",
        1,
        3,
        int(config.fine_baseline.polynomial_order),
        disabled=shared_shape_mode,
    )
    if st.button("保存锚点并计算预览", type="primary"):
        try:
            records = []
            for row in anchor_table.to_dict("records"):
                if pd.isna(row.get("start")) or pd.isna(row.get("end")):
                    continue
                records.append(
                    {
                        "enabled": bool(row.get("enabled", True)),
                        "start": float(row["start"]),
                        "end": float(row["end"]),
                        "statistic": str(row.get("statistic", statistic)),
                    }
                )
            _set_config_section(
                "fine_baseline",
                {
                    "enabled": True if shared_shape_mode else enabled and method != "none",
                    "method": "endpoint_window_linear" if shared_shape_mode else method,
                    "endpoint_window_width_cm1": float(width),
                    "statistic": statistic,
                    "strict_endpoint": False if shared_shape_mode else strict,
                    "anchors": records,
                    "polynomial_order": order,
                },
            )
            _set_config(series_mode=series_mode)
            result = run_pipeline(data, _config())
            st.session_state.result = result
        except Exception as exc:
            st.error(str(exc))
    result = st.session_state.result
    if result is None:
        return
    index = st.slider("预览谱序号", 0, result.absorbance_selected.n_spectra - 1, 0)
    x = result.absorbance_selected.wavenumber
    y = result.absorbance_selected.spectra[index]
    figure = _line_figure(
        x,
        (
            ("A_raw", y),
            ("A_for_baseline", result.baseline_estimation_spectra[index]),
            ("B_coarse", result.baseline.coarse_baseline[index]),
            ("B_fine", result.baseline.fine_baseline[index]),
            ("B_total", result.baseline.total_baseline[index]),
            ("Corrected", result.baseline.corrected[index]),
        ),
        title=f"Fine-baseline preview · {result.absorbance_selected.perturbation_labels[index]}",
    )
    windows, centers, anchor_values = _fine_anchor_visuals(result, index)
    for anchor_index, window in enumerate(windows, start=1):
        start, end = sorted((float(window["start"]), float(window["end"])))
        figure.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(255, 165, 0, 0.16)",
            line_width=1,
            line_color="rgba(220, 120, 0, 0.55)",
            annotation_text=f"A{anchor_index}",
            annotation_position="top left",
        )
    if centers.size:
        figure.add_trace(
            go.Scatter(
                x=centers,
                y=anchor_values,
                mode="markers+text",
                text=[f"{value:.4g}" for value in anchor_values],
                textposition="top center",
                marker={"size": 9, "symbol": "diamond", "color": "darkorange"},
                name="Anchor statistic on B_fine",
            )
        )
    st.plotly_chart(figure, width="stretch")
    if centers.size:
        if not windows:
            windows = [
                {"start": center, "end": center, "statistic": "exact endpoint"}
                for center in centers
            ]
        st.caption("标记值是粗调后残差在固定窗口内的统计值，即 B_fine 所使用的锚点。")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Anchor": f"A{anchor_index}",
                        "Start": float(window["start"]),
                        "End": float(window["end"]),
                        "Statistic": str(window.get("statistic", "median")),
                        "Representative wavenumber": float(center),
                        "Representative value (B_fine)": float(value),
                    }
                    for anchor_index, (window, center, value) in enumerate(
                        zip(windows, centers, anchor_values, strict=True), start=1
                    )
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def _ensure_result(data: SpectrumSet) -> Any | None:
    if st.session_state.result is None:
        try:
            st.session_state.result = run_pipeline(data, _config())
        except Exception as exc:
            st.error(str(exc))
            return None
    return st.session_state.result


def _heatmap(x: np.ndarray, perturbation: np.ndarray, values: np.ndarray, title: str) -> go.Figure:
    figure = px.imshow(
        values,
        x=x,
        y=perturbation,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        origin="lower",
        labels={"x": "Wavenumber (cm⁻¹)", "y": "Perturbation", "color": "Absorbance"},
        title=title,
    )
    figure.update_xaxes(autorange="reversed")
    return figure


def page_series() -> None:
    st.header("Series Consistency")
    data = _require_dataset()
    if data is None:
        return
    if st.button("按当前锁定配方重新运行", type="primary"):
        st.session_state.result = None
    result = _ensure_result(data)
    if result is None:
        return
    x = result.absorbance_selected.wavenumber
    p = result.absorbance_selected.perturbation
    tabs = st.tabs(("原始吸光度", "总基线", "校正谱"))
    for tab, values, title in zip(
        tabs,
        (
            result.absorbance_selected.spectra,
            result.baseline.total_baseline,
            result.baseline.corrected,
        ),
        ("Raw absorbance", "Total baseline", "Corrected absorbance"),
        strict=True,
    ):
        tab.plotly_chart(_heatmap(x, p, values, title), width="stretch")
    metrics = pd.DataFrame(dict(result.qc.per_spectrum))
    metrics["label"] = result.absorbance_selected.perturbation_labels
    st.dataframe(metrics, width="stretch", hide_index=True)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=p, y=metrics["baseline_area"], name="Baseline area"))
    figure.add_trace(go.Scatter(x=p, y=metrics["anchor_error"], name="Anchor residual"))
    figure.add_trace(
        go.Scatter(x=p, y=metrics["adjacent_baseline_rms"], name="Adjacent baseline RMS")
    )
    figure.update_layout(
        xaxis_title="Perturbation", yaxis_title="Diagnostic", legend_orientation="h"
    )
    st.plotly_chart(figure, width="stretch")
    if result.warnings:
        with st.expander(f"{len(result.warnings)} 条警告", expanded=True):
            for warning in result.warnings:
                st.warning(warning)
    st.caption(result.qc.summary["diagnostic_score_disclaimer"])


def _normalization_interval_defaults(
    config: PipelineConfig, x: np.ndarray, method: str
) -> tuple[float, float]:
    """Choose a valid, at-least-two-point interval inside the selected axis."""

    ordered = np.sort(np.asarray(x, dtype=np.float64))
    available_low = float(ordered[0])
    available_high = float(ordered[-1])
    configured: tuple[float, float] | None = None
    if method.startswith("internal_peak"):
        configured = config.normalization.internal_reference_range
    elif method == "area":
        configured = config.normalization.integration_range

    for candidate in (configured, (1650.0, 1550.0)):
        if candidate is None:
            continue
        candidate_low, candidate_high = sorted((float(candidate[0]), float(candidate[1])))
        low = max(candidate_low, available_low)
        high = min(candidate_high, available_high)
        if high > low and np.count_nonzero((ordered >= low) & (ordered <= high)) >= 2:
            return high, low

    low_index = min(ordered.size - 2, ordered.size // 3)
    high_index = min(ordered.size - 1, max(low_index + 1, (2 * ordered.size) // 3))
    return float(ordered[high_index]), float(ordered[low_index])


def page_normalization() -> None:
    st.header("Normalization")
    st.caption("归一化只在基线校正后建立独立支路；主 analysis_data 永远保留未归一化校正谱。")
    data = _require_dataset()
    if data is None:
        return
    config = _config()
    methods = (
        "none",
        "internal_peak_height",
        "internal_peak_area",
        "vector",
        "area",
        "minmax_display",
    )
    method = st.selectbox("方法", methods, index=methods.index(config.normalization.method))
    try:
        selected = _selected_absorbance(data, config)
    except Exception as exc:
        st.error(str(exc))
        return
    available_low = float(np.min(selected.wavenumber))
    available_high = float(np.max(selected.wavenumber))
    default_high, default_low = _normalization_interval_defaults(
        config, selected.wavenumber, method
    )
    col_a, col_b = st.columns(2)
    interval_high = col_a.number_input(
        "参考/积分高波数",
        min_value=available_low,
        max_value=available_high,
        value=default_high,
    )
    interval_low = col_b.number_input(
        "参考/积分低波数",
        min_value=available_low,
        max_value=available_high,
        value=default_low,
    )
    target = st.number_input(
        "目标值", min_value=1e-12, value=float(config.normalization.target_value)
    )
    absolute = st.checkbox("面积/高度取绝对值", value=config.normalization.absolute)
    application_failed = False
    if st.button("应用归一化支路", type="primary"):
        values: dict[str, Any] = {"method": method, "target_value": target, "absolute": absolute}
        if method.startswith("internal_peak"):
            values["internal_reference_range"] = [interval_high, interval_low]
        if method == "area":
            values["integration_range"] = [interval_high, interval_low]
        try:
            payload = dict(st.session_state.config_payload)
            normalization_payload = dict(payload.get("normalization", {}))
            normalization_payload.update(values)
            payload["normalization"] = normalization_payload
            validated = _validated_config_payload(payload)
            if hasattr(PipelineConfig, "model_validate"):
                candidate_config = PipelineConfig.model_validate(validated)
            else:  # pragma: no cover - Pydantic 1
                candidate_config = PipelineConfig.parse_obj(validated)
            candidate_result = run_pipeline(data, candidate_config)
        except Exception as exc:
            st.error(str(exc))
            application_failed = True
        else:
            _set_config_section("normalization", values)
            st.session_state.result = candidate_result
    result = st.session_state.result if application_failed else _ensure_result(data)
    if result is None:
        return
    rows = sorted(
        {0, result.absorbance_selected.n_spectra // 2, result.absorbance_selected.n_spectra - 1}
    )
    for index in rows:
        st.plotly_chart(
            _line_figure(
                result.absorbance_selected.wavenumber,
                (
                    ("analysis_data (unchanged)", result.normalization.analysis_data[index]),
                    ("view_data", result.normalization.view_data[index]),
                ),
                title=result.absorbance_selected.perturbation_labels[index],
            ),
            width="stretch",
        )
    factors = pd.DataFrame(
        {
            "Perturbation": result.absorbance_selected.perturbation,
            "Label": result.absorbance_selected.perturbation_labels,
            "Factor": result.normalization.factors,
        }
    )
    st.dataframe(factors, width="stretch", hide_index=True)
    st.line_chart(factors, x="Perturbation", y="Factor")


def page_export() -> None:
    st.header("Export")
    data = _require_dataset()
    if data is None:
        return
    result = _ensure_result(data)
    if result is None:
        return
    st.success(
        f"重建检查：{'通过' if result.qc.summary['reconstruction_passed'] else '失败'}；"
        f"最大误差 {result.qc.summary['maximum_reconstruction_error']:.3g}"
    )
    st.json(
        {
            "input_sha256": result.input_sha256,
            "software_version": result.software_version,
            "series_mode": result.config.series_mode,
            "coarse": result.config.coarse_baseline.method,
            "fine": result.config.fine_baseline.method,
            "normalization": result.normalization.method,
            "warnings": len(result.warnings),
        }
    )
    with st.spinner("生成 CSV、HTML、QC 图、recipe 与 manifest…"):
        bundle = build_export_zip(result)
    st.download_button(
        "下载完整可复现 ZIP",
        data=bundle,
        file_name="ftir_baseline_workbench_export.zip",
        mime="application/zip",
        type="primary",
    )
    recipe_text = json.dumps(result.recipe_dict(), ensure_ascii=False, indent=2)
    st.download_button(
        "仅下载 JSON recipe",
        data=recipe_text,
        file_name="processing_recipe.json",
        mime="application/json",
    )
    st.markdown(
        "ZIP 包含原始输入、完整/选区吸光度、粗/细/总基线、校正谱、可选归一化、"
        "质量指标、2D-COS-ready 校正谱、HTML 报告、QC 图和带 SHA-256 的 manifest。"
    )


def main() -> None:
    st.set_page_config(
        page_title="FTIR Baseline Workbench",
        page_icon="〰️",
        layout="wide",
    )
    _initialise_state()
    st.sidebar.title("FTIR Baseline Workbench")
    st.sidebar.caption("可检查 · 可比较 · 可复现")
    page = st.sidebar.radio("工作流", PAGES)
    st.sidebar.divider()
    st.sidebar.code("raw → absorbance → range\n→ coarse → fine → normalize\n→ QC → export")
    handlers = {
        PAGES[0]: page_import,
        PAGES[1]: page_range,
        PAGES[2]: page_gallery,
        PAGES[3]: page_anchors,
        PAGES[4]: page_series,
        PAGES[5]: page_normalization,
        PAGES[6]: page_export,
    }
    handlers[page]()


if __name__ == "__main__":
    main()
