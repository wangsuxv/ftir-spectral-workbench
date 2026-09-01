from __future__ import annotations

import ast
import io
import zipfile
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from streamlit.testing.v1 import AppTest

from ftir_baseline.config import PipelineConfig
from ftir_baseline.io import read_spectrum_file
from ftir_workbench.services import BaselineWorkflowService, TwoDCOSWorkflowService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "ui" / "streamlit_app.py"


def _load_app_module():  # type: ignore[no-untyped-def]
    spec = spec_from_file_location("unified_streamlit_app_under_test", APP_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("could not load unified Streamlit app")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unified_app_starts_and_exposes_ten_stage_workflow() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "FTIR Spectral Workbench"
    assert list(app.sidebar.radio[0].options) == [
        "1. Import & Perturbation",
        "2. Absorbance & Range",
        "3. Coarse Baseline",
        "4. Fine Baseline",
        "5. Series Consistency & QC",
        "6. Normalization / Branches",
        "7. Baseline Result & Export",
        "8. Post-Baseline Smoothing",
        "9. Optional 2D-COS Setup",
        "10. 2D-COS Results",
    ]
    assert any("Start from corrected absorbance" in tab.label for tab in app.tabs)


def test_unified_ui_never_imports_legacy_2d_preprocessing() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert "ftir2dcos.pipeline" not in imported
    assert "ftir2dcos.preprocessing" not in imported
    assert "ftir2dcos.conversion" not in imported


def test_default_twodcos_ranges_remain_separate_blocks() -> None:
    app_module = _load_app_module()
    rows = app_module._default_twodcos_ranges(np.linspace(900.0, 1800.0, 901))

    assert [(row["High"], row["Low"]) for row in rows] == [
        (1736.0, 1509.0),
        (1250.0, 1140.0),
    ]
    assert all(row["Enabled"] for row in rows)


def test_scientific_2d_draft_change_immediately_invalidates_old_result() -> None:
    app_module = _load_app_module()
    current = app_module.TwoDCOSConfig(
        ranges=(app_module.TwoDCOSRange(1800.0, 1500.0, "amide"),),
        convention="2dpy_compatible",
    )
    changed = replace(
        current,
        ranges=(app_module.TwoDCOSRange(1750.0, 1500.0, "amide"),),
    )
    old_result = object()
    state = {
        "twodcos_config": current,
        "twodcos_result": old_result,
        "twodcos_bundle": b"old bundle",
        "peak_order_result": object(),
        "twodcos_status": "TWODCOS_COMPLETED",
    }

    outcome = app_module._reconcile_twodcos_config(state, changed)

    assert outcome == "science_invalidated"
    assert state["twodcos_config"] is changed
    assert state["twodcos_result"] is None
    assert state["twodcos_bundle"] is None
    assert state["peak_order_result"] is None
    assert str(state["twodcos_status"]).startswith("TWODCOS_INVALIDATED")


def test_display_only_change_preserves_matrix_and_never_calls_compute(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    app_module = _load_app_module()
    current = app_module.TwoDCOSConfig(
        ranges=(app_module.TwoDCOSRange(1800.0, 1500.0, "amide"),),
    )
    changed = replace(
        current,
        display=app_module.TwoDCOSDisplayConfig(
            contour_levels=51,
            display_percentile=92.0,
        ),
    )
    matrix_result = object()
    peak_result = object()
    state = {
        "twodcos_config": current,
        "twodcos_result": matrix_result,
        "twodcos_bundle": b"old bundle",
        "peak_order_result": peak_result,
        "twodcos_status": "TWODCOS_COMPLETED",
    }

    def forbidden_compute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("display-only changes must not compute 2D")

    monkeypatch.setattr(
        app_module.TwoDCOSWorkflowService,
        "compute",
        forbidden_compute,
    )
    outcome = app_module._reconcile_twodcos_config(state, changed)

    assert outcome == "display_updated"
    assert state["twodcos_result"] is matrix_result
    assert state["peak_order_result"] is peak_result
    assert state["twodcos_bundle"] is None
    assert state["twodcos_config"] is changed
    assert str(state["twodcos_status"]).startswith("DISPLAY_UPDATED")


def test_display_config_changes_contour_render_without_matrix_recompute() -> None:
    app_module = _load_app_module()
    axis = np.array([1800.0, 1700.0, 1600.0])
    matrix = np.array(
        [
            [1.0, 0.5, -0.2],
            [0.5, 0.8, -0.4],
            [-0.2, -0.4, 0.3],
        ]
    )

    coarse = app_module._heatmap(
        axis,
        axis,
        matrix,
        title="coarse",
        percentile=100.0,
        contour_levels=11,
    )
    fine = app_module._heatmap(
        axis,
        axis,
        matrix,
        title="fine",
        percentile=80.0,
        contour_levels=41,
    )

    assert coarse.data[0].type == "contour"
    assert fine.data[0].type == "contour"
    assert coarse.data[0].contours.size != fine.data[0].contours.size
    assert coarse.data[0].zmax != fine.data[0].zmax


def test_switching_prepared_branch_invalidates_all_2d_descendants() -> None:
    app_module = _load_app_module()
    branch = SimpleNamespace(prepared_data_sha256="3" * 64)
    state = {
        "active_prepared": object(),
        "prepared_source": "old",
        "twodcos_config": object(),
        "twodcos_result": object(),
        "twodcos_bundle": b"old bundle",
        "peak_order_result": object(),
        "twodcos_status": "TWODCOS_COMPLETED",
    }

    app_module._activate_prepared_for_twodcos(
        state,
        branch,
        source="scientific sensitivity branch (vector)",
    )

    assert state["active_prepared"] is branch
    assert state["prepared_source"] == "scientific sensitivity branch (vector)"
    assert state["twodcos_config"] is None
    assert state["twodcos_result"] is None
    assert state["twodcos_bundle"] is None
    assert state["peak_order_result"] is None
    assert str(state["twodcos_status"]).startswith("PREPARED_FOR_2DCOS")
    assert "scientific sensitivity branch (vector)" in state["twodcos_status"]
    assert branch.prepared_data_sha256 in state["twodcos_status"]


def test_baseline_export_page_renders_without_calling_twodcos(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    data = read_spectrum_file(
        PROJECT_ROOT / "examples" / "baseline" / "example_absorbance.csv",
        input_unit="absorbance",
    )
    service = BaselineWorkflowService()
    result = service.run(
        data,
        PipelineConfig(
            input_unit="absorbance",
            wavenumber_range=(1800.0, 900.0),
            series_mode="independent_locked",
            coarse_baseline={"method": "none"},
            fine_baseline={"enabled": False, "method": "none"},
            normalization={"method": "none"},
        ),
    )
    prepared = service.prepared(result, baseline_run_id="ui-baseline-only")
    bundle = service.export_baseline_only(result, prepared=prepared, qc_figures={})

    def forbidden_compute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("baseline export UI must not compute 2D")

    monkeypatch.setattr(TwoDCOSWorkflowService, "compute", forbidden_compute)
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.session_state["baseline_result"] = result
    app.session_state["prepared"] = prepared
    app.session_state["baseline_bundle"] = bundle
    matrix_result = object()
    app.session_state["twodcos_result"] = matrix_result
    app.sidebar.radio[0].set_value("7. Baseline Result & Export").run()

    assert not app.exception
    assert app.session_state["twodcos_result"] is matrix_result
    assert app.session_state["active_prepared"] is None
    assert any(
        element.label == "导出校正谱并结束"
        for element in app.get("download_button")
    )
    assert {
        "Download derived fraction transmittance",
        "Download derived percent transmittance",
    } <= {element.label for element in app.get("download_button")}
    display = next(item for item in app.radio if item.label == "Display intensity:")
    baseline_config = dict(app.session_state["baseline_config"])
    prepared_hash = app.session_state["prepared"].prepared_data_sha256

    display.set_value("percent_transmittance").run()

    assert not app.exception
    assert app.session_state["baseline_config"] == baseline_config
    assert app.session_state["prepared"].prepared_data_sha256 == prepared_hash
    assert app.session_state["twodcos_result"] is matrix_result
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        names = set(archive.namelist())
    assert not any("derived_" in name for name in names)


def test_baseline_preview_draft_is_non_destructive_until_explicit_commit() -> None:
    app_module = _load_app_module()
    committed = PipelineConfig(
        input_unit="absorbance",
        coarse_baseline={"method": "none"},
        fine_baseline={"enabled": False, "method": "none"},
    )
    sentinels = {key: object() for key in app_module._BASELINE_DESCENDANT_KEYS}
    state = {"baseline_config": committed.to_dict(), **sentinels}

    draft = app_module._coarse_draft_payload(
        committed,
        method="arpls",
        series_mode="collaborative_pls",
        lam=1.0e6,
        p=0.01,
        max_iter=50,
        tol=1.0e-3,
        smoothing=True,
    )

    assert state["baseline_config"] == committed.to_dict()
    assert all(state[key] is value for key, value in sentinels.items())
    assert app_module._commit_baseline_payload_to_state(state, draft) is True
    assert state["baseline_config"]["coarse_baseline"]["method"] == "arpls"
    assert all(state[key] is None for key in app_module._BASELINE_DESCENDANT_KEYS)


def test_preview_runner_passes_complete_loaded_series_to_frozen_pipeline(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    app_module = _load_app_module()
    complete_series = object()
    expected_result = object()
    calls: list[tuple[object, PipelineConfig]] = []

    def fake_run_pipeline(data: object, config: PipelineConfig) -> object:
        calls.append((data, config))
        return expected_result

    monkeypatch.setattr(app_module, "run_pipeline", fake_run_pipeline)
    payload = PipelineConfig(input_unit="absorbance").to_dict()

    actual = app_module._run_baseline_preview(complete_series, payload)

    assert actual is expected_result
    assert calls == [(complete_series, PipelineConfig(input_unit="absorbance"))]


def test_series_consistency_page_exposes_five_heatmaps_qc_and_drill_down() -> None:
    data = read_spectrum_file(
        PROJECT_ROOT / "examples" / "baseline" / "example_absorbance.csv",
        input_unit="absorbance",
    )
    result = BaselineWorkflowService().run(
        data,
        PipelineConfig(
            input_unit="absorbance",
            wavenumber_range=(1800.0, 900.0),
            series_mode="independent_locked",
            coarse_baseline={"method": "none"},
            fine_baseline={"enabled": False, "method": "none"},
            normalization={"method": "none"},
        ),
    )
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.session_state["raw_data"] = data
    app.session_state["baseline_result"] = result

    app.sidebar.radio[0].set_value("5. Series Consistency & QC").run()

    assert not app.exception
    tab_labels = {tab.label for tab in app.tabs}
    assert {
        "Raw absorbance",
        "Coarse baseline",
        "Fine baseline",
        "Total baseline",
        "Corrected absorbance",
        "Baseline continuity",
        "Residual diagnostics",
        "Peak preservation",
    } <= tab_labels
    assert any(
        button.label == "Download per-spectrum QC CSV"
        for button in app.get("download_button")
    )
    assert len(app.dataframe) >= 2
