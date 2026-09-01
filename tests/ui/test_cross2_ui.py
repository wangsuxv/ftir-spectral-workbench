from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from ftir_baseline.config import PipelineConfig
from ftir_baseline.io import read_spectrum_file
from ftir_workbench.config import TwoDCOSConfig, TwoDCOSRange
from ftir_workbench.services import BaselineWorkflowService, TwoDCOSWorkflowService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "ui" / "streamlit_app.py"
UI_ROOT = APP_PATH.parent


def _load_app_module():  # type: ignore[no-untyped-def]
    sys.path.insert(0, str(UI_ROOT))
    spec = spec_from_file_location("cross2_streamlit_app_under_test", APP_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("could not load unified Streamlit app")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepared_and_analysis():  # type: ignore[no-untyped-def]
    data = read_spectrum_file(
        PROJECT_ROOT / "examples" / "baseline" / "example_absorbance.csv",
        input_unit="absorbance",
    )
    baseline = BaselineWorkflowService()
    baseline_result = baseline.run(
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
    prepared = baseline.prepared(baseline_result, baseline_run_id="cross2-ui")
    config = TwoDCOSConfig(
        ranges=(
            TwoDCOSRange(1800.0, 1500.0, "upper"),
            TwoDCOSRange(1400.0, 900.0, "lower"),
        ),
        convention="2dpy_compatible",
        nonuniform_perturbation_policy="allow",
        cross_range_enabled=True,
    )
    analysis = TwoDCOSWorkflowService().compute(prepared, config)
    return prepared, config, analysis


def test_cross_numeric_preview_is_bounded_and_preserves_orientation() -> None:
    app_module = _load_app_module()
    matrix = np.arange(40 * 35, dtype=np.float64).reshape(40, 35)
    rows = np.arange(40, dtype=np.float64) + 1000.0
    columns = np.arange(35, dtype=np.float64) + 1500.0

    preview = app_module._cross_numeric_preview(matrix, rows, columns)

    assert preview.shape == (30, 30)
    np.testing.assert_array_equal(preview.to_numpy(), matrix[:30, :30])
    np.testing.assert_array_equal(preview.index.to_numpy(), rows[:30])


def test_cross2_ui_exposes_both_orientations_and_never_recomputes(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    prepared, config, analysis = _prepared_and_analysis()
    original_fingerprint = analysis.twodcos_fingerprint

    def forbidden_compute(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Cross 2 display must not recompute a unique pair")

    monkeypatch.setattr(TwoDCOSWorkflowService, "compute", forbidden_compute)
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.session_state["active_prepared"] = prepared
    app.session_state["prepared_source"] = "cross2 UI fixture"
    app.session_state["twodcos_config"] = config
    app.session_state["twodcos_result"] = analysis
    app.session_state["twodcos_status"] = "TWODCOS_COMPLETED"

    app.sidebar.radio[0].set_value("9. 2D-COS Results").run()

    assert not app.exception
    assert len(app.session_state["twodcos_result"].cross_results) == 1
    assert app.session_state["twodcos_result"].twodcos_fingerprint == original_fingerprint
    tab_labels = {tab.label for tab in app.tabs}
    assert {
        "Cross 1",
        "Cross 2",
        "QC / identities",
        "Synchronous block overview",
        "Asynchronous block overview",
    } <= tab_labels
    assert any(
        button.label == "下载 2D-COS bundle"
        for button in app.get("download_button")
    )
    orientation = next(
        item for item in app.radio if item.label == "Orientation focus (display only)"
    )

    orientation.set_value("reverse").run()

    assert not app.exception
    assert app.session_state["cross_selected_orientation"] == "reverse"
    assert app.session_state["twodcos_result"].twodcos_fingerprint == original_fingerprint
    assert app.session_state["peak_order_result"] is None
