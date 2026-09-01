from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from ftir_workbench import (
    PostBaselineSmoothingConfig,
    PostBaselineSmoothingService,
    PreparedSpectralDataset,
    TwoDCOSConfig,
    TwoDCOSRange,
    TwoDCOSWorkflowService,
)
from ftir_workbench.fingerprints import prepared_data_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "ui" / "streamlit_app.py"


def _load_app_module():  # type: ignore[no-untyped-def]
    sys.path.insert(0, str(APP_PATH.parent))
    spec = spec_from_file_location("smoothing_streamlit_app_under_test", APP_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("could not load unified Streamlit app")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _element(elements, label: str):  # type: ignore[no-untyped-def]
    return next(element for element in elements if element.label == label)


def _prepared() -> PreparedSpectralDataset:
    axis = np.linspace(1800.0, 1000.0, 41)
    position = np.linspace(-1.0, 1.0, axis.size)
    peak = np.exp(-0.5 * (position / 0.18) ** 2)
    ripple = 0.02 * np.sin(np.arange(axis.size) * 1.7)
    spectra = np.vstack(
        (
            0.10 + 0.35 * peak + ripple,
            0.15 + 0.55 * peak - 0.7 * ripple,
            0.20 + 0.75 * peak + 1.2 * ripple,
        )
    )
    perturbation = np.array([0.0, 2.0, 5.0])
    labels = ("0 min", "2 min", "5 min")
    digest = prepared_data_sha256(axis, perturbation, labels, spectra)
    return PreparedSpectralDataset(
        wavenumber=axis,
        perturbation=perturbation,
        perturbation_labels=labels,
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="smoothing-ui-fixture.csv",
        source_sha256="1" * 64,
        baseline_run_id="smoothing-ui-baseline",
        baseline_fingerprint="2" * 64,
        prepared_data_sha256=digest,
        original_axis_direction="descending",
        current_axis_direction="descending",
        perturbation_order_policy="preserve_file_order",
        baseline_recipe={
            "prepared_data_contract": {
                "source_channel": "PipelineResult.analysis_data",
                "scientific_normalization": False,
            }
        },
        baseline_qc={"passed": True},
        warnings=(),
    )


def test_smoothing_source_prefers_primary_and_marks_active_fallback() -> None:
    app_module = _load_app_module()
    primary = object()
    active = object()

    selected, source, fallback = app_module._smoothing_page_source(
        {
            "prepared": primary,
            "active_prepared": active,
            "prepared_source": "active scientific branch",
        }
    )
    assert selected is primary
    assert source == "primary unsmoothed Prepared from the current baseline run"
    assert fallback is False

    selected, source, fallback = app_module._smoothing_page_source(
        {
            "prepared": None,
            "active_prepared": active,
            "prepared_source": "reloaded corrected absorbance",
        }
    )
    assert selected is active
    assert source == "fallback active Prepared: reloaded corrected absorbance"
    assert fallback is True


def test_baseline_invalidation_clears_preview_and_formal_smoothing_not_draft() -> None:
    app_module = _load_app_module()
    draft = PostBaselineSmoothingConfig(enabled=True)
    state = {
        key: object() for key in app_module._BASELINE_DESCENDANT_KEYS
    }
    state["smoothing_draft_config"] = draft

    app_module._invalidate_from_baseline_state(state)

    for key in (
        "smoothing_preview_config",
        "smoothing_preview_parent_hash",
        "smoothing_preview_result",
        "smoothing_result",
        "smoothed_prepared",
        "smoothing_bundle",
    ):
        assert state[key] is None
    assert state["smoothing_draft_config"] is draft


def test_branch_activation_clears_only_2d_and_preserves_all_smoothing_state() -> None:
    app_module = _load_app_module()
    primary = _prepared()
    _, child = PostBaselineSmoothingService().apply(
        primary,
        PostBaselineSmoothingConfig(enabled=True, method="savgol"),
    )
    smoothing_state = {
        "smoothing_preview_config": object(),
        "smoothing_preview_parent_hash": "parent-hash",
        "smoothing_preview_result": object(),
        "smoothing_result": object(),
        "smoothed_prepared": child,
        "smoothing_bundle": b"future-bundle",
    }
    state = {
        "prepared": primary,
        "baseline_result": object(),
        **smoothing_state,
        "active_prepared": primary,
        "prepared_source": "primary",
        "twodcos_config": object(),
        "twodcos_result": object(),
        "twodcos_bundle": b"old-2d",
        "peak_order_result": object(),
        "twodcos_status": "TWODCOS_COMPLETED",
    }

    app_module._activate_prepared_for_twodcos(
        state,
        child,
        source="post-baseline smoothing branch (savgol)",
    )

    assert state["prepared"] is primary
    assert state["baseline_result"] is not None
    assert state["active_prepared"] is child
    assert state["prepared_source"] == "post-baseline smoothing branch (savgol)"
    for key, value in smoothing_state.items():
        assert state[key] is value
    assert state["twodcos_config"] is None
    assert state["twodcos_result"] is None
    assert state["twodcos_bundle"] is None
    assert state["peak_order_result"] is None
    assert "post-baseline smoothing branch (savgol)" in state["twodcos_status"]
    assert child.prepared_data_sha256 in state["twodcos_status"]


def test_smoothing_page_preview_apply_staleness_and_branch_switching() -> None:
    primary = _prepared()
    previous_active = object()
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.session_state["prepared"] = primary
    app.session_state["active_prepared"] = previous_active
    app.session_state["prepared_source"] = "unrelated active branch"

    app.sidebar.radio[0].set_value("8. Post-Baseline Smoothing").run()

    assert not app.exception
    enable = _element(app.checkbox, "Enable post-baseline smoothing")
    algorithm = _element(app.selectbox, "Algorithm")
    assert enable.value is False
    assert list(algorithm.options) == [
        "Savitzky–Golay",
        "Gaussian",
        "Moving Average / Uniform",
        "Median / despike (expert)",
    ]
    assert _element(app.button, "Create Smoothed Scientific Branch").disabled is True
    assert _element(app.button, "Use Smoothed Branch for 2D-COS").disabled is True
    assert app.session_state["active_prepared"] is previous_active

    enable.set_value(True).run()
    _element(app.button, "Generate Preview").click().run()

    assert not app.exception
    preview = app.session_state["smoothing_preview_result"]
    assert preview is not None
    assert app.session_state["smoothing_preview_config"].enabled is True
    assert app.session_state["smoothing_preview_parent_hash"] == (
        primary.prepared_data_sha256
    )
    assert app.session_state["smoothing_result"] is None
    assert app.session_state["smoothed_prepared"] is None
    assert app.session_state["active_prepared"] is previous_active
    assert _element(app.button, "Create Smoothed Scientific Branch").disabled is False
    assert len(app.get("image")) >= 2
    metric_labels = {metric.label for metric in app.get("metric")}
    assert {
        "Mean relative RMS removed",
        "Min derivative correlation",
        "Max absolute-area change",
        "Mean roughness ratio",
        "Max edge-effect ratio",
    }.issubset(metric_labels)
    assert any(
        "spectrum_index" in dataframe.value.columns
        for dataframe in app.dataframe
    )

    _element(app.button, "Create Smoothed Scientific Branch").click().run()

    assert not app.exception
    formal = app.session_state["smoothing_result"]
    child = app.session_state["smoothed_prepared"]
    assert formal is not None
    assert child is not None
    assert app.session_state["active_prepared"] is previous_active
    assert app.session_state["prepared"] is primary
    assert app.session_state["smoothing_bundle"] is None

    formal_fingerprint = formal.smoothing_fingerprint
    _element(app.selectbox, "Preview spectrum").set_value("mean").run()
    assert app.session_state["smoothing_result"].smoothing_fingerprint == (
        formal_fingerprint
    )
    _element(app.selectbox, "Algorithm").set_value("gaussian").run()
    assert app.session_state["smoothing_result"].smoothing_fingerprint == (
        formal_fingerprint
    )
    assert _element(app.button, "Create Smoothed Scientific Branch").disabled is True
    assert any(
        "Current draft differs from committed smoothed branch." in warning.value
        for warning in app.warning
    )

    app.session_state["twodcos_config"] = object()
    app.session_state["twodcos_result"] = object()
    app.session_state["twodcos_bundle"] = b"stale"
    app.session_state["peak_order_result"] = object()
    _element(app.button, "Use Smoothed Branch for 2D-COS").click().run()

    assert app.session_state["active_prepared"].prepared_data_sha256 == (
        child.prepared_data_sha256
    )
    assert app.session_state["prepared"] is primary
    assert app.session_state["smoothed_prepared"].prepared_data_sha256 == (
        child.prepared_data_sha256
    )
    assert app.session_state["smoothing_result"].smoothing_fingerprint == (
        formal_fingerprint
    )
    assert app.session_state["twodcos_config"] is None
    assert app.session_state["twodcos_result"] is None
    assert app.session_state["twodcos_bundle"] is None
    assert app.session_state["peak_order_result"] is None
    assert "post-baseline smoothing branch (savgol)" in (
        app.session_state["twodcos_status"]
    )
    assert child.prepared_data_sha256 in app.session_state["twodcos_status"]

    _element(app.button, "Use Unsmoothed Branch for 2D-COS").click().run()
    assert app.session_state["active_prepared"] is primary
    assert app.session_state["smoothed_prepared"].prepared_data_sha256 == (
        child.prepared_data_sha256
    )


def test_smoothing_method_controls_are_conditional() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.session_state["prepared"] = _prepared()
    app.sidebar.radio[0].set_value("8. Post-Baseline Smoothing").run()

    _element(app.checkbox, "Enable post-baseline smoothing").set_value(True).run()
    assert app.session_state["smoothing_draft_config"].enabled is True
    _element(app.checkbox, "Enable post-baseline smoothing").set_value(False).run()
    assert app.session_state["smoothing_draft_config"].enabled is False

    number_labels = {item.label for item in app.number_input}
    select_labels = {item.label for item in app.selectbox}
    assert {
        "Savitzky–Golay window length",
        "Savitzky–Golay polynomial order",
    }.issubset(number_labels)
    assert "Savitzky–Golay boundary mode" in select_labels
    assert "Gaussian sigma (points)" not in number_labels

    _element(app.selectbox, "Algorithm").set_value("gaussian").run()
    number_labels = {item.label for item in app.number_input}
    select_labels = {item.label for item in app.selectbox}
    assert {"Gaussian sigma (points)", "Gaussian truncate"}.issubset(number_labels)
    assert "Convolution boundary mode" in select_labels
    assert "Savitzky–Golay window length" not in number_labels

    _element(app.selectbox, "Algorithm").set_value("moving_average").run()
    number_labels = {item.label for item in app.number_input}
    select_labels = {item.label for item in app.selectbox}
    assert "Moving-average window length" in number_labels
    assert "Convolution boundary mode" in select_labels
    assert "Gaussian sigma (points)" not in number_labels

    _element(app.selectbox, "Algorithm").set_value("median").run()
    number_labels = {item.label for item in app.number_input}
    select_labels = {item.label for item in app.selectbox}
    assert "Median window length" in number_labels
    assert "Convolution boundary mode" in select_labels
    assert "Moving-average window length" not in number_labels
    assert any("非线性 expert 方法" in warning.value for warning in app.warning)


def test_smoothed_2d_setup_shows_active_branch_lineage() -> None:
    primary = _prepared()
    _, child = PostBaselineSmoothingService().apply(
        primary,
        PostBaselineSmoothingConfig(enabled=True, method="gaussian"),
    )
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.session_state["prepared"] = primary
    app.session_state["active_prepared"] = child
    app.session_state["prepared_source"] = "post-baseline smoothing branch (gaussian)"

    app.sidebar.radio[0].set_value("9. Optional 2D-COS Setup").run()

    assert not app.exception
    assert any(
        "Prepared branch kind: post_baseline_smoothing" in caption.value
        for caption in app.caption
    )
    assert any(
        child.prepared_data_sha256 in code.value
        and primary.prepared_data_sha256 in code.value
        for code in app.code
    )


def test_smoothing_page_rejects_chained_fallback_before_controls() -> None:
    primary = _prepared()
    _, child = PostBaselineSmoothingService().apply(
        primary,
        PostBaselineSmoothingConfig(enabled=True, method="gaussian"),
    )
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.session_state["prepared"] = None
    app.session_state["active_prepared"] = child
    app.session_state["prepared_source"] = "reloaded smoothed Prepared"

    app.sidebar.radio[0].set_value("8. Post-Baseline Smoothing").run()

    assert not app.exception
    assert any("禁止 chained smoothing" in error.value for error in app.error)
    assert not any(
        checkbox.label == "Enable post-baseline smoothing"
        for checkbox in app.checkbox
    )
    assert not any(
        button.label == "Use Unsmoothed Branch for 2D-COS"
        for button in app.button
    )


def test_smoothed_2d_results_show_lineage_and_skip_legacy_project_embedding() -> None:
    primary = _prepared()
    _, child = PostBaselineSmoothingService().apply(
        primary,
        PostBaselineSmoothingConfig(enabled=True, method="gaussian"),
    )
    config = TwoDCOSConfig(
        ranges=(
            TwoDCOSRange(1800.0, 1450.0, "upper"),
            TwoDCOSRange(1350.0, 1000.0, "lower"),
        ),
        convention="canonical",
        nonuniform_perturbation_policy="allow",
        cross_range_enabled=True,
    )
    analysis = TwoDCOSWorkflowService().compute(child, config)
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.session_state["prepared"] = primary
    app.session_state["active_prepared"] = child
    app.session_state["prepared_source"] = "post-baseline smoothing branch (gaussian)"
    app.session_state["twodcos_config"] = config
    app.session_state["twodcos_result"] = analysis
    app.session_state["twodcos_status"] = "TWODCOS_COMPLETED"
    app.session_state["baseline_bundle"] = b"invalid-if-project-builder-is-called"

    app.sidebar.radio[0].set_value("10. 2D-COS Results").run()

    assert not app.exception
    assert any(
        "Prepared branch kind: post_baseline_smoothing" in caption.value
        for caption in app.caption
    )
    assert any(
        "旧 .ftirw project schema" in info.value
        for info in app.info
    )
    download_labels = {item.label for item in app.get("download_button")}
    assert "下载 2D-COS bundle" in download_labels
    assert "下载完整 project.ftirw" not in download_labels
