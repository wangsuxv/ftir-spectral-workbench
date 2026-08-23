from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ftir2dcos.config import PipelineConfig, WavenumberRange
from ftir2dcos.peak_order import PeakRequest, ResolvedPairValues, infer_peak_order

_APP_PATH = Path(__file__).resolve().parents[2] / "legacy" / "twodcos_streamlit_app.py"
_SPEC = importlib.util.spec_from_file_location("twodcos_streamlit_app_under_test", _APP_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError("could not load legacy/twodcos_streamlit_app.py for testing")
streamlit_app = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = streamlit_app
_SPEC.loader.exec_module(streamlit_app)


class NamedBytesIO(io.BytesIO):
    def __init__(self, content: bytes, name: str) -> None:
        super().__init__(content)
        self.name = name


def wide_upload() -> NamedBytesIO:
    rows = ["Wavenumber,0 min,5 min,10 min,15 min,20 min"]
    for index, wavenumber in enumerate(np.linspace(1800.0, 1000.0, 81)):
        intensities = [0.1 + index * 0.001 + spectrum * 0.01 for spectrum in range(5)]
        rows.append(",".join([f"{wavenumber:.1f}", *(f"{value:.6f}" for value in intensities)]))
    return NamedBytesIO(("\n".join(rows) + "\n").encode(), "wide.csv")


def test_anchor_range_parser() -> None:
    assert streamlit_app.parse_anchor_ranges("1509, 1520; 1720 - 1736") == (
        (1509.0, 1520.0),
        (1720.0, 1736.0),
    )
    assert streamlit_app.parse_anchor_ranges("") == ()
    with pytest.raises(ValueError, match="two wavenumbers"):
        streamlit_app.parse_anchor_ranges("1509, 1520, 1530")
    with pytest.raises(ValueError, match="distinct"):
        streamlit_app.parse_anchor_ranges("1509, 1509")


def test_upload_dispatch_and_preview_frame() -> None:
    dataset = streamlit_app.load_uploaded_dataset([wide_upload()])

    assert dataset.shape == (5, 81)
    assert dataset.metadata["delimiter_name"] == "comma"
    assert dataset.perturbation_labels == (
        "0 min",
        "5 min",
        "10 min",
        "15 min",
        "20 min",
    )
    frame = streamlit_app.dataset_preview_frame(dataset)
    assert frame.shape == (10, 6)
    assert frame.columns.tolist()[0] == "Wavenumber"
    np.testing.assert_array_equal(frame.iloc[:, 0], dataset.wavenumber[:10])


def test_multiple_dpt_uploads_keep_upload_order() -> None:
    axis = np.linspace(1736.0, 1509.0, 12)
    uploads = []
    for label, offset in (("10MIN.dpt", 0.1), ("0MIN.dpt", 0.2), ("5MIN.dpt", 0.3)):
        text = "\n".join(f"{x:.3f},{offset + index * 0.001:.6f}" for index, x in enumerate(axis))
        uploads.append(NamedBytesIO((text + "\n").encode(), label))

    dataset = streamlit_app.load_uploaded_dataset(uploads)

    assert dataset.perturbation_labels == ("10MIN", "0MIN", "5MIN")
    np.testing.assert_array_equal(dataset.perturbation, [10.0, 0.0, 5.0])
    with pytest.raises(ValueError, match="cannot be mixed"):
        streamlit_app.load_uploaded_dataset([wide_upload(), uploads[0]])


def test_default_range_prefers_requested_window() -> None:
    assert streamlit_app.default_analysis_range(np.array([1800.0, 1700.0, 1400.0])) == (
        1509.0,
        1736.0,
    )
    assert streamlit_app.default_analysis_range(np.array([900.0, 1000.0, 1100.0])) == (
        900.0,
        1100.0,
    )


def test_default_ranges_include_only_intervals_covered_by_axis() -> None:
    both = streamlit_app.default_wavenumber_ranges(np.linspace(1000.0, 1800.0, 81))
    first_only = streamlit_app.default_wavenumber_ranges(np.linspace(1400.0, 1800.0, 41))
    fallback = streamlit_app.default_wavenumber_ranges(np.linspace(800.0, 1000.0, 21))

    assert [item.bounds for item in both] == [(1509.0, 1736.0), (1140.0, 1250.0)]
    assert [item.bounds for item in first_only] == [(1509.0, 1736.0)]
    assert fallback[0].bounds == (800.0, 1000.0)
    assert fallback[0].label == "full range"


def test_dynamic_range_validation_is_strict() -> None:
    axis = np.linspace(1000.0, 1800.0, 801)
    frame = streamlit_app.range_editor_frame(axis)

    ranges = streamlit_app.validate_wavenumber_ranges(frame, axis)
    assert [item.bounds for item in ranges] == [(1509.0, 1736.0), (1140.0, 1250.0)]

    duplicate = pd.concat(
        [
            frame,
            pd.DataFrame(
                [{"Label": "duplicate", "High wavenumber": 1509.0, "Low wavenumber": 1736.0}]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicates"):
        streamlit_app.validate_wavenumber_ranges(duplicate, axis)

    outside = frame.iloc[[0]].copy()
    outside.loc[0, "High wavenumber"] = 1900.0
    with pytest.raises(ValueError, match="outside"):
        streamlit_app.validate_wavenumber_ranges(outside, axis)

    zero_width = frame.iloc[[0]].copy()
    zero_width.loc[0, "Low wavenumber"] = zero_width.loc[0, "High wavenumber"]
    with pytest.raises(ValueError, match="distinct"):
        streamlit_app.validate_wavenumber_ranges(zero_width, axis)

    missing = frame.iloc[[0]].copy()
    missing.loc[0, "Low wavenumber"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        streamlit_app.validate_wavenumber_ranges(missing, axis)

    too_narrow = pd.DataFrame(
        [{"Label": "tiny", "High wavenumber": 1701.0, "Low wavenumber": 1700.0}]
    )
    with pytest.raises(ValueError, match="measured points"):
        streamlit_app.validate_wavenumber_ranges(too_narrow, axis)


def test_common_wavenumber_bounds_detects_shared_windows() -> None:
    assert streamlit_app.common_wavenumber_bounds(
        (WavenumberRange(1736, 1509), WavenumberRange(1650, 1500))
    ) == (1509.0, 1650.0)
    assert (
        streamlit_app.common_wavenumber_bounds(
            (WavenumberRange(1736, 1509), WavenumberRange(1250, 1140))
        )
        is None
    )


def test_matrix_memory_estimate_includes_unique_rectangular_cross_blocks() -> None:
    axis = np.arange(1.0, 11.0)
    ranges = (
        WavenumberRange(4.0, 1.0, label="first"),
        WavenumberRange(10.0, 6.0, label="second"),
    )

    canonical, canonical_bytes = streamlit_app.matrix_memory_estimate_frame(
        axis, ranges, convention="canonical"
    )
    compatible, compatible_bytes = streamlit_app.matrix_memory_estimate_frame(
        axis, ranges, convention="2dpy_compatible"
    )

    assert canonical["Type"].tolist() == ["Self", "Self", "Cross"]
    assert canonical["Matrix shape"].tolist() == ["4 x 4", "5 x 5", "4 x 5"]
    assert compatible["Matrix shape"].tolist() == ["4 x 4", "5 x 5", "5 x 4"]
    assert canonical_bytes == compatible_bytes == 976
    assert canonical["Two matrices (MiB)"].sum() == pytest.approx(976 / 1024**2)


def test_range_editor_uses_dynamic_rows_and_stable_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = streamlit_app.load_uploaded_dataset([wide_upload()])
    captured: dict[str, object] = {}

    def fake_editor(data: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return data

    monkeypatch.setattr(streamlit_app.st, "data_editor", fake_editor)
    monkeypatch.setattr(streamlit_app.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "caption", lambda *args, **kwargs: None)
    ranges = streamlit_app._render_wavenumber_ranges(dataset, key_prefix="source123")

    assert ranges is not None and len(ranges) == 2
    assert captured["num_rows"] == "dynamic"
    assert captured["key"] == "wavenumber_ranges_source123"


def test_peak_editor_validates_optional_one_based_range_choices() -> None:
    dataset = streamlit_app.load_uploaded_dataset([wide_upload()])
    result = streamlit_app.run_multi_range_pipeline(
        dataset,
        streamlit_app.default_wavenumber_ranges(dataset.wavenumber),
        PipelineConfig(),
    )
    options = streamlit_app.peak_range_options(result)
    frame = pd.DataFrame(
        [
            {
                streamlit_app.PEAK_WAVENUMBER_COLUMN: 1700.0,
                streamlit_app.PEAK_LABEL_COLUMN: "amide",
                streamlit_app.PEAK_RANGE_COLUMN: options[1],
            },
            {
                streamlit_app.PEAK_WAVENUMBER_COLUMN: 1200.0,
                streamlit_app.PEAK_LABEL_COLUMN: "",
                streamlit_app.PEAK_RANGE_COLUMN: streamlit_app.AUTO_PEAK_RANGE,
            },
            {
                streamlit_app.PEAK_WAVENUMBER_COLUMN: np.nan,
                streamlit_app.PEAK_LABEL_COLUMN: "",
                streamlit_app.PEAK_RANGE_COLUMN: "",
            },
        ]
    )

    requests = streamlit_app.validate_peak_requests(frame, result)

    assert requests == (
        PeakRequest(1700.0, label="amide", range_index=0),
        PeakRequest(1200.0),
    )
    assert streamlit_app.peak_editor_frame().dtypes.astype(str).tolist() == [
        "float64",
        "string",
        "string",
    ]

    frame.loc[1, streamlit_app.PEAK_RANGE_COLUMN] = "unknown"
    with pytest.raises(ValueError, match="unknown analysis range"):
        streamlit_app.validate_peak_requests(frame, result)


def test_peak_order_form_is_dynamic_stable_and_submit_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = streamlit_app.load_uploaded_dataset([wide_upload()])
    result = streamlit_app.run_multi_range_pipeline(
        dataset,
        streamlit_app.default_wavenumber_ranges(dataset.wavenumber),
        PipelineConfig(),
    )
    cached = streamlit_app.CachedRun(
        result=result, bundle_bytes=b"", file_names=(), config_json="{}"
    )
    captured: dict[str, object] = {}

    class FakeColumn:
        def number_input(self, label: str, **kwargs: object) -> float:
            return float(kwargs["value"])

    def fake_editor(data: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return data

    monkeypatch.setattr(streamlit_app.st, "session_state", {"peak_order_cache": {}})
    monkeypatch.setattr(streamlit_app.st, "subheader", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "form", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(streamlit_app.st, "data_editor", fake_editor)
    monkeypatch.setattr(streamlit_app.st, "columns", lambda count: [FakeColumn()] * count)
    monkeypatch.setattr(streamlit_app.st, "form_submit_button", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        streamlit_app,
        "analyze_multi_range_peak_order",
        lambda *args, **kwargs: pytest.fail("analysis must wait for explicit form submission"),
    )

    streamlit_app._render_peak_response_order(cached, "stable123")

    assert captured["num_rows"] == "dynamic"
    assert captured["key"] == "peak_order_editor_stable123"


def test_peak_order_chain_and_evidence_expose_partial_order_audit() -> None:
    first = PeakRequest(1700.0, label="A", range_index=0)
    second = PeakRequest(1600.0, label="B", range_index=0)
    third = PeakRequest(1200.0, label="C", range_index=1)
    order = infer_peak_order(
        (first, second, third),
        (
            ResolvedPairValues(
                first,
                second,
                synchronous=2.0,
                asynchronous=1.0,
                matched_first_wavenumber=1699.5,
                matched_second_wavenumber=1600.5,
                sync_threshold=0.1,
                async_threshold=0.1,
                relative_signal_strength=0.5,
            ),
        ),
    )
    dataset = streamlit_app.load_uploaded_dataset([wide_upload()])
    result = streamlit_app.run_multi_range_pipeline(
        dataset,
        streamlit_app.default_wavenumber_ranges(dataset.wavenumber),
        PipelineConfig(),
    )

    chain = streamlit_app.peak_order_chain_text(order)
    evidence = streamlit_app.peak_evidence_frame(order, result)

    assert "∥" in chain
    assert not order.is_unique_total_order
    assert evidence.loc[0, "First requested (cm⁻¹)"] == 1700.0
    assert evidence.loc[0, "First matched (cm⁻¹)"] == 1699.5
    assert evidence.loc[0, "First range"].startswith("R1:")
    assert evidence.loc[0, "Synchronous Φ"] == 2.0
    assert evidence.loc[0, "Asynchronous Ψ"] == 1.0
    assert evidence.loc[0, "Sign product"] == 1
    assert evidence.loc[0, "Relative signal strength"] == 0.5
    assert "A" in evidence.loc[0, "Decision"]


def test_fingerprints_separate_scientific_and_display_settings() -> None:
    dataset = streamlit_app.load_uploaded_dataset([wide_upload()])
    config = PipelineConfig()
    ranges = (
        WavenumberRange(1736, 1509),
        WavenumberRange(1250, 1140),
    )
    display_changed = replace(config, contour_levels=31, display_percentile=95.0)
    convention_changed = replace(config, convention="canonical")

    assert streamlit_app.config_fingerprint(
        dataset, config, ranges=ranges
    ) == streamlit_app.config_fingerprint(dataset, display_changed, ranges=ranges)
    assert streamlit_app.config_fingerprint(
        dataset, config, ranges=ranges
    ) != streamlit_app.config_fingerprint(dataset, convention_changed, ranges=ranges)
    assert streamlit_app.config_fingerprint(
        dataset, config, ranges=ranges, preprocessing_only=True
    ) == streamlit_app.config_fingerprint(
        dataset, convention_changed, ranges=ranges, preprocessing_only=True
    )
    assert streamlit_app.config_fingerprint(
        dataset, config, ranges=ranges
    ) != streamlit_app.config_fingerprint(dataset, config, ranges=tuple(reversed(ranges)))

    edited = dataset.with_updates(perturbation=dataset.perturbation + 1.0)
    assert streamlit_app.config_fingerprint(
        dataset, config, ranges=ranges
    ) != streamlit_app.config_fingerprint(edited, config, ranges=ranges)
    assert streamlit_app.source_fingerprint(dataset) == streamlit_app.source_fingerprint(edited)


def test_axis_direction_and_bundle_member_helpers() -> None:
    assert streamlit_app.axis_direction(np.array([3.0, 2.0, 1.0])) == ("descending (high → low)")
    assert streamlit_app.axis_direction(np.array([1.0, 2.0, 3.0])) == ("ascending (low → high)")
    assert streamlit_app.axis_direction(np.array([1.0, 3.0, 2.0])) == "non-monotonic"
    with pytest.raises(ValueError, match="non-empty"):
        streamlit_app.axis_direction(np.array([]))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("figures/multi_range_synchronous_blocks.png", b"fake-png")
    assert (
        streamlit_app.bundle_member_bytes(
            buffer.getvalue(), "figures/multi_range_synchronous_blocks.png"
        )
        == b"fake-png"
    )
    assert streamlit_app.bundle_member_bytes(buffer.getvalue(), "missing.png") is None


def test_previews_are_generated_independently_for_every_range() -> None:
    dataset = streamlit_app.load_uploaded_dataset([wide_upload()])
    ranges = streamlit_app.default_wavenumber_ranges(dataset.wavenumber)

    previews = streamlit_app.generate_range_previews(dataset, ranges, PipelineConfig())

    assert len(previews) == 2
    for analysis_range, preview in zip(ranges, previews, strict=True):
        assert preview.selected_raw.wavenumber[0] >= analysis_range.low_wavenumber
        assert preview.selected_raw.wavenumber[-1] <= analysis_range.high_wavenumber


def test_cross_range_axis_metadata_follows_active_convention() -> None:
    dataset = streamlit_app.load_uploaded_dataset([wide_upload()])
    ranges = streamlit_app.default_wavenumber_ranges(dataset.wavenumber)

    result = streamlit_app.run_multi_range_pipeline(dataset, ranges, PipelineConfig())

    assert len(result.cross_results) == 1
    cross_result = result.cross_results[0]
    frame = streamlit_app.cross_axis_frame(cross_result)
    assert frame["Matrix axis"].tolist() == ["Row (vertical)", "Column (horizontal)"]
    assert frame["Analysis range"].tolist() == [
        cross_result.row_range.display_name,
        cross_result.column_range.display_name,
    ]
    assert frame["Variable"].tolist() == [
        cross_result.twodcos.row_variable,
        cross_result.twodcos.column_variable,
    ]
    assert frame["Points"].tolist() == list(cross_result.twodcos.synchronous.shape)
    assert frame["Displayed contour direction"].tolist() == [
        "high → low cm⁻¹",
        "high → low cm⁻¹",
    ]
    np.testing.assert_array_equal(
        streamlit_app._block_matrix(
            result,
            row_index=cross_result.row_index,
            column_index=cross_result.column_index,
            kind="synchronous",
        ),
        cross_result.twodcos.synchronous,
    )
    np.testing.assert_array_equal(
        streamlit_app._block_matrix(
            result,
            row_index=cross_result.column_index,
            column_index=cross_result.row_index,
            kind="asynchronous",
        ),
        cross_result.twodcos.reverse_asynchronous,
    )
    assert bool(cross_result.qc_metrics["all_checks_passed"])


def test_results_fall_back_to_independent_view_for_legacy_empty_cross_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    range_item = SimpleNamespace(result=SimpleNamespace(qc_metrics={"all_checks_passed": True}))
    legacy_result = SimpleNamespace(range_results=(range_item,))
    cached = streamlit_app.CachedRun(
        result=legacy_result,
        bundle_bytes=b"",
        file_names=(),
        config_json="{}",
    )
    rendered: list[str] = []

    monkeypatch.setattr(streamlit_app.st, "header", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        streamlit_app.st,
        "segmented_control",
        lambda *args, **kwargs: pytest.fail("single-range fallback needs no view selector"),
    )
    monkeypatch.setattr(
        streamlit_app,
        "_render_independent_results",
        lambda *args, **kwargs: rendered.append("independent"),
    )

    streamlit_app._render_results(cached, PipelineConfig(), "legacy")

    assert rendered == ["independent"]


def test_block_overview_uses_plotting_fallback_when_export_png_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = streamlit_app.CachedRun(
        result=SimpleNamespace(),
        bundle_bytes=b"legacy-zip",
        file_names=(),
        config_json="{}",
    )
    shown: list[object] = []
    sentinel_figure = object()

    monkeypatch.setattr(streamlit_app, "bundle_member_bytes", lambda *args: None)
    monkeypatch.setattr(
        streamlit_app,
        "_create_block_overview_figure",
        lambda *args, **kwargs: sentinel_figure,
    )
    monkeypatch.setattr(streamlit_app, "_show_figure", shown.append)
    monkeypatch.setattr(streamlit_app.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(streamlit_app.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        streamlit_app.st,
        "image",
        lambda *args, **kwargs: pytest.fail("missing PNG must use plotting fallback"),
    )

    streamlit_app._render_block_figure(cached, PipelineConfig(), kind="synchronous")

    assert shown == [sentinel_figure]


def test_execute_pipeline_bundle_retains_zip_after_temporary_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = streamlit_app.load_uploaded_dataset([wide_upload()])
    config = PipelineConfig()
    ranges = streamlit_app.default_wavenumber_ranges(dataset.wavenumber)
    calls: list[tuple[object, object, object]] = []

    def fake_run_multi_range_pipeline(
        source: object,
        range_values: object,
        pipeline_config: object,
        *,
        output_root: str,
        input_paths: tuple[()],
    ) -> object:
        calls.append((source, range_values, pipeline_config))
        bundle = Path(output_root) / "multi_range_bundle.zip"
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr("multi_range_manifest.json", "{}")
        return SimpleNamespace(bundle_path=bundle)

    monkeypatch.setattr(
        streamlit_app,
        "run_multi_range_pipeline",
        fake_run_multi_range_pipeline,
    )
    cached = streamlit_app.execute_pipeline_bundle(dataset, ranges, config)

    assert calls == [(dataset, ranges, config)]
    assert cached.file_names == ("multi_range_manifest.json",)
    with zipfile.ZipFile(io.BytesIO(cached.bundle_bytes)) as archive:
        assert archive.namelist() == ["multi_range_manifest.json"]
    assert '"analysis_ranges"' in cached.config_json


def test_streamlit_app_starts_without_an_upload() -> None:
    app = AppTest.from_file(_APP_PATH).run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "FTIR preprocessing and 2D-COS"
    assert "Upload data to begin" in app.info[0].value
