from __future__ import annotations

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from ftir_baseline.config import PipelineConfig
from ftir_baseline.models import SpectrumSet

_SPEC = spec_from_file_location(
    "streamlit_app_under_test",
    Path(__file__).resolve().parents[2] / "legacy" / "baseline_streamlit_app.py",
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError("could not load streamlit_app.py for testing")
app = module_from_spec(_SPEC)
_SPEC.loader.exec_module(app)
_APP_PATH = Path(__file__).resolve().parents[2] / "legacy" / "baseline_streamlit_app.py"


class _Uploaded:
    def __init__(self, name: str, payload: bytes = b"1000,1\n999,2\n") -> None:
        self.name = name
        self._payload = payload

    def getvalue(self) -> bytes:
        return self._payload


def _fake_streamlit_state() -> SimpleNamespace:
    return SimpleNamespace(
        session_state=SimpleNamespace(
            config_payload=PipelineConfig(input_unit="absorbance").to_dict(),
            result=object(),
            gallery=object(),
            selected_candidate=object(),
        )
    )


def test_duplicate_upload_basenames_are_rejected_before_writing() -> None:
    uploads: list[Any] = [_Uploaded("same.dpt"), _Uploaded("SAME.DPT")]
    with pytest.raises(ValueError, match="重复文件名"):
        app._load_uploaded(uploads, "absorbance", True)


def test_invalid_config_edit_is_atomic(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _fake_streamlit_state()
    monkeypatch.setattr(app, "st", fake_st)
    original = deepcopy(fake_st.session_state.config_payload)

    with pytest.raises(ValueError, match="must differ"):
        app._set_config(wavenumber_range=[1000.0, 1000.0])
    assert fake_st.session_state.config_payload == original
    assert fake_st.session_state.result is not None

    with pytest.raises(ValueError, match="must differ"):
        app._set_config_section(
            "fine_baseline",
            {"anchors": [{"start": 1200.0, "end": 1200.0}]},
        )
    assert fake_st.session_state.config_payload == original
    assert fake_st.session_state.gallery is not None


def test_new_dataset_resets_range_unit_and_transmittance_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _fake_streamlit_state()
    fake_st.session_state.config_payload["input_unit"] = "percent_transmittance"
    fake_st.session_state.config_payload["transmittance_floor"] = 1e-6
    fake_st.session_state.config_payload["fine_baseline"].update(
        {
            "method": "pchip",
            "anchors": [
                {"enabled": True, "start": 1790.0, "end": 1800.0, "statistic": "median"},
                {"enabled": True, "start": 900.0, "end": 910.0, "statistic": "median"},
            ],
        }
    )
    fake_st.session_state.config_payload["normalization"].update(
        {
            "method": "internal_peak_height",
            "internal_reference_range": [1650.0, 1550.0],
        }
    )
    monkeypatch.setattr(app, "st", fake_st)
    data = SpectrumSet(
        wavenumber=np.array([1000.0, 950.0, 900.0]),
        perturbation=np.array([0.0]),
        perturbation_labels=("sample",),
        spectra=np.array([[0.1, 0.2, 0.3]]),
        intensity_unit="absorbance",
        source_name="narrow dataset",
        metadata={},
    )

    app._set_dataset(data)

    assert fake_st.session_state.dataset is data
    assert fake_st.session_state.config_payload["input_unit"] == "absorbance"
    assert fake_st.session_state.config_payload["transmittance_floor"] is None
    assert fake_st.session_state.config_payload["wavenumber_range"] == [1000.0, 900.0]
    assert fake_st.session_state.config_payload["fine_baseline"]["method"] == (
        "endpoint_window_linear"
    )
    assert fake_st.session_state.config_payload["fine_baseline"]["anchors"] == []
    assert fake_st.session_state.config_payload["normalization"]["method"] == "none"


def test_narrow_axis_normalization_defaults_are_valid() -> None:
    config = PipelineConfig(input_unit="absorbance", wavenumber_range=(1000.0, 900.0))
    x = np.array([1000.0, 950.0, 900.0])

    high, low = app._normalization_interval_defaults(config, x, "internal_peak_height")

    assert 900.0 <= low < high <= 1000.0
    assert np.count_nonzero((x >= low) & (x <= high)) >= 2


def test_ui_candidate_specs_include_anchor_pchip() -> None:
    anchors = [
        {"enabled": True, "start": 1796.0, "end": 1804.0, "statistic": "median"},
        {"enabled": True, "start": 896.0, "end": 904.0, "statistic": "median"},
    ]
    specs = app._gallery_specs(["Anchor PCHIP"], [6], 0.01, 8.0, anchors)

    assert len(specs) == 1
    assert specs[0].name == "Anchor PCHIP"
    assert specs[0].fine_method == "pchip"


def test_private_local_data_runs_through_all_streamlit_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_data = Path(__file__).resolve().parents[2] / "data" / "original"
    if not any(private_data.glob("*MIN.dpt")):
        pytest.skip("private local data is not available")
    matplotlib_config = tmp_path / "matplotlib"
    matplotlib_config.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(matplotlib_config))
    streamlit_test = AppTest.from_file(_APP_PATH, default_timeout=60)

    streamlit_test.run()
    assert not streamlit_test.exception
    demo_button = next(
        button
        for button in streamlit_test.button
        if button.label == "载入本机 data/original 中的 DPT"
    )
    demo_button.click().run()
    assert not streamlit_test.exception

    for page in app.PAGES[1:]:
        streamlit_test.sidebar.radio[0].set_value(page).run()
        assert not streamlit_test.exception, page
