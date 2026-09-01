from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from ftir_baseline.io import TextImportOptions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = PROJECT_ROOT / "ui" / "streamlit_app.py"


def _load_app_module():  # type: ignore[no-untyped-def]
    spec = spec_from_file_location("import_compat_streamlit_app_under_test", APP_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("could not load unified Streamlit app")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _element(elements, label: str):  # type: ignore[no-untyped-def]
    return next(element for element in elements if element.label == label)


@dataclass(frozen=True)
class _Upload:
    name: str
    content: bytes

    def getvalue(self) -> bytes:
        return self.content


DECIMAL_COMMA_WIDE = (
    b"wavenumber;0MIN;1MIN\n"
    b"1800,0;0,10;0,20\n"
    b"1700,0;0,11;0,21\n"
    b"1600,0;0,12;0,22\n"
)


def test_import_page_exposes_all_advanced_text_controls() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not app.exception
    assert [expander.label for expander in app.get("expander")] == [
        "Advanced text import options"
    ]
    assert {selectbox.label for selectbox in app.selectbox} >= {
        "Delimiter",
        "Decimal mark",
        "Encoding",
        "Header",
    }
    assert list(_element(app.selectbox, "Delimiter").options) == [
        "Auto",
        "Comma",
        "Tab",
        "Semicolon",
        "Whitespace",
    ]
    assert list(_element(app.selectbox, "Encoding").options) == [
        "Auto",
        "UTF-8",
        "UTF-16",
        "UTF-16 LE",
        "UTF-16 BE",
        "GB18030",
        "CP1252",
    ]
    assert _element(app.number_input, "Skip leading rows").value == 0
    assert _element(app.checkbox, "Trim all-empty edge columns").value is True
    assert any(subheader.value == "Import Diagnosis" for subheader in app.subheader)
    assert _element(app.button, "Analyze uploaded files").disabled is True
    assert _element(app.button, "Load using these settings").disabled is True
    uploader = app.get("file_uploader")[0]
    assert set(uploader.allowed_type) == {
        ".dpt",
        ".csv",
        ".tsv",
        ".tab",
        ".txt",
        ".asc",
        ".dat",
        ".xy",
    }


def test_probe_and_load_receive_the_same_text_import_options() -> None:
    app_module = _load_app_module()
    upload = _Upload("decimal-comma.csv", DECIMAL_COMMA_WIDE)
    options = TextImportOptions(delimiter="semicolon", decimal_mark="comma")

    probes = app_module._uploaded_import_probes(
        [upload],
        import_options=options,
    )
    data = app_module._uploaded_raw(
        [upload],
        unit="absorbance",
        sort_by_perturbation=True,
        import_options=options,
    )

    assert len(probes) == 1
    assert probes[0].selected_delimiter == "semicolon"
    assert probes[0].selected_decimal_mark == "comma"
    assert data.metadata["delimiter_name"] == probes[0].selected_delimiter
    assert data.metadata["decimal_mark"] == probes[0].selected_decimal_mark
    np.testing.assert_array_equal(data.wavenumber, [1800.0, 1700.0, 1600.0])
    np.testing.assert_array_equal(
        data.spectra,
        [[0.10, 0.11, 0.12], [0.20, 0.21, 0.22]],
    )


def test_analyze_reports_diagnosis_without_changing_scientific_state() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    uploader = app.get("file_uploader")[0]
    uploader.upload("decimal-comma.csv", DECIMAL_COMMA_WIDE, "text/plain").run()
    _element(app.selectbox, "Delimiter").set_value("Semicolon").run()
    _element(app.selectbox, "Decimal mark").set_value("Comma").run()
    baseline_config = deepcopy(dict(app.session_state["baseline_config"]))
    app.session_state["baseline_result"] = "existing-result"

    _element(app.button, "Analyze uploaded files").click().run()

    assert not app.exception
    assert not app.error
    assert app.session_state["raw_data"] is None
    assert app.session_state["baseline_config"] == baseline_config
    assert app.session_state["baseline_result"] == "existing-result"
    assert len(app.dataframe) == 1
    diagnosis = app.dataframe[0].value
    assert list(diagnosis.columns) == [
        "File",
        "Extension",
        "Encoding",
        "Delimiter",
        "Decimal",
        "Header",
        "Layout",
        "Rows",
        "Columns",
        "Warnings",
    ]
    assert diagnosis.loc[0, "Extension"] == ".csv"
    assert diagnosis.loc[0, "Delimiter"] == "semicolon"
    assert diagnosis.loc[0, "Decimal"] == "comma"
    assert diagnosis.loc[0, "Layout"] == "wide_table"


def test_changed_import_options_invalidate_uploaded_raw_and_descendants() -> None:
    content = (
        b"wavenumber\t0MIN\n"
        b"1800\t0.10\n"
        b"1700\t0.11\n"
        b"1600\t0.12\n"
    )
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.get("file_uploader")[0].upload("single.tsv", content, "text/tab-separated-values").run()

    _element(app.button, "Load using these settings").click().run()

    assert not app.exception
    assert not app.error
    assert app.session_state["raw_data"] is not None
    baseline_config = deepcopy(dict(app.session_state["baseline_config"]))
    for key in ("baseline_result", "prepared", "twodcos_result", "peak_order_result"):
        app.session_state[key] = f"stale-{key}"

    _element(app.selectbox, "Header").set_value("Present").run()

    assert not app.exception
    assert app.session_state["raw_data"] is None
    assert app.session_state["baseline_config"] == baseline_config
    for key in ("baseline_result", "prepared", "twodcos_result", "peak_order_result"):
        assert app.session_state[key] is None
    assert any("旧的原始数据和下游结果已失效" in info.value for info in app.info)
