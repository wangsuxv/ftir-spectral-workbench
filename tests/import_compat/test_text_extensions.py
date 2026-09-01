from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from ftir_baseline.io import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_TEXT_EXTENSIONS,
    load_spectrum_files,
    read_spectrum_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "import_compat"
APP_PATH = PROJECT_ROOT / "ui" / "streamlit_app.py"


def test_supported_text_extensions_are_the_public_backward_compatible_set() -> None:
    expected = frozenset(
        {".csv", ".tsv", ".tab", ".txt", ".dpt", ".asc", ".dat", ".xy"}
    )

    assert expected == SUPPORTED_TEXT_EXTENSIONS
    assert SUPPORTED_EXTENSIONS is SUPPORTED_TEXT_EXTENSIONS


@pytest.mark.parametrize(
    "fixture_name",
    ("single.tsv", "single.tab", "single.asc", "single.dat", "single.xy"),
)
def test_new_single_spectrum_extensions_use_existing_text_parser(
    fixture_name: str,
) -> None:
    data = read_spectrum_file(FIXTURES / fixture_name)

    np.testing.assert_array_equal(data.wavenumber, [1800, 1700, 1600, 1500, 1400])
    np.testing.assert_array_equal(data.spectra, [[0.020, 0.025, 0.040, 0.080, 0.150]])
    assert data.metadata["source_format"] == Path(fixture_name).suffix[1:]


def test_wide_tsv_is_read_as_a_three_spectrum_table() -> None:
    data = read_spectrum_file(FIXTURES / "wide.tsv")

    assert data.spectra.shape == (3, 5)
    assert data.perturbation_labels == ("0MIN", "5MIN", "10MIN")
    np.testing.assert_array_equal(data.perturbation, [0, 5, 10])
    assert data.metadata["delimiter"] == repr("\t")


def test_mixed_extension_multifile_series_preserves_explicit_order() -> None:
    paths = [
        FIXTURES / "single.tsv",
        FIXTURES / "single.asc",
        FIXTURES / "single.xy",
    ]
    data = load_spectrum_files(
        paths,
        perturbations=(0.0, 5.0, 10.0),
        perturbation_labels=("0MIN", "5MIN", "10MIN"),
    )

    assert data.perturbation_labels == ("0MIN", "5MIN", "10MIN")
    np.testing.assert_array_equal(data.perturbation, [0, 5, 10])
    assert data.metadata["final_file_order"] == tuple(path.name for path in paths)


def test_import_uploader_accepts_all_supported_text_extensions() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not app.exception
    assert app.file_uploader[0].allowed_type == [
        ".dpt",
        ".csv",
        ".tsv",
        ".tab",
        ".txt",
        ".asc",
        ".dat",
        ".xy",
    ]
