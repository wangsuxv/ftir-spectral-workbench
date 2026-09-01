from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ftir2dcos.io import read_wide_file
from ftir_baseline.io import load_spectrum_files, read_spectrum_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "import_compat"


@pytest.mark.parametrize(
    "fixture_name",
    ("legacy_wide.csv", "wide.tsv"),
)
def test_headered_wide_table_matches_frozen_twodcos_reader(
    fixture_name: str,
) -> None:
    path = FIXTURES / fixture_name

    workbench = read_spectrum_file(path, input_unit="absorbance")
    frozen = read_wide_file(path, intensity_unit="absorbance")

    np.testing.assert_array_equal(workbench.wavenumber, frozen.wavenumber)
    np.testing.assert_array_equal(workbench.spectra, frozen.spectra)
    np.testing.assert_array_equal(workbench.perturbation, frozen.perturbation)
    assert workbench.perturbation_labels == frozen.perturbation_labels


@pytest.mark.parametrize(
    ("fixture_name", "delimiter_name"),
    (
        ("single.tab", "tab"),
        ("single.asc", "whitespace"),
        ("single.dat", "semicolon"),
        ("single.xy", "comma"),
    ),
)
def test_public_reader_loads_existing_two_column_extension_fixture(
    fixture_name: str,
    delimiter_name: str,
) -> None:
    spectrum = read_spectrum_file(FIXTURES / fixture_name)

    np.testing.assert_array_equal(spectrum.wavenumber, [1800, 1700, 1600, 1500, 1400])
    np.testing.assert_array_equal(
        spectrum.spectra,
        [[0.020, 0.025, 0.040, 0.080, 0.150]],
    )
    assert spectrum.perturbation_labels == ("single",)
    np.testing.assert_array_equal(spectrum.perturbation, [0.0])
    assert spectrum.metadata["delimiter_name"] == delimiter_name
    assert spectrum.metadata["input_layout"] == "two_column"
    assert spectrum.metadata["source_format"] == Path(fixture_name).suffix.removeprefix(".")


def test_public_reader_loads_existing_mixed_extension_two_column_series() -> None:
    paths = [
        FIXTURES / "mixed_series" / "0MIN.dpt",
        FIXTURES / "mixed_series" / "5MIN.tsv",
        FIXTURES / "mixed_series" / "10MIN.asc",
    ]

    series = load_spectrum_files(paths)

    np.testing.assert_array_equal(series.wavenumber, [1002, 1001, 1000])
    np.testing.assert_array_equal(
        series.spectra,
        [
            [0.125, 0.150, 0.175],
            [0.225, 0.250, 0.275],
            [0.325, 0.350, 0.375],
        ],
    )
    assert series.perturbation_labels == ("0MIN", "5MIN", "10MIN")
    np.testing.assert_array_equal(series.perturbation, [0.0, 5.0, 10.0])
    assert series.metadata["final_file_order"] == tuple(path.name for path in paths)
