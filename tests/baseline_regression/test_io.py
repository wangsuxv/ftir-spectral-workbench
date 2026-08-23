from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ftir_baseline.io import (
    SpectrumReadError,
    load_spectrum_directory,
    load_spectrum_files,
    read_spectrum_file,
)
from ftir_baseline.validation import SpectrumValidationError


def _write_spectrum(path: Path, y: list[float], x: list[float] | None = None) -> Path:
    axis = x or [1002.0, 1001.0, 1000.0]
    path.write_text(
        "\n".join(
            f"{wavenumber},{intensity}" for wavenumber, intensity in zip(axis, y, strict=True)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_read_headerless_dpt_preserves_descending_axis(tmp_path: Path) -> None:
    path = _write_spectrum(tmp_path / "12MIN.dpt", [0.1, 0.2, 0.3])
    data = read_spectrum_file(path, input_unit="absorbance")

    np.testing.assert_array_equal(data.wavenumber, [1002, 1001, 1000])
    np.testing.assert_array_equal(data.spectra, [[0.1, 0.2, 0.3]])
    np.testing.assert_array_equal(data.perturbation, [12])
    assert data.perturbation_labels == ("12MIN",)
    assert data.axis_direction == "descending"
    assert data.metadata["source_format"] == "dpt"
    assert len(data.metadata["source_sha256"]) == 64


def test_read_wide_csv_header_and_multiple_spectra(tmp_path: Path) -> None:
    path = tmp_path / "wide.csv"
    path.write_text(
        "Wavenumber,0 min,5 min\n1000,1.0,2.0\n1001,1.1,2.1\n1002,1.2,2.2\n",
        encoding="utf-8",
    )
    data = read_spectrum_file(path, input_unit="percent_transmittance")

    assert data.spectra.shape == (2, 3)
    assert data.perturbation_labels == ("0 min", "5 min")
    np.testing.assert_array_equal(data.perturbation, [0, 5])
    assert data.intensity_unit == "percent_transmittance"
    assert data.axis_direction == "ascending"


def test_wide_table_sort_is_explicit_and_recorded(tmp_path: Path) -> None:
    path = tmp_path / "wide.csv"
    path.write_text(
        "Wavenumber,10 min,2 min,5 min\n1000,10,2,5\n1001,11,3,6\n",
        encoding="utf-8",
    )
    unsorted = read_spectrum_file(path)
    sorted_data = read_spectrum_file(path, sort_by_perturbation=True)

    assert unsorted.perturbation_labels == ("10 min", "2 min", "5 min")
    assert sorted_data.perturbation_labels == ("2 min", "5 min", "10 min")
    np.testing.assert_array_equal(sorted_data.spectra[:, 0], [2, 5, 10])
    assert sorted_data.metadata["sorted_by_perturbation"] is True
    assert sorted_data.metadata["original_spectrum_order"] == (
        "10 min",
        "2 min",
        "5 min",
    )


def test_wide_table_sort_rejects_unparseable_labels(tmp_path: Path) -> None:
    path = tmp_path / "wide.csv"
    path.write_text("Wavenumber,early,late\n1000,1,2\n1001,2,3\n", encoding="utf-8")
    with pytest.raises(SpectrumReadError, match="cannot sort wide-table"):
        read_spectrum_file(path, sort_by_perturbation=True)


def test_read_whitespace_txt_and_comments(tmp_path: Path) -> None:
    path = tmp_path / "single.txt"
    path.write_text(
        "# exported spectrum\nWavenumber Intensity\n1000 0.1\n1001 0.2\n1002 0.3\n",
        encoding="utf-8",
    )
    data = read_spectrum_file(path, delimiter=None)
    np.testing.assert_allclose(data.spectra[0], [0.1, 0.2, 0.3])
    assert data.metadata["delimiter"] == "whitespace"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("1000,1\n1000,2\n999,3\n", "duplicate"),
        ("1000,1\n999,2\n1000,3\n", "duplicate"),
        ("1000,1\n999,2\n1001,3\n", "strictly monotonic"),
        ("1000,1\n999,nan\n998,3\n", "non-finite"),
        ("1000,1\n999,inf\n998,3\n", "non-finite"),
    ],
)
def test_reader_rejects_invalid_scientific_data(tmp_path: Path, rows: str, message: str) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(rows, encoding="utf-8")
    with pytest.raises(SpectrumValidationError, match=message):
        read_spectrum_file(path)


def test_reader_rejects_ragged_or_midfile_text(tmp_path: Path) -> None:
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("1000,1\n999,2,3\n", encoding="utf-8")
    with pytest.raises(SpectrumReadError, match="columns"):
        read_spectrum_file(ragged)

    text = tmp_path / "text.csv"
    text.write_text("x,y\n1000,1\noops,2\n", encoding="utf-8")
    with pytest.raises(SpectrumReadError, match="non-numeric"):
        read_spectrum_file(text)


def test_multifile_loader_preserves_order_by_default(tmp_path: Path) -> None:
    paths = [
        _write_spectrum(tmp_path / "10MIN.dpt", [10, 11, 12]),
        _write_spectrum(tmp_path / "2MIN.dpt", [2, 3, 4]),
        _write_spectrum(tmp_path / "5MIN.dpt", [5, 6, 7]),
    ]
    data = load_spectrum_files(paths)

    assert data.perturbation_labels == ("10MIN", "2MIN", "5MIN")
    np.testing.assert_array_equal(data.perturbation, [10, 2, 5])
    np.testing.assert_array_equal(data.spectra[:, 0], [10, 2, 5])
    assert data.metadata["sorted_by_perturbation"] is False
    assert data.metadata["order_policy"] == "input_order"


def test_multifile_sort_is_explicit_stable_and_recorded(tmp_path: Path) -> None:
    paths = [
        _write_spectrum(tmp_path / "10MIN.dpt", [10, 11, 12]),
        _write_spectrum(tmp_path / "2MIN.dpt", [2, 3, 4]),
        _write_spectrum(tmp_path / "5MIN.dpt", [5, 6, 7]),
    ]
    data = load_spectrum_files(paths, sort_by_perturbation=True)

    assert data.perturbation_labels == ("2MIN", "5MIN", "10MIN")
    np.testing.assert_array_equal(data.perturbation, [2, 5, 10])
    assert data.metadata["original_file_order"] == (
        "10MIN.dpt",
        "2MIN.dpt",
        "5MIN.dpt",
    )
    assert data.metadata["final_file_order"] == (
        "2MIN.dpt",
        "5MIN.dpt",
        "10MIN.dpt",
    )
    assert data.metadata["sorted_by_perturbation"] is True


def test_directory_requires_explicit_portable_order(tmp_path: Path) -> None:
    _write_spectrum(tmp_path / "10MIN.dpt", [10, 11, 12])
    _write_spectrum(tmp_path / "2MIN.dpt", [2, 3, 4])

    with pytest.raises(SpectrumReadError, match="no portable acquisition order"):
        load_spectrum_directory(tmp_path)

    data = load_spectrum_directory(tmp_path, sort_by_perturbation=True)
    assert data.perturbation_labels == ("2MIN", "10MIN")
    assert data.metadata["directory_input"] is True
    assert data.metadata["directory_discovery_policy"] == "lexical_filename"
    assert data.metadata["order_policy"] == "numeric_perturbation_stable"


def test_exclusion_is_explicit_and_recorded(tmp_path: Path) -> None:
    time_file = _write_spectrum(tmp_path / "0MIN.dpt", [1, 2, 3])
    baseline = _write_spectrum(tmp_path / "BASELINE.dpt", [0, 0, 0])
    data = load_spectrum_files([time_file, baseline], exclude_names=("BASELINE.dpt",))
    assert data.n_spectra == 1
    assert data.metadata["excluded_files"] == ("BASELINE.dpt",)


def test_multifile_loader_rejects_duplicate_basenames(tmp_path: Path) -> None:
    first_directory = tmp_path / "a"
    second_directory = tmp_path / "b"
    first_directory.mkdir()
    second_directory.mkdir()
    first = _write_spectrum(first_directory / "0MIN.dpt", [1, 2, 3])
    second = _write_spectrum(second_directory / "0min.DPT", [4, 5, 6])

    with pytest.raises(SpectrumReadError, match="unique basenames"):
        load_spectrum_files([first, second])


def test_multifile_loader_requires_identical_axes(tmp_path: Path) -> None:
    first = _write_spectrum(tmp_path / "0MIN.dpt", [1, 2, 3])
    second = _write_spectrum(tmp_path / "1MIN.dpt", [2, 3, 4], x=[1002, 1000.5, 1000])
    with pytest.raises(SpectrumValidationError, match="point-for-point"):
        load_spectrum_files([first, second])


def test_sort_requires_parseable_or_explicit_perturbations(tmp_path: Path) -> None:
    sample = _write_spectrum(tmp_path / "sample.dpt", [1, 2, 3])
    with pytest.raises(SpectrumReadError, match="cannot sort"):
        load_spectrum_files([sample], sort_by_perturbation=True)


def test_private_local_directory_loads_when_available() -> None:
    directory = Path(__file__).resolve().parents[2] / "data" / "original"
    if not any(directory.glob("*MIN.dpt")):
        pytest.skip("private local data is not available")
    data = load_spectrum_directory(directory, sort_by_perturbation=True)

    assert data.n_spectra > 0
    assert data.n_points > 1
    assert data.axis_direction == "descending"
    assert np.all(np.diff(data.perturbation) > 0)
    assert data.metadata["excluded_files"] == ("BASELINE.dpt",)
