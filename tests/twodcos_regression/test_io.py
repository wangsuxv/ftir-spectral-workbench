from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ftir2dcos.io import (
    extract_perturbation_value,
    load_input,
    read_dpt_directory,
    read_dpt_files,
    read_wide_file,
)


@pytest.mark.parametrize("delimiter", [",", "\t", ";"])
def test_wide_reader_detects_supported_delimiters_and_transposes(
    tmp_path: Path, delimiter: str
) -> None:
    path = tmp_path / "spectra.txt"
    content = delimiter.join(["Wavenumber", "0", "5 min", "RH_60", "25C"])
    for index, wavenumber in enumerate(range(1800, 1788, -1)):
        content += "\n" + delimiter.join(
            [str(wavenumber), str(index), str(index + 1), str(index + 2), str(index + 3)]
        )
    path.write_text(content, encoding="utf-8-sig")

    dataset = read_wide_file(path, intensity_unit="absorbance")

    assert dataset.shape == (4, 12)
    assert dataset.spectra.dtype == np.float64
    assert dataset.wavenumber.dtype == np.float64
    assert dataset.perturbation_labels == ("0", "5 min", "RH_60", "25C")
    np.testing.assert_array_equal(dataset.perturbation, [0.0, 5.0, 60.0, 25.0])
    assert dataset.metadata["delimiter"] == delimiter
    assert dataset.metadata["original_wavenumber_direction"] == "descending"
    assert not dataset.spectra.flags.writeable


def test_reader_keeps_missing_values_for_explicit_validation(tmp_path: Path) -> None:
    path = tmp_path / "nan.csv"
    rows = ["wn,0,1,2"]
    rows.extend(
        f"{1000 + index},{index},{'NaN' if index == 4 else index + 1},{index + 2}"
        for index in range(10)
    )
    path.write_text("\n".join(rows), encoding="utf-8")

    dataset = read_wide_file(path)

    assert np.isnan(dataset.spectra[1, 4])
    assert path.read_text(encoding="utf-8") == "\n".join(rows)


def test_reader_rejects_text_in_numeric_column_with_location(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("wn,0\n1000,1\n999,not-a-number\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Non-numeric.*data row 2.*column 2"):
        read_wide_file(path)


@pytest.mark.parametrize(
    ("label", "expected"),
    [("0", 0.0), ("5 min", 5.0), ("RH_60", 60.0), ("25C", 25.0), ("none", None)],
)
def test_extract_perturbation_value(label: str, expected: float | None) -> None:
    assert extract_perturbation_value(label) == expected


def _write_dpt(path: Path, axis: np.ndarray, values: np.ndarray) -> None:
    path.write_text(
        "\n".join(f"{x:.3f},{y:.8f}" for x, y in zip(axis, values, strict=True)),
        encoding="utf-8",
    )


def test_dpt_reader_preserves_supplied_file_order_and_never_interpolates(
    tmp_path: Path,
) -> None:
    axis = np.linspace(1800.0, 1700.0, 11)
    first = tmp_path / "10MIN.dpt"
    second = tmp_path / "0MIN.dpt"
    _write_dpt(first, axis, np.arange(axis.size, dtype=float) + 10)
    _write_dpt(second, axis, np.arange(axis.size, dtype=float))

    dataset = read_dpt_files([first, second], intensity_unit="absorbance")

    assert dataset.perturbation_labels == ("10MIN", "0MIN")
    np.testing.assert_array_equal(dataset.perturbation, [10.0, 0.0])
    np.testing.assert_allclose(dataset.spectra[0], np.arange(axis.size) + 10)
    assert dataset.metadata["processing_history"][0]["interpolation_performed"] is False

    mismatched = tmp_path / "20MIN.dpt"
    shifted_axis = axis.copy()
    shifted_axis[4] += 0.01
    _write_dpt(mismatched, shifted_axis, np.arange(axis.size, dtype=float))
    with pytest.raises(ValueError, match="No sorting or interpolation was performed"):
        read_dpt_files([first, mismatched])


def test_dpt_directory_default_excludes_baseline_and_records_enumeration(
    tmp_path: Path,
) -> None:
    axis = np.linspace(1800.0, 1700.0, 11)
    for name, value in [("10MIN.dpt", 10.0), ("0MIN.dpt", 0.0), ("BASELINE.dpt", -1.0)]:
        _write_dpt(tmp_path / name, axis, np.full(axis.size, value))

    dataset = read_dpt_directory(tmp_path)

    assert dataset.perturbation_labels == ("0MIN", "10MIN")
    assert "BASELINE.dpt" not in dataset.metadata["directory_enumeration_order"]
    assert dataset.metadata["directory_ordering_rule"] == "filename_casefold_ascending"
    assert any(
        "not perturbation-value sorting" in item for item in dataset.metadata["parse_warnings"]
    )


def test_load_input_only_sorts_perturbations_when_explicitly_requested(tmp_path: Path) -> None:
    axis = np.linspace(1800.0, 1700.0, 11)
    for name, value in [("10MIN.dpt", 10.0), ("2MIN.dpt", 2.0), ("0MIN.dpt", 0.0)]:
        _write_dpt(tmp_path / name, axis, np.full(axis.size, value))

    preserved = load_input(tmp_path)
    sorted_dataset = load_input(tmp_path, perturbation_order="sort_by_perturbation")

    assert preserved.perturbation_labels == ("0MIN", "10MIN", "2MIN")
    assert sorted_dataset.perturbation_labels == ("0MIN", "2MIN", "10MIN")
    assert sorted_dataset.metadata["perturbation_order_changed"] is True
    assert sorted_dataset.metadata["perturbation_order_indices"] == [0, 2, 1]
