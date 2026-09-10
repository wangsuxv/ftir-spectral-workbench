from __future__ import annotations

import csv
import io

import numpy as np
import pytest

from ftir_workbench.batch.importing import import_sources
from ftir_workbench.batch.models import BatchWorkspace


def table_bytes(x: np.ndarray, *ys: np.ndarray, labels: tuple[str, ...] = ()) -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["wavenumber", *(labels or tuple(f"sample_{i}" for i in range(len(ys))))])
    writer.writerows(zip(x, *ys, strict=True))
    return stream.getvalue().encode("utf-8")


def test_i01_two_column_preserves_values_direction_and_full_domain() -> None:
    ws = BatchWorkspace()
    x = np.linspace(4000, 400, 31)
    y = np.linspace(0.1, 0.5, 31)
    ids = import_sources(ws, [("sample.csv", table_bytes(x, y))])
    assert len(ids) == 1
    record, state = ws.records[ids[0]], ws.states[ids[0]]
    np.testing.assert_array_equal(record.wavenumber, x)
    np.testing.assert_array_equal(record.raw_intensity, y)
    assert record.original_axis_direction == "descending"
    assert state.preparation_draft.wavenumber_range == (4000.0, 400.0)
    assert not state.preparation_draft.baseline_smoothing.enabled
    assert state.coarse_snapshot is None and state.fine_snapshot is None
    with pytest.raises(ValueError):
        record.raw_intensity.setflags(write=True)


def test_i02_i05_wide_columns_have_separate_identity_state_and_hashes() -> None:
    ws = BatchWorkspace()
    x = np.arange(10.0)
    ids = import_sources(ws, [("wide.csv", table_bytes(x, x, x + 1, x + 2,
                                                     labels=("中文", "12MIN", "中文")))])
    assert len(ids) == len(set(ids)) == 3
    assert [ws.records[sid].original_column_label for sid in ids] == ["中文", "12MIN", "中文"]
    assert [ws.records[sid].original_column_index for sid in ids] == [1, 2, 3]
    assert len({ws.records[sid].scientific_input_sha256 for sid in ids}) == 3
    assert len({id(ws.states[sid]) for sid in ids}) == 3
    assert len(ws.sources) == 1


def test_i03_i04_i06_mixed_files_axes_and_duplicate_names_do_not_merge() -> None:
    ws = BatchWorkspace()
    x = np.arange(10.0)
    other = np.linspace(4000, 500, 17)
    ids = import_sources(ws, [("same.csv", table_bytes(x, x)),
                              ("same.csv", table_bytes(other, other, other + 1))])
    assert len(ids) == 3
    assert len(ws.sources) == 2
    assert [ws.records[sid].wavenumber.size for sid in ids] == [10, 17, 17]
    assert [ws.records[sid].original_axis_direction for sid in ids] == [
        "ascending", "descending", "descending"]
    assert len({ws.records[sid].source_id for sid in ids}) == 2


def test_i07_bad_file_isolated_and_duplicate_upload_marked_without_deletion() -> None:
    ws = BatchWorkspace()
    data = table_bytes(np.arange(5.0), np.arange(5.0))
    ids = import_sources(ws, [("good.csv", data), ("bad.csv", b"x,y\n1,NaN\n2,3\n"),
                              ("good.csv", data)])
    assert len(ids) == 2
    assert not ws.records[ids[0]].duplicate_candidate
    assert ws.records[ids[1]].duplicate_candidate
    assert ws.import_issues[0]["original_filename"] == "bad.csv"
    assert ws.import_issues[0]["code"] == "IMPORT_FAILED"
    assert "line" in ws.import_issues[0]["message"]


@pytest.mark.parametrize("unit", ["absorbance", "percent_transmittance", "fraction_transmittance"])
def test_i08_explicit_units_are_preserved_without_conversion(unit: str) -> None:
    ws = BatchWorkspace()
    y = np.linspace(0.1, 0.9, 5)
    ids = import_sources(ws, [("spectrum.csv", table_bytes(np.arange(5.0), y))], input_unit=unit)
    assert ws.records[ids[0]].confirmed_input_unit == unit
    np.testing.assert_array_equal(ws.records[ids[0]].raw_intensity, y)
    assert next(iter(ws.sources.values())).default_input_unit == unit


@pytest.mark.parametrize("data", [
    b"x,y\n1,2\n1,3\n2,4\n", b"x,y\n1,2\n3,3\n2,4\n",
    b"x,y\n1,2\n2,inf\n3,4\n", b"x,y\n1,2\n2,\n3,4\n",
])
def test_i09_strict_invalid_axis_or_data_rejected(data: bytes) -> None:
    ws = BatchWorkspace()
    assert import_sources(ws, [("invalid.csv", data)]) == []
    assert len(ws.import_issues) == 1
    assert not ws.records


def test_upload_filename_cannot_escape_temp_directory() -> None:
    ws = BatchWorkspace()
    ids = import_sources(ws, [("../../same.csv", table_bytes(np.arange(5.0), np.arange(5.0)))])
    assert len(ids) == 1
    assert next(iter(ws.sources.values())).original_filename == "../../same.csv"
    assert "source_path" not in next(iter(ws.sources.values())).import_probe
