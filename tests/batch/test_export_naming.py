"""Readable names are presentation only; scientific snapshots remain authoritative."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
from dataclasses import replace
from pathlib import PurePosixPath

import numpy as np
import pytest

import ftir_workbench.batch.export_naming as naming
import ftir_workbench.batch.service as service
from ftir_baseline.io import TextImportOptions
from ftir_workbench.batch.export import (
    _table,
    build_batch_export,
    preview_batch_export,
    verify_batch_export,
)
from ftir_workbench.batch.export_naming import (
    allocate_bases,
    get_naming_options,
    make_name_row,
    name_map_table,
    processing_suffix,
    set_name_override,
    set_naming_policy,
)
from ftir_workbench.batch.importing import import_sources
from ftir_workbench.batch.models import BatchError, BatchWorkspace
from ftir_workbench.batch.service import preview_coarse
from ftir_workbench.batch.state import confirm_coarse, skip_fine
from ftir_workbench.batch.workspace import (
    _read_json,
    json_bytes,
    make_archive,
    read_verified_archive,
)

from .test_importing import table_bytes


def imported(names: list[str], *, labels: tuple[str, ...] = ("Absorbance",)) -> tuple[BatchWorkspace, list[str]]:
    ws = BatchWorkspace()
    x = np.linspace(1800, 900, 31)
    y = 0.1 + np.exp(-((x - 1300) / 100) ** 2)
    payload = table_bytes(x, *(y + index for index in range(len(labels))), labels=labels)
    ids = import_sources(ws, [(name, payload) for name in names])
    assert len(ids) == len(names) * len(labels)
    return ws, ids


def finalized() -> tuple[BatchWorkspace, list[str]]:
    ws, ids = imported(["G2_90RH.csv", "样品 α (25°C 80%RH).csv"])
    for sid in ids:
        preview_coarse(ws, sid)
        confirm_coarse(ws, sid)
        skip_fine(ws, sid)
    return ws, ids


def test_n01_n02_original_filename_and_wide_column_selection() -> None:
    ws, ids = imported([r"C:\lab\G2_90RH.csv", "/lab/中文 (25°C 90%RH).csv"])
    assert [allocate_bases(ws)[sid]["base_name"] for sid in ids] == ["G2_90RH", "中文 (25°C 90%RH)"]
    ws, ids = imported(["wide.csv"], labels=("G1", "G2"))
    preview_coarse(ws, ids[1])
    confirm_coarse(ws, ids[1])
    artifact = build_batch_export(ws, [ids[1]], stage="coarse")
    row = artifact.metadata["name_map"][0]
    assert row["base_name"] == "G2"
    assert row["source_layout"] == "wide_table" and row["source_column_index"] == 2
    assert row["spectra_csv"] == "spectra/G2__baseline-coarse.csv"


def test_n03_headerless_and_old_probe_fallback_uses_original_indices() -> None:
    ws = BatchWorkspace()
    ids = import_sources(ws, [("batch01.csv", b"1000,1,2\n1100,2,4\n1200,3,5\n")], options=TextImportOptions(delimiter="comma", decimal_mark="dot"))
    assert [allocate_bases(ws)[sid]["base_name"] for sid in ids] == ["batch01__col01", "batch01__col02"]
    ws, ids = imported(["wide.csv"], labels=("A", "B"))
    source_id = ws.records[ids[0]].source_id
    ws.sources[source_id] = replace(ws.sources[source_id], import_probe={})
    assert allocate_bases(ws)[ids[1]]["base_name"] == "B"
    del ws.records[ids[0]]
    del ws.states[ids[0]]
    assert allocate_bases(ws)[ids[1]]["source_layout"] == "wide_table"


def test_n04_display_override_and_explicit_empty_fallback_do_not_rewrite_records() -> None:
    ws, ids = imported(["G2.csv"])
    sid = ids[0]
    record, source = ws.records[sid], ws.sources[ws.records[sid].source_id]
    set_naming_policy(ws, "display")
    assert allocate_bases(ws)[sid]["base_name"] == "Absorbance"
    set_name_override(ws, sid, "我的 α 样品")
    assert allocate_bases(ws)[sid]["base_name"] == "我的 α 样品"
    set_name_override(ws, sid, "")
    decision = allocate_bases(ws)[sid]
    assert decision["base_name"] == "spectrum__col01" and decision["notes"]
    set_name_override(ws, sid, None)
    assert allocate_bases(ws)[sid]["base_name"] == "Absorbance"
    assert ws.records[sid] is record and ws.sources[source.source_id] is source


@pytest.mark.parametrize(("name", "expected"), [
    ("CON.csv", "_CON"), ("nul.txt.csv", "_nul.txt"), ("LPT9.csv", "_LPT9"),
    ("A<>:*?|.csv", "A______"), ("你好 α (25°C 90%RH).csv", "你好 α (25°C 90%RH)"),
    ("e\u0301.csv", "é"), ("ab\x00\u200bcd.csv", "abcd"), ("  .csv", "spectrum__col01"),
    ("foo... .csv", "foo"), ("baseline_normalized.csv", "baseline_normalized"),
])
def test_n05_safe_unicode_names(name: str, expected: str) -> None:
    ws, ids = imported([name])
    assert allocate_bases(ws)[ids[0]]["base_name"] == expected


@pytest.mark.parametrize("names", [
    ["same.csv", "same.csv"], ["a.csv", "A.csv"], ["é.csv", "e\u0301.csv"],
    ["a?.csv", "a*.csv"], ["长" * 100 + "a.csv", "长" * 100 + "b.csv"],
])
def test_n06_collision_tokens_and_subset_order_stability(names: list[str]) -> None:
    ws, ids = imported(names)
    before = allocate_bases(ws)
    assert len({item["base_name"].casefold() for item in before.values()}) == 2
    assert all(len(item["collision_token"]) == 8 for item in before.values())
    ws.display_order.reverse()
    ws.records = dict(reversed(list(ws.records.items())))
    assert allocate_bases(ws) == before
    for sid in ids:
        preview_coarse(ws, sid)
        confirm_coarse(ws, sid)
    both = build_batch_export(ws, ids, stage="coarse").metadata["name_map"]
    for sid in reversed(ids):
        one = build_batch_export(ws, [sid], stage="coarse").metadata["name_map"]
        assert one == [row for row in both if row["spectrum_id"] == sid]


def test_n07_cross_file_wide_labels_use_readable_file_qualifier() -> None:
    ws, ids = imported(["first.csv", "second.csv"], labels=("A", "B"))
    decisions = allocate_bases(ws)
    assert [decisions[sid]["base_name"] for sid in ids] == ["first__A", "first__B", "second__A", "second__B"]
    assert all(not decision["collision_token"] for decision in decisions.values())


def test_n08_short_token_collision_extends_or_fails_without_overwrite(monkeypatch: pytest.MonkeyPatch) -> None:
    ws, _ = imported(["same.csv", "same.csv"])
    monkeypatch.setattr(naming, "_identity_token", lambda sid, source: "f" * 8 + hashlib.sha256(sid.encode()).hexdigest()[8:])
    assert all(len(row["collision_token"]) == 12 for row in allocate_bases(ws).values())
    monkeypatch.setattr(naming, "_identity_token", lambda sid, source: "f" * 64)
    with pytest.raises(BatchError, match="full identity token collision"):
        allocate_bases(ws)


def test_n09_longest_suffix_budget_includes_tokens_and_extension() -> None:
    ws, ids = imported(["长" * 100 + ".csv", "长" * 100 + ".csv"])
    for sid in ids:
        row = make_name_row(allocate_bases(ws)[sid], branch="normalized_smoothed", quantity="minmax_scaled_intensity",
                            purpose="display_only", fingerprint="f", parent_fingerprint="p", recipe_json=f"recipes/{sid}.json",
                            included_as="requested", effective_recipe={"method": "minmax_display"},
                            parent_effective_recipe={"method": "moving_average"})
        assert len(PurePosixPath(row["spectra_csv"]).name.encode()) <= 200
        assert "�" not in row["spectra_csv"]


@pytest.mark.parametrize(("method", "suffix"), [("maximum", "norm-max"), ("internal_peak_height", "norm-peak-window"),
    ("internal_peak_area", "norm-area-window"), ("area", "norm-area"), ("vector", "norm-l2"),
    ("minmax_display", "minmax_scaled_intensity_display_only")])
def test_n10_committed_suffix_mapping(method: str, suffix: str) -> None:
    assert processing_suffix("normalized_baseline", {"method": method}) == "__baseline__" + suffix
    assert processing_suffix("normalized_smoothed", {"method": method}, {"method": "savgol"}) == "__baseline__smooth-savgol__" + suffix
    with pytest.raises(BatchError):
        processing_suffix("normalized_baseline", {"method": "invented"})


def test_n11_n12_n17_preview_and_export_only_serialize_confirmed_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    ws, ids = finalized()
    snapshots = {sid: ws.states[sid].fine_snapshot for sid in ids}
    source_hashes = {sid: ws.records[sid].scientific_input_sha256 for sid in ids}
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("naming/export must never fit")
    monkeypatch.setattr(service, "run_pipeline", forbidden)
    for policy in ("original", "display", "legacy_id"):
        set_naming_policy(ws, policy)
        plan = preview_batch_export(ws, ids, stage="fine", derived_units=("percent_transmittance",))
        artifact = build_batch_export(ws, ids, stage="fine", include_wide=True, derived_units=("percent_transmittance",))
        assert verify_batch_export(artifact.zip_bytes)
        members = read_verified_archive(artifact.zip_bytes, "independent_baseline_batch")
        for sid, row in zip(ids, plan["name_map"], strict=True):
            assert row["spectra_csv"] in members
            values = np.loadtxt(io.BytesIO(members[row["spectra_csv"]]), delimiter=",", skiprows=1)
            np.testing.assert_array_equal(values[:, 1], snapshots[sid].result.analysis_data[0])
            assert ws.states[sid].fine_snapshot is snapshots[sid]
            assert source_hashes[sid] == ws.records[sid].scientific_input_sha256
        if policy == "legacy_id":
            assert artifact.metadata["schema_version"] == "1.0"
            assert "export_name_map.csv" not in members
            assert all(sid in artifact.wide_csv_bytes.decode().splitlines()[0] for sid in ids)
        else:
            assert artifact.metadata["name_map"] == plan["name_map"]
            assert artifact.metadata["schema_version"] == "1.1"
            assert "__baseline.csv" in plan["name_map"][0]["spectra_csv"]


def test_n13_formula_csv_escaping_preserves_authoritative_json() -> None:
    ws, ids = finalized()
    set_name_override(ws, ids[0], "=SUM(A1:A2)")
    artifact = build_batch_export(ws, [ids[0]], stage="fine", include_wide=True)
    assert verify_batch_export(artifact.zip_bytes)
    members = read_verified_archive(artifact.zip_bytes, "independent_baseline_batch")
    rows = list(csv.DictReader(io.StringIO(members["export_name_map.csv"].decode())))
    assert rows[0]["override"] == "'=SUM(A1:A2)"
    assert artifact.metadata["name_map"][0]["override"] == "=SUM(A1:A2)"


@pytest.mark.parametrize("tamper", ["base", "identity", "token", "path", "index", "extra", "wide", "schema", "csv", "role"])
def test_n14_new_baseline_semantic_verifier_rejects_rehashed_tampering(tamper: str) -> None:
    ws, ids = finalized()
    artifact = build_batch_export(ws, ids, stage="fine", include_wide=True)
    members = read_verified_archive(artifact.zip_bytes, "independent_baseline_batch")
    metadata = _read_json(members["batch_metadata.json"])
    row = metadata["name_map"][0]
    if tamper == "base":
        row["base_name"] = "fraud"
    elif tamper == "identity":
        row["spectrum_id"] = ids[1]
    elif tamper == "token":
        row["collision_token"] = "f" * 8
    elif tamper == "path":
        row["spectra_csv"] = "spectra/other.csv"
    elif tamper == "index":
        metadata["spectra"][0]["source_id"] = "other"
    elif tamper == "extra":
        members["unused.csv"] = b"x,y\n1,2\n"
    elif tamper == "wide":
        metadata["wide_headers"][0] = "wrong"
    elif tamper == "schema":
        metadata["schema_version"] = "2.0"
    elif tamper == "role":
        row["included_as"] = "parent"
    members["batch_metadata.json"] = json_bytes(metadata)
    members["export_name_map.csv"] = b"wrong\n" if tamper == "csv" else _table(*name_map_table(metadata["name_map"]))
    assert not verify_batch_export(make_archive(members, "independent_baseline_batch"))


def test_n15_empty_preview_reports_missing_without_fallback() -> None:
    ws, ids = imported(["A.csv"])
    preview = preview_batch_export(ws, ids, stage="fine")
    assert preview["name_map"] == [] and not preview["exportable"]
    assert preview["report"][0]["status"] == "excluded"
    assert preview["report"][0]["reason"]


def test_n16_name_preferences_validate_before_partial_mutation() -> None:
    ws, ids = imported(["A.csv", "B.csv"])
    before = copy.deepcopy([ws.states[sid].display_preferences for sid in ids])
    with pytest.raises(BatchError):
        set_naming_policy(ws, "invented")
    assert before == [ws.states[sid].display_preferences for sid in ids]
    with pytest.raises(BatchError):
        get_naming_options(ws, {"overrides": {"missing": "foo"}})


@pytest.mark.parametrize(("field", "replacement"), [
    ("spectrum_id", "unrelated-spectrum"), ("source_id", "unrelated-source"),
    ("source_column_index", 99), ("display_name", "incorrect report label"),
    ("stage_fingerprint", "0" * 64), ("fine_decision", "not_decided"),
    ("stage", "coarse"), ("status", "excluded"),
])
def test_n18_rehashed_processing_report_must_match_confirmed_export_identity(
    field: str, replacement: object,
) -> None:
    ws, ids = finalized()
    artifact = build_batch_export(ws, ids, stage="fine")
    assert verify_batch_export(artifact.zip_bytes)
    members = read_verified_archive(artifact.zip_bytes, "independent_baseline_batch")
    metadata = _read_json(members["batch_metadata.json"])
    metadata["processing_report"][0][field] = replacement
    report = metadata["processing_report"]
    headers = sorted({key for item in report for key in item})
    # Both human-readable CSV and authoritative JSON agree with each other;
    # rejection must come from their mismatch with the exported spectrum.
    members["processing_report.csv"] = _table(
        headers, [[item.get(key, "") for key in headers] for item in report],
    )
    members["batch_metadata.json"] = json_bytes(metadata)
    assert not verify_batch_export(make_archive(members, "independent_baseline_batch"))
