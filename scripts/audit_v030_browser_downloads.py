"""Verify actual synthetic browser downloads without fitting or writing arrays.

The directory must contain coarse.zip, final.zip, workspace.zip,
restored-final.zip and restored-workspace.zip saved from real UI downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import zipfile
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from pydantic import BaseModel

from ftir_workbench.batch.export import verify_batch_export
from ftir_workbench.batch.state import get_ready_snapshot
from ftir_workbench.batch.workspace import load_batch_workspace


def _exact(a: Any, b: Any, path: str = "workspace") -> None:
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    elif is_dataclass(a):
        assert type(a) is type(b), path
        for field in fields(a):
            if field.name not in {"export_history", "last_export_summary"}:
                _exact(getattr(a, field.name), getattr(b, field.name), path + "." + field.name)
    elif isinstance(a, BaseModel):
        _exact(a.model_dump(), b.model_dump(), path)
    elif isinstance(a, Mapping):
        assert a.keys() == b.keys(), path
        for key in a:
            _exact(a[key], b[key], path + "." + str(key))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            _exact(left, right, f"{path}[{index}]")
    elif isinstance(a, float) and math.isnan(a):
        assert math.isnan(b), path
    else:
        assert a == b, path


def audit(directory: Path) -> dict[str, Any]:
    names = ("coarse.zip", "final.zip", "workspace.zip", "restored-final.zip", "restored-workspace.zip")
    payloads = {name: (directory / name).read_bytes() for name in names}
    with patch("ftir_baseline.pipeline.run_pipeline", side_effect=AssertionError("restore must not fit")), patch(
        "ftir_workbench.batch.service.run_pipeline", side_effect=AssertionError("restore must not fit")
    ):
        before = load_batch_workspace(payloads["workspace.zip"])
        after = load_batch_workspace(payloads["restored-workspace.zip"])
        _exact(before, after)
        assert len(after.records) == 5 and len(after.sources) == 3
        assert sorted(state.fine_decision for state in after.states.values()) == ["applied"] * 4 + ["explicitly_skipped"]
        assert {r.confirmed_input_unit for r in after.records.values()} == {
            "absorbance", "percent_transmittance", "fraction_transmittance"
        }
        assert sorted(r.wavenumber.size for r in after.records.values()) == [291, 401, 401, 401, 581]
        checked = []
        for name, stage in (("coarse.zip", "coarse"), ("final.zip", "fine"), ("restored-final.zip", "fine")):
            assert verify_batch_export(payloads[name])
            with zipfile.ZipFile(io.BytesIO(payloads[name])) as archive:
                metadata = json.loads(archive.read("batch_metadata.json"))
                assert metadata["artifact_type"] == "independent_baseline_batch"
                assert metadata["is_2d_ready"] is False and metadata["wide_available"] is False
                assert not any("for_2dcos" in member or "prepared" in member.lower() or "wide.csv" in member for member in archive.namelist())
                assert metadata["exported_spectrum_ids"] == after.display_order
                for entry in metadata["spectra"]:
                    sid = entry["spectrum_id"]
                    record = after.records[sid]
                    snapshot = get_ready_snapshot(after, sid, stage)
                    assert archive.read(entry["spectra_csv"]).splitlines()[0] == b"wavenumber_cm-1,corrected_absorbance"
                    values = np.loadtxt(io.BytesIO(archive.read(entry["spectra_csv"])), delimiter=",", skiprows=1)
                    np.testing.assert_array_equal(values[:, 0], snapshot.result.absorbance_selected.wavenumber)
                    np.testing.assert_array_equal(values[:, 1], snapshot.result.analysis_data[0])
                    assert entry["stage_fingerprint"] == snapshot.fingerprint
                    assert entry["scientific_input_sha256"] == record.scientific_input_sha256
                    assert entry["source_id"] == record.source_id
                    assert entry["source_sha256"] == after.sources[record.source_id].original_bytes_sha256
                    assert entry["source_column_index"] == record.original_column_index
                    assert entry["source_column_label"] == record.original_column_label
                    assert entry["input_unit"] == record.confirmed_input_unit
                checked.append({"download": name, "confirmed_arrays_exact": 5, "stage": stage})
        with zipfile.ZipFile(io.BytesIO(payloads["final.zip"])) as left, zipfile.ZipFile(io.BytesIO(payloads["restored-final.zip"])) as right:
            assert left.namelist() == right.namelist()
            changed_members = []
            for name in left.namelist():
                a, b = left.read(name), right.read(name)
                if a != b:
                    changed_members.append(name)
                    # JSON mapping key order changes QC row order on restore.
                    # Every metric/value row must still match exactly; each
                    # manifest was independently verified above.
                    if name == "qc_summary.csv":
                        assert sorted(a.splitlines()) == sorted(b.splitlines())
                    else:
                        assert name == "manifest.json", name
    return {
        "status": "pass", "source": "actual Playwright upload/download UI session; synthetic files only",
        "stable_ids_sources_drafts_formal_arrays_parents_exact": True,
        "ignored_comparison_fields": ["export_history", "last_export_summary (new export event)"],
        "spectra": 5, "sources": 3, "fine_applied": 4, "fine_explicitly_skipped": 1,
        "lengths": [291, 401, 401, 401, 581], "wide_export_disabled": True,
        "final_zip_before_after_restore_byte_identical": payloads["final.zip"] == payloads["restored-final.zip"],
        "differing_zip_members": changed_members,
        "qc_metric_value_rows_exact_ignoring_row_order": True,
        "all_spectra_components_recipes_metadata_reports_byte_identical": True,
        "pipeline_calls_during_download_audit": 0, "exports": checked,
        "downloads": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in payloads.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(audit(args.directory), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
