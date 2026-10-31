"""Actual v0.3.1 producers and verifiers exercise both naming compatibility paths."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from ftir_workbench.batch.export import build_batch_export, verify_batch_export
from ftir_workbench.batch.export_naming import get_naming_options
from ftir_workbench.batch.postprocessing import BRANCHES, branch_arrays, get_branch
from ftir_workbench.batch.postprocessing_export import (
    ARTIFACT_TYPE,
    build_postprocessing_export,
    verify_postprocessing_export,
)
from ftir_workbench.batch.workspace import (
    load_batch_workspace,
    make_archive,
    read_verified_archive,
    save_batch_workspace,
)
from tests.postprocessing.test_workspace_migration import forbid_calculations
from tests.postprocessing.test_workspace_migration import legacy_directory as legacy_directory

ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "14a514d7032f5f7c302f4f47d79a72d895251c1d"
BASELINE_FILES = tuple(f"legacy_baseline_{stage}_v031.zip"
                       for stage in ("coarse", "final", "final_wide"))
POST_FILES = tuple(f"legacy_post_{branch}_v031.zip" for branch in (*BRANCHES, "all", "all_wide"))


@pytest.fixture(scope="module")
def old_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = ROOT / "outputs/validation-private/v032/legacy-v031"
    if not (path / "legacy_generation.json").exists():
        path = tmp_path_factory.mktemp("exact-v031-naming")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/generate_v031_naming_fixtures.py"),
             "--base-commit", BASE_COMMIT, "--output", str(path)],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
    evidence = json.loads((path / "legacy_generation.json").read_text())
    assert evidence["base_commit"] == BASE_COMMIT
    assert evidence["workbench_version"] == "0.3.1"
    assert evidence["dependencies"]["ftir-spectral-workbench"] == "0.3.1"
    assert evidence["experimental_data_read"] is False
    assert evidence["source_environment"] == "exact temporary Git archive"
    for name, entry in evidence["artifacts"].items():
        payload = (path / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        assert len(payload) == entry["size_bytes"]
    return path


@pytest.fixture(scope="module")
def old_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("v031-verifier").resolve()
    payload = subprocess.check_output(["git", "archive", "--format=tar", BASE_COMMIT], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        assert all(not member.issym() and not member.islnk()
                   and (path / member.name).resolve().is_relative_to(path)
                   for member in archive.getmembers())
        archive.extractall(path, filter="data")
    return path


@pytest.mark.parametrize("filename", BASELINE_FILES + POST_FILES)
def test_current_verifier_accepts_actual_v031_legacy_bundle(
    old_directory: Path, filename: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_calculations(monkeypatch)
    payload = (old_directory / filename).read_bytes()
    if filename in BASELINE_FILES:
        assert verify_batch_export(payload)
    else:
        assert verify_postprocessing_export(payload, recompute=False)


def test_actual_v031_workspace_preserves_every_array_and_defaults_naming(
    old_directory: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_calculations(monkeypatch)
    evidence = json.loads((old_directory / "legacy_generation.json").read_text())
    original = load_batch_workspace((old_directory / "legacy_workspace_v031.zip").read_bytes())
    restored = load_batch_workspace(save_batch_workspace(original))
    assert get_naming_options(restored) == {"policy": "original", "overrides": {}}
    assert restored.workspace_id == evidence["workspace_id"]
    assert restored.display_order == evidence["spectrum_ids"]
    count = 0
    with np.load(old_directory / "expected_arrays.npz", allow_pickle=False) as expected:
        for sid in restored.display_order:
            record, state = restored.records[sid], restored.states[sid]
            for suffix, actual in (("raw_x", record.wavenumber), ("raw_y", record.raw_intensity)):
                np.testing.assert_array_equal(actual, expected[f"{sid}__{suffix}"])
                count += 1
            assert record.scientific_input_sha256 == evidence["states"][sid]["scientific_input_sha256"]
            assert record.source_id == evidence["states"][sid]["source_id"]
            for stage in ("coarse", "fine"):
                snapshot = getattr(state, stage + "_snapshot")
                for field, wanted in evidence["states"][sid]["snapshots"][stage].items():
                    assert getattr(snapshot, field) == wanted
                values = {"x": snapshot.result.absorbance_selected.wavenumber,
                          "estimate": snapshot.result.baseline_estimation_spectra}
                values.update({name: getattr(snapshot.result.baseline, name) for name in (
                    "coarse_baseline", "fine_baseline", "total_baseline", "corrected",
                )})
                for suffix, actual in values.items():
                    np.testing.assert_array_equal(actual, expected[f"{sid}__{stage}__{suffix}"])
                    count += 1
            for branch in BRANCHES:
                snapshot = get_branch(restored, sid, branch)
                assert snapshot.fingerprint == evidence["states"][sid]["branches"][branch]["fingerprint"]
                x, y = branch_arrays(snapshot)
                np.testing.assert_array_equal(x, expected[f"{sid}__{branch}__x"])
                np.testing.assert_array_equal(y, expected[f"{sid}__{branch}__y"])
                count += 2
    assert count == evidence["expected_array_count"] == 176


def test_actual_v030_schema1_workspace_defaults_naming(legacy_directory: Path) -> None:
    payload = (legacy_directory / "legacy_workspace_v030.zip").read_bytes()
    old = read_verified_archive(payload, "independent_baseline_workspace")
    assert json.loads(old["workspace.json"])["schema_version"] == "1.0"
    workspace = load_batch_workspace(payload)
    assert get_naming_options(workspace) == {"policy": "original", "overrides": {}}
    restored = load_batch_workspace(save_batch_workspace(workspace))
    assert restored.workspace_id == workspace.workspace_id
    assert get_naming_options(restored) == {"policy": "original", "overrides": {}}


@pytest.mark.parametrize("filename", BASELINE_FILES + POST_FILES)
def test_new_legacy_id_member_paths_arrays_and_old_verifier_remain_compatible(
    old_directory: Path, old_source: Path, tmp_path: Path,
    filename: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbid_calculations(monkeypatch)
    ws = load_batch_workspace((old_directory / "legacy_workspace_v031.zip").read_bytes())
    evidence = json.loads((old_directory / "legacy_generation.json").read_text())
    wide = "wide" in filename
    ids = evidence["same_axis_ids"] if wide else ws.display_order
    if filename in BASELINE_FILES:
        artifact = build_batch_export(
            ws, ids, stage="coarse" if "coarse" in filename else "fine", include_wide=wide,
            derived_units=("fraction_transmittance", "percent_transmittance"), naming="legacy_id",
        )
        artifact_type = "independent_baseline_batch"
    else:
        label = filename.removeprefix("legacy_post_").removesuffix("_v031.zip")
        artifact = build_postprocessing_export(
            ws, ids, branches=BRANCHES if label in {"all", "all_wide"} else (label,),
            include_wide=wide, naming="legacy_id",
        )
        artifact_type = ARTIFACT_TYPE
    actual = read_verified_archive(artifact.zip_bytes, artifact_type)
    old = read_verified_archive((old_directory / filename).read_bytes(), artifact_type)
    assert set(actual) == set(old)
    assert not any("export_name_map" in member for member in actual)
    assert json.loads(actual["batch_metadata.json"])["schema_version"] == "1.0"
    for name, payload in actual.items():
        if name.endswith(".csv"):
            # JSON restoration may sort QC metric rows; the actual numerical
            # table and every label remain identical to the old producer.
            assert sorted(csv.reader(io.StringIO(payload.decode()))) == sorted(
                csv.reader(io.StringIO(old[name].decode()))
            ), name
    result_path = tmp_path / "new_legacy.zip"
    result_path.write_bytes(artifact.zip_bytes)
    program = """
import json, pathlib, sys
import ftir_workbench
from ftir_workbench.batch.export import verify_batch_export
from ftir_workbench.batch.postprocessing_export import verify_postprocessing_export
assert pathlib.Path(ftir_workbench.__file__).resolve().is_relative_to(pathlib.Path.cwd())
assert ftir_workbench.__version__ == '0.3.1'
payload = pathlib.Path(sys.argv[1]).read_bytes()
if sys.argv[2] == 'independent_baseline_batch':
    results = {'legacy_baseline': verify_batch_export(payload)}
else:
    results = {'legacy_no_recompute': verify_postprocessing_export(payload, recompute=False),
               'legacy_recompute': verify_postprocessing_export(payload, recompute=True)}
assert all(results.values()), results
print(json.dumps(results))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(result_path), artifact_type],
        cwd=old_source, env=dict(os.environ, PYTHONPATH=str(old_source / "src")),
        check=True, capture_output=True, text=True,
    )
    assert all(json.loads(completed.stdout).values())


@pytest.mark.parametrize("filename", [BASELINE_FILES[0], POST_FILES[-1]])
def test_schema1_bundle_cannot_silently_accept_unknown_name_mapping(
    old_directory: Path, filename: str,
) -> None:
    artifact_type = "independent_baseline_batch" if filename in BASELINE_FILES else ARTIFACT_TYPE
    members = read_verified_archive((old_directory / filename).read_bytes(), artifact_type)
    members["export_name_map.csv"] = b"spectrum_id,filename\nforged,wrong.csv\n"
    payload = make_archive(members, artifact_type)
    if artifact_type == ARTIFACT_TYPE:
        assert not verify_postprocessing_export(payload, recompute=False)
    else:
        assert not verify_batch_export(payload)
