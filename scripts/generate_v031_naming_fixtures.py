#!/usr/bin/env python3
"""Generate synthetic old-naming bundles using the exact v0.3.1 Git archive.

No experimental files are read. Binary output belongs in ignored outputs. The
archive supplies both source code and its project version metadata, so this
generator also works after the active installation advances to a newer release.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import itertools
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

import numpy as np

BASE_COMMIT = "14a514d7032f5f7c302f4f47d79a72d895251c1d"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def table(x: np.ndarray, *ys: np.ndarray, labels: tuple[str, ...] | None) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    if labels is not None:
        writer.writerow(["wavenumber", *labels])
    writer.writerows(zip(x, *ys, strict=True))
    return stream.getvalue().encode("utf-8")


def generate(source_root: Path, output: Path, commit: str, archive_hash: str) -> dict:
    import ftir_workbench
    from ftir_baseline.config import CoarseBaselineConfig
    from ftir_workbench.batch import export, importing, postprocessing_export, workspace
    from ftir_workbench.batch.models import BatchWorkspace
    from ftir_workbench.batch.postprocessing import (
        BRANCHES,
        branch_arrays,
        confirm_branch,
        get_branch,
        get_postprocessing_state,
        plain,
        preview_branch,
        update_draft,
    )
    from ftir_workbench.batch.service import preview_coarse, preview_fine
    from ftir_workbench.batch.state import confirm_coarse, confirm_fine, set_coarse_draft, skip_fine

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose a fresh output directory; existing evidence is preserved")
    output.mkdir(parents=True, exist_ok=True)
    module_sources = {}
    for module in (ftir_workbench, export, importing, postprocessing_export, workspace):
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(source_root):
            raise RuntimeError("worker imported code outside the exact archived producer")
        module_sources[module.__name__] = {
            "path": path.relative_to(source_root).as_posix(), "sha256": sha256(path.read_bytes()),
        }
    if (ftir_workbench.__version__ != "0.3.1"
            or importlib.metadata.version("ftir-spectral-workbench") != "0.3.1"):
        raise RuntimeError("fixture requires the source and package metadata of v0.3.1")

    rng = np.random.default_rng(3200)
    x = np.linspace(1800.0, 900.0, 181)
    a = 0.2 + 0.7 * np.exp(-((x - 1450.0) / 35.0) ** 2)
    a += rng.normal(0.0, 0.001, x.size)
    b = 0.3 + 0.6 * np.exp(-((x - 1250.0) / 50.0) ** 2)
    other_x = np.linspace(900.0, 1800.0, 91)
    c = 0.25 + 0.4 * np.exp(-((other_x - 1600.0) / 50.0) ** 2)
    sources = [
        ("G2_90RH.csv", table(x, a, labels=("Absorbance",))),
        ("G3_80RH.csv", table(x, b, labels=("Absorbance",))),
        ("混合 α (25°C).csv", table(x, a * 0.8, b * 1.1, labels=("样品 A 90%RH", "重复"))),
        ("batch01.csv", table(x, a * 0.9, b * 0.9, labels=None)),
        ("other_axis.csv", table(other_x, c, labels=("Absorbance",))),
        ("G2_90RH.csv", table(x, a * 1.2, labels=("Absorbance",))),
    ]
    ws = BatchWorkspace(workspace_id=uuid5(NAMESPACE_URL, "ftir-v032-old-naming").hex)
    counter = itertools.count()
    with patch("ftir_workbench.batch.importing.uuid4", lambda: uuid5(
        NAMESPACE_URL, f"ftir-v032-old-naming-import:{next(counter)}"
    )):
        ids = importing.import_sources(ws, sources)
    if len(ids) != 8 or ws.import_issues:
        raise RuntimeError(f"synthetic import failed: {ws.import_issues}")
    smooth_methods = ("savgol", "gaussian", "moving_average", "median")
    normalize_methods = ("maximum", "minmax_display", "area", "vector")
    for index, sid in enumerate(ids):
        set_coarse_draft(ws, sid, CoarseBaselineConfig(method="none"))
        preview_coarse(ws, sid)
        confirm_coarse(ws, sid)
        if index == 2:
            preview_fine(ws, sid)
            confirm_fine(ws, sid)
        else:
            skip_fine(ws, sid)
        recipes = {
            "smoothed": {"enabled": True, "method": smooth_methods[index % 4]},
            "normalized_baseline": {"enabled": True, "method": normalize_methods[index % 4]},
            "normalized_smoothed": {
                "enabled": True, "method": normalize_methods[(index + 1) % 4],
            },
        }
        for branch, recipe in recipes.items():
            update_draft(ws, sid, branch, recipe)
            preview_branch(ws, sid, branch)
            confirm_branch(ws, sid, branch)
    a_id = ids[0]
    ws.records[a_id] = replace(ws.records[a_id], display_name="用户名称 α 90%RH")
    ws.states[a_id].display_preferences = {"zoom": [1650.0, 1150.0], "unit": "absorbance"}
    update_draft(ws, a_id, "smoothed", {"enabled": True, "method": "savgol",
                                       "savgol_window_length": 9})
    update_draft(ws, a_id, "normalized_baseline", {"enabled": True, "method": "vector"})
    get_postprocessing_state(ws, a_id).normalization_source = "smoothed"
    get_postprocessing_state(ws, a_id).export_choice = "all"
    ws.selected_spectrum_id = a_id
    same_axis = [sid for sid in ids if np.array_equal(ws.records[sid].wavenumber, x)]
    payloads: dict[str, bytes] = {}
    evidence: dict[str, dict] = {}
    for suffix, stage, selection, wide in (
        ("coarse", "coarse", ids, False), ("final", "fine", ids, False),
        ("final_wide", "fine", same_axis, True),
    ):
        artifact = export.build_batch_export(
            ws, selection, stage=stage, include_wide=wide,
            derived_units=("fraction_transmittance", "percent_transmittance"),
        )
        name = f"legacy_baseline_{suffix}_v031.zip"
        payloads[name] = artifact.zip_bytes
        evidence[name] = {"metadata": artifact.metadata,
                          "old_verifier": export.verify_batch_export(artifact.zip_bytes)}
    for label, branches, selection, wide in [
        *((branch, (branch,), ids, False) for branch in BRANCHES),
        ("all", BRANCHES, ids, False), ("all_wide", BRANCHES, same_axis, True),
    ]:
        artifact = postprocessing_export.build_postprocessing_export(
            ws, selection, branches=branches, include_wide=wide,
        )
        name = f"legacy_post_{label}_v031.zip"
        payloads[name] = artifact.zip_bytes
        evidence[name] = {
            "metadata": artifact.metadata,
            "old_verifier_no_recompute": postprocessing_export.verify_postprocessing_export(
                artifact.zip_bytes, recompute=False,
            ),
            "old_verifier_recompute": postprocessing_export.verify_postprocessing_export(
                artifact.zip_bytes, recompute=True,
            ),
        }
    payloads["legacy_workspace_v031.zip"] = workspace.save_batch_workspace(ws)
    restored = workspace.load_batch_workspace(payloads["legacy_workspace_v031.zip"])
    if (restored.workspace_id != ws.workspace_id or restored.display_order != ids
            or restored.selected_spectrum_id != a_id):
        raise RuntimeError("old workspace failed identity self-roundtrip")
    arrays = {}
    state_evidence = {}
    for sid in ids:
        record, state = ws.records[sid], ws.states[sid]
        arrays[f"{sid}__raw_x"] = record.wavenumber
        arrays[f"{sid}__raw_y"] = record.raw_intensity
        state_evidence[sid] = {
            "source_id": record.source_id, "source_filename": ws.sources[record.source_id].original_filename,
            "source_column_index": record.original_column_index,
            "source_column_label": record.original_column_label, "display_name": record.display_name,
            "scientific_input_sha256": record.scientific_input_sha256,
            "fine_decision": state.fine_decision, "snapshots": {}, "branches": {},
        }
        for stage in ("coarse", "fine"):
            snapshot = getattr(state, stage + "_snapshot")
            state_evidence[sid]["snapshots"][stage] = {
                "fingerprint": snapshot.fingerprint,
                "parent_coarse_fingerprint": snapshot.parent_coarse_fingerprint,
                "implementation_fingerprint": snapshot.implementation_fingerprint,
            }
            arrays[f"{sid}__{stage}__x"] = snapshot.result.absorbance_selected.wavenumber
            arrays[f"{sid}__{stage}__estimate"] = snapshot.result.baseline_estimation_spectra
            for field in ("coarse_baseline", "fine_baseline", "total_baseline", "corrected"):
                arrays[f"{sid}__{stage}__{field}"] = getattr(snapshot.result.baseline, field)
        for branch in BRANCHES:
            snapshot = get_branch(ws, sid, branch)
            branch_x, branch_y = branch_arrays(snapshot)
            arrays[f"{sid}__{branch}__x"] = branch_x
            arrays[f"{sid}__{branch}__y"] = branch_y
            actual_x, actual_y = branch_arrays(get_branch(restored, sid, branch))
            np.testing.assert_array_equal(actual_x, branch_x)
            np.testing.assert_array_equal(actual_y, branch_y)
            item = {"fingerprint": snapshot.fingerprint}
            if branch != "baseline":
                item.update(recipe=plain(snapshot.recipe), versions=dict(snapshot.versions),
                            parent_fingerprint=snapshot.parent_fingerprint)
            state_evidence[sid]["branches"][branch] = item
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    payloads["expected_arrays.npz"] = stream.getvalue()
    for name, payload in payloads.items():
        (output / name).write_bytes(payload)
    if not all(value for item in evidence.values() for key, value in item.items()
               if key.startswith("old_verifier")):
        raise RuntimeError("a generated bundle failed its own old verifier")
    metadata = {
        "generator": "scripts/generate_v031_naming_fixtures.py", "base_commit": commit,
        "source_environment": "exact temporary Git archive", "archive_sha256": archive_hash,
        "distribution_metadata_source": "exact archived pyproject.toml name/version",
        "dataset": "fixed seed 3200; analytic peaks and synthetic noise only",
        "experimental_data_read": False, "workspace_schema": ws.schema_version,
        "workspace_id": ws.workspace_id, "spectrum_ids": ids, "same_axis_ids": same_axis,
        "module_sources": module_sources, "workbench_version": ftir_workbench.__version__,
        "python": sys.version,
        "dependencies": {name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "pybaselines", "pydantic", "ftir-spectral-workbench",
        )},
        "states": state_evidence, "bundle_evidence": evidence,
        "artifacts": {name: {
            "size_bytes": len(payload), "sha256": sha256(payload),
            "members": zipfile.ZipFile(io.BytesIO(payload)).namelist(),
        } for name, payload in payloads.items()},
        "self_roundtrip_arrays_exact": True, "expected_array_count": len(arrays),
    }
    (output / "legacy_generation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-commit", default=BASE_COMMIT)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/validation-private/v032/legacy-v031"))
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--archive-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.source_root is not None:
        metadata = generate(args.source_root.resolve(), output, args.base_commit, args.archive_sha256)
        print(json.dumps({key: metadata[key] for key in (
            "base_commit", "archive_sha256", "workbench_version", "dependencies",
            "workspace_schema", "expected_array_count", "self_roundtrip_arrays_exact",
        )}, indent=2))
        print(f"Wrote {len(metadata['artifacts'])} synthetic artifacts to {output}")
        return 0
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.base_commit}^{{commit}}"], cwd=root, text=True,
    ).strip()
    payload = subprocess.check_output(["git", "archive", "--format=tar", commit], cwd=root)
    with tempfile.TemporaryDirectory(prefix="ftir-v031-naming-") as temporary:
        snapshot = (Path(temporary) / "source").resolve()
        snapshot.mkdir()
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            if any(member.issym() or member.islnk()
                   or not (snapshot / member.name).resolve().is_relative_to(snapshot)
                   for member in archive.getmembers()):
                raise RuntimeError("unsafe member in archived producer")
            archive.extractall(snapshot, filter="data")
        project = tomllib.loads((snapshot / "pyproject.toml").read_text())["project"]
        distribution = snapshot / "src" / f"ftir_spectral_workbench-{project['version']}.dist-info"
        distribution.mkdir()
        (distribution / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {project['name']}\nVersion: {project['version']}\n",
        )
        environment = dict(os.environ, PYTHONPATH=str(snapshot / "src"))
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--source-root", str(snapshot),
             "--base-commit", commit, "--archive-sha256", sha256(payload), "--output", str(output)],
            cwd=snapshot, env=environment, check=False,
        )
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
