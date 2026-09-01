from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

import scripts.audit_v025_release as audit_module
from scripts.audit_v025_release import (
    EXPECTED_FROZEN_FILE_COUNT,
    _build_smoothing_fixture,
    audit_release_metadata,
    audit_science_freeze,
    audit_smoothing_workflow,
    run_release_audit,
)

REPOSITORY = Path(__file__).resolve().parents[2]


def test_v025_release_audit_passes_without_private_path_or_raw_data_output() -> None:
    summary = run_release_audit(REPOSITORY)
    serialized = json.dumps(summary, ensure_ascii=False, sort_keys=True)

    assert summary["status"] == "pass"
    assert summary["repository"] == REPOSITORY.name
    assert str(REPOSITORY) not in serialized
    assert "/Users/" not in serialized
    assert "/private/var/" not in serialized
    assert summary["privacy"] == {
        "absolute_paths_emitted": False,
        "raw_data_read": False,
        "fixture_scope": "deterministic synthetic Prepared only",
    }

    freeze = summary["checks"]["v021_science_freeze"]
    assert freeze["expected_file_count"] == EXPECTED_FROZEN_FILE_COUNT
    assert freeze["manifest_file_count"] == EXPECTED_FROZEN_FILE_COUNT
    assert freeze["checked_worktree_count"] == EXPECTED_FROZEN_FILE_COUNT
    assert freeze["checked_start_commit_count"] == EXPECTED_FROZEN_FILE_COUNT
    assert freeze["git_diff_changed_paths"] == []

    release_metadata = summary["checks"]["release_metadata"]
    assert release_metadata["status"] == "pass"
    assert release_metadata["distribution_version"] == "0.2.5"
    assert release_metadata["workbench_version"] == "0.2.5"
    assert all(release_metadata["checks"].values())

    smoothing = summary["checks"]["smoothing_and_2d"]
    assert smoothing["status"] == "pass"
    assert all(smoothing["checks"].values())
    assert smoothing["parent_prepared_data_sha256"] != smoothing[
        "child_prepared_data_sha256"
    ]
    assert smoothing["unsmoothed_twodcos_fingerprint"] != smoothing[
        "smoothed_twodcos_fingerprint"
    ]

    compatibility = summary["checks"]["v021_compatibility_and_privacy"]
    assert compatibility["status"] == "pass"
    assert compatibility["legacy_base_commit"] == freeze["start_commit"]
    assert compatibility["legacy_generator_version"] == "0.2.1"
    assert all(compatibility["checks"].values())


def test_v025_science_freeze_reports_manifest_tampering(tmp_path: Path) -> None:
    source = REPOSITORY / "artifacts/v0.2.1_science_freeze_manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest["files"][0]["size_bytes"] += 1
    manifest["files"][0]["sha256"] = "0" * 64
    tampered = tmp_path / "tampered-v021-freeze.json"
    tampered.write_text(json.dumps(manifest), encoding="utf-8")

    result = audit_science_freeze(REPOSITORY, tampered)
    serialized = json.dumps(result, sort_keys=True)

    assert result["status"] == "fail"
    assert any("mismatch" in item["reason"] for item in result["mismatches"])
    assert result["manifest"] == "<external manifest override>"
    assert str(tampered) not in serialized


def test_v025_science_freeze_rejects_a_replaced_manifest_path(
    tmp_path: Path,
) -> None:
    source = REPOSITORY / "artifacts/v0.2.1_science_freeze_manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    replacement = (REPOSITORY / "LICENSE").read_bytes()
    manifest["files"][0] = {
        "path": "LICENSE",
        "size_bytes": len(replacement),
        "sha256": hashlib.sha256(replacement).hexdigest(),
    }
    tampered = tmp_path / "replaced-path-v021-freeze.json"
    tampered.write_text(json.dumps(manifest), encoding="utf-8")

    result = audit_science_freeze(REPOSITORY, tampered)

    assert result["status"] == "fail"
    assert any(
        item["reason"] == "manifest path set differs from the start-tree freeze"
        for item in result["mismatches"]
    )


def test_v025_science_freeze_rejects_an_untracked_file_in_frozen_root() -> None:
    probe = REPOSITORY / "src" / "ftir_baseline" / "_v025_freeze_probe.py"
    assert not probe.exists()
    try:
        probe.write_text("# synthetic freeze-audit probe\n", encoding="utf-8")
        result = audit_science_freeze(REPOSITORY)
    finally:
        probe.unlink(missing_ok=True)

    assert result["status"] == "fail"
    assert result["current_frozen_extra_paths"] == [
        "src/ftir_baseline/_v025_freeze_probe.py"
    ]
    assert any(
        item["reason"] == "current frozen-root path set differs from start tree"
        for item in result["mismatches"]
    )


def test_v025_smoothing_audit_rejects_tampered_bundle() -> None:
    _, _, _, bundle = _build_smoothing_fixture()
    tampered_stream = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(bundle), "r") as source,
        zipfile.ZipFile(
            tampered_stream,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as target,
    ):
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "smoothing_config.json":
                payload += b" "
            target.writestr(info, payload)

    result = audit_smoothing_workflow(
        smoothing_bundle_override=tampered_stream.getvalue()
    )

    assert result["status"] == "fail"
    assert result["checks"]["smoothing_bundle_verified"] is False
    assert result["checks"]["smoothing_child_exact_reload"] is False
    assert "smoothing_bundle_verified" in result["failed_checks"]


def test_v025_audit_rejects_runtime_modules_outside_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _reject_external(*_args: object, **_kwargs: object) -> dict[str, str]:
        raise RuntimeError("simulated external module source")

    monkeypatch.setattr(
        audit_module,
        "_assert_modules_from_snapshot",
        _reject_external,
    )

    metadata = audit_release_metadata(REPOSITORY)
    assert metadata["status"] == "fail"
    assert metadata["checks"]["runtime_modules_from_repository"] is False
    assert metadata["module_sources"] == {}
    with pytest.raises(RuntimeError, match="simulated external module source"):
        audit_smoothing_workflow(repository=REPOSITORY)
