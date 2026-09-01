from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_v021_release import (
    DEFAULT_BASE_COMMIT,
    audit_repository_privacy,
    audit_science_freeze,
    run_release_audit,
)

REPOSITORY = Path(__file__).resolve().parents[2]


def test_exact_v020_bundles_verify_and_reload_with_current_code() -> None:
    summary = run_release_audit(REPOSITORY)
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["status"] == "pass"
    assert summary["repository"] == REPOSITORY.name
    assert str(REPOSITORY) not in serialized
    assert ".audit-ignore-probe" not in serialized
    assert summary["base_commit"] == DEFAULT_BASE_COMMIT
    compatibility = summary["checks"]["legacy_bundle_compatibility"]
    assert compatibility["base_commit_resolved"] == DEFAULT_BASE_COMMIT
    assert compatibility["generation"]["base_workbench_version"] == "0.2.0"
    assert all(compatibility["checks"].values())
    assert compatibility["prepared_shape"] == [3, 61]
    assert compatibility["reload_scope"]["not_claimed"].startswith(
        "No public load_project"
    )


def test_science_freeze_audit_reports_manifest_tampering(tmp_path: Path) -> None:
    source = REPOSITORY / "artifacts/v0.2_science_freeze_manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest["files"][0]["size_bytes"] += 1
    manifest["files"][0]["sha256"] = "0" * 64
    tampered = tmp_path / "tampered-freeze.json"
    tampered.write_text(json.dumps(manifest), encoding="utf-8")

    result = audit_science_freeze(REPOSITORY, tampered)

    assert result["status"] == "fail"
    assert result["mismatches"][0]["reason"] == "size or SHA-256 mismatch"


def test_privacy_audit_checks_index_without_reading_ignored_spectra() -> None:
    result = audit_repository_privacy(REPOSITORY)

    assert result["status"] == "pass"
    assert result["tracked_original_files"] == ["data/original/README.md"]
    assert result["tracked_outputs"] == []
    assert result["tracked_forbidden_archives_or_binaries"] == []
    assert result["tracked_sensitive_manifests"] == []
    assert all(result["ignore_rule_checks"].values())
    assert "ignored spectrum contents were not read" in result["scope"]
