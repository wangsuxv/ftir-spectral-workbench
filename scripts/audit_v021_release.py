#!/usr/bin/env python3
"""Run the local, machine-readable v0.2.1 release audit.

The compatibility check deliberately generates its artifacts with an exported
snapshot of the v0.2.0 base commit.  The working-tree code then verifies those
artifacts and reloads the public Prepared checkpoint contained in each bundle.
No ignored or untracked spectrum is opened or hashed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

DEFAULT_BASE_COMMIT = "5976ecedeee4a22391a9d426280b272ad66d802a"
DEFAULT_FREEZE_MANIFEST = Path("artifacts/v0.2_science_freeze_manifest.json")
ALLOWED_ORIGINAL_TRACKED = {"data/original/README.md"}
FORBIDDEN_TRACKED_SUFFIXES = {
    ".dx",
    ".ftirw",
    ".jcamp",
    ".jdx",
    ".sp",
    ".spa",
    ".spc",
    ".spg",
    ".srs",
    ".xls",
    ".xlsx",
    ".zip",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        timeout=120,
    )
    return completed.stdout


def _git_text(repository: Path, *arguments: str) -> str:
    return _git_bytes(repository, *arguments).decode("utf-8").strip()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def audit_science_freeze(
    repository: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Compare every recorded frozen file's size and SHA-256 with the worktree."""

    path = manifest_path or repository / DEFAULT_FREEZE_MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        entries = manifest["files"]
        if not isinstance(entries, list):
            raise TypeError("manifest 'files' must be a list")
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        return {
            "status": "fail",
            "manifest": str(path.relative_to(repository)) if _inside(path, repository) else str(path),
            "error": f"{type(error).__name__}: {error}",
            "checked_file_count": 0,
            "mismatches": [],
        }

    mismatches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            mismatches.append({"reason": "file entry is not an object"})
            continue
        relative_name = str(entry.get("path", ""))
        candidate = repository / relative_name
        if not relative_name or relative_name in seen:
            mismatches.append(
                {
                    "path": relative_name,
                    "reason": "empty or duplicate path",
                }
            )
            continue
        seen.add(relative_name)
        if not _inside(candidate, repository):
            mismatches.append({"path": relative_name, "reason": "path escapes repository"})
            continue
        if not candidate.is_file():
            mismatches.append({"path": relative_name, "reason": "missing file"})
            continue
        actual_size = candidate.stat().st_size
        actual_sha256 = _sha256_file(candidate)
        expected_size = entry.get("size_bytes")
        expected_sha256 = entry.get("sha256")
        if actual_size != expected_size or actual_sha256 != expected_sha256:
            mismatches.append(
                {
                    "path": relative_name,
                    "reason": "size or SHA-256 mismatch",
                    "expected_size_bytes": expected_size,
                    "actual_size_bytes": actual_size,
                    "expected_sha256": expected_sha256,
                    "actual_sha256": actual_sha256,
                }
            )

    recorded_count = manifest.get("file_count")
    if recorded_count != len(entries):
        mismatches.append(
            {
                "reason": "manifest file_count does not match files list",
                "recorded_file_count": recorded_count,
                "listed_file_count": len(entries),
            }
        )
    return {
        "status": "pass" if not mismatches else "fail",
        "manifest": str(path.relative_to(repository)) if _inside(path, repository) else str(path),
        "manifest_base_commit": manifest.get("base_commit"),
        "expected_file_count": recorded_count,
        "checked_file_count": len(seen),
        "mismatches": mismatches,
        "algorithm": "SHA-256 over exact file bytes",
    }


def _is_ignored(repository: Path, relative_name: str) -> bool:
    completed = subprocess.run(
        ("git", "check-ignore", "--quiet", "--", relative_name),
        cwd=repository,
        check=False,
        capture_output=True,
        timeout=30,
    )
    return completed.returncode == 0


def audit_repository_privacy(repository: Path) -> dict[str, Any]:
    """Audit only the Git index and ignore rules; never read ignored spectra."""

    tracked_payload = _git_bytes(repository, "ls-files", "-z")
    tracked = sorted(
        item.decode("utf-8") for item in tracked_payload.split(b"\0") if item
    )
    tracked_original = [name for name in tracked if name.startswith("data/original/")]
    tracked_outputs = [name for name in tracked if name.startswith("outputs/")]
    tracked_archives = [
        name for name in tracked if Path(name).suffix.lower() in FORBIDDEN_TRACKED_SUFFIXES
    ]
    tracked_sensitive_manifests = [
        name for name in tracked if name == "artifacts/real_data_demo_manifest.json"
    ]
    ignore_rule_checks = {
        "raw_spectrum_path_is_ignored": _is_ignored(
            repository, "data/original/.audit-ignore-probe.dpt"
        ),
        "generated_output_path_is_ignored": _is_ignored(
            repository, "outputs/.audit-ignore-probe.zip"
        ),
    }
    failures: list[str] = []
    if set(tracked_original) != ALLOWED_ORIGINAL_TRACKED:
        failures.append("data/original tracking differs from the README-only allowlist")
    if tracked_outputs:
        failures.append("generated outputs are tracked")
    if tracked_archives:
        failures.append("bundle, archive, vendor-binary, or structured binary files are tracked")
    if tracked_sensitive_manifests:
        failures.append("sensitive real-data replay metadata is tracked")
    if not all(ignore_rule_checks.values()):
        failures.append("required raw-data/output ignore rules are ineffective")
    return {
        "status": "pass" if not failures else "fail",
        "scope": "Git index and ignore rules only; ignored spectrum contents were not read",
        "allowed_original_files": sorted(ALLOWED_ORIGINAL_TRACKED),
        "tracked_original_files": tracked_original,
        "tracked_outputs": tracked_outputs,
        "tracked_forbidden_archives_or_binaries": tracked_archives,
        "tracked_sensitive_manifests": tracked_sensitive_manifests,
        "ignore_rule_checks": ignore_rule_checks,
        "failures": failures,
    }


def _assert_modules_from_snapshot(source_root: Path, modules: list[Any]) -> dict[str, str]:
    expected_root = (source_root / "src").resolve()
    resolved: dict[str, str] = {}
    for module in modules:
        module_path = Path(module.__file__).resolve()
        if not _inside(module_path, expected_root):
            raise RuntimeError(
                f"{module.__name__} was imported from {module_path}, not {expected_root}"
            )
        resolved[module.__name__] = str(module_path.relative_to(source_root))
    return resolved


def _generate_legacy_artifacts(source_root: Path, artifact_dir: Path) -> dict[str, Any]:
    """Generate compatibility artifacts in a process importing only base sources."""

    import numpy as np

    import ftir2dcos
    import ftir_baseline
    import ftir_workbench
    from ftir2dcos.twodcos import compute_2dcos
    from ftir_baseline.config import PipelineConfig
    from ftir_baseline.models import SpectrumSet
    from ftir_workbench.export import build_project_bundle, build_twodcos_bundle
    from ftir_workbench.services import BaselineWorkflowService

    module_sources = _assert_modules_from_snapshot(
        source_root, [ftir_baseline, ftir2dcos, ftir_workbench]
    )
    if ftir_workbench.__version__ != "0.2.0":
        raise RuntimeError(
            f"base snapshot reported ftir_workbench {ftir_workbench.__version__}, expected 0.2.0"
        )

    wavenumber = np.linspace(1800.0, 900.0, 61, dtype=np.float64)
    baseline = 0.02 + 2.0e-5 * (1800.0 - wavenumber)
    peak = np.exp(-0.5 * ((wavenumber - 1250.0) / 32.0) ** 2)
    spectra = np.vstack(
        [baseline + scale * 0.2 * peak for scale in (0.8, 1.0, 1.25)]
    )
    series = SpectrumSet(
        wavenumber=wavenumber,
        perturbation=np.array([0.0, 2.0, 5.0], dtype=np.float64),
        perturbation_labels=("0MIN", "2MIN", "5MIN"),
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="synthetic-v0.2.0-release-audit",
        metadata={"order_policy": "numeric_perturbation_stable"},
    )
    config = PipelineConfig(
        input_unit="absorbance",
        wavenumber_range=(1800.0, 900.0),
        series_mode="independent_locked",
        coarse_baseline={"method": "none"},
        fine_baseline={"enabled": False, "method": "none"},
        normalization={"method": "none"},
    )
    service = BaselineWorkflowService()
    result = service.run(series, config)
    prepared = service.prepared(baseline_run_id="v0.2.0-release-audit")
    baseline_bundle = service.export_baseline_only(
        result,
        prepared=prepared,
        qc_figures={},
    )
    if not isinstance(baseline_bundle, bytes):
        raise TypeError("baseline exporter did not return bytes")
    analysis = compute_2dcos(
        prepared.spectra,
        prepared.wavenumber,
        convention="2dpy_compatible",
    )
    twodcos_bundle = build_twodcos_bundle(
        prepared,
        analysis,
        {"ranges": [[1800.0, 900.0]], "convention": "2dpy_compatible"},
    )
    project_bundle = build_project_bundle(
        baseline_bundle,
        (twodcos_bundle,),
        {"project_name": "v0.2.0 synthetic compatibility audit"},
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "baseline": ("baseline_run_v020.zip", baseline_bundle),
        "twodcos": ("twodcos_run_v020.zip", twodcos_bundle),
        "project": ("project_v020.ftirw", project_bundle),
    }
    for _, (name, payload) in artifacts.items():
        (artifact_dir / name).write_bytes(payload)
    np.savez(
        artifact_dir / "prepared_expected_v020.npz",
        wavenumber=prepared.wavenumber,
        perturbation=prepared.perturbation,
        spectra=prepared.spectra,
    )
    metadata = {
        "generator": "exact exported v0.2.0 Git snapshot",
        "versions": {
            "ftir_workbench": ftir_workbench.__version__,
            "ftir_baseline": ftir_baseline.__version__,
            "ftir2dcos": ftir2dcos.__version__,
        },
        "module_sources": module_sources,
        "prepared": {
            "baseline_run_id": prepared.baseline_run_id,
            "baseline_fingerprint": prepared.baseline_fingerprint,
            "prepared_data_sha256": prepared.prepared_data_sha256,
            "perturbation_labels": list(prepared.perturbation_labels),
            "wavenumber_shape": list(prepared.wavenumber.shape),
            "spectra_shape": list(prepared.spectra.shape),
        },
        "artifacts": {
            kind: {"file": name, "size_bytes": len(payload), "sha256": _sha256(payload)}
            for kind, (name, payload) in artifacts.items()
        },
    }
    (artifact_dir / "legacy_generation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _load_prepared_from_twodcos(bundle: bytes) -> Any:
    from ftir_workbench.export import load_prepared

    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        return load_prepared(
            (
                archive.read("source_prepared_spectrum.csv"),
                archive.read("source_prepared_spectrum.meta.json"),
            )
        )


def _prepared_matches_expected(prepared: Any, artifact_dir: Path) -> bool:
    import numpy as np

    metadata = json.loads((artifact_dir / "legacy_generation.json").read_text("utf-8"))
    with np.load(artifact_dir / "prepared_expected_v020.npz", allow_pickle=False) as expected:
        arrays_match = all(
            np.array_equal(getattr(prepared, name), expected[name])
            for name in ("wavenumber", "perturbation", "spectra")
        )
    expected_prepared = metadata["prepared"]
    return bool(
        arrays_match
        and list(prepared.perturbation_labels) == expected_prepared["perturbation_labels"]
        and prepared.baseline_run_id == expected_prepared["baseline_run_id"]
        and prepared.baseline_fingerprint == expected_prepared["baseline_fingerprint"]
        and prepared.prepared_data_sha256 == expected_prepared["prepared_data_sha256"]
    )


def _same_prepared(first: Any, second: Any) -> bool:
    import numpy as np

    return bool(
        np.array_equal(first.wavenumber, second.wavenumber)
        and np.array_equal(first.perturbation, second.perturbation)
        and np.array_equal(first.spectra, second.spectra)
        and first.perturbation_labels == second.perturbation_labels
        and first.prepared_data_sha256 == second.prepared_data_sha256
        and first.baseline_fingerprint == second.baseline_fingerprint
        and first.baseline_run_id == second.baseline_run_id
    )


def _verify_generated_artifact_hashes(
    artifact_dir: Path,
    generation: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    for kind, record in generation["artifacts"].items():
        path = artifact_dir / record["file"]
        if not path.is_file():
            failures.append(f"missing generated {kind} artifact")
            continue
        if path.stat().st_size != record["size_bytes"] or _sha256_file(path) != record["sha256"]:
            failures.append(f"generated {kind} artifact hash mismatch")
    return failures


def _audit_legacy_artifacts(artifact_dir: Path) -> dict[str, Any]:
    from ftir_baseline.export import verify_export_manifest
    from ftir_workbench.export import (
        load_prepared,
        verify_project_bundle,
        verify_twodcos_bundle,
        verify_workbench_manifest,
    )

    generation = json.loads((artifact_dir / "legacy_generation.json").read_text("utf-8"))
    baseline = (artifact_dir / generation["artifacts"]["baseline"]["file"]).read_bytes()
    twodcos = (artifact_dir / generation["artifacts"]["twodcos"]["file"]).read_bytes()
    project = (artifact_dir / generation["artifacts"]["project"]["file"]).read_bytes()

    checks: dict[str, bool] = {
        "generated_artifact_hashes": not _verify_generated_artifact_hashes(
            artifact_dir, generation
        ),
        "baseline_legacy_manifest": verify_export_manifest(baseline),
        "baseline_workbench_manifest": verify_workbench_manifest(baseline),
        "twodcos_bundle": verify_twodcos_bundle(twodcos),
        "project_bundle": verify_project_bundle(project),
    }
    baseline_prepared = load_prepared(baseline)
    twodcos_prepared = _load_prepared_from_twodcos(twodcos)
    checks["baseline_prepared_exact_reload"] = _prepared_matches_expected(
        baseline_prepared, artifact_dir
    )
    checks["twodcos_source_prepared_exact_reload"] = _prepared_matches_expected(
        twodcos_prepared, artifact_dir
    )
    checks["standalone_prepared_checkpoints_equal"] = _same_prepared(
        baseline_prepared, twodcos_prepared
    )

    with zipfile.ZipFile(io.BytesIO(project), "r") as project_archive:
        nested_baseline = project_archive.read("baseline_run.zip")
        nested_twodcos = project_archive.read("twodcos_run_01.zip")
        project_config = json.loads(project_archive.read("project_config.json"))
    checks.update(
        {
            "project_nested_baseline_byte_identical": nested_baseline == baseline,
            "project_nested_twodcos_byte_identical": nested_twodcos == twodcos,
            "project_nested_baseline_manifest": verify_export_manifest(nested_baseline),
            "project_nested_twodcos_bundle": verify_twodcos_bundle(nested_twodcos),
            "project_config_reload": project_config
            == {"project_name": "v0.2.0 synthetic compatibility audit"},
        }
    )
    nested_baseline_prepared = load_prepared(nested_baseline)
    nested_twodcos_prepared = _load_prepared_from_twodcos(nested_twodcos)
    checks["project_nested_baseline_prepared_exact_reload"] = _prepared_matches_expected(
        nested_baseline_prepared, artifact_dir
    )
    checks["project_nested_twodcos_source_prepared_exact_reload"] = (
        _prepared_matches_expected(nested_twodcos_prepared, artifact_dir)
    )
    checks["project_nested_prepared_checkpoints_equal"] = _same_prepared(
        nested_baseline_prepared, nested_twodcos_prepared
    )

    # Matrix reload has no public loader in v0.2.0/v0.2.1.  The public verifier
    # above validates their recorded sizes and hashes; report that boundary
    # explicitly instead of claiming a nonexistent project loader.
    with zipfile.ZipFile(io.BytesIO(nested_twodcos), "r") as archive:
        matrix_members = [
            name
            for name in (
                "synchronous_matrix.csv",
                "asynchronous_matrix.csv",
                "dynamic_spectra.csv",
            )
            if name in archive.namelist() and archive.getinfo(name).file_size > 0
        ]
    checks["twodcos_numeric_payloads_present"] = len(matrix_members) == 3

    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "pass" if not failed_checks else "fail",
        "generation": {
            "base_workbench_version": generation["versions"]["ftir_workbench"],
            "module_sources": generation["module_sources"],
            "artifacts": generation["artifacts"],
        },
        "checks": checks,
        "failed_checks": failed_checks,
        "reload_scope": {
            "public_apis": [
                "ftir_baseline.export.verify_export_manifest",
                "ftir_workbench.export.verify_workbench_manifest",
                "ftir_workbench.export.verify_twodcos_bundle",
                "ftir_workbench.export.verify_project_bundle",
                "ftir_workbench.export.load_prepared",
            ],
            "project_reload": (
                "verify .ftirw, open its documented ZIP members, verify nested bundles, "
                "and reload both nested Prepared checkpoints"
            ),
            "not_claimed": "No public load_project or 2D-matrix loader exists.",
        },
        "prepared_shape": generation["prepared"]["spectra_shape"],
        "matrix_members_verified": matrix_members,
        "numpy_comparison": "exact array equality; no tolerance, sorting, or interpolation",
    }


def audit_legacy_bundle_compatibility(
    repository: Path,
    base_commit: str = DEFAULT_BASE_COMMIT,
) -> dict[str, Any]:
    """Generate bundles with the exact base snapshot, then audit with current code."""

    resolved_commit = _git_text(repository, "rev-parse", "--verify", f"{base_commit}^{{commit}}")
    archive_payload = _git_bytes(repository, "archive", "--format=tar", resolved_commit)
    with tempfile.TemporaryDirectory(prefix="ftir-v021-release-audit-") as temporary:
        temporary_root = Path(temporary)
        snapshot = temporary_root / "v020-source"
        artifacts = temporary_root / "legacy-artifacts"
        snapshot.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:") as archive:
            members = archive.getmembers()
            unsafe = [
                member.name
                for member in members
                if not _inside(snapshot / member.name, snapshot)
                or member.issym()
                or member.islnk()
            ]
            if unsafe:
                raise RuntimeError(f"git archive contains unsafe members: {unsafe!r}")
            if "filter" in inspect.signature(archive.extractall).parameters:
                archive.extractall(snapshot, members=members, filter="data")
            else:  # pragma: no cover - compatibility with early Python 3.11
                archive.extractall(snapshot, members=members)
        command = (
            sys.executable,
            str(Path(__file__).resolve()),
            "_generate-legacy",
            "--source-root",
            str(snapshot),
            "--artifact-dir",
            str(artifacts),
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(snapshot / "src")
        subprocess.run(
            command,
            cwd=snapshot,
            env=environment,
            check=True,
            capture_output=True,
            timeout=180,
        )
        result = _audit_legacy_artifacts(artifacts)
    result.update(
        {
            "base_commit_requested": base_commit,
            "base_commit_resolved": resolved_commit,
            "generation_environment": "temporary git archive; artifacts deleted after audit",
        }
    )
    return result


def _safe_check(name: str, function: Any) -> tuple[str, dict[str, Any]]:
    try:
        result = function()
    except Exception as error:
        result = {
            "status": "fail",
            "error": f"{type(error).__name__}: {error}",
        }
    return name, result


def run_release_audit(
    repository: Path,
    *,
    base_commit: str = DEFAULT_BASE_COMMIT,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Run every v0.2.1 release audit and return a JSON-serializable summary."""

    repository = repository.resolve()
    checks = dict(
        (
            _safe_check(
                "science_freeze",
                lambda: audit_science_freeze(repository, manifest_path),
            ),
            _safe_check(
                "legacy_bundle_compatibility",
                lambda: audit_legacy_bundle_compatibility(repository, base_commit),
            ),
            _safe_check(
                "repository_privacy",
                lambda: audit_repository_privacy(repository),
            ),
        )
    )
    failed_checks = sorted(name for name, result in checks.items() if result["status"] != "pass")
    return {
        "schema_version": "1.0",
        "audit": "FTIR Spectral Workbench v0.2.1 local release audit",
        "status": "pass" if not failed_checks else "fail",
        "base_commit": base_commit,
        "repository": repository.name,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def _main_generate_legacy(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    namespace = parser.parse_args(arguments)
    _generate_legacy_artifacts(namespace.source_root.resolve(), namespace.artifact_dir.resolve())
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the v0.2.1 science freeze, v0.2.0 bundle compatibility, "
            "and repository privacy."
        )
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--base-commit",
        default=DEFAULT_BASE_COMMIT,
        help="exact v0.2.0 commit used to generate legacy bundles",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="optional science-freeze manifest override",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the JSON result to this path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_generate-legacy":
        return _main_generate_legacy(arguments[1:])
    namespace = _build_parser().parse_args(arguments)
    repository = namespace.repository.resolve()
    manifest = namespace.manifest
    if manifest is not None and not manifest.is_absolute():
        manifest = repository / manifest
    summary = run_release_audit(
        repository,
        base_commit=namespace.base_commit,
        manifest_path=manifest,
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if namespace.output is not None:
        output = namespace.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
