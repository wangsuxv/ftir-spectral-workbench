#!/usr/bin/env python3
"""Run the privacy-safe, machine-readable v0.2.5 local release audit.

The audit uses deterministic synthetic spectra only. It never opens ignored raw
data, and its JSON result contains repository-relative labels rather than local
absolute paths.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ftir_workbench.config import TwoDCOSConfig
    from ftir_workbench.models import PreparedSpectralDataset

try:
    from scripts.audit_v021_release import (
        _assert_modules_from_snapshot,
        _audit_legacy_artifacts,
        audit_repository_privacy,
    )
except ModuleNotFoundError:  # Direct ``python scripts/audit_v025_release.py`` execution.
    from audit_v021_release import (  # type: ignore[import-not-found,no-redef]
        _assert_modules_from_snapshot,
        _audit_legacy_artifacts,
        audit_repository_privacy,
    )

DEFAULT_FREEZE_MANIFEST = Path("artifacts/v0.2.1_science_freeze_manifest.json")
EXPECTED_START_COMMIT = "92513def080001de4c226fcea0fde484ae8d97fb"
EXPECTED_START_TREE = "0f5fb43486e39bc9158cad07860c5bc9c149d5ff"
EXPECTED_FROZEN_FILE_COUNT = 34
EXPECTED_FROZEN_ROOTS = (
    "src/ftir_baseline/**",
    "src/ftir2dcos/twodcos/**",
    "src/ftir2dcos/preprocessing/smoothing.py",
    "src/ftir2dcos/peak_order.py",
    "src/ftir_workbench/cross_views.py",
    "src/ftir_workbench/display_units.py",
    "src/ftir_workbench/services/baseline_service.py",
    "src/ftir_workbench/services/twodcos_service.py",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _configure_private_runtime() -> None:
    """Keep third-party cache diagnostics away from private home-directory paths."""

    cache = Path(tempfile.gettempdir()) / "ftir-workbench-v025-matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)


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


def _manifest_label(repository: Path, path: Path) -> str:
    if _inside(path, repository):
        return path.resolve().relative_to(repository.resolve()).as_posix()
    return "<external manifest override>"


def _frozen_paths_from_start_tree(
    repository: Path,
    resolved_commit: str,
) -> tuple[str, ...]:
    """Expand the fixed freeze contract independently of manifest file entries."""

    tree_paths = {
        line
        for line in _git_text(
            repository,
            "ls-tree",
            "-r",
            "--name-only",
            resolved_commit,
        ).splitlines()
        if line
    }
    expected: set[str] = set()
    for root in EXPECTED_FROZEN_ROOTS:
        if root.endswith("/**"):
            prefix = root.removesuffix("**")
            expected.update(path for path in tree_paths if path.startswith(prefix))
        elif root in tree_paths:
            expected.add(root)
        else:
            raise ValueError(f"frozen root is absent from start tree: {root}")
    return tuple(sorted(expected))


def _frozen_pathspecs() -> tuple[str, ...]:
    return tuple(
        root.removesuffix("/**") if root.endswith("/**") else root
        for root in EXPECTED_FROZEN_ROOTS
    )


def _runtime_module_sources(repository: Path) -> dict[str, str]:
    """Prove that scientific/runtime imports come from this repository checkout."""

    import ftir2dcos
    import ftir_baseline
    import ftir_workbench

    return _assert_modules_from_snapshot(
        repository,
        [ftir_baseline, ftir2dcos, ftir_workbench],
    )


def audit_science_freeze(
    repository: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Audit all 34 frozen files in both the start commit and current worktree."""

    repository = repository.resolve()
    path = (manifest_path or repository / DEFAULT_FREEZE_MANIFEST).resolve()
    label = _manifest_label(repository, path)
    try:
        manifest_value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest_value, dict):
            raise TypeError("manifest root must be an object")
        entries = manifest_value["files"]
        if not isinstance(entries, list):
            raise TypeError("manifest files must be a list")
        manifest_base_commit = str(manifest_value["base_commit"])
        resolved_commit = _git_text(
            repository,
            "rev-parse",
            "--verify",
            f"{EXPECTED_START_COMMIT}^{{commit}}",
        )
        resolved_tree = _git_text(repository, "rev-parse", f"{resolved_commit}^{{tree}}")
        expected_paths = _frozen_paths_from_start_tree(repository, resolved_commit)
        pathspecs = _frozen_pathspecs()
        current_paths = tuple(
            sorted(
                line
                for line in _git_text(
                    repository,
                    "ls-files",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "--",
                    *pathspecs,
                ).splitlines()
                if line
            )
        )
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        return {
            "status": "fail",
            "manifest": label,
            "error_type": type(error).__name__,
            "expected_file_count": EXPECTED_FROZEN_FILE_COUNT,
            "checked_worktree_count": 0,
            "checked_start_commit_count": 0,
            "mismatches": [],
            "git_diff_changed_paths": [],
        }

    mismatches: list[dict[str, Any]] = []
    seen: set[str] = set()
    checked_worktree = 0
    checked_start_commit = 0
    for entry in entries:
        if not isinstance(entry, dict):
            mismatches.append({"reason": "file entry is not an object"})
            continue
        relative_name = str(entry.get("path", ""))
        candidate = repository / relative_name
        if (
            not relative_name
            or relative_name in seen
            or Path(relative_name).is_absolute()
            or not _inside(candidate, repository)
        ):
            mismatches.append({"path": "<unsafe-or-duplicate>", "reason": "invalid path"})
            continue
        seen.add(relative_name)
        expected_size = entry.get("size_bytes")
        expected_sha256 = entry.get("sha256")

        if not candidate.is_file():
            mismatches.append({"path": relative_name, "reason": "worktree file missing"})
        else:
            worktree_payload = candidate.read_bytes()
            checked_worktree += 1
            if (
                len(worktree_payload) != expected_size
                or _sha256(worktree_payload) != expected_sha256
            ):
                mismatches.append(
                    {"path": relative_name, "reason": "worktree size or SHA-256 mismatch"}
                )

        try:
            start_payload = _git_bytes(
                repository,
                "show",
                f"{resolved_commit}:{relative_name}",
            )
        except subprocess.SubprocessError:
            mismatches.append({"path": relative_name, "reason": "start-commit file missing"})
        else:
            checked_start_commit += 1
            if len(start_payload) != expected_size or _sha256(start_payload) != expected_sha256:
                mismatches.append(
                    {
                        "path": relative_name,
                        "reason": "start-commit size or SHA-256 mismatch",
                    }
                )

    recorded_count = manifest_value.get("file_count")
    if recorded_count != EXPECTED_FROZEN_FILE_COUNT or len(entries) != recorded_count:
        mismatches.append({"reason": "freeze manifest must contain exactly 34 files"})
    manifest_roots = manifest_value.get("frozen_roots")
    if manifest_roots != list(EXPECTED_FROZEN_ROOTS):
        mismatches.append({"reason": "frozen_roots differ from the fixed release contract"})
    if tuple(sorted(seen)) != expected_paths:
        mismatches.append({"reason": "manifest path set differs from the start-tree freeze"})
    if len(expected_paths) != EXPECTED_FROZEN_FILE_COUNT:
        mismatches.append({"reason": "start-tree freeze does not expand to exactly 34 files"})
    if manifest_base_commit != EXPECTED_START_COMMIT:
        mismatches.append({"reason": "manifest base commit differs from fixed start"})
    if resolved_commit != EXPECTED_START_COMMIT:
        mismatches.append({"reason": "fixed start commit did not resolve canonically"})
    if (
        resolved_tree != EXPECTED_START_TREE
        or manifest_value.get("base_tree") != EXPECTED_START_TREE
    ):
        mismatches.append({"reason": "base tree differs from fixed start tree"})
    current_path_set = set(current_paths)
    expected_path_set = set(expected_paths)
    current_extra_paths = sorted(current_path_set - expected_path_set)
    current_missing_paths = sorted(expected_path_set - current_path_set)
    if current_extra_paths or current_missing_paths:
        mismatches.append({"reason": "current frozen-root path set differs from start tree"})

    changed_paths: list[str] = []
    if expected_paths:
        try:
            changed_paths = sorted(
                line
                for line in _git_text(
                    repository,
                    "diff",
                    "--name-only",
                    "--no-renames",
                    resolved_commit,
                    "--",
                    *pathspecs,
                ).splitlines()
                if line
            )
        except subprocess.SubprocessError:
            mismatches.append({"reason": "git diff failed"})
    if changed_paths:
        mismatches.append({"reason": "frozen paths differ from start commit"})

    return {
        "status": "pass" if not mismatches else "fail",
        "manifest": label,
        "start_commit": resolved_commit,
        "start_tree": resolved_tree,
        "expected_file_count": EXPECTED_FROZEN_FILE_COUNT,
        "manifest_file_count": recorded_count,
        "start_tree_frozen_file_count": len(expected_paths),
        "current_frozen_file_count": len(current_paths),
        "current_frozen_extra_paths": current_extra_paths,
        "current_frozen_missing_paths": current_missing_paths,
        "checked_worktree_count": checked_worktree,
        "checked_start_commit_count": checked_start_commit,
        "git_diff_changed_paths": changed_paths,
        "mismatches": mismatches,
        "algorithm": "SHA-256 over exact bytes; no tolerance or normalization",
    }


def audit_release_metadata(repository: Path) -> dict[str, Any]:
    """Confirm the distribution/workbench version without changing frozen packages."""

    import ftir2dcos
    import ftir_baseline
    import ftir_workbench

    try:
        project = tomllib.loads((repository / "pyproject.toml").read_text("utf-8"))[
            "project"
        ]
        distribution_version = str(project["version"])
    except (KeyError, OSError, TypeError, ValueError, tomllib.TOMLDecodeError):
        distribution_version = "unavailable"
    try:
        module_sources = _runtime_module_sources(repository)
    except RuntimeError:
        module_sources = {}
    checks = {
        "runtime_modules_from_repository": set(module_sources)
        == {"ftir_baseline", "ftir2dcos", "ftir_workbench"},
        "distribution_version_is_0_2_5": distribution_version == "0.2.5",
        "workbench_version_is_0_2_5": ftir_workbench.__version__ == "0.2.5",
        "baseline_package_version_frozen": ftir_baseline.__version__ == "0.1.0",
        "twodcos_package_version_frozen": ftir2dcos.__version__ == "0.4.0",
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "pass" if not failed_checks else "fail",
        "distribution_version": distribution_version,
        "workbench_version": ftir_workbench.__version__,
        "baseline_package_version": ftir_baseline.__version__,
        "twodcos_package_version": ftir2dcos.__version__,
        "module_sources": module_sources,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def _synthetic_prepared() -> PreparedSpectralDataset:
    from ftir_workbench.fingerprints import prepared_data_sha256
    from ftir_workbench.models import PreparedSpectralDataset

    axis = np.linspace(1800.0, 1000.0, 81, dtype=np.float64)
    peak = np.exp(-0.5 * ((axis - 1540.0) / 38.0) ** 2)
    shoulder = np.exp(-0.5 * ((axis - 1210.0) / 27.0) ** 2)
    ripple = 0.012 * np.sin(np.arange(axis.size, dtype=np.float64) * 1.8)
    spectra = np.vstack(
        [
            0.08 + scale * peak + (0.12 - scale / 10.0) * shoulder + phase * ripple
            for scale, phase in ((0.25, 1.0), (0.38, -0.8), (0.53, 1.3), (0.67, -1.1))
        ]
    )
    perturbation = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    labels = ("step-0", "step-1", "step-2", "step-3")
    digest = prepared_data_sha256(axis, perturbation, labels, spectra)
    return PreparedSpectralDataset(
        wavenumber=axis,
        perturbation=perturbation,
        perturbation_labels=labels,
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="synthetic-v0.2.5-release-audit",
        source_sha256=_sha256(b"synthetic-v0.2.5-release-audit"),
        baseline_run_id="synthetic-v0.2.5-release-audit",
        baseline_fingerprint="b" * 64,
        prepared_data_sha256=digest,
        original_axis_direction="descending",
        current_axis_direction="descending",
        perturbation_order_policy="preserve_file_order",
        baseline_recipe={
            "prepared_data_contract": {
                "source_channel": "synthetic release-audit Prepared",
                "scientific_normalization": False,
            }
        },
        baseline_qc={"all_checks_passed": True},
        warnings=(),
    )


def _prepared_exact(first: PreparedSpectralDataset, second: PreparedSpectralDataset) -> bool:
    return bool(
        np.array_equal(first.wavenumber, second.wavenumber)
        and np.array_equal(first.perturbation, second.perturbation)
        and np.array_equal(first.spectra, second.spectra)
        and first.perturbation_labels == second.perturbation_labels
        and first.to_metadata_dict() == second.to_metadata_dict()
    )


def _build_smoothing_fixture() -> tuple[
    PreparedSpectralDataset,
    Any,
    PreparedSpectralDataset,
    bytes,
]:
    from ftir_workbench.post_baseline_smoothing import PostBaselineSmoothingConfig
    from ftir_workbench.services.smoothing_service import PostBaselineSmoothingService

    parent = _synthetic_prepared()
    service = PostBaselineSmoothingService()
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="gaussian",
        gaussian_sigma_points=1.0,
        gaussian_truncate=4.0,
        convolution_mode="reflect",
    )
    result, child = service.apply(parent, config)
    return parent, result, child, service.build_bundle(result, child)


def _twodcos_config() -> TwoDCOSConfig:
    from ftir_workbench.config import TwoDCOSConfig, TwoDCOSRange

    return TwoDCOSConfig(
        ranges=(
            TwoDCOSRange(1800.0, 1450.0, "upper"),
            TwoDCOSRange(1350.0, 1000.0, "lower"),
        ),
        convention="canonical",
        nonuniform_perturbation_policy="allow",
        cross_range_enabled=True,
    )


def _load_twodcos_source(bundle: bytes) -> PreparedSpectralDataset:
    from ftir_workbench.export import load_prepared

    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        return load_prepared(
            (
                archive.read("source_prepared_spectrum.csv"),
                archive.read("source_prepared_spectrum.meta.json"),
            )
        )


def audit_smoothing_workflow(
    *,
    repository: Path | None = None,
    smoothing_bundle_override: bytes | None = None,
) -> dict[str, Any]:
    """Audit smoothing roundtrip and both Prepared-only 2D branches."""

    _configure_private_runtime()
    repository_value = (
        Path(__file__).resolve().parents[1]
        if repository is None
        else repository.resolve()
    )
    module_sources = _runtime_module_sources(repository_value)

    from ftir_workbench.export import (
        build_twodcos_bundle,
        load_prepared,
        verify_twodcos_bundle,
    )
    from ftir_workbench.services.twodcos_service import TwoDCOSWorkflowService
    from ftir_workbench.smoothing_export import verify_smoothing_bundle

    parent, smoothing_result, child, generated_bundle = _build_smoothing_fixture()
    smoothing_bundle = (
        generated_bundle if smoothing_bundle_override is None else smoothing_bundle_override
    )
    checks: dict[str, bool] = {
        "smoothing_bundle_verified": verify_smoothing_bundle(smoothing_bundle),
        "smoothing_child_hash_changed": (
            child.prepared_data_sha256 != parent.prepared_data_sha256
        ),
        "smoothing_residual_identity": np.array_equal(
            smoothing_result.removed_component,
            parent.spectra - smoothing_result.smoothed_spectra,
        ),
    }
    try:
        reloaded_child = load_prepared(smoothing_bundle)
    except (OSError, TypeError, ValueError, zipfile.BadZipFile):
        checks["smoothing_child_exact_reload"] = False
    else:
        checks["smoothing_child_exact_reload"] = _prepared_exact(reloaded_child, child)

    config = _twodcos_config()
    twodcos_service = TwoDCOSWorkflowService()
    unsmoothed = twodcos_service.compute(parent, config)
    smoothed = twodcos_service.compute(child, config)
    checks.update(
        {
            "unsmoothed_self_cross_complete": (
                len(unsmoothed.homo_results) == 2
                and len(unsmoothed.cross_results) == 1
                and unsmoothed.all_checks_passed
            ),
            "smoothed_self_cross_complete": (
                len(smoothed.homo_results) == 2
                and len(smoothed.cross_results) == 1
                and smoothed.all_checks_passed
            ),
            "twodcos_parent_hashes_exact": (
                unsmoothed.parent_prepared_data_sha256
                == parent.prepared_data_sha256
                and smoothed.parent_prepared_data_sha256
                == child.prepared_data_sha256
            ),
            "twodcos_fingerprints_distinct": (
                unsmoothed.twodcos_fingerprint != smoothed.twodcos_fingerprint
            ),
            "smoothed_matrices_differ_from_unsmoothed": any(
                not np.array_equal(left.synchronous, right.synchronous)
                for left, right in zip(
                    unsmoothed.homo_results,
                    smoothed.homo_results,
                    strict=True,
                )
            ),
        }
    )
    cross = smoothed.cross_results[0].result
    checks["smoothed_cross_reverse_identities"] = bool(
        np.array_equal(cross.reverse_synchronous, cross.synchronous.T)
        and np.array_equal(cross.reverse_asynchronous, -cross.asynchronous.T)
    )

    twodcos_bundle = build_twodcos_bundle(child, smoothed, config)
    checks["smoothed_twodcos_bundle_verified"] = verify_twodcos_bundle(twodcos_bundle)
    try:
        embedded_child = _load_twodcos_source(twodcos_bundle)
    except (KeyError, OSError, TypeError, ValueError, zipfile.BadZipFile):
        checks["smoothed_twodcos_source_child_exact"] = False
    else:
        checks["smoothed_twodcos_source_child_exact"] = _prepared_exact(
            embedded_child,
            child,
        )

    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "pass" if not failed_checks else "fail",
        "input": "deterministic synthetic Prepared; no raw-data file was read",
        "module_sources": module_sources,
        "shape": [parent.n_spectra, parent.n_points],
        "parent_prepared_data_sha256": parent.prepared_data_sha256,
        "child_prepared_data_sha256": child.prepared_data_sha256,
        "smoothing_fingerprint": smoothing_result.smoothing_fingerprint,
        "unsmoothed_twodcos_fingerprint": unsmoothed.twodcos_fingerprint,
        "smoothed_twodcos_fingerprint": smoothed.twodcos_fingerprint,
        "checks": checks,
        "failed_checks": failed_checks,
    }


def _generate_v021_artifacts(source_root: Path, artifact_dir: Path) -> dict[str, Any]:
    """Generate baseline/2D/project artifacts with only the v0.2.1 snapshot."""

    import ftir2dcos
    import ftir_baseline
    import ftir_workbench
    from ftir2dcos.twodcos import compute_2dcos
    from ftir_baseline.config import PipelineConfig
    from ftir_baseline.models import SpectrumSet
    from ftir_workbench.export import build_project_bundle, build_twodcos_bundle
    from ftir_workbench.services import BaselineWorkflowService

    module_sources = _assert_modules_from_snapshot(
        source_root,
        [ftir_baseline, ftir2dcos, ftir_workbench],
    )
    if ftir_workbench.__version__ != "0.2.1":
        raise RuntimeError("start snapshot does not report ftir_workbench 0.2.1")

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
        source_name="synthetic-v0.2.1-release-audit",
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
    prepared = service.prepared(baseline_run_id="v0.2.1-release-audit")
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
        # Keep the legacy audit helper's documented project-config sentinel;
        # the generator version and Git snapshot prove this is a v0.2.1 artifact.
        {"project_name": "v0.2.0 synthetic compatibility audit"},
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "baseline": ("baseline_run_v021.zip", baseline_bundle),
        "twodcos": ("twodcos_run_v021.zip", twodcos_bundle),
        "project": ("project_v021.ftirw", project_bundle),
    }
    for name, payload in artifacts.values():
        (artifact_dir / name).write_bytes(payload)
    np.savez(
        artifact_dir / "prepared_expected_v020.npz",
        wavenumber=prepared.wavenumber,
        perturbation=prepared.perturbation,
        spectra=prepared.spectra,
    )
    metadata = {
        "generator": "exact exported v0.2.1 Git snapshot",
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
            kind: {
                "file": name,
                "size_bytes": len(payload),
                "sha256": _sha256(payload),
            }
            for kind, (name, payload) in artifacts.items()
        },
    }
    (artifact_dir / "legacy_generation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def audit_v021_bundle_compatibility(
    repository: Path,
    start_commit: str,
) -> dict[str, Any]:
    """Generate with the exact v0.2.1 start tree and verify with current code."""

    resolved_commit = _git_text(
        repository,
        "rev-parse",
        "--verify",
        f"{start_commit}^{{commit}}",
    )
    archive_payload = _git_bytes(repository, "archive", "--format=tar", resolved_commit)
    with tempfile.TemporaryDirectory(prefix="ftir-v025-release-audit-") as temporary:
        temporary_root = Path(temporary)
        snapshot = temporary_root / "v021-source"
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
                raise RuntimeError("git archive contains unsafe members")
            if "filter" in inspect.signature(archive.extractall).parameters:
                archive.extractall(snapshot, members=members, filter="data")
            else:  # pragma: no cover - compatibility with early Python 3.11
                archive.extractall(snapshot, members=members)
        command = (
            sys.executable,
            str(Path(__file__).resolve()),
            "_generate-v021",
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
            "start_commit": resolved_commit,
            "generation_environment": "temporary Git archive; artifacts deleted",
        }
    )
    return result


def audit_v021_compatibility_and_privacy(repository: Path) -> dict[str, Any]:
    """Audit exact v0.2.1 bundles and reuse the privacy audit safely."""

    compatibility = audit_v021_bundle_compatibility(
        repository,
        EXPECTED_START_COMMIT,
    )
    compatibility_checks = compatibility.get("checks", {})
    if not isinstance(compatibility_checks, dict):
        compatibility_checks = {}
    privacy = audit_repository_privacy(repository)
    ignore_checks = privacy.get("ignore_rule_checks", {})
    if not isinstance(ignore_checks, dict):
        ignore_checks = {}
    checks = {
        "exact_v021_bundle_audit_passed": compatibility.get("status") == "pass",
        "old_baseline_reload": all(
            bool(compatibility_checks.get(name))
            for name in (
                "baseline_legacy_manifest",
                "baseline_workbench_manifest",
                "baseline_prepared_exact_reload",
                "project_nested_baseline_prepared_exact_reload",
            )
        ),
        "old_twodcos_reload": all(
            bool(compatibility_checks.get(name))
            for name in (
                "twodcos_bundle",
                "twodcos_source_prepared_exact_reload",
                "project_nested_twodcos_source_prepared_exact_reload",
            )
        ),
        "old_project_reload": all(
            bool(compatibility_checks.get(name))
            for name in (
                "project_bundle",
                "project_nested_baseline_byte_identical",
                "project_nested_twodcos_byte_identical",
                "project_config_reload",
            )
        ),
        "repository_privacy": privacy.get("status") == "pass",
        "raw_and_output_ignore_rules": bool(ignore_checks) and all(ignore_checks.values()),
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "status": "pass" if not failed_checks else "fail",
        "reused_audit": "scripts/audit_v021_release.py",
        "legacy_base_commit": compatibility.get("start_commit"),
        "legacy_generator_version": compatibility.get("generation", {}).get(
            "base_workbench_version"
        ),
        "legacy_prepared_shape": compatibility.get("prepared_shape"),
        "checks": checks,
        "failed_checks": failed_checks,
        "privacy_scope": "Git index and ignore rules only; ignored spectrum contents not read",
    }


def _safe_check(name: str, function: Any) -> tuple[str, dict[str, Any]]:
    try:
        result = function()
    except Exception as error:
        result = {"status": "fail", "error_type": type(error).__name__}
    return name, result


def run_release_audit(
    repository: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Run every v0.2.5 release check and return a safe JSON object."""

    repository = repository.resolve()
    _configure_private_runtime()
    checks = dict(
        (
            _safe_check(
                "v021_science_freeze",
                lambda: audit_science_freeze(repository, manifest_path),
            ),
            _safe_check(
                "release_metadata",
                lambda: audit_release_metadata(repository),
            ),
            _safe_check(
                "smoothing_and_2d",
                lambda: audit_smoothing_workflow(repository=repository),
            ),
            _safe_check(
                "v021_compatibility_and_privacy",
                lambda: audit_v021_compatibility_and_privacy(repository),
            ),
        )
    )
    failed_checks = sorted(name for name, result in checks.items() if result["status"] != "pass")
    try:
        head_commit = _git_text(repository, "rev-parse", "HEAD")
    except subprocess.SubprocessError:
        head_commit = "unavailable"
    return {
        "schema_version": "1.0",
        "audit": "FTIR Spectral Workbench v0.2.5 local release audit",
        "status": "pass" if not failed_checks else "fail",
        "repository": repository.name,
        "audited_head_commit": head_commit,
        "checks": checks,
        "failed_checks": failed_checks,
        "privacy": {
            "absolute_paths_emitted": False,
            "raw_data_read": False,
            "fixture_scope": "deterministic synthetic Prepared only",
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the v0.2.1 science freeze, v0.2.5 smoothing/2D lineage, "
            "legacy bundle reload, and repository privacy."
        )
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _main_generate_v021(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    namespace = parser.parse_args(arguments)
    _generate_v021_artifacts(
        namespace.source_root.resolve(),
        namespace.artifact_dir.resolve(),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_generate-v021":
        return _main_generate_v021(arguments[1:])
    namespace = _build_parser().parse_args(arguments)
    repository = namespace.repository.resolve()
    manifest = namespace.manifest
    if manifest is not None and not manifest.is_absolute():
        manifest = repository / manifest
    summary = run_release_audit(repository, manifest_path=manifest)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if namespace.output is not None:
        output = namespace.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
