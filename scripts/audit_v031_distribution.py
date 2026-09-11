#!/usr/bin/env python3
"""Audit v0.3.1 wheel/sdist contents and already-installed wheel entry points.

This does not build, install, download, or modify dependencies. Installed checks
run in a temporary neutral directory with source PYTHONPATH removed, and the API
probe uses Python isolated mode. Documentation finality is deliberately outside
this audit; rebuild and repeat after documentation edits are complete.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import UTC, datetime
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

PACKAGES = ("ftir_baseline", "ftir2dcos", "ftir_workbench")
CONSOLE_SCRIPTS = {
    "ftir-workbench": "ftir_workbench.cli:main",
    "ftir-baseline": "ftir_baseline.cli:main",
    "ftir2dcos": "ftir2dcos.cli:main",
}
FORBIDDEN_PARTS = {"outputs", "rawdata", "raw_data", ".venv", "venv", "private", ".git", "__pycache__"}
SPECTRAL_SUFFIXES = {".csv", ".tsv", ".dpt", ".npy", ".npz", ".spc", ".spa", ".jdx", ".dx", ".xlsx", ".xls"}

INSTALLED_PROBE = r'''
import importlib, importlib.metadata, json, sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import ftir_baseline, ftir2dcos, ftir_workbench
from ftir_baseline.normalization import apply_normalization
from ftir_workbench.batch.normalization_adapter import normalize_spectral_arrays
from ftir_workbench.post_baseline_smoothing import smooth_spectral_arrays

assert importlib.metadata.version("ftir-spectral-workbench") == "0.3.1"
assert ftir_workbench.__version__ == "0.3.1"
origins = {}
for name in ("ftir_baseline", "ftir2dcos", "ftir_workbench",
             "ftir_workbench.batch.postprocessing", "ftir_workbench.batch.normalization_adapter",
             "ftir_workbench.batch.postprocessing_export", "ftir_workbench.batch.workspace",
             "ftir_workbench.post_baseline_smoothing", "ftir_workbench.export",
             "ftir_workbench.services.smoothing_service"):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    assert "site-packages" in path.parts, (name, str(path))
    origins[name] = "site-packages/" + str(path).split("site-packages/", 1)[1]
public = {
    "ftir_workbench.batch.importing": ("import_sources",),
    "ftir_workbench.batch.service": ("preview_coarse", "preview_fine"),
    "ftir_workbench.batch.state": ("confirm_coarse", "confirm_fine", "skip_fine", "get_ready_snapshot"),
    "ftir_workbench.batch.postprocessing": ("preview_branch", "confirm_branch", "get_branch", "update_draft", "copy_drafts", "preview_selected", "confirm_selected"),
    "ftir_workbench.batch.postprocessing_export": ("build_postprocessing_export", "verify_postprocessing_export", "can_export_postprocessing_wide"),
    "ftir_workbench.batch.workspace": ("save_batch_workspace", "load_batch_workspace"),
    "ftir_workbench.export": ("build_baseline_bundle", "build_project_bundle", "build_twodcos_bundle", "verify_project_bundle"),
    "ftir_workbench.post_baseline_smoothing": ("apply_post_baseline_smoothing", "smooth_spectral_arrays", "validate_smoothing_request"),
}
for module_name, names in public.items():
    module = importlib.import_module(module_name)
    for name in names:
        assert callable(getattr(module, name)), (module_name, name)
def forbidden(*args, **kwargs):
    raise AssertionError("Ordinary installed API must not create Prepared or run baseline/2D")
x = np.array([1000.0, 1002.0, 1004.0])
y = np.array([[2.0, 4.0, 6.0]])
with patch("ftir_baseline.pipeline.run_pipeline", side_effect=forbidden), \
     patch("ftir_workbench.batch.service.run_pipeline", side_effect=forbidden), \
     patch("ftir_workbench.models.PreparedSpectralDataset.__init__", side_effect=forbidden), \
     patch("ftir_workbench.services.twodcos_service.TwoDCOSWorkflowService.compute", side_effect=forbidden):
    core = apply_normalization(x, y, config={"method": "minmax_display"})
    np.testing.assert_array_equal(core.analysis_data, y)
    assert core.optional_normalized is None
    np.testing.assert_array_equal(core.view_data, [[0.0, 0.5, 1.0]])
    normalized = normalize_spectral_arrays(x, y, {"enabled": True, "method": "maximum"})
    np.testing.assert_array_equal(normalized.normalized_spectra, y / 6.0)
    assert normalized.quantity == "normalized_intensity"
    scaled = normalize_spectral_arrays(x, y, {"enabled": True, "method": "minmax_display"})
    np.testing.assert_array_equal(scaled.normalized_spectra, core.view_data)
    assert scaled.quantity == "minmax_scaled_intensity" and scaled.purpose == "display_only"
    np.testing.assert_array_equal(scaled.scale, [0.25])
    np.testing.assert_array_equal(scaled.offset, [-0.5])
    smoothed = smooth_spectral_arrays(x, y, {"enabled": True, "method": "moving_average", "moving_average_window_length": 3})
    np.testing.assert_array_equal(smoothed.wavenumber, x)
    assert smoothed.smoothed_spectra.shape == (1, 3)
    np.testing.assert_array_equal(smoothed.removed_component, y - smoothed.smoothed_spectra)
print(json.dumps({
    "isolated_mode": bool(sys.flags.isolated), "origins": origins,
    "distribution_version": importlib.metadata.version("ftir-spectral-workbench"),
    "workbench_version": ftir_workbench.__version__,
    "baseline_version": ftir_baseline.__version__, "twodcos_version": ftir2dcos.__version__,
    "callable_public_apis": public,
    "normalization_core_analysis_unchanged": True,
    "normalization_core_minmax_optional_normalized_is_none": True,
    "normalization_core_minmax_uses_view_data": True,
    "ordinary_maximum_uses_transformed_output": True,
    "ordinary_minmax_quantity_purpose_scale_offset_correct": True,
    "ordinary_smoothing_array_api_smoke": True,
    "baseline_prepared_2d_calls": 0,
}))
'''


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def wheel_audit(root: Path, wheel: Path) -> dict[str, Any]:
    source_files = {str(path.relative_to(root / "src")): path
                    for package in PACKAGES for path in (root / "src" / package).rglob("*.py")}
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        expected = set(source_files)
        wheel_py = {name for name in names if name.endswith(".py")}
        require(expected == wheel_py, f"Wheel Python module set mismatch; missing={sorted(expected - wheel_py)}, extra={sorted(wheel_py - expected)}")
        differences = [name for name, path in source_files.items() if archive.read(name) != path.read_bytes()]
        require(not differences, f"Wheel runtime modules differ from current source: {differences}")
        info = "ftir_spectral_workbench-0.3.1.dist-info/"
        metadata = BytesParser().parsebytes(archive.read(info + "METADATA"))
        require(metadata["Version"] == "0.3.1", "Wheel version is not 0.3.1")
        entrypoints = configparser.ConfigParser()
        entrypoints.read_string(archive.read(info + "entry_points.txt").decode())
        require(dict(entrypoints["console_scripts"]) == CONSOLE_SCRIPTS, "Console scripts changed or unexpected new command added")
        manifest = json.loads((root / "artifacts/v0.2.1_science_freeze_manifest.json").read_text())
        for record in manifest["files"]:
            path = record["path"].removeprefix("src/")
            require(sha(archive.read(path)) == record["sha256"], "Wheel frozen scientific file mismatch: " + path)
        installed = importlib.metadata.distribution("ftir-spectral-workbench")
        installed_mismatches = []
        for name in sorted(wheel_py):
            path = Path(installed.locate_file(name))
            if not path.is_file() or path.read_bytes() != archive.read(name):
                installed_mismatches.append(name)
        require(not installed_mismatches, f"Installed files differ from this wheel: {installed_mismatches}")
        legacy_modules = (
            "ftir_baseline/pipeline.py", "ftir2dcos/twodcos/engine.py", "ftir_workbench/export.py",
            "ftir_workbench/post_baseline_smoothing.py", "ftir_workbench/services/smoothing_service.py",
        )
        require(all(name in wheel_py for name in legacy_modules), "Legacy runtime capability is missing")
        return {
            "filename": wheel.name, "sha256": sha(wheel.read_bytes()), "bytes": wheel.stat().st_size,
            "version": metadata["Version"], "python_module_count": len(wheel_py),
            "batch_modules": sorted(name for name in wheel_py if name.startswith("ftir_workbench/batch/")),
            "package_module_counts": {package: sum(name.startswith(package + "/") for name in wheel_py) for package in PACKAGES},
            "console_scripts": dict(entrypoints["console_scripts"]),
            "all_runtime_modules_match_source": True, "all_installed_modules_match_wheel": True,
            "frozen_scientific_files_verified": len(manifest["files"]),
            "legacy_baseline_2d_project_smoothing_present": True,
            "legacy_runtime_anchors": list(legacy_modules),
        }


def sdist_audit(root: Path, path: Path) -> dict[str, Any]:
    prefix = "ftir_spectral_workbench-0.3.1/"
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        names = []
        unexpected_spectra = []
        for member in members:
            require(member.name.startswith(prefix), "Unexpected sdist root")
            relative = member.name.removeprefix(prefix)
            parts = PurePosixPath(relative).parts
            require(not set(parts) & FORBIDDEN_PARTS and ".." not in parts,
                    "Private/output/environment path in sdist: " + relative)
            require(not member.issym() and not member.islnk(), "Unexpected archive link: " + relative)
            if relative.startswith("data/original/"):
                require(relative == "data/original/README.md", "Experimental source file included in sdist")
            if (PurePosixPath(relative).suffix.lower() in SPECTRAL_SUFFIXES
                    and not relative.startswith(("examples/", "tests/fixtures/"))):
                unexpected_spectra.append(relative)
            names.append(relative)
        require(not unexpected_spectra, f"Spectrum-like files outside synthetic example/test locations: {unexpected_spectra}")
        for package in PACKAGES:
            for source in (root / "src" / package).rglob("*.py"):
                relative = str(source.relative_to(root))
                member_stream = archive.extractfile(prefix + relative)
                require(member_stream is not None and member_stream.read() == source.read_bytes(),
                        "Sdist runtime file missing or outdated: " + relative)
        require("ui/batch_postprocessing.py" in names and "ui/batch_workflow.py" in names,
                "Ordinary UI is missing from sdist")
        pkg_info = archive.extractfile(prefix + "PKG-INFO")
        require(pkg_info is not None and BytesParser().parsebytes(pkg_info.read())["Version"] == "0.3.1", "Sdist version mismatch")
        return {
            "filename": path.name, "sha256": sha(path.read_bytes()), "bytes": path.stat().st_size,
            "member_count": len(names), "version": "0.3.1", "runtime_sources_match_current": True,
            "outputs_rawdata_venv_private_paths_absent": True,
            "data_original_only_readme_placeholder": True,
            "spectrum_like_members_only_under_examples_or_test_fixtures": True,
            "ordinary_ui_present": True,
            "documentation_finality": "Not asserted; rebuild and rerun after final documentation edits.",
        }


def installed_audit(root: Path) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("MYPYPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    output: dict[str, Any] = {"source_pythonpath_removed": True, "cli_help": {}}
    with tempfile.TemporaryDirectory(prefix="ftir-installed-audit-") as neutral:
        probe = subprocess.run([sys.executable, "-I", "-c", INSTALLED_PROBE], cwd=neutral,
                               env=environment, capture_output=True, text=True, timeout=60)
        require(probe.returncode == 0, "Installed API probe failed: " + probe.stderr)
        output["installed_api"] = json.loads(probe.stdout)
        for command in CONSOLE_SCRIPTS:
            executable = Path(sys.executable).parent / command
            completed = subprocess.run([str(executable), "--help"], cwd=neutral, env=environment,
                                       capture_output=True, text=True, timeout=30)
            require(completed.returncode == 0 and "usage:" in completed.stdout.lower(), command + " --help failed")
            output["cli_help"][command] = {"command": command + " --help", "exit_code": completed.returncode,
                                           "stdout": completed.stdout, "stderr": completed.stderr}
    comparison_path = root / "artifacts/validation/v0.3.1/phase5/dependency_comparison.json"
    comparison = json.loads(comparison_path.read_text())
    checked = []
    for record in comparison["packages"]:
        actual = importlib.metadata.version(record["package"])
        require(actual == record["baseline"] == record["actual"], "Dependency version changed: " + record["package"])
        checked.append({"package": record["package"], "version": actual})
    output["dependency_versions_rechecked"] = checked
    output["dependency_count"] = len(checked)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report destination")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {"started_utc": datetime.now(UTC).isoformat(), "checks": {}, "failures": []}
    checks = {
        "wheel": lambda: wheel_audit(root, root / "dist/ftir_spectral_workbench-0.3.1-py3-none-any.whl"),
        "sdist": lambda: sdist_audit(root, root / "dist/ftir_spectral_workbench-0.3.1.tar.gz"),
        "installed": lambda: installed_audit(root),
    }
    for name, action in checks.items():
        try:
            report["checks"][name] = {"status": "pass", "details": action()}
        except Exception as exc:
            report["checks"][name] = {"status": "fail", "error": str(exc)}
            report["failures"].append(name)
    report["status"] = "fail" if report["failures"] else "pass"
    report["completed_utc"] = datetime.now(UTC).isoformat()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    text = text.replace(str(root), "<repository>").replace(str(Path.home()), "<home>")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
