from __future__ import annotations

import json
from pathlib import Path

import pytest

from ftir_workbench.cli import main
from ftir_workbench.export import verify_twodcos_bundle, verify_workbench_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_cli_baseline_then_prepared_only_twodcos(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    source = PROJECT_ROOT / "examples" / "baseline" / "example_absorbance.csv"
    baseline_path = tmp_path / "baseline.zip"

    exit_code = main(
        [
            "baseline",
            str(source),
            "--unit",
            "absorbance",
            "--baseline-high",
            "1800",
            "--baseline-low",
            "900",
            "--coarse-method",
            "none",
            "--series-mode",
            "independent_locked",
            "--output",
            str(baseline_path),
        ]
    )

    assert exit_code == 0
    baseline_summary = json.loads(capsys.readouterr().out)
    assert baseline_summary["state"] == "baseline_completed"
    assert baseline_summary["manifest_verified"] is True
    assert baseline_path.is_file()
    assert verify_workbench_manifest(baseline_path)

    twodcos_path = tmp_path / "twodcos.zip"
    exit_code = main(
        [
            "twodcos",
            str(baseline_path),
            "--range",
            "1800:900:full",
            "--output",
            str(twodcos_path),
        ]
    )

    assert exit_code == 0
    twodcos_summary = json.loads(capsys.readouterr().out)
    assert twodcos_summary["state"] == "twodcos_completed"
    assert twodcos_summary["self_results"] == 1
    assert twodcos_summary["cross_results"] == 0
    assert twodcos_summary["all_checks_passed"] is True
    assert twodcos_path.is_file()
    assert verify_twodcos_bundle(twodcos_path)


def test_cli_inspect_reports_explicit_unit(capsys) -> None:  # type: ignore[no-untyped-def]
    source = PROJECT_ROOT / "examples" / "baseline" / "example_percent_transmittance.csv"

    exit_code = main(["inspect", str(source), "--unit", "percent_transmittance"])

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["unit"] == "percent_transmittance"
    assert summary["spectra"] == 3


def test_cli_ftirw_verify_cannot_fall_back_to_outer_manifest_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    from ftir_workbench import cli

    project = tmp_path / "mismatched-project.ftirw"
    project.write_bytes(b"placeholder")
    monkeypatch.setattr(cli, "verify_project_bundle", lambda _source: False)
    monkeypatch.setattr(cli, "verify_workbench_manifest", lambda _source: True)

    exit_code = main(["verify", str(project)])

    assert exit_code == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["bundle_type"] == "project"
    assert summary["verified"] is False
