from __future__ import annotations

import json
from pathlib import Path

import pytest

from ftir_baseline.cli import main as baseline_main
from ftir_workbench.cli import build_parser, main
from ftir_workbench.export import verify_twodcos_bundle, verify_workbench_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMPORT_FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "import_compat"


def write_semicolon_decimal_comma_table(path: Path) -> None:
    path.write_text(
        "Instrument export\n"
        ";Wavenumber;0 min;5 min;\n"
        ";1002,0;0,10;0,20;\n"
        ";1001,0;0,11;0,21;\n"
        ";1000,0;0,12;0,22;\n",
        encoding="utf-8-sig",
    )


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


@pytest.mark.parametrize(
    "argv",
    [
        ["inspect", "raw.dat"],
        ["baseline", "raw.dat"],
        ["demo"],
    ],
)
def test_raw_input_commands_expose_text_import_options(argv: list[str]) -> None:
    arguments = build_parser().parse_args(
        [
            *argv,
            "--delimiter",
            "whitespace",
            "--decimal-mark",
            "dot",
            "--encoding",
            "cp1252",
            "--header",
            "absent",
            "--skip-rows",
            "4",
            "--no-trim-empty-edge-columns",
        ]
    )

    assert arguments.delimiter == "whitespace"
    assert arguments.decimal_mark == "dot"
    assert arguments.encoding == "cp1252"
    assert arguments.header == "absent"
    assert arguments.skip_rows == 4
    assert arguments.trim_empty_edge_columns is False


def test_cli_inspect_applies_explicit_text_import_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "semicolon-decimal-comma.dat"
    write_semicolon_decimal_comma_table(source)

    exit_code = main(
        [
            "inspect",
            str(source),
            "--unit",
            "absorbance",
            "--delimiter",
            "semicolon",
            "--decimal-mark",
            "comma",
            "--encoding",
            "utf-8-sig",
            "--header",
            "present",
            "--skip-rows",
            "1",
            "--trim-empty-edge-columns",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["spectra"] == 2
    assert summary["points"] == 3


def test_prepared_twodcos_command_does_not_receive_raw_import_options() -> None:
    arguments = build_parser().parse_args(
        ["twodcos", "prepared.csv", "--range", "1800:900"]
    )

    assert not hasattr(arguments, "delimiter")
    assert not hasattr(arguments, "decimal_mark")
    assert not hasattr(arguments, "encoding")


@pytest.mark.parametrize(
    "relative_path",
    [
        "legacy_wide.csv",
        "wide.tsv",
        "single.tab",
        "legacy_wide.txt",
        "legacy_series/0MIN.dpt",
        "single.asc",
        "single.dat",
        "single.xy",
    ],
    ids=("csv", "tsv", "tab", "txt", "dpt", "asc", "dat", "xy"),
)
def test_supported_extension_cli_entry_points(
    relative_path: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = IMPORT_FIXTURES / relative_path

    assert (
        main(
            [
                "inspect",
                str(source),
                "--unit",
                "absorbance",
                "--no-sort-by-perturbation",
            ]
        )
        == 0
    )
    inspect_summary = json.loads(capsys.readouterr().out)
    assert inspect_summary["points"] >= 5

    output = tmp_path / f"{source.suffix.lstrip('.')}-baseline.zip"
    assert (
        main(
            [
                "baseline",
                str(source),
                "--unit",
                "absorbance",
                "--baseline-high",
                "1800",
                "--baseline-low",
                "1400",
                "--coarse-method",
                "none",
                "--fine-method",
                "none",
                "--series-mode",
                "independent_locked",
                "--no-sort-by-perturbation",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    baseline_summary = json.loads(capsys.readouterr().out)
    assert baseline_summary["state"] == "baseline_completed"
    assert baseline_summary["manifest_verified"] is True

    assert baseline_main(["inspect", str(source), "--unit", "absorbance"]) == 0
    legacy_summary = json.loads(capsys.readouterr().out)
    assert legacy_summary["n_points"] >= 5
