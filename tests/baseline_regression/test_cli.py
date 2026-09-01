from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ftir_baseline.cli import build_parser, main
from ftir_baseline.export import verify_export_manifest


def write_wide_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Wavenumber", "0 min", "1 min"])
        for point in range(1800, 899, -5):
            baseline = 0.01 + (1800 - point) * 1e-5
            writer.writerow([point, baseline, baseline + 0.002])


def write_semicolon_decimal_comma_table(path: Path) -> None:
    path.write_text(
        "Instrument export\n"
        ";Wavenumber;0 min;5 min;\n"
        ";1002,0;0,10;0,20;\n"
        ";1001,0;0,11;0,21;\n"
        ";1000,0;0,12;0,22;\n",
        encoding="utf-8-sig",
    )


def test_cli_init_recipe(tmp_path: Path) -> None:
    recipe = tmp_path / "recipe.json"
    assert main(["init-recipe", "--unit", "absorbance", "--output", str(recipe)]) == 0
    assert '"input_unit": "absorbance"' in recipe.read_text(encoding="utf-8")


def test_cli_run_writes_verified_bundle(tmp_path: Path) -> None:
    source = tmp_path / "series.csv"
    output = tmp_path / "out"
    write_wide_csv(source)
    status = main(
        [
            "run",
            str(source),
            "--unit",
            "absorbance",
            "--series-mode",
            "independent_locked",
            "--coarse-method",
            "none",
            "--output",
            str(output),
        ]
    )
    assert status == 0
    bundles = list(output.glob("*.zip"))
    assert len(bundles) == 1
    assert verify_export_manifest(bundles[0])
    with zipfile.ZipFile(bundles[0]) as archive:
        assert "corrected_absorbance_for_2dcos.csv" in archive.namelist()
        assert "10_processing_recipe.json" in archive.namelist()


def test_cli_scan_emits_standard_json_with_anchor_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "series.csv"
    write_wide_csv(source)
    assert (
        main(
            [
                "scan",
                str(source),
                "--unit",
                "absorbance",
                "--arpls-grid",
                "6",
                "--asls-grid",
                "6",
                "--asls-p",
                "0.01",
                "--airpls-grid",
                "6",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out, parse_constant=lambda token: pytest.fail(token))
    assert payload["candidates"]
    assert "Anchor PCHIP" in {row["name"] for row in payload["candidates"]}
    assert all(np.isfinite(row["mean_anchor_error"]) for row in payload["candidates"])


@pytest.mark.parametrize(
    "argv",
    [
        ["inspect", "raw.dat", "--unit", "absorbance"],
        ["run", "raw.dat", "--unit", "absorbance"],
        ["scan", "raw.dat", "--unit", "absorbance"],
        ["demo"],
    ],
)
def test_raw_input_commands_expose_text_import_options(argv: list[str]) -> None:
    arguments = build_parser().parse_args(
        [
            *argv,
            "--delimiter",
            "semicolon",
            "--decimal-mark",
            "comma",
            "--encoding",
            "gb18030",
            "--header",
            "present",
            "--skip-rows",
            "2",
            "--no-trim-empty-edge-columns",
        ]
    )

    assert arguments.delimiter == "semicolon"
    assert arguments.decimal_mark == "comma"
    assert arguments.encoding == "gb18030"
    assert arguments.header == "present"
    assert arguments.skip_rows == 2
    assert arguments.trim_empty_edge_columns is False


def test_cli_inspect_applies_explicit_text_import_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "semicolon-decimal-comma.dat"
    write_semicolon_decimal_comma_table(source)

    assert (
        main(
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
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["shape"] == [2, 3]
    assert payload["metadata"]["delimiter_name"] == "semicolon"
    assert payload["metadata"]["decimal_mark"] == "comma"
    assert payload["metadata"]["encoding"] == "utf-8-sig"
    assert payload["metadata"]["header_mode"] == "present"
    assert payload["metadata"]["skip_rows"] == 1
    assert payload["metadata"]["trimmed_empty_edge_columns"] == 2
