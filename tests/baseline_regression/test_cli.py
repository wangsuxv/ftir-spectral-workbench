from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ftir_baseline.cli import main
from ftir_baseline.export import verify_export_manifest


def write_wide_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Wavenumber", "0 min", "1 min"])
        for point in range(1800, 899, -5):
            baseline = 0.01 + (1800 - point) * 1e-5
            writer.writerow([point, baseline, baseline + 0.002])


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
