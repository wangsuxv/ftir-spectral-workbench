from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ftir2dcos.config import BaselineConfig, PipelineConfig
from ftir2dcos.export import (
    create_unique_run_directory,
    export_run,
    sha256_file,
    write_matrix_csv,
    write_spectra_csv,
)
from ftir2dcos.twodcos import TwoDCOSResult, compute_2dcos


@dataclass(frozen=True)
class ExampleDataset:
    wavenumber: np.ndarray
    perturbation: np.ndarray
    perturbation_labels: tuple[str, ...]
    spectra: np.ndarray
    intensity_unit: str = "absorbance"
    source_name: str = "example.csv"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExampleResult:
    imported: ExampleDataset
    selected_raw: ExampleDataset
    baselines: np.ndarray
    baseline_corrected: ExampleDataset
    processed: ExampleDataset
    twodcos: TwoDCOSResult
    warnings: tuple[str, ...]

    @property
    def qc_metrics(self) -> dict[str, float | bool]:
        return self.twodcos.qc_metrics


def _example_result() -> ExampleResult:
    wavenumber = np.linspace(1509.0, 1736.0, 10)
    perturbation = np.asarray([0.0, 1.0, 3.0], dtype=np.float64)
    labels = ("0 min", "1 min", "3 min")
    raw = np.vstack([0.08 + np.sin(wavenumber / 40.0 + phase) * 0.02 for phase in (0.0, 0.6, 1.2)])
    baselines = np.vstack(
        [0.01 + index * 0.001 + 2e-5 * (wavenumber - wavenumber.mean()) for index in range(3)]
    )
    corrected = raw - baselines
    processed = corrected.copy()
    twodcos = compute_2dcos(processed, wavenumber, convention="canonical")

    def dataset(values: np.ndarray, metadata: dict[str, object] | None = None) -> ExampleDataset:
        return ExampleDataset(
            wavenumber,
            perturbation,
            labels,
            values,
            metadata={} if metadata is None else metadata,
        )

    return ExampleResult(
        imported=dataset(
            raw,
            metadata={
                "source_files": [{"name": "uploaded.csv", "sha256": "a" * 64, "size_bytes": 123}],
                "source_sha256": "a" * 64,
                "original_wavenumber_direction": "ascending",
            },
        ),
        selected_raw=dataset(raw),
        baselines=baselines,
        baseline_corrected=dataset(corrected),
        processed=dataset(processed),
        twodcos=twodcos,
        warnings=("Nonuniform perturbation spacing detected; index order is used.",),
    )


def test_write_spectra_csv_is_origin_ready(tmp_path: Path) -> None:
    axis = np.asarray([1736.0, 1720.5, 1700.0])
    spectra = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    path = write_spectra_csv(tmp_path / "spectra.csv", axis, spectra, ("0 min", "5 min"))
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["wavenumber_cm-1", "0 min", "5 min"]
    assert rows[1] == ["1736", "1", "4"]
    assert rows[-1] == ["1700", "3", "6"]


def test_write_matrix_csv_preserves_row_and_column_orientation(tmp_path: Path) -> None:
    row_axis = np.asarray([1736.0, 1700.0])
    column_axis = np.asarray([1600.0, 1509.0])
    matrix = np.asarray([[1.0, -2.0], [3.0, -4.0]])
    path = write_matrix_csv(tmp_path / "matrix.csv", matrix, row_axis, column_axis)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["wavenumber_cm-1", "1600", "1509"],
        ["1736", "1", "-2"],
        ["1700", "3", "-4"],
    ]


def test_unique_run_directories_never_overwrite(tmp_path: Path) -> None:
    fixed = datetime(2026, 8, 21, 12, 30, 45)
    first = create_unique_run_directory(tmp_path, now=fixed)
    marker = first / "keep.txt"
    marker.write_text("do not overwrite", encoding="utf-8")
    second = create_unique_run_directory(tmp_path, now=fixed)

    assert first.name == "run_20260821_123045"
    assert second.name == "run_20260821_123045_001"
    assert marker.read_text(encoding="utf-8") == "do not overwrite"


def test_export_run_creates_complete_manifest_csv_figures_and_zip(tmp_path: Path) -> None:
    result = _example_result()
    input_path = tmp_path / "source.csv"
    original_input = b"original raw input must remain unchanged\n"
    input_path.write_bytes(original_input)
    figures_before = set(plt.get_fignums())

    run_directory = export_run(
        result,
        PipelineConfig(
            low_wavenumber=1509,
            high_wavenumber=1736,
            baseline=BaselineConfig(method="offset"),
            contour_levels=11,
            display_percentile=75.0,
            convention="canonical",
        ),
        tmp_path / "results",
    )

    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    output_files = set(manifest["output_files"])
    actual_files = {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file()
    }
    assert output_files == actual_files
    assert manifest["input_file_name"] == "uploaded.csv"
    assert manifest["input_sha256"] == "a" * 64
    assert manifest["original_data_shape"] == [3, 10]
    assert manifest["final_data_shape"] == [3, 10]
    assert manifest["convention"] == "canonical"
    assert manifest["matrix_axes"]["row_variable"] == "nu1"
    assert manifest["matrix_axes"]["column_variable"] == "nu2"
    assert manifest["plot_display"]["display_percentile"] == 75.0
    assert manifest["nonuniform_perturbation_warning"] is not None
    assert input_path.read_bytes() == original_input

    sync_csv = run_directory / "data/09_synchronous_matrix.csv"
    with sync_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    exported_sync = np.asarray([[float(value) for value in row[1:]] for row in rows[1:]])
    assert np.allclose(exported_sync, result.twodcos.synchronous, rtol=0.0, atol=0.0)

    assert (run_directory / "figures/raw_spectra.png").read_bytes().startswith(b"\x89PNG")
    assert (run_directory / "figures/synchronous_2dcos.pdf").read_bytes().startswith(b"%PDF")
    with zipfile.ZipFile(run_directory / "run_bundle.zip") as archive:
        assert set(archive.namelist()) == output_files - {"run_bundle.zip"}
        assert archive.testzip() is None
    assert set(plt.get_fignums()) == figures_before


def test_sha256_file_reads_without_modifying_input(tmp_path: Path) -> None:
    source = tmp_path / "raw.dat"
    content = b"immutable source bytes\n"
    source.write_bytes(content)
    assert sha256_file(source) == "bf782e46fcf08ea44f997a629201422491adfc7da33f84daa479a73d677d941d"
    assert source.read_bytes() == content
