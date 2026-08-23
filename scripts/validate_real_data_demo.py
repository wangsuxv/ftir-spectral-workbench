"""Validate the real-data bundles and recompute their prepared-only 2D lineage."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from ftir_baseline.export import verify_export_manifest
from ftir_workbench.config import TwoDCOSConfig, TwoDCOSRange
from ftir_workbench.export import (
    load_prepared,
    verify_project_bundle,
    verify_twodcos_bundle,
)
from ftir_workbench.services.twodcos_service import TwoDCOSWorkflowService


def _matrix_from_csv(
    archive: zipfile.ZipFile,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the exported row axis, column axis, and matrix without reordering."""

    payload = archive.read(name)
    first_line = payload.splitlines()[0].decode("utf-8")
    column_axis = np.fromstring(first_line.split(",", maxsplit=1)[1], sep=",")
    table = np.loadtxt(io.BytesIO(payload), delimiter=",", skiprows=1, ndmin=2)
    return table[:, 0], column_axis, table[:, 1:]


def _assert_exported_matrix(
    archive: zipfile.ZipFile,
    name: str,
    expected_row_axis: np.ndarray,
    expected_column_axis: np.ndarray,
    expected_matrix: np.ndarray,
) -> None:
    row_axis, column_axis, matrix = _matrix_from_csv(archive, name)
    np.testing.assert_array_equal(row_axis, expected_row_axis)
    np.testing.assert_array_equal(column_axis, expected_column_axis)
    np.testing.assert_array_equal(matrix, expected_matrix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    return parser


def main() -> int:
    directory = build_parser().parse_args().directory.expanduser().resolve()
    baseline_path = directory / "baseline_run.zip"
    twodcos_path = directory / "twodcos_run.zip"
    project_path = directory / "project.ftirw"
    if not verify_export_manifest(baseline_path):
        raise RuntimeError("baseline manifest verification failed")
    if not verify_twodcos_bundle(twodcos_path):
        raise RuntimeError("2D manifest verification failed")
    if not verify_project_bundle(project_path):
        raise RuntimeError("project archive verification failed")

    prepared = load_prepared(baseline_path)
    config = TwoDCOSConfig(
        ranges=(
            TwoDCOSRange(1736.0, 1509.0, "amide_1736_1509"),
            TwoDCOSRange(1250.0, 1140.0, "fingerprint_1250_1140"),
        ),
        convention="2dpy_compatible",
        nonuniform_perturbation_policy="warn",
        cross_range_enabled=True,
    )
    result = TwoDCOSWorkflowService().compute(prepared, config)
    actual_homo_shapes = tuple(item.result.synchronous.shape for item in result.homo_results)
    if len(result.homo_results) != len(config.ranges):
        raise RuntimeError("one or more configured self-range blocks are missing")
    for item in result.homo_results:
        expected_shape = (
            item.result.row_wavenumber.size,
            item.result.column_wavenumber.size,
        )
        if item.result.synchronous.shape != expected_shape:
            raise RuntimeError("a self matrix does not match its exported axes")
    if len(result.cross_results) != 1:
        raise RuntimeError("expected exactly one unique cross-range block")
    cross_shape = result.cross_results[0].result.synchronous.shape
    expected_cross_shape = (
        result.cross_results[0].result.row_wavenumber.size,
        result.cross_results[0].result.column_wavenumber.size,
    )
    if cross_shape != expected_cross_shape:
        raise RuntimeError("the cross matrix does not match its exported axes")
    if not result.all_checks_passed:
        raise RuntimeError("one or more 2D numerical checks failed")

    with zipfile.ZipFile(twodcos_path) as archive:
        reloaded = load_prepared(
            (
                archive.read("source_prepared_spectrum.csv"),
                archive.read("source_prepared_spectrum.meta.json"),
            )
        )
        manifest = json.loads(archive.read("manifest.json"))
        for index, item in enumerate(result.homo_results, start=1):
            prefix = f"ranges/range_{index:02d}"
            _assert_exported_matrix(
                archive,
                f"{prefix}/synchronous_matrix.csv",
                item.result.row_wavenumber,
                item.result.column_wavenumber,
                item.result.synchronous,
            )
            _assert_exported_matrix(
                archive,
                f"{prefix}/asynchronous_matrix.csv",
                item.result.row_wavenumber,
                item.result.column_wavenumber,
                item.result.asynchronous,
            )
        for index, item in enumerate(result.cross_results, start=1):
            prefix = f"cross_ranges/cross_{index:02d}"
            _assert_exported_matrix(
                archive,
                f"{prefix}/synchronous_matrix.csv",
                item.result.row_wavenumber,
                item.result.column_wavenumber,
                item.result.synchronous,
            )
            _assert_exported_matrix(
                archive,
                f"{prefix}/asynchronous_matrix.csv",
                item.result.row_wavenumber,
                item.result.column_wavenumber,
                item.result.asynchronous,
            )
    np.testing.assert_array_equal(reloaded.wavenumber, prepared.wavenumber)
    np.testing.assert_array_equal(reloaded.perturbation, prepared.perturbation)
    np.testing.assert_array_equal(reloaded.spectra, prepared.spectra)
    if reloaded.perturbation_labels != prepared.perturbation_labels:
        raise RuntimeError("prepared perturbation labels changed during bundle round-trip")

    summary = {
        "baseline_manifest_verified": True,
        "twodcos_manifest_verified": True,
        "project_manifest_and_nested_bundles_verified": True,
        "prepared_shape": list(prepared.spectra.shape),
        "prepared_axis_direction": prepared.current_axis_direction,
        "perturbation_first_last": [
            float(prepared.perturbation[0]),
            float(prepared.perturbation[-1]),
        ],
        "prepared_roundtrip_exact": True,
        "exported_matrices_match_recomputation_exactly": True,
        "self_matrix_shapes": [list(shape) for shape in actual_homo_shapes],
        "cross_matrix_shape": list(cross_shape),
        "all_2d_qc_checks_passed": True,
        "parent_baseline_run_id": manifest["parent_baseline_run_id"],
        "parent_baseline_fingerprint": manifest["parent_baseline_fingerprint"],
        "parent_prepared_data_sha256": manifest["parent_prepared_data_sha256"],
        "warnings": list(result.warnings),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
