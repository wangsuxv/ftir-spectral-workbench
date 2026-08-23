from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ftir2dcos.config import PipelineConfig as LegacyPipelineConfig
from ftir2dcos.pipeline import run_pipeline as run_legacy_pipeline
from ftir_workbench.config import TwoDCOSConfig, TwoDCOSRange
from ftir_workbench.export import serialize_prepared
from ftir_workbench.fingerprints import prepared_data_sha256
from ftir_workbench.models import PreparedSpectralDataset
from ftir_workbench.services import TwoDCOSWorkflowService


def _prepared() -> PreparedSpectralDataset:
    axis = np.linspace(900.0, 1800.0, 37, dtype=np.float64)
    perturbation = np.array([0.0, 1.0, 3.0, 6.0, 10.0], dtype=np.float64)
    labels = tuple(f"{value:g}MIN" for value in perturbation)
    peak = np.exp(-0.5 * ((axis - 1630.0) / 35.0) ** 2)
    shoulder = np.exp(-0.5 * ((axis - 1210.0) / 28.0) ** 2)
    spectra = np.vstack(
        [
            0.02 + (0.3 + value / 30.0) * peak - (value / 50.0) * shoulder
            for value in perturbation
        ]
    )
    digest = prepared_data_sha256(axis, perturbation, labels, spectra)
    return PreparedSpectralDataset(
        wavenumber=axis,
        perturbation=perturbation,
        perturbation_labels=labels,
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="memory-prepared",
        source_sha256="1" * 64,
        baseline_run_id="baseline-legacy-parity",
        baseline_fingerprint="2" * 64,
        prepared_data_sha256=digest,
        original_axis_direction="ascending",
        current_axis_direction="ascending",
        perturbation_order_policy="preserve_file_order",
        baseline_recipe={"already_corrected": True},
        baseline_qc={"all_checks_passed": True},
        warnings=(),
    )


@pytest.mark.parametrize("convention", ["canonical", "2dpy_compatible"])
def test_in_memory_prepared_chain_matches_legacy_csv_baseline_none(
    tmp_path: Path,
    convention: str,
) -> None:
    """The migration changes orchestration, not synchronous/asynchronous values."""

    prepared = _prepared()
    csv_path = tmp_path / "旧链 输入 含空格.csv"
    csv_path.write_bytes(serialize_prepared(prepared).csv_bytes)
    legacy = run_legacy_pipeline(
        csv_path,
        LegacyPipelineConfig(
            low_wavenumber=900.0,
            high_wavenumber=1800.0,
            input_intensity_unit="absorbance",
            perturbation_order="preserve_file_order",
            convention=convention,
        ),
    )
    current = TwoDCOSWorkflowService().compute(
        prepared,
        TwoDCOSConfig(
            ranges=(TwoDCOSRange(1800.0, 900.0, "full"),),
            convention=convention,
            nonuniform_perturbation_policy="allow",
        ),
    ).homo_results[0].result

    np.testing.assert_array_equal(current.synchronous, legacy.twodcos.synchronous)
    np.testing.assert_array_equal(current.asynchronous, legacy.twodcos.asynchronous)
    np.testing.assert_array_equal(current.dynamic, legacy.twodcos.dynamic)
    np.testing.assert_array_equal(current.row_wavenumber, legacy.twodcos.row_wavenumber)
    np.testing.assert_array_equal(
        current.column_wavenumber,
        legacy.twodcos.column_wavenumber,
    )
