from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ftir_workbench.fingerprints import prepared_data_sha256
from ftir_workbench.models import NormalizationState, PreparedSpectralDataset


def make_prepared(
    *,
    wavenumber: ArrayLike | None = None,
    spectra: ArrayLike | None = None,
    normalization_state: NormalizationState = "none",
    baseline_recipe: Mapping[str, Any] | None = None,
) -> PreparedSpectralDataset:
    axis = (
        np.linspace(1800.0, 1000.0, 41)
        if wavenumber is None
        else np.asarray(wavenumber, dtype=np.float64)
    )
    if spectra is None:
        position = np.linspace(-1.0, 1.0, axis.size)
        peak = np.exp(-0.5 * (position / 0.20) ** 2)
        shoulder = np.exp(-0.5 * ((position - 0.38) / 0.10) ** 2)
        ripple = 0.015 * np.sin(np.arange(axis.size, dtype=np.float64) * 2.1)
        matrix = np.vstack(
            (
                0.10 + 0.40 * peak + 0.08 * shoulder + ripple,
                0.15 + 0.55 * peak + 0.05 * shoulder - 0.7 * ripple,
                0.20 + 0.70 * peak + 0.03 * shoulder + 1.3 * ripple,
            )
        )
    else:
        matrix = np.asarray(spectra, dtype=np.float64)
    perturbation = np.array([0.0, 2.0, 5.0], dtype=np.float64)
    if matrix.shape[0] != perturbation.size:
        perturbation = np.arange(matrix.shape[0], dtype=np.float64)
    labels = tuple(f"spectrum-{index}" for index in range(matrix.shape[0]))
    digest = prepared_data_sha256(
        axis,
        perturbation,
        labels,
        matrix,
        normalization_state=normalization_state,
    )
    direction = "ascending" if axis[-1] > axis[0] else "descending"
    recipe = (
        {
            "prepared_data_contract": {
                "source_channel": "PipelineResult.analysis_data",
                "scientific_normalization": False,
            }
        }
        if baseline_recipe is None
        else dict(baseline_recipe)
    )
    return PreparedSpectralDataset(
        wavenumber=axis,
        perturbation=perturbation,
        perturbation_labels=labels,
        spectra=matrix,
        intensity_unit="absorbance",
        source_name="synthetic-focused-test.csv",
        source_sha256="1" * 64,
        baseline_run_id="baseline-focused-test",
        baseline_fingerprint="2" * 64,
        prepared_data_sha256=digest,
        original_axis_direction=direction,
        current_axis_direction=direction,
        perturbation_order_policy="preserve_file_order",
        baseline_recipe=recipe,
        baseline_qc={"passed": True},
        warnings=(),
        normalization_state=normalization_state,
    )
