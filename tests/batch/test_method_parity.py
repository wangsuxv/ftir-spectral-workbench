"""All public baseline method families stay delegated to the frozen pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from ftir_baseline.config import CoarseBaselineConfig, FineBaselineConfig
from ftir_baseline.pipeline import run_pipeline
from ftir_workbench.batch.importing import import_sources
from ftir_workbench.batch.models import BatchWorkspace
from tests.batch.test_importing import table_bytes


def synthetic_workspace(unit: str = "absorbance") -> tuple[BatchWorkspace, str]:
    x = np.linspace(1800, 900, 181)
    a = 0.2 + 0.0001 * (x - 900) + 0.5 * np.exp(-(((x - 1450) / 35) ** 2))
    a += 0.002 * np.sin(x / 7)
    y = a if unit == "absorbance" else 10 ** (-a)
    if unit == "percent_transmittance":
        y *= 100
    ws = BatchWorkspace()
    sid = import_sources(ws, [("synthetic.csv", table_bytes(x, y))], input_unit=unit)[0]
    return ws, sid


def assert_exact_decomposition(actual, expected) -> None:  # type: ignore[no-untyped-def]
    for name in ("coarse_baseline", "fine_baseline", "total_baseline", "corrected"):
        np.testing.assert_array_equal(
            getattr(actual.baseline, name), getattr(expected.baseline, name)
        )
    np.testing.assert_array_equal(
        actual.absorbance_selected.wavenumber, expected.absorbance_selected.wavenumber
    )
    np.testing.assert_array_equal(
        actual.baseline_estimation_spectra, expected.baseline_estimation_spectra
    )
    np.testing.assert_array_equal(actual.analysis_data, expected.analysis_data)


@pytest.mark.parametrize(
    "method", ["none", "offset", "linear", "arpls", "asls", "airpls", "rubberband", "pspline_arpls"]
)
@pytest.mark.parametrize("unit", ["absorbance", "percent_transmittance", "fraction_transmittance"])
def test_all_coarse_methods_and_input_units_equal_core(method: str, unit: str) -> None:
    from ftir_workbench.batch.service import IndependentBatchBaselineService
    from ftir_workbench.batch.state import set_coarse_draft

    ws, sid = synthetic_workspace(unit)
    set_coarse_draft(ws, sid, CoarseBaselineConfig(method=method))
    service = IndependentBatchBaselineService()
    preview = service.preview_coarse(ws, sid)
    authoritative = run_pipeline(preview.result.raw_input, preview.config)
    assert preview.result.raw_input.n_spectra == 1
    assert preview.config.series_mode == "independent_locked"
    assert preview.config.normalization.method == "none"
    assert preview.config.fine_baseline == FineBaselineConfig(enabled=False, method="none")
    assert_exact_decomposition(preview.result, authoritative)


@pytest.mark.parametrize(
    "method,order",
    [
        ("none", 1),
        ("endpoint_window_linear", 1),
        ("piecewise_linear", 1),
        ("pchip", 1),
        ("polynomial", 1),
        ("polynomial", 2),
        ("polynomial", 3),
    ],
)
def test_all_fine_methods_equal_single_authoritative_run(method: str, order: int) -> None:
    from ftir_workbench.batch.service import IndependentBatchBaselineService
    from ftir_workbench.batch.state import confirm_coarse, set_fine_draft, skip_fine

    ws, sid = synthetic_workspace()
    service = IndependentBatchBaselineService()
    service.preview_coarse(ws, sid)
    confirm_coarse(ws, sid)
    anchors = [{"start": center - 4, "end": center + 4} for center in [900, 1150, 1600, 1800]]
    set_fine_draft(
        ws, sid, FineBaselineConfig(method=method, polynomial_order=order, anchors=anchors)
    )
    fine = skip_fine(ws, sid) if method == "none" else service.preview_fine(ws, sid)
    expected = run_pipeline(fine.result.raw_input, fine.config)
    assert_exact_decomposition(fine.result, expected)
    np.testing.assert_array_equal(
        fine.result.baseline.coarse_baseline,
        ws.states[sid].coarse_snapshot.result.baseline.coarse_baseline,
    )
