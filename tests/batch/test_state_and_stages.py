from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ftir_baseline.config import CoarseBaselineConfig, FineBaselineConfig, SmoothingConfig
from ftir_baseline.pipeline import run_pipeline
from ftir_workbench.batch.importing import import_sources
from ftir_workbench.batch.models import BatchError, BatchWorkspace, PreparationConfig
from ftir_workbench.batch.service import (
    preview_coarse,
    preview_fine,
    preview_selected,
    to_spectrum_set,
)
from ftir_workbench.batch.state import (
    confirm_coarse,
    confirm_fine,
    confirm_preparation,
    copy_parameters,
    draft_changes,
    get_ready_snapshot,
    restore_committed_draft,
    set_coarse_draft,
    set_fine_draft,
    set_preparation_draft,
    skip_fine,
)

from .test_importing import table_bytes


def workspace_pair() -> tuple[BatchWorkspace, str, str]:
    ws = BatchWorkspace()
    x = np.linspace(1800, 900, 181)
    y = 0.05 + 0.0001 * x + 0.2 * np.exp(-((x - 1300) / 30) ** 2)
    y += 0.002 * np.sin(x / 6)
    ids = import_sources(ws, [("synthetic.csv", table_bytes(x, y, y * 2))])
    return ws, *ids


@pytest.mark.parametrize("smoothing", [False, True])
def test_s04_s06_s07_full_pipeline_parity_and_estimation_channel(smoothing: bool) -> None:
    ws, a, _ = workspace_pair()
    prep = ws.states[a].preparation_draft.to_dict()
    prep["baseline_smoothing"] = SmoothingConfig(enabled=smoothing).to_dict()
    set_preparation_draft(ws, a, PreparationConfig(**prep))
    confirm_preparation(ws, a)
    original = ws.records[a].raw_intensity.copy()
    coarse = preview_coarse(ws, a)
    confirm_coarse(ws, a)
    fine = preview_fine(ws, a)
    direct = run_pipeline(to_spectrum_set(ws, a), fine.config)
    for actual, expected in (
        (fine.result.baseline.coarse_baseline, direct.baseline.coarse_baseline),
        (fine.result.baseline.fine_baseline, direct.baseline.fine_baseline),
        (fine.result.baseline.total_baseline, direct.baseline.total_baseline),
        (fine.result.analysis_data, direct.analysis_data),
        (fine.result.baseline_estimation_spectra, direct.baseline_estimation_spectra),
        (fine.result.baseline.coarse_baseline, coarse.result.baseline.coarse_baseline),
    ):
        np.testing.assert_array_equal(actual, expected)
    raw_residual = fine.result.absorbance_selected.spectra - fine.result.baseline.coarse_baseline
    estimate_residual = fine.result.baseline_estimation_spectra - fine.result.baseline.coarse_baseline
    assert np.array_equal(raw_residual, estimate_residual) == (not smoothing)
    np.testing.assert_array_equal(ws.records[a].raw_intensity, original)
    np.testing.assert_array_equal(fine.result.analysis_data,
                                  fine.result.absorbance_selected.spectra - fine.result.baseline.total_baseline)


def test_t01_t02_switching_retains_independent_draft_preview_and_formal_result() -> None:
    ws, a, b = workspace_pair()
    preview_coarse(ws, a)
    original = confirm_coarse(ws, a)
    set_coarse_draft(ws, a, CoarseBaselineConfig(method="linear"))
    draft_preview = preview_coarse(ws, a)
    ws.selected_spectrum_id = b
    set_coarse_draft(ws, b, CoarseBaselineConfig(method="offset"))
    preview_coarse(ws, b)
    confirm_coarse(ws, b)
    ws.selected_spectrum_id = a
    assert ws.states[a].coarse_draft.method == "linear"
    assert ws.states[a].coarse_preview is draft_preview
    assert ws.states[a].coarse_snapshot is original
    assert get_ready_snapshot(ws, a, "coarse") is original
    assert original.fingerprint != draft_preview.fingerprint
    assert draft_changes(ws, a)["coarse"]
    restore_committed_draft(ws, a, "coarse")
    assert ws.states[a].coarse_draft == original.config.coarse_baseline


def test_t03_t04_coarse_commit_stales_only_current_fine_and_retains_fine_draft() -> None:
    ws, a, b = workspace_pair()
    for sid in (a, b):
        preview_coarse(ws, sid)
        confirm_coarse(ws, sid)
        preview_fine(ws, sid)
        confirm_fine(ws, sid)
    b_fine = ws.states[b].fine_snapshot
    old_fine, old_coarse = ws.states[a].fine_snapshot, ws.states[a].coarse_snapshot
    set_fine_draft(ws, a, FineBaselineConfig(endpoint_window_width_cm1=15))
    assert ws.states[a].coarse_snapshot is old_coarse
    set_coarse_draft(ws, a, CoarseBaselineConfig(method="linear"))
    assert get_ready_snapshot(ws, a, "fine") is old_fine
    preview_coarse(ws, a)
    confirm_coarse(ws, a)
    assert ws.states[a].fine_snapshot is old_fine
    assert ws.states[a].fine_draft.endpoint_window_width_cm1 == 15
    with pytest.raises(BatchError, match="FINE_STALE"):
        get_ready_snapshot(ws, a, "fine")
    assert get_ready_snapshot(ws, b, "fine") is b_fine


@pytest.mark.parametrize("change", ["unit", "range", "smoothing"])
def test_t05_preparation_requires_explicit_confirmation_before_invalidation(change: str) -> None:
    ws, a, _ = workspace_pair()
    preview_coarse(ws, a)
    original = confirm_coarse(ws, a)
    skip_fine(ws, a)
    payload = ws.states[a].preparation_draft.to_dict()
    if change == "unit":
        payload["input_unit"] = "fraction_transmittance"
    elif change == "range":
        payload["wavenumber_range"] = [1700, 1000]
    else:
        payload["baseline_smoothing"] = {"enabled": True}
    set_preparation_draft(ws, a, PreparationConfig(**payload))
    assert get_ready_snapshot(ws, a, "coarse") is original
    confirm_preparation(ws, a)
    assert ws.states[a].coarse_stale and ws.states[a].fine_stale
    with pytest.raises(BatchError, match="COARSE_STALE"):
        get_ready_snapshot(ws, a, "coarse")


def test_t08_copy_writes_detached_drafts_without_units_or_results() -> None:
    ws, a, b = workspace_pair()
    preview_coarse(ws, b)
    original = confirm_coarse(ws, b)
    set_coarse_draft(ws, a, CoarseBaselineConfig(method="linear"))
    payload = ws.states[a].preparation_draft.to_dict()
    payload.update(input_unit="fraction_transmittance", wavenumber_range=[1700, 1000],
                   baseline_smoothing={"enabled": True}, transmittance_floor=0.001)
    set_preparation_draft(ws, a, PreparationConfig(**payload))
    assert copy_parameters(ws, a, [b], stage="coarse") == {b: None}
    assert ws.states[b].coarse_draft.method == "linear"
    assert ws.states[b].preparation_draft.input_unit == "absorbance"
    assert ws.states[b].preparation_draft.wavenumber_range == (1800, 900)
    assert ws.states[b].coarse_snapshot is original
    copy_parameters(ws, a, [b], stage="coarse", include_range=True, include_smoothing=True)
    target = ws.states[b].preparation_draft
    assert target.wavenumber_range == (1700, 1000) and target.baseline_smoothing.enabled
    assert target.input_unit == "absorbance" and target.transmittance_floor is None
    set_coarse_draft(ws, b, CoarseBaselineConfig(method="offset"))
    assert ws.states[a].coarse_draft.method == "linear"


def test_copy_invalid_range_or_anchors_is_atomic_per_target() -> None:
    ws, a, b = workspace_pair()
    old = ws.states[b].fine_draft
    set_fine_draft(ws, a, FineBaselineConfig(method="pchip", anchors=(
        {"start": 4000, "end": 4008}, {"start": 5000, "end": 5008},
    )))
    assert "ANCHOR_OUT_OF_RANGE" in copy_parameters(ws, a, [b], stage="fine")[b]
    assert ws.states[b].fine_draft is old
    payload = ws.states[a].preparation_draft.to_dict()
    payload["wavenumber_range"] = [4000, 900]
    set_preparation_draft(ws, a, PreparationConfig(**payload))
    original = ws.states[b].preparation_draft
    assert "RANGE_INVALID" in copy_parameters(ws, a, [b], stage="coarse", include_range=True)[b]
    assert ws.states[b].preparation_draft is original


def test_t09_not_decided_and_explicit_skip_are_distinct_and_skip_does_not_fit() -> None:
    ws, a, _ = workspace_pair()
    preview_coarse(ws, a)
    coarse = confirm_coarse(ws, a)
    assert ws.states[a].fine_decision == "not_decided"
    with pytest.raises(BatchError, match="FINE_REQUIRED"):
        get_ready_snapshot(ws, a, "fine")
    final = skip_fine(ws, a)
    assert final.result is coarse.result
    assert ws.states[a].fine_decision == "explicitly_skipped"
    assert get_ready_snapshot(ws, a, "fine") is final
    assert final.parent_coarse_fingerprint == coarse.fingerprint
    assert np.all(final.result.baseline.fine_baseline == 0)


def test_t10_batch_failure_isolated_and_does_not_stop_later_item() -> None:
    ws, a, b = workspace_pair()
    c = import_sources(ws, [("third.csv", table_bytes(ws.records[a].wavenumber,
                                                     ws.records[a].raw_intensity))])[0]
    calls = []

    def runner(data, config):
        calls.append(data.metadata["spectrum_id"])
        if data.metadata["spectrum_id"] == b:
            raise RuntimeError("synthetic algorithm failure")
        return run_pipeline(data, config)

    results = preview_selected(ws, [a, b, c], stage="coarse", runner=runner)
    assert results[a] is None and results[c] is None
    assert "PIPELINE_FAILED" in results[b]
    assert calls == [a, b, c]
    assert ws.states[a].coarse_preview is not None and ws.states[c].coarse_preview is not None
    assert ws.states[b].coarse_preview is None
    assert ws.states[a].coarse_snapshot is None


@pytest.mark.parametrize("unit", ["fraction_transmittance", "percent_transmittance"])
def test_i09_nonpositive_transmittance_requires_explicit_floor(unit: str) -> None:
    ws, a, _ = workspace_pair()
    record = ws.records[a]
    y = record.raw_intensity.copy()
    y[10] = 0
    ws.records[a] = replace(record, raw_intensity=y, scientific_input_sha256="")
    payload = ws.states[a].preparation_draft.to_dict()
    payload["input_unit"] = unit
    set_preparation_draft(ws, a, PreparationConfig(**payload))
    confirm_preparation(ws, a)
    with pytest.raises(BatchError, match="PIPELINE_FAILED"):
        preview_coarse(ws, a)
    payload["transmittance_floor"] = 0.0001
    set_preparation_draft(ws, a, PreparationConfig(**payload))
    confirm_preparation(ws, a)
    result = preview_coarse(ws, a).result
    assert result.unit_conversion.repaired
    assert result.unit_conversion.repaired_count == 1
