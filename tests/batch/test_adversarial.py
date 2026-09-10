"""Independent adversarial checks of identity, cache, and stage dependencies."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ftir_baseline.config import CoarseBaselineConfig, FineBaselineConfig, PipelineConfig
from ftir_baseline.models import SpectrumSet
from ftir_baseline.pipeline import PipelineResult, run_pipeline
from ftir_workbench.batch.fingerprints import stage_fingerprint
from ftir_workbench.batch.importing import import_sources
from ftir_workbench.batch.models import BatchError, BatchWorkspace, StageSnapshot
from ftir_workbench.batch.recipes import coarse_config
from ftir_workbench.batch.service import (
    IndependentBatchBaselineService,
    preview_fine,
    run_singleton,
)
from ftir_workbench.batch.state import (
    confirm_coarse,
    confirm_fine,
    set_coarse_draft,
    set_fine_draft,
)
from tests.batch.test_importing import table_bytes


def make_batch(*, same_values: bool = False, same_names: bool = False,
               wide: bool = False) -> tuple[BatchWorkspace, list[str]]:
    x = np.linspace(1800.0, 900.0, 181)
    y = 0.2 + 0.0001 * x + 0.45 * np.exp(-((x - 1450.0) / 35.0) ** 2)
    y += 0.002 * np.sin(x / 7.0)
    ys = [y, y if same_values else y * 3.0, y if same_values else y + 0.1]
    names = ["same.csv"] * 3 if same_names else ["A.csv", "B.csv", "C.csv"]
    uploads = ([("wide.csv", table_bytes(x, *ys, labels=("same",) * 3))] if wide
               else [(name, table_bytes(x, values))
                     for name, values in zip(names, ys, strict=True)])
    workspace = BatchWorkspace()
    ids = import_sources(workspace, uploads)
    assert len(ids) == 3, workspace.import_issues
    for sid in ids:
        set_coarse_draft(workspace, sid, CoarseBaselineConfig(method="linear"))
    return workspace, ids


def complete_fine(workspace: BatchWorkspace, spectrum_id: str,
                  service: IndependentBatchBaselineService) -> StageSnapshot:
    service.preview_coarse(workspace, spectrum_id)
    confirm_coarse(workspace, spectrum_id)
    preview = service.preview_fine(workspace, spectrum_id)
    confirm_fine(workspace, spectrum_id)
    assert workspace.states[spectrum_id].fine_snapshot is preview
    return preview


@pytest.mark.parametrize("same_names,wide", [(False, False), (True, False), (True, True)])
def test_identical_values_never_reuse_another_sources_result_provenance(
    same_names: bool, wide: bool,
) -> None:
    workspace, ids = make_batch(same_values=True, same_names=same_names, wide=wide)
    assert len({workspace.records[sid].scientific_input_sha256 for sid in ids}) == 1
    service = IndependentBatchBaselineService()
    results = [complete_fine(workspace, sid, service) for sid in ids]

    assert len({snapshot.fingerprint for snapshot in results}) == 3
    assert len({id(snapshot.result) for snapshot in results}) == 3
    assert len({id(workspace.states[sid].coarse_snapshot.result) for sid in ids}) == 3
    for sid, snapshot in zip(ids, results, strict=True):
        record = workspace.records[sid]
        source = workspace.sources[record.source_id]
        assert snapshot.spectrum_id == sid
        assert snapshot.result.raw_input.source_name == source.original_filename
        metadata = snapshot.result.raw_input.metadata
        assert metadata["spectrum_id"] == sid
        assert metadata["source_id"] == record.source_id
        assert metadata["original_column_index"] == record.original_column_index
        assert metadata["perturbation_semantics"] == "not_applicable"
        np.testing.assert_array_equal(snapshot.result.raw_input.perturbation, [0.0])
        np.testing.assert_array_equal(snapshot.result.analysis_data, results[0].result.analysis_data)
        # Revisit after processing another identity: cached data retains this source.
        again = service.preview_fine(workspace, sid)
        assert again.fingerprint == snapshot.fingerprint
        assert again.result.raw_input.metadata["spectrum_id"] == sid


@pytest.mark.parametrize("change", ["edit", "delete", "reorder", "rename"])
def test_unrelated_entry_changes_preserve_a_arrays_and_stage_fingerprint(change: str) -> None:
    workspace, (a, b, c) = make_batch()
    service = IndependentBatchBaselineService()
    original = complete_fine(workspace, a, service)
    before = original.result.analysis_data.copy()
    if change == "edit":
        set_coarse_draft(workspace, b, CoarseBaselineConfig(method="offset"))
        service.preview_coarse(workspace, b)
        confirm_coarse(workspace, b)
    elif change == "delete":
        workspace.states.pop(b)
        workspace.records.pop(b)
        workspace.display_order.remove(b)
    elif change == "reorder":
        workspace.display_order[:] = [c, b, a]
    else:
        workspace.records[b] = replace(workspace.records[b], display_name="renamed B")

    assert workspace.states[a].fine_snapshot is original
    assert not workspace.states[a].fine_stale
    assert stage_fingerprint(workspace, a, original.config, "fine",
                             original.parent_coarse_fingerprint) == original.fingerprint
    again = service.preview_fine(workspace, a)
    assert again.fingerprint == original.fingerprint
    np.testing.assert_array_equal(again.result.analysis_data, before)


@pytest.mark.parametrize("series_mode", ["collaborative_pls", "shared_shape"])
def test_injected_joint_processing_recipe_is_rejected_before_pipeline(series_mode: str) -> None:
    workspace, (a, _, _) = make_batch()
    state = workspace.states[a]
    payload = coarse_config(state.preparation_draft, state.coarse_draft).to_dict()
    payload["series_mode"] = series_mode

    def forbidden_runner(data: SpectrumSet, config: PipelineConfig) -> PipelineResult:
        pytest.fail("an incompatible recipe must fail before invoking the scientific pipeline")

    with pytest.raises(BatchError, match="INCOMPATIBLE_RECIPE"):
        run_singleton(workspace, a, PipelineConfig(**payload), runner=forbidden_runner)
    assert state.coarse_snapshot is None


def test_injected_normalization_is_rejected_before_pipeline() -> None:
    workspace, (a, _, _) = make_batch()
    state = workspace.states[a]
    payload = coarse_config(state.preparation_draft, state.coarse_draft).to_dict()
    payload["normalization"] = {"method": "vector"}

    def forbidden_runner(data: SpectrumSet, config: PipelineConfig) -> PipelineResult:
        pytest.fail("ordinary mode must not invoke a normalizing pipeline")

    with pytest.raises(BatchError, match="INCOMPATIBLE_RECIPE"):
        run_singleton(workspace, a, PipelineConfig(**payload), runner=forbidden_runner)


def test_late_coarse_preview_cannot_commit_after_draft_changes() -> None:
    workspace, (a, _, _) = make_batch()
    service = IndependentBatchBaselineService()
    original = complete_fine(workspace, a, service)
    coarse_before = workspace.states[a].coarse_snapshot
    late_preview = service.preview_coarse(workspace, a)
    set_coarse_draft(workspace, a, CoarseBaselineConfig(method="offset"))
    workspace.states[a].coarse_preview = late_preview
    with pytest.raises(BatchError):
        confirm_coarse(workspace, a)
    assert workspace.states[a].coarse_snapshot is coarse_before
    assert workspace.states[a].fine_snapshot is original
    assert not workspace.states[a].fine_stale


def test_late_fine_preview_cannot_commit_after_fine_draft_changes() -> None:
    workspace, (a, _, _) = make_batch()
    service = IndependentBatchBaselineService()
    original = complete_fine(workspace, a, service)
    late_preview = service.preview_fine(workspace, a)
    set_fine_draft(workspace, a, FineBaselineConfig(endpoint_window_width_cm1=20.0))
    workspace.states[a].fine_preview = late_preview
    with pytest.raises(BatchError):
        confirm_fine(workspace, a)
    assert workspace.states[a].fine_snapshot is original
    assert not workspace.states[a].fine_stale


def test_late_fine_preview_cannot_commit_after_new_coarse_parent() -> None:
    workspace, (a, _, _) = make_batch()
    service = IndependentBatchBaselineService()
    original = complete_fine(workspace, a, service)
    set_coarse_draft(workspace, a, CoarseBaselineConfig(method="offset"))
    service.preview_coarse(workspace, a)
    confirm_coarse(workspace, a)
    workspace.states[a].fine_preview = original
    with pytest.raises(BatchError):
        confirm_fine(workspace, a)
    assert workspace.states[a].fine_stale


def test_fine_pipeline_coarse_component_must_match_confirmed_parent() -> None:
    workspace, (a, _, _) = make_batch()
    service = IndependentBatchBaselineService()
    service.preview_coarse(workspace, a)
    confirm_coarse(workspace, a)

    def mismatched_runner(data: SpectrumSet, config: PipelineConfig) -> PipelineResult:
        result = run_pipeline(data, config)
        baseline = result.baseline
        shifted = replace(baseline, coarse_baseline=baseline.coarse_baseline + 0.01,
                          total_baseline=baseline.total_baseline + 0.01,
                          corrected=baseline.corrected - 0.01)
        normalization = replace(result.normalization, analysis_data=shifted.corrected,
                                view_data=shifted.corrected)
        return replace(result, baseline=shifted, normalization=normalization)

    with pytest.raises(BatchError, match="FINE_STALE"):
        preview_fine(workspace, a, runner=mismatched_runner)
    assert workspace.states[a].fine_snapshot is None


@pytest.mark.parametrize("stage", ["coarse", "fine"])
def test_another_spectrums_preview_cannot_be_confirmed(stage: str) -> None:
    workspace, (a, b, _) = make_batch(same_values=True, same_names=True)
    service = IndependentBatchBaselineService()
    original = complete_fine(workspace, a, service)
    complete_fine(workspace, b, service)
    if stage == "coarse":
        workspace.states[a].coarse_preview = workspace.states[b].coarse_preview
        with pytest.raises(BatchError):
            confirm_coarse(workspace, a)
    else:
        workspace.states[a].fine_preview = workspace.states[b].fine_preview
        with pytest.raises(BatchError):
            confirm_fine(workspace, a)
    assert workspace.states[a].fine_snapshot is original


def test_confirming_identical_coarse_preview_keeps_confirmed_fine_valid() -> None:
    workspace, (a, _, _) = make_batch()
    service = IndependentBatchBaselineService()
    original = complete_fine(workspace, a, service)
    state = workspace.states[a]
    coarse_fingerprint = state.coarse_snapshot.fingerprint
    service.preview_coarse(workspace, a)
    confirm_coarse(workspace, a)
    assert state.coarse_snapshot.fingerprint == coarse_fingerprint
    assert state.fine_snapshot is original
    assert not state.fine_stale
    assert state.fine_decision == "applied"
    assert state.fine_parent_coarse_fingerprint == coarse_fingerprint


def test_display_name_and_view_state_do_not_change_own_scientific_fingerprint() -> None:
    workspace, (a, b, _) = make_batch()
    service = IndependentBatchBaselineService()
    original = complete_fine(workspace, a, service)
    workspace.records[a] = replace(workspace.records[a], display_name="My display label")
    workspace.states[a].display_preferences.update({"x_range": [1400.0, 1200.0],
                                                    "display_unit": "%T"})
    workspace.selected_spectrum_id = b
    workspace.selected_spectrum_id = a
    again = service.preview_fine(workspace, a)
    assert again.fingerprint == original.fingerprint
    assert workspace.states[a].fine_snapshot is original
    np.testing.assert_array_equal(again.result.analysis_data, original.result.analysis_data)
