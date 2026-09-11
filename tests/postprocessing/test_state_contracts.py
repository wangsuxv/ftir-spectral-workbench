"""Independent acceptance of ordinary four-branch ownership and transitions."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ftir_baseline.config import CoarseBaselineConfig
from ftir_workbench.batch.importing import import_sources
from ftir_workbench.batch.models import BatchWorkspace
from ftir_workbench.batch.postprocessing import (
    branch_status,
    confirm_branch,
    confirm_selected,
    copy_drafts,
    get_branch,
    get_postprocessing_state,
    preview_branch,
    preview_normalization,
    preview_selected,
    preview_smoothing,
    update_draft,
)
from ftir_workbench.batch.service import preview_coarse
from ftir_workbench.batch.state import confirm_coarse, set_coarse_draft, skip_fine
from tests.batch.test_importing import table_bytes
from tests.postprocessing.helpers import confirmed_workspace

SMOOTH = {"enabled": True, "method": "gaussian", "gaussian_sigma_points": 1.0}
NORMALIZE = {"enabled": True, "method": "maximum"}
DERIVED = ("smoothed", "normalized_baseline", "normalized_smoothed")


def apply_branch(ws: BatchWorkspace, sid: str, branch: str, values: dict):
    """Commit only the preview requested for this explicit branch and record."""
    update_draft(ws, sid, branch, values)
    preview = preview_branch(ws, sid, branch)
    confirm_branch(ws, sid, branch)
    actual = get_branch(ws, sid, branch)
    assert actual is preview
    return actual


def graph(ws: BatchWorkspace, sid: str) -> dict:
    return {
        "smoothed": apply_branch(ws, sid, "smoothed", SMOOTH),
        "normalized_baseline": apply_branch(ws, sid, "normalized_baseline", NORMALIZE),
        "normalized_smoothed": apply_branch(ws, sid, "normalized_smoothed", NORMALIZE),
    }


def test_pp002_pp003_postprocessing_never_calls_baseline_prepared_or_twod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, sid = confirmed_workspace()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("ordinary derived operations must not fit baseline or create Prepared/2D")

    for target in (
        "ftir_baseline.pipeline.run_pipeline",
        "ftir_workbench.batch.service.run_pipeline",
        "ftir_workbench.services.baseline_service.BaselineWorkflowService.run",
        "ftir_workbench.models.PreparedSpectralDataset.__init__",
        "ftir_workbench.services.twodcos_service.TwoDCOSWorkflowService.compute",
    ):
        monkeypatch.setattr(target, forbidden)
    results = graph(ws, sid)
    assert all(result.spectra.shape == (1, 181) for result in results.values())
    assert get_postprocessing_state(ws, sid).normalization_source == "baseline"
    assert get_postprocessing_state(ws, sid).export_choice == "baseline"


@pytest.mark.parametrize("branch", DERIVED)
def test_pp004_coarse_only_requires_explicit_final_baseline_decision(branch: str) -> None:
    ws, sid = confirmed_workspace(skip=False)
    update_draft(ws, sid, branch, SMOOTH if branch == "smoothed" else NORMALIZE)
    with pytest.raises(ValueError):
        preview_branch(ws, sid, branch)
    assert ws.states[sid].fine_decision == "not_decided"
    assert getattr(get_postprocessing_state(ws, sid), branch).committed is None
    skip_fine(ws, sid)
    if branch == "normalized_smoothed":
        apply_branch(ws, sid, "smoothed", SMOOTH)
    preview_branch(ws, sid, branch)
    confirm_branch(ws, sid, branch)
    assert branch_status(ws, sid, branch)["status"] == "ready"


def test_pp005_parent_arrays_and_recipe_stay_immutable_across_all_branches() -> None:
    ws, sid = confirmed_workspace()
    parent = get_branch(ws, sid, "baseline")
    recipe = parent.result.recipe_dict()
    arrays = {name: getattr(parent.result.baseline, name).copy() for name in (
        "coarse_baseline", "fine_baseline", "total_baseline", "corrected",
    )}
    x = parent.result.absorbance_selected.wavenumber.copy()
    snapshots = graph(ws, sid)
    assert get_branch(ws, sid, "baseline") is parent
    assert parent.result.recipe_dict() == recipe
    np.testing.assert_array_equal(parent.result.absorbance_selected.wavenumber, x)
    for name, expected in arrays.items():
        np.testing.assert_array_equal(getattr(parent.result.baseline, name), expected)
    for branch, result in snapshots.items():
        assert result.baseline is parent
        np.testing.assert_array_equal(result.wavenumber, x)
        for array in (result.wavenumber, result.spectra):
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.setflags(write=True)
        if branch == "smoothed":
            np.testing.assert_array_equal(result.removed_component,
                                          parent.result.analysis_data - result.spectra)


@pytest.mark.parametrize("source", ["smoothed", "normalized_baseline", "normalized_smoothed"])
def test_pp006_smoothing_rejects_every_nonbaseline_source(source: str) -> None:
    ws, sid = confirmed_workspace()
    graph(ws, sid)
    with pytest.raises(ValueError):
        preview_smoothing(ws, sid, source_kind=source)


@pytest.mark.parametrize("source", ["normalized_baseline", "normalized_smoothed", "raw"])
def test_pp006_normalization_rejects_every_noncontract_source(source: str) -> None:
    ws, sid = confirmed_workspace()
    graph(ws, sid)
    with pytest.raises(ValueError):
        preview_normalization(ws, sid, source_kind=source)


def test_pp007_single_and_unrelated_batch_have_identical_derived_arrays() -> None:
    standalone, a = confirmed_workspace()
    isolated = graph(standalone, a)
    x, y = standalone.records[a].wavenumber, standalone.records[a].raw_intensity
    batch, together_a = confirmed_workspace(x, y)
    confirmed_workspace(x, y * 17.0 + 3.0, workspace=batch, name="unrelated.csv")
    other_x = np.linspace(2200.0, 500.0, 341)
    confirmed_workspace(other_x, 2.0 + np.sin(other_x / 100.0), workspace=batch, name="other-grid.csv")
    together = graph(batch, together_a)
    for branch in DERIVED:
        np.testing.assert_array_equal(together[branch].spectra, isolated[branch].spectra)
        np.testing.assert_array_equal(together[branch].wavenumber, isolated[branch].wavenumber)


@pytest.mark.parametrize("branch", DERIVED)
def test_pp043_preview_alone_is_never_a_confirmed_branch(branch: str) -> None:
    ws, sid = confirmed_workspace()
    if branch == "normalized_smoothed":
        apply_branch(ws, sid, "smoothed", SMOOTH)
    update_draft(ws, sid, branch, SMOOTH if branch == "smoothed" else NORMALIZE)
    preview = preview_branch(ws, sid, branch)
    state = getattr(get_postprocessing_state(ws, sid), branch)
    assert state.preview is preview
    assert state.committed is None
    assert branch_status(ws, sid, branch)["preview_current"]
    with pytest.raises(ValueError):
        get_branch(ws, sid, branch)


@pytest.mark.parametrize("branch", DERIVED)
def test_pp044_pp045_edit_preserves_confirmed_but_blocks_old_preview_commit(branch: str) -> None:
    ws, sid = confirmed_workspace()
    original = graph(ws, sid)
    update_draft(ws, sid, branch,
                 {"gaussian_sigma_points": 2.0} if branch == "smoothed" else {"target": 2.0})
    state = branch_status(ws, sid, branch)
    assert state["status"] == "ready"
    assert state["draft_modified"]
    assert not state["preview_current"]
    assert get_branch(ws, sid, branch) is original[branch]
    with pytest.raises(ValueError):
        confirm_branch(ws, sid, branch)
    for key, snapshot in original.items():
        assert get_branch(ws, sid, key) is snapshot


def test_pp046_new_smoothing_invalidates_only_its_normalized_child() -> None:
    ws, sid = confirmed_workspace()
    original = graph(ws, sid)
    new_smoothing = apply_branch(ws, sid, "smoothed", {**SMOOTH, "gaussian_sigma_points": 2.0})
    assert new_smoothing.fingerprint != original["smoothed"].fingerprint
    assert get_branch(ws, sid, "normalized_baseline") is original["normalized_baseline"]
    assert get_branch(ws, sid, "baseline") is original["smoothed"].baseline
    assert branch_status(ws, sid, "normalized_smoothed")["status"] == "stale"
    assert get_postprocessing_state(ws, sid).normalized_smoothed.committed is original["normalized_smoothed"]
    assert original["normalized_smoothed"].parent_smoothed is original["smoothed"]
    with pytest.raises(ValueError):
        get_branch(ws, sid, "normalized_smoothed")
    with pytest.raises(ValueError):
        confirm_branch(ws, sid, "normalized_smoothed")


def test_pp047_confirmed_new_baseline_invalidates_only_that_records_descendants() -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="unrelated.csv")
    original_a, original_b = graph(ws, a), graph(ws, b)
    set_coarse_draft(ws, a, CoarseBaselineConfig(method="offset"))
    preview_coarse(ws, a)
    confirm_coarse(ws, a)
    skip_fine(ws, a)
    for branch in DERIVED:
        assert branch_status(ws, a, branch)["status"] == "stale"
        assert getattr(get_postprocessing_state(ws, a), branch).committed is original_a[branch]
        with pytest.raises(ValueError):
            get_branch(ws, a, branch)
        assert get_branch(ws, b, branch) is original_b[branch]


@pytest.mark.parametrize("branch,other", [("normalized_baseline", "normalized_smoothed"),
                                         ("normalized_smoothed", "normalized_baseline")])
def test_pp048_normalization_branches_do_not_supersede_each_other(branch: str, other: str) -> None:
    ws, sid = confirmed_workspace()
    original = graph(ws, sid)
    changed = apply_branch(ws, sid, branch, {**NORMALIZE, "target": 2.0})
    assert changed.fingerprint != original[branch].fingerprint
    np.testing.assert_array_equal(changed.spectra, original[branch].spectra * 2.0)
    assert get_branch(ws, sid, other) is original[other]
    assert get_branch(ws, sid, "smoothed") is original["smoothed"]


def test_pp049_equal_smoothing_arrays_with_different_recipe_still_change_parent() -> None:
    ws, sid = confirmed_workspace(np.arange(21.0), np.ones(21))
    old_smoothing = apply_branch(ws, sid, "smoothed", {
        "enabled": True, "method": "median", "median_window_length": 3,
    })
    normalized = apply_branch(ws, sid, "normalized_smoothed", NORMALIZE)
    new_smoothing = apply_branch(ws, sid, "smoothed", {"median_window_length": 5})
    np.testing.assert_array_equal(new_smoothing.spectra, old_smoothing.spectra)
    assert new_smoothing.fingerprint != old_smoothing.fingerprint
    assert normalized.parent_fingerprint == old_smoothing.fingerprint
    assert branch_status(ws, sid, "normalized_smoothed")["status"] == "stale"


def test_pp050_reconfirm_and_unused_parameters_do_not_invalidate_descendants() -> None:
    ws, sid = confirmed_workspace()
    original = graph(ws, sid)
    update_draft(ws, sid, "smoothed", {"savgol_window_length": 9999,
                                      "median_window_length": 111})
    assert get_postprocessing_state(ws, sid).smoothed.draft["savgol_window_length"] == 9999
    again = preview_branch(ws, sid, "smoothed")
    assert again.fingerprint == original["smoothed"].fingerprint
    confirm_branch(ws, sid, "smoothed")
    confirm_branch(ws, sid, "smoothed")
    for branch in DERIVED:
        assert get_branch(ws, sid, branch).fingerprint == original[branch].fingerprint
        assert branch_status(ws, sid, branch)["status"] == "ready"
    update_draft(ws, sid, "normalized_baseline", {"area_interval": [9000.0, 8000.0]})
    assert preview_branch(ws, sid, "normalized_baseline").fingerprint == original["normalized_baseline"].fingerprint
    confirm_branch(ws, sid, "normalized_baseline")
    assert get_branch(ws, sid, "normalized_smoothed") is original["normalized_smoothed"]


def test_pp051_per_source_drafts_and_display_selection_remain_independent() -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="B.csv")
    original = graph(ws, a)
    update_draft(ws, a, "smoothed", {"gaussian_sigma_points": 2.0})
    update_draft(ws, a, "normalized_baseline", {"method": "vector"})
    update_draft(ws, a, "normalized_smoothed", {"target": 3.0})
    state_a = get_postprocessing_state(ws, a)
    state_a.normalization_source = "smoothed"
    state_a.export_choice = "normalized_baseline"
    ws.selected_spectrum_id = b
    update_draft(ws, b, "smoothed", {**SMOOTH, "gaussian_sigma_points": 4.0})
    ws.selected_spectrum_id = a
    assert get_postprocessing_state(ws, a) is state_a
    assert state_a.smoothed.draft["gaussian_sigma_points"] == 2.0
    assert state_a.normalized_baseline.draft["method"] == "vector"
    assert state_a.normalized_smoothed.draft["target"] == 3.0
    assert state_a.normalization_source == "smoothed"
    assert state_a.export_choice == "normalized_baseline"
    assert get_postprocessing_state(ws, b).smoothed.draft["gaussian_sigma_points"] == 4.0
    for branch in DERIVED:
        assert get_branch(ws, a, branch) is original[branch]


def test_pp052_copy_resolves_each_targets_smoothed_parent_without_fallback_or_coefficient_copy() -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="no-S.csv")
    _, c = confirmed_workspace(np.linspace(1500.0, 1400.0, 21), np.ones(21),
                               workspace=ws, name="missing-range.csv")
    _, d = confirmed_workspace(ws.records[a].wavenumber, ws.records[a].raw_intensity * 2.0,
                               workspace=ws, name="own-amplitude.csv")
    for sid in (a, c, d):
        apply_branch(ws, sid, "smoothed", SMOOTH)
    source = apply_branch(ws, a, "normalized_smoothed", {
        "enabled": True, "method": "internal_peak_height", "reference_interval": [1700.0, 1600.0],
    })
    parents = {sid: get_branch(ws, sid, "smoothed") for sid in (a, c, d)}
    outcomes = copy_drafts(ws, a, [b, c, d], branch="normalized_smoothed")
    assert outcomes[b] and outcomes[c]
    assert outcomes[d] is None
    target = get_postprocessing_state(ws, d).normalized_smoothed
    assert target.committed is None and target.preview is None
    assert target.draft == get_postprocessing_state(ws, a).normalized_smoothed.draft
    assert target.draft is not get_postprocessing_state(ws, a).normalized_smoothed.draft
    own = preview_branch(ws, d, "normalized_smoothed")
    assert own.parent_fingerprint == parents[d].fingerprint
    assert own.parent_fingerprint != source.parent_fingerprint
    np.testing.assert_allclose(own.scale, source.scale / 2.0, rtol=1e-15, atol=0)
    assert get_postprocessing_state(ws, b).smoothed.committed is None


def test_pp053_batch_preview_and_confirm_isolate_invalid_windows_from_good_and_committed_items() -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(np.arange(5.0), np.ones(5), workspace=ws, name="short.csv")
    _, c = confirmed_workspace(workspace=ws, name="previous-valid.csv")
    previous = apply_branch(ws, c, "smoothed", SMOOTH)
    for sid in (a, b):
        update_draft(ws, sid, "smoothed", {"enabled": True, "method": "savgol", "savgol_window_length": 7})
    update_draft(ws, c, "smoothed", {"method": "moving_average", "moving_average_window_length": 999})
    previews = preview_selected(ws, [a, b, c], branch="smoothed")
    assert previews[a] is None and previews[b] and previews[c]
    assert get_postprocessing_state(ws, a).smoothed.committed is None
    confirmations = confirm_selected(ws, [a, b, c], branch="smoothed")
    assert confirmations[a] is None and confirmations[b] and confirmations[c]
    assert get_branch(ws, a, "smoothed") is not None
    assert get_postprocessing_state(ws, b).smoothed.committed is None
    assert get_branch(ws, c, "smoothed") is previous


def test_pp054_identical_values_from_different_sources_never_share_snapshot_provenance() -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(ws.records[a].wavenumber, ws.records[a].raw_intensity,
                               workspace=ws, name="synthetic.csv")
    original_a, original_b = graph(ws, a), graph(ws, b)
    assert ws.records[a].scientific_input_sha256 == ws.records[b].scientific_input_sha256
    assert ws.records[a].source_id != ws.records[b].source_id
    for branch in DERIVED:
        left, right = original_a[branch], original_b[branch]
        np.testing.assert_array_equal(left.spectra, right.spectra)
        assert left is not right
        assert left.fingerprint != right.fingerprint
        assert left.baseline.spectrum_id == a and right.baseline.spectrum_id == b
        assert left.baseline.result.raw_input.metadata["source_id"] == ws.records[a].source_id
        assert right.baseline.result.raw_input.metadata["source_id"] == ws.records[b].source_id
    ws.records[a] = replace(ws.records[a], display_name="a fresh label")
    ws.display_order.reverse()
    ws.records.pop(b)
    ws.states.pop(b)
    ws.postprocessing.pop(b)
    ws.display_order.remove(b)
    for branch in DERIVED:
        assert get_branch(ws, a, branch) is original_a[branch]
        assert preview_branch(ws, a, branch).fingerprint == original_a[branch].fingerprint


@pytest.mark.parametrize("branch", DERIVED)
def test_preview_from_another_identity_cannot_be_confirmed(branch: str) -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(ws.records[a].wavenumber, ws.records[a].raw_intensity,
                               workspace=ws, name="same-data.csv")
    original_a, original_b = graph(ws, a), graph(ws, b)
    getattr(get_postprocessing_state(ws, a), branch).preview = original_b[branch]
    with pytest.raises(ValueError):
        confirm_branch(ws, a, branch)
    assert get_branch(ws, a, branch) is original_a[branch]


@pytest.mark.parametrize("change", [
    "output", "axis", "baseline", "parent", "workspace", "spectrum", "source",
    "input", "request", "effective_recipe", "quantity", "purpose", "scale", "offset",
])
def test_forged_snapshot_arrays_or_parent_cannot_be_promoted(change: str) -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="other.csv")
    original = graph(ws, a)
    other = graph(ws, b)
    snapshot = original["normalized_smoothed"]
    changes = {
        "output": {"spectra": snapshot.spectra + 0.1},
        "axis": {"wavenumber": snapshot.wavenumber + 0.1},
        "baseline": {"baseline": other["smoothed"].baseline},
        "parent": {"parent_fingerprint": other["smoothed"].fingerprint},
        "workspace": {"workspace_id": "another-workspace"},
        "spectrum": {"spectrum_id": b},
        "source": {"source_id": ws.records[b].source_id},
        "input": {"input_sha256": "0" * 64},
        "request": {"request_fingerprint": "0" * 64},
        "effective_recipe": {"effective_recipe": {"enabled": True, "method": "vector", "target": 1.0}},
        "quantity": {"quantity": "absorbance"},
        "purpose": {"purpose": "display_only"},
        "scale": {"scale": snapshot.scale * 2.0},
        "offset": {"offset": snapshot.offset + 0.1},
    }
    with pytest.raises(ValueError):
        altered = replace(snapshot, **changes[change])
        get_postprocessing_state(ws, a).normalized_smoothed.preview = altered
        confirm_branch(ws, a, "normalized_smoothed")
    assert get_branch(ws, a, "normalized_smoothed") is snapshot


def test_pp018_reads_confirmation_cached_preview_and_copy_do_not_recalculate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, a = confirmed_workspace()
    _, b = confirmed_workspace(workspace=ws, name="copy-target.csv")
    original = graph(ws, a)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("reading, confirming, copying or reusing a cached preview must not calculate")

    monkeypatch.setattr("ftir_workbench.batch.postprocessing.smooth_spectral_arrays", forbidden)
    monkeypatch.setattr("ftir_workbench.batch.postprocessing.normalize_spectral_arrays", forbidden)
    state = get_postprocessing_state(ws, a)
    state.display_preferences.update({"x_limits": [1500.0, 1400.0], "show_removed": True})
    for branch in DERIVED:
        assert get_branch(ws, a, branch) is original[branch]
        assert branch_status(ws, a, branch)["status"] == "ready"
        assert preview_branch(ws, a, branch) is original[branch]
        confirm_branch(ws, a, branch)
    assert copy_drafts(ws, a, [b], branch="smoothed")[b] is None
    assert get_postprocessing_state(ws, b).smoothed.preview is None
    assert get_postprocessing_state(ws, b).smoothed.committed is None


def test_pp054_wide_columns_have_independent_derived_cache_and_parent_identity() -> None:
    ws = BatchWorkspace()
    x = np.linspace(1800.0, 900.0, 181)
    y = 0.4 + np.exp(-((x - 1400.0) / 45.0) ** 2)
    a, b = import_sources(ws, [("wide.csv", table_bytes(x, y, y * 2.0, labels=("same", "same")))])
    for sid in (a, b):
        set_coarse_draft(ws, sid, CoarseBaselineConfig(method="none"))
        preview_coarse(ws, sid)
        confirm_coarse(ws, sid)
        skip_fine(ws, sid)
    left, right = graph(ws, a), graph(ws, b)
    assert ws.records[a].source_id == ws.records[b].source_id
    assert ws.records[a].scientific_input_sha256 != ws.records[b].scientific_input_sha256
    for branch in DERIVED:
        assert left[branch].spectrum_id == a and right[branch].spectrum_id == b
        assert left[branch].fingerprint != right[branch].fingerprint
        assert left[branch].parent_fingerprint != right[branch].parent_fingerprint
        assert get_postprocessing_state(ws, a) is not get_postprocessing_state(ws, b)
    np.testing.assert_array_equal(right["smoothed"].spectra, left["smoothed"].spectra * 2.0)
    for branch in ("normalized_baseline", "normalized_smoothed"):
        np.testing.assert_array_equal(right[branch].spectra, left[branch].spectra)
