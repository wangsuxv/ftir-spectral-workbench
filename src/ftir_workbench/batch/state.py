"""Explicit draft, preview and commit transitions; no numerical fitting occurs here."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from ftir_baseline.config import CoarseBaselineConfig, FineBaselineConfig

from .fingerprints import stage_fingerprint
from .models import (
    BatchError,
    BatchWorkspace,
    PreparationConfig,
    Stage,
    StageSnapshot,
    scientific_input_hash,
)
from .recipes import (
    coarse_config,
    fine_config,
    preparation_from_config,
    require_independent_config,
    validate_fine,
    validate_preparation,
)


def set_preparation_draft(workspace: BatchWorkspace, spectrum_id: str,
                          preparation: PreparationConfig) -> None:
    workspace.states[spectrum_id].preparation_draft = PreparationConfig(**preparation.to_dict())


def confirm_preparation(workspace: BatchWorkspace, spectrum_id: str) -> None:
    state = workspace.states[spectrum_id]
    validate_preparation(workspace.records[spectrum_id], state.preparation_draft)
    if state.preparation_draft == state.preparation_committed:
        return
    record = workspace.records[spectrum_id]
    workspace.records[spectrum_id] = replace(
        record, confirmed_input_unit=state.preparation_draft.input_unit,
        scientific_input_sha256="",
    )
    state.preparation_committed = PreparationConfig(**state.preparation_draft.to_dict())
    state.coarse_stale = state.coarse_snapshot is not None
    state.fine_stale = state.fine_snapshot is not None
    workspace.last_export_summary = None


def set_coarse_draft(workspace: BatchWorkspace, spectrum_id: str,
                     coarse: CoarseBaselineConfig) -> None:
    workspace.states[spectrum_id].coarse_draft = CoarseBaselineConfig(**coarse.to_dict())


def set_fine_draft(workspace: BatchWorkspace, spectrum_id: str,
                   fine: FineBaselineConfig) -> None:
    workspace.states[spectrum_id].fine_draft = FineBaselineConfig(**fine.to_dict())


def _validate_snapshot(workspace: BatchWorkspace, spectrum_id: str,
                       snapshot: StageSnapshot) -> None:
    result = snapshot.result
    require_independent_config(snapshot.config)
    if snapshot.stage == "coarse":
        if snapshot.parent_coarse_implementation_fingerprint is not None:
            raise BatchError("PREVIEW_OUTDATED", "coarse snapshot has an unexpected parent implementation")
    elif snapshot.stage == "fine":
        parent_implementation = snapshot.parent_coarse_implementation_fingerprint
        if not parent_implementation:
            raise BatchError("PREVIEW_OUTDATED", "fine snapshot is missing its parent implementation")
        expected_parent = stage_fingerprint(
            workspace, spectrum_id,
            coarse_config(preparation_from_config(snapshot.config), snapshot.config.coarse_baseline),
            "coarse", implementation=parent_implementation,
        )
        if snapshot.parent_coarse_fingerprint != expected_parent:
            raise BatchError("PREVIEW_OUTDATED", "fine snapshot parent implementation or fingerprint mismatch")
    if snapshot.spectrum_id != spectrum_id:
        raise BatchError("PREVIEW_OUTDATED", "preview belongs to a different spectrum")
    expected = stage_fingerprint(
        workspace, spectrum_id, snapshot.config, snapshot.stage,
        snapshot.parent_coarse_fingerprint, implementation=snapshot.implementation_fingerprint,
    )
    if expected != snapshot.fingerprint or snapshot.config != result.config:
        raise BatchError("PREVIEW_OUTDATED", "snapshot dependency fingerprint mismatch")
    if snapshot.input_sha256 != result.input_sha256:
        raise BatchError("PREVIEW_OUTDATED", "snapshot input hash mismatch")
    if result.raw_input.n_spectra != 1:
        raise BatchError("PIPELINE_FAILED", "ordinary stage result must contain one spectrum")
    record = workspace.records[spectrum_id]
    source = workspace.sources[record.source_id]
    raw = result.raw_input
    if (
        not np.array_equal(raw.wavenumber, record.wavenumber)
        or not np.array_equal(raw.spectra[0], record.raw_intensity)
        or raw.metadata.get("spectrum_id") != spectrum_id
        or raw.metadata.get("source_id") != record.source_id
        or raw.metadata.get("workspace_id") != workspace.workspace_id
        or raw.metadata.get("original_column_index") != record.original_column_index
        or raw.source_name != source.original_filename
        or raw.intensity_unit != snapshot.config.input_unit
        or snapshot.input_sha256 != scientific_input_hash(
            record.wavenumber, record.raw_intensity, snapshot.config.input_unit
        )
    ):
        raise BatchError("PREVIEW_OUTDATED", "snapshot raw data or provenance mismatch")
    baseline = result.baseline
    arrays = (result.absorbance_selected.spectra, result.baseline_estimation_spectra,
              baseline.coarse_baseline, baseline.fine_baseline, baseline.total_baseline,
              baseline.corrected, result.analysis_data)
    if any(not np.isfinite(array).all() for array in arrays):
        raise BatchError("PIPELINE_FAILED", "non-finite stage output")
    if (
        not np.allclose(baseline.total_baseline, baseline.coarse_baseline + baseline.fine_baseline,
                        rtol=1e-12, atol=1e-14)
        or not np.allclose(result.absorbance_selected.spectra,
                           baseline.corrected + baseline.total_baseline, rtol=1e-12, atol=1e-14)
        or not np.array_equal(result.analysis_data, baseline.corrected)
    ):
        raise BatchError("PIPELINE_FAILED", "baseline decomposition/reconstruction mismatch")


def validate_parent(coarse: StageSnapshot, fine: StageSnapshot) -> None:
    if (fine.parent_coarse_fingerprint != coarse.fingerprint
            or fine.parent_coarse_implementation_fingerprint != coarse.implementation_fingerprint):
        raise BatchError("FINE_STALE", "fine result belongs to another coarse parent")
    if coarse_config(preparation_from_config(fine.config),
                     fine.config.coarse_baseline) != coarse.config:
        raise BatchError("FINE_STALE", "fine recipe changed the confirmed coarse recipe")
    left, right = coarse.result, fine.result
    for a, b in (
        (left.absorbance_selected.wavenumber, right.absorbance_selected.wavenumber),
        (left.absorbance_selected.spectra, right.absorbance_selected.spectra),
        (left.baseline_estimation_spectra, right.baseline_estimation_spectra),
        (left.baseline.coarse_baseline, right.baseline.coarse_baseline),
    ):
        if not np.array_equal(a, b):
            raise BatchError("FINE_STALE", "fine pipeline does not reproduce confirmed coarse arrays")


def get_ready_snapshot(workspace: BatchWorkspace, spectrum_id: str, stage: Stage) -> StageSnapshot:
    if stage not in {"coarse", "fine"}:
        raise ValueError("stage must be coarse or fine")
    state = workspace.states[spectrum_id]
    coarse = state.coarse_snapshot
    if coarse is None:
        raise BatchError("COARSE_REQUIRED", "a confirmed coarse result is required")
    if state.coarse_stale:
        raise BatchError("COARSE_STALE", "coarse result predates the confirmed preparation")
    if preparation_from_config(coarse.config) != state.preparation_committed:
        raise BatchError("COARSE_STALE", "coarse result does not match the confirmed preparation")
    _validate_snapshot(workspace, spectrum_id, coarse)
    if stage == "coarse":
        return coarse
    fine = state.fine_snapshot
    if fine is None or state.fine_decision == "not_decided":
        raise BatchError("FINE_REQUIRED", "confirm fine adjustment or explicitly skip it")
    if state.fine_stale:
        raise BatchError("FINE_STALE", "fine result predates the current coarse result")
    _validate_snapshot(workspace, spectrum_id, fine)
    validate_parent(coarse, fine)
    return fine


def confirm_coarse(workspace: BatchWorkspace, spectrum_id: str) -> StageSnapshot:
    state = workspace.states[spectrum_id]
    preview = state.coarse_preview
    if preview is None:
        raise BatchError("PREVIEW_OUTDATED", "calculate a coarse preview before confirming")
    config = coarse_config(state.preparation_committed, state.coarse_draft)
    expected = stage_fingerprint(workspace, spectrum_id, config, "coarse")
    if preview.stage != "coarse" or preview.fingerprint != expected:
        raise BatchError("PREVIEW_OUTDATED", "coarse preview no longer matches the current draft/input")
    _validate_snapshot(workspace, spectrum_id, preview)
    if np.any(preview.result.baseline.fine_baseline != 0.0):
        raise BatchError("PIPELINE_FAILED", "coarse-only preview contains a fine component")
    previous = state.coarse_snapshot
    changed = previous is None or previous.fingerprint != preview.fingerprint
    state.coarse_snapshot = preview
    state.coarse_committed = CoarseBaselineConfig(**state.coarse_draft.to_dict())
    state.coarse_stale = False
    state.errors.clear()
    if changed:
        state.fine_stale = state.fine_snapshot is not None
        workspace.last_export_summary = None
    return preview


def confirm_fine(workspace: BatchWorkspace, spectrum_id: str) -> StageSnapshot:
    state = workspace.states[spectrum_id]
    coarse = get_ready_snapshot(workspace, spectrum_id, "coarse")
    preview = state.fine_preview
    if preview is None:
        raise BatchError("PREVIEW_OUTDATED", "calculate a fine preview before confirming")
    config = fine_config(coarse.config, state.fine_draft)
    expected = stage_fingerprint(workspace, spectrum_id, config, "fine", coarse.fingerprint)
    if preview.stage != "fine" or preview.fingerprint != expected:
        raise BatchError("PREVIEW_OUTDATED", "fine preview no longer matches the draft/coarse parent")
    _validate_snapshot(workspace, spectrum_id, preview)
    validate_parent(coarse, preview)
    state.fine_snapshot = preview
    state.fine_committed = FineBaselineConfig(**state.fine_draft.to_dict())
    state.fine_parent_coarse_fingerprint = coarse.fingerprint
    state.fine_decision = "applied"
    state.fine_stale = False
    state.errors.clear()
    workspace.last_export_summary = None
    return preview


def skip_fine(workspace: BatchWorkspace, spectrum_id: str) -> StageSnapshot:
    coarse = get_ready_snapshot(workspace, spectrum_id, "coarse")
    state = workspace.states[spectrum_id]
    snapshot = StageSnapshot(
        stage="fine", spectrum_id=spectrum_id, input_sha256=coarse.input_sha256,
        config=coarse.config, result=coarse.result,
        fingerprint=stage_fingerprint(workspace, spectrum_id, coarse.config, "fine", coarse.fingerprint,
                                      implementation=coarse.implementation_fingerprint),
        parent_coarse_fingerprint=coarse.fingerprint,
        implementation_fingerprint=coarse.implementation_fingerprint,
        parent_coarse_implementation_fingerprint=coarse.implementation_fingerprint,
    )
    state.fine_snapshot = snapshot
    state.fine_committed = FineBaselineConfig(**snapshot.config.fine_baseline.to_dict())
    state.fine_parent_coarse_fingerprint = coarse.fingerprint
    state.fine_decision = "explicitly_skipped"
    state.fine_stale = False
    state.errors.clear()
    workspace.last_export_summary = None
    return snapshot


def restore_committed_draft(workspace: BatchWorkspace, spectrum_id: str, stage: Stage) -> None:
    state = workspace.states[spectrum_id]
    if stage == "coarse":
        if state.coarse_committed is None:
            raise BatchError("COARSE_REQUIRED", "there is no committed coarse recipe")
        set_coarse_draft(workspace, spectrum_id, state.coarse_committed)
    elif stage == "fine":
        if state.fine_committed is None:
            raise BatchError("FINE_REQUIRED", "there is no committed fine recipe")
        set_fine_draft(workspace, spectrum_id, state.fine_committed)
    else:
        raise ValueError("stage must be coarse or fine")


def draft_changes(workspace: BatchWorkspace, spectrum_id: str) -> dict[str, bool]:
    state = workspace.states[spectrum_id]
    return {
        "preparation": state.preparation_draft != state.preparation_committed,
        "coarse": state.coarse_committed is None or state.coarse_draft != state.coarse_committed,
        "fine": state.fine_committed is None or state.fine_draft != state.fine_committed,
    }


def copy_parameters(workspace: BatchWorkspace, source_id: str, target_ids: Sequence[str],
                    *, stage: Stage, include_range: bool = False,
                    include_smoothing: bool = False) -> dict[str, str | None]:
    """Copy validated independent drafts; never units, raw data, or confirmed results."""

    if stage not in {"coarse", "fine"}:
        raise ValueError("stage must be coarse or fine")
    source = workspace.states[source_id]
    outcomes: dict[str, str | None] = {}
    for target_id in dict.fromkeys(target_ids):
        try:
            target = workspace.states[target_id]
            preparation_data = target.preparation_draft.to_dict()
            if include_range:
                preparation_data["wavenumber_range"] = source.preparation_draft.wavenumber_range
            if include_smoothing:
                preparation_data["baseline_smoothing"] = source.preparation_draft.baseline_smoothing.to_dict()
            preparation = PreparationConfig(**preparation_data)
            record = workspace.records[target_id]
            validate_preparation(record, preparation)
            if stage == "fine":
                validate_fine(record, preparation, source.fine_draft)
            # All validation precedes writes for this target; earlier targets stay intact on failure.
            if include_range or include_smoothing:
                set_preparation_draft(workspace, target_id, preparation)
            if stage == "coarse":
                set_coarse_draft(workspace, target_id, source.coarse_draft)
            else:
                set_fine_draft(workspace, target_id, source.fine_draft)
            outcomes[target_id] = None
        except (ValueError, KeyError) as exc:
            outcomes[target_id] = str(exc)
    return outcomes
