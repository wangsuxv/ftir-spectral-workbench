"""Single-spectrum numerical orchestration through the frozen run_pipeline only."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ftir_baseline.config import PipelineConfig
from ftir_baseline.models import SpectrumSet, thaw_mapping
from ftir_baseline.pipeline import PipelineResult, run_pipeline

from .fingerprints import (
    implementation_fingerprint,
    source_selection_fingerprint,
    stage_fingerprint,
)
from .models import BatchError, BatchWorkspace, Stage, StageSnapshot
from .recipes import (
    coarse_config,
    fine_config,
    require_independent_config,
    validate_fine,
    validate_preparation,
)
from .state import _validate_snapshot, get_ready_snapshot, validate_parent

Runner = Callable[[SpectrumSet, PipelineConfig], PipelineResult]


def to_spectrum_set(workspace: BatchWorkspace, spectrum_id: str) -> SpectrumSet:
    record = workspace.records[spectrum_id]
    source = workspace.sources[record.source_id]
    return SpectrumSet(
        wavenumber=record.wavenumber,
        spectra=record.raw_intensity[np.newaxis, :],
        perturbation=np.array([0.0]),
        perturbation_labels=("independent",),
        intensity_unit=record.confirmed_input_unit,
        source_name=source.original_filename,
        metadata={
            "workflow_mode": "independent_batch", "perturbation_semantics": "not_applicable",
            "spectrum_id": spectrum_id, "workspace_id": workspace.workspace_id,
            "source_id": record.source_id, "original_filename": source.original_filename,
            "original_column_index": record.original_column_index,
            "original_column_label": record.original_column_label,
            "source_sha256": record.scientific_input_sha256,
            "original_source_sha256": source.original_bytes_sha256,
            "source_selection_sha256": source_selection_fingerprint(workspace, spectrum_id),
            "import_options": thaw_mapping(source.import_options),
            "import_probe": thaw_mapping(source.import_probe),
            "original_axis_direction": record.original_axis_direction,
        },
    )


def run_singleton(workspace: BatchWorkspace, spectrum_id: str, config: PipelineConfig,
                  *, runner: Runner | None = None) -> PipelineResult:
    checked = require_independent_config(config)
    data = to_spectrum_set(workspace, spectrum_id)
    if data.n_spectra != 1:
        raise BatchError("PIPELINE_FAILED", "independent adapter requires exactly one spectrum")
    return (runner or run_pipeline)(data, checked)


def _preview(workspace: BatchWorkspace, spectrum_id: str, stage: Stage,
             runner: Runner | None) -> StageSnapshot:
    state = workspace.states[spectrum_id]
    record = workspace.records[spectrum_id]
    try:
        validate_preparation(record, state.preparation_committed)
        parent = None
        if stage == "coarse":
            config = coarse_config(state.preparation_committed, state.coarse_draft)
            candidates = (state.coarse_preview, state.coarse_snapshot)
        else:
            parent = get_ready_snapshot(workspace, spectrum_id, "coarse")
            if not state.fine_draft.enabled or state.fine_draft.method == "none":
                raise BatchError("FINE_REQUIRED", "use explicit skip to confirm no fine adjustment")
            validate_fine(record, state.preparation_committed, state.fine_draft)
            config = fine_config(parent.config, state.fine_draft)
            candidates = (state.fine_preview, state.fine_snapshot)
        parent_hash = None if parent is None else parent.fingerprint
        fingerprint = stage_fingerprint(workspace, spectrum_id, config, stage, parent_hash)
        snapshot = next((item for item in candidates if item is not None and
                         item.fingerprint == fingerprint), None)
        if snapshot is None:
            result = run_singleton(workspace, spectrum_id, config, runner=runner)
            snapshot = StageSnapshot(
                stage=stage, spectrum_id=spectrum_id, input_sha256=result.input_sha256,
                config=config, fingerprint=fingerprint, result=result,
                parent_coarse_fingerprint=parent_hash,
                implementation_fingerprint=implementation_fingerprint(),
                parent_coarse_implementation_fingerprint=(
                    None if parent is None else parent.implementation_fingerprint
                ),
            )
        _validate_snapshot(workspace, spectrum_id, snapshot)
        if parent is not None:
            validate_parent(parent, snapshot)
        if stage == "coarse":
            state.coarse_preview = snapshot
        else:
            state.fine_preview = snapshot
        state.errors.clear()
        state.warnings = list(snapshot.result.warnings)
        return snapshot
    except Exception as exc:
        error = exc if isinstance(exc, BatchError) else BatchError("PIPELINE_FAILED", str(exc))
        state.errors[:] = [str(error)]
        raise error from (None if error is exc else exc)


def preview_coarse(workspace: BatchWorkspace, spectrum_id: str,
                   *, runner: Runner | None = None) -> StageSnapshot:
    return _preview(workspace, spectrum_id, "coarse", runner)


def preview_fine(workspace: BatchWorkspace, spectrum_id: str,
                 *, runner: Runner | None = None) -> StageSnapshot:
    return _preview(workspace, spectrum_id, "fine", runner)


def preview_selected(workspace: BatchWorkspace, spectrum_ids: Sequence[str], *, stage: Stage,
                     runner: Runner | None = None) -> dict[str, str | None]:
    if stage not in {"coarse", "fine"}:
        raise ValueError("stage must be coarse or fine")
    outcomes: dict[str, str | None] = {}
    for spectrum_id in dict.fromkeys(spectrum_ids):
        try:
            _preview(workspace, spectrum_id, stage, runner)
            outcomes[spectrum_id] = None
        except (BatchError, KeyError) as exc:
            outcomes[spectrum_id] = str(exc)
    return outcomes


class IndependentBatchBaselineService:
    """Injectable runner facade; all state/cache lifetime belongs to the workspace."""

    def __init__(self, runner: Runner | None = None) -> None:
        self.runner = runner

    def preview_coarse(self, workspace: BatchWorkspace, spectrum_id: str) -> StageSnapshot:
        return preview_coarse(workspace, spectrum_id, runner=self.runner)

    def preview_fine(self, workspace: BatchWorkspace, spectrum_id: str) -> StageSnapshot:
        return preview_fine(workspace, spectrum_id, runner=self.runner)

    def preview_selected(self, workspace: BatchWorkspace, spectrum_ids: Sequence[str], *,
                         stage: Stage) -> dict[str, str | None]:
        return preview_selected(workspace, spectrum_ids, stage=stage, runner=self.runner)
