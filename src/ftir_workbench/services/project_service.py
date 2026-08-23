"""High-level project facade with explicit state and invalidation semantics."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..config import (
    BaselineWorkflowConfig,
    ImportConfig,
    TwoDCOSConfig,
    TwoDCOSDisplayConfig,
    WorkbenchProjectConfig,
)
from ..models import PreparedSpectralDataset
from ..project import WorkbenchProject, WorkflowState
from ..validation import validate_prepared_dataset
from ..workflow import ChangeScope, InvalidWorkflowTransition, twodcos_science_changed
from .twodcos_service import TwoDCOSAnalysisResult, TwoDCOSWorkflowService


def _project_config(project: WorkbenchProject) -> WorkbenchProjectConfig:
    return project.config or WorkbenchProjectConfig()


def _baseline_scientific_payload(config: BaselineWorkflowConfig | None) -> object:
    if config is None:
        return None
    to_dict = getattr(config, "to_dict", None)
    payload = to_dict() if callable(to_dict) else config
    if isinstance(payload, dict):
        payload = dict(payload)
        # ftir_baseline keeps this as a non-destructive view/optional branch;
        # PreparedSpectralDataset uses analysis_data, never view_data.
        payload.pop("normalization", None)
    return payload


class WorkbenchProjectService:
    """Maintain the current immutable project snapshot.

    Methods update ``project`` synchronously and return the new snapshot so the
    same API works in CLI, Streamlit session state, or a future desktop model.
    No export or 2D calculation is triggered implicitly.
    """

    def __init__(self, project: WorkbenchProject | None = None) -> None:
        self._project = project or WorkbenchProject(config=WorkbenchProjectConfig())

    @property
    def project(self) -> WorkbenchProject:
        return self._project

    def _commit(self, project: WorkbenchProject) -> WorkbenchProject:
        self._project = project
        return project

    def configure_import(self, config: ImportConfig) -> WorkbenchProject:
        """Set unit/order decisions and invalidate every downstream result."""

        normalized = config if isinstance(config, ImportConfig) else ImportConfig.from_dict(config)
        old_config = _project_config(self.project)
        if old_config.import_config == normalized:
            return self.project
        combined = replace(old_config, import_config=normalized)
        state = WorkflowState.RAW_IMPORTED if self.project.raw_data is not None else WorkflowState.EMPTY
        return self._commit(
            self.project.with_updates(
                config=combined,
                state=state,
                baseline_result=None,
                prepared=None,
                twodcos_config=None,
                twodcos_result=None,
                baseline_exported=False,
            )
        )

    def import_raw(self, raw_data: Any) -> WorkbenchProject:
        """Record new raw data and invalidate all derived science."""

        if raw_data is None:
            raise ValueError("raw_data must not be None")
        return self._commit(
            self.project.with_updates(
                state=WorkflowState.RAW_IMPORTED,
                raw_data=raw_data,
                baseline_result=None,
                prepared=None,
                twodcos_config=None,
                twodcos_result=None,
                baseline_exported=False,
            )
        )

    def configure_baseline(
        self,
        config: BaselineWorkflowConfig,
    ) -> WorkbenchProject:
        """Record the authoritative baseline recipe and clear derived results."""

        if self.project.raw_data is None:
            raise InvalidWorkflowTransition("baseline configuration requires imported raw data")
        old_project_config = _project_config(self.project)
        combined = replace(old_project_config, baseline=config)
        existing_science = _baseline_scientific_payload(self.project.baseline_config)
        new_science = _baseline_scientific_payload(config)
        if existing_science == new_science and self.project.baseline_config is not None:
            if self.project.baseline_config == config:
                return self.project
            # A view-normalization-only edit does not alter analysis_data and
            # therefore must preserve the prepared object and matrices.
            return self._commit(
                self.project.with_updates(
                    config=combined,
                    baseline_config=config,
                )
            )
        return self._commit(
            self.project.with_updates(
                config=combined,
                baseline_config=config,
                state=WorkflowState.BASELINE_CONFIGURED,
                baseline_result=None,
                prepared=None,
                twodcos_config=None,
                twodcos_result=None,
                baseline_exported=False,
            )
        )

    def complete_baseline(self, result: Any) -> WorkbenchProject:
        """Record a successful baseline result without starting optional 2D."""

        if result is None:
            raise ValueError("baseline result must not be None")
        if self.project.state is not WorkflowState.BASELINE_CONFIGURED:
            raise InvalidWorkflowTransition(
                "baseline completion requires state BASELINE_CONFIGURED"
            )
        return self._commit(
            self.project.with_updates(
                state=WorkflowState.BASELINE_COMPLETED,
                baseline_result=result,
                prepared=None,
                twodcos_config=None,
                twodcos_result=None,
                baseline_exported=False,
            )
        )

    def export_baseline_and_stop(self) -> WorkbenchProject:
        """Mark baseline-only completion; deliberately imports/calls no 2D code."""

        if self.project.baseline_result is None:
            raise InvalidWorkflowTransition("baseline export requires a completed baseline")
        return self._commit(self.project.with_updates(baseline_exported=True))

    def prepare_for_twodcos(
        self,
        prepared: PreparedSpectralDataset,
    ) -> WorkbenchProject:
        """Attach the in-memory handoff, or start directly from corrected absorbance."""

        if not isinstance(prepared, PreparedSpectralDataset):
            raise TypeError("prepared must be a PreparedSpectralDataset")
        validate_prepared_dataset(prepared)
        return self._commit(
            self.project.with_updates(
                state=WorkflowState.PREPARED_FOR_2DCOS,
                prepared=prepared,
                twodcos_config=None,
                twodcos_result=None,
            )
        )

    # Direct-from-corrected mode uses the same contract and state.
    load_prepared = prepare_for_twodcos

    def continue_to_twodcos(self) -> WorkbenchProject:
        """Adapt the current in-memory baseline result and enter optional 2D.

        No CSV handoff or second baseline execution occurs.  The adapter is the
        single authority for selecting ``PipelineResult.analysis_data``.
        """

        if self.project.baseline_result is None:
            raise InvalidWorkflowTransition(
                "continuing to 2D requires a completed baseline result"
            )
        from ..adapters import prepared_from_baseline_result

        prepared = prepared_from_baseline_result(self.project.baseline_result)
        return self.prepare_for_twodcos(prepared)

    continue_with_current_baseline = continue_to_twodcos

    def configure_twodcos(self, config: TwoDCOSConfig) -> WorkbenchProject:
        """Set 2D options, preserving matrices when only display fields changed."""

        if self.project.prepared is None:
            raise InvalidWorkflowTransition("2D configuration requires prepared absorbance")
        if not isinstance(config, TwoDCOSConfig):
            config = TwoDCOSConfig.from_dict(config)
        old = self.project.twodcos_config
        science_changed = twodcos_science_changed(old, config)
        combined = replace(_project_config(self.project), twodcos=config)
        if not science_changed and old is not None:
            # This includes contour levels and display percentile.  Preserve the
            # exact matrix container and completed state.
            return self._commit(
                self.project.with_updates(
                    config=combined,
                    twodcos_config=config,
                )
            )
        return self._commit(
            self.project.with_updates(
                config=combined,
                twodcos_config=config,
                twodcos_result=None,
                state=WorkflowState.TWODCOS_CONFIGURED,
            )
        )

    def update_display_config(
        self,
        display: TwoDCOSDisplayConfig,
    ) -> WorkbenchProject:
        """Change plot settings without invalidating existing matrices."""

        if self.project.twodcos_config is None:
            raise InvalidWorkflowTransition("display configuration requires 2D configuration")
        normalized = (
            display
            if isinstance(display, TwoDCOSDisplayConfig)
            else TwoDCOSDisplayConfig.from_dict(display)
        )
        return self.configure_twodcos(
            replace(self.project.twodcos_config, display=normalized)
        )

    def complete_twodcos(
        self,
        result: TwoDCOSAnalysisResult,
    ) -> WorkbenchProject:
        """Attach matrices only when they belong to the current prepared data."""

        if self.project.prepared is None or self.project.twodcos_config is None:
            raise InvalidWorkflowTransition(
                "2D completion requires prepared data and a 2D configuration"
            )
        if not isinstance(result, TwoDCOSAnalysisResult):
            raise TypeError("result must be a TwoDCOSAnalysisResult")
        prepared = self.project.prepared
        if result.parent_prepared_data_sha256 != prepared.prepared_data_sha256:
            raise ValueError("2D result parent prepared fingerprint is stale")
        if result.parent_baseline_fingerprint != prepared.baseline_fingerprint:
            raise ValueError("2D result parent baseline fingerprint is stale")
        if twodcos_science_changed(self.project.twodcos_config, result.config):
            raise ValueError("2D result was computed with a different scientific config")
        return self._commit(
            self.project.with_updates(
                state=WorkflowState.TWODCOS_COMPLETED,
                twodcos_result=result,
            )
        )

    def run_twodcos(
        self,
        config: TwoDCOSConfig | None = None,
        *,
        service: TwoDCOSWorkflowService | None = None,
    ) -> WorkbenchProject:
        """Explicitly compute 2D from the current prepared object in memory."""

        if config is not None:
            self.configure_twodcos(config)
        if self.project.prepared is None or self.project.twodcos_config is None:
            raise InvalidWorkflowTransition(
                "run_twodcos requires prepared data and a 2D configuration"
            )
        runner = service or TwoDCOSWorkflowService()
        result = runner.compute(self.project.prepared, self.project.twodcos_config)
        return self.complete_twodcos(result)

    def invalidate(
        self,
        scope: ChangeScope,
        *,
        warning: str | None = None,
    ) -> WorkbenchProject:
        """Expose explicit invalidation for UI/controller dependency changes."""

        from ..workflow import invalidate_project

        return self._commit(invalidate_project(self.project, scope, warning=warning))


# Naming used in the implementation specification and a concise convenience alias.
ProjectService = WorkbenchProjectService


__all__ = ["ProjectService", "WorkbenchProjectService"]
