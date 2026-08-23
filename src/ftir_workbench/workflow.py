"""State transitions and scientific-result invalidation rules.

All functions in this module are pure: a transition returns a new immutable
project snapshot and never mutates a result in place.  This makes dependency
changes reviewable and keeps display-only edits out of the scientific graph.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from .project import WorkbenchProject, WorkflowState


class InvalidWorkflowTransition(ValueError):
    """Raised when a caller skips a required scientific workflow state."""


class ChangeScope(StrEnum):
    """Dependency scope of a project change."""

    RAW_DATA = "raw_data"
    BASELINE_SCIENCE = "baseline_science"
    PREPARED_SCIENCE = "prepared_science"
    TWODCOS_SCIENCE = "twodcos_science"
    DISPLAY_ONLY = "display_only"


_FORWARD_TRANSITIONS: dict[WorkflowState, frozenset[WorkflowState]] = {
    WorkflowState.EMPTY: frozenset({WorkflowState.RAW_IMPORTED}),
    WorkflowState.RAW_IMPORTED: frozenset({WorkflowState.BASELINE_CONFIGURED}),
    WorkflowState.BASELINE_CONFIGURED: frozenset({WorkflowState.BASELINE_COMPLETED}),
    WorkflowState.BASELINE_COMPLETED: frozenset({WorkflowState.PREPARED_FOR_2DCOS}),
    WorkflowState.PREPARED_FOR_2DCOS: frozenset({WorkflowState.TWODCOS_CONFIGURED}),
    WorkflowState.TWODCOS_CONFIGURED: frozenset({WorkflowState.TWODCOS_COMPLETED}),
    WorkflowState.TWODCOS_COMPLETED: frozenset(),
}


def can_transition(current: WorkflowState | str, target: WorkflowState | str) -> bool:
    """Return whether ``target`` is the next explicit forward state."""

    source_state = WorkflowState(current)
    target_state = WorkflowState(target)
    return target_state in _FORWARD_TRANSITIONS[source_state]


def transition_state(
    current: WorkflowState | str,
    target: WorkflowState | str,
) -> WorkflowState:
    """Validate and return a forward state transition."""

    source_state = WorkflowState(current)
    target_state = WorkflowState(target)
    if source_state is target_state:
        return target_state
    if not can_transition(source_state, target_state):
        raise InvalidWorkflowTransition(
            f"cannot transition from {source_state.name} to {target_state.name}"
        )
    return target_state


def transition_project(
    project: WorkbenchProject,
    target: WorkflowState | str,
    **changes: Any,
) -> WorkbenchProject:
    """Advance a project by one valid state and apply associated values."""

    target_state = transition_state(project.state, target)
    return project.with_updates(state=target_state, **changes)


def _configured_or_imported_state(project: WorkbenchProject) -> WorkflowState:
    if project.raw_data is None:
        return WorkflowState.EMPTY
    if project.baseline_config is not None:
        return WorkflowState.BASELINE_CONFIGURED
    return WorkflowState.RAW_IMPORTED


def invalidate_project(
    project: WorkbenchProject,
    scope: ChangeScope | str,
    *,
    warning: str | None = None,
) -> WorkbenchProject:
    """Invalidate exactly the results downstream of ``scope``.

    Raw/baseline changes clear baseline, prepared, and 2D results.  A prepared
    scientific branch clears 2D only but keeps the completed baseline result.
    A 2D scientific change clears only matrices.  Display-only changes preserve
    every scientific object and the current state.
    """

    normalized_scope = ChangeScope(scope)
    warnings = project.warnings
    if warning is not None:
        warnings = (*warnings, str(warning))

    if normalized_scope is ChangeScope.DISPLAY_ONLY:
        return project.with_updates(warnings=warnings)

    if normalized_scope is ChangeScope.RAW_DATA:
        state = WorkflowState.RAW_IMPORTED if project.raw_data is not None else WorkflowState.EMPTY
        return project.with_updates(
            state=state,
            baseline_result=None,
            prepared=None,
            twodcos_config=None,
            twodcos_result=None,
            baseline_exported=False,
            warnings=warnings,
        )

    if normalized_scope is ChangeScope.BASELINE_SCIENCE:
        return project.with_updates(
            state=_configured_or_imported_state(project),
            baseline_result=None,
            prepared=None,
            twodcos_config=None,
            twodcos_result=None,
            baseline_exported=False,
            warnings=warnings,
        )

    if normalized_scope is ChangeScope.PREPARED_SCIENCE:
        return project.with_updates(
            state=(
                WorkflowState.BASELINE_COMPLETED
                if project.baseline_result is not None
                else _configured_or_imported_state(project)
            ),
            prepared=None,
            twodcos_config=None,
            twodcos_result=None,
            warnings=warnings,
        )

    # TWODCOS_SCIENCE
    if project.prepared is None:
        state = (
            WorkflowState.BASELINE_COMPLETED
            if project.baseline_result is not None
            else _configured_or_imported_state(project)
        )
    elif project.twodcos_config is None:
        state = WorkflowState.PREPARED_FOR_2DCOS
    else:
        state = WorkflowState.TWODCOS_CONFIGURED
    return project.with_updates(
        state=state,
        twodcos_result=None,
        warnings=warnings,
    )


def scientific_twodcos_config(config: Any | None) -> Any:
    """Return a display-free representation for change detection."""

    if config is None:
        return None
    scientific_dict = getattr(config, "scientific_dict", None)
    if callable(scientific_dict):
        return scientific_dict()
    to_dict = getattr(config, "to_dict", None)
    payload = to_dict() if callable(to_dict) else config
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.pop("display", None)
        payload.pop("contour_levels", None)
        payload.pop("display_percentile", None)
    return payload


def twodcos_science_changed(old: Any | None, new: Any | None) -> bool:
    """Compare only fields that influence 2D matrices."""

    return scientific_twodcos_config(old) != scientific_twodcos_config(new)


# Short aliases make the state-machine API pleasant in functional callers.
invalidate = invalidate_project
transition = transition_project


__all__ = [
    "ChangeScope",
    "InvalidWorkflowTransition",
    "can_transition",
    "invalidate",
    "invalidate_project",
    "scientific_twodcos_config",
    "transition",
    "transition_project",
    "transition_state",
    "twodcos_science_changed",
]
