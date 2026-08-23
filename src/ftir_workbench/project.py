"""Immutable project snapshots for the unified FTIR workflow.

The project model deliberately stores results as already-computed artifacts.  It
does not run either scientific pipeline; orchestration and invalidation live in
``workflow`` and ``services.project_service``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:  # pragma: no cover - imports are for static checkers only
    from .config import BaselineWorkflowConfig, TwoDCOSConfig, WorkbenchProjectConfig
    from .models import PreparedSpectralDataset
    from .services.twodcos_service import TwoDCOSAnalysisResult


class WorkflowState(StrEnum):
    """Auditable states in the baseline-first, optional-2D workflow."""

    EMPTY = "empty"
    RAW_IMPORTED = "raw_imported"
    BASELINE_CONFIGURED = "baseline_configured"
    BASELINE_COMPLETED = "baseline_completed"
    PREPARED_FOR_2DCOS = "prepared_for_2dcos"
    TWODCOS_CONFIGURED = "twodcos_configured"
    TWODCOS_COMPLETED = "twodcos_completed"


@dataclass(frozen=True, slots=True)
class WorkbenchProject:
    """One immutable snapshot of a workbench project.

    ``baseline_result`` intentionally uses ``Any`` at this boundary so the
    coordination package does not duplicate or wrap the authoritative
    :mod:`ftir_baseline` result model.  The same rule applies to ``raw_data``.
    Prepared and 2D results use the new cross-package contracts.
    """

    project_id: str = field(default_factory=lambda: uuid4().hex)
    state: WorkflowState = WorkflowState.EMPTY
    config: WorkbenchProjectConfig | None = None
    raw_data: Any | None = None
    baseline_config: BaselineWorkflowConfig | None = None
    baseline_result: Any | None = None
    prepared: PreparedSpectralDataset | None = None
    twodcos_config: TwoDCOSConfig | None = None
    twodcos_result: TwoDCOSAnalysisResult | None = None
    baseline_exported: bool = False
    revision: int = 0
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", str(self.project_id))
        object.__setattr__(self, "state", WorkflowState(self.state))
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        if self.revision < 0:
            raise ValueError("revision must be non-negative")
        if self.state is WorkflowState.RAW_IMPORTED and self.raw_data is None:
            raise ValueError("RAW_IMPORTED requires raw_data")
        if self.state is WorkflowState.BASELINE_CONFIGURED and (
            self.raw_data is None or self.baseline_config is None
        ):
            raise ValueError(
                "BASELINE_CONFIGURED requires raw_data and baseline_config"
            )
        if self.state is WorkflowState.BASELINE_COMPLETED and self.baseline_result is None:
            raise ValueError("BASELINE_COMPLETED requires a baseline_result")
        if self.state is WorkflowState.TWODCOS_CONFIGURED and self.twodcos_config is None:
            raise ValueError("TWODCOS_CONFIGURED requires a twodcos_config")
        if self.state is WorkflowState.TWODCOS_COMPLETED and self.twodcos_result is None:
            raise ValueError("TWODCOS_COMPLETED requires a twodcos_result")
        if self.state is WorkflowState.TWODCOS_COMPLETED and self.twodcos_config is None:
            raise ValueError("TWODCOS_COMPLETED requires a twodcos_config")
        if self.state in {
            WorkflowState.PREPARED_FOR_2DCOS,
            WorkflowState.TWODCOS_CONFIGURED,
            WorkflowState.TWODCOS_COMPLETED,
        } and self.prepared is None:
            raise ValueError(f"{self.state.name} requires prepared data")

    @property
    def prepared_data(self) -> PreparedSpectralDataset | None:
        """Readable alias used by adapters and UI clients."""

        return self.prepared

    @property
    def two_dcos_result(self) -> TwoDCOSAnalysisResult | None:
        """Readable alias for callers that spell out ``two_dcos``."""

        return self.twodcos_result

    @property
    def baseline_complete(self) -> bool:
        """Whether a baseline result exists, independent of optional 2D work."""

        return self.baseline_result is not None

    @property
    def can_export_baseline(self) -> bool:
        """Baseline export is a successful terminal action at any later state."""

        return self.baseline_result is not None

    @property
    def can_run_twodcos(self) -> bool:
        """Whether prepared absorbance is available for a 2D calculation."""

        return self.prepared is not None

    def with_updates(self, **changes: Any) -> WorkbenchProject:
        """Return the next immutable project revision."""

        if "revision" in changes:
            raise ValueError("revision is managed by WorkbenchProject.with_updates")
        return replace(self, revision=self.revision + 1, **changes)


# A concise alias is useful for code that already has a Workbench namespace.
Project = WorkbenchProject


__all__ = ["Project", "WorkbenchProject", "WorkflowState"]
