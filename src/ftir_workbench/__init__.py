"""Coordination layer for FTIR baseline and prepared-only 2D-COS workflows."""

from .adapters import (
    prepared_from_baseline_result,
    prepared_scientific_branch_from_baseline_result,
    to_prepared_dataset,
)
from .config import (
    BaselineWorkflowConfig,
    ImportConfig,
    TwoDCOSConfig,
    TwoDCOSDisplayConfig,
    TwoDCOSRange,
    WorkbenchProjectConfig,
)
from .fingerprints import (
    array_sha256,
    baseline_fingerprint,
    canonical_json_bytes,
    canonical_json_sha256,
    prepared_data_sha256,
    project_fingerprint,
    twodcos_fingerprint,
)
from .models import PreparedSpectralDataset
from .project import Project, WorkbenchProject, WorkflowState
from .services import (
    BaselineWorkflowService,
    CrossPreparedConfirmationRequired,
    CrossRangeResult,
    HomoRangeResult,
    ProjectService,
    TwoDCOSAnalysisResult,
    TwoDCOSWorkflowService,
    WorkbenchProjectService,
    analyze_peak_order,
    compute_cross_from_prepared,
    compute_homo_from_prepared,
    cross_result_fingerprint,
)
from .validation import (
    PreparedDatasetValidationError,
    validate_cross_prepared_compatibility,
    validate_prepared_dataset,
)
from .workflow import ChangeScope, InvalidWorkflowTransition

__version__ = "0.1.0"

__all__ = [
    "BaselineWorkflowConfig",
    "BaselineWorkflowService",
    "ChangeScope",
    "CrossPreparedConfirmationRequired",
    "CrossRangeResult",
    "HomoRangeResult",
    "ImportConfig",
    "InvalidWorkflowTransition",
    "PreparedDatasetValidationError",
    "PreparedSpectralDataset",
    "Project",
    "ProjectService",
    "TwoDCOSAnalysisResult",
    "TwoDCOSConfig",
    "TwoDCOSDisplayConfig",
    "TwoDCOSRange",
    "TwoDCOSWorkflowService",
    "WorkbenchProject",
    "WorkbenchProjectConfig",
    "WorkbenchProjectService",
    "WorkflowState",
    "analyze_peak_order",
    "array_sha256",
    "baseline_fingerprint",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "compute_cross_from_prepared",
    "compute_homo_from_prepared",
    "cross_result_fingerprint",
    "prepared_data_sha256",
    "prepared_from_baseline_result",
    "prepared_scientific_branch_from_baseline_result",
    "project_fingerprint",
    "to_prepared_dataset",
    "twodcos_fingerprint",
    "validate_cross_prepared_compatibility",
    "validate_prepared_dataset",
]
