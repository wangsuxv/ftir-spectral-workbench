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
from .cross_views import (
    BlockKind,
    CrossOrientation,
    FullBlockOverview,
    OrientedCrossView,
    full_block_overview,
    oriented_cross_views,
)
from .display_units import (
    DisplayConversionResult,
    DisplayIntensityUnit,
    absorbance_to_fraction_transmittance,
    absorbance_to_percent_transmittance,
    convert_absorbance_for_display,
    derived_transmittance_csv_bytes,
    derived_transmittance_filename,
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
from .post_baseline_smoothing import (
    PostBaselineSmoothingConfig,
    PostBaselineSmoothingResult,
    apply_post_baseline_smoothing,
    post_baseline_smoothing_fingerprint,
)
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

__version__ = "0.2.1"

__all__ = [
    "BaselineWorkflowConfig",
    "BaselineWorkflowService",
    "BlockKind",
    "ChangeScope",
    "CrossOrientation",
    "CrossPreparedConfirmationRequired",
    "CrossRangeResult",
    "DisplayConversionResult",
    "DisplayIntensityUnit",
    "FullBlockOverview",
    "HomoRangeResult",
    "ImportConfig",
    "InvalidWorkflowTransition",
    "OrientedCrossView",
    "PostBaselineSmoothingConfig",
    "PostBaselineSmoothingResult",
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
    "absorbance_to_fraction_transmittance",
    "absorbance_to_percent_transmittance",
    "analyze_peak_order",
    "apply_post_baseline_smoothing",
    "array_sha256",
    "baseline_fingerprint",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "compute_cross_from_prepared",
    "compute_homo_from_prepared",
    "convert_absorbance_for_display",
    "cross_result_fingerprint",
    "derived_transmittance_csv_bytes",
    "derived_transmittance_filename",
    "full_block_overview",
    "oriented_cross_views",
    "post_baseline_smoothing_fingerprint",
    "prepared_data_sha256",
    "prepared_from_baseline_result",
    "prepared_scientific_branch_from_baseline_result",
    "project_fingerprint",
    "to_prepared_dataset",
    "twodcos_fingerprint",
    "validate_cross_prepared_compatibility",
    "validate_prepared_dataset",
]
