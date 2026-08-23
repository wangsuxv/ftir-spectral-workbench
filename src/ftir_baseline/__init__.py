"""Scientific core for FTIR Baseline Workbench."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .models import BaselineResult, SpectrumSet
from .validation import SpectrumValidationError

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "AnchorWindowConfig": "config",
    "CoarseBaselineConfig": "config",
    "FineBaselineConfig": "config",
    "NormalizationConfig": "config",
    "PipelineConfig": "config",
    "SmoothingConfig": "config",
    "InvalidTransmittanceError": "units",
    "UnitConversionRecord": "units",
    "UnitConversionResult": "units",
    "convert_to_absorbance": "units",
    "to_absorbance": "units",
    "SpectrumReadError": "io",
    "load_spectrum_directory": "io",
    "load_spectrum_files": "io",
    "read_spectrum_file": "io",
    "crop_spectrum_set": "ranges",
    "orient_spectrum_set": "ranges",
    "restore_original_axis": "ranges",
    "BaselineEstimationChannels": "smoothing",
    "prepare_baseline_channels": "smoothing",
    "smooth_for_baseline": "smoothing",
    "collaborative_pls_baseline": "baseline",
    "compose_baselines": "baseline",
    "estimate_coarse": "baseline",
    "estimate_fine": "baseline",
    "estimate_series_baseline": "baseline",
    "shared_shape_baseline": "baseline",
    "NormalizationResult": "normalization",
    "apply_normalization": "normalization",
    "normalize_spectra": "normalization",
    "QCResult": "qc",
    "generate_qc_figures": "qc",
    "run_quality_control": "qc",
    "PipelineResult": "pipeline",
    "pipeline_result_fingerprint": "pipeline",
    "run_pipeline": "pipeline",
    "CandidateGallery": "gallery",
    "CandidateSpec": "gallery",
    "default_candidate_specs": "gallery",
    "scan_baseline_candidates": "gallery",
    "build_export_zip": "export",
    "export_result": "export",
    "verify_export_manifest": "export",
    "SYNTHETIC_SCENARIOS": "synthetic",
    "SyntheticDataset": "synthetic",
    "calculate_benchmark_metrics": "synthetic",
    "generate_synthetic_ftir": "synthetic",
    "run_synthetic_benchmarks": "synthetic",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "SYNTHETIC_SCENARIOS",
    "AnchorWindowConfig",
    "BaselineEstimationChannels",
    "BaselineResult",
    "CandidateGallery",
    "CandidateSpec",
    "CoarseBaselineConfig",
    "FineBaselineConfig",
    "InvalidTransmittanceError",
    "NormalizationConfig",
    "NormalizationResult",
    "PipelineConfig",
    "PipelineResult",
    "QCResult",
    "SmoothingConfig",
    "SpectrumReadError",
    "SpectrumSet",
    "SpectrumValidationError",
    "SyntheticDataset",
    "UnitConversionRecord",
    "UnitConversionResult",
    "apply_normalization",
    "build_export_zip",
    "calculate_benchmark_metrics",
    "collaborative_pls_baseline",
    "compose_baselines",
    "convert_to_absorbance",
    "crop_spectrum_set",
    "default_candidate_specs",
    "estimate_coarse",
    "estimate_fine",
    "estimate_series_baseline",
    "export_result",
    "generate_qc_figures",
    "generate_synthetic_ftir",
    "load_spectrum_directory",
    "load_spectrum_files",
    "normalize_spectra",
    "orient_spectrum_set",
    "pipeline_result_fingerprint",
    "prepare_baseline_channels",
    "read_spectrum_file",
    "restore_original_axis",
    "run_pipeline",
    "run_quality_control",
    "run_synthetic_benchmarks",
    "scan_baseline_candidates",
    "shared_shape_baseline",
    "smooth_for_baseline",
    "to_absorbance",
    "verify_export_manifest",
]
