"""Run two requested wavenumber intervals on a local DPT directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from ftir2dcos.config import (
    BaselineConfig,
    NormalizationConfig,
    PipelineConfig,
    SmoothingConfig,
    WavenumberRange,
)
from ftir2dcos.pipeline import run_multi_range_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_RANGES = (
    WavenumberRange(
        high_wavenumber=1736.0,
        low_wavenumber=1509.0,
        label="amide_1736_1509",
    ),
    WavenumberRange(
        high_wavenumber=1250.0,
        low_wavenumber=1140.0,
        label="fingerprint_1250_1140",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Build a small wrapper around the shared scientific pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Run local *MIN.dpt spectra over 1736-1509 and 1250-1140 "
            "cm^-1, then export one aggregate result bundle."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "original",
        help="Directory containing the two-column *MIN.dpt spectra.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "legacy-twodcos",
        help="Root directory for the timestamped multi-range result directory.",
    )
    return parser


def main() -> int:
    """Execute the demo without duplicating any preprocessing or 2D-COS code."""

    args = build_parser().parse_args()
    config = PipelineConfig(
        input_intensity_unit="absorbance",
        perturbation_order="sort_by_perturbation",
        baseline=BaselineConfig(method="none"),
        smoothing=SmoothingConfig(enabled=False),
        normalization=NormalizationConfig(method="none"),
        convention="2dpy_compatible",
    )
    result = run_multi_range_pipeline(
        args.input_dir,
        ANALYSIS_RANGES,
        config,
        output_root=args.output,
        dpt_pattern="*MIN.dpt",
    )
    output_directory = result.output_directory
    if output_directory is None:  # pragma: no cover - defensive only
        raise RuntimeError("The pipeline completed without creating an output directory")
    bundle_path = result.bundle_path
    if bundle_path is None or not bundle_path.is_file():  # pragma: no cover - defensive only
        raise RuntimeError("The pipeline completed without creating the aggregate ZIP")
    block_figure_paths = (
        output_directory / "figures" / "multi_range_synchronous_blocks.png",
        output_directory / "figures" / "multi_range_synchronous_blocks.pdf",
        output_directory / "figures" / "multi_range_asynchronous_blocks.png",
        output_directory / "figures" / "multi_range_asynchronous_blocks.pdf",
    )
    missing_figures = [path for path in block_figure_paths if not path.is_file()]
    if missing_figures:  # pragma: no cover - defensive only
        missing = ", ".join(map(str, missing_figures))
        raise RuntimeError(f"The pipeline did not create the full block figures: {missing}")

    print(f"Multi-range result directory: {output_directory}")
    for item in result.range_results:
        axis = item.result.processed.wavenumber
        actual_range = f"{axis.max():g}-{axis.min():g} cm^-1"
        qc_status = "PASS" if item.result.qc_metrics["all_checks_passed"] else "FAIL"
        print(
            f"- {item.analysis_range.display_name}: actual={actual_range}; "
            f"spectra x wavenumbers={item.result.processed.spectra.shape}; QC={qc_status}"
        )
    print(f"Cross-range correlations: {result.cross_count} unique pair(s)")
    for item in result.cross_results:
        analysis = item.twodcos
        row_axis = analysis.row_wavenumber
        column_axis = analysis.column_wavenumber
        row_actual = f"{row_axis.max():g}-{row_axis.min():g} cm^-1"
        column_actual = f"{column_axis.max():g}-{column_axis.min():g} cm^-1"
        qc_status = "PASS" if item.qc_metrics["all_checks_passed"] else "FAIL"
        print(
            f"- {item.pair_label}: rows={item.row_range.display_name} [{row_actual}]; "
            f"columns={item.column_range.display_name} [{column_actual}]; "
            f"matrix={analysis.synchronous.shape}; QC={qc_status}"
        )
    print(f"All self/cross checks passed: {result.all_checks_passed}")
    print("Full block figures:")
    for path in block_figure_paths:
        print(f"- {path}")
    print(f"Aggregate ZIP: {bundle_path}")
    print(f"Warnings: {len(result.warnings)}")
    for warning in result.warnings:
        print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
