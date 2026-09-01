"""Command-line entry point for the unified baseline-first workflow."""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from ftir_baseline.config import PipelineConfig
from ftir_baseline.export import verify_export_manifest as verify_baseline_bundle
from ftir_baseline.io import TextImportOptions, load_spectrum_directory, read_spectrum_file

from .config import TwoDCOSConfig, TwoDCOSRange
from .export import (
    build_baseline_bundle,
    build_project_bundle,
    build_twodcos_bundle,
    load_prepared,
    verify_project_bundle,
    verify_twodcos_bundle,
    verify_workbench_manifest,
)
from .post_baseline_smoothing import ConvolutionMode, PostBaselineSmoothingConfig
from .services.baseline_service import BaselineWorkflowService
from .services.twodcos_service import TwoDCOSWorkflowService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "original"


def _range(value: str) -> TwoDCOSRange:
    normalized = value.replace(",", ":")
    parts = [part.strip() for part in normalized.split(":")]
    if len(parts) not in {2, 3} or not all(parts[:2]):
        raise argparse.ArgumentTypeError("range must be HIGH:LOW or HIGH:LOW:LABEL")
    try:
        high, low = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("range endpoints must be numbers") from exc
    try:
        return TwoDCOSRange(high, low, None if len(parts) == 2 else parts[2])
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _baseline_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        input_unit=args.unit,
        wavenumber_range=(args.baseline_high, args.baseline_low),
        baseline_smoothing={
            "enabled": bool(args.smoothing),
            "window_length": args.smoothing_window,
            "polyorder": args.smoothing_polyorder,
            "estimate_only": True,
        },
        coarse_baseline={
            "method": args.coarse_method,
            "lambda": args.lam,
            "p": args.asls_p,
            "max_iter": args.max_iter,
            "tol": args.tol,
        },
        fine_baseline={
            "enabled": args.fine_method != "none",
            "method": args.fine_method,
            "endpoint_window_width_cm1": args.endpoint_width,
            "statistic": "median",
            "strict_endpoint": False,
            "anchors": [],
        },
        normalization={"method": "none"},
        series_mode=args.series_mode,
        restore_descending_axis_on_export=True,
    )


def _load_raw(args: argparse.Namespace) -> Any:
    source = Path(args.input).expanduser()
    import_options = _text_import_options(args)
    if source.is_dir():
        return load_spectrum_directory(
            source,
            input_unit=args.unit,
            exclude_names=("BASELINE.dpt",),
            sort_by_perturbation=bool(args.sort_by_perturbation),
            source_name=source.name,
            import_options=import_options,
        )
    return read_spectrum_file(
        source,
        input_unit=args.unit,
        sort_by_perturbation=bool(args.sort_by_perturbation),
        import_options=import_options,
    )


def _write_bundle(output: str | Path, name: str, payload: bytes) -> Path:
    destination = Path(output).expanduser()
    if destination.suffix.lower() in {".zip", ".ftirw"}:
        path = destination
    else:
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path.resolve()


def _add_baseline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--unit", choices=("absorbance", "percent_transmittance", "fraction_transmittance"), default="absorbance")
    parser.add_argument("--baseline-high", type=float, default=1800.0)
    parser.add_argument("--baseline-low", type=float, default=900.0)
    parser.add_argument("--coarse-method", choices=("none", "offset", "linear", "arpls", "asls", "airpls", "rubberband", "pspline_arpls"), default="arpls")
    parser.add_argument("--lam", type=float, default=1_000_000.0)
    parser.add_argument("--asls-p", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=0.001)
    parser.add_argument("--fine-method", choices=("none", "endpoint_window_linear"), default="endpoint_window_linear")
    parser.add_argument("--endpoint-width", type=float, default=8.0)
    parser.add_argument("--series-mode", choices=("independent_locked", "collaborative_pls", "shared_shape"), default="collaborative_pls")
    parser.add_argument("--smoothing", action="store_true", help="Use an estimate-only Savitzky–Golay channel.")
    parser.add_argument("--smoothing-window", type=int, default=7)
    parser.add_argument("--smoothing-polyorder", type=int, default=2)
    parser.add_argument("--sort-by-perturbation", action=argparse.BooleanOptionalAction, default=True)
    _add_text_import_arguments(parser)


def _add_text_import_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--delimiter",
        choices=("auto", "comma", "tab", "semicolon", "whitespace"),
        default="auto",
        help="Text delimiter (default: detect automatically).",
    )
    parser.add_argument(
        "--decimal-mark",
        choices=("auto", "dot", "comma"),
        default="auto",
        help="Decimal mark (comma is valid only with a non-comma delimiter).",
    )
    parser.add_argument(
        "--encoding",
        choices=(
            "auto",
            "utf-8",
            "utf-8-sig",
            "utf-16",
            "utf-16-le",
            "utf-16-be",
            "gb18030",
            "cp1252",
        ),
        default="auto",
        help="Text encoding (default: BOM-aware automatic detection).",
    )
    parser.add_argument(
        "--header",
        choices=("auto", "present", "absent"),
        default="auto",
        help="Whether a column header is present immediately before the numeric block.",
    )
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        metavar="N",
        help="Skip exactly N leading physical rows before parsing.",
    )
    parser.add_argument(
        "--trim-empty-edge-columns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim columns that are empty on every parsed row (default: enabled).",
    )


def _add_smoothing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--method",
        choices=("savgol", "gaussian", "moving_average", "median"),
        required=True,
        help="Post-baseline smoothing algorithm (the command always creates an enabled branch).",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        help="Odd point window for savgol, moving_average, or median.",
    )
    parser.add_argument(
        "--polyorder",
        type=int,
        help="Savitzky–Golay polynomial order.",
    )
    parser.add_argument(
        "--sigma-points",
        type=float,
        help="Gaussian sigma measured in points.",
    )
    parser.add_argument(
        "--truncate",
        type=float,
        help="Gaussian kernel truncation in sigma units.",
    )
    parser.add_argument(
        "--mode",
        choices=("interp", "reflect", "mirror", "nearest"),
        help=(
            "Boundary mode: interp/mirror/nearest for savgol; "
            "reflect/mirror/nearest for the other methods."
        ),
    )
    parser.add_argument(
        "--uniformity-rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for approximately uniform wavenumber spacing.",
    )
    parser.add_argument(
        "--nonuniform-axis-policy",
        choices=("error", "allow_index_space_with_warning"),
        default="error",
        help="Reject a nonuniform axis or explicitly smooth it in index space with a warning.",
    )


def _text_import_options(args: argparse.Namespace) -> TextImportOptions:
    return TextImportOptions(
        delimiter=args.delimiter,
        decimal_mark=args.decimal_mark,
        encoding=args.encoding,
        header_mode=args.header,
        skip_rows=args.skip_rows,
        trim_empty_edge_columns=args.trim_empty_edge_columns,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ftir-workbench", description="Baseline-first FTIR processing with optional prepared-only 2D-COS")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Validate and summarize raw FTIR input")
    inspect_parser.add_argument("input", type=Path)
    _add_baseline_arguments(inspect_parser)
    inspect_parser.set_defaults(handler=_command_inspect)

    baseline_parser = subparsers.add_parser("baseline", help="Run and export a baseline-only workflow")
    baseline_parser.add_argument("input", type=Path)
    baseline_parser.add_argument("--output", type=Path, default=Path("outputs"))
    _add_baseline_arguments(baseline_parser)
    baseline_parser.set_defaults(handler=_command_baseline)

    twodcos_help = (
        "Run 2D-COS from a prepared CSV, sidecar, baseline ZIP, "
        "or smoothing bundle"
    )
    twodcos_parser = subparsers.add_parser(
        "twodcos",
        help=twodcos_help,
        description=twodcos_help,
    )
    twodcos_parser.add_argument("input", type=Path)
    twodcos_parser.add_argument("--metadata", type=Path)
    twodcos_parser.add_argument("--range", dest="ranges", type=_range, action="append", required=True)
    twodcos_parser.add_argument("--convention", choices=("canonical", "2dpy_compatible"), default="2dpy_compatible")
    twodcos_parser.add_argument("--nonuniform-policy", choices=("warn", "allow", "error"), default="warn")
    twodcos_parser.add_argument("--no-cross", action="store_true")
    twodcos_parser.add_argument("--output", type=Path, default=Path("outputs"))
    twodcos_parser.set_defaults(handler=_command_twodcos)

    smooth_help = (
        "Create an explicit post-baseline smoothed Prepared branch and bundle"
    )
    smooth_parser = subparsers.add_parser(
        "smooth",
        help=smooth_help,
        description=smooth_help,
    )
    smooth_parser.add_argument(
        "input",
        type=Path,
        help="Prepared CSV/sidecar, baseline ZIP, or smoothing ZIP.",
    )
    smooth_parser.add_argument("--metadata", type=Path)
    smooth_parser.add_argument("--output", type=Path, default=Path("outputs"))
    _add_smoothing_arguments(smooth_parser)
    smooth_parser.set_defaults(handler=_command_smooth)

    demo_parser = subparsers.add_parser("demo", help="Run a local DPT series through baseline and optional 2D demonstration")
    demo_parser.add_argument("--input-dir", dest="input", type=Path, default=DEFAULT_DATA)
    demo_parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "real-data-demo")
    demo_parser.add_argument("--baseline-only", action="store_true")
    demo_parser.add_argument("--convention", choices=("canonical", "2dpy_compatible"), default="2dpy_compatible")
    _add_baseline_arguments(demo_parser)
    demo_parser.set_defaults(handler=_command_demo)

    verify_parser = subparsers.add_parser("verify", help="Verify a workbench ZIP/.ftirw manifest")
    verify_parser.add_argument("bundle", type=Path)
    verify_parser.set_defaults(handler=_command_verify)
    return parser


def _dataset_summary(data: Any) -> dict[str, Any]:
    return {
        "source": data.source_name,
        "unit": data.intensity_unit,
        "spectra": data.n_spectra,
        "points": data.n_points,
        "wavenumber_min": float(np.min(data.wavenumber)),
        "wavenumber_max": float(np.max(data.wavenumber)),
        "axis_direction": data.axis_direction,
        "perturbation_first": float(data.perturbation[0]),
        "perturbation_last": float(data.perturbation[-1]),
        "perturbation_uniform": bool(
            data.n_spectra < 3
            or np.allclose(np.diff(data.perturbation), np.diff(data.perturbation)[0])
        ),
    }


def _command_inspect(args: argparse.Namespace) -> int:
    print(json.dumps(_dataset_summary(_load_raw(args)), ensure_ascii=False, indent=2))
    return 0


def _run_baseline(args: argparse.Namespace) -> tuple[Any, Any, bytes]:
    data = _load_raw(args)
    service = BaselineWorkflowService()
    result = service.run(data, _baseline_config(args))
    prepared = service.prepared(result)
    bundle = build_baseline_bundle(result, prepared=prepared)
    return result, prepared, bundle


def _command_baseline(args: argparse.Namespace) -> int:
    result, prepared, bundle = _run_baseline(args)
    path = _write_bundle(args.output, "baseline_run.zip", bundle)
    payload = {
        **_dataset_summary(result.absorbance_selected),
        "state": "baseline_completed",
        "baseline_run_id": prepared.baseline_run_id,
        "baseline_fingerprint": prepared.baseline_fingerprint,
        "prepared_data_sha256": prepared.prepared_data_sha256,
        "bundle": str(path),
        "manifest_verified": verify_workbench_manifest(bundle),
        "warnings": list(prepared.warnings),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _twodcos_config(args: argparse.Namespace) -> TwoDCOSConfig:
    return TwoDCOSConfig(
        ranges=tuple(args.ranges),
        convention=args.convention,
        nonuniform_perturbation_policy=args.nonuniform_policy,
        cross_range_enabled=not args.no_cross,
    )


def _command_twodcos(args: argparse.Namespace) -> int:
    prepared = load_prepared(args.input, metadata=args.metadata)
    config = _twodcos_config(args)
    result = TwoDCOSWorkflowService().compute(prepared, config)
    bundle = build_twodcos_bundle(prepared, result, config)
    path = _write_bundle(args.output, "twodcos_run.zip", bundle)
    print(
        json.dumps(
            {
                "state": "twodcos_completed",
                "self_results": len(result.homo_results),
                "cross_results": len(result.cross_results),
                "all_checks_passed": result.all_checks_passed,
                "twodcos_fingerprint": result.twodcos_fingerprint,
                "bundle": str(path),
                "manifest_verified": verify_workbench_manifest(bundle),
                "warnings": list(result.warnings),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


def _smoothing_config(args: argparse.Namespace) -> PostBaselineSmoothingConfig:
    method = str(args.method)
    supplied = {
        "window_length": args.window_length,
        "polyorder": args.polyorder,
        "sigma_points": args.sigma_points,
        "truncate": args.truncate,
    }
    allowed_parameters = {
        "savgol": frozenset({"window_length", "polyorder"}),
        "gaussian": frozenset({"sigma_points", "truncate"}),
        "moving_average": frozenset({"window_length"}),
        "median": frozenset({"window_length"}),
    }
    invalid = sorted(
        name
        for name, value in supplied.items()
        if value is not None and name not in allowed_parameters[method]
    )
    if invalid:
        rendered = ", ".join(f"--{name.replace('_', '-')}" for name in invalid)
        raise ValueError(f"{rendered} is not valid with --method {method}")

    mode = args.mode
    if method == "savgol":
        if mode == "reflect":
            raise ValueError(
                "--mode reflect is not valid with --method savgol; "
                "choose interp, mirror, or nearest"
            )
        return PostBaselineSmoothingConfig(
            enabled=True,
            method="savgol",
            savgol_window_length=(7 if args.window_length is None else args.window_length),
            savgol_polyorder=(2 if args.polyorder is None else args.polyorder),
            savgol_mode="interp" if mode is None else mode,
            uniformity_rtol=args.uniformity_rtol,
            nonuniform_axis_policy=args.nonuniform_axis_policy,
        )

    if mode == "interp":
        raise ValueError(
            f"--mode interp is not valid with --method {method}; "
            "choose reflect, mirror, or nearest"
        )
    convolution_mode = cast(ConvolutionMode, "reflect" if mode is None else mode)
    if method == "gaussian":
        return PostBaselineSmoothingConfig(
            enabled=True,
            method="gaussian",
            gaussian_sigma_points=(
                1.0 if args.sigma_points is None else args.sigma_points
            ),
            gaussian_truncate=4.0 if args.truncate is None else args.truncate,
            convolution_mode=convolution_mode,
            uniformity_rtol=args.uniformity_rtol,
            nonuniform_axis_policy=args.nonuniform_axis_policy,
        )
    if method == "moving_average":
        return PostBaselineSmoothingConfig(
            enabled=True,
            method="moving_average",
            moving_average_window_length=(
                3 if args.window_length is None else args.window_length
            ),
            convolution_mode=convolution_mode,
            uniformity_rtol=args.uniformity_rtol,
            nonuniform_axis_policy=args.nonuniform_axis_policy,
        )
    return PostBaselineSmoothingConfig(
        enabled=True,
        method="median",
        median_window_length=3 if args.window_length is None else args.window_length,
        convolution_mode=convolution_mode,
        uniformity_rtol=args.uniformity_rtol,
        nonuniform_axis_policy=args.nonuniform_axis_policy,
    )


def _command_smooth(args: argparse.Namespace) -> int:
    config = _smoothing_config(args)

    from .services.smoothing_service import PostBaselineSmoothingService
    from .smoothing_export import verify_smoothing_bundle

    parent = load_prepared(args.input, metadata=args.metadata)
    service = PostBaselineSmoothingService()
    result, prepared = service.apply(parent, config)
    bundle = service.build_bundle(result, prepared)
    if not verify_smoothing_bundle(bundle):
        raise RuntimeError("generated smoothing bundle failed verification")
    path = _write_bundle(args.output, "post_baseline_smoothing_run.zip", bundle)
    scientific_config = result.config.scientific_dict()
    print(
        json.dumps(
            {
                "state": "smoothing_completed",
                "method": result.config.method,
                "effective_parameters": scientific_config["parameters"],
                "parent_prepared_data_sha256": parent.prepared_data_sha256,
                "prepared_data_sha256": prepared.prepared_data_sha256,
                "smoothing_fingerprint": result.smoothing_fingerprint,
                "bundle": str(path),
                "manifest_verified": True,
                "warnings": list(result.warnings),
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


def _command_demo(args: argparse.Namespace) -> int:
    baseline_result, prepared, baseline_bundle = _run_baseline(args)
    baseline_path = _write_bundle(args.output, "baseline_run.zip", baseline_bundle)
    payload: dict[str, Any] = {
        "state": "baseline_completed",
        "baseline_bundle": str(baseline_path),
        "baseline_fingerprint": prepared.baseline_fingerprint,
        "prepared_data_sha256": prepared.prepared_data_sha256,
        "baseline_qc": dict(baseline_result.qc.summary),
    }
    if not args.baseline_only:
        config = TwoDCOSConfig(
            ranges=(
                TwoDCOSRange(1736.0, 1509.0, "amide_1736_1509"),
                TwoDCOSRange(1250.0, 1140.0, "fingerprint_1250_1140"),
            ),
            convention=args.convention,
            nonuniform_perturbation_policy="warn",
            cross_range_enabled=True,
        )
        result = TwoDCOSWorkflowService().compute(prepared, config)
        twodcos_bundle = build_twodcos_bundle(prepared, result, config)
        twodcos_path = _write_bundle(args.output, "twodcos_run.zip", twodcos_bundle)
        project_bundle = build_project_bundle(
            baseline_bundle,
            twodcos_bundles=(twodcos_bundle,),
            project_config={"baseline": _baseline_config(args).to_dict(), "twodcos": config.to_dict()},
        )
        project_path = _write_bundle(args.output, "project.ftirw", project_bundle)
        payload.update(
            {
                "state": "twodcos_completed",
                "twodcos_bundle": str(twodcos_path),
                "project_bundle": str(project_path),
                "self_results": len(result.homo_results),
                "cross_results": len(result.cross_results),
                "all_checks_passed": result.all_checks_passed,
                "twodcos_fingerprint": result.twodcos_fingerprint,
                "warnings": list(result.warnings),
            }
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _command_verify(args: argparse.Namespace) -> int:
    bundle_type = "generic"
    artifact_type = ""
    if args.bundle.suffix.lower() == ".ftirw":
        bundle_type = "project"
    else:
        try:
            payload = args.bundle.read_bytes()
            with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
                manifest = json.loads(archive.read("manifest.json"))
            artifact_type = str(manifest.get("artifact_type", ""))
            if artifact_type == "ftir_workbench_project":
                bundle_type = "project"
            elif artifact_type == "twodcos_run":
                bundle_type = "twodcos"
            elif artifact_type == "post_baseline_smoothing_run":
                bundle_type = "smoothing"
            elif "prepared_spectrum" in manifest:
                bundle_type = "baseline"
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, json.JSONDecodeError):
            pass
    if bundle_type == "project":
        verified = verify_project_bundle(args.bundle)
    elif bundle_type == "twodcos":
        verified = verify_twodcos_bundle(args.bundle)
    elif bundle_type == "smoothing":
        from .smoothing_export import verify_smoothing_bundle

        verified = verify_smoothing_bundle(args.bundle)
    elif bundle_type == "baseline":
        verified = verify_baseline_bundle(args.bundle)
    else:
        verified = verify_workbench_manifest(args.bundle)
    print(
        json.dumps(
            {
                "bundle": str(args.bundle.resolve()),
                "bundle_type": bundle_type,
                "verified": verified,
            },
            indent=2,
        )
    )
    return 0 if verified else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
