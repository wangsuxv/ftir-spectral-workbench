"""Command-line interface for the same :func:`run_pipeline` used by the UI."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .export import export_result, verify_export_manifest
from .gallery import (
    default_candidate_specs,
    scan_baseline_candidates,
    starter_pchip_anchor_windows,
)
from .io import TextImportOptions, load_spectrum_directory, load_spectrum_files, read_spectrum_file
from .models import SpectrumSet
from .pipeline import run_pipeline
from .ranges import crop_spectrum_set
from .units import IntensityUnit, convert_to_absorbance

UNITS = ("absorbance", "percent_transmittance", "fraction_transmittance")


def _candidate_anchor_windows(config: PipelineConfig, x: np.ndarray) -> list[dict[str, Any]]:
    explicit = [anchor.to_dict() for anchor in config.fine_baseline.anchors if anchor.enabled]
    return explicit or list(
        starter_pchip_anchor_windows(
            x,
            endpoint_window_width_cm1=config.fine_baseline.endpoint_window_width_cm1,
            statistic=config.fine_baseline.statistic,
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON recipe {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("recipe JSON root must be an object")
    nested = payload.get("config")
    return dict(nested) if isinstance(nested, dict) else payload


def load_recipe(path: str | Path) -> PipelineConfig:
    payload = _read_json(Path(path))
    if hasattr(PipelineConfig, "model_validate"):
        return PipelineConfig.model_validate(payload)
    return PipelineConfig.parse_obj(payload)  # pragma: no cover - Pydantic 1


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    if getattr(args, "recipe", None):
        config = load_recipe(args.recipe)
        if args.unit is not None and args.unit != config.input_unit:
            raise ValueError("--unit conflicts with input_unit recorded in --recipe")
        payload = config.to_dict()
    else:
        if args.unit is None:
            raise ValueError("input unit confirmation is mandatory: pass --unit or a JSON --recipe")
        payload = PipelineConfig(input_unit=args.unit).to_dict()

    if getattr(args, "wavenumber_range", None):
        payload["wavenumber_range"] = [float(value) for value in args.wavenumber_range]
    if getattr(args, "series_mode", None):
        payload["series_mode"] = args.series_mode
    coarse = dict(payload.get("coarse_baseline", {}))
    if getattr(args, "coarse_method", None):
        coarse["method"] = args.coarse_method
    if getattr(args, "lam", None) is not None:
        coarse["lambda"] = float(args.lam)
    payload["coarse_baseline"] = coarse
    if hasattr(PipelineConfig, "model_validate"):
        return PipelineConfig.model_validate(payload)
    return PipelineConfig.parse_obj(payload)  # pragma: no cover - Pydantic 1


def _load_inputs(args: argparse.Namespace, unit: IntensityUnit) -> SpectrumSet:
    paths = [Path(value) for value in args.inputs]
    import_options = _text_import_options(args)
    if len(paths) == 1 and paths[0].is_dir():
        exclude = () if args.include_baseline_file else ("BASELINE.dpt",)
        return load_spectrum_directory(
            paths[0],
            input_unit=unit,
            exclude_names=exclude,
            sort_by_perturbation=bool(args.sort_by_perturbation),
            import_options=import_options,
        )
    if len(paths) == 1:
        return read_spectrum_file(
            paths[0],
            input_unit=unit,
            sort_by_perturbation=bool(args.sort_by_perturbation),
            import_options=import_options,
        )
    return load_spectrum_files(
        paths,
        input_unit=unit,
        exclude_names=() if args.include_baseline_file else ("BASELINE.dpt",),
        sort_by_perturbation=bool(args.sort_by_perturbation),
        import_options=import_options,
    )


def _summary(result: Any, output: Path | None = None) -> dict[str, Any]:
    summary = {
        "source": result.raw_input.source_name,
        "input_sha256": result.input_sha256,
        "n_spectra": result.absorbance_selected.n_spectra,
        "n_points_full": result.absorbance_full.n_points,
        "n_points_selected": result.absorbance_selected.n_points,
        "selected_wavenumber_min": float(np.min(result.absorbance_selected.wavenumber)),
        "selected_wavenumber_max": float(np.max(result.absorbance_selected.wavenumber)),
        "series_mode": result.config.series_mode,
        "coarse_method": result.config.coarse_baseline.method,
        "fine_method": result.config.fine_baseline.method,
        "normalization": result.normalization.method,
        "qc": dict(result.qc.summary),
        "warning_count": len(result.warnings),
        "warnings": list(result.warnings),
    }
    if output is not None:
        summary["output_zip"] = str(output)
        summary["manifest_verified"] = verify_export_manifest(output)
    return summary


def command_run(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    data = _load_inputs(args, config.input_unit)
    result = run_pipeline(data, config)
    output = export_result(result, args.output, filename=args.filename)
    print(json.dumps(_summary(result, output), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    if args.unit is None:
        raise ValueError("--unit is mandatory; the program does not infer physical units")
    data = _load_inputs(args, args.unit)
    information = {
        "source": data.source_name,
        "n_spectra": data.n_spectra,
        "n_points": data.n_points,
        "shape": list(data.spectra.shape),
        "wavenumber_min": float(np.min(data.wavenumber)),
        "wavenumber_max": float(np.max(data.wavenumber)),
        "axis_direction": data.axis_direction,
        "intensity_unit_confirmed": data.intensity_unit,
        "intensity_min": float(np.min(data.spectra)),
        "intensity_max": float(np.max(data.spectra)),
        "perturbation": data.perturbation.tolist(),
        "labels": list(data.perturbation_labels),
        "metadata": data.mutable_metadata(),
    }
    print(json.dumps(information, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def command_init_recipe(args: argparse.Namespace) -> int:
    config = PipelineConfig(input_unit=args.unit)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(config.to_json(indent=2) + "\n", encoding="utf-8")
    print(str(destination.resolve()))
    return 0


def command_scan(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    data = _load_inputs(args, config.input_unit)
    conversion = convert_to_absorbance(
        data.spectra,
        config.input_unit,
        transmittance_floor=config.transmittance_floor,
    )
    absorbance = SpectrumSet(
        wavenumber=data.wavenumber,
        perturbation=data.perturbation,
        perturbation_labels=data.perturbation_labels,
        spectra=conversion.absorbance,
        intensity_unit="absorbance",
        source_name=data.source_name,
        metadata=data.mutable_metadata(),
    )
    selected = crop_spectrum_set(absorbance, config.wavenumber_range, strict_bounds=True)
    anchors = _candidate_anchor_windows(config, selected.wavenumber)
    specs = default_candidate_specs(
        anchor_windows=anchors,
        arpls_log10_lambda=args.arpls_grid,
        asls_log10_lambda=args.asls_grid,
        asls_p=args.asls_p,
        airpls_log10_lambda=args.airpls_grid,
        endpoint_window_width_cm1=config.fine_baseline.endpoint_window_width_cm1,
    )
    gallery = scan_baseline_candidates(
        selected.wavenumber,
        selected.spectra,
        specs,
        representative=args.representative,
        smoothing=config.baseline_smoothing,
        anchor_windows=anchors,
    )
    rows = [
        {
            "rank": rank,
            "name": item.name,
            "score": item.score,
            **dict(
                next(
                    evaluation.qc.summary
                    for evaluation in gallery.evaluations
                    if evaluation.name == item.name
                )
            ),
        }
        for rank, item in enumerate(gallery.ranking, start=1)
    ]
    print(
        json.dumps(
            {
                "representative": gallery.representative_name,
                "disclaimer": gallery.disclaimer,
                "candidates": rows,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


def command_demo(args: argparse.Namespace) -> int:
    source = Path(args.input_dir)
    config = PipelineConfig(input_unit="absorbance")
    data = load_spectrum_directory(
        source,
        input_unit="absorbance",
        exclude_names=("BASELINE.dpt",),
        sort_by_perturbation=True,
        source_name="local DPT series",
        import_options=_text_import_options(args),
    )
    result = run_pipeline(data, config)
    output_dir = Path(args.output)
    output = export_result(result, output_dir, filename="ftir_baseline_demo.zip")
    anchors = _candidate_anchor_windows(config, result.absorbance_selected.wavenumber)
    gallery = scan_baseline_candidates(
        result.absorbance_selected.wavenumber,
        result.absorbance_selected.spectra,
        default_candidate_specs(anchor_windows=anchors),
        representative="median",
        anchor_windows=anchors,
    )
    evaluations = {item.name: item for item in gallery.evaluations}
    candidate_rows = [
        {
            "rank": rank,
            "candidate": ranked.name,
            "heuristic_score": ranked.score,
            "qc": dict(evaluations[ranked.name].qc.summary),
            "warnings": list(evaluations[ranked.name].result.warnings),
        }
        for rank, ranked in enumerate(gallery.ranking, start=1)
    ]
    candidate_path = output_dir / "candidate_gallery.json"
    candidate_path.write_text(
        json.dumps(
            {
                "representative": gallery.representative_name,
                "disclaimer": gallery.disclaimer,
                "candidates": candidate_rows,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.extract:
        extracted = output_dir / "ftir_baseline_demo"
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output) as archive:
            archive.extractall(extracted)
    summary = _summary(result, output)
    summary["candidate_gallery"] = str(candidate_path.resolve())
    summary["candidate_gallery_count"] = len(candidate_rows)
    summary_path = output_dir / "demo_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def _add_input_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "inputs",
        nargs="+",
        help="plain-text spectrum files (.csv/.tsv/.tab/.txt/.dpt/.asc/.dat/.xy) or one directory",
    )
    parser.add_argument("--unit", choices=UNITS, help="explicitly confirmed input unit")
    parser.add_argument(
        "--sort-by-perturbation",
        action="store_true",
        help="explicitly sort by the number parsed from each label/filename",
    )
    parser.add_argument(
        "--include-baseline-file",
        action="store_true",
        help="include BASELINE.dpt as a spectrum (normally excluded from a time series)",
    )
    _add_text_import_arguments(parser)


def _add_text_import_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--delimiter",
        choices=("auto", "comma", "tab", "semicolon", "whitespace"),
        default="auto",
        help="text delimiter (default: detect automatically)",
    )
    parser.add_argument(
        "--decimal-mark",
        choices=("auto", "dot", "comma"),
        default="auto",
        help="decimal mark (comma is valid only with a non-comma delimiter)",
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
        help="text encoding (default: BOM-aware automatic detection)",
    )
    parser.add_argument(
        "--header",
        choices=("auto", "present", "absent"),
        default="auto",
        help="whether a column header is present immediately before the numeric block",
    )
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        metavar="N",
        help="skip exactly N leading physical rows before parsing",
    )
    parser.add_argument(
        "--trim-empty-edge-columns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="trim columns that are empty on every parsed row (default: enabled)",
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
    parser = argparse.ArgumentParser(
        prog="ftir-baseline",
        description="Inspectable baseline correction for in-situ FTIR series",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="validate and describe input")
    _add_input_options(inspect_parser)
    inspect_parser.set_defaults(handler=command_inspect)

    init_parser = subparsers.add_parser("init-recipe", help="write a complete starter recipe")
    init_parser.add_argument("--unit", required=True, choices=UNITS)
    init_parser.add_argument("--output", default="recipe.json")
    init_parser.set_defaults(handler=command_init_recipe)

    run_parser = subparsers.add_parser("run", help="run the pipeline and export a ZIP")
    _add_input_options(run_parser)
    run_parser.add_argument("--recipe", help="JSON recipe or recipe from a previous export")
    run_parser.add_argument("--range", dest="wavenumber_range", nargs=2, type=float)
    run_parser.add_argument(
        "--series-mode",
        choices=("independent_locked", "collaborative_pls", "shared_shape"),
    )
    run_parser.add_argument(
        "--coarse-method",
        choices=(
            "none",
            "offset",
            "linear",
            "arpls",
            "asls",
            "airpls",
            "rubberband",
            "pspline_arpls",
        ),
    )
    run_parser.add_argument("--lambda", dest="lam", type=float)
    run_parser.add_argument("--output", default="output")
    run_parser.add_argument("--filename")
    run_parser.set_defaults(handler=command_run)

    scan_parser = subparsers.add_parser("scan", help="compare the prescribed parameter grid")
    _add_input_options(scan_parser)
    scan_parser.add_argument("--recipe")
    scan_parser.add_argument("--range", dest="wavenumber_range", nargs=2, type=float)
    scan_parser.add_argument(
        "--series-mode", choices=("independent_locked", "collaborative_pls", "shared_shape")
    )
    scan_parser.add_argument("--coarse-method")
    scan_parser.add_argument("--lambda", dest="lam", type=float)
    scan_parser.add_argument("--representative", default="median")
    scan_parser.add_argument("--arpls-grid", nargs="+", type=float, default=[3, 4, 5, 6, 7, 8, 9])
    scan_parser.add_argument("--asls-grid", nargs="+", type=float, default=[4, 5, 6, 7, 8, 9])
    scan_parser.add_argument("--asls-p", nargs="+", type=float, default=[0.001, 0.01, 0.05])
    scan_parser.add_argument("--airpls-grid", nargs="+", type=float, default=[4, 5, 6, 7, 8])
    scan_parser.set_defaults(handler=command_scan)

    demo_parser = subparsers.add_parser("demo", help="run a local raw DPT series")
    demo_parser.add_argument("--input-dir", default="data/original")
    demo_parser.add_argument("--output", default="outputs/baseline-demo")
    demo_parser.add_argument("--extract", action=argparse.BooleanOptionalAction, default=True)
    _add_text_import_arguments(demo_parser)
    demo_parser.set_defaults(handler=command_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
