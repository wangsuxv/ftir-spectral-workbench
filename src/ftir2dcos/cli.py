"""Command-line interface for the shared FTIR 2D-COS pipeline.

The CLI accepts either a plain :class:`~ftir2dcos.config.PipelineConfig` JSON
object or a wrapper with input/output settings, optional ``ranges``, and a
``pipeline`` object.  Explicit command-line values override JSON values. Paths
originating in a wrapper are resolved relative to the JSON file; paths entered
on the command line retain normal shell/current-directory semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig, WavenumberRange
from .peak_order import PeakRequest
from .pipeline import run_multi_range_pipeline, run_pipeline

_WRAPPER_KEYS = frozenset(
    {
        "input",
        "output",
        "dpt_pattern",
        "ranges",
        "peaks",
        "peak_analysis",
        "pipeline",
    }
)
_WRAPPER_ONLY_KEYS = _WRAPPER_KEYS - {"pipeline"}
_PEAK_ANALYSIS_KEYS = frozenset(
    {
        "match_tolerance_cm-1",
        "synchronous_threshold",
        "asynchronous_threshold",
        "relative_threshold",
        "analysis_order_note",
    }
)


class CLIError(ValueError):
    """A user-facing command-line or configuration error."""


@dataclass(frozen=True, slots=True)
class _Invocation:
    source: Path
    output_root: Path
    dpt_pattern: str
    delimiter: str | None
    perturbation: tuple[float, ...] | None
    config: PipelineConfig
    ranges: tuple[WavenumberRange, ...]
    peaks: tuple[PeakRequest, ...]
    peak_match_tolerance: float
    synchronous_threshold: float
    asynchronous_threshold: float
    relative_threshold: float
    analysis_order_note: str | None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def _fraction(value: str) -> float:
    parsed = _non_negative_float(value)
    if parsed > 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1 inclusive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser without importing Streamlit."""

    parser = argparse.ArgumentParser(
        prog="ftir2dcos",
        description=(
            "Preprocess FTIR spectra and calculate single-range or all-pairs "
            "cross-range 2D-COS matrices."
        ),
        epilog=(
            "Configuration precedence: explicit CLI options override JSON values. "
            "Ranges use: repeated --range, then wrapper JSON ranges, then legacy "
            "pipeline low/high fields. A plain PipelineConfig JSON still requires "
            "--input. Wrapper-relative paths are resolved from the JSON file directory."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON wrapper or plain PipelineConfig file",
    )

    input_group = parser.add_argument_group("input and output")
    input_group.add_argument(
        "--input",
        type=Path,
        help="wide CSV/TXT/TSV file, DPT file, or directory of DPT files",
    )
    input_group.add_argument(
        "--output",
        type=Path,
        help="output root (default: wrapper value, otherwise ./results)",
    )
    input_group.add_argument(
        "--dpt-pattern",
        help="case-insensitive filename pattern for a DPT directory (default: *MIN.dpt)",
    )
    input_group.add_argument(
        "--delimiter",
        choices=("auto", "comma", "tab", "semicolon"),
        help="wide-table delimiter (default: detect comma, tab, or semicolon)",
    )
    input_group.add_argument(
        "--perturbation",
        nargs="+",
        type=float,
        metavar="VALUE",
        help="explicit perturbation values in input spectrum order",
    )

    pipeline_group = parser.add_argument_group("pipeline")
    pipeline_group.add_argument(
        "--range",
        dest="wavenumber_ranges",
        action="append",
        nargs="+",
        metavar="HIGH:LOW[:LABEL]",
        help=(
            "inclusive range; repeat for batch analysis. Colon or comma compact syntax "
            "is preferred; legacy '--range HIGH LOW' remains supported. For negative "
            "compact values use '--range=-1:-5[:LABEL]'."
        ),
    )
    pipeline_group.add_argument(
        "--intensity",
        choices=(
            "absorbance",
            "percent_transmittance",
            "fraction_transmittance",
            "unknown",
        ),
        help="input intensity unit; conversion is only done for an explicit transmittance unit",
    )
    pipeline_group.add_argument(
        "--perturbation-order",
        choices=("preserve_file_order", "sort_by_perturbation"),
        help="preserve acquisition/file order by default, or explicitly sort by perturbation",
    )
    pipeline_group.add_argument(
        "--convention",
        choices=("canonical", "2dpy_compatible"),
        help="2D-COS matrix orientation/sign convention",
    )
    pipeline_group.add_argument(
        "--contour-levels",
        type=_positive_int,
        help="number of contour levels used for exported figures",
    )
    pipeline_group.add_argument(
        "--display-percentile",
        type=float,
        help="symmetric display-only color percentile in (0, 100]",
    )

    peak_group = parser.add_argument_group("peak response-order analysis")
    peak_group.add_argument(
        "--peak",
        dest="peaks",
        action="append",
        metavar="WAVENUMBER[:LABEL][@RANGE]",
        help=(
            "peak position to order; repeat at least twice. RANGE is an optional 1-based "
            "analysis-range index required when intervals overlap (examples: 1650:amide-I, "
            "1650:amide-I@1, 1200@2)."
        ),
    )
    peak_group.add_argument(
        "--peak-match-tolerance",
        type=_non_negative_float,
        metavar="CM-1",
        help="maximum distance to the nearest sampled grid point (default: 1 cm^-1)",
    )
    peak_group.add_argument(
        "--sync-threshold",
        type=_non_negative_float,
        metavar="VALUE",
        help="absolute synchronous signal cutoff for an indeterminate pair (default: 0)",
    )
    peak_group.add_argument(
        "--async-threshold",
        type=_non_negative_float,
        metavar="VALUE",
        help="absolute asynchronous signal cutoff for an indeterminate pair (default: 0)",
    )
    peak_group.add_argument(
        "--peak-relative-threshold",
        type=_fraction,
        metavar="FRACTION",
        help=(
            "matrix-relative numerical signal cutoff; the effective cutoff is the larger "
            "of this fraction (0..1) times matrix max-abs and the absolute cutoff "
            "(default: 1e-6)"
        ),
    )
    peak_group.add_argument(
        "--analysis-order-note",
        help=(
            "optional context appended to the automatically derived perturbation-order facts; "
            "it cannot override the stored sequence"
        ),
    )

    baseline_group = parser.add_argument_group("baseline correction")
    baseline_group.add_argument(
        "--baseline",
        dest="baseline_method",
        choices=(
            "none",
            "offset",
            "constant",
            "anchor_polynomial",
            "anchor",
            "asls",
            "rubberband",
        ),
        help="baseline method",
    )
    baseline_group.add_argument(
        "--offset-mode",
        choices=("minimum", "window_median"),
        help="constant/offset estimator",
    )
    baseline_group.add_argument(
        "--offset-window",
        nargs=2,
        type=float,
        metavar=("HIGH", "LOW"),
        help="wavenumber interval used by offset window_median mode",
    )
    baseline_group.add_argument(
        "--anchor-range",
        dest="anchor_ranges",
        action="append",
        nargs=2,
        type=float,
        metavar=("HIGH", "LOW"),
        help="anchor interval; repeat this option to supply multiple intervals",
    )
    baseline_group.add_argument(
        "--polynomial-order",
        type=int,
        choices=(0, 1, 2, 3),
        help="anchor-polynomial order",
    )
    baseline_group.add_argument("--asls-lam", type=float, help="AsLS smoothness lambda")
    baseline_group.add_argument("--asls-p", type=float, help="AsLS asymmetry in (0, 1)")
    baseline_group.add_argument(
        "--asls-diff-order",
        type=_positive_int,
        help="AsLS finite-difference order",
    )
    baseline_group.add_argument(
        "--asls-max-iter",
        type=_positive_int,
        help="AsLS maximum iterations",
    )
    baseline_group.add_argument(
        "--asls-tol",
        type=float,
        help="AsLS convergence tolerance",
    )
    baseline_group.add_argument(
        "--rubberband-segments",
        nargs="+",
        type=int,
        metavar="SEGMENT",
        help="segment count (one value) or explicit segment boundary indices",
    )
    baseline_group.add_argument(
        "--rubberband-lam",
        type=float,
        help="optional rubberband smoothing lambda",
    )
    baseline_group.add_argument(
        "--rubberband-diff-order",
        type=_positive_int,
        help="rubberband smoothing finite-difference order",
    )
    baseline_group.add_argument(
        "--rubberband-smooth-half-window",
        type=_non_negative_int,
        help="optional rubberband pre-smoothing half-window",
    )

    smoothing_group = parser.add_argument_group("optional Savitzky-Golay smoothing")
    smoothing_toggle = smoothing_group.add_mutually_exclusive_group()
    smoothing_toggle.add_argument(
        "--smooth",
        dest="smoothing_enabled",
        action="store_true",
        help="enable smoothing (disabled by default)",
    )
    smoothing_toggle.add_argument(
        "--no-smooth",
        dest="smoothing_enabled",
        action="store_false",
        help="explicitly disable smoothing",
    )
    smoothing_group.add_argument(
        "--smoothing-window",
        type=_positive_int,
        help="Savitzky-Golay window length",
    )
    smoothing_group.add_argument(
        "--smoothing-polyorder",
        type=_non_negative_int,
        help="Savitzky-Golay polynomial order",
    )
    smoothing_group.add_argument(
        "--smoothing-mode",
        choices=("interp", "mirror", "constant", "nearest", "wrap"),
        help="Savitzky-Golay edge mode",
    )
    parser.set_defaults(smoothing_enabled=None)

    normalization_group = parser.add_argument_group("optional normalization")
    normalization_group.add_argument(
        "--normalization",
        dest="normalization_method",
        choices=("none", "vector", "reference_peak"),
        help="spectrum-wise normalization method (none by default)",
    )
    normalization_group.add_argument(
        "--reference-peak-range",
        nargs=2,
        type=float,
        metavar=("HIGH", "LOW"),
        help="reference interval required by reference_peak normalization",
    )
    return parser


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CLIError(f"could not read config file {path}: {exc}") from exc
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CLIError(
            f"invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise CLIError(f"config file {path} must contain a JSON object")
    return parsed


def _split_config(
    path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], Path | None]:
    if path is None:
        return {}, {}, None
    config_path = path.expanduser()
    parsed = _load_json(config_path)
    is_wrapper = "pipeline" in parsed or bool(_WRAPPER_ONLY_KEYS.intersection(parsed))
    if not is_wrapper:
        return dict(parsed), {}, config_path

    unknown = set(parsed) - _WRAPPER_KEYS
    if unknown:
        joined = ", ".join(sorted(map(str, unknown)))
        raise CLIError(f"unknown wrapper config field(s): {joined}")
    pipeline = parsed.get("pipeline", {})
    if not isinstance(pipeline, Mapping):
        raise CLIError("wrapper field 'pipeline' must contain a JSON object")
    wrapper = {key: parsed[key] for key in _WRAPPER_ONLY_KEYS if key in parsed}
    return dict(pipeline), wrapper, config_path


def _nested_mapping(values: dict[str, Any], key: str) -> dict[str, Any]:
    current = values.get(key, {})
    if not isinstance(current, Mapping):
        raise CLIError(f"pipeline field {key!r} must contain a JSON object")
    return dict(current)


def _range_number(value: Any, *, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CLIError(f"{context} must be a finite number; got {value!r}") from exc
    if not math.isfinite(number):
        raise CLIError(f"{context} must be a finite number; got {value!r}")
    return number


def _range_label(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CLIError(f"{context} must be a non-empty string or null")
    return value.strip()


def _make_range(
    high_value: Any,
    low_value: Any,
    label_value: Any = None,
    *,
    context: str,
) -> WavenumberRange:
    first = _range_number(high_value, context=f"{context} high wavenumber")
    second = _range_number(low_value, context=f"{context} low wavenumber")
    if first == second:
        raise CLIError(f"{context} boundaries must be different")
    label = _range_label(label_value, context=f"{context} label")
    try:
        return WavenumberRange(
            high_wavenumber=max(first, second),
            low_wavenumber=min(first, second),
            label=label,
        )
    except (TypeError, ValueError) as exc:
        raise CLIError(f"invalid {context}: {exc}") from exc


def _parse_cli_ranges(raw_ranges: Sequence[Sequence[str]]) -> tuple[WavenumberRange, ...]:
    parsed: list[WavenumberRange] = []
    for index, tokens in enumerate(raw_ranges, start=1):
        context = f"--range #{index}"
        parts: list[str]
        if len(tokens) == 1:
            compact = tokens[0]
            delimiter = ":" if ":" in compact else "," if "," in compact else None
            if delimiter is None:
                raise CLIError(
                    f"{context} must use HIGH:LOW[:LABEL], HIGH,LOW[,LABEL], or the "
                    "legacy two-token form HIGH LOW. A hyphen is intentionally not a "
                    "separator because it is ambiguous with negative numbers."
                )
            parts = compact.split(delimiter)
        else:
            parts = list(tokens)
        if len(parts) not in {2, 3}:
            raise CLIError(f"{context} must contain two boundaries and at most one label")
        label = parts[2] if len(parts) == 3 else None
        parsed.append(_make_range(parts[0], parts[1], label, context=context))
    return tuple(parsed)


def _parse_peak_text(raw_value: str, *, context: str) -> PeakRequest:
    compact = str(raw_value).strip()
    if not compact:
        raise CLIError(f"{context} cannot be empty")

    range_index: int | None = None
    peak_part = compact
    if "@" in compact:
        peak_part, separator, range_part = compact.rpartition("@")
        if not separator or not peak_part or not range_part:
            raise CLIError(f"{context} must use WAVENUMBER[:LABEL][@RANGE] with a 1-based RANGE")
        try:
            range_one_based = int(range_part)
        except ValueError as exc:
            raise CLIError(f"{context} RANGE must be a positive 1-based integer") from exc
        if range_one_based < 1:
            raise CLIError(f"{context} RANGE must be a positive 1-based integer")
        range_index = range_one_based - 1

    if ":" in peak_part:
        wavenumber_part, label_part = peak_part.split(":", 1)
        label = label_part.strip()
        if not label:
            raise CLIError(f"{context} label must be non-empty when ':' is present")
    else:
        wavenumber_part = peak_part
        label = None
    wavenumber = _range_number(
        wavenumber_part,
        context=f"{context} wavenumber",
    )
    try:
        return PeakRequest(
            wavenumber=wavenumber,
            label=label,
            range_index=range_index,
        )
    except (TypeError, ValueError) as exc:
        raise CLIError(f"invalid {context}: {exc}") from exc


def _parse_cli_peaks(raw_peaks: Sequence[str] | None) -> tuple[PeakRequest, ...]:
    """Parse repeatable peak specifications with an optional 1-based range suffix."""

    if not raw_peaks:
        return ()
    parsed = tuple(
        _parse_peak_text(raw_value, context=f"--peak #{index}")
        for index, raw_value in enumerate(raw_peaks, start=1)
    )
    if len(parsed) == 1:
        raise CLIError("peak response-order analysis requires at least two --peak options")
    return parsed


def _parse_wrapper_peaks(value: Any) -> tuple[PeakRequest, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise CLIError("wrapper field 'peaks' must be an array with at least two entries")
    entries = list(value)
    if len(entries) < 2:
        raise CLIError("wrapper field 'peaks' must contain at least two entries")

    parsed: list[PeakRequest] = []
    for index, entry in enumerate(entries, start=1):
        context = f"wrapper peak #{index}"
        if isinstance(entry, str):
            parsed.append(_parse_peak_text(entry, context=context))
            continue
        if isinstance(entry, Mapping):
            expected = {"wavenumber", "label", "range_index_one_based"}
            unknown = set(entry) - expected
            if unknown:
                joined = ", ".join(sorted(map(str, unknown)))
                raise CLIError(f"{context} has unknown field(s): {joined}")
            if "wavenumber" not in entry:
                raise CLIError(f"{context} is missing required field 'wavenumber'")
            range_index = None
            if "range_index_one_based" in entry:
                raw_index = entry["range_index_one_based"]
                if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                    raise CLIError(f"{context} range_index_one_based must be a positive integer")
                if raw_index < 1:
                    raise CLIError(f"{context} range_index_one_based must be a positive integer")
                range_index = raw_index - 1
            try:
                parsed.append(
                    PeakRequest(
                        wavenumber=_range_number(
                            entry["wavenumber"],
                            context=f"{context} wavenumber",
                        ),
                        label=entry.get("label"),  # type: ignore[arg-type]
                        range_index=range_index,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise CLIError(f"invalid {context}: {exc}") from exc
            continue
        try:
            parsed.append(PeakRequest(_range_number(entry, context=f"{context} wavenumber")))
        except (TypeError, ValueError) as exc:
            raise CLIError(f"invalid {context}: {exc}") from exc
    return tuple(parsed)


def _parse_wrapper_peak_analysis(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CLIError("wrapper field 'peak_analysis' must contain a JSON object")
    unknown = set(value) - _PEAK_ANALYSIS_KEYS
    if unknown:
        joined = ", ".join(sorted(map(str, unknown)))
        raise CLIError(f"wrapper peak_analysis has unknown field(s): {joined}")

    parsed: dict[str, Any] = {}
    numeric_fields = {
        "match_tolerance_cm-1",
        "synchronous_threshold",
        "asynchronous_threshold",
        "relative_threshold",
    }
    for key in numeric_fields.intersection(value):
        number = _range_number(value[key], context=f"wrapper peak_analysis {key}")
        if number < 0.0:
            raise CLIError(f"wrapper peak_analysis {key} must be non-negative")
        if key == "relative_threshold" and number > 1.0:
            raise CLIError("wrapper peak_analysis relative_threshold must be between 0 and 1")
        parsed[key] = number
    if "analysis_order_note" in value:
        note = value["analysis_order_note"]
        if not isinstance(note, str) or not note.strip():
            raise CLIError("wrapper peak_analysis analysis_order_note must be a non-empty string")
        parsed["analysis_order_note"] = note.strip()
    return parsed


def _is_direct_range_array(values: Sequence[Any]) -> bool:
    return len(values) in {2, 3} and not any(
        isinstance(item, Mapping)
        or (isinstance(item, Sequence) and not isinstance(item, str | bytes | bytearray))
        for item in values[:2]
    )


def _parse_wrapper_ranges(value: Any) -> tuple[WavenumberRange, ...]:
    if isinstance(value, Mapping):
        entries: list[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        direct = list(value)
        if not direct:
            raise CLIError("wrapper field 'ranges' cannot be empty")
        entries = [direct] if _is_direct_range_array(direct) else direct
    else:
        raise CLIError("wrapper field 'ranges' must be a range object or an array of ranges")

    parsed: list[WavenumberRange] = []
    expected_keys = {"high_wavenumber", "low_wavenumber", "label"}
    for index, entry in enumerate(entries, start=1):
        context = f"wrapper range #{index}"
        if isinstance(entry, Mapping):
            unknown = set(entry) - expected_keys
            if unknown:
                joined = ", ".join(sorted(map(str, unknown)))
                raise CLIError(f"{context} has unknown field(s): {joined}")
            missing = {"high_wavenumber", "low_wavenumber"} - set(entry)
            if missing:
                joined = ", ".join(sorted(missing))
                raise CLIError(f"{context} is missing required field(s): {joined}")
            parsed.append(
                _make_range(
                    entry["high_wavenumber"],
                    entry["low_wavenumber"],
                    entry.get("label"),
                    context=context,
                )
            )
            continue
        if not isinstance(entry, Sequence) or isinstance(entry, str | bytes | bytearray):
            raise CLIError(f"{context} must be an object or a two/three-element array")
        parts = list(entry)
        if len(parts) not in {2, 3}:
            raise CLIError(f"{context} must contain two or three elements")
        label = parts[2] if len(parts) == 3 else None
        parsed.append(_make_range(parts[0], parts[1], label, context=context))
    return tuple(parsed)


def _resolve_ranges(
    args: argparse.Namespace,
    wrapper: Mapping[str, Any],
    config: PipelineConfig,
) -> tuple[tuple[WavenumberRange, ...], PipelineConfig]:
    if args.wavenumber_ranges is not None:
        ranges = _parse_cli_ranges(args.wavenumber_ranges)
    elif "ranges" in wrapper:
        ranges = _parse_wrapper_ranges(wrapper["ranges"])
    elif config.wavenumber_range is not None:
        low, high = config.wavenumber_range
        ranges = (_make_range(high, low, context="legacy pipeline wavenumber range"),)
    else:
        ranges = ()

    if ranges:
        config = config.for_range(ranges[0])
    return ranges, config


def _apply_pipeline_overrides(
    pipeline_values: dict[str, Any], args: argparse.Namespace
) -> PipelineConfig:
    values = dict(pipeline_values)

    top_level_overrides = {
        "input_intensity_unit": args.intensity,
        "perturbation_order": args.perturbation_order,
        "convention": args.convention,
        "contour_levels": args.contour_levels,
        "display_percentile": args.display_percentile,
    }
    values.update({key: value for key, value in top_level_overrides.items() if value is not None})

    baseline_overrides: dict[str, Any] = {
        "method": args.baseline_method,
        "offset_mode": args.offset_mode,
        "offset_window": args.offset_window,
        "anchor_ranges": args.anchor_ranges,
        "polynomial_order": args.polynomial_order,
        "asls_lam": args.asls_lam,
        "asls_p": args.asls_p,
        "asls_diff_order": args.asls_diff_order,
        "asls_max_iter": args.asls_max_iter,
        "asls_tol": args.asls_tol,
        "rubberband_lam": args.rubberband_lam,
        "rubberband_diff_order": args.rubberband_diff_order,
        "rubberband_smooth_half_window": args.rubberband_smooth_half_window,
    }
    if args.rubberband_segments is not None:
        baseline_overrides["rubberband_segments"] = (
            args.rubberband_segments[0]
            if len(args.rubberband_segments) == 1
            else tuple(args.rubberband_segments)
        )
    baseline_overrides = {
        key: value for key, value in baseline_overrides.items() if value is not None
    }
    if baseline_overrides:
        baseline = _nested_mapping(values, "baseline")
        baseline.update(baseline_overrides)
        values["baseline"] = baseline

    smoothing_overrides = {
        "enabled": args.smoothing_enabled,
        "window_length": args.smoothing_window,
        "polyorder": args.smoothing_polyorder,
        "mode": args.smoothing_mode,
    }
    smoothing_overrides = {
        key: value for key, value in smoothing_overrides.items() if value is not None
    }
    if smoothing_overrides:
        smoothing = _nested_mapping(values, "smoothing")
        smoothing.update(smoothing_overrides)
        values["smoothing"] = smoothing

    normalization_overrides = {
        "method": args.normalization_method,
        "reference_peak_range": args.reference_peak_range,
    }
    normalization_overrides = {
        key: value for key, value in normalization_overrides.items() if value is not None
    }
    if normalization_overrides:
        normalization = _nested_mapping(values, "normalization")
        normalization.update(normalization_overrides)
        values["normalization"] = normalization

    try:
        return PipelineConfig.from_dict(values)
    except (TypeError, ValueError) as exc:
        raise CLIError(f"invalid pipeline configuration: {exc}") from exc


def _wrapper_path(value: Any, *, name: str, config_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CLIError(f"wrapper field {name!r} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path


def _resolve_invocation(args: argparse.Namespace) -> _Invocation:
    pipeline_values, wrapper, config_path = _split_config(args.config)
    config = _apply_pipeline_overrides(pipeline_values, args)
    ranges, config = _resolve_ranges(args, wrapper, config)
    wrapper_peak_analysis = _parse_wrapper_peak_analysis(wrapper.get("peak_analysis"))
    if args.peaks is not None:
        peaks = _parse_cli_peaks(args.peaks)
    elif "peaks" in wrapper:
        peaks = _parse_wrapper_peaks(wrapper["peaks"])
    else:
        peaks = ()
    explicit_cli_peak_setting = any(
        value is not None
        for value in (
            args.peak_match_tolerance,
            args.sync_threshold,
            args.async_threshold,
            args.peak_relative_threshold,
            args.analysis_order_note,
        )
    )
    if not peaks and (wrapper_peak_analysis or explicit_cli_peak_setting):
        raise CLIError("peak-analysis settings require at least two configured peaks")
    if peaks and not ranges:
        raise CLIError("peak response-order analysis requires at least one analysis range")

    peak_match_tolerance = (
        args.peak_match_tolerance
        if args.peak_match_tolerance is not None
        else wrapper_peak_analysis.get("match_tolerance_cm-1", 1.0)
    )
    synchronous_threshold = (
        args.sync_threshold
        if args.sync_threshold is not None
        else wrapper_peak_analysis.get("synchronous_threshold", 0.0)
    )
    asynchronous_threshold = (
        args.async_threshold
        if args.async_threshold is not None
        else wrapper_peak_analysis.get("asynchronous_threshold", 0.0)
    )
    relative_threshold = (
        args.peak_relative_threshold
        if args.peak_relative_threshold is not None
        else wrapper_peak_analysis.get("relative_threshold", 1.0e-6)
    )
    note_value = (
        args.analysis_order_note
        if args.analysis_order_note is not None
        else wrapper_peak_analysis.get("analysis_order_note")
    )
    analysis_order_note = None if note_value is None else str(note_value).strip()

    if args.input is not None:
        source = args.input.expanduser()
    elif "input" in wrapper and config_path is not None:
        source = _wrapper_path(wrapper["input"], name="input", config_path=config_path)
    else:
        detail = (
            " when --config contains a plain PipelineConfig object"
            if args.config is not None and not wrapper
            else " (or wrapper JSON field 'input')"
        )
        raise CLIError(f"--input is required{detail}")
    if not source.exists():
        raise CLIError(f"input path does not exist: {source}")

    if args.output is not None:
        output_root = args.output.expanduser()
    elif "output" in wrapper and config_path is not None:
        output_root = _wrapper_path(wrapper["output"], name="output", config_path=config_path)
    else:
        output_root = Path("results")

    if args.dpt_pattern is not None:
        dpt_pattern = args.dpt_pattern
    else:
        pattern_value = wrapper.get("dpt_pattern", "*MIN.dpt")
        if not isinstance(pattern_value, str) or not pattern_value.strip():
            raise CLIError("dpt_pattern must be a non-empty string")
        dpt_pattern = pattern_value
    if not dpt_pattern.strip():
        raise CLIError("dpt_pattern must be a non-empty string")

    perturbation = None
    if args.perturbation is not None:
        perturbation = tuple(args.perturbation)
    return _Invocation(
        source=source,
        output_root=output_root,
        dpt_pattern=dpt_pattern,
        delimiter=args.delimiter,
        perturbation=perturbation,
        config=config,
        ranges=ranges,
        peaks=peaks,
        peak_match_tolerance=peak_match_tolerance,
        synchronous_threshold=synchronous_threshold,
        asynchronous_threshold=asynchronous_threshold,
        relative_threshold=relative_threshold,
        analysis_order_note=analysis_order_note,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        invocation = _resolve_invocation(args)
    except CLIError as exc:
        parser.error(str(exc))

    shared_options = {
        "output_root": invocation.output_root,
        "delimiter": invocation.delimiter,
        "perturbation": invocation.perturbation,
        "dpt_pattern": invocation.dpt_pattern,
    }
    multi_mode = len(invocation.ranges) > 1 or bool(invocation.peaks)
    if invocation.peaks:
        shared_options.update(
            {
                "peaks": invocation.peaks,
                "peak_match_tolerance": invocation.peak_match_tolerance,
                "synchronous_threshold": invocation.synchronous_threshold,
                "asynchronous_threshold": invocation.asynchronous_threshold,
                "relative_threshold": invocation.relative_threshold,
                "analysis_order_note": invocation.analysis_order_note,
            }
        )
    try:
        if multi_mode:
            result = run_multi_range_pipeline(
                invocation.source,
                invocation.ranges,
                invocation.config,
                **shared_options,
            )
        else:
            result = run_pipeline(
                invocation.source,
                invocation.config,
                **shared_options,
            )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: error: {exc}\n")

    if multi_mode:
        if result.output_directory is None:
            print("Multi-range analysis complete (no export directory was produced).")
        else:
            print(f"Multi-range analysis complete: {Path(result.output_directory).resolve()}")
        for index, item in enumerate(result.range_results, start=1):
            if item.output_directory is None:
                print(f"Range {index} ({item.analysis_range.display_name}): no export directory")
            else:
                print(
                    f"Range {index} ({item.analysis_range.display_name}): "
                    f"{Path(item.output_directory).resolve()}"
                )
        cross_results = tuple(getattr(result, "cross_results", ()))
        print(f"Cross-range correlations: {len(cross_results)} unique pair(s)")
        for item in cross_results:
            pair_label = getattr(item, "pair_label", "cross-range pair")
            pair_output = getattr(item, "output_directory", None)
            if pair_output is None:
                print(f"Cross pair ({pair_label}): no export directory")
            else:
                print(f"Cross pair ({pair_label}): {Path(pair_output).resolve()}")
        peak_order = getattr(result, "peak_order", None)
        if peak_order is not None:
            peak_count = len(getattr(peak_order, "peaks", ()))
            unresolved_count = len(getattr(peak_order, "unresolved_relations", ()))
            print(
                "Peak response-order analysis: "
                f"{peak_count} peak(s), {unresolved_count} unresolved pair(s)"
            )
            if peak_order.is_unique_total_order:
                rendered_order = " -> ".join(peak.display_label for peak in peak_order.unique_order)
                print(f"Response order (analysis sequence): {rendered_order}")
            elif peak_order.has_cycles:
                print(
                    "Response order: no total order; resolved relations contain directed cycle(s)."
                )
                rendered_cycles = "; ".join(
                    "{" + " | ".join(peak.display_label for peak in group) + "}"
                    for group in peak_order.cyclic_groups
                )
                print(f"Cyclic group(s): {rendered_cycles}")
            else:
                rendered_layers = " -> ".join(
                    (
                        layer[0].display_label
                        if len(layer) == 1
                        else "{" + " | ".join(peak.display_label for peak in layer) + "}"
                    )
                    for layer in peak_order.topological_layers
                )
                print(f"Response-order layers (analysis sequence): {rendered_layers}")
                print("Peaks in the same layer are not ordered relative to one another.")
            if result.output_directory is not None:
                print(
                    "Peak-order JSON: "
                    f"{(Path(result.output_directory) / 'peak_order' / 'peak_order.json').resolve()}"
                )
        if result.bundle_path is not None:
            print(f"Aggregate bundle: {Path(result.bundle_path).resolve()}")
    else:
        if result.output_directory is None:
            print("Analysis complete (no export directory was produced).")
        else:
            print(f"Analysis complete: {Path(result.output_directory).resolve()}")
        if result.bundle_path is not None:
            print(f"Bundle: {Path(result.bundle_path).resolve()}")
    for warning in result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    return 0


__all__ = ["build_parser", "main"]
