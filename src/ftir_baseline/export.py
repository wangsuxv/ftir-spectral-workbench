"""Deterministic, auditable CSV/HTML/ZIP export for pipeline results."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from importlib import metadata as importlib_metadata
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .pipeline import PROCESSING_ORDER

FloatArray = NDArray[np.float64]
_MISSING = object()
_NO_DEFAULT = object()
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class ExportPayload:
    """Pipeline-shaped payload for callers that do not use ``PipelineResult``."""

    raw_input: Any
    absorbance_full: Any
    absorbance_selected: Any
    baseline: Any
    normalization: Any
    config: Any
    metrics: Any = field(default_factory=dict)
    recipe: Any = None
    sensitivity_branches: Mapping[str, ArrayLike] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    input_hash: str | None = None
    version: str | None = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get(obj: Any, name: str, default: Any = _NO_DEFAULT) -> Any:
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
    elif hasattr(obj, name):
        return getattr(obj, name)
    if default is _NO_DEFAULT:
        raise AttributeError(f"export result is missing required field {name!r}")
    return default


def _first(obj: Any, names: Sequence[str], default: Any = _NO_DEFAULT) -> Any:
    for name in names:
        value = _get(obj, name, _MISSING)
        if value is not _MISSING:
            return value
    if default is _NO_DEFAULT:
        raise AttributeError(f"export result is missing all fields {tuple(names)!r}")
    return default


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json", exclude_none=False))
        except TypeError:
            return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if is_dataclass(value):
        return _jsonable(asdict(cast(Any, value)))
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    return str(value)


def _software_version(explicit: Any = None) -> str:
    if explicit:
        return str(explicit)
    try:
        return importlib_metadata.version("ftir-baseline-workbench")
    except importlib_metadata.PackageNotFoundError:
        try:
            from . import __version__

            return str(__version__)
        except (ImportError, AttributeError):
            return "0+unknown"


def _as_matrix(values: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 1:
        result = result[np.newaxis, :]
    if result.ndim != 2 or result.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_spectra, n_points)")
    invalid = np.argwhere(~np.isfinite(result))
    if invalid.size:
        row, column = invalid[0]
        raise ValueError(f"{name} has NaN or Inf at spectrum {row}, point {column}")
    return result


def _spectrum_fields(spectrum_set: Any, name: str) -> tuple[FloatArray, FloatArray]:
    x = np.asarray(_first(spectrum_set, ("wavenumber", "x")), dtype=np.float64)
    spectra = _as_matrix(_first(spectrum_set, ("spectra", "data", "values")), name)
    if x.ndim != 1 or x.shape != (spectra.shape[1],):
        raise ValueError(f"{name}.wavenumber must have shape ({spectra.shape[1]},)")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name}.wavenumber contains NaN or Inf")
    return x, spectra


def _config_dict(config: Any) -> dict[str, Any]:
    value = _jsonable(config)
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": value}


def _restore_descending(config: Any) -> bool:
    value = _get(config, "restore_descending_axis_on_export", True)
    return bool(value)


def _labels_for(spectrum_set: Any, n_spectra: int) -> tuple[str, ...]:
    labels = _first(
        spectrum_set,
        ("perturbation_labels", "spectrum_labels", "labels"),
        default=(),
    )
    if labels is not None:
        labels_tuple = tuple(str(item) for item in labels)
        if len(labels_tuple) == n_spectra:
            return labels_tuple
    perturbation = _get(spectrum_set, "perturbation", None)
    if perturbation is not None:
        values = np.asarray(perturbation)
        if values.shape == (n_spectra,):
            return tuple(str(item) for item in values.tolist())
    return tuple(f"Spectrum_{index + 1}" for index in range(n_spectra))


def _oriented(
    x: FloatArray,
    spectra: FloatArray,
    restore_descending: bool,
) -> tuple[FloatArray, FloatArray]:
    if restore_descending and x.size > 1 and x[0] < x[-1]:
        return x[::-1], spectra[:, ::-1]
    return x, spectra


def _format_number(value: float) -> str:
    return format(float(value), ".17g")


def _matrix_csv(
    x: FloatArray,
    spectra: FloatArray,
    labels: Sequence[str],
    *,
    restore_descending: bool,
) -> bytes:
    if spectra.shape != (len(labels), x.size):
        raise ValueError(
            "spectral matrix shape must equal (number of perturbation labels, number of points)"
        )
    x_out, spectra_out = _oriented(x, spectra, restore_descending)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["Wavenumber", *labels])
    for point, row_values in zip(x_out, spectra_out.T, strict=True):
        writer.writerow([_format_number(point), *(_format_number(value) for value in row_values)])
    return stream.getvalue().encode("utf-8")


def format_2dcos_csv(
    wavenumber: ArrayLike,
    spectra: ArrayLike,
    perturbation_labels: Sequence[str] | None = None,
    *,
    restore_descending_axis: bool = True,
) -> bytes:
    """Format a 2D-COS matrix with spectra in perturbation order as columns."""

    values = _as_matrix(spectra, "2D-COS spectra")
    x = np.asarray(wavenumber, dtype=np.float64)
    if x.shape != (values.shape[1],):
        raise ValueError(f"wavenumber must have shape ({values.shape[1]},)")
    labels = (
        tuple(str(item) for item in perturbation_labels)
        if perturbation_labels is not None
        else tuple(f"Spectrum_{index + 1}" for index in range(values.shape[0]))
    )
    if len(labels) != values.shape[0]:
        raise ValueError("one perturbation label is required for every spectrum")
    return _matrix_csv(x, values, labels, restore_descending=restore_descending_axis)


def build_2dcos_sensitivity_report(
    wavenumber: ArrayLike,
    sensitivity_branches: Mapping[str, ArrayLike],
) -> dict[str, Any]:
    """Summarize how preprocessing branches change 2D-COS input dynamics.

    Dynamic spectra are formed by subtracting the spectrum-series mean at every
    wavenumber (mean over perturbation axis 0).  The result intentionally stops
    short of claiming that a stable RMS or sign identifies the physical
    baseline; it is a sensitivity screen to be reviewed with the spectra.
    """

    x = np.asarray(wavenumber, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("sensitivity-report wavenumber must be a non-empty 1-D array")
    if not np.all(np.isfinite(x)):
        raise ValueError("sensitivity-report wavenumber contains NaN or Inf")
    if not isinstance(sensitivity_branches, Mapping) or not sensitivity_branches:
        raise ValueError("sensitivity_branches must be a non-empty mapping")

    matrices: dict[str, FloatArray] = {}
    dynamics: dict[str, FloatArray] = {}
    expected_shape: tuple[int, int] | None = None
    for raw_name, raw_values in sensitivity_branches.items():
        name = str(raw_name)
        if not name:
            raise ValueError("sensitivity branch names must be non-empty")
        values = _as_matrix(raw_values, f"sensitivity_branches.{name}")
        if values.shape[1] != x.size:
            raise ValueError(
                f"sensitivity branch {name!r} has {values.shape[1]} points; expected {x.size}"
            )
        if expected_shape is None:
            expected_shape = values.shape
        elif values.shape != expected_shape:
            raise ValueError(
                "all sensitivity branches must have the same shape; "
                f"{name!r} has {values.shape}, expected {expected_shape}"
            )
        matrices[name] = values
        dynamics[name] = values - np.mean(values, axis=0, keepdims=True)

    branch_statistics: dict[str, Any] = {}
    for name, values in matrices.items():
        dynamic = dynamics[name]
        dynamic_rms_by_wavenumber = np.sqrt(np.mean(dynamic**2, axis=0))
        maximum_index = int(np.argmax(dynamic_rms_by_wavenumber))
        branch_statistics[name] = {
            "shape": list(values.shape),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "matrix_rms": float(np.sqrt(np.mean(values**2))),
            "mean_spectrum_rms": float(np.sqrt(np.mean(np.mean(values, axis=0) ** 2))),
            "mean_centered_dynamic_rms": float(np.sqrt(np.mean(dynamic**2))),
            "dynamic_rms_by_spectrum": np.sqrt(np.mean(dynamic**2, axis=1)).tolist(),
            "maximum_dynamic_rms_wavenumber": float(x[maximum_index]),
            "maximum_dynamic_rms_at_wavenumber": float(dynamic_rms_by_wavenumber[maximum_index]),
        }

    pairwise: dict[str, Any] = {}
    for left_name, right_name in combinations(matrices, 2):
        left = matrices[left_name]
        right = matrices[right_name]
        left_dynamic = dynamics[left_name]
        right_dynamic = dynamics[right_name]
        matrix_difference = left - right
        dynamic_difference = left_dynamic - right_dynamic
        denominator = float(np.linalg.norm(left_dynamic) * np.linalg.norm(right_dynamic))
        correlation: float | None
        if np.array_equal(left_dynamic, right_dynamic):
            correlation = 1.0
        elif denominator > np.finfo(np.float64).tiny:
            correlation = float(
                np.clip(
                    np.vdot(left_dynamic.ravel(), right_dynamic.ravel()) / denominator,
                    -1.0,
                    1.0,
                )
            )
        elif np.allclose(left_dynamic, right_dynamic):
            correlation = 1.0
        else:
            correlation = None

        sign_threshold = (
            64.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                float(np.max(np.abs(left_dynamic))),
                float(np.max(np.abs(right_dynamic))),
            )
        )
        active = (np.abs(left_dynamic) > sign_threshold) | (np.abs(right_dynamic) > sign_threshold)
        sign_agreement = (
            float(np.mean(np.signbit(left_dynamic[active]) == np.signbit(right_dynamic[active])))
            if np.any(active)
            else None
        )
        reference_rms = float(np.sqrt(np.mean(right_dynamic**2)))
        difference_rms = float(np.sqrt(np.mean(dynamic_difference**2)))
        pairwise[f"{left_name}__vs__{right_name}"] = {
            "left_branch": left_name,
            "right_branch": right_name,
            "matrix_difference_rms": float(np.sqrt(np.mean(matrix_difference**2))),
            "matrix_difference_max_abs": float(np.max(np.abs(matrix_difference))),
            "mean_spectrum_difference_rms": float(
                np.sqrt(np.mean(np.mean(matrix_difference, axis=0) ** 2))
            ),
            "mean_centered_dynamic_difference_rms": difference_rms,
            "dynamic_flattened_correlation": correlation,
            "dynamic_sign_agreement_fraction": sign_agreement,
            "dynamic_difference_to_right_rms_ratio": (
                difference_rms / reference_rms if reference_rms > 0 else None
            ),
        }

    preferred_reference = (
        "coarse_plus_fine" if "coarse_plus_fine" in matrices else next(reversed(matrices))
    )
    return {
        "schema_version": "1.0",
        "purpose": "2D-COS preprocessing sensitivity screen",
        "mean_centering": {
            "applied": True,
            "axis": 0,
            "axis_meaning": "perturbation/spectrum series",
            "formula": "dynamic = branch - mean(branch, axis=0)",
        },
        "perturbation_order_preserved": True,
        "preferred_reference_branch": preferred_reference,
        "branch_statistics": branch_statistics,
        "pairwise_branch_differences": pairwise,
        "scientific_disclaimer": (
            "These RMS, correlation, and sign-agreement diagnostics show sensitivity to "
            "preprocessing. They do not prove the true baseline or replace inspection of "
            "critical peak regions and 2D-COS cross-peak signs."
        ),
    }


def _empty_optional_csv(x: FloatArray, *, restore_descending: bool) -> bytes:
    x_out = x[::-1] if restore_descending and x.size > 1 and x[0] < x[-1] else x
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["Wavenumber"])
    writer.writerows([_format_number(value)] for value in x_out)
    return stream.getvalue().encode("utf-8")


def _flatten_metric_columns(metrics: Any, n_spectra: int) -> dict[str, list[Any]]:
    if metrics is None:
        return {}
    if hasattr(metrics, "per_spectrum"):
        metrics = metrics.per_spectrum
    elif isinstance(metrics, Mapping) and isinstance(metrics.get("per_spectrum"), Mapping):
        metrics = metrics["per_spectrum"]
    elif hasattr(metrics, "as_dict"):
        metrics = metrics.as_dict()
        if isinstance(metrics, Mapping) and isinstance(metrics.get("per_spectrum"), Mapping):
            metrics = metrics["per_spectrum"]
    if not isinstance(metrics, Mapping):
        return {}

    columns: dict[str, list[Any]] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), item)
            return
        array = np.asarray(value) if not isinstance(value, str) else np.asarray([value])
        if array.dtype.kind not in "biuf":
            return
        if array.ndim == 0:
            scalar = array.item()
            columns[prefix] = [scalar] * n_spectra
        elif array.ndim == 1 and array.size == n_spectra:
            columns[prefix] = array.tolist()

    for key, value in metrics.items():
        visit(str(key), value)
    return columns


def _metrics_csv(
    metrics: Any,
    baseline_metrics: Any,
    n_spectra: int,
    perturbation: ArrayLike | None,
) -> bytes:
    columns: dict[str, list[Any]] = {"spectrum_index": list(range(n_spectra))}
    if perturbation is not None:
        perturbation_values = np.asarray(perturbation)
        if perturbation_values.shape == (n_spectra,):
            columns["perturbation"] = perturbation_values.tolist()
    for source in (baseline_metrics, metrics):
        for key, metric_values in _flatten_metric_columns(source, n_spectra).items():
            columns[key] = metric_values

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    names = list(columns)
    writer.writerow(names)
    for row in range(n_spectra):
        output: list[Any] = []
        for name in names:
            value = columns[name][row]
            if isinstance(value, (float, np.floating)):
                output.append("" if not np.isfinite(value) else _format_number(float(value)))
            elif isinstance(value, np.integer):
                output.append(int(value))
            else:
                output.append(value)
        writer.writerow(output)
    return stream.getvalue().encode("utf-8")


def _fallback_input_hash(x: FloatArray, spectra: FloatArray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(spectra, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _safe_figure_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(str(name)).name).strip("._")
    if not safe:
        raise ValueError("QC figure names must contain at least one safe character")
    return safe


def _summary_from_metrics(metrics: Any) -> Mapping[str, Any]:
    if hasattr(metrics, "summary"):
        value = metrics.summary
        return value if isinstance(value, Mapping) else {}
    if isinstance(metrics, Mapping):
        summary = metrics.get("summary")
        return summary if isinstance(summary, Mapping) else {}
    return {}


def _html_report(
    recipe: Mapping[str, Any],
    metrics: Any,
    warnings: Sequence[str],
    figure_names: Sequence[str],
    sensitivity_report: Mapping[str, Any] | None = None,
) -> bytes:
    summary = _summary_from_metrics(metrics)
    warning_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in warnings)
    if not warning_items:
        warning_items = "<li>No pipeline warnings were recorded.</li>"
    metric_rows = "".join(
        "<tr><th>"
        + html.escape(str(key))
        + "</th><td>"
        + html.escape(str(_jsonable(value)))
        + "</td></tr>"
        for key, value in summary.items()
    )
    if not metric_rows:
        metric_rows = "<tr><td colspan='2'>No aggregate QC metrics supplied.</td></tr>"
    figures = "".join(
        f"<figure><img src='12_qc_figures/{html.escape(name)}' alt='{html.escape(name)}'>"
        f"<figcaption>{html.escape(name)}</figcaption></figure>"
        for name in figure_names
    )
    if not figures:
        figures = "<p>No QC figures were generated; numerical QC remains in the CSV.</p>"
    sensitivity_html = ""
    if sensitivity_report is not None:
        branch_rows = "".join(
            "<tr><th>"
            + html.escape(str(name))
            + "</th><td>"
            + html.escape(f"{float(values['mean_centered_dynamic_rms']):.6g}")
            + "</td><td>"
            + html.escape(f"{float(values['matrix_rms']):.6g}")
            + "</td></tr>"
            for name, values in sensitivity_report["branch_statistics"].items()
        )
        sensitivity_html = (
            "<h2>2D-COS preprocessing sensitivity</h2>"
            "<p class='notice'>This is a sensitivity screen, not proof of the true baseline "
            "and not a substitute for inspecting critical peak regions or cross-peak signs. "
            "Complete pairwise statistics are in <code>13_2dcos_sensitivity_report.json</code>."
            "</p><table><thead><tr><th>Branch</th><th>Mean-centered dynamic RMS</th>"
            f"<th>Matrix RMS</th></tr></thead><tbody>{branch_rows}</tbody></table>"
        )
    recipe_json = json.dumps(_jsonable(recipe), ensure_ascii=False, indent=2, sort_keys=True)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FTIR Baseline Workbench Processing Report</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1f2937}}
h1,h2{{color:#12344d}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ccd5df;padding:.45rem;text-align:left}}
th{{background:#eef4f8}} .notice{{background:#fff8d8;border-left:4px solid #d99b00;padding:.8rem}}
img{{max-width:100%;height:auto}} pre{{overflow:auto;background:#f5f7f9;padding:1rem}} figure{{margin:1rem 0}}
</style></head><body>
<h1>FTIR Baseline Workbench Processing Report</h1>
<p class="notice"><strong>Scientific note:</strong> Diagnostic scores rank candidates only; they do not prove the unknown true baseline. Negative points and temporal jumps were never silently removed.</p>
<h2>Provenance</h2><p>Source: {html.escape(str(recipe.get("source_name", "unknown")))}</p>
<p>Input SHA-256: <code>{html.escape(str(recipe.get("input_sha256", "unknown")))}</code></p>
<p>Software version: {html.escape(str(recipe.get("software_version", "unknown")))}</p>
<h2>Warnings</h2><ul>{warning_items}</ul>
<h2>QC summary</h2><table><tbody>{metric_rows}</tbody></table>
<h2>QC figures</h2>{figures}
{sensitivity_html}
<h2>Complete processing recipe</h2><pre>{html.escape(recipe_json)}</pre>
</body></html>"""
    return document.encode("utf-8")


def _zip_info(name: str, *, directory: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name if not directory or name.endswith("/") else name + "/")
    info.date_time = _ZIP_TIMESTAMP
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = ((0o755 if directory else 0o644) & 0xFFFF) << 16
    if directory:
        info.external_attr |= 0x10
    return info


def _media_type(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return {
        ".csv": "text/csv",
        ".json": "application/json",
        ".html": "text/html",
        ".png": "image/png",
        ".svg": "image/svg+xml",
    }.get(suffix, "application/octet-stream")


def build_export_zip(
    result: Any,
    qc_figures: Mapping[str, bytes] | None = None,
) -> bytes:
    """Build the complete in-memory export ZIP from a duck-typed pipeline result."""

    raw_set = _first(result, ("raw_input", "raw"))
    full_set = _first(result, ("absorbance_full", "full_absorbance"))
    selected_set = _first(result, ("absorbance_selected", "selected_absorbance"))
    baseline_result = _first(result, ("baseline", "baseline_result"))
    normalization = _first(result, ("normalization", "normalization_result"), default=None)
    config = _get(result, "config", {})
    restore_descending = _restore_descending(config)

    raw_x, raw_spectra = _spectrum_fields(raw_set, "raw_input")
    full_x, full_spectra = _spectrum_fields(full_set, "absorbance_full")
    selected_x, selected_spectra = _spectrum_fields(selected_set, "absorbance_selected")
    labels = _labels_for(selected_set, selected_spectra.shape[0])
    raw_labels = _labels_for(raw_set, raw_spectra.shape[0])
    full_labels = _labels_for(full_set, full_spectra.shape[0])

    coarse = _as_matrix(_get(baseline_result, "coarse_baseline"), "coarse_baseline")
    fine = _as_matrix(_get(baseline_result, "fine_baseline"), "fine_baseline")
    total = _as_matrix(_get(baseline_result, "total_baseline"), "total_baseline")
    corrected = _as_matrix(
        _first(baseline_result, ("corrected", "corrected_absorbance")), "corrected"
    )
    expected_shape = selected_spectra.shape
    for name, values in (
        ("coarse_baseline", coarse),
        ("fine_baseline", fine),
        ("total_baseline", total),
        ("corrected", corrected),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {values.shape}")

    optional_normalized: FloatArray | None = None
    normalization_method = "none"
    normalization_warnings: tuple[str, ...] = ()
    normalization_params: Any = {}
    normalization_factors: Any = []
    if normalization is not None:
        optional_value = _get(normalization, "optional_normalized", None)
        if optional_value is not None:
            optional_normalized = _as_matrix(optional_value, "optional_normalized")
            if optional_normalized.shape != expected_shape:
                raise ValueError(
                    f"optional_normalized must have shape {expected_shape}, got {optional_normalized.shape}"
                )
        normalization_method = str(_get(normalization, "method", "none"))
        normalization_warnings = tuple(_get(normalization, "warnings", ()))
        normalization_params = _get(normalization, "params", {})
        normalization_factors = _get(normalization, "factors", [])

    metrics = _first(result, ("metrics", "qc"), default={})
    baseline_metrics = _get(baseline_result, "metrics", {})
    result_warnings = tuple(str(item) for item in _get(result, "warnings", ()))
    baseline_warnings = tuple(str(item) for item in _get(baseline_result, "warnings", ()))
    warnings = tuple(dict.fromkeys((*result_warnings, *baseline_warnings, *normalization_warnings)))
    perturbation = _get(selected_set, "perturbation", None)

    input_hash = _first(result, ("input_hash", "input_sha256"), default=None)
    if not input_hash:
        input_hash = _fallback_input_hash(raw_x, raw_spectra)
    version = _software_version(_first(result, ("version", "software_version"), default=None))
    source_name = str(_get(raw_set, "source_name", "unknown"))

    config_payload = _config_dict(config)
    supplied_recipe = _get(result, "recipe", None)
    supplied_recipe_json = _jsonable(supplied_recipe)
    recipe: dict[str, Any] = (
        dict(supplied_recipe_json) if isinstance(supplied_recipe_json, Mapping) else {}
    )
    recipe.setdefault("processing_order", list(PROCESSING_ORDER))
    recipe.update(
        {
            "schema_version": "1.0",
            "software_name": "ftir-baseline-workbench",
            "software_version": version,
            "source_name": source_name,
            "input_sha256": str(input_hash),
            "config": config_payload,
            "normalization": {
                "method": normalization_method,
                "params": _jsonable(normalization_params),
                "factors": _jsonable(normalization_factors),
                "optional_branch_written": optional_normalized is not None,
                "analysis_branch_overwritten": False,
            },
            "warnings": list(warnings),
            "axis_export": {
                "restore_descending_axis": restore_descending,
                "selected_input_direction": (
                    "descending"
                    if selected_x.size < 2 or selected_x[0] > selected_x[-1]
                    else "ascending"
                ),
            },
        }
    )

    files: dict[str, bytes] = {
        "01_raw_input.csv": _matrix_csv(
            raw_x, raw_spectra, raw_labels, restore_descending=restore_descending
        ),
        "02_absorbance_full.csv": _matrix_csv(
            full_x, full_spectra, full_labels, restore_descending=restore_descending
        ),
        "03_absorbance_selected.csv": _matrix_csv(
            selected_x, selected_spectra, labels, restore_descending=restore_descending
        ),
        "04_coarse_baseline.csv": _matrix_csv(
            selected_x, coarse, labels, restore_descending=restore_descending
        ),
        "05_fine_baseline.csv": _matrix_csv(
            selected_x, fine, labels, restore_descending=restore_descending
        ),
        "06_total_baseline.csv": _matrix_csv(
            selected_x, total, labels, restore_descending=restore_descending
        ),
        "07_corrected_absorbance.csv": _matrix_csv(
            selected_x, corrected, labels, restore_descending=restore_descending
        ),
        "08_normalized_optional.csv": (
            _empty_optional_csv(selected_x, restore_descending=restore_descending)
            if optional_normalized is None
            else _matrix_csv(
                selected_x,
                optional_normalized,
                labels,
                restore_descending=restore_descending,
            )
        ),
        "09_baseline_metrics.csv": _metrics_csv(
            metrics, baseline_metrics, selected_spectra.shape[0], perturbation
        ),
        "corrected_absorbance_for_2dcos.csv": _matrix_csv(
            selected_x, corrected, labels, restore_descending=restore_descending
        ),
    }
    if optional_normalized is not None:
        files["normalized_optional_for_sensitivity_analysis.csv"] = _matrix_csv(
            selected_x,
            optional_normalized,
            labels,
            restore_descending=restore_descending,
        )

    sensitivity_report: dict[str, Any] | None = None
    sensitivity_branches = _get(result, "sensitivity_branches", None)
    if sensitivity_branches is not None and not isinstance(sensitivity_branches, Mapping):
        raise ValueError("result.sensitivity_branches must be a mapping when supplied")
    if isinstance(sensitivity_branches, Mapping) and sensitivity_branches:
        sensitivity_report = build_2dcos_sensitivity_report(selected_x, sensitivity_branches)
        files["13_2dcos_sensitivity_report.json"] = json.dumps(
            sensitivity_report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        recipe["2dcos_sensitivity_report"] = {
            "generated": True,
            "path": "13_2dcos_sensitivity_report.json",
            "branches": list(sensitivity_report["branch_statistics"]),
            "mean_centered_over_perturbation_axis": True,
            "scientific_disclaimer": sensitivity_report["scientific_disclaimer"],
        }

    if qc_figures is None:
        try:
            from .qc import generate_qc_figures

            qc_figures = generate_qc_figures(
                selected_x,
                selected_spectra,
                total,
                corrected,
                metrics if hasattr(metrics, "per_spectrum") else None,
                perturbation=perturbation,
            )
        except (ImportError, RuntimeError):
            qc_figures = {}
    safe_figures: dict[str, bytes] = {}
    for name, contents in (qc_figures or {}).items():
        safe_name = _safe_figure_name(str(name))
        if not isinstance(contents, (bytes, bytearray, memoryview)):
            raise TypeError(f"QC figure {name!r} must be bytes")
        safe_figures[f"12_qc_figures/{safe_name}"] = bytes(contents)

    recipe_bytes = json.dumps(
        _jsonable(recipe), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    files["10_processing_recipe.json"] = recipe_bytes
    files.update(safe_figures)
    files["11_processing_report.html"] = _html_report(
        recipe,
        metrics,
        warnings,
        [Path(name).name for name in sorted(safe_figures)],
        sensitivity_report,
    )

    manifest_core: dict[str, Any] = {
        "schema_version": "1.0",
        "input_sha256": str(input_hash),
        "software_version": version,
        "hash_algorithm": "SHA-256",
        "files": [
            {
                "path": name,
                "size_bytes": len(contents),
                "sha256": sha256_bytes(contents),
                "media_type": _media_type(name),
            }
            for name, contents in sorted(files.items())
        ],
        "notes": [
            "manifest_sha256 authenticates this manifest object with that field omitted",
            "08_normalized_optional.csv has only a Wavenumber column when normalization is disabled",
            "normalized_optional_for_sensitivity_analysis.csv is written only for explicit scientific normalization",
            "13_2dcos_sensitivity_report.json is written when pipeline sensitivity branches are available",
        ],
    }
    core_bytes = json.dumps(
        manifest_core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    manifest = dict(manifest_core)
    manifest["manifest_sha256"] = sha256_bytes(core_bytes)
    files["manifest.json"] = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(_zip_info("12_qc_figures/", directory=True), b"")
        for name in sorted(files):
            archive.writestr(_zip_info(name), files[name])
    return output.getvalue()


create_export_zip = build_export_zip


def export_result(
    result: Any,
    output_dir: str | Path,
    qc_figures: Mapping[str, bytes] | None = None,
    *,
    filename: str | None = None,
) -> Path:
    """Write the auditable ZIP atomically and return its absolute path."""

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if filename is None:
        raw_set = _first(result, ("raw_input", "raw"))
        stem = Path(str(_get(raw_set, "source_name", "ftir_baseline"))).stem
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "ftir_baseline"
        filename = f"{safe_stem}_processed.zip"
    safe_filename = Path(filename).name
    if not safe_filename.lower().endswith(".zip"):
        safe_filename += ".zip"
    destination = directory / safe_filename
    temporary = directory / f".{safe_filename}.tmp"
    temporary.write_bytes(build_export_zip(result, qc_figures))
    temporary.replace(destination)
    return destination


def verify_export_manifest(bundle: bytes | bytearray | memoryview | str | Path) -> bool:
    """Verify every file hash and the canonical manifest hash in an export ZIP."""

    if isinstance(bundle, (str, Path)):
        source: Any = Path(bundle)
    else:
        source = io.BytesIO(bytes(bundle))
    try:
        with zipfile.ZipFile(source, "r") as archive:
            archive_names = archive.namelist()
            if len(archive_names) != len(set(archive_names)):
                return False
            manifest = json.loads(archive.read("manifest.json"))
            expected_manifest_hash = manifest.pop("manifest_sha256")
            core_bytes = json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if sha256_bytes(core_bytes) != expected_manifest_hash:
                return False
            listed_names = {str(entry["path"]) for entry in manifest["files"]}
            expected_names = listed_names | {"manifest.json", "12_qc_figures/"}
            if set(archive_names) != expected_names:
                return False
            for entry in manifest["files"]:
                contents = archive.read(entry["path"])
                if len(contents) != int(entry["size_bytes"]):
                    return False
                if sha256_bytes(contents) != entry["sha256"]:
                    return False
    except (
        KeyError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ):
        return False
    return True


verify_manifest = verify_export_manifest


__all__ = [
    "ExportPayload",
    "build_2dcos_sensitivity_report",
    "build_export_zip",
    "create_export_zip",
    "export_result",
    "format_2dcos_csv",
    "sha256_bytes",
    "sha256_file",
    "verify_export_manifest",
    "verify_manifest",
]
