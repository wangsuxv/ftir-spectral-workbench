"""Portable prepared-data and workbench bundle serialization.

The coordination layer owns these artifact formats.  The baseline bundle is
extended *after* the authoritative :mod:`ftir_baseline` exporter has produced
its normal output; no baseline or 2D-COS calculation is duplicated here.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, TextIO, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ftir_baseline.export import build_export_zip as build_legacy_baseline_zip
from ftir_baseline.export import verify_export_manifest as verify_legacy_baseline_manifest

from .adapters import prepared_from_baseline_result
from .fingerprints import prepared_data_sha256
from .models import PreparedSpectralDataset
from .validation import wavenumber_direction

FloatArray = NDArray[np.float64]
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_DEFAULT_PREPARED_CSV = "corrected_absorbance_for_2dcos.csv"
_DEFAULT_PREPARED_META = "prepared_spectrum.meta.json"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_PARENT_LINEAGE_FIELDS = (
    "parent_baseline_run_id",
    "parent_baseline_fingerprint",
    "parent_prepared_data_sha256",
)


@dataclass(frozen=True, slots=True)
class PreparedExport:
    """In-memory prepared CSV and its mandatory metadata sidecar."""

    csv_bytes: bytes
    metadata_bytes: bytes
    csv_name: str = _DEFAULT_PREPARED_CSV
    metadata_name: str = _DEFAULT_PREPARED_META

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Allow ``csv_bytes, metadata_bytes = serialize_prepared(...)``."""

        yield self.csv_bytes
        yield self.metadata_bytes


@dataclass(frozen=True, slots=True)
class PreparedPaths:
    """Paths written by :func:`export_prepared`."""

    csv_path: Path
    metadata_path: Path

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield self.csv_path
        yield self.metadata_path


PreparedSource = (
    PreparedExport
    | bytes
    | bytearray
    | memoryview
    | str
    | Path
    | BinaryIO
    | TextIO
    | tuple[object, object]
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("metadata must not contain NaN or infinite values")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(cast(Any, value)))
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    raise TypeError(f"cannot serialize metadata value of type {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _format_float(value: float) -> str:
    """Use enough significant digits to recover the same binary64 value."""

    return format(float(value), ".17g")


def _wide_spectra_csv(
    wavenumber: ArrayLike,
    spectra: ArrayLike,
    labels: Sequence[str],
) -> bytes:
    axis = np.asarray(wavenumber, dtype=np.float64)
    matrix = np.asarray(spectra, dtype=np.float64)
    label_tuple = tuple(str(label) for label in labels)
    if axis.ndim != 1:
        raise ValueError("wavenumber must be one-dimensional")
    if matrix.shape != (len(label_tuple), axis.size):
        raise ValueError(
            "spectra must have shape (number of labels, number of wavenumbers); "
            f"got {matrix.shape}"
        )
    if not np.isfinite(axis).all() or not np.isfinite(matrix).all():
        raise ValueError("prepared CSV cannot contain NaN or infinite values")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["Wavenumber", *label_tuple])
    for point, intensities in zip(axis, matrix.T, strict=True):
        writer.writerow(
            [_format_float(point), *(_format_float(value) for value in intensities)]
        )
    return stream.getvalue().encode("utf-8")


def prepared_csv_bytes(prepared: PreparedSpectralDataset) -> bytes:
    """Serialize a prepared dataset without changing either axis order."""

    return _wide_spectra_csv(
        prepared.wavenumber,
        prepared.spectra,
        prepared.perturbation_labels,
    )


def prepared_metadata_dict(
    prepared: PreparedSpectralDataset,
    *,
    csv_name: str = _DEFAULT_PREPARED_CSV,
    recipe_file: str | None = None,
    csv_payload: bytes | None = None,
) -> dict[str, Any]:
    """Build the complete, JSON-compatible prepared-data sidecar."""

    csv_name = Path(csv_name).name
    payload = prepared_csv_bytes(prepared) if csv_payload is None else bytes(csv_payload)
    if hasattr(prepared, "to_metadata_dict"):
        metadata = dict(prepared.to_metadata_dict(schema_version="1.0"))
    else:  # pragma: no cover - compatibility with early coordination models
        metadata = {
            "schema_version": "1.0",
            "source_name": prepared.source_name,
            "source_sha256": prepared.source_sha256,
            "baseline_run_id": prepared.baseline_run_id,
            "baseline_fingerprint": prepared.baseline_fingerprint,
            "prepared_data_sha256": prepared.prepared_data_sha256,
            "original_axis_direction": prepared.original_axis_direction,
            "current_axis_direction": prepared.current_axis_direction,
            "perturbation_order_policy": prepared.perturbation_order_policy,
            "baseline_recipe": prepared.baseline_recipe,
            "baseline_qc": prepared.baseline_qc,
            "warnings": prepared.warnings,
            "normalization_state": prepared.normalization_state,
        }
    metadata.update(
        {
            "artifact_type": "prepared_spectral_dataset",
            "csv_file": csv_name,
            "csv_sha256": _sha256(payload),
            "unit": "absorbance",
            "intensity_unit": "absorbance",
            "axis_direction": prepared.current_axis_direction,
            "original_axis_direction": prepared.original_axis_direction,
            "current_axis_direction": prepared.current_axis_direction,
            "perturbation_values": np.asarray(
                prepared.perturbation, dtype=np.float64
            ).tolist(),
            "perturbation_labels": list(prepared.perturbation_labels),
            "spectra_shape": [int(value) for value in prepared.spectra.shape],
            "recipe_file": recipe_file,
            "baseline_recipe": _jsonable(prepared.baseline_recipe),
            "baseline_qc": _jsonable(prepared.baseline_qc),
            "warnings": list(prepared.warnings),
        }
    )
    return _jsonable(metadata)


def serialize_prepared(
    prepared: PreparedSpectralDataset,
    *,
    csv_name: str = _DEFAULT_PREPARED_CSV,
    metadata_name: str = _DEFAULT_PREPARED_META,
    recipe_file: str | None = None,
) -> PreparedExport:
    """Return deterministic UTF-8 CSV and JSON sidecar bytes."""

    safe_csv_name = Path(csv_name).name
    safe_metadata_name = Path(metadata_name).name
    csv_payload = prepared_csv_bytes(prepared)
    metadata = prepared_metadata_dict(
        prepared,
        csv_name=safe_csv_name,
        recipe_file=recipe_file,
        csv_payload=csv_payload,
    )
    return PreparedExport(
        csv_bytes=csv_payload,
        metadata_bytes=_json_bytes(metadata),
        csv_name=safe_csv_name,
        metadata_name=safe_metadata_name,
    )


def _atomic_write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path


def export_prepared(
    prepared: PreparedSpectralDataset,
    destination: str | Path | None = None,
    *,
    csv_name: str = _DEFAULT_PREPARED_CSV,
    metadata_name: str = _DEFAULT_PREPARED_META,
) -> PreparedExport | PreparedPaths:
    """Serialize prepared data in memory, or atomically write the CSV/sidecar pair.

    A destination ending in ``.csv`` is treated as the CSV path.  Every other
    destination is treated as an output directory.
    """

    destination_path = None if destination is None else Path(destination)
    output_csv_name = csv_name
    if destination_path is not None and destination_path.suffix.lower() == ".csv":
        output_csv_name = destination_path.name
    artifact = serialize_prepared(
        prepared,
        csv_name=output_csv_name,
        metadata_name=metadata_name,
    )
    if destination_path is None:
        return artifact
    requested = destination_path.expanduser()
    if requested.suffix.lower() == ".csv":
        csv_path = requested.resolve()
        metadata_path = csv_path.with_name(artifact.metadata_name)
    else:
        directory = requested.resolve()
        csv_path = directory / artifact.csv_name
        metadata_path = directory / artifact.metadata_name
    _atomic_write(csv_path, artifact.csv_bytes)
    _atomic_write(metadata_path, artifact.metadata_bytes)
    return PreparedPaths(csv_path=csv_path, metadata_path=metadata_path)


def _read_bytes(source: object, *, default_name: str) -> tuple[bytes, str, Path | None]:
    if isinstance(source, Path):
        path = source.expanduser().resolve()
        return path.read_bytes(), path.name, path
    if isinstance(source, str):
        if "\n" in source or "\r" in source or source.lstrip().startswith(("{", "[")):
            return source.encode("utf-8"), default_name, None
        path = Path(source).expanduser().resolve()
        return path.read_bytes(), path.name, path
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), default_name, None
    getvalue = getattr(source, "getvalue", None)
    if callable(getvalue):
        value = getvalue()
        payload = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        return payload, str(getattr(source, "name", default_name)), None
    read = getattr(source, "read", None)
    if callable(read):
        position: int | None = None
        tell = getattr(source, "tell", None)
        if callable(tell):
            with suppress(AttributeError, OSError, TypeError, ValueError):
                position = int(tell())
        value = read()
        if position is not None:
            seek = getattr(source, "seek", None)
            if callable(seek):
                with suppress(AttributeError, OSError):
                    seek(position)
        payload = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        return payload, str(getattr(source, "name", default_name)), None
    raise TypeError(f"unsupported prepared-data source type: {type(source).__name__}")


def _load_metadata(source: object) -> tuple[dict[str, Any], Path | None]:
    if isinstance(source, Mapping):
        return dict(source), None
    payload, _, path = _read_bytes(source, default_name=_DEFAULT_PREPARED_META)
    try:
        value = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("prepared metadata must be valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("prepared metadata JSON root must be an object")
    return dict(value), path


def _parse_prepared_csv(payload: bytes) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("prepared CSV must be UTF-8") from exc
    rows = list(csv.reader(io.StringIO(text, newline="")))
    while rows and not any(cell.strip() for cell in rows[-1]):
        rows.pop()
    if len(rows) < 3:
        raise ValueError("prepared CSV requires a header and at least two wavenumber rows")
    header = rows[0]
    if len(header) < 2 or header[0].strip().casefold() not in {
        "wavenumber",
        "wavenumber_cm-1",
        "wavenumber_cm^-1",
    }:
        raise ValueError("prepared CSV first column must be Wavenumber")
    labels = tuple(cell for cell in header[1:])
    expected_columns = len(header)
    values = np.empty((len(rows) - 1, expected_columns), dtype=np.float64)
    for row_index, row in enumerate(rows[1:], start=2):
        if len(row) != expected_columns:
            raise ValueError(
                f"prepared CSV row {row_index} has {len(row)} columns; "
                f"expected {expected_columns}"
            )
        try:
            values[row_index - 2] = [float(cell.strip()) for cell in row]
        except ValueError as exc:
            raise ValueError(f"prepared CSV row {row_index} contains non-numeric data") from exc
    if not np.isfinite(values).all():
        raise ValueError("prepared CSV contains NaN or infinite values")
    axis = values[:, 0]
    matrix = values[:, 1:].T
    wavenumber_direction(axis)
    if any(not label.strip() for label in labels):
        raise ValueError("prepared CSV perturbation labels must not be blank")
    return axis, matrix, labels


_PERTURBATION_NUMBER = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def _infer_perturbation(labels: Sequence[str]) -> tuple[FloatArray, str]:
    values: list[float] = []
    for label in labels:
        match = _PERTURBATION_NUMBER.search(label)
        if match is None:
            return np.arange(len(labels), dtype=np.float64), "sequential_index"
        value = float(match.group(0))
        if not np.isfinite(value):
            return np.arange(len(labels), dtype=np.float64), "sequential_index"
        values.append(value)
    return np.asarray(values, dtype=np.float64), "parsed_from_labels"


def _find_zip_member(archive: zipfile.ZipFile, basename: str) -> str | None:
    exact = [name for name in archive.namelist() if name == basename]
    if exact:
        return exact[0]
    matches = [name for name in archive.namelist() if Path(name).name == basename]
    if len(matches) > 1:
        raise ValueError(f"archive contains multiple files named {basename!r}")
    return matches[0] if matches else None


def _prepared_from_payloads(
    csv_payload: bytes,
    *,
    source_name: str,
    metadata: Mapping[str, Any] | None,
) -> PreparedSpectralDataset:
    axis, spectra, csv_labels = _parse_prepared_csv(csv_payload)
    detected_direction = wavenumber_direction(axis)
    provenance_warning = (
        "Provenance incomplete: loaded a bare corrected-absorbance CSV without "
        "prepared_spectrum.meta.json; the original source, baseline recipe, and QC "
        "cannot be independently reconstructed."
    )
    if metadata is None:
        perturbation, order_policy = _infer_perturbation(csv_labels)
        normalization_state = "none"
        data_hash = prepared_data_sha256(
            axis,
            perturbation,
            csv_labels,
            spectra,
            normalization_state=normalization_state,
        )
        return PreparedSpectralDataset(
            wavenumber=axis,
            perturbation=perturbation,
            perturbation_labels=csv_labels,
            spectra=spectra,
            intensity_unit="absorbance",
            source_name=source_name,
            source_sha256="unknown",
            baseline_run_id="unknown",
            baseline_fingerprint="unknown",
            prepared_data_sha256=data_hash,
            original_axis_direction=detected_direction,
            current_axis_direction=detected_direction,
            perturbation_order_policy=order_policy,
            baseline_recipe={},
            baseline_qc={},
            warnings=(provenance_warning,),
            normalization_state="none",
        )

    meta = dict(metadata)
    unit = str(meta.get("unit", meta.get("intensity_unit", "absorbance"))).strip().lower()
    if unit != "absorbance":
        raise ValueError(f"prepared metadata unit must be 'absorbance'; got {unit!r}")
    expected_csv_hash = meta.get("csv_sha256")
    if expected_csv_hash is not None and str(expected_csv_hash).lower() != _sha256(csv_payload):
        raise ValueError("prepared CSV SHA-256 does not match the metadata sidecar")
    meta_labels = tuple(str(value) for value in meta.get("perturbation_labels", csv_labels))
    if meta_labels != csv_labels:
        raise ValueError("prepared CSV labels do not match the metadata sidecar")
    perturbation_values = meta.get("perturbation_values", meta.get("perturbation"))
    missing_fields: list[str] = []
    if perturbation_values is None:
        perturbation, inferred_policy = _infer_perturbation(csv_labels)
        missing_fields.append("perturbation_values")
    else:
        perturbation = np.asarray(perturbation_values, dtype=np.float64)
        inferred_policy = "metadata_sidecar"
    if perturbation.shape != (spectra.shape[0],) or not np.isfinite(perturbation).all():
        raise ValueError(
            "prepared metadata perturbation_values must contain one finite value per spectrum"
        )
    expected_shape = meta.get("spectra_shape")
    if expected_shape is not None and tuple(int(value) for value in expected_shape) != spectra.shape:
        raise ValueError("prepared CSV shape does not match the metadata sidecar")
    recorded_direction = str(
        meta.get("current_axis_direction", meta.get("axis_direction", detected_direction))
    )
    if recorded_direction != detected_direction:
        raise ValueError("prepared CSV axis direction does not match the metadata sidecar")
    original_direction = str(meta.get("original_axis_direction", recorded_direction))
    if original_direction not in {"ascending", "descending"}:
        raise ValueError("prepared metadata original_axis_direction is invalid")
    normalization_state = str(meta.get("normalization_state", "none"))
    if normalization_state not in {"none", "display_only", "scientific_explicit"}:
        raise ValueError("prepared metadata normalization_state is invalid")
    typed_normalization_state = cast(
        "Literal['none', 'display_only', 'scientific_explicit']",
        normalization_state,
    )
    actual_data_hash = prepared_data_sha256(
        axis,
        perturbation,
        csv_labels,
        spectra,
        normalization_state=typed_normalization_state,
    )
    expected_data_hash = str(meta.get("prepared_data_sha256", actual_data_hash)).lower()
    if not _HEX_64.fullmatch(expected_data_hash):
        raise ValueError("prepared_data_sha256 must be a 64-character lowercase SHA-256")
    if expected_data_hash != actual_data_hash:
        raise ValueError("prepared CSV values do not match prepared_data_sha256")
    for field_name in ("source_sha256", "baseline_run_id", "baseline_fingerprint"):
        if field_name not in meta or str(meta[field_name]).strip().lower() in {
            "",
            "unknown",
            "unavailable",
        }:
            missing_fields.append(field_name)
    warnings = [str(value) for value in meta.get("warnings", ())]
    if missing_fields:
        warnings.append(
            "Provenance incomplete: metadata is missing authoritative values for "
            + ", ".join(sorted(set(missing_fields)))
            + "."
        )
    warnings = list(dict.fromkeys(warnings))
    baseline_recipe = meta.get("baseline_recipe", {})
    baseline_qc = meta.get("baseline_qc", meta.get("qc_summary", {}))
    if not isinstance(baseline_recipe, Mapping) or not isinstance(baseline_qc, Mapping):
        raise ValueError("baseline_recipe and baseline_qc must be JSON objects")
    return PreparedSpectralDataset(
        wavenumber=axis,
        perturbation=perturbation,
        perturbation_labels=csv_labels,
        spectra=spectra,
        intensity_unit="absorbance",
        source_name=str(meta.get("source_name", source_name)),
        source_sha256=str(meta.get("source_sha256", "unknown")),
        baseline_run_id=str(meta.get("baseline_run_id", "unknown")),
        baseline_fingerprint=str(meta.get("baseline_fingerprint", "unknown")),
        prepared_data_sha256=actual_data_hash,
        original_axis_direction=cast(
            "Literal['ascending', 'descending']", original_direction
        ),
        current_axis_direction=detected_direction,
        perturbation_order_policy=str(
            meta.get("perturbation_order_policy", inferred_policy)
        ),
        baseline_recipe=dict(baseline_recipe),
        baseline_qc=dict(baseline_qc),
        warnings=tuple(warnings),
        normalization_state=typed_normalization_state,
    )


def load_prepared(
    source: PreparedSource,
    metadata: object | None = None,
) -> PreparedSpectralDataset:
    """Load prepared data from CSV, CSV+sidecar, a sidecar path, or baseline ZIP.

    Bare CSV is intentionally accepted as a public checkpoint format.  Its
    result carries an explicit provenance warning and sentinel parent IDs.
    """

    if isinstance(source, PreparedExport):
        return _prepared_from_payloads(
            source.csv_bytes,
            source_name=source.csv_name,
            metadata=_load_metadata(source.metadata_bytes)[0],
        )
    if isinstance(source, tuple) and len(source) == 2:
        return load_prepared(cast(PreparedSource, source[0]), source[1])

    payload, input_name, input_path = _read_bytes(source, default_name=_DEFAULT_PREPARED_CSV)
    if zipfile.is_zipfile(io.BytesIO(payload)):
        if not (
            verify_legacy_baseline_manifest(payload) or verify_workbench_manifest(payload)
        ):
            raise ValueError("baseline/workbench ZIP manifest verification failed")
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            metadata_member = _find_zip_member(archive, _DEFAULT_PREPARED_META)
            metadata_value: dict[str, Any] | None = None
            if metadata_member is not None:
                metadata_value = _load_metadata(archive.read(metadata_member))[0]
                requested_csv = Path(
                    str(metadata_value.get("csv_file", _DEFAULT_PREPARED_CSV))
                ).name
            else:
                requested_csv = _DEFAULT_PREPARED_CSV
            csv_member = _find_zip_member(archive, requested_csv)
            if csv_member is None:
                raise ValueError(f"archive does not contain {requested_csv!r}")
            return _prepared_from_payloads(
                archive.read(csv_member),
                source_name=f"{input_name}!/{csv_member}",
                metadata=metadata_value,
            )

    if input_path is not None and input_path.suffix.lower() == ".json":
        sidecar, _ = _load_metadata(payload)
        csv_name = Path(str(sidecar.get("csv_file", _DEFAULT_PREPARED_CSV))).name
        csv_path = input_path.with_name(csv_name)
        if not csv_path.is_file():
            raise FileNotFoundError(f"prepared CSV referenced by sidecar does not exist: {csv_path}")
        return _prepared_from_payloads(
            csv_path.read_bytes(),
            source_name=csv_path.name,
            metadata=sidecar,
        )

    sidecar_metadata: Mapping[str, Any] | None = None
    if metadata is not None:
        sidecar_metadata = _load_metadata(metadata)[0]
    elif input_path is not None:
        sidecar_path = input_path.with_name(_DEFAULT_PREPARED_META)
        if sidecar_path.is_file():
            sidecar_metadata = _load_metadata(sidecar_path)[0]
    return _prepared_from_payloads(
        payload,
        source_name=input_name,
        metadata=sidecar_metadata,
    )


def _zip_info(name: str, *, directory: bool = False) -> zipfile.ZipInfo:
    normalized = name.rstrip("/") + "/" if directory else name.rstrip("/")
    info = zipfile.ZipInfo(normalized, _ZIP_TIMESTAMP)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = ((0o755 if directory else 0o644) << 16) | (0x10 if directory else 0)
    return info


def _media_type(name: str) -> str:
    return {
        ".csv": "text/csv",
        ".json": "application/json",
        ".html": "text/html",
        ".png": "image/png",
        ".pdf": "application/pdf",
        ".svg": "image/svg+xml",
        ".zip": "application/zip",
        ".ftirw": "application/zip",
    }.get(Path(name).suffix.lower(), "application/octet-stream")


def _safe_archive_name(name: str, *, prefix: str | None = None) -> str:
    basename = Path(str(name)).name
    if not basename or basename in {".", ".."}:
        raise ValueError(f"invalid archive member name: {name!r}")
    return basename if prefix is None else f"{prefix.rstrip('/')}/{basename}"


def _manifest_bytes(manifest_core: Mapping[str, Any]) -> bytes:
    canonical = json.dumps(
        _jsonable(manifest_core),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    manifest = dict(manifest_core)
    manifest["manifest_sha256"] = _sha256(canonical)
    return _json_bytes(manifest)


def _build_manifest_archive(
    files: Mapping[str, bytes],
    *,
    directories: Sequence[str] = (),
    manifest_base: Mapping[str, Any] | None = None,
) -> bytes:
    payloads = {str(name): bytes(payload) for name, payload in files.items()}
    if "manifest.json" in payloads:
        raise ValueError("manifest.json is built by the archive writer")
    directory_names = tuple(sorted({name.rstrip("/") + "/" for name in directories}))
    manifest_core = dict(manifest_base or {})
    manifest_core.pop("manifest_sha256", None)
    manifest_core.update(
        {
            "schema_version": str(manifest_core.get("schema_version", "1.0")),
            "hash_algorithm": "SHA-256",
            "files": [
                {
                    "path": name,
                    "size_bytes": len(payload),
                    "sha256": _sha256(payload),
                    "media_type": _media_type(name),
                }
                for name, payload in sorted(payloads.items())
            ],
        }
    )
    if directory_names:
        manifest_core["directories"] = list(directory_names)
    manifest_payload = _manifest_bytes(manifest_core)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in directory_names:
            archive.writestr(_zip_info(name, directory=True), b"")
        for name, payload in sorted(payloads.items()):
            archive.writestr(_zip_info(name), payload)
        archive.writestr(_zip_info("manifest.json"), manifest_payload)
    return output.getvalue()


def build_baseline_bundle(
    result: Any,
    *,
    prepared: PreparedSpectralDataset | None = None,
    qc_figures: Mapping[str, bytes] | None = None,
) -> bytes:
    """Extend the authoritative baseline ZIP with a verifiable prepared sidecar."""

    prepared_value = prepared or prepared_from_baseline_result(result)
    selected = result.absorbance_selected
    if not np.array_equal(prepared_value.wavenumber, selected.wavenumber):
        raise ValueError("prepared wavenumber does not belong to this baseline result")
    if not np.array_equal(prepared_value.perturbation, selected.perturbation):
        raise ValueError("prepared perturbation does not belong to this baseline result")
    if prepared_value.perturbation_labels != selected.perturbation_labels:
        raise ValueError("prepared perturbation labels do not belong to this baseline result")
    if not np.array_equal(prepared_value.spectra, result.analysis_data):
        raise ValueError(
            "baseline-only bundle prepared spectra must be PipelineResult.analysis_data"
        )
    artifact = serialize_prepared(
        prepared_value,
        csv_name=_DEFAULT_PREPARED_CSV,
        metadata_name=_DEFAULT_PREPARED_META,
        recipe_file="10_processing_recipe.json",
    )
    base_bundle = build_legacy_baseline_zip(result, qc_figures=qc_figures)
    if not verify_legacy_baseline_manifest(base_bundle):
        raise ValueError("authoritative baseline exporter produced an unverifiable manifest")
    with zipfile.ZipFile(io.BytesIO(base_bundle), "r") as archive:
        base_manifest = json.loads(archive.read("manifest.json"))
        files = {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/") and name != "manifest.json"
        }
        directories = tuple(name for name in archive.namelist() if name.endswith("/"))
    # This public checkpoint must represent the exact in-memory prepared object,
    # even when the legacy export preference would reverse an ascending axis.
    files[artifact.csv_name] = artifact.csv_bytes
    files[artifact.metadata_name] = artifact.metadata_bytes
    base_manifest.pop("files", None)
    notes = [str(value) for value in base_manifest.get("notes", ())]
    note = "prepared_spectrum.meta.json authenticates the exact 2D-COS-ready CSV contract"
    if note not in notes:
        notes.append(note)
    base_manifest.update(
        {
            "notes": notes,
            "prepared_spectrum": {
                "csv_path": artifact.csv_name,
                "metadata_path": artifact.metadata_name,
                "baseline_run_id": prepared_value.baseline_run_id,
                "baseline_fingerprint": prepared_value.baseline_fingerprint,
                "prepared_data_sha256": prepared_value.prepared_data_sha256,
            },
        }
    )
    bundle = _build_manifest_archive(
        files,
        directories=directories or ("12_qc_figures/",),
        manifest_base=base_manifest,
    )
    if not verify_legacy_baseline_manifest(bundle):
        raise ValueError("extended baseline bundle manifest verification failed")
    return bundle


def _write_or_return_bundle(
    bundle: bytes,
    destination: str | Path | None,
    *,
    default_name: str,
) -> bytes | Path:
    if destination is None:
        return bundle
    path = Path(destination).expanduser()
    if (path.exists() and path.is_dir()) or not path.suffix:
        path = path / default_name
    return _atomic_write(path.resolve(), bundle)


def export_baseline_bundle(
    result: Any,
    destination: str | Path | None = None,
    *,
    prepared: PreparedSpectralDataset | None = None,
    qc_figures: Mapping[str, bytes] | None = None,
) -> bytes | Path:
    """Build a baseline-only bundle and optionally write it to disk."""

    bundle = build_baseline_bundle(result, prepared=prepared, qc_figures=qc_figures)
    return _write_or_return_bundle(bundle, destination, default_name="baseline_run.zip")


def _matrix_csv(
    matrix: ArrayLike,
    row_axis: ArrayLike,
    column_axis: ArrayLike,
) -> bytes:
    values = np.asarray(matrix, dtype=np.float64)
    rows = np.asarray(row_axis, dtype=np.float64)
    columns = np.asarray(column_axis, dtype=np.float64)
    if values.shape != (rows.size, columns.size):
        raise ValueError(
            f"2D matrix shape {values.shape} does not match axes {(rows.size, columns.size)}"
        )
    if not np.isfinite(values).all():
        raise ValueError("2D matrix contains NaN or infinite values")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["wavenumber_cm-1", *(_format_float(value) for value in columns)])
    for coordinate, row in zip(rows, values, strict=True):
        writer.writerow([_format_float(coordinate), *(_format_float(value) for value in row)])
    return stream.getvalue().encode("utf-8")


def _unwrap_homo_result(result: Any) -> Any:
    candidate = result
    homo_results = getattr(candidate, "homo_results", None)
    if homo_results is not None:
        if not homo_results:
            raise ValueError("2D analysis contains no homo results")
        candidate = homo_results[0]
    nested = getattr(candidate, "result", None)
    if nested is not None:
        candidate = nested
    nested = getattr(candidate, "twodcos", None)
    if nested is not None:
        candidate = nested
    for name in ("dynamic", "synchronous", "asynchronous"):
        if not hasattr(candidate, name):
            raise TypeError(f"2D result is missing required field {name!r}")
    return candidate


def _core_result(value: Any) -> Any:
    candidate = getattr(value, "result", value)
    return getattr(candidate, "twodcos", candidate)


def _validate_twodcos_parent(prepared: PreparedSpectralDataset, result: Any) -> None:
    """Reject exporting a stale service result under a different parent lineage."""

    candidates = [result]
    homo_results = getattr(result, "homo_results", None)
    if homo_results is not None:
        candidates.extend(homo_results)
    expected = {
        "parent_baseline_run_id": prepared.baseline_run_id,
        "parent_baseline_fingerprint": prepared.baseline_fingerprint,
        "parent_prepared_data_sha256": prepared.prepared_data_sha256,
    }
    for candidate in candidates:
        for field_name, expected_value in expected.items():
            recorded = getattr(candidate, field_name, None)
            if recorded is not None and str(recorded) != expected_value:
                raise ValueError(
                    f"2D result {field_name} does not match the supplied prepared dataset"
                )
    for candidate in getattr(result, "cross_results", ()):
        for prefix in ("first", "second"):
            for suffix, expected_value in (
                ("baseline_run_id", prepared.baseline_run_id),
                ("baseline_fingerprint", prepared.baseline_fingerprint),
                ("prepared_data_sha256", prepared.prepared_data_sha256),
            ):
                field_name = f"{prefix}_parent_{suffix}"
                recorded = getattr(candidate, field_name, None)
                if recorded is not None and str(recorded) != expected_value:
                    raise ValueError(
                        f"2D result {field_name} does not match the supplied prepared dataset"
                    )


def build_twodcos_bundle(
    prepared: PreparedSpectralDataset,
    result: Any,
    config: Any | None = None,
    *,
    figures: Mapping[str, bytes] | None = None,
    peak_order_files: Mapping[str, bytes] | None = None,
) -> bytes:
    """Build a self-contained, parent-linked homo 2D-COS run bundle."""

    _validate_twodcos_parent(prepared, result)
    analysis = _unwrap_homo_result(result)
    config_value = config if config is not None else getattr(result, "config", {})
    artifact = serialize_prepared(
        prepared,
        csv_name="source_prepared_spectrum.csv",
        metadata_name="source_prepared_spectrum.meta.json",
    )
    dynamic = np.asarray(analysis.dynamic, dtype=np.float64)
    if dynamic.ndim != 2 or dynamic.shape[0] != prepared.n_spectra:
        raise ValueError(
            "2D dynamic spectra must contain the same number of spectra as the source prepared data"
        )
    row_axis = np.asarray(
        getattr(analysis, "row_wavenumber", prepared.wavenumber), dtype=np.float64
    )
    column_axis = np.asarray(
        getattr(analysis, "column_wavenumber", prepared.wavenumber), dtype=np.float64
    )
    if dynamic.shape[1] != row_axis.size:
        raise ValueError("2D dynamic spectra width must match the homo result wavenumber axis")
    files: dict[str, bytes] = {
        artifact.csv_name: artifact.csv_bytes,
        artifact.metadata_name: artifact.metadata_bytes,
        "twodcos_config.json": _json_bytes(config_value),
        "dynamic_spectra.csv": _wide_spectra_csv(
            row_axis,
            dynamic,
            prepared.perturbation_labels,
        ),
        "synchronous_matrix.csv": _matrix_csv(
            analysis.synchronous, row_axis, column_axis
        ),
        "asynchronous_matrix.csv": _matrix_csv(
            analysis.asynchronous, row_axis, column_axis
        ),
        "qc_metrics.json": _json_bytes(getattr(analysis, "qc_metrics", {})),
    }
    for name, payload in (figures or {}).items():
        files[_safe_archive_name(name, prefix="figures")] = bytes(payload)
    for name, payload in (peak_order_files or {}).items():
        files[_safe_archive_name(name, prefix="peak_order")] = bytes(payload)
    directories = ["figures/", "peak_order/"]
    homo_items = getattr(result, "homo_results", None)
    if homo_items is not None:
        directories.append("ranges/")
        for index, item in enumerate(homo_items, start=1):
            core = _core_result(item)
            prefix = f"ranges/range_{index:02d}"
            item_row_axis = np.asarray(core.row_wavenumber, dtype=np.float64)
            item_column_axis = np.asarray(core.column_wavenumber, dtype=np.float64)
            item_dynamic = np.asarray(core.dynamic, dtype=np.float64)
            if item_dynamic.shape != (prepared.n_spectra, item_row_axis.size):
                raise ValueError(f"homo result {index} dynamic spectra/axis shape mismatch")
            files[f"{prefix}/range.json"] = _json_bytes(
                getattr(item, "analysis_range", {"index": index})
            )
            files[f"{prefix}/dynamic_spectra.csv"] = _wide_spectra_csv(
                item_row_axis,
                item_dynamic,
                prepared.perturbation_labels,
            )
            files[f"{prefix}/synchronous_matrix.csv"] = _matrix_csv(
                core.synchronous,
                item_row_axis,
                item_column_axis,
            )
            files[f"{prefix}/asynchronous_matrix.csv"] = _matrix_csv(
                core.asynchronous,
                item_row_axis,
                item_column_axis,
            )
            files[f"{prefix}/qc_metrics.json"] = _json_bytes(core.qc_metrics)
    cross_items = getattr(result, "cross_results", None)
    if cross_items:
        directories.append("cross_ranges/")
        for index, item in enumerate(cross_items, start=1):
            core = _core_result(item)
            prefix = f"cross_ranges/cross_{index:02d}"
            cross_row_axis = np.asarray(core.row_wavenumber, dtype=np.float64)
            cross_column_axis = np.asarray(core.column_wavenumber, dtype=np.float64)
            files[f"{prefix}/ranges.json"] = _json_bytes(
                {
                    "first_range": getattr(item, "first_range", None),
                    "second_range": getattr(item, "second_range", None),
                }
            )
            files[f"{prefix}/synchronous_matrix.csv"] = _matrix_csv(
                core.synchronous,
                cross_row_axis,
                cross_column_axis,
            )
            files[f"{prefix}/asynchronous_matrix.csv"] = _matrix_csv(
                core.asynchronous,
                cross_row_axis,
                cross_column_axis,
            )
            files[f"{prefix}/qc_metrics.json"] = _json_bytes(core.qc_metrics)
    manifest_base = {
        "schema_version": "1.0",
        "artifact_type": "twodcos_run",
        "parent_baseline_run_id": prepared.baseline_run_id,
        "parent_baseline_fingerprint": prepared.baseline_fingerprint,
        "parent_prepared_data_sha256": prepared.prepared_data_sha256,
        "convention": str(getattr(analysis, "convention", "unknown")),
        "homo_result_count": len(homo_items) if homo_items is not None else 1,
        "cross_result_count": len(cross_items) if cross_items is not None else 0,
        "twodcos_fingerprint": getattr(result, "twodcos_fingerprint", None),
    }
    bundle = _build_manifest_archive(
        files,
        directories=directories,
        manifest_base=manifest_base,
    )
    if not verify_twodcos_bundle(bundle):
        raise ValueError("2D-COS bundle manifest verification failed")
    return bundle


def export_twodcos_bundle(
    prepared: PreparedSpectralDataset,
    result: Any,
    config: Any | None = None,
    destination: str | Path | None = None,
    *,
    figures: Mapping[str, bytes] | None = None,
    peak_order_files: Mapping[str, bytes] | None = None,
) -> bytes | Path:
    """Build a 2D-COS bundle and optionally write it to disk."""

    bundle = build_twodcos_bundle(
        prepared,
        result,
        config,
        figures=figures,
        peak_order_files=peak_order_files,
    )
    return _write_or_return_bundle(bundle, destination, default_name="twodcos_run.zip")


def _bundle_bytes(bundle: bytes | bytearray | memoryview | str | Path) -> bytes:
    if isinstance(bundle, (bytes, bytearray, memoryview)):
        return bytes(bundle)
    return Path(bundle).expanduser().read_bytes()


def _prepared_lineage(prepared: PreparedSpectralDataset) -> dict[str, str]:
    return {
        "parent_baseline_run_id": prepared.baseline_run_id,
        "parent_baseline_fingerprint": prepared.baseline_fingerprint,
        "parent_prepared_data_sha256": prepared.prepared_data_sha256,
    }


def _manifest_lineage(
    manifest: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, str]:
    lineage: dict[str, str] = {}
    for field_name in _PARENT_LINEAGE_FIELDS:
        value = manifest.get(field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context} is missing non-empty {field_name}")
        lineage[field_name] = value
    return lineage


def _assert_same_lineage(
    expected: Mapping[str, str],
    actual: Mapping[str, str],
    *,
    expected_context: str,
    actual_context: str,
) -> None:
    for field_name in _PARENT_LINEAGE_FIELDS:
        if expected[field_name] != actual[field_name]:
            raise ValueError(
                f"{field_name} mismatch between {expected_context} and {actual_context}"
            )


def _prepared_from_archive_members(
    archive: zipfile.ZipFile,
    *,
    csv_member: str,
    metadata_member: str,
    context: str,
) -> PreparedSpectralDataset:
    metadata = _load_metadata(archive.read(metadata_member))[0]
    recorded_csv = metadata.get("csv_file")
    if not isinstance(recorded_csv, str) or recorded_csv != csv_member:
        raise ValueError(
            f"{context} metadata csv_file must be exactly {csv_member!r}"
        )
    return _prepared_from_payloads(
        archive.read(csv_member),
        source_name=f"{context}!/{csv_member}",
        metadata=metadata,
    )


def _twodcos_lineage_from_verified_bundle(bundle: bytes) -> dict[str, str]:
    """Read already integrity-verified 2D lineage and revalidate its source data."""

    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if not isinstance(manifest, Mapping):
            raise ValueError("2D-COS manifest root must be an object")
        manifest_lineage = _manifest_lineage(manifest, context="2D-COS manifest")
        prepared = _prepared_from_archive_members(
            archive,
            csv_member="source_prepared_spectrum.csv",
            metadata_member="source_prepared_spectrum.meta.json",
            context="2D-COS bundle",
        )
    _assert_same_lineage(
        _prepared_lineage(prepared),
        manifest_lineage,
        expected_context="embedded prepared spectrum",
        actual_context="2D-COS manifest",
    )
    return manifest_lineage


def _baseline_lineage_from_verified_bundle(
    bundle: bytes,
    *,
    required: bool,
) -> dict[str, str] | None:
    """Read and cross-check baseline prepared sidecar/manifest lineage."""

    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        names = set(archive.namelist())
        has_csv = _DEFAULT_PREPARED_CSV in names
        has_metadata = _DEFAULT_PREPARED_META in names
        if not has_csv or not has_metadata:
            if required:
                raise ValueError(
                    "baseline bundle requires corrected prepared CSV and metadata "
                    "before it can parent a 2D-COS run"
                )
            return None
        prepared = _prepared_from_archive_members(
            archive,
            csv_member=_DEFAULT_PREPARED_CSV,
            metadata_member=_DEFAULT_PREPARED_META,
            context="baseline bundle",
        )
        manifest = json.loads(archive.read("manifest.json"))
        if not isinstance(manifest, Mapping):
            raise ValueError("baseline manifest root must be an object")
        prepared_entry = manifest.get("prepared_spectrum")
        if not isinstance(prepared_entry, Mapping):
            raise ValueError(
                "baseline manifest requires prepared_spectrum lineage before it can "
                "parent a 2D-COS run"
            )
        expected_paths = {
            "csv_path": _DEFAULT_PREPARED_CSV,
            "metadata_path": _DEFAULT_PREPARED_META,
        }
        for field_name, expected_path in expected_paths.items():
            if prepared_entry.get(field_name) != expected_path:
                raise ValueError(
                    f"baseline manifest prepared_spectrum.{field_name} must be "
                    f"exactly {expected_path!r}"
                )
        baseline_manifest_lineage = _manifest_lineage(
            {
                "parent_baseline_run_id": prepared_entry.get("baseline_run_id"),
                "parent_baseline_fingerprint": prepared_entry.get(
                    "baseline_fingerprint"
                ),
                "parent_prepared_data_sha256": prepared_entry.get(
                    "prepared_data_sha256"
                ),
            },
            context="baseline manifest prepared_spectrum",
        )
    prepared_lineage = _prepared_lineage(prepared)
    _assert_same_lineage(
        prepared_lineage,
        baseline_manifest_lineage,
        expected_context="baseline prepared sidecar",
        actual_context="baseline manifest prepared_spectrum",
    )
    return prepared_lineage


def verify_workbench_manifest(
    bundle: bytes | bytearray | memoryview | str | Path,
) -> bool:
    """Verify the canonical manifest, exact member set, sizes, and file hashes."""

    try:
        with zipfile.ZipFile(io.BytesIO(_bundle_bytes(bundle)), "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                return False
            for name in names:
                path = Path(name)
                if path.is_absolute() or ".." in path.parts:
                    return False
            manifest = json.loads(archive.read("manifest.json"))
            if not isinstance(manifest, dict):
                return False
            expected_manifest_hash = str(manifest.pop("manifest_sha256"))
            canonical = json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if _sha256(canonical) != expected_manifest_hash:
                return False
            file_entries = manifest["files"]
            listed = {str(entry["path"]) for entry in file_entries}
            directories = {str(name) for name in manifest.get("directories", ())}
            if set(names) != listed | directories | {"manifest.json"}:
                return False
            for entry in file_entries:
                payload = archive.read(str(entry["path"]))
                if len(payload) != int(entry["size_bytes"]):
                    return False
                if _sha256(payload) != str(entry["sha256"]):
                    return False
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ):
        return False
    return True


def verify_twodcos_bundle(
    bundle: bytes | bytearray | memoryview | str | Path,
) -> bool:
    """Verify both generic integrity and the required 2D run contract."""

    required = {
        "source_prepared_spectrum.csv",
        "source_prepared_spectrum.meta.json",
        "twodcos_config.json",
        "dynamic_spectra.csv",
        "synchronous_matrix.csv",
        "asynchronous_matrix.csv",
        "qc_metrics.json",
        "figures/",
        "peak_order/",
        "manifest.json",
    }
    try:
        payload = _bundle_bytes(bundle)
        if not verify_workbench_manifest(payload):
            return False
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            if not required.issubset(archive.namelist()):
                return False
        _twodcos_lineage_from_verified_bundle(payload)
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


def build_project_bundle(
    baseline_bundle: bytes | bytearray | memoryview | str | Path,
    twodcos_bundles: Sequence[bytes | bytearray | memoryview | str | Path] = (),
    project_config: Any | None = None,
) -> bytes:
    """Build a portable ``.ftirw`` archive from already verified run bundles."""

    baseline_payload = _bundle_bytes(baseline_bundle)
    if not verify_legacy_baseline_manifest(baseline_payload):
        raise ValueError("baseline_run.zip manifest verification failed")
    twodcos_payloads: list[bytes] = []
    for index, item in enumerate(twodcos_bundles, start=1):
        payload = _bundle_bytes(item)
        if not verify_twodcos_bundle(payload):
            raise ValueError(f"twodcos bundle {index} failed manifest verification")
        twodcos_payloads.append(payload)
    if twodcos_payloads:
        baseline_lineage = _baseline_lineage_from_verified_bundle(
            baseline_payload,
            required=True,
        )
        if baseline_lineage is None:  # pragma: no cover - required=True is exhaustive
            raise ValueError("baseline bundle has no prepared lineage")
        for index, payload in enumerate(twodcos_payloads, start=1):
            _assert_same_lineage(
                baseline_lineage,
                _twodcos_lineage_from_verified_bundle(payload),
                expected_context="baseline bundle",
                actual_context=f"twodcos bundle {index}",
            )
    files: dict[str, bytes] = {
        "baseline_run.zip": baseline_payload,
        "project_config.json": _json_bytes({} if project_config is None else project_config),
    }
    for index, payload in enumerate(twodcos_payloads, start=1):
        files[f"twodcos_run_{index:02d}.zip"] = payload
    bundle = _build_manifest_archive(
        files,
        manifest_base={
            "schema_version": "1.0",
            "artifact_type": "ftir_workbench_project",
            "twodcos_run_count": len(twodcos_payloads),
        },
    )
    if not verify_workbench_manifest(bundle):  # pragma: no cover - defensive
        raise ValueError("project bundle manifest verification failed")
    return bundle


def export_project_bundle(
    baseline_bundle: bytes | bytearray | memoryview | str | Path,
    twodcos_bundles: Sequence[bytes | bytearray | memoryview | str | Path] = (),
    project_config: Any | None = None,
    destination: str | Path | None = None,
) -> bytes | Path:
    """Build a project archive and optionally write ``project.ftirw``."""

    bundle = build_project_bundle(baseline_bundle, twodcos_bundles, project_config)
    return _write_or_return_bundle(bundle, destination, default_name="project.ftirw")


def verify_project_bundle(
    bundle: bytes | bytearray | memoryview | str | Path,
) -> bool:
    """Verify a project manifest and every nested baseline/2D run manifest."""

    try:
        payload = _bundle_bytes(bundle)
        if not verify_workbench_manifest(payload):
            return False
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            names = archive.namelist()
            if not {
                "baseline_run.zip",
                "project_config.json",
                "manifest.json",
            }.issubset(names):
                return False
            baseline_payload = archive.read("baseline_run.zip")
            if not verify_legacy_baseline_manifest(baseline_payload):
                return False
            twodcos_names = sorted(
                name
                for name in names
                if re.fullmatch(r"twodcos_run_\d{2}\.zip", name)
            )
            if twodcos_names:
                baseline_lineage = _baseline_lineage_from_verified_bundle(
                    baseline_payload,
                    required=True,
                )
                if baseline_lineage is None:
                    return False
                for name in twodcos_names:
                    twodcos_payload = archive.read(name)
                    if not verify_twodcos_bundle(twodcos_payload):
                        return False
                    _assert_same_lineage(
                        baseline_lineage,
                        _twodcos_lineage_from_verified_bundle(twodcos_payload),
                        expected_context="baseline bundle",
                        actual_context=name,
                    )
            manifest = json.loads(archive.read("manifest.json"))
            return int(manifest.get("twodcos_run_count", -1)) == len(twodcos_names)
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


# Readable compatibility aliases used by the implementation specification.
load_prepared_dataset = load_prepared
prepared_to_csv_bytes = prepared_csv_bytes
build_2dcos_bundle = build_twodcos_bundle
export_2dcos_bundle = export_twodcos_bundle
verify_bundle_manifest = verify_workbench_manifest


__all__ = [
    "PreparedExport",
    "PreparedPaths",
    "build_2dcos_bundle",
    "build_baseline_bundle",
    "build_project_bundle",
    "build_twodcos_bundle",
    "export_2dcos_bundle",
    "export_baseline_bundle",
    "export_prepared",
    "export_project_bundle",
    "export_twodcos_bundle",
    "load_prepared",
    "load_prepared_dataset",
    "prepared_csv_bytes",
    "prepared_metadata_dict",
    "prepared_to_csv_bytes",
    "serialize_prepared",
    "verify_bundle_manifest",
    "verify_project_bundle",
    "verify_twodcos_bundle",
    "verify_workbench_manifest",
]
