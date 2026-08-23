"""Strict readers for wide FTIR tables and collections of two-column DPT files."""

from __future__ import annotations

import csv
import fnmatch
import hashlib
import io
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import numpy as np

from .models import SpectralDataset

PathSource = str | Path
ReadableSource = PathSource | bytes | bytearray | BinaryIO | TextIO
_NUMERIC_TOKEN = re.compile(r"(?<![A-Za-z0-9.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_DELIMITERS = (",", "\t", ";")
_DELIMITER_NAMES = {",": "comma", "\t": "tab", ";": "semicolon"}


def sha256_bytes(content: bytes) -> str:
    """Return a lowercase SHA-256 digest for input provenance."""

    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file without altering it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_perturbation_value(label: str) -> float | None:
    """Extract the first standalone numeric token from a perturbation label.

    Examples include ``"0"``, ``"5 min"``, ``"RH_60"``, and ``"25C"``.
    The function does not use the value to reorder spectra.
    """

    text = str(label).strip()
    # The look-behind in the general expression prevents matching digits that
    # are embedded after letters, while separators such as '_' remain valid.
    match = _NUMERIC_TOKEN.search(text)
    if match is None:
        # Temperature-style labels such as ``25C`` already match; this fallback
        # permits a common prefix form such as ``RH60`` without treating every
        # alphanumeric identifier as numeric.
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if match is None:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def extract_perturbation_values(labels: Iterable[str]) -> np.ndarray:
    """Extract numeric perturbations, using NaN for labels needing user input."""

    extracted = [extract_perturbation_value(label) for label in labels]
    return np.asarray([np.nan if item is None else item for item in extracted], dtype=np.float64)


def detect_delimiter(sample: str | bytes) -> str:
    """Detect comma, tab, or semicolon delimiters without accepting others."""

    if isinstance(sample, bytes):
        try:
            text = sample.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Input must be UTF-8 text (an UTF-8 BOM is supported)") from exc
    else:
        text = sample.lstrip("\ufeff")
    nonempty = [line for line in text.splitlines() if line.strip()]
    if not nonempty:
        raise ValueError("Input file is empty")
    sniff_sample = "\n".join(nonempty[:30])
    try:
        dialect = csv.Sniffer().sniff(sniff_sample, delimiters="".join(_DELIMITERS))
        return dialect.delimiter
    except csv.Error:
        counts = {delimiter: nonempty[0].count(delimiter) for delimiter in _DELIMITERS}
        best = max(counts, key=counts.get)
        if counts[best] == 0:
            raise ValueError("Could not detect comma, tab, or semicolon delimiter") from None
        return best


def _normalise_delimiter(delimiter: str | None, text: str) -> str:
    if delimiter is None or str(delimiter).lower() in {"auto", "detect"}:
        return detect_delimiter(text)
    aliases = {
        "comma": ",",
        "tab": "\t",
        "tsv": "\t",
        "semicolon": ";",
        ",": ",",
        "\t": "\t",
        ";": ";",
    }
    try:
        return aliases[str(delimiter).lower()]
    except KeyError as exc:
        raise ValueError("delimiter must be comma, tab, semicolon, or auto") from exc


def _read_source(source: ReadableSource) -> tuple[bytes, str]:
    if isinstance(source, bytes | bytearray):
        return bytes(source), "uploaded_data"
    if isinstance(source, str | Path):
        path = Path(source)
        return path.read_bytes(), path.name

    name = Path(str(getattr(source, "name", "uploaded_data"))).name
    # Streamlit UploadedFile provides getvalue(), which avoids changing its cursor.
    getvalue = getattr(source, "getvalue", None)
    if callable(getvalue):
        content = getvalue()
    else:
        tell = getattr(source, "tell", None)
        seek = getattr(source, "seek", None)
        original_position = tell() if callable(tell) else None
        content = source.read()
        if original_position is not None and callable(seek):
            seek(original_position)
    if isinstance(content, str):
        return content.encode("utf-8"), name
    return bytes(content), name


def _decode_utf8(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Input must be UTF-8 text (an UTF-8 BOM is supported)") from exc


def _nonempty_rows(text: str, delimiter: str) -> list[list[str]]:
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter, strict=True))
    return [row for row in rows if any(cell.strip() for cell in row)]


def _float_cell(value: str, *, row: int, column: int, column_label: str) -> float:
    text = value.strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric value {value!r} at data row {row}, column {column} ({column_label!r})"
        ) from exc


def _direction(values: np.ndarray) -> str:
    if values.size < 2 or not np.all(np.isfinite(values)):
        return "undetermined"
    differences = np.diff(values)
    if np.all(differences > 0):
        return "ascending"
    if np.all(differences < 0):
        return "descending"
    return "non_monotonic"


def read_wide_file(
    source: ReadableSource,
    *,
    delimiter: str | None = None,
    intensity_unit: str = "unknown",
    perturbation: Sequence[float] | None = None,
) -> SpectralDataset:
    """Read a headered wide CSV/TXT/TSV file.

    The first column is interpreted as wavenumber.  Every remaining column is
    one spectrum and is transposed into the internal ``(m, n)`` orientation.
    Text in a numeric cell raises a location-aware error.  Missing, NaN, and Inf
    values remain represented and are rejected by validation rather than being
    silently dropped.
    """

    content, source_name = _read_source(source)
    text = _decode_utf8(content)
    actual_delimiter = _normalise_delimiter(delimiter, text)
    rows = _nonempty_rows(text, actual_delimiter)
    if len(rows) < 2:
        raise ValueError("Wide spectral input must contain a header and at least one data row")
    header = rows[0]
    if len(header) < 2:
        raise ValueError("Wide spectral input must contain wavenumber and at least one spectrum")
    expected_columns = len(header)
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != expected_columns:
            raise ValueError(
                f"Row {row_number} has {len(row)} columns; expected {expected_columns}"
            )

    labels = tuple(header[1:])
    numeric = np.empty((len(rows) - 1, expected_columns), dtype=np.float64)
    for row_index, row in enumerate(rows[1:], start=1):
        for column_index, cell in enumerate(row, start=1):
            numeric[row_index - 1, column_index - 1] = _float_cell(
                cell,
                row=row_index,
                column=column_index,
                column_label=header[column_index - 1],
            )

    wavenumber = numeric[:, 0]
    spectra = numeric[:, 1:].T.copy()
    extracted = extract_perturbation_values(labels)
    warnings: list[str] = []
    extraction_details: list[dict[str, Any]] = []
    for label, value in zip(labels, extracted, strict=True):
        success = bool(np.isfinite(value))
        extraction_details.append(
            {"label": label, "value": float(value) if success else None, "success": success}
        )
        if not success:
            warnings.append(
                f"Could not extract a numeric perturbation from label {label!r}; user input is required."
            )
    if perturbation is not None:
        perturbation_array = np.asarray(perturbation, dtype=np.float64)
        if perturbation_array.shape != (len(labels),):
            raise ValueError(
                f"Explicit perturbation must have shape {(len(labels),)}; "
                f"got {perturbation_array.shape}"
            )
        warnings.append(
            "Numeric perturbations were explicitly supplied and replace label extraction."
        )
    else:
        perturbation_array = extracted
    if len(set(labels)) != len(labels):
        warnings.append("Duplicate perturbation labels were preserved; verify their meaning.")

    direction = _direction(wavenumber)
    metadata: dict[str, Any] = {
        "input_format": "wide_table",
        "encoding": "utf-8-sig",
        "delimiter": actual_delimiter,
        "delimiter_name": _DELIMITER_NAMES[actual_delimiter],
        "source_sha256": sha256_bytes(content),
        "source_files": [{"name": source_name, "sha256": sha256_bytes(content)}],
        "original_column_names": list(header),
        "original_table_shape": [len(rows) - 1, expected_columns],
        "original_data_shape": [len(rows) - 1, len(labels)],
        "internal_shape": [len(labels), len(rows) - 1],
        "perturbation_extraction": extraction_details,
        "perturbation_original_labels": list(labels),
        "perturbation_final_labels": list(labels),
        "perturbation_order_changed": False,
        "original_wavenumber_direction": direction,
        "wavenumber_direction": direction,
        "wavenumber_order_changed": False,
        "parse_warnings": warnings,
        "processing_history": [
            {
                "operation": "read_wide_file",
                "delimiter": _DELIMITER_NAMES[actual_delimiter],
                "spectra_transposed_to_internal_m_by_n": True,
            }
        ],
    }
    return SpectralDataset(
        wavenumber=wavenumber,
        perturbation=perturbation_array,
        perturbation_labels=labels,
        spectra=spectra,
        intensity_unit=intensity_unit,
        source_name=source_name,
        metadata=metadata,
    )


def _read_dpt_spectrum(
    source: ReadableSource,
    *,
    delimiter: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    content, source_name = _read_source(source)
    text = _decode_utf8(content)
    actual_delimiter = _normalise_delimiter(delimiter, text)
    rows = _nonempty_rows(text, actual_delimiter)
    if not rows:
        raise ValueError(f"DPT file {source_name!r} is empty")

    header_skipped = False
    try:
        float(rows[0][0].strip())
        float(rows[0][1].strip())
    except (ValueError, IndexError):
        header_skipped = True
        rows = rows[1:]
    if not rows:
        raise ValueError(f"DPT file {source_name!r} contains no numeric data")

    values = np.empty((len(rows), 2), dtype=np.float64)
    for row_number, row in enumerate(rows, start=2 if header_skipped else 1):
        if len(row) != 2:
            raise ValueError(
                f"DPT file {source_name!r}, row {row_number} has {len(row)} columns; expected 2"
            )
        values[row_number - (2 if header_skipped else 1), 0] = _float_cell(
            row[0], row=row_number, column=1, column_label="wavenumber"
        )
        values[row_number - (2 if header_skipped else 1), 1] = _float_cell(
            row[1], row=row_number, column=2, column_label="intensity"
        )
    details = {
        "name": source_name,
        "sha256": sha256_bytes(content),
        "delimiter": actual_delimiter,
        "delimiter_name": _DELIMITER_NAMES[actual_delimiter],
        "header_skipped": header_skipped,
        "n_points": len(rows),
    }
    return values[:, 0], values[:, 1], details


def read_dpt_files(
    sources: Sequence[ReadableSource],
    *,
    intensity_unit: str = "unknown",
    delimiter: str | None = None,
    perturbation: Sequence[float] | None = None,
) -> SpectralDataset:
    """Read ordered, two-column DPT spectra without sorting or interpolation."""

    if not sources:
        raise ValueError("At least one DPT file is required")
    axes: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    for source in sources:
        axis, intensity, source_details = _read_dpt_spectrum(source, delimiter=delimiter)
        axes.append(axis)
        spectra.append(intensity)
        details.append(source_details)

    reference_axis = axes[0]
    for index, axis in enumerate(axes[1:], start=1):
        if axis.shape != reference_axis.shape:
            raise ValueError(
                f"DPT wavenumber length mismatch: {details[0]['name']!r} has "
                f"{reference_axis.size} points but {details[index]['name']!r} has {axis.size}. "
                "No interpolation was performed."
            )
        if not np.array_equal(axis, reference_axis, equal_nan=True):
            finite = np.isfinite(axis) & np.isfinite(reference_axis)
            max_difference = (
                float(np.max(np.abs(axis[finite] - reference_axis[finite])))
                if np.any(finite)
                else None
            )
            raise ValueError(
                f"DPT wavenumber axes differ between {details[0]['name']!r} and "
                f"{details[index]['name']!r} (max finite difference={max_difference}). "
                "No sorting or interpolation was performed."
            )

    labels = tuple(Path(item["name"]).stem for item in details)
    extracted = extract_perturbation_values(labels)
    warnings: list[str] = []
    extraction_details: list[dict[str, Any]] = []
    for label, value in zip(labels, extracted, strict=True):
        success = bool(np.isfinite(value))
        extraction_details.append(
            {"label": label, "value": float(value) if success else None, "success": success}
        )
        if not success:
            warnings.append(
                f"Could not extract a numeric perturbation from DPT filename {label!r}; "
                "user input is required."
            )
    if perturbation is not None:
        perturbation_array = np.asarray(perturbation, dtype=np.float64)
        if perturbation_array.shape != (len(labels),):
            raise ValueError(
                f"Explicit perturbation must have shape {(len(labels),)}; "
                f"got {perturbation_array.shape}"
            )
        warnings.append(
            "Numeric perturbations were explicitly supplied and replace filename extraction."
        )
    else:
        perturbation_array = extracted

    direction = _direction(reference_axis)
    common_parents: set[str] = set()
    for source in sources:
        if isinstance(source, str | Path):
            common_parents.add(str(Path(source).parent))
    source_name = Path(next(iter(common_parents))).name if len(common_parents) == 1 else "DPT_files"
    metadata: dict[str, Any] = {
        "input_format": "dpt_collection",
        "encoding": "utf-8-sig",
        "source_files": details,
        "source_sha256": {item["name"]: item["sha256"] for item in details},
        "original_data_shape": [int(reference_axis.size), len(spectra)],
        "internal_shape": [len(spectra), int(reference_axis.size)],
        "perturbation_extraction": extraction_details,
        "perturbation_original_labels": list(labels),
        "perturbation_final_labels": list(labels),
        "perturbation_order_changed": False,
        "original_wavenumber_direction": direction,
        "wavenumber_direction": direction,
        "wavenumber_order_changed": False,
        "parse_warnings": warnings,
        "processing_history": [
            {
                "operation": "read_dpt_files",
                "file_order": list(labels),
                "file_order_changed": False,
                "interpolation_performed": False,
            }
        ],
    }
    return SpectralDataset(
        wavenumber=reference_axis,
        perturbation=perturbation_array,
        perturbation_labels=labels,
        spectra=np.vstack(spectra).astype(np.float64, copy=False),
        intensity_unit=intensity_unit,
        source_name=source_name,
        metadata=metadata,
    )


def read_dpt_directory(
    directory: str | Path,
    *,
    pattern: str = "*MIN.dpt",
    intensity_unit: str = "unknown",
    delimiter: str | None = None,
    perturbation: Sequence[float] | None = None,
) -> SpectralDataset:
    """Enumerate a DPT directory deterministically and record that enumeration.

    The default pattern deliberately excludes a file named ``BASELINE.dpt``.
    Filename sorting here only makes directory enumeration deterministic; it is
    not numeric perturbation sorting.
    """

    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    matches = sorted(
        (
            item
            for item in path.iterdir()
            if item.is_file() and fnmatch.fnmatch(item.name.lower(), pattern.lower())
        ),
        key=lambda item: (item.name.casefold(), item.name),
    )
    if not matches:
        raise ValueError(f"No DPT files in {path} matched pattern {pattern!r}")
    dataset = read_dpt_files(
        matches,
        intensity_unit=intensity_unit,
        delimiter=delimiter,
        perturbation=perturbation,
    )
    metadata = dict(dataset.metadata)
    parse_warnings = list(metadata.get("parse_warnings", []))
    parse_warnings.append(
        "Directory entries were deterministically sorted by filename; this is not "
        "perturbation-value sorting."
    )
    history = list(metadata.get("processing_history", []))
    history.insert(
        0,
        {
            "operation": "enumerate_dpt_directory",
            "directory": str(path),
            "pattern": pattern,
            "enumeration_order": [item.name for item in matches],
            "ordering_rule": "filename_casefold_ascending",
        },
    )
    metadata.update(
        {
            "input_directory": str(path),
            "directory_pattern": pattern,
            "directory_enumeration_order": [item.name for item in matches],
            "directory_ordering_rule": "filename_casefold_ascending",
            "parse_warnings": parse_warnings,
            "processing_history": history,
        }
    )
    return dataset.with_updates(source_name=path.name, metadata=metadata)


def load_input(
    source: ReadableSource | Sequence[ReadableSource],
    *,
    intensity_unit: str = "unknown",
    delimiter: str | None = None,
    perturbation: Sequence[float] | None = None,
    perturbation_order: str = "preserve_file_order",
    dpt_pattern: str = "*MIN.dpt",
) -> SpectralDataset:
    """Dispatch a path, upload, directory, or DPT sequence to the strict reader."""

    if isinstance(source, Sequence) and not isinstance(
        source, str | bytes | bytearray | Path | io.IOBase
    ):
        dataset = read_dpt_files(
            source,
            intensity_unit=intensity_unit,
            delimiter=delimiter,
            perturbation=perturbation,
        )
    elif isinstance(source, str | Path) and Path(source).is_dir():
        dataset = read_dpt_directory(
            source,
            pattern=dpt_pattern,
            intensity_unit=intensity_unit,
            delimiter=delimiter,
            perturbation=perturbation,
        )
    elif isinstance(source, str | Path) and Path(source).suffix.lower() == ".dpt":
        dataset = read_dpt_files(
            [source],
            intensity_unit=intensity_unit,
            delimiter=delimiter,
            perturbation=perturbation,
        )
    else:
        dataset = read_wide_file(
            source,
            delimiter=delimiter,
            intensity_unit=intensity_unit,
            perturbation=perturbation,
        )

    # Import locally to avoid making validation depend on reader internals.
    from .validation import apply_perturbation_order

    return apply_perturbation_order(dataset, perturbation_order)


# Clear aliases for callers using either "read" or "load" terminology.
load_spectral_file = read_wide_file
load_wide_file = read_wide_file
load_dpt_files = read_dpt_files


__all__ = [
    "detect_delimiter",
    "extract_perturbation_value",
    "extract_perturbation_values",
    "load_dpt_files",
    "load_input",
    "load_spectral_file",
    "load_wide_file",
    "read_dpt_directory",
    "read_dpt_files",
    "read_wide_file",
    "sha256_bytes",
    "sha256_file",
]
