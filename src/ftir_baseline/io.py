"""Strict readers for delimited plain-text FTIR spectra."""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .models import SpectrumSet
from .validation import SpectrumValidationError, validate_matching_axes

PathLike = str | Path
SupportedUnit = Literal[
    "absorbance",
    "percent_transmittance",
    "fraction_transmittance",
]
SUPPORTED_TEXT_EXTENSIONS = frozenset(
    {".csv", ".tsv", ".tab", ".txt", ".dpt", ".asc", ".dat", ".xy"}
)
# Backward-compatible name used by callers and directory discovery.
SUPPORTED_EXTENSIONS = SUPPORTED_TEXT_EXTENSIONS


class SpectrumReadError(SpectrumValidationError):
    """Raised when a spectral text file cannot be parsed without guessing."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_text(path: Path, encoding: str | None) -> tuple[str, str]:
    if encoding is not None:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeError as exc:
            raise SpectrumReadError(f"cannot decode {path} as {encoding}") from exc
    failures: list[str] = []
    for candidate in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return path.read_text(encoding=candidate), candidate
        except UnicodeError:
            failures.append(candidate)
    raise SpectrumReadError(f"cannot decode {path}; attempted {', '.join(failures)}")


def _first_content_line(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            return line
    raise SpectrumReadError("file contains no data rows")


def _detect_delimiter(first_line: str) -> str | None:
    counts = {delimiter: first_line.count(delimiter) for delimiter in (",", "\t", ";")}
    delimiter, count = max(counts.items(), key=lambda item: item[1])
    return delimiter if count else None


def _tokenize(line: str, delimiter: str | None) -> list[str]:
    if delimiter is None:
        return re.split(r"\s+", line.strip())
    return next(csv.reader([line], delimiter=delimiter, skipinitialspace=True))


def _float_token(token: str) -> float:
    return float(token.strip())


def _parse_numeric_table(
    path: Path,
    *,
    delimiter: str | None,
    encoding: str | None,
) -> tuple[np.ndarray, tuple[str, ...] | None, str, str]:
    text, used_encoding = _read_text(path, encoding)
    if delimiter == "auto":
        delimiter = _detect_delimiter(_first_content_line(text))
    elif delimiter is not None and len(delimiter) != 1:
        raise ValueError("delimiter must be one character, None for whitespace, or 'auto'")

    header: tuple[str, ...] | None = None
    rows: list[list[float]] = []
    expected_columns: int | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = [token.strip() for token in _tokenize(stripped, delimiter)]
        if any(token == "" for token in tokens):
            raise SpectrumReadError(f"{path}:{line_number} contains an empty field")
        try:
            numeric = [_float_token(token) for token in tokens]
        except ValueError as exc:
            if not rows and header is None:
                header = tuple(tokens)
                expected_columns = len(tokens)
                continue
            raise SpectrumReadError(
                f"{path}:{line_number} contains a non-numeric value; only one optional "
                "header row is supported"
            ) from exc
        if expected_columns is None:
            expected_columns = len(numeric)
        if len(numeric) != expected_columns:
            raise SpectrumReadError(
                f"{path}:{line_number} has {len(numeric)} columns; expected {expected_columns}"
            )
        rows.append(numeric)

    if not rows:
        raise SpectrumReadError(f"{path} contains no numeric data rows")
    if expected_columns is None or expected_columns < 2:
        raise SpectrumReadError(
            f"{path} must contain wavenumber plus at least one intensity column"
        )
    table = np.asarray(rows, dtype=np.float64)
    delimiter_name = "whitespace" if delimiter is None else repr(delimiter)
    return table, header, used_encoding, delimiter_name


_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_perturbation_label(label: str) -> float | None:
    """Extract the first finite number from a label such as ``12MIN``."""

    match = _NUMBER_PATTERN.search(label)
    if match is None:
        return None
    value = float(match.group(0))
    return value if np.isfinite(value) else None


def _unit(value: str) -> SupportedUnit:
    allowed = {"absorbance", "percent_transmittance", "fraction_transmittance"}
    if value not in allowed:
        raise ValueError(
            f"input_unit must be one of {sorted(allowed)}; got {value!r}. "
            "The reader does not infer physical units."
        )
    return value  # type: ignore[return-value]


def _coordinates_from_labels(labels: Sequence[str]) -> tuple[np.ndarray, str, tuple[str, ...]]:
    parsed = tuple(parse_perturbation_label(label) for label in labels)
    missing = tuple(label for label, value in zip(labels, parsed, strict=True) if value is None)
    if not missing:
        return np.asarray(parsed, dtype=np.float64), "parsed_from_labels", ()
    return np.arange(len(labels), dtype=np.float64), "sequential_index", missing


def read_spectrum_file(
    path: PathLike,
    *,
    input_unit: SupportedUnit = "absorbance",
    perturbation: Sequence[float] | float | None = None,
    perturbation_labels: Sequence[str] | None = None,
    delimiter: str | None = "auto",
    encoding: str | None = None,
    source_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    sort_by_perturbation: bool = False,
) -> SpectrumSet:
    """Read one narrow or wide plain-text spectrum table.

    The supported wide layout is ``wavenumber, spectrum_1, spectrum_2, ...``.
    One optional header row is accepted.  Wavenumber rows are never sorted or
    deduplicated.  Wide-table spectrum columns are reordered only when
    ``sort_by_perturbation=True`` is explicitly requested and auditable labels
    contain numeric perturbation values.
    """

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"spectrum file does not exist: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise SpectrumReadError(
            f"unsupported spectrum extension {file_path.suffix!r}; expected one of "
            f"{sorted(SUPPORTED_TEXT_EXTENSIONS)}. Structured or vendor-binary spectra must "
            "first be exported as a supported text table."
        )
    table, header, used_encoding, used_delimiter = _parse_numeric_table(
        file_path, delimiter=delimiter, encoding=encoding
    )
    wavenumber = table[:, 0]
    spectra = table[:, 1:].T
    n_spectra = spectra.shape[0]

    if perturbation_labels is not None:
        labels = tuple(str(value) for value in perturbation_labels)
    elif header is not None:
        labels = tuple(str(value) for value in header[1:])
    elif n_spectra == 1:
        labels = (file_path.stem,)
    else:
        labels = tuple(f"spectrum_{index}" for index in range(n_spectra))
    if len(labels) != n_spectra:
        raise SpectrumReadError(f"expected {n_spectra} perturbation labels, received {len(labels)}")

    if perturbation is None:
        coordinates, inference, unparsed = _coordinates_from_labels(labels)
    else:
        coordinates = np.asarray(perturbation, dtype=np.float64)
        if coordinates.ndim == 0:
            coordinates = coordinates.reshape(1)
        inference = "explicit"
        unparsed = ()

    file_metadata: dict[str, Any] = dict(metadata or {})
    file_metadata.update(
        {
            "source_path": str(file_path.resolve()),
            "source_sha256": _sha256(file_path),
            "source_format": file_path.suffix.lower().lstrip("."),
            "encoding": used_encoding,
            "delimiter": used_delimiter,
            "header": header,
            "perturbation_inference": inference,
            "unparsed_perturbation_labels": unparsed,
        }
    )
    result = SpectrumSet(
        wavenumber=wavenumber,
        perturbation=coordinates,
        perturbation_labels=labels,
        spectra=spectra,
        intensity_unit=_unit(input_unit),
        source_name=source_name or file_path.name,
        metadata=file_metadata,
    )
    if not sort_by_perturbation:
        return result
    if inference == "sequential_index":
        raise SpectrumReadError(
            "cannot sort wide-table spectra by perturbation because these labels have no "
            f"numeric value: {list(unparsed)}"
        )
    order = np.argsort(result.perturbation, kind="stable")
    sorted_metadata = result.mutable_metadata()
    sorted_metadata.update(
        {
            "original_spectrum_order": list(result.perturbation_labels),
            "final_spectrum_order": [result.perturbation_labels[int(index)] for index in order],
            "sorted_by_perturbation": True,
            "order_policy": "numeric_perturbation_stable",
        }
    )
    return SpectrumSet(
        wavenumber=result.wavenumber,
        perturbation=result.perturbation[order],
        perturbation_labels=tuple(result.perturbation_labels[int(index)] for index in order),
        spectra=result.spectra[order],
        intensity_unit=result.intensity_unit,
        source_name=result.source_name,
        metadata=sorted_metadata,
    )


def _resolve_paths(paths: Sequence[PathLike] | PathLike) -> list[Path]:
    if isinstance(paths, (str, Path)):
        candidate = Path(paths).expanduser()
        if candidate.is_dir():
            discovered = (
                path
                for path in candidate.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            return sorted(discovered, key=lambda path: (path.name.casefold(), path.name))
        return [candidate]
    return [Path(path).expanduser() for path in paths]


def load_spectrum_files(
    paths: Sequence[PathLike] | PathLike,
    *,
    input_unit: SupportedUnit = "absorbance",
    perturbations: Sequence[float] | None = None,
    perturbation_labels: Sequence[str] | None = None,
    label_parser: Callable[[str], float | None] = parse_perturbation_label,
    sort_by_perturbation: bool = False,
    exclude_names: Sequence[str] = (),
    delimiter: str | None = "auto",
    encoding: str | None = None,
    source_name: str | None = None,
) -> SpectrumSet:
    """Load a sequence of single-spectrum files with strict axis matching.

    Explicit path order is the default and is never silently changed.  Numerical
    sorting occurs only with ``sort_by_perturbation=True`` and is recorded, along
    with the original/final order and exclusions, in metadata.
    """

    directory_input = isinstance(paths, (str, Path)) and Path(paths).expanduser().is_dir()
    if directory_input and not sort_by_perturbation:
        raise SpectrumReadError(
            "a directory has no portable acquisition order; pass "
            "sort_by_perturbation=True for an explicit numeric order, or pass an ordered "
            "path list to preserve a scientifically defined acquisition order"
        )

    original_paths = _resolve_paths(paths)
    if not original_paths:
        raise SpectrumReadError(
            "no supported text-spectrum files were supplied; expected one of "
            f"{sorted(SUPPORTED_TEXT_EXTENSIONS)}"
        )
    excluded_lookup = {name.casefold() for name in exclude_names}
    excluded = [path for path in original_paths if path.name.casefold() in excluded_lookup]
    selected = [path for path in original_paths if path.name.casefold() not in excluded_lookup]
    if not selected:
        raise SpectrumReadError("all supplied spectrum files were excluded")
    folded_names = [path.name.casefold() for path in selected]
    duplicate_names = sorted(
        {path.name for path in selected if folded_names.count(path.name.casefold()) > 1},
        key=str.casefold,
    )
    if duplicate_names:
        raise SpectrumReadError(
            "spectrum files must have unique basenames for unambiguous labels and per-file "
            f"hash provenance; duplicates: {duplicate_names}"
        )
    missing = [path for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"spectrum files do not exist: {[str(path) for path in missing]}")

    labels = (
        tuple(str(label) for label in perturbation_labels)
        if perturbation_labels is not None
        else tuple(path.stem for path in selected)
    )
    if len(labels) != len(selected):
        raise SpectrumReadError(
            f"expected {len(selected)} perturbation labels, received {len(labels)}"
        )

    if perturbations is not None:
        coordinates = np.asarray(perturbations, dtype=np.float64)
        coordinate_source = "explicit"
        unparsed: tuple[str, ...] = ()
    else:
        parsed = tuple(label_parser(label) for label in labels)
        unparsed = tuple(
            label for label, value in zip(labels, parsed, strict=True) if value is None
        )
        if unparsed:
            if sort_by_perturbation:
                raise SpectrumReadError(
                    "cannot sort by perturbation because these labels have no numeric value: "
                    f"{list(unparsed)}"
                )
            coordinates = np.arange(len(selected), dtype=np.float64)
            coordinate_source = "sequential_index"
        else:
            coordinates = np.asarray(parsed, dtype=np.float64)
            coordinate_source = "parsed_from_labels"
    if coordinates.shape != (len(selected),):
        raise SpectrumReadError(
            f"perturbations must have shape ({len(selected)},); got {coordinates.shape}"
        )
    if not np.isfinite(coordinates).all():
        raise SpectrumReadError("perturbations contain NaN or Inf")

    order = np.arange(len(selected))
    if sort_by_perturbation:
        order = np.argsort(coordinates, kind="stable")
        selected = [selected[int(index)] for index in order]
        labels = tuple(labels[int(index)] for index in order)
        coordinates = coordinates[order]

    loaded = [
        read_spectrum_file(
            path,
            input_unit=input_unit,
            delimiter=delimiter,
            encoding=encoding,
        )
        for path in selected
    ]
    for path, spectrum_set in zip(selected, loaded, strict=True):
        if spectrum_set.n_spectra != 1:
            raise SpectrumReadError(
                f"{path} contains {spectrum_set.n_spectra} spectra; multi-file loading "
                "requires exactly one intensity column per file"
            )
    reference = loaded[0].wavenumber
    for path, spectrum_set in zip(selected[1:], loaded[1:], strict=True):
        validate_matching_axes(
            reference,
            spectrum_set.wavenumber,
            reference_name=selected[0].name,
            candidate_name=path.name,
        )

    hashes = {
        path.name: data.metadata["source_sha256"]
        for path, data in zip(selected, loaded, strict=True)
    }
    combined_hash = hashlib.sha256(
        "".join(str(data.metadata["source_sha256"]) for data in loaded).encode("ascii")
    ).hexdigest()
    metadata = {
        "source_paths": tuple(str(path.resolve()) for path in selected),
        "source_sha256_by_file": hashes,
        "combined_source_sha256": combined_hash,
        "original_file_order": tuple(path.name for path in original_paths),
        "final_file_order": tuple(path.name for path in selected),
        "excluded_files": tuple(path.name for path in excluded),
        "sorted_by_perturbation": bool(sort_by_perturbation),
        "order_policy": "numeric_perturbation_stable" if sort_by_perturbation else "input_order",
        "directory_input": bool(directory_input),
        "directory_discovery_policy": "lexical_filename" if directory_input else None,
        "perturbation_inference": coordinate_source,
        "unparsed_perturbation_labels": unparsed,
    }
    return SpectrumSet(
        wavenumber=reference,
        perturbation=coordinates,
        perturbation_labels=labels,
        spectra=np.vstack([data.spectra[0] for data in loaded]),
        intensity_unit=_unit(input_unit),
        source_name=source_name or f"{len(selected)}-file spectrum series",
        metadata=metadata,
    )


def load_spectrum_directory(
    directory: PathLike,
    *,
    input_unit: SupportedUnit = "absorbance",
    exclude_names: Sequence[str] = ("BASELINE.dpt",),
    sort_by_perturbation: bool = False,
    **kwargs: Any,
) -> SpectrumSet:
    """Convenience loader for an acquisition directory.

    ``BASELINE.dpt`` is excluded by this explicitly named directory workflow; the
    exclusion is recorded.  Because filesystem iteration order is not portable,
    directory loading requires ``sort_by_perturbation=True``.  Callers that have a
    separate acquisition order should pass that ordered path list to
    :func:`load_spectrum_files` instead.
    """

    path = Path(directory).expanduser()
    if not path.is_dir():
        raise NotADirectoryError(f"spectrum directory does not exist: {path}")
    return load_spectrum_files(
        path,
        input_unit=input_unit,
        exclude_names=exclude_names,
        sort_by_perturbation=sort_by_perturbation,
        **kwargs,
    )


# Concise aliases used by CLI/pipeline callers.
read_spectrum = read_spectrum_file
read_spectrum_series = load_spectrum_files


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_TEXT_EXTENSIONS",
    "SpectrumReadError",
    "load_spectrum_directory",
    "load_spectrum_files",
    "parse_perturbation_label",
    "read_spectrum",
    "read_spectrum_file",
    "read_spectrum_series",
]
