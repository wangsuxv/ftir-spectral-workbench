"""Strict readers for delimited plain-text FTIR spectra."""

from __future__ import annotations

import codecs
import csv
import hashlib
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
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

_DELIMITER_NAMES = {"auto", "comma", "tab", "semicolon", "whitespace"}
_DECIMAL_MARKS = {"auto", "dot", "comma"}
_HEADER_MODES = {"auto", "present", "absent"}
_SUPPORTED_ENCODINGS = {
    "auto",
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "gb18030",
    "cp1252",
}
_DELIMITER_CHARACTERS: dict[str, str | None] = {
    "comma": ",",
    "tab": "\t",
    "semicolon": ";",
    "whitespace": None,
}
_EXTENSION_DELIMITER_HINTS = {
    ".tsv": "tab",
    ".tab": "tab",
}
_DOT_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_COMMA_NUMBER = re.compile(r"^[+-]?(?:\d+(?:,\d*)?|,\d+)(?:[eE][+-]?\d+)?$")
_INTEGER_NUMBER = re.compile(r"^[+-]?\d+(?:[eE][+-]?\d+)?$")
_NONFINITE_TOKENS = frozenset(
    {
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }
)


def _freeze_json_value(value: Any) -> Any:
    """Recursively freeze a JSON-like value used by public probe evidence."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: Any) -> Any:
    """Return an independent JSON-serializable copy of a frozen evidence value."""

    if isinstance(value, Mapping):
        return {str(key): _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _normalize_delimiter(value: str | None, *, allow_auto: bool = True) -> str:
    if value is None:
        return "whitespace"
    aliases = {",": "comma", "\t": "tab", ";": "semicolon", "space": "whitespace"}
    normalized = aliases.get(str(value), str(value).strip().lower())
    if normalized == "auto" and allow_auto:
        return normalized
    if normalized in _DELIMITER_NAMES - {"auto"}:
        return normalized
    if len(str(value)) == 1:
        return str(value)
    expected = sorted(_DELIMITER_NAMES if allow_auto else _DELIMITER_NAMES - {"auto"})
    raise ValueError(
        f"delimiter must be one of {expected}, a legacy one-character delimiter, or None"
    )


def _normalize_encoding(value: str | None) -> str:
    if value is None:
        return "auto"
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "utf8": "utf-8",
        "utf8-sig": "utf-8-sig",
        "utf16": "utf-16",
        "utf16-le": "utf-16-le",
        "utf16-be": "utf-16-be",
        "windows-1252": "cp1252",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _SUPPORTED_ENCODINGS:
        try:
            codecs.lookup(normalized)
        except LookupError as exc:
            raise ValueError(f"unknown text encoding {value!r}") from exc
    return normalized


@dataclass(frozen=True, slots=True)
class TextImportOptions:
    """Auditable controls for a delimited plain-text spectrum import."""

    delimiter: str = "auto"
    decimal_mark: str = "auto"
    encoding: str = "auto"
    header_mode: str = "auto"
    skip_rows: int = 0
    allow_preamble: bool = True
    trim_empty_edge_columns: bool = True
    comment_prefixes: tuple[str, ...] = ("#", "//", "%")

    def __post_init__(self) -> None:
        delimiter = _normalize_delimiter(self.delimiter)
        decimal_mark = str(self.decimal_mark).strip().lower()
        if decimal_mark not in _DECIMAL_MARKS:
            raise ValueError(f"decimal_mark must be one of {sorted(_DECIMAL_MARKS)}")
        encoding = _normalize_encoding(self.encoding)
        header_mode = str(self.header_mode).strip().lower()
        if header_mode not in _HEADER_MODES:
            raise ValueError(f"header_mode must be one of {sorted(_HEADER_MODES)}")
        if isinstance(self.skip_rows, bool) or not isinstance(self.skip_rows, int):
            raise TypeError("skip_rows must be an integer")
        if self.skip_rows < 0:
            raise ValueError("skip_rows must be non-negative")
        prefixes = tuple(str(prefix) for prefix in self.comment_prefixes)
        if any(not prefix for prefix in prefixes):
            raise ValueError("comment prefixes must be non-empty strings")
        if delimiter == "comma" and decimal_mark == "comma":
            raise ValueError(
                "comma delimiter and comma decimal mark are ambiguous; choose a non-comma "
                "delimiter or dot decimal mark"
            )
        object.__setattr__(self, "delimiter", delimiter)
        object.__setattr__(self, "decimal_mark", decimal_mark)
        object.__setattr__(self, "encoding", encoding)
        object.__setattr__(self, "header_mode", header_mode)
        object.__setattr__(self, "allow_preamble", bool(self.allow_preamble))
        object.__setattr__(
            self,
            "trim_empty_edge_columns",
            bool(self.trim_empty_edge_columns),
        )
        object.__setattr__(self, "comment_prefixes", prefixes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "delimiter": self.delimiter,
            "decimal_mark": self.decimal_mark,
            "encoding": self.encoding,
            "header_mode": self.header_mode,
            "skip_rows": self.skip_rows,
            "allow_preamble": self.allow_preamble,
            "trim_empty_edge_columns": self.trim_empty_edge_columns,
            "comment_prefixes": list(self.comment_prefixes),
        }


@dataclass(frozen=True, slots=True)
class ImportProbe:
    """Structured diagnosis emitted by the same parser used for formal loading."""

    source_name: str
    extension: str
    source_sha256: str
    size_bytes: int
    selected_encoding: str
    encoding_evidence: str
    selected_delimiter: str
    delimiter_evidence: Mapping[str, Any]
    selected_decimal_mark: str
    decimal_evidence: Mapping[str, Any]
    header_present: bool
    header: tuple[str, ...] | None
    skipped_explicit_rows: int
    skipped_preamble_lines: tuple[int, ...]
    skipped_comment_lines: tuple[int, ...]
    trimmed_empty_edge_columns: int
    numeric_block_start_line: int
    numeric_block_end_line: int
    data_rows: int
    columns: int
    layout: Literal["two_column", "wide_table"]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "delimiter_evidence",
            _freeze_json_value(self.delimiter_evidence),
        )
        object.__setattr__(
            self,
            "decimal_evidence",
            _freeze_json_value(self.decimal_evidence),
        )
        object.__setattr__(self, "header", None if self.header is None else tuple(self.header))
        object.__setattr__(self, "skipped_preamble_lines", tuple(self.skipped_preamble_lines))
        object.__setattr__(self, "skipped_comment_lines", tuple(self.skipped_comment_lines))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable provenance payload."""

        return {
            "source_name": self.source_name,
            "extension": self.extension,
            "source_sha256": self.source_sha256,
            "size_bytes": self.size_bytes,
            "selected_encoding": self.selected_encoding,
            "encoding_evidence": self.encoding_evidence,
            "selected_delimiter": self.selected_delimiter,
            "delimiter_evidence": _thaw_json_value(self.delimiter_evidence),
            "selected_decimal_mark": self.selected_decimal_mark,
            "decimal_evidence": _thaw_json_value(self.decimal_evidence),
            "header_present": self.header_present,
            "header": None if self.header is None else list(self.header),
            "skipped_explicit_rows": self.skipped_explicit_rows,
            "skipped_preamble_lines": list(self.skipped_preamble_lines),
            "skipped_comment_lines": list(self.skipped_comment_lines),
            "trimmed_empty_edge_columns": self.trimmed_empty_edge_columns,
            "numeric_block_start_line": self.numeric_block_start_line,
            "numeric_block_end_line": self.numeric_block_end_line,
            "data_rows": self.data_rows,
            "columns": self.columns,
            "layout": self.layout,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class _ContentLine:
    number: int
    raw: str


@dataclass(frozen=True, slots=True)
class _NumericRow:
    line: _ContentLine
    tokens: tuple[str, ...]
    values: tuple[float, ...]
    leading_empty: int
    trailing_empty: int


@dataclass(frozen=True, slots=True)
class _CandidateRun:
    start_index: int
    end_index: int
    rows: tuple[_NumericRow, ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def columns(self) -> int:
        return len(self.rows[0].values)


@dataclass(frozen=True, slots=True)
class _DelimiterCandidate:
    delimiter: str
    decimal_mark: str
    runs: tuple[_CandidateRun, ...]
    evidence_count: int

    @property
    def best_row_count(self) -> int:
        return max((run.row_count for run in self.runs), default=0)

    @property
    def best_columns(self) -> int:
        best = max(self.runs, key=lambda run: (run.row_count, run.columns), default=None)
        return 0 if best is None else best.columns

    @property
    def numeric_row_count(self) -> int:
        return sum(run.row_count for run in self.runs)


@dataclass(frozen=True, slots=True)
class _ParsedTextTable:
    table: np.ndarray
    header: tuple[str, ...] | None
    probe: ImportProbe


class SpectrumReadError(SpectrumValidationError):
    """Raised when a spectral text file cannot be parsed without guessing."""


def _unsupported_extension_error(path: Path) -> SpectrumReadError:
    suffix = path.suffix.lower()
    vendor_formats = {
        ".spa": "Thermo OMNIC",
        ".spg": "Thermo OMNIC",
        ".srs": "Thermo OMNIC",
        ".spc": "Galactic/SPC",
        ".sp": "PerkinElmer",
    }
    structured = {".jdx", ".dx", ".jcamp", ".xls", ".xlsx", ".zip"}
    format_name = vendor_formats.get(suffix)
    if suffix[1:].isdigit():
        format_name = "Bruker OPUS"
    if format_name is not None or suffix in structured:
        kind = format_name or "structured"
        return SpectrumReadError(
            f"{path.name} appears to be {kind} data, not a delimited text table. v0.2.1 "
            "supports text tables only. Export the spectrum as CSV, TSV, TXT, DPT, ASC, "
            "DAT, or XY."
        )
    return SpectrumReadError(
        f"unsupported spectrum extension {path.suffix!r}; expected one of "
        f"{sorted(SUPPORTED_TEXT_EXTENSIONS)}. Structured or vendor-binary spectra must "
        "first be exported as a supported text table."
    )


def _utf16_nul_pattern(raw: bytes) -> str | None:
    if len(raw) < 4 or b"\x00" not in raw:
        return None
    even = raw[0::2]
    odd = raw[1::2]
    even_ratio = even.count(0) / max(len(even), 1)
    odd_ratio = odd.count(0) / max(len(odd), 1)
    if odd_ratio >= 0.35 and even_ratio <= 0.10:
        return "utf-16-le"
    if even_ratio >= 0.35 and odd_ratio <= 0.10:
        return "utf-16-be"
    return None


def _reject_binary(path: Path, raw: bytes, *, utf16_hint: str | None) -> None:
    signatures = {
        b"PK\x03\x04": "ZIP archive",
        b"PK\x05\x06": "ZIP archive",
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1": "OLE/Excel binary",
        b"%PDF-": "PDF document",
        b"GIF87a": "GIF image",
        b"GIF89a": "GIF image",
        b"\x89PNG\r\n\x1a\n": "PNG image",
        b"\xff\xd8\xff": "JPEG image",
        b"II*\x00": "TIFF image",
        b"MM\x00*": "TIFF image",
    }
    for signature, description in signatures.items():
        if raw.startswith(signature):
            raise SpectrumReadError(
                f"{path.name} appears to be {description}, not a text spectrum. Export it "
                "as CSV, TSV, TXT, DPT, ASC, DAT, or XY."
            )
    if (
        len(raw) >= 14
        and raw.startswith(b"BM")
        and raw[6:10] == b"\x00\x00\x00\x00"
        and int.from_bytes(raw[2:6], "little") >= 14
        and int.from_bytes(raw[10:14], "little") >= 14
    ):
        raise SpectrumReadError(
            f"{path.name} appears to be a BMP image, not a text spectrum. Export it as "
            "CSV, TSV, TXT, DPT, ASC, DAT, or XY."
        )
    pdf_position = raw.find(b"%PDF-")
    if 0 <= pdf_position < 1024:
        raise SpectrumReadError(
            f"{path.name} appears to be a PDF document, not a text spectrum. Export it "
            "as CSV, TSV, TXT, DPT, ASC, DAT, or XY."
        )
    sample = raw[:4096]
    if b"\x00" in sample and utf16_hint is None:
        raise SpectrumReadError(
            f"{path.name} contains a non-UTF-16 NUL-byte pattern and appears binary; "
            "v0.2.1 supports delimited text tables only"
        )
    if utf16_hint is None:
        controls = sum(
            byte < 32 and byte not in {0, 9, 10, 12, 13}
            for byte in sample
        )
        if sample and controls / len(sample) > 0.02:
            raise SpectrumReadError(
                f"{path.name} contains binary control bytes; v0.2.1 supports delimited "
                "text tables only"
            )


def _reject_decoded_controls(path: Path, text: str) -> None:
    sample = text[:4096]
    controls = sum(
        unicodedata.category(character) == "Cc"
        and character not in {"\t", "\n", "\f", "\r"}
        for character in sample
    )
    if sample and controls / len(sample) > 0.02:
        raise SpectrumReadError(
            f"{path.name} contains binary control characters after decoding; v0.2.1 "
            "supports delimited text tables only"
        )


def _decode_source(
    path: Path,
    raw: bytes,
    requested_encoding: str,
) -> tuple[str, str, str, tuple[str, ...]]:
    bom_encoding: str | None = None
    bom_evidence: str | None = None
    if raw.startswith(codecs.BOM_UTF8):
        bom_encoding, bom_evidence = "utf-8-sig", "UTF-8 BOM"
    elif raw.startswith(codecs.BOM_UTF32_LE) or raw.startswith(codecs.BOM_UTF32_BE):
        raise SpectrumReadError(
            f"{path.name} uses UTF-32, which is outside the supported text encodings"
        )
    elif raw.startswith(codecs.BOM_UTF16_LE):
        bom_encoding, bom_evidence = "utf-16-le", "UTF-16 LE BOM"
    elif raw.startswith(codecs.BOM_UTF16_BE):
        bom_encoding, bom_evidence = "utf-16-be", "UTF-16 BE BOM"
    nul_hint = _utf16_nul_pattern(raw)
    _reject_binary(path, raw, utf16_hint=bom_encoding or nul_hint)

    if requested_encoding != "auto":
        try:
            text = raw.decode(requested_encoding)
        except UnicodeError as exc:
            raise SpectrumReadError(
                f"cannot decode {path.name} as {requested_encoding}"
            ) from exc
        if text.startswith("\ufeff"):
            text = text[1:]
        _reject_decoded_controls(path, text)
        evidence = "explicit user selection"
        if bom_evidence is not None:
            evidence += f"; file has {bom_evidence}"
        return text, requested_encoding, evidence, ()

    if bom_encoding is not None:
        codec_name = "utf-16" if bom_encoding.startswith("utf-16") else bom_encoding
        try:
            text = raw.decode(codec_name)
        except UnicodeError as exc:  # pragma: no cover - malformed BOM edge
            raise SpectrumReadError(
                f"cannot decode {path.name} despite its {bom_evidence}"
            ) from exc
        _reject_decoded_controls(path, text)
        return text, bom_encoding, str(bom_evidence), ()

    if nul_hint is not None:
        try:
            text = raw.decode(nul_hint)
        except UnicodeError as exc:
            raise SpectrumReadError(
                f"{path.name} has a {nul_hint} NUL pattern but cannot be decoded"
            ) from exc
        if text.startswith("\ufeff"):
            text = text[1:]
        _reject_decoded_controls(path, text)
        return text, nul_hint, "UTF-16 alternating NUL-byte pattern", ()

    decoded: list[tuple[str, str]] = []
    failures: list[str] = []
    for candidate in ("utf-8-sig", "gb18030", "cp1252"):
        try:
            decoded.append((candidate, raw.decode(candidate)))
        except UnicodeError:
            failures.append(candidate)
    if not decoded:
        raise SpectrumReadError(
            f"cannot decode {path.name}; attempted UTF-8, GB18030, and CP1252"
        )
    selected_encoding, text = decoded[0]
    _reject_decoded_controls(path, text)
    differing = [name for name, candidate_text in decoded[1:] if candidate_text != text]
    warnings: tuple[str, ...] = ()
    if differing:
        warnings = (
            "Encoding auto-detection selected "
            f"{selected_encoding}; {', '.join(differing)} also decoded to different text. "
            "Choose the encoding explicitly if the header looks incorrect.",
        )
    evidence = "deterministic fallback order: UTF-8-SIG, GB18030, CP1252"
    if failures:
        evidence += f"; rejected {', '.join(failures)}"
    return text, selected_encoding, evidence, warnings


def _delimiter_character(delimiter: str) -> str | None:
    if delimiter in _DELIMITER_CHARACTERS:
        return _DELIMITER_CHARACTERS[delimiter]
    if len(delimiter) == 1:
        return delimiter
    raise ValueError(f"unsupported delimiter {delimiter!r}")


def _tokenize(line: str, delimiter: str) -> list[str]:
    character = _delimiter_character(delimiter)
    if character is None:
        return re.split(r"[ \t]+", line.strip())
    try:
        return next(csv.reader([line], delimiter=character, skipinitialspace=True))
    except csv.Error as exc:
        raise SpectrumReadError(f"cannot tokenize line using {delimiter} delimiter") from exc


def _trim_candidate_edges(
    tokens: Sequence[str],
    *,
    enabled: bool,
) -> tuple[tuple[str, ...], int, int]:
    stripped = tuple(token.strip() for token in tokens)
    if not enabled:
        return stripped, 0, 0
    left = 0
    while left < len(stripped) and stripped[left] == "":
        left += 1
    right = 0
    while right < len(stripped) - left and stripped[len(stripped) - right - 1] == "":
        right += 1
    end = len(stripped) - right if right else len(stripped)
    return stripped[left:end], left, right


def _parse_number(token: str, decimal_mark: str) -> float:
    stripped = token.strip()
    pattern = _DOT_NUMBER if decimal_mark == "dot" else _COMMA_NUMBER
    if pattern.fullmatch(stripped) is None:
        raise ValueError(stripped)
    normalized = stripped if decimal_mark == "dot" else stripped.replace(",", ".")
    value = float(normalized)
    if not np.isfinite(value):
        raise ValueError(stripped)
    return value


def _delimiter_evidence_count(lines: Sequence[_ContentLine], delimiter: str) -> int:
    character = _delimiter_character(delimiter)
    if character is None:
        return sum(bool(re.search(r"[ \t]+", line.raw.strip())) for line in lines)
    return sum(line.raw.count(character) for line in lines)


def _candidate_runs(
    lines: Sequence[_ContentLine],
    *,
    delimiter: str,
    decimal_mark: str,
    trim_empty_edge_columns: bool,
) -> tuple[_CandidateRun, ...]:
    numeric_rows: list[_NumericRow | None] = []
    for line in lines:
        tokens, leading, trailing = _trim_candidate_edges(
            _tokenize(line.raw, delimiter),
            enabled=trim_empty_edge_columns,
        )
        if len(tokens) < 2 or any(token == "" for token in tokens):
            numeric_rows.append(None)
            continue
        try:
            values = tuple(_parse_number(token, decimal_mark) for token in tokens)
        except ValueError:
            numeric_rows.append(None)
            continue
        numeric_rows.append(
            _NumericRow(
                line=line,
                tokens=tokens,
                values=values,
                leading_empty=leading,
                trailing_empty=trailing,
            )
        )

    runs: list[_CandidateRun] = []
    start: int | None = None
    current: list[_NumericRow] = []
    expected_columns: int | None = None
    for index, row in enumerate(numeric_rows):
        if row is None or (expected_columns is not None and len(row.values) != expected_columns):
            if current and start is not None:
                runs.append(_CandidateRun(start, index - 1, tuple(current)))
            start = None
            current = []
            expected_columns = None
        if row is None:
            continue
        if start is None:
            start = index
            expected_columns = len(row.values)
        current.append(row)
    if current and start is not None:
        runs.append(_CandidateRun(start, len(numeric_rows) - 1, tuple(current)))
    return tuple(runs)


def _evaluate_delimiter_candidate(
    lines: Sequence[_ContentLine],
    *,
    delimiter: str,
    requested_decimal_mark: str,
    trim_empty_edge_columns: bool,
) -> _DelimiterCandidate:
    decimal_marks = (
        (requested_decimal_mark,)
        if requested_decimal_mark != "auto"
        else (("dot",) if delimiter == "comma" else ("dot", "comma"))
    )
    variants: list[_DelimiterCandidate] = []
    for decimal_mark in decimal_marks:
        runs = _candidate_runs(
            lines,
            delimiter=delimiter,
            decimal_mark=decimal_mark,
            trim_empty_edge_columns=trim_empty_edge_columns,
        )
        variants.append(
            _DelimiterCandidate(
                delimiter=delimiter,
                decimal_mark=decimal_mark,
                runs=runs,
                evidence_count=_delimiter_evidence_count(lines, delimiter),
            )
        )
    variants.sort(
        key=lambda item: (
            item.best_row_count,
            item.evidence_count,
            item.decimal_mark == "dot",
        ),
        reverse=True,
    )
    return variants[0]


def _candidate_signature(candidate: _DelimiterCandidate) -> tuple[tuple[str, ...], ...]:
    best = max(candidate.runs, key=lambda run: run.row_count, default=None)
    return () if best is None else tuple(row.tokens for row in best.rows)


def _choose_delimiter(
    path: Path,
    lines: Sequence[_ContentLine],
    options: TextImportOptions,
) -> tuple[_DelimiterCandidate, dict[str, Any], tuple[str, ...]]:
    delimiters = (
        (options.delimiter,)
        if options.delimiter != "auto"
        else ("comma", "tab", "semicolon", "whitespace")
    )
    candidates = [
        _evaluate_delimiter_candidate(
            lines,
            delimiter=delimiter,
            requested_decimal_mark=options.decimal_mark,
            trim_empty_edge_columns=options.trim_empty_edge_columns,
        )
        for delimiter in delimiters
    ]
    best_strength = max(
        (
            (candidate.best_row_count, candidate.numeric_row_count)
            for candidate in candidates
        ),
        default=(0, 0),
    )
    viable = [
        candidate
        for candidate in candidates
        if (candidate.best_row_count, candidate.numeric_row_count) == best_strength
        and candidate.best_columns >= 2
        and candidate.numeric_row_count >= 1
    ]
    if not viable:
        requested = options.delimiter
        raise SpectrumReadError(
            f"{path.name}: no rectangular numeric block with at least two rows and two "
            f"columns was found (delimiter={requested}, decimal={options.decimal_mark})"
        )

    signatures = {_candidate_signature(candidate) for candidate in viable}
    if len(signatures) > 1:
        names = ", ".join(candidate.delimiter for candidate in viable)
        raise SpectrumReadError(
            f"{path.name}: delimiter detection is ambiguous between {names}. Choose one "
            "explicitly in Advanced text import options."
        )

    hint = _EXTENSION_DELIMITER_HINTS.get(path.suffix.lower())
    priority = {"comma": 4, "tab": 3, "semicolon": 2, "whitespace": 1}
    viable.sort(
        key=lambda item: (
            item.delimiter == hint,
            item.evidence_count,
            priority.get(item.delimiter, 0),
        ),
        reverse=True,
    )
    selected = viable[0]
    if selected.delimiter == "comma" and options.decimal_mark == "comma":
        raise SpectrumReadError(
            f"{path.name}: comma delimiter and comma decimal mark are ambiguous; choose a "
            "non-comma delimiter or dot decimal mark explicitly"
        )
    warnings: list[str] = []
    if hint is not None and selected.delimiter != hint:
        warnings.append(
            f"Extension {path.suffix.lower()} suggests {hint}, but content was parsed as "
            f"{selected.delimiter}."
        )
    evidence = {
        "mode": options.delimiter,
        "extension_hint": hint,
        "extension_hint_agreed": hint is None or selected.delimiter == hint,
        "candidates": [
            {
                "delimiter": candidate.delimiter,
                "best_decimal_mark": candidate.decimal_mark,
                "numeric_rows": candidate.best_row_count,
                "total_numeric_rows": candidate.numeric_row_count,
                "columns": candidate.best_columns,
                "separator_evidence": candidate.evidence_count,
            }
            for candidate in candidates
        ],
    }
    return selected, evidence, tuple(warnings)


def _is_header_candidate(
    tokens: Sequence[str],
    *,
    columns: int,
    decimal_mark: str,
    require_text_majority: bool = True,
) -> bool:
    if len(tokens) != columns or any(token == "" for token in tokens):
        return False
    if _token_is_numeric_like(tokens[0]):
        return False
    if any(token.strip().casefold() in _NONFINITE_TOKENS for token in tokens):
        return False
    numeric = 0
    for token in tokens:
        try:
            _parse_number(token, decimal_mark)
        except ValueError:
            continue
        numeric += 1
    if require_text_majority:
        return numeric * 2 < len(tokens)
    return numeric < len(tokens)


def _token_is_numeric_like(token: str) -> bool:
    stripped = token.strip()
    lowered = stripped.casefold()
    if lowered in _NONFINITE_TOKENS:
        return True
    for decimal_mark in ("dot", "comma"):
        try:
            _parse_number(stripped, decimal_mark)
        except ValueError:
            continue
        return True
    return re.match(r"^[+-]?(?:\d|[.,]\d)", stripped) is not None


def _is_partial_numeric_row(
    line: _ContentLine,
    *,
    delimiter: str,
    decimal_mark: str,
    columns: int,
    trim_empty_edge_columns: bool,
) -> bool:
    tokens, _, _ = _trim_candidate_edges(
        _tokenize(line.raw, delimiter),
        enabled=trim_empty_edge_columns,
    )
    if len(tokens) != columns:
        return False
    for token in tokens:
        if token.strip().casefold() in _NONFINITE_TOKENS:
            return True
        try:
            _parse_number(token, decimal_mark)
        except ValueError:
            continue
        return True
    return False


def _has_strong_axis_label(token: str) -> bool:
    """Recognize an axis label without using it to infer the physical data unit."""

    normalized = unicodedata.normalize("NFKC", token).casefold().strip()
    compact = re.sub(r"[\s_\-]", "", normalized)
    return (
        compact.startswith("wavenumber")
        or compact.startswith("waveno")
        or compact.startswith("wavenum")
        or compact.startswith("波数")
        or compact == "wn"
    )


def _safe_token(token: str, *, limit: int = 80) -> str:
    normalized = token.replace("\n", "\\n").replace("\r", "\\r")
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def _strict_row_error(
    path: Path,
    *,
    line_number: int,
    column_number: int,
    token: str,
    header: tuple[str, ...] | None,
    delimiter: str,
    decimal_mark: str,
    encoding: str,
    reason: str,
) -> SpectrumReadError:
    label = None
    if header is not None and 0 < column_number <= len(header):
        label = header[column_number - 1]
    label_text = "" if label is None else f" ({label!r})"
    return SpectrumReadError(
        f"{path.name}: line {line_number}, column {column_number}{label_text}: "
        f"token {_safe_token(token)!r} {reason} after {delimiter}/{decimal_mark} parsing "
        f"with {encoding} encoding"
    )


def _looks_numeric_like_in_any_profile(
    line: _ContentLine,
    *,
    trim_empty_edge_columns: bool,
) -> bool:
    for delimiter in ("comma", "tab", "semicolon", "whitespace"):
        tokens, leading_empty, trailing_empty = _trim_candidate_edges(
            _tokenize(line.raw, delimiter),
            enabled=trim_empty_edge_columns,
        )
        if not tokens:
            continue
        if _token_is_numeric_like(tokens[0]):
            return True
        if any(token.strip().casefold() in _NONFINITE_TOKENS for token in tokens):
            return True
        for decimal_mark in ("dot", "comma"):
            if delimiter == "comma" and decimal_mark == "comma":
                continue
            parsed: list[bool] = []
            for token in tokens:
                try:
                    _parse_number(token, decimal_mark)
                except ValueError:
                    parsed.append(False)
                else:
                    parsed.append(True)
            if all(parsed) and len(parsed) >= 2:
                return True
            if (leading_empty or trailing_empty) and any(parsed):
                return True
    return False


def _raise_preblock_issue(
    path: Path,
    *,
    line: _ContentLine,
    columns: int,
    candidate: _DelimiterCandidate,
    options: TextImportOptions,
    encoding: str,
    header: tuple[str, ...] | None,
) -> None:
    tokens, _, _ = _trim_candidate_edges(
        _tokenize(line.raw, candidate.delimiter),
        enabled=options.trim_empty_edge_columns,
    )
    if len(tokens) != columns:
        raise _strict_row_error(
            path,
            line_number=line.number,
            column_number=1,
            token=line.raw.strip(),
            header=header,
            delimiter=candidate.delimiter,
            decimal_mark=candidate.decimal_mark,
            encoding=encoding,
            reason=f"has {len(tokens)} columns; expected {columns} before the numeric block",
        )
    for column_number, token in enumerate(tokens, start=1):
        try:
            _parse_number(token, candidate.decimal_mark)
        except ValueError as exc:
            lowered = token.strip().lower()
            if lowered in _NONFINITE_TOKENS:
                reason = "is non-finite"
            else:
                reason = "is non-numeric or uses a conflicting decimal mark/delimiter"
            raise _strict_row_error(
                path,
                line_number=line.number,
                column_number=column_number,
                token=token,
                header=header,
                delimiter=candidate.delimiter,
                decimal_mark=candidate.decimal_mark,
                encoding=encoding,
                reason=reason,
            ) from exc
    raise _strict_row_error(
        path,
        line_number=line.number,
        column_number=1,
        token=tokens[0],
        header=header,
        delimiter=candidate.delimiter,
        decimal_mark=candidate.decimal_mark,
        encoding=encoding,
        reason="belongs to an earlier numeric block and cannot be skipped as preamble",
    )


def _select_numeric_run(
    path: Path,
    lines: Sequence[_ContentLine],
    candidate: _DelimiterCandidate,
    options: TextImportOptions,
    *,
    encoding: str,
) -> tuple[_CandidateRun, tuple[str, ...] | None, tuple[int, ...], int]:
    if not candidate.runs:
        raise SpectrumReadError(f"{path.name} contains no numeric data rows")
    strong_headers: list[tuple[int, tuple[str, ...], int, int]] = []
    for index, line in enumerate(lines):
        tokens, left, right = _trim_candidate_edges(
            _tokenize(line.raw, candidate.delimiter),
            enabled=options.trim_empty_edge_columns,
        )
        if tokens and _has_strong_axis_label(tokens[0]):
            strong_headers.append((index, tuple(tokens), left, right))
    if len(strong_headers) > 1:
        locations = ", ".join(
            str(lines[index].number) for index, _, _, _ in strong_headers
        )
        raise SpectrumReadError(
            f"{path.name}: multiple spectrum-axis headers were found at lines "
            f"{locations}; set skip_rows/header_mode explicitly"
        )

    anchored_header: tuple[str, ...] | None = None
    anchored_header_index: int | None = None
    anchored_header_edges: tuple[int, int] | None = None
    anchored_run: _CandidateRun | None = None
    if strong_headers:
        (
            anchored_header_index,
            anchored_header,
            anchored_left,
            anchored_right,
        ) = strong_headers[0]
        if options.header_mode == "absent":
            raise SpectrumReadError(
                f"{path.name}: line {lines[anchored_header_index].number} looks like a "
                "spectrum header, but header_mode='absent' requires numeric data"
            )
        anchored_header_edges = (anchored_left, anchored_right)
        if options.header_mode == "auto":
            prefix_runs = [
                item
                for item in candidate.runs
                if item.end_index < anchored_header_index
            ]
            prefix_numeric_indices = {
                index
                for item in prefix_runs
                for index in range(item.start_index, item.end_index + 1)
            }
            if len(prefix_numeric_indices) > 2:
                first_prefix_index = min(prefix_numeric_indices)
                _raise_preblock_issue(
                    path,
                    line=lines[first_prefix_index],
                    columns=candidate.runs[0].columns,
                    candidate=candidate,
                    options=options,
                    encoding=encoding,
                    header=None,
                )
            malformed_prefix = next(
                (
                    line
                    for index, line in enumerate(lines[:anchored_header_index])
                    if index not in prefix_numeric_indices
                    and _looks_numeric_like_in_any_profile(
                        line,
                        trim_empty_edge_columns=options.trim_empty_edge_columns,
                    )
                ),
                None,
            )
            if malformed_prefix is not None:
                _raise_preblock_issue(
                    path,
                    line=malformed_prefix,
                    columns=len(anchored_header),
                    candidate=candidate,
                    options=options,
                    encoding=encoding,
                    header=None,
                )
        anchored_run = next(
            (
                item
                for item in candidate.runs
                if item.start_index == anchored_header_index + 1
            ),
            None,
        )
        if anchored_run is not None and len(anchored_header) != anchored_run.columns:
            raise _strict_row_error(
                path,
                line_number=lines[anchored_header_index].number,
                column_number=1,
                token=lines[anchored_header_index].raw.strip(),
                header=None,
                delimiter=candidate.delimiter,
                decimal_mark=candidate.decimal_mark,
                encoding=encoding,
                reason=(
                    f"has {len(anchored_header)} header columns; the following numeric "
                    f"rows have {anchored_run.columns}"
                ),
            )
        if anchored_run is None:
            next_index = anchored_header_index + 1
            if next_index >= len(lines):
                raise SpectrumReadError(
                    f"{path.name}: spectrum header on line "
                    f"{lines[anchored_header_index].number} has no following numeric data"
                )
            _raise_preblock_issue(
                path,
                line=lines[next_index],
                columns=len(anchored_header),
                candidate=candidate,
                options=options,
                encoding=encoding,
                header=anchored_header,
            )

    if anchored_run is not None:
        run = anchored_run
    else:
        first_run = candidate.runs[0]
        first_run_has_header = False
        if first_run.start_index > 0 and options.header_mode != "absent":
            first_header_tokens, _, _ = _trim_candidate_edges(
                _tokenize(lines[first_run.start_index - 1].raw, candidate.delimiter),
                enabled=options.trim_empty_edge_columns,
            )
            first_run_has_header = _is_header_candidate(
                first_header_tokens,
                columns=first_run.columns,
                decimal_mark=candidate.decimal_mark,
                require_text_majority=options.header_mode != "present",
            )
        if first_run_has_header or first_run.start_index == 0:
            run = first_run
        else:
            max_rows = max(item.row_count for item in candidate.runs)
            best = [item for item in candidate.runs if item.row_count == max_rows]
            if len(best) > 1:
                spans = ", ".join(
                    f"{item.rows[0].line.number}-{item.rows[-1].line.number}"
                    for item in best
                )
                raise SpectrumReadError(
                    f"{path.name}: multiple equally long numeric blocks were found at "
                    f"lines {spans}; set skip_rows/header_mode explicitly"
                )
            run = best[0]
    before = list(lines[: run.start_index])
    header: tuple[str, ...] | None = None
    header_line_index: int | None = None
    possible_header: tuple[str, ...] | None = None
    header_edge_counts: tuple[int, int] | None = None
    if anchored_run is not None:
        possible_header = anchored_header
        header_line_index = anchored_header_index
        header_edge_counts = anchored_header_edges
    else:
        header_candidates: list[tuple[int, tuple[str, ...], int, int]] = []
        for index, earlier_line in enumerate(before):
            raw_tokens, header_left, header_right = _trim_candidate_edges(
                _tokenize(earlier_line.raw, candidate.delimiter),
                enabled=options.trim_empty_edge_columns,
            )
            if _is_header_candidate(
                raw_tokens,
                columns=run.columns,
                decimal_mark=candidate.decimal_mark,
                require_text_majority=options.header_mode != "present",
            ):
                header_candidates.append(
                    (index, tuple(raw_tokens), header_left, header_right)
                )
        if header_candidates and header_candidates[-1][0] == len(before) - 1:
            if len(header_candidates) > 1:
                raise SpectrumReadError(
                    f"{path.name}: multiple same-width text rows before the numeric "
                    "block are ambiguous; set skip_rows/header_mode explicitly"
                )
            header_line_index, possible_header, header_left, header_right = (
                header_candidates[-1]
            )
            header_edge_counts = (header_left, header_right)
        elif header_candidates:
            earlier_index, earlier_header, _, _ = header_candidates[-1]
            issue_index = earlier_index + 1
            if issue_index < len(before):
                _raise_preblock_issue(
                    path,
                    line=before[issue_index],
                    columns=run.columns,
                    candidate=candidate,
                    options=options,
                    encoding=encoding,
                    header=earlier_header,
                )

        partial_numeric = next(
            (
                earlier_line
                for index, earlier_line in enumerate(before)
                if index != header_line_index
                and _is_partial_numeric_row(
                    earlier_line,
                    delimiter=candidate.delimiter,
                    decimal_mark=candidate.decimal_mark,
                    columns=run.columns,
                    trim_empty_edge_columns=options.trim_empty_edge_columns,
                )
            ),
            None,
        )
        if partial_numeric is not None and options.header_mode == "auto":
            _raise_preblock_issue(
                path,
                line=partial_numeric,
                columns=run.columns,
                candidate=candidate,
                options=options,
                encoding=encoding,
                header=None,
            )

        numeric_like = next(
            (
                earlier_line
                for index, earlier_line in enumerate(before)
                if index != header_line_index
                and _looks_numeric_like_in_any_profile(
                    earlier_line,
                    trim_empty_edge_columns=options.trim_empty_edge_columns,
                )
            ),
            None,
        )
        if numeric_like is not None and options.header_mode == "auto":
            _raise_preblock_issue(
                path,
                line=numeric_like,
                columns=run.columns,
                candidate=candidate,
                options=options,
                encoding=encoding,
                header=None,
            )
    if options.header_mode == "absent" and possible_header is not None:
        raise SpectrumReadError(
            f"{path.name}: line {before[-1].number} looks like a header, but "
            "header_mode='absent' requires numeric data"
        )
    if possible_header is not None and options.header_mode != "absent":
        header = possible_header
        header_line_index = len(before) - 1
    if options.header_mode == "present" and header is None:
        raise SpectrumReadError(
            f"{path.name}: header_mode='present' requires a header immediately before the "
            f"numeric block beginning on line {run.rows[0].line.number}"
        )
    preamble = [
        line
        for index, line in enumerate(before)
        if index != header_line_index
    ]
    if preamble and not options.allow_preamble:
        raise SpectrumReadError(
            f"{path.name}: leading preamble is disabled; first unexpected line is "
            f"{preamble[0].number}"
        )
    if run.end_index != len(lines) - 1:
        offending = lines[run.end_index + 1]
        tokens, _, _ = _trim_candidate_edges(
            _tokenize(offending.raw, candidate.delimiter),
            enabled=options.trim_empty_edge_columns,
        )
        if len(tokens) != run.columns:
            raise _strict_row_error(
                path,
                line_number=offending.number,
                column_number=1,
                token=offending.raw.strip(),
                header=header,
                delimiter=candidate.delimiter,
                decimal_mark=candidate.decimal_mark,
                encoding=encoding,
                reason=f"has {len(tokens)} columns; expected {run.columns}",
            )
        for column_number, token in enumerate(tokens, start=1):
            if token == "":
                raise _strict_row_error(
                    path,
                    line_number=offending.number,
                    column_number=column_number,
                    token="<empty>",
                    header=header,
                    delimiter=candidate.delimiter,
                    decimal_mark=candidate.decimal_mark,
                    encoding=encoding,
                    reason="is an internal missing value",
                )
            try:
                _parse_number(token, candidate.decimal_mark)
            except ValueError as exc:
                reason = (
                    "is non-finite"
                    if token.strip().casefold() in _NONFINITE_TOKENS
                    else "is non-numeric"
                )
                raise _strict_row_error(
                    path,
                    line_number=offending.number,
                    column_number=column_number,
                    token=token,
                    header=header,
                    delimiter=candidate.delimiter,
                    decimal_mark=candidate.decimal_mark,
                    encoding=encoding,
                    reason=reason,
                ) from exc
        raise SpectrumReadError(
            f"{path.name}: line {offending.number} starts a second numeric block; "
            "set skip_rows/header_mode explicitly"
        )

    if (
        preamble
        and run.row_count < 3
        and options.header_mode == "auto"
        and options.skip_rows == 0
    ):
        raise SpectrumReadError(
            f"{path.name}: an auto-detected numeric block after a preamble must contain at "
            "least three rows; set skip_rows/header_mode explicitly for a 1-2 point file"
        )

    left_counts = {row.leading_empty for row in run.rows}
    right_counts = {row.trailing_empty for row in run.rows}
    if len(left_counts) != 1 or len(right_counts) != 1:
        expected_left = max(
            left_counts,
            key=lambda count: (
                sum(numeric_row.leading_empty == count for numeric_row in run.rows),
                -count,
            ),
        )
        expected_right = max(
            right_counts,
            key=lambda count: (
                sum(numeric_row.trailing_empty == count for numeric_row in run.rows),
                -count,
            ),
        )
        offending_edge_row = next(
            numeric_row
            for numeric_row in run.rows
            if numeric_row.leading_empty != expected_left
            or numeric_row.trailing_empty != expected_right
        )
        raise _strict_row_error(
            path,
            line_number=offending_edge_row.line.number,
            column_number=1,
            token="<inconsistent edge empties>",
            header=header,
            delimiter=candidate.delimiter,
            decimal_mark=candidate.decimal_mark,
            encoding=encoding,
            reason="does not form an all-empty edge column across every data row",
        )
    expected_left = next(iter(left_counts))
    expected_right = next(iter(right_counts))
    if header is not None and header_edge_counts != (expected_left, expected_right):
        raise _strict_row_error(
            path,
            line_number=before[-1].number,
            column_number=1,
            token="<inconsistent header edge empties>",
            header=header,
            delimiter=candidate.delimiter,
            decimal_mark=candidate.decimal_mark,
            encoding=encoding,
            reason="does not form an all-empty edge column across header and data rows",
        )
    trimmed = expected_left + expected_right
    return run, header, tuple(line.number for line in preamble), trimmed


def _reject_headerless_comma_ambiguity(
    path: Path,
    run: _CandidateRun,
    header: tuple[str, ...] | None,
    candidate: _DelimiterCandidate,
    options: TextImportOptions,
    *,
    encoding: str,
) -> None:
    if not (
        header is None
        and options.delimiter == "auto"
        and options.decimal_mark == "auto"
        and candidate.delimiter == "comma"
        and candidate.decimal_mark == "dot"
        and run.columns >= 3
    ):
        return
    adjacent_decimal_comma = any(
        _COMMA_NUMBER.fullmatch(f"{left.strip()},{right.strip()}") is not None
        for row in run.rows
        for left, right in zip(row.tokens, row.tokens[1:], strict=False)
    )
    thousands_pattern = any(
        (
            re.fullmatch(r"[+-]?\d{1,3}", left.strip()) is not None
            and re.fullmatch(
                r"\d{3}(?:\.\d+)?(?:[eE][+-]?\d+)?",
                right.strip(),
            )
            is not None
        )
        or (
            re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+", left.strip())
            is not None
            and re.fullmatch(r"\d+(?:[eE][+-]?\d+)?", right.strip()) is not None
        )
        for row in run.rows
        for left, right in zip(row.tokens, row.tokens[1:], strict=False)
    )
    if not adjacent_decimal_comma and not thousands_pattern:
        return
    raise _strict_row_error(
        path,
        line_number=run.rows[0].line.number,
        column_number=1,
        token=run.rows[0].line.raw.strip(),
        header=None,
        delimiter=candidate.delimiter,
        decimal_mark=candidate.decimal_mark,
        encoding=encoding,
        reason=(
            "is ambiguous between a comma-delimited wide table, decimal-comma values, "
            "and/or thousands grouping; choose comma delimiter with dot decimal mark "
            "explicitly only if the wide-table interpretation is intended"
        ),
    )


def _resolve_import_options(
    *,
    import_options: TextImportOptions | None,
    delimiter: str | None,
    encoding: str | None,
    decimal_mark: str,
    header_mode: str,
    skip_rows: int,
    allow_preamble: bool,
    trim_empty_edge_columns: bool,
    comment_prefixes: Sequence[str],
) -> TextImportOptions:
    if import_options is not None:
        if not isinstance(import_options, TextImportOptions):
            raise TypeError("import_options must be a TextImportOptions instance")
        conflicting = (
            delimiter != "auto"
            or encoding is not None
            or decimal_mark != "auto"
            or header_mode != "auto"
            or skip_rows != 0
            or allow_preamble is not True
            or trim_empty_edge_columns is not True
            or tuple(comment_prefixes) != ("#", "//", "%")
        )
        if conflicting:
            raise ValueError(
                "pass either import_options or individual text parser arguments, not both"
            )
        return import_options
    return TextImportOptions(
        delimiter=_normalize_delimiter(delimiter),
        decimal_mark=decimal_mark,
        encoding=_normalize_encoding(encoding),
        header_mode=header_mode,
        skip_rows=skip_rows,
        allow_preamble=allow_preamble,
        trim_empty_edge_columns=trim_empty_edge_columns,
        comment_prefixes=tuple(comment_prefixes),
    )


def _parse_text_table(path: Path, options: TextImportOptions) -> _ParsedTextTable:
    if path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
        raise _unsupported_extension_error(path)
    raw = path.read_bytes()
    if not raw:
        raise SpectrumReadError(f"{path.name} is empty")
    source_sha256 = hashlib.sha256(raw).hexdigest()
    text, encoding, encoding_evidence, encoding_warnings = _decode_source(
        path,
        raw,
        options.encoding,
    )
    physical_lines = text.splitlines()
    if options.skip_rows > len(physical_lines):
        raise SpectrumReadError(
            f"{path.name}: skip_rows={options.skip_rows} exceeds the "
            f"{len(physical_lines)} physical lines in the file"
        )
    content: list[_ContentLine] = []
    comment_lines: list[int] = []
    for line_number, raw_line in enumerate(physical_lines, start=1):
        if line_number <= options.skip_rows:
            continue
        stripped = raw_line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(prefix) for prefix in options.comment_prefixes):
            comment_lines.append(line_number)
            continue
        content.append(_ContentLine(line_number, raw_line))
    if not content:
        raise SpectrumReadError(f"{path.name} contains no data rows")

    candidate, delimiter_evidence, delimiter_warnings = _choose_delimiter(
        path,
        content,
        options,
    )
    # Re-evaluate the selected profile across the complete file; the detection sample is
    # capped at 30 lines, while strict validation must inspect every row.
    candidate = _evaluate_delimiter_candidate(
        content,
        delimiter=candidate.delimiter,
        requested_decimal_mark=candidate.decimal_mark,
        trim_empty_edge_columns=options.trim_empty_edge_columns,
    )
    run, header, preamble_lines, trimmed_columns = _select_numeric_run(
        path,
        content,
        candidate,
        options,
        encoding=encoding,
    )
    _reject_headerless_comma_ambiguity(
        path,
        run,
        header,
        candidate,
        options,
        encoding=encoding,
    )

    rows: list[list[float]] = []
    expected_columns = run.columns
    for row in run.rows:
        if len(row.tokens) != expected_columns:
            raise SpectrumReadError(
                f"{path.name}: line {row.line.number} has {len(row.tokens)} columns; "
                f"expected {expected_columns} after {candidate.delimiter}/"
                f"{candidate.decimal_mark} parsing with {encoding} encoding"
            )
        parsed: list[float] = []
        for column_number, token in enumerate(row.tokens, start=1):
            if token == "":
                raise _strict_row_error(
                    path,
                    line_number=row.line.number,
                    column_number=column_number,
                    token="<empty>",
                    header=header,
                    delimiter=candidate.delimiter,
                    decimal_mark=candidate.decimal_mark,
                    encoding=encoding,
                    reason="is an internal missing value",
                )
            try:
                parsed.append(_parse_number(token, candidate.decimal_mark))
            except ValueError as exc:  # pragma: no cover - candidate rows are prevalidated
                raise _strict_row_error(
                    path,
                    line_number=row.line.number,
                    column_number=column_number,
                    token=token,
                    header=header,
                    delimiter=candidate.delimiter,
                    decimal_mark=candidate.decimal_mark,
                    encoding=encoding,
                    reason="is not numeric",
                ) from exc
        rows.append(parsed)
    if expected_columns < 2:
        raise SpectrumReadError(
            f"{path.name} must contain wavenumber plus at least one intensity column"
        )

    dot_tokens = 0
    comma_tokens = 0
    integer_tokens = 0
    for row in run.rows:
        for token in row.tokens:
            stripped = token.strip()
            if _INTEGER_NUMBER.fullmatch(stripped):
                integer_tokens += 1
            elif "." in stripped:
                dot_tokens += 1
            elif "," in stripped:
                comma_tokens += 1
    decimal_evidence = {
        "mode": options.decimal_mark,
        "dot_decimal_tokens": dot_tokens,
        "comma_decimal_tokens": comma_tokens,
        "integer_tokens": integer_tokens,
        "no_decimal_evidence": dot_tokens == 0 and comma_tokens == 0,
    }
    warnings = tuple(dict.fromkeys((*encoding_warnings, *delimiter_warnings)))
    probe = ImportProbe(
        source_name=path.name,
        extension=path.suffix.lower(),
        source_sha256=source_sha256,
        size_bytes=len(raw),
        selected_encoding=encoding,
        encoding_evidence=encoding_evidence,
        selected_delimiter=candidate.delimiter,
        delimiter_evidence=delimiter_evidence,
        selected_decimal_mark=candidate.decimal_mark,
        decimal_evidence=decimal_evidence,
        header_present=header is not None,
        header=header,
        skipped_explicit_rows=options.skip_rows,
        skipped_preamble_lines=preamble_lines,
        skipped_comment_lines=tuple(comment_lines),
        trimmed_empty_edge_columns=trimmed_columns,
        numeric_block_start_line=run.rows[0].line.number,
        numeric_block_end_line=run.rows[-1].line.number,
        data_rows=run.row_count,
        columns=expected_columns,
        layout="two_column" if expected_columns == 2 else "wide_table",
        warnings=warnings,
    )
    return _ParsedTextTable(
        table=np.asarray(rows, dtype=np.float64),
        header=header,
        probe=probe,
    )


def probe_spectrum_file(
    path: PathLike,
    *,
    options: TextImportOptions | None = None,
) -> ImportProbe:
    """Diagnose one text spectrum with the exact parser used by formal loading."""

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"spectrum file does not exist: {file_path}")
    return _parse_text_table(file_path, options or TextImportOptions()).probe


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
    import_options: TextImportOptions | None = None,
    decimal_mark: str = "auto",
    header_mode: str = "auto",
    skip_rows: int = 0,
    allow_preamble: bool = True,
    trim_empty_edge_columns: bool = True,
    comment_prefixes: Sequence[str] = ("#", "//", "%"),
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
    options = _resolve_import_options(
        import_options=import_options,
        delimiter=delimiter,
        encoding=encoding,
        decimal_mark=decimal_mark,
        header_mode=header_mode,
        skip_rows=skip_rows,
        allow_preamble=allow_preamble,
        trim_empty_edge_columns=trim_empty_edge_columns,
        comment_prefixes=comment_prefixes,
    )
    parsed = _parse_text_table(file_path, options)
    table = parsed.table
    header = parsed.header
    probe = parsed.probe
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
    delimiter_character = _delimiter_character(probe.selected_delimiter)
    legacy_delimiter = (
        "whitespace" if delimiter_character is None else repr(delimiter_character)
    )
    file_metadata.update(
        {
            "source_path": str(file_path.resolve()),
            "source_sha256": probe.source_sha256,
            "source_format": file_path.suffix.lower().lstrip("."),
            "encoding": probe.selected_encoding,
            "encoding_evidence": probe.encoding_evidence,
            "delimiter": legacy_delimiter,
            "delimiter_name": probe.selected_delimiter,
            "delimiter_detection_candidates": probe.delimiter_evidence["candidates"],
            "decimal_mark": probe.selected_decimal_mark,
            "decimal_detection_evidence": dict(probe.decimal_evidence),
            "header_mode": options.header_mode,
            "header": header,
            "skip_rows": options.skip_rows,
            "skipped_preamble_lines": probe.skipped_preamble_lines,
            "skipped_comment_lines": probe.skipped_comment_lines,
            "trimmed_empty_edge_columns": probe.trimmed_empty_edge_columns,
            "numeric_block_start_line": probe.numeric_block_start_line,
            "numeric_block_end_line": probe.numeric_block_end_line,
            "input_layout": probe.layout,
            "import_parser": "ftir_text_table",
            "import_parser_version": "2.1",
            "import_warnings": probe.warnings,
            "import_probe": probe.to_dict(),
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
    import_options: TextImportOptions | None = None,
    decimal_mark: str = "auto",
    header_mode: str = "auto",
    skip_rows: int = 0,
    allow_preamble: bool = True,
    trim_empty_edge_columns: bool = True,
    comment_prefixes: Sequence[str] = ("#", "//", "%"),
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
            import_options=import_options,
            decimal_mark=decimal_mark,
            header_mode=header_mode,
            skip_rows=skip_rows,
            allow_preamble=allow_preamble,
            trim_empty_edge_columns=trim_empty_edge_columns,
            comment_prefixes=comment_prefixes,
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
        "import_probe_by_file": {
            path.name: data.metadata["import_probe"]
            for path, data in zip(selected, loaded, strict=True)
        },
        "encoding_by_file": {
            path.name: data.metadata["encoding"]
            for path, data in zip(selected, loaded, strict=True)
        },
        "delimiter_by_file": {
            path.name: data.metadata["delimiter_name"]
            for path, data in zip(selected, loaded, strict=True)
        },
        "decimal_mark_by_file": {
            path.name: data.metadata["decimal_mark"]
            for path, data in zip(selected, loaded, strict=True)
        },
        "preamble_lines_by_file": {
            path.name: data.metadata["skipped_preamble_lines"]
            for path, data in zip(selected, loaded, strict=True)
        },
        "trimmed_empty_columns_by_file": {
            path.name: data.metadata["trimmed_empty_edge_columns"]
            for path, data in zip(selected, loaded, strict=True)
        },
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
    "ImportProbe",
    "SpectrumReadError",
    "TextImportOptions",
    "load_spectrum_directory",
    "load_spectrum_files",
    "parse_perturbation_label",
    "probe_spectrum_file",
    "read_spectrum",
    "read_spectrum_file",
    "read_spectrum_series",
]
