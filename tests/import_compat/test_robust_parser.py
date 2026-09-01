from __future__ import annotations

import codecs
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from ftir_baseline.io import (
    ImportProbe,
    SpectrumReadError,
    TextImportOptions,
    load_spectrum_files,
    probe_spectrum_file,
    read_spectrum_file,
)
from ftir_baseline.validation import SpectrumValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "import_compat"


def test_text_import_options_defaults_are_stable_and_json_serializable() -> None:
    options = TextImportOptions()

    assert asdict(options) == {
        "delimiter": "auto",
        "decimal_mark": "auto",
        "encoding": "auto",
        "header_mode": "auto",
        "skip_rows": 0,
        "allow_preamble": True,
        "trim_empty_edge_columns": True,
        "comment_prefixes": ("#", "//", "%"),
    }
    assert json.loads(json.dumps(asdict(options)))["comment_prefixes"] == ["#", "//", "%"]


@pytest.mark.parametrize(
    "kwargs",
    (
        {"skip_rows": -1},
        {"delimiter": "comma", "decimal_mark": "comma"},
        {"comment_prefixes": ("#", "")},
    ),
)
def test_text_import_options_reject_invalid_combinations(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TextImportOptions(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("fixture_name", "delimiter", "delimiter_name"),
    (
        ("single.xy", ",", "comma"),
        ("single.tsv", "\t", "tab"),
        ("single.dat", ";", "semicolon"),
        ("single.asc", None, "whitespace"),
    ),
)
def test_legacy_delimiter_and_encoding_keywords_keep_their_meaning(
    fixture_name: str,
    delimiter: str | None,
    delimiter_name: str,
) -> None:
    data = read_spectrum_file(
        FIXTURES / fixture_name,
        delimiter=delimiter,
        encoding="utf-8",
    )

    assert data.metadata["delimiter_name"] == delimiter_name
    assert data.metadata["encoding"] == "utf-8"


@pytest.mark.parametrize(
    ("fixture_name", "options", "delimiter_name"),
    (
        ("decimal_comma_semicolon.csv", TextImportOptions(), "semicolon"),
        ("decimal_comma_tab.tsv", TextImportOptions(), "tab"),
        (
            "decimal_comma_whitespace.asc",
            TextImportOptions(delimiter="whitespace", decimal_mark="comma"),
            "whitespace",
        ),
    ),
)
def test_decimal_comma_is_supported_only_with_noncomma_delimiters(
    fixture_name: str,
    options: TextImportOptions,
    delimiter_name: str,
) -> None:
    data = read_spectrum_file(FIXTURES / fixture_name, import_options=options)
    probe = probe_spectrum_file(FIXTURES / fixture_name, options=options)

    np.testing.assert_allclose(data.wavenumber, [1002.5, 1001.5, 1000.5])
    expected_first_spectrum = (
        [0.00125, 0.0015, 0.00175]
        if data.n_spectra == 2
        else [0.125, 0.15, 0.175]
    )
    np.testing.assert_allclose(data.spectra[0], expected_first_spectrum)
    assert probe.selected_delimiter == delimiter_name
    assert probe.selected_decimal_mark == "comma"
    assert data.metadata["delimiter_name"] == delimiter_name
    assert data.metadata["decimal_mark"] == "comma"


def test_auto_delimiter_records_extension_hint_mismatch_warning() -> None:
    path = FIXTURES / "mismatched_extension.tsv"
    probe = probe_spectrum_file(path)
    data = read_spectrum_file(path)

    assert probe.selected_delimiter == "semicolon"
    assert data.metadata["delimiter_name"] == "semicolon"
    warning_text = " ".join(probe.warnings).lower()
    assert "semicolon" in warning_text
    assert "tab" in warning_text or "extension" in warning_text


def test_explicit_comma_decimal_with_auto_comma_delimiter_is_rejected() -> None:
    options = TextImportOptions(delimiter="auto", decimal_mark="comma")

    with pytest.raises(SpectrumReadError, match=r"(?i)comma|ambiguous|decimal"):
        read_spectrum_file(FIXTURES / "single.xy", import_options=options)


def test_integer_comma_table_cannot_hide_explicit_comma_decimal_conflict(
    tmp_path: Path,
) -> None:
    path = tmp_path / "integer_columns.csv"
    path.write_text("1002,1\n1001,2\n1000,3\n", encoding="utf-8")

    with pytest.raises(SpectrumReadError, match=r"(?i)comma|ambiguous|decimal"):
        read_spectrum_file(
            path,
            import_options=TextImportOptions(delimiter="auto", decimal_mark="comma"),
        )


def test_mixed_dot_and_comma_decimals_are_rejected() -> None:
    with pytest.raises(SpectrumReadError, match=r"(?i)mixed|decimal|numeric"):
        read_spectrum_file(FIXTURES / "mixed_decimal_semicolon.csv")


@pytest.mark.parametrize(
    ("fixture_name", "decimal_mark"),
    (
        ("thousands_eu_semicolon.csv", "comma"),
        ("thousands_us_semicolon.csv", "dot"),
    ),
)
def test_thousands_separators_are_not_guessed(
    fixture_name: str,
    decimal_mark: str,
) -> None:
    options = TextImportOptions(delimiter="semicolon", decimal_mark=decimal_mark)

    with pytest.raises(SpectrumReadError, match=r"(?i)thousands|ambiguous|numeric|decimal"):
        read_spectrum_file(FIXTURES / fixture_name, import_options=options)


def test_scientific_notation_supports_signs_and_both_exponent_cases() -> None:
    data = read_spectrum_file(FIXTURES / "scientific_notation.dat")
    probe = probe_spectrum_file(FIXTURES / "scientific_notation.dat")

    np.testing.assert_array_equal(data.wavenumber, [1002.0, 1001.0, 1000.0])
    np.testing.assert_allclose(
        data.spectra,
        [[0.00125, 0.0015, 0.00175], [0.0025, 0.00275, 0.003]],
    )
    assert probe.selected_decimal_mark == "dot"
    assert probe.layout == "wide_table"


@pytest.mark.parametrize(
    ("case_name", "payload", "selected_encoding", "expected_header"),
    (
        (
            "utf8",
            b"Wavenumber\tIntensity\n1002\t0.1\n1001\t0.2\n1000\t0.3\n",
            "utf-8-sig",
            ("Wavenumber", "Intensity"),
        ),
        (
            "utf8_bom",
            codecs.BOM_UTF8
            + b"Wavenumber\tIntensity\n1002\t0.1\n1001\t0.2\n1000\t0.3\n",
            "utf-8-sig",
            ("Wavenumber", "Intensity"),
        ),
        (
            "utf16_le_bom",
            codecs.BOM_UTF16_LE
            + "波数\t吸光丁\n1002\t0.1\n1001\t0.2\n1000\t0.3\n".encode(
                "utf-16-le"
            ),
            "utf-16-le",
            ("波数", "吸光丁"),
        ),
        (
            "utf16_be_bom",
            codecs.BOM_UTF16_BE
            + "波数\t吸光丁\n1002\t0.1\n1001\t0.2\n1000\t0.3\n".encode(
                "utf-16-be"
            ),
            "utf-16-be",
            ("波数", "吸光丁"),
        ),
        (
            "gb18030",
            "波数\t吸光度\n1002\t0.1\n1001\t0.2\n1000\t0.3\n".encode("gb18030"),
            "gb18030",
            ("波数", "吸光度"),
        ),
        (
            "cp1252",
            "Wavenumber\tAbsorbancé\n1002\t0.1\n1001\t0.2\n1000\t0.3\n".encode(
                "cp1252"
            ),
            "cp1252",
            ("Wavenumber", "Absorbancé"),
        ),
    ),
)
def test_encoding_detection_preserves_raw_bytes_hash_and_header(
    tmp_path: Path,
    case_name: str,
    payload: bytes,
    selected_encoding: str,
    expected_header: tuple[str, ...],
) -> None:
    path = tmp_path / f"{case_name}.tsv"
    path.write_bytes(payload)

    probe = probe_spectrum_file(path)
    data = read_spectrum_file(path)

    assert probe.selected_encoding == selected_encoding
    assert probe.encoding_evidence
    assert probe.header == expected_header
    assert probe.source_sha256 == hashlib.sha256(payload).hexdigest()
    assert probe.size_bytes == len(payload)
    assert data.metadata["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert data.metadata["encoding"] == selected_encoding
    np.testing.assert_array_equal(data.wavenumber, [1002, 1001, 1000])
    if case_name in {"utf8_bom", "utf16_le_bom", "utf16_be_bom"}:
        assert "bom" in probe.encoding_evidence.lower()
    if case_name == "gb18030":
        warning_text = " ".join(probe.warnings).lower()
        assert "cp1252" in warning_text
        assert "explicit" in warning_text


@pytest.mark.parametrize(
    ("fixture_name", "selected_encoding", "expected_header"),
    (
        ("encoding_utf8.tsv", "utf-8-sig", ("Wavenumber", "Intensity")),
        ("encoding_utf8_bom.tsv", "utf-8-sig", ("Wavenumber", "Intensity")),
        ("encoding_utf16le_bom.tsv", "utf-16-le", ("波数", "吸光丁")),
        ("encoding_utf16be_bom.tsv", "utf-16-be", ("波数", "吸光丁")),
        ("encoding_gb18030.tsv", "gb18030", ("波数", "吸光度")),
        ("encoding_cp1252.tsv", "cp1252", ("Wavenumber", "Absorbancé")),
    ),
)
def test_repository_encoding_fixtures_are_parseable_and_hashed_from_raw_bytes(
    fixture_name: str,
    selected_encoding: str,
    expected_header: tuple[str, ...],
) -> None:
    path = FIXTURES / fixture_name
    raw = path.read_bytes()

    probe = probe_spectrum_file(path)
    data = read_spectrum_file(path)

    assert probe.selected_encoding == selected_encoding
    assert probe.header == expected_header
    assert probe.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert data.metadata["source_sha256"] == hashlib.sha256(raw).hexdigest()
    np.testing.assert_array_equal(data.wavenumber, [1002, 1001, 1000])


def test_explicit_encoding_option_uses_requested_decoder(tmp_path: Path) -> None:
    path = tmp_path / "explicit_cp1252.csv"
    path.write_bytes(
        "Wavenumber;Absorbancé\n1002;0.1\n1001;0.2\n1000;0.3\n".encode("cp1252")
    )
    options = TextImportOptions(encoding="cp1252", delimiter="semicolon")

    probe = probe_spectrum_file(path, options=options)

    assert probe.selected_encoding == "cp1252"
    assert probe.selected_delimiter == "semicolon"
    assert probe.header == ("Wavenumber", "Absorbancé")


def test_invalid_bytes_are_not_forced_through_a_fallback_decoder(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"\x81")

    with pytest.raises(SpectrumReadError, match=r"(?i)decode|encoding|binary|text"):
        probe_spectrum_file(path)


def test_binary_looking_payload_is_rejected_even_with_text_extension(tmp_path: Path) -> None:
    path = tmp_path / "not_really_text.txt"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(32)) * 16)

    with pytest.raises(SpectrumReadError, match=r"(?i)binary|text|export"):
        read_spectrum_file(path)


@pytest.mark.parametrize(
    "signature",
    (
        b"%PDF-1.7\n",
        b"GIF89a",
        b"\x89PNG\r\n\x1a\n",
        b"\xff\xd8\xff\xe0JFIF",
        b"II*\x00",
        b"MM\x00*",
        b"BM\x3a\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00",
    ),
)
def test_obvious_binary_signature_cannot_hide_in_text_preamble(
    tmp_path: Path,
    signature: bytes,
) -> None:
    path = tmp_path / "disguised.txt"
    path.write_bytes(signature + b"\n1002,0.1\n1001,0.2\n1000,0.3\n")

    with pytest.raises(SpectrumReadError, match=r"(?i)image|pdf|binary|text|export"):
        read_spectrum_file(path)


def test_plain_text_beginning_with_bm_is_not_misclassified_as_bitmap(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bm_header.txt"
    path.write_text(
        "BM,Intensity\n1002,0.1\n1001,0.2\n1000,0.3\n",
        encoding="utf-8",
    )

    data = read_spectrum_file(path)

    assert data.perturbation_labels == ("Intensity",)


@pytest.mark.parametrize(
    "prefix",
    (
        codecs.BOM_UTF8,
        b"\n",
        b"JUNK BEFORE HEADER\n",
    ),
)
def test_pdf_signature_within_leading_bytes_is_rejected(
    tmp_path: Path,
    prefix: bytes,
) -> None:
    path = tmp_path / "pdf_with_prefix.txt"
    path.write_bytes(prefix + b"%PDF-1.7\n1002,0.1\n1001,0.2\n1000,0.3\n")

    with pytest.raises(SpectrumReadError, match=r"(?i)pdf|text|export"):
        read_spectrum_file(path)


def test_pdf_signature_starting_at_last_allowed_offset_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pdf_at_1023.txt"
    path.write_bytes(
        b"#" + b"x" * 1022 + b"%PDF-1.7\n1002,0.1\n1001,0.2\n1000,0.3\n"
    )

    with pytest.raises(SpectrumReadError, match=r"(?i)pdf|text|export"):
        read_spectrum_file(path)


@pytest.mark.parametrize(
    "suffix",
    (
        ".spa",
        ".spg",
        ".srs",
        ".0",
        ".1",
        ".spc",
        ".sp",
        ".jdx",
        ".dx",
        ".jcamp",
        ".xls",
        ".xlsx",
        ".zip",
    ),
)
def test_vendor_binary_and_structured_extensions_are_actionably_rejected(
    tmp_path: Path,
    suffix: str,
) -> None:
    path = tmp_path / f"unsupported{suffix}"
    path.write_bytes(b"1002,0.1\n1001,0.2\n1000,0.3\n")

    with pytest.raises(SpectrumReadError, match=r"(?i)text|export|structured|opus"):
        probe_spectrum_file(path)


def test_zip_signature_is_rejected_when_disguised_as_text(tmp_path: Path) -> None:
    path = tmp_path / "archive_disguised_as_text.csv"
    path.write_bytes(b"PK\x03\x04" + bytes(range(32)))

    with pytest.raises(SpectrumReadError, match=r"(?i)zip|text|export"):
        read_spectrum_file(path)


def test_preamble_comments_header_and_probe_metadata_are_auditable() -> None:
    path = FIXTURES / "preamble_header_comments.txt"
    raw_bytes = path.read_bytes()
    probe = probe_spectrum_file(path)
    data = read_spectrum_file(path)

    assert isinstance(probe, ImportProbe)
    assert probe.source_name == path.name
    assert probe.extension == ".txt"
    assert probe.source_sha256 == hashlib.sha256(raw_bytes).hexdigest()
    assert probe.size_bytes == len(raw_bytes)
    assert probe.selected_encoding == "utf-8-sig"
    assert probe.selected_delimiter == "semicolon"
    assert probe.selected_decimal_mark == "dot"
    assert probe.header_present is True
    assert probe.header == ("Wavenumber", "0MIN", "5MIN")
    assert probe.skipped_explicit_rows == 0
    assert probe.skipped_preamble_lines == (1, 2)
    assert probe.skipped_comment_lines == (4, 5, 6)
    assert probe.trimmed_empty_edge_columns == 0
    assert probe.numeric_block_start_line == 8
    assert probe.numeric_block_end_line == 10
    assert probe.data_rows == 3
    assert probe.columns == 3
    assert probe.layout == "wide_table"
    assert isinstance(probe.delimiter_evidence, Mapping)
    assert isinstance(probe.decimal_evidence, Mapping)

    expected_metadata_keys = {
        "import_parser",
        "import_parser_version",
        "source_format",
        "source_sha256",
        "encoding",
        "encoding_evidence",
        "delimiter",
        "delimiter_name",
        "delimiter_detection_candidates",
        "decimal_mark",
        "decimal_detection_evidence",
        "header_mode",
        "header",
        "skip_rows",
        "skipped_preamble_lines",
        "skipped_comment_lines",
        "trimmed_empty_edge_columns",
        "numeric_block_start_line",
        "numeric_block_end_line",
        "input_layout",
        "import_warnings",
    }
    assert expected_metadata_keys <= data.metadata.keys()
    assert data.metadata["import_parser"] == "ftir_text_table"
    assert data.metadata["import_parser_version"] == "2.1"
    assert data.metadata["source_sha256"] == probe.source_sha256
    assert data.metadata["encoding"] == probe.selected_encoding
    assert data.metadata["delimiter_name"] == probe.selected_delimiter
    assert data.metadata["decimal_mark"] == probe.selected_decimal_mark
    assert data.metadata["header"] == probe.header
    assert data.metadata["skipped_preamble_lines"] == probe.skipped_preamble_lines
    assert data.metadata["skipped_comment_lines"] == probe.skipped_comment_lines
    assert data.metadata["numeric_block_start_line"] == probe.numeric_block_start_line
    assert data.metadata["numeric_block_end_line"] == probe.numeric_block_end_line
    assert data.metadata["input_layout"] == probe.layout
    assert data.metadata["import_warnings"] == probe.warnings
    np.testing.assert_array_equal(data.wavenumber, [1002, 1001, 1000])
    np.testing.assert_allclose(
        data.spectra,
        [[0.125, 0.15, 0.175], [0.225, 0.25, 0.275]],
    )


def test_probe_evidence_is_deeply_immutable_and_to_dict_is_independent() -> None:
    probe = probe_spectrum_file(FIXTURES / "decimal_comma_semicolon.csv")
    candidates = probe.delimiter_evidence["candidates"]

    with pytest.raises(TypeError):
        candidates[0]["delimiter"] = "changed"  # type: ignore[index]

    serialized = probe.to_dict()
    serialized["delimiter_evidence"]["candidates"][0]["delimiter"] = "changed"
    assert probe.delimiter_evidence["candidates"][0]["delimiter"] != "changed"
    json.dumps(serialized)


def test_explicit_skip_rows_and_present_header_mode(tmp_path: Path) -> None:
    path = tmp_path / "skip_rows.csv"
    path.write_text(
        "discard this exporter line\n"
        "Wavenumber,Intensity\n"
        "1002,0.1\n"
        "1001,0.2\n"
        "1000,0.3\n",
        encoding="utf-8",
    )
    options = TextImportOptions(
        delimiter="comma",
        header_mode="present",
        skip_rows=1,
        allow_preamble=False,
    )

    probe = probe_spectrum_file(path, options=options)
    data = read_spectrum_file(path, import_options=options)

    assert probe.skipped_explicit_rows == 1
    assert probe.skipped_preamble_lines == ()
    assert probe.header == ("Wavenumber", "Intensity")
    assert probe.numeric_block_start_line == 3
    assert data.metadata["skip_rows"] == 1


def test_custom_comment_prefix_is_honored_without_preamble(tmp_path: Path) -> None:
    path = tmp_path / "custom_comment.csv"
    path.write_text(
        "! synthetic export note\n"
        "Wavenumber,Intensity\n"
        "1002,0.1\n"
        "1001,0.2\n"
        "1000,0.3\n",
        encoding="utf-8",
    )
    options = TextImportOptions(comment_prefixes=("!",), allow_preamble=False)

    probe = probe_spectrum_file(path, options=options)

    assert probe.skipped_comment_lines == (1,)
    assert probe.skipped_preamble_lines == ()
    assert probe.numeric_block_start_line == 3


def test_comments_and_blank_lines_inside_numeric_region_do_not_drop_data() -> None:
    path = FIXTURES / "interleaved_comments.csv"

    probe = probe_spectrum_file(path)
    data = read_spectrum_file(path)

    assert probe.skipped_comment_lines == (3,)
    assert probe.numeric_block_start_line == 2
    assert probe.numeric_block_end_line == 6
    assert probe.data_rows == 3
    np.testing.assert_array_equal(data.wavenumber, [1002, 1001, 1000])


def test_header_modes_are_strict_and_preserve_two_row_legacy_tables(tmp_path: Path) -> None:
    headered = tmp_path / "headered.csv"
    headered.write_text(
        "Wavenumber,Intensity\n1002,0.1\n1001,0.2\n1000,0.3\n",
        encoding="utf-8",
    )
    headerless = tmp_path / "headerless.csv"
    headerless.write_text("1001,0.1\n1000,0.2\n", encoding="utf-8")

    auto_headered = read_spectrum_file(headered)
    auto_headerless = read_spectrum_file(headerless)
    explicit_absent = read_spectrum_file(
        headerless,
        import_options=TextImportOptions(header_mode="absent"),
    )

    np.testing.assert_array_equal(auto_headered.wavenumber, [1002, 1001, 1000])
    np.testing.assert_array_equal(auto_headerless.spectra, [[0.1, 0.2]])
    np.testing.assert_array_equal(explicit_absent.spectra, [[0.1, 0.2]])
    with pytest.raises(SpectrumReadError, match=r"(?i)header|non-numeric"):
        read_spectrum_file(
            headered,
            import_options=TextImportOptions(header_mode="absent"),
        )
    with pytest.raises(SpectrumReadError, match=r"(?i)header"):
        read_spectrum_file(
            headerless,
            import_options=TextImportOptions(header_mode="present"),
        )


def test_preamble_can_be_explicitly_forbidden() -> None:
    options = TextImportOptions(allow_preamble=False)

    with pytest.raises(SpectrumReadError, match=r"(?i)preamble|line 1|metadata"):
        read_spectrum_file(
            FIXTURES / "preamble_header_comments.txt",
            import_options=options,
        )


def test_equal_length_numeric_blocks_require_explicit_disambiguation() -> None:
    with pytest.raises(SpectrumReadError, match=r"(?i)ambiguous|skip|block"):
        probe_spectrum_file(FIXTURES / "ambiguous_numeric_blocks.csv")


def test_all_empty_edge_columns_are_trimmed_only_when_enabled() -> None:
    path = FIXTURES / "empty_edge_columns.csv"
    probe = probe_spectrum_file(path)
    data = read_spectrum_file(path)

    assert probe.trimmed_empty_edge_columns == 2
    assert probe.header == ("Wavenumber", "0MIN", "5MIN")
    assert probe.columns == 3
    np.testing.assert_array_equal(data.wavenumber, [1002, 1001, 1000])
    assert data.spectra.shape == (2, 3)

    options = TextImportOptions(trim_empty_edge_columns=False)
    with pytest.raises(SpectrumReadError, match=r"(?i)empty|column"):
        read_spectrum_file(path, import_options=options)


def test_internal_missing_value_is_never_trimmed() -> None:
    path = FIXTURES / "internal_missing.csv"

    with pytest.raises(SpectrumReadError) as caught:
        read_spectrum_file(path)

    message = str(caught.value).lower()
    assert path.name.lower() in message
    assert "line 3" in message
    assert "column 2" in message
    assert "0min" in message
    assert "empty" in message or "missing" in message


def test_bad_row_inside_numeric_block_reports_full_parse_context() -> None:
    path = FIXTURES / "bad_inside_numeric.csv"

    with pytest.raises(SpectrumReadError) as caught:
        read_spectrum_file(path)

    message = str(caught.value).lower()
    for expected in (
        path.name.lower(),
        "line 3",
        "column 2",
        "absorbance",
        "oops",
        "comma",
        "dot",
        "utf-8",
    ):
        assert expected in message


def test_bad_row_cannot_be_reclassified_as_header_of_a_later_block(tmp_path: Path) -> None:
    path = tmp_path / "split_block.csv"
    path.write_text(
        "Wavenumber,Intensity\n"
        "1004,0.1\n"
        "bad,token\n"
        "1003,0.2\n"
        "1002,0.3\n"
        "1001,0.4\n",
        encoding="utf-8",
    )

    with pytest.raises(SpectrumReadError) as caught:
        read_spectrum_file(path)

    message = str(caught.value).lower()
    assert path.name.lower() in message
    assert "line 3" in message
    assert "bad" in message


def test_preamble_does_not_weaken_bad_row_strictness(tmp_path: Path) -> None:
    path = tmp_path / "preamble_split_block.csv"
    path.write_text(
        "Instrument: synthetic\n"
        "Wavenumber,Intensity\n"
        "1004,0.1\n"
        "bad,token\n"
        "1003,0.2\n"
        "1002,0.3\n"
        "1001,0.4\n",
        encoding="utf-8",
    )

    with pytest.raises(SpectrumReadError) as caught:
        read_spectrum_file(path)

    message = str(caught.value).lower()
    assert "line 4" in message
    assert "bad" in message


def test_same_width_text_metadata_is_allowed_before_axis_labelled_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata_semicolon.dat"
    path.write_text(
        "Instrument;Synthetic FTIR\n"
        "Resolution;4 cm-1\n"
        "Wavenumber (cm-1);Intensity\n"
        "1002;0.1\n"
        "1001;0.2\n"
        "1000;0.3\n",
        encoding="utf-8",
    )

    probe = probe_spectrum_file(path)
    data = read_spectrum_file(path)

    assert probe.skipped_preamble_lines == (1, 2)
    assert probe.header == ("Wavenumber (cm-1)", "Intensity")
    np.testing.assert_array_equal(data.wavenumber, [1002, 1001, 1000])


def test_short_numeric_block_cannot_be_silently_restarted_by_weak_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "split_headerless.csv"
    path.write_text(
        "1006,0.1\n"
        "1005,0.2\n"
        "bad,token\n"
        "1004,0.3\n"
        "1003,0.4\n"
        "1002,0.5\n"
        "1001,0.6\n",
        encoding="utf-8",
    )

    with pytest.raises(SpectrumReadError, match=r"(?i)line 3|bad|block"):
        read_spectrum_file(path)


@pytest.mark.parametrize(
    "first_line",
    (
        "nan,0.1",
        "Infinity,oops",
        ",0.1",
        "1004,",
        "1003,5;0.1",
    ),
)
def test_numeric_like_bad_first_line_is_not_consumed_as_header_or_preamble(
    tmp_path: Path,
    first_line: str,
) -> None:
    delimiter = ";" if ";" in first_line else ","
    path = tmp_path / "bad_first.csv"
    path.write_text(
        first_line
        + "\n"
        + delimiter.join(("1003", "0.2"))
        + "\n"
        + delimiter.join(("1002", "0.3"))
        + "\n"
        + delimiter.join(("1001", "0.4"))
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SpectrumReadError):
        read_spectrum_file(path)


def test_half_numeric_weak_header_is_ambiguous_unless_explicit(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed_first.csv"
    malformed.write_text(
        "oops,0.1\n1002,0.2\n1001,0.3\n1000,0.4\n",
        encoding="utf-8",
    )
    with pytest.raises(SpectrumReadError, match=r"(?i)oops|numeric|line 1"):
        read_spectrum_file(malformed)

    wide = tmp_path / "explicit_numeric_labels.csv"
    wide.write_text(
        "X,0,5\n1002,0.1,0.2\n1001,0.2,0.3\n1000,0.3,0.4\n",
        encoding="utf-8",
    )
    with pytest.raises(SpectrumReadError, match=r"(?i)numeric|line 1"):
        read_spectrum_file(wide)
    explicit = read_spectrum_file(
        wide,
        import_options=TextImportOptions(header_mode="present"),
    )
    assert explicit.perturbation_labels == ("0", "5")


def test_majority_text_weak_header_may_contain_one_numeric_label(
    tmp_path: Path,
) -> None:
    path = tmp_path / "majority_text_header.csv"
    path.write_text(
        "Axis,Early,5\n1002,0.1,0.2\n1001,0.2,0.3\n1000,0.3,0.4\n",
        encoding="utf-8",
    )

    data = read_spectrum_file(path)

    assert data.perturbation_labels == ("Early", "5")


def test_majority_text_weak_header_with_global_empty_edges_is_preserved(
    tmp_path: Path,
) -> None:
    path = tmp_path / "majority_text_header_edges.csv"
    path.write_text(
        ",Axis,Early,5,\n"
        ",1002,0.1,0.2,\n"
        ",1001,0.2,0.3,\n"
        ",1000,0.3,0.4,\n",
        encoding="utf-8",
    )

    probe = probe_spectrum_file(path)
    data = read_spectrum_file(path)

    assert probe.trimmed_empty_edge_columns == 2
    assert data.perturbation_labels == ("Early", "5")


def test_short_numeric_metadata_can_precede_explicit_header_and_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "numeric_metadata.csv"
    path.write_text(
        "1,64\n"
        "2,4\n"
        "Wavenumber,Intensity\n"
        "1002,0.1\n"
        "1001,0.2\n"
        "1000,0.3\n",
        encoding="utf-8",
    )

    probe = probe_spectrum_file(path)
    data = read_spectrum_file(path)

    assert probe.skipped_preamble_lines == (1, 2)
    assert probe.header == ("Wavenumber", "Intensity")
    assert probe.numeric_block_start_line == 4
    np.testing.assert_array_equal(data.wavenumber, [1002, 1001, 1000])


@pytest.mark.parametrize("prefix_rows", (3, 4))
def test_full_numeric_block_before_axis_header_is_not_silently_preamble(
    tmp_path: Path,
    prefix_rows: int,
) -> None:
    path = tmp_path / "numeric_block_before_header.csv"
    prefix = "".join(
        f"{1010 - index},{0.1 + index / 100:.3f}\n"
        for index in range(prefix_rows)
    )
    path.write_text(
        prefix
        + "Wavenumber,Intensity\n"
        + "1004,0.5\n1003,0.6\n1002,0.7\n1001,0.8\n",
        encoding="utf-8",
    )

    with pytest.raises(SpectrumReadError, match=r"(?i)earlier|block|preamble|line 1"):
        read_spectrum_file(path)


def test_two_point_block_after_preamble_requires_explicit_header_mode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "short_after_preamble.csv"
    path.write_text(
        "Instrument: synthetic\n"
        "Wavenumber,Intensity\n"
        "1001,0.1\n"
        "1000,0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(SpectrumReadError, match=r"(?i)three|1-2|explicit|header"):
        read_spectrum_file(path)

    data = read_spectrum_file(
        path,
        import_options=TextImportOptions(header_mode="present"),
    )
    np.testing.assert_array_equal(data.wavenumber, [1001, 1000])


def test_ragged_row_inside_numeric_block_is_not_skipped(tmp_path: Path) -> None:
    path = tmp_path / "ragged.csv"
    path.write_text(
        "Wavenumber,Intensity\n"
        "1002,0.1\n"
        "1001,0.2,unexpected\n"
        "1000,0.3\n",
        encoding="utf-8",
    )

    with pytest.raises(SpectrumReadError) as caught:
        read_spectrum_file(path)

    message = str(caught.value).lower()
    assert path.name.lower() in message
    assert "line 3" in message
    assert "column" in message


def test_edge_empty_cell_on_only_one_row_is_not_globally_trimmed(tmp_path: Path) -> None:
    path = tmp_path / "partial_edge_empty.csv"
    path.write_text(
        "Wavenumber,Intensity\n"
        "1002,0.1\n"
        ",1001,0.2\n"
        "1000,0.3\n",
        encoding="utf-8",
    )

    with pytest.raises(SpectrumReadError) as caught:
        read_spectrum_file(path)

    message = str(caught.value).lower()
    assert path.name.lower() in message
    assert "line 3" in message
    assert "empty" in message or "column" in message


def test_edge_column_must_be_empty_across_header_and_data(tmp_path: Path) -> None:
    path = tmp_path / "header_edge_mismatch.csv"
    path.write_text(
        "Wavenumber,Intensity\n"
        ",1002,0.1\n"
        ",1001,0.2\n"
        ",1000,0.3\n",
        encoding="utf-8",
    )

    with pytest.raises(SpectrumReadError) as caught:
        read_spectrum_file(path)

    message = str(caught.value).lower()
    assert "line 1" in message
    assert "header" in message
    assert "empty" in message or "column" in message


@pytest.mark.parametrize(
    "payload",
    (
        "Wavenumber,A,B\n1002,0.1\n1001,0.2\n1000,0.3\n",
        "Wavenumber,Intensity\n1002,0.1,0.2\n1001,0.2,0.3\n1000,0.3,0.4\n",
        "Wavenumber,Intensity\n1002,5,0,125\n1001,5,0,150\n1000,5,0,175\n",
    ),
)
def test_axis_header_width_mismatch_cannot_be_demoted_to_preamble(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "header_width_mismatch.csv"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(SpectrumReadError) as caught:
        read_spectrum_file(path)

    message = str(caught.value).lower()
    assert "line 1" in message
    assert "header" in message
    assert "column" in message


@pytest.mark.parametrize(
    "metadata_line",
    (
        "Instrument metadata",
        "Instrument: Synthetic FTIR",
        "Instrument,Synthetic,FTIR",
    ),
)
def test_different_width_text_line_is_headerless_preamble(
    tmp_path: Path,
    metadata_line: str,
) -> None:
    path = tmp_path / "headerless_with_preamble.csv"
    path.write_text(
        metadata_line + "\n1002,0.1\n1001,0.2\n1000,0.3\n",
        encoding="utf-8",
    )

    probe = probe_spectrum_file(path)
    data = read_spectrum_file(path)

    assert probe.header is None
    assert probe.skipped_preamble_lines == (1,)
    np.testing.assert_array_equal(data.wavenumber, [1002, 1001, 1000])


@pytest.mark.parametrize(
    "payload",
    (
        "1002,5,0,125\n1001,5,0,150\n1000,5,0,175\n",
        "4,000.5,0.1\n3,000.5,0.2\n2,000.5,0.3\n",
        "1.234,56,0.1\n1.233,56,0.2\n1.232,56,0.3\n",
        "1002,5,0.1\n1001,5,0.2\n1000,5,0.3\n",
        "1002.0,0,5\n1001.0,0,6\n1000.0,0,7\n",
    ),
)
def test_headerless_comma_decimal_or_thousands_shape_is_ambiguous_in_auto(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "ambiguous_headerless.csv"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(SpectrumReadError, match=r"(?i)ambiguous|thousands|decimal"):
        read_spectrum_file(path)

    explicit = read_spectrum_file(
        path,
        import_options=TextImportOptions(delimiter="comma", decimal_mark="dot"),
    )
    assert explicit.n_points == 3


@pytest.mark.parametrize(
    "payload",
    (
        "1e3,2,0.1\n9e2,3,0.2\n8e2,4,0.3\n",
        "1002,-1,0.1\n1001,-2,0.2\n1000,-3,0.3\n",
        "1002,+1,0.1\n1001,+2,0.2\n1000,+3,0.3\n",
    ),
)
def test_unambiguous_scientific_or_signed_comma_wide_table_stays_supported(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "signed_or_scientific_wide.csv"
    path.write_text(payload, encoding="utf-8")

    data = read_spectrum_file(path)

    assert data.n_spectra == 2
    assert data.n_points == 3


def test_long_preamble_is_not_limited_by_delimiter_sampling_window(tmp_path: Path) -> None:
    path = tmp_path / "long_preamble.dat"
    preamble = "".join(f"Metadata field {index}: synthetic\n" for index in range(35))
    path.write_text(
        preamble
        + "Wavenumber;Intensity\n"
        + "1002;0.1\n"
        + "1001;0.2\n"
        + "1000;0.3\n",
        encoding="utf-8",
    )

    probe = probe_spectrum_file(path)
    data = read_spectrum_file(path)

    assert probe.skipped_preamble_lines == tuple(range(1, 36))
    assert probe.numeric_block_start_line == 37
    np.testing.assert_array_equal(data.wavenumber, [1002, 1001, 1000])


def test_noncomment_footer_is_rejected_instead_of_truncating_the_block() -> None:
    path = FIXTURES / "footer_text.csv"

    with pytest.raises(SpectrumReadError) as caught:
        read_spectrum_file(path)

    message = str(caught.value).lower()
    assert path.name.lower() in message
    assert "line 5" in message
    assert "end of exported spectrum" in message


def test_probe_and_read_share_identical_parser_decisions() -> None:
    path = FIXTURES / "decimal_comma_semicolon.csv"
    options = TextImportOptions()
    probe = probe_spectrum_file(path, options=options)
    data = read_spectrum_file(path, import_options=options)

    decision_pairs = {
        "encoding": probe.selected_encoding,
        "delimiter_name": probe.selected_delimiter,
        "decimal_mark": probe.selected_decimal_mark,
        "header": probe.header,
        "skipped_preamble_lines": probe.skipped_preamble_lines,
        "skipped_comment_lines": probe.skipped_comment_lines,
        "trimmed_empty_edge_columns": probe.trimmed_empty_edge_columns,
        "numeric_block_start_line": probe.numeric_block_start_line,
        "numeric_block_end_line": probe.numeric_block_end_line,
        "input_layout": probe.layout,
        "import_warnings": probe.warnings,
    }
    for metadata_key, probe_value in decision_pairs.items():
        assert data.metadata[metadata_key] == probe_value


def test_mixed_format_multifile_series_uses_options_per_file() -> None:
    series = FIXTURES / "mixed_series"
    paths = [series / "0MIN.dpt", series / "5MIN.tsv", series / "10MIN.asc"]
    data = load_spectrum_files(paths, import_options=TextImportOptions())

    np.testing.assert_array_equal(data.wavenumber, [1002, 1001, 1000])
    np.testing.assert_allclose(
        data.spectra,
        [[0.125, 0.15, 0.175], [0.225, 0.25, 0.275], [0.325, 0.35, 0.375]],
    )
    assert data.perturbation_labels == ("0MIN", "5MIN", "10MIN")
    np.testing.assert_array_equal(data.perturbation, [0, 5, 10])

    expected_names = tuple(path.name for path in paths)
    assert tuple(data.metadata["import_probe_by_file"]) == expected_names
    assert dict(data.metadata["delimiter_by_file"]) == {
        "0MIN.dpt": "comma",
        "5MIN.tsv": "tab",
        "10MIN.asc": "whitespace",
    }
    assert dict(data.metadata["decimal_mark_by_file"]) == {
        "0MIN.dpt": "dot",
        "5MIN.tsv": "comma",
        "10MIN.asc": "dot",
    }
    assert set(data.metadata["encoding_by_file"]) == set(expected_names)
    assert set(data.metadata["preamble_lines_by_file"]) == set(expected_names)
    assert set(data.metadata["trimmed_empty_columns_by_file"]) == set(expected_names)
    for serialized_probe in data.metadata["import_probe_by_file"].values():
        assert isinstance(serialized_probe, Mapping)
        assert serialized_probe["source_sha256"]
        assert serialized_probe["layout"] == "two_column"


def test_multifile_import_options_do_not_relax_point_for_point_axis_matching(
    tmp_path: Path,
) -> None:
    first = tmp_path / "0MIN.csv"
    first.write_text("1002,0.1\n1001,0.2\n1000,0.3\n", encoding="utf-8")
    second = tmp_path / "5MIN.tsv"
    second.write_text("1002\t0,4\n1000,5\t0,5\n1000\t0,6\n", encoding="utf-8")

    with pytest.raises(
        SpectrumValidationError,
        match=r"(?i)point-for-point|axis|wavenumber",
    ):
        load_spectrum_files(
            [first, second],
            import_options=TextImportOptions(),
        )
