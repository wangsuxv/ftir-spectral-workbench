"""Regenerate the synthetic encoded-text fixtures used by import tests."""

from __future__ import annotations

import codecs
from pathlib import Path

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "import_compat"
)
ASCII_TABLE = "Wavenumber\tIntensity\n1002\t0.1\n1001\t0.2\n1000\t0.3\n"
CHINESE_TABLE = "波数\t吸光丁\n1002\t0.1\n1001\t0.2\n1000\t0.3\n"


def main() -> None:
    """Write deterministic, private-data-free encoding fixtures."""

    payloads = {
        "encoding_utf8.tsv": ASCII_TABLE.encode("utf-8"),
        "encoding_utf8_bom.tsv": codecs.BOM_UTF8 + ASCII_TABLE.encode("utf-8"),
        "encoding_utf16le_bom.tsv": (
            codecs.BOM_UTF16_LE + CHINESE_TABLE.encode("utf-16-le")
        ),
        "encoding_utf16be_bom.tsv": (
            codecs.BOM_UTF16_BE + CHINESE_TABLE.encode("utf-16-be")
        ),
        "encoding_gb18030.tsv": "波数\t吸光度\n1002\t0.1\n1001\t0.2\n1000\t0.3\n".encode(
            "gb18030"
        ),
        "encoding_cp1252.tsv": (
            "Wavenumber\tAbsorbancé\n1002\t0.1\n1001\t0.2\n1000\t0.3\n".encode(
                "cp1252"
            )
        ),
    }
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (FIXTURE_DIRECTORY / name).write_bytes(payload)


if __name__ == "__main__":
    main()
