from __future__ import annotations

from pathlib import Path

import numpy as np

from ftir_baseline.io import read_spectrum_file


def test_wide_csv_supports_chinese_header_and_path_with_spaces(tmp_path: Path) -> None:
    source_directory = tmp_path / "中文 原始数据 目录"
    source_directory.mkdir()
    source = source_directory / "样品 时间序列.csv"
    source.write_text(
        "波数,10分钟,0分钟,20分钟\n"
        "1800,0.20,0.10,0.30\n"
        "1600,0.40,0.25,0.55\n"
        "1400,0.15,0.12,0.18\n"
        "1200,0.35,0.22,0.48\n"
        "900,0.08,0.05,0.11\n",
        encoding="utf-8",
    )

    result = read_spectrum_file(
        source,
        input_unit="absorbance",
        sort_by_perturbation=True,
    )

    assert result.source_name == "样品 时间序列.csv"
    assert result.metadata["source_path"] == str(source.resolve())
    assert result.perturbation_labels == ("0分钟", "10分钟", "20分钟")
    np.testing.assert_array_equal(result.perturbation, [0.0, 10.0, 20.0])
    np.testing.assert_array_equal(result.spectra[:, 0], [0.10, 0.20, 0.30])
    np.testing.assert_array_equal(result.wavenumber, [1800, 1600, 1400, 1200, 900])
