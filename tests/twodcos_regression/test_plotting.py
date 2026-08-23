from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ftir2dcos.plotting import (
    create_2d_contour,
    create_baseline_qc_representative,
    create_spectra_overlay,
    plot_dynamic_spectra_overlay,
    save_figure,
)


@pytest.fixture
def spectral_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    wavenumber = np.linspace(1509.0, 1736.0, 12)
    raw = np.vstack(
        [0.05 + np.sin(wavenumber / 55.0 + phase) * 0.01 for phase in (0.0, 0.4, 0.8, 1.2, 1.6)]
    )
    baselines = np.vstack(
        [0.01 + index * 0.001 + (wavenumber - wavenumber.mean()) * 1e-5 for index in range(5)]
    )
    corrected = raw - baselines
    return wavenumber, raw, baselines, corrected


def test_1d_overlay_has_high_wavenumber_on_left(
    spectral_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    wavenumber, raw, _, _ = spectral_arrays
    figure = create_spectra_overlay(
        wavenumber,
        raw,
        labels=("0", "1", "2", "3", "4"),
        title="Raw",
    )
    try:
        axis = figure.axes[0]
        assert axis.xaxis_inverted()
        assert len(axis.lines) == raw.shape[0]
        assert "Wavenumber" in axis.get_xlabel()
    finally:
        plt.close(figure)


def test_representative_baseline_uses_first_middle_last(
    spectral_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    wavenumber, raw, baselines, corrected = spectral_arrays
    figure = create_baseline_qc_representative(
        wavenumber,
        raw,
        baselines,
        corrected,
        labels=("first", "second", "middle", "fourth", "last"),
    )
    try:
        assert [axis.get_title(loc="left") for axis in figure.axes] == [
            "first",
            "middle",
            "last",
        ]
        assert all(axis.xaxis_inverted() for axis in figure.axes)
        assert all(len(axis.lines) == 3 for axis in figure.axes)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("kind", ["synchronous", "asynchronous"])
def test_2d_contour_uses_symmetric_display_scale_without_changing_matrix(kind: str) -> None:
    wavenumber = np.linspace(1509.0, 1736.0, 10)
    matrix = np.arange(100, dtype=np.float64).reshape(10, 10) - 20.0
    original = matrix.copy()
    figure = create_2d_contour(
        wavenumber,
        matrix,
        kind=kind,
        convention="2dpy_compatible",
        display_percentile=80.0,
        contour_levels=11,
    )
    try:
        axis = figure.axes[0]
        assert axis.xaxis_inverted()
        assert axis.yaxis_inverted()
        assert np.array_equal(matrix, original)
        assert "matrix values unchanged" in figure.texts[0].get_text()
        assert "2dpy_compatible" in axis.get_title(loc="left")
        if kind == "synchronous":
            assert len(axis.lines) == 1
        else:
            assert not axis.lines
    finally:
        plt.close(figure)


def test_save_figure_writes_png_pdf_and_closes_figure(
    tmp_path: Path,
    spectral_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    wavenumber, raw, _, _ = spectral_arrays
    figure = create_spectra_overlay(wavenumber, raw)
    figure_number = figure.number
    paths = save_figure(figure, tmp_path / "overlay")

    assert set(paths) == {"png", "pdf"}
    assert paths["png"].read_bytes().startswith(b"\x89PNG")
    assert paths["pdf"].read_bytes().startswith(b"%PDF")
    assert figure_number not in plt.get_fignums()


def test_plot_wrapper_closes_figure(
    tmp_path: Path,
    spectral_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    wavenumber, raw, _, _ = spectral_arrays
    before = set(plt.get_fignums())
    paths = plot_dynamic_spectra_overlay(
        wavenumber,
        raw - raw.mean(axis=0),
        tmp_path / "dynamic",
        formats=("png",),
    )
    assert paths["png"].is_file()
    assert set(plt.get_fignums()) == before


def test_plotting_rejects_nonfinite_values() -> None:
    wavenumber = np.linspace(1509.0, 1736.0, 10)
    spectra = np.ones((3, 10), dtype=np.float64)
    spectra[0, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        create_spectra_overlay(wavenumber, spectra)
