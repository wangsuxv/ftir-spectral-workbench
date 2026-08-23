"""Plotting helpers for FTIR spectra and two-dimensional correlation maps.

The functions in this module never change scientific arrays.  Figure builders are
kept separate from the saving wrappers so a UI can render a preview, while batch
exports can guarantee that every Matplotlib figure is closed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

# File exports and Streamlit rendering both require a non-interactive, headless-safe backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DPI = 300
DEFAULT_CONTOUR_LEVELS = 21
DEFAULT_DISPLAY_PERCENTILE = 99.0


def _validate_wavenumber(wavenumber: np.ndarray | Sequence[float]) -> np.ndarray:
    axis = np.asarray(wavenumber, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("wavenumber must be a one-dimensional array with at least 2 points")
    if not np.all(np.isfinite(axis)):
        raise ValueError("wavenumber contains NaN or infinite values")
    return axis


def _validate_spectra(
    spectra: np.ndarray | Sequence[Sequence[float]],
    n_wavenumbers: int,
    *,
    name: str = "spectra",
) -> np.ndarray:
    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != n_wavenumbers:
        raise ValueError(f"{name} must have shape (n_spectra, {n_wavenumbers}); got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return values


def _validate_labels(labels: Sequence[str] | None, n_spectra: int) -> tuple[str, ...]:
    if labels is None:
        return tuple(f"Spectrum {index + 1}" for index in range(n_spectra))
    normalized = tuple(str(label) for label in labels)
    if len(normalized) != n_spectra:
        raise ValueError(f"labels must contain {n_spectra} values; got {len(normalized)}")
    return normalized


def _style_ftir_axis(axis: Axes, intensity_label: str) -> None:
    axis.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    axis.set_ylabel(intensity_label)
    axis.grid(axis="y", color="#D7DCE2", linewidth=0.6, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    if not axis.xaxis_inverted():
        axis.invert_xaxis()


def _line_colors(count: int, *, palette: str = "cividis") -> np.ndarray:
    if count == 1:
        return np.asarray([[0.10, 0.31, 0.52, 1.0]])
    return plt.get_cmap(palette)(np.linspace(0.12, 0.88, count))


def create_spectra_overlay(
    wavenumber: np.ndarray | Sequence[float],
    spectra: np.ndarray | Sequence[Sequence[float]],
    *,
    labels: Sequence[str] | None = None,
    title: str = "FTIR spectra",
    intensity_label: str = "Intensity",
    show_legend: bool | None = None,
    palette: str = "cividis",
) -> Figure:
    """Create a one-dimensional FTIR overlay with high wavenumber on the left."""

    axis_values = _validate_wavenumber(wavenumber)
    matrix = _validate_spectra(spectra, axis_values.size)
    series_labels = _validate_labels(labels, matrix.shape[0])
    if show_legend is None:
        show_legend = matrix.shape[0] <= 12

    figure, axis = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    colors = _line_colors(matrix.shape[0], palette=palette)
    for index, (series, label) in enumerate(zip(matrix, series_labels, strict=True)):
        axis.plot(
            axis_values,
            series,
            color=colors[index],
            linewidth=1.15,
            alpha=0.88,
            label=label,
        )
    axis.set_title(title, loc="left", color="#202830", fontweight="bold")
    _style_ftir_axis(axis, intensity_label)
    if show_legend:
        axis.legend(frameon=False, fontsize=8, ncols=min(3, max(1, matrix.shape[0])))
    return figure


def create_baseline_qc_representative(
    wavenumber: np.ndarray | Sequence[float],
    raw: np.ndarray | Sequence[Sequence[float]],
    baselines: np.ndarray | Sequence[Sequence[float]],
    corrected: np.ndarray | Sequence[Sequence[float]],
    *,
    labels: Sequence[str] | None = None,
    title: str = "Representative baseline quality control",
    intensity_label: str = "Intensity",
) -> Figure:
    """Create raw/baseline/corrected panels for first, middle and last spectra."""

    axis_values = _validate_wavenumber(wavenumber)
    raw_matrix = _validate_spectra(raw, axis_values.size, name="raw")
    baseline_matrix = _validate_spectra(baselines, axis_values.size, name="baselines")
    corrected_matrix = _validate_spectra(corrected, axis_values.size, name="corrected")
    if raw_matrix.shape != baseline_matrix.shape or raw_matrix.shape != corrected_matrix.shape:
        raise ValueError("raw, baselines and corrected must have identical shapes")
    series_labels = _validate_labels(labels, raw_matrix.shape[0])

    representative = tuple(dict.fromkeys((0, raw_matrix.shape[0] // 2, raw_matrix.shape[0] - 1)))
    figure, axes = plt.subplots(
        len(representative),
        1,
        figsize=(9.2, 3.35 * len(representative)),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
    )
    for row, spectrum_index in enumerate(representative):
        axis = axes[row, 0]
        axis.plot(
            axis_values,
            raw_matrix[spectrum_index],
            color="#606A73",
            linewidth=1.15,
            label="Raw",
        )
        axis.plot(
            axis_values,
            baseline_matrix[spectrum_index],
            color="#C78A15",
            linewidth=1.35,
            label="Baseline",
        )
        axis.plot(
            axis_values,
            corrected_matrix[spectrum_index],
            color="#1A5A96",
            linewidth=1.15,
            label="Corrected",
        )
        axis.set_title(series_labels[spectrum_index], loc="left", fontsize=10)
        _style_ftir_axis(axis, intensity_label)
        if row == 0:
            axis.legend(frameon=False, ncols=3, fontsize=8, loc="best")
    figure.suptitle(title, x=0.01, ha="left", color="#202830", fontweight="bold")
    return figure


def _display_limit(matrix: np.ndarray, display_percentile: float | None) -> tuple[float, str]:
    absolute = np.abs(matrix)
    maximum = float(np.max(absolute))
    if display_percentile is None:
        limit = maximum
        description = "full matrix range"
    else:
        percentile = float(display_percentile)
        if not 0.0 < percentile <= 100.0:
            raise ValueError("display_percentile must be in (0, 100] or None")
        limit = float(np.percentile(absolute, percentile))
        description = f"{percentile:g}th percentile display limit"
        if limit == 0.0 and maximum > 0.0:
            limit = maximum
    if limit == 0.0:
        limit = float(np.finfo(np.float64).eps)
    return limit, description


def create_2d_contour(
    wavenumber: np.ndarray | Sequence[float],
    matrix: np.ndarray | Sequence[Sequence[float]],
    *,
    kind: str,
    column_wavenumber: np.ndarray | Sequence[float] | None = None,
    convention: str = "canonical",
    row_variable: str = "nu1",
    column_variable: str = "nu2",
    method: str = "processed",
    contour_levels: int = DEFAULT_CONTOUR_LEVELS,
    filled: bool = True,
    display_percentile: float | None = DEFAULT_DISPLAY_PERCENTILE,
    cmap: str = "RdBu_r",
    show_diagonal: bool | None = None,
) -> Figure:
    """Create a synchronous or asynchronous 2D-COS contour map.

    Percentile limiting affects only the color normalization.  The input matrix is
    never clipped, copied back, or otherwise modified.
    """

    row_axis = _validate_wavenumber(wavenumber)
    column_axis = row_axis if column_wavenumber is None else _validate_wavenumber(column_wavenumber)
    values = np.asarray(matrix, dtype=np.float64)
    expected_shape = (row_axis.size, column_axis.size)
    if values.shape != expected_shape:
        raise ValueError(f"matrix must have shape {expected_shape}; got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix contains NaN or infinite values")
    if contour_levels < 2:
        raise ValueError("contour_levels must be at least 2")

    normalized_kind = kind.strip().lower()
    if normalized_kind not in {"synchronous", "asynchronous"}:
        raise ValueError("kind must be 'synchronous' or 'asynchronous'")
    if show_diagonal is None:
        show_diagonal = normalized_kind == "synchronous"

    limit, limit_description = _display_limit(values, display_percentile)
    levels = np.linspace(-limit, limit, int(contour_levels), dtype=np.float64)
    figure, axis = plt.subplots(figsize=(7.5, 6.6), constrained_layout=True)
    contour = (
        axis.contourf(
            column_axis,
            row_axis,
            values,
            levels=levels,
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
            extend="both",
        )
        if filled
        else axis.contour(
            column_axis,
            row_axis,
            values,
            levels=levels,
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
        )
    )
    colorbar = figure.colorbar(contour, ax=axis, pad=0.025)
    colorbar.set_label("Correlation intensity")

    row_lower = float(np.min(row_axis))
    row_upper = float(np.max(row_axis))
    column_lower = float(np.min(column_axis))
    column_upper = float(np.max(column_axis))
    axis.set_xlim(column_upper, column_lower)
    axis.set_ylim(row_upper, row_lower)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(f"Wavenumber ({column_variable}, cm$^{{-1}}$)")
    axis.set_ylabel(f"Wavenumber ({row_variable}, cm$^{{-1}}$)")
    if show_diagonal:
        diagonal_lower = max(row_lower, column_lower)
        diagonal_upper = min(row_upper, column_upper)
        if diagonal_lower <= diagonal_upper:
            axis.plot(
                [diagonal_lower, diagonal_upper],
                [diagonal_lower, diagonal_upper],
                color="#303840",
                linewidth=0.75,
                linestyle="--",
                alpha=0.7,
            )

    display_name = normalized_kind.capitalize()
    if np.array_equal(row_axis, column_axis):
        range_description = f"{row_upper:g}\u2013{row_lower:g} cm$^{{-1}}$"
    else:
        range_description = (
            f"rows {row_upper:g}\u2013{row_lower:g}; "
            f"columns {column_upper:g}\u2013{column_lower:g} cm$^{{-1}}$"
        )
    axis.set_title(
        f"{display_name} 2D-COS | {method} | {convention}\nRange: {range_description}",
        loc="left",
        fontsize=10.5,
        color="#202830",
        fontweight="bold",
    )
    figure.text(
        0.01,
        0.005,
        f"Symmetric color scale \u00b1{limit:.5g} ({limit_description}); matrix values unchanged.",
        fontsize=8,
        color="#505A64",
    )
    return figure


def _validate_block_axes(
    axes: Sequence[np.ndarray | Sequence[float]],
    *,
    name: str,
) -> tuple[np.ndarray, ...]:
    """Validate one non-empty collection of monotonic spectral axes."""

    try:
        supplied_axes = tuple(axes)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of wavenumber arrays") from error
    if not supplied_axes:
        raise ValueError(f"{name} must contain at least one wavenumber array")

    validated: list[np.ndarray] = []
    for index, supplied_axis in enumerate(supplied_axes):
        if np.iscomplexobj(supplied_axis):
            raise TypeError(f"{name}[{index}] must contain real values")
        try:
            axis = _validate_wavenumber(supplied_axis)
        except (TypeError, ValueError) as error:
            raise type(error)(f"{name}[{index}]: {error}") from error
        differences = np.diff(axis)
        if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
            raise ValueError(f"{name}[{index}] must be strictly monotonic")
        validated.append(axis)
    return tuple(validated)


def _validate_block_matrices(
    block_matrices: Sequence[Sequence[np.ndarray | Sequence[Sequence[float]]]],
    row_axes: Sequence[np.ndarray],
    column_axes: Sequence[np.ndarray],
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Validate an n-by-m rectangular collection without altering its arrays."""

    try:
        supplied_rows = tuple(block_matrices)
    except TypeError as error:
        raise TypeError("block_matrices must be a rectangular sequence of matrix rows") from error
    if len(supplied_rows) != len(row_axes):
        raise ValueError(
            "block_matrices must contain one row per row axis; "
            f"expected {len(row_axes)}, got {len(supplied_rows)}"
        )

    validated_rows: list[tuple[np.ndarray, ...]] = []
    for row_index, supplied_row in enumerate(supplied_rows):
        try:
            row = tuple(supplied_row)
        except TypeError as error:
            raise TypeError(f"block_matrices[{row_index}] must be a sequence") from error
        if len(row) != len(column_axes):
            raise ValueError(
                f"block_matrices[{row_index}] must contain {len(column_axes)} blocks; "
                f"got {len(row)}"
            )

        validated_blocks: list[np.ndarray] = []
        for column_index, supplied_matrix in enumerate(row):
            if np.iscomplexobj(supplied_matrix):
                raise TypeError(
                    f"block_matrices[{row_index}][{column_index}] must contain real values"
                )
            try:
                matrix = np.asarray(supplied_matrix, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"block_matrices[{row_index}][{column_index}] must be numeric"
                ) from error
            expected_shape = (row_axes[row_index].size, column_axes[column_index].size)
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"block_matrices[{row_index}][{column_index}] must have shape "
                    f"{expected_shape}; got {matrix.shape}"
                )
            if not np.all(np.isfinite(matrix)):
                raise ValueError(
                    f"block_matrices[{row_index}][{column_index}] contains NaN or infinite values"
                )
            validated_blocks.append(matrix)
        validated_rows.append(tuple(validated_blocks))
    return tuple(validated_rows)


def _normalize_block_labels(
    labels: Sequence[str] | None,
    count: int,
    *,
    name: str,
) -> tuple[str, ...]:
    if labels is None:
        return tuple(f"Range {index + 1}" for index in range(count))
    normalized = tuple(str(label).strip() for label in labels)
    if len(normalized) != count:
        raise ValueError(f"{name} must contain {count} values; got {len(normalized)}")
    if any(not label for label in normalized):
        raise ValueError(f"{name} cannot contain blank labels")
    return normalized


def _same_spectral_axis(first: np.ndarray, second: np.ndarray) -> bool:
    """Return true only for exactly identical coordinate sets."""

    return first.shape == second.shape and (
        np.array_equal(first, second) or np.array_equal(first, second[::-1])
    )


def _normalize_diagonal_blocks(
    diagonal_blocks: Iterable[tuple[int, int]] | Sequence[Sequence[bool]] | np.ndarray | None,
    row_axes: Sequence[np.ndarray],
    column_axes: Sequence[np.ndarray],
) -> frozenset[tuple[int, int]]:
    n_rows = len(row_axes)
    n_columns = len(column_axes)
    if diagonal_blocks is None:
        matching_axes = {
            (row_index, column_index)
            for row_index, row_axis in enumerate(row_axes)
            for column_index, column_axis in enumerate(column_axes)
            if _same_spectral_axis(row_axis, column_axis)
        }
        # Duplicate coordinate arrays are ambiguous: they could represent distinct
        # modalities or repeated ranges.  Require a one-to-one match unless the
        # caller explicitly identifies the genuine auto-correlation blocks.
        return frozenset(
            (row_index, column_index)
            for row_index, column_index in matching_axes
            if sum(match[0] == row_index for match in matching_axes) == 1
            and sum(match[1] == column_index for match in matching_axes) == 1
        )

    try:
        supplied = tuple(diagonal_blocks)
    except TypeError as error:
        raise TypeError(
            "diagonal_blocks must be an iterable of (row, column) pairs or a boolean mask"
        ) from error

    is_boolean_mask = len(supplied) == n_rows and all(
        isinstance(mask_row, (Sequence, np.ndarray))
        and not isinstance(mask_row, (str, bytes))
        and len(mask_row) == n_columns
        and all(isinstance(value, (bool, np.bool_)) for value in mask_row)
        for mask_row in supplied
    )
    if is_boolean_mask:
        return frozenset(
            (row_index, column_index)
            for row_index, mask_row in enumerate(supplied)
            for column_index, enabled in enumerate(mask_row)
            if enabled
        )

    normalized: set[tuple[int, int]] = set()
    for entry in supplied:
        if isinstance(entry, (str, bytes)):
            raise ValueError("diagonal_blocks entries must be (row, column) pairs")
        try:
            pair = tuple(entry)
        except TypeError as error:
            raise ValueError("diagonal_blocks entries must be (row, column) pairs") from error
        if len(pair) != 2 or any(
            isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral) for index in pair
        ):
            raise ValueError("diagonal_blocks entries must be integer (row, column) pairs")
        row_index, column_index = (int(pair[0]), int(pair[1]))
        if not (0 <= row_index < n_rows and 0 <= column_index < n_columns):
            raise ValueError(
                "diagonal_blocks index out of bounds: "
                f"({row_index}, {column_index}) for a {n_rows}x{n_columns} grid"
            )
        normalized.add((row_index, column_index))
    return frozenset(normalized)


def _panel_ratios(axes: Sequence[np.ndarray]) -> np.ndarray:
    """Use physical spectral span while keeping narrow panels readable."""

    spans = np.asarray([np.ptp(axis) for axis in axes], dtype=np.float64)
    median_span = float(np.median(spans))
    return np.clip(spans, median_span * 0.25, median_span * 6.0)


def _range_description(axis: np.ndarray) -> str:
    return f"{float(np.max(axis)):g}\N{EN DASH}{float(np.min(axis)):g} cm$^{{-1}}$"


def create_multi_range_2d_contour(
    row_wavenumbers: Sequence[np.ndarray | Sequence[float]],
    column_wavenumbers: Sequence[np.ndarray | Sequence[float]],
    block_matrices: Sequence[Sequence[np.ndarray | Sequence[Sequence[float]]]],
    *,
    kind: str,
    row_labels: Sequence[str] | None = None,
    column_labels: Sequence[str] | None = None,
    convention: str = "canonical",
    method: str = "processed",
    contour_levels: int = DEFAULT_CONTOUR_LEVELS,
    filled: bool = True,
    display_percentile: float | None = DEFAULT_DISPLAY_PERCENTILE,
    cmap: str = "RdBu_r",
    diagonal_blocks: (
        Iterable[tuple[int, int]] | Sequence[Sequence[bool]] | np.ndarray | None
    ) = None,
) -> Figure:
    """Create an arbitrary rectangular grid of auto- and cross-region 2D-COS blocks.

    Each block is rendered against its own row and column coordinates.  No axes or
    matrices are concatenated, regridded, or interpolated.  A single symmetric
    color normalization is calculated from all blocks; percentile limiting changes
    only that display normalization and never the supplied matrices.

    ``diagonal_blocks`` may be an iterable of ``(row, column)`` indices or an
    n-by-m boolean mask.  When omitted, a diagonal is drawn only where the row and
    column coordinate arrays are exactly the same (in either direction).
    """

    row_axes = _validate_block_axes(row_wavenumbers, name="row_wavenumbers")
    column_axes = _validate_block_axes(column_wavenumbers, name="column_wavenumbers")
    matrices = _validate_block_matrices(block_matrices, row_axes, column_axes)
    normalized_row_labels = _normalize_block_labels(row_labels, len(row_axes), name="row_labels")
    normalized_column_labels = _normalize_block_labels(
        column_labels, len(column_axes), name="column_labels"
    )
    marked_diagonals = _normalize_diagonal_blocks(diagonal_blocks, row_axes, column_axes)

    normalized_kind = str(kind).strip().lower()
    if normalized_kind not in {"synchronous", "asynchronous"}:
        raise ValueError("kind must be 'synchronous' or 'asynchronous'")
    if isinstance(contour_levels, (bool, np.bool_)):
        raise ValueError("contour_levels must be an integer of at least 2")
    try:
        level_count = int(contour_levels)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("contour_levels must be an integer of at least 2") from error
    if level_count < 2 or level_count != contour_levels:
        raise ValueError("contour_levels must be an integer of at least 2")
    try:
        color_map = plt.get_cmap(cmap)
    except ValueError as error:
        raise ValueError(f"unknown Matplotlib colormap {cmap!r}") from error

    pooled_absolute_values = np.concatenate(
        [np.abs(matrix).reshape(-1) for matrix_row in matrices for matrix in matrix_row]
    )
    limit, limit_description = _display_limit(pooled_absolute_values, display_percentile)
    colorbar_extend = "both" if limit < float(np.max(pooled_absolute_values)) else "neither"
    levels = np.linspace(-limit, limit, level_count, dtype=np.float64)
    normalization = Normalize(vmin=-limit, vmax=limit, clip=False)

    width_ratios = _panel_ratios(column_axes)
    height_ratios = _panel_ratios(row_axes)
    width_units = float(np.sum(width_ratios) / np.median(width_ratios))
    height_units = float(np.sum(height_ratios) / np.median(height_ratios))
    figure_width = float(np.clip(2.7 * width_units + 1.7, 7.0, 22.0))
    figure_height = float(np.clip(2.45 * height_units + 1.8, 5.8, 22.0))

    figure, axes = plt.subplots(
        len(row_axes),
        len(column_axes),
        figsize=(figure_width, figure_height),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={
            "width_ratios": width_ratios,
            "height_ratios": height_ratios,
            "wspace": 0.035,
            "hspace": 0.035,
        },
    )
    try:
        for row_index, row_axis in enumerate(row_axes):
            for column_index, column_axis in enumerate(column_axes):
                axis = axes[row_index, column_index]
                matrix = matrices[row_index][column_index]
                contour_options = {
                    "levels": levels,
                    "cmap": color_map,
                    "norm": normalization,
                }
                if filled:
                    axis.contourf(
                        column_axis,
                        row_axis,
                        matrix,
                        extend="both",
                        **contour_options,
                    )
                else:
                    axis.contour(column_axis, row_axis, matrix, **contour_options)

                axis.set_xlim(float(np.max(column_axis)), float(np.min(column_axis)))
                axis.set_ylim(float(np.max(row_axis)), float(np.min(row_axis)))
                axis.tick_params(
                    axis="x",
                    labelbottom=row_index == len(row_axes) - 1,
                    labelsize=8,
                    length=3,
                )
                axis.tick_params(
                    axis="y",
                    labelleft=column_index == 0,
                    labelsize=8,
                    length=3,
                )
                if row_index == 0:
                    axis.set_title(
                        f"{normalized_column_labels[column_index]}\n"
                        f"{_range_description(column_axis)}",
                        fontsize=9.5,
                        color="#202830",
                        fontweight="bold",
                        pad=7,
                    )
                if row_index == len(row_axes) - 1:
                    axis.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=9)
                if column_index == 0:
                    axis.set_ylabel(
                        f"{normalized_row_labels[row_index]}\n"
                        f"{_range_description(row_axis)}\n"
                        r"Wavenumber (cm$^{-1}$)",
                        fontsize=9,
                    )

                if (row_index, column_index) in marked_diagonals:
                    overlap_lower = max(float(np.min(row_axis)), float(np.min(column_axis)))
                    overlap_upper = min(float(np.max(row_axis)), float(np.max(column_axis)))
                    if overlap_lower <= overlap_upper:
                        axis.plot(
                            [overlap_lower, overlap_upper],
                            [overlap_lower, overlap_upper],
                            color="#303840",
                            linewidth=0.75,
                            linestyle="--",
                            alpha=0.75,
                        )

        scalar_mappable = ScalarMappable(norm=normalization, cmap=color_map)
        scalar_mappable.set_array([])
        colorbar = figure.colorbar(
            scalar_mappable,
            ax=axes.ravel().tolist(),
            pad=0.025,
            shrink=0.92,
            extend=colorbar_extend,
        )
        colorbar.set_label("Correlation intensity (shared scale)", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

        display_name = normalized_kind.capitalize()
        figure.suptitle(
            f"{display_name} multi-range 2D-COS | {method} | {convention}",
            x=0.01,
            ha="left",
            fontsize=12,
            color="#202830",
            fontweight="bold",
        )
        figure.text(
            0.01,
            0.002,
            f"Global symmetric color scale ±{limit:.5g} "
            f"({limit_description} across all {len(row_axes) * len(column_axes)} blocks); "
            "matrix values unchanged.",
            fontsize=8,
            color="#505A64",
        )
    except Exception:
        plt.close(figure)
        raise
    return figure


def save_figure(
    figure: Figure,
    output_base: str | Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = DEFAULT_DPI,
) -> dict[str, Path]:
    """Save a figure and always close it, including when saving raises an error."""

    base = Path(output_base)
    if base.suffix.lower() in {".png", ".pdf"}:
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    normalized_formats = tuple(str(fmt).lower().lstrip(".") for fmt in formats)
    unsupported = sorted(set(normalized_formats) - {"png", "pdf"})
    if unsupported:
        plt.close(figure)
        raise ValueError(f"unsupported figure format(s): {', '.join(unsupported)}")

    written: dict[str, Path] = {}
    try:
        for file_format in normalized_formats:
            destination = base.with_suffix(f".{file_format}")
            save_options: dict[str, object] = {"bbox_inches": "tight", "format": file_format}
            if file_format == "png":
                save_options["dpi"] = int(dpi)
            figure.savefig(destination, **save_options)
            written[file_format] = destination
    finally:
        plt.close(figure)
    return written


def plot_raw_spectra_overlay(
    wavenumber: np.ndarray | Sequence[float],
    spectra: np.ndarray | Sequence[Sequence[float]],
    output_base: str | Path,
    *,
    labels: Sequence[str] | None = None,
    intensity_label: str = "Intensity",
    formats: Sequence[str] = ("png", "pdf"),
) -> dict[str, Path]:
    figure = create_spectra_overlay(
        wavenumber,
        spectra,
        labels=labels,
        title="Raw FTIR spectra",
        intensity_label=intensity_label,
    )
    return save_figure(figure, output_base, formats=formats)


def plot_baseline_qc_representative(
    wavenumber: np.ndarray | Sequence[float],
    raw: np.ndarray | Sequence[Sequence[float]],
    baselines: np.ndarray | Sequence[Sequence[float]],
    corrected: np.ndarray | Sequence[Sequence[float]],
    output_base: str | Path,
    *,
    labels: Sequence[str] | None = None,
    intensity_label: str = "Intensity",
    formats: Sequence[str] = ("png", "pdf"),
) -> dict[str, Path]:
    figure = create_baseline_qc_representative(
        wavenumber,
        raw,
        baselines,
        corrected,
        labels=labels,
        intensity_label=intensity_label,
    )
    return save_figure(figure, output_base, formats=formats)


def plot_all_baselines_overlay(
    wavenumber: np.ndarray | Sequence[float],
    baselines: np.ndarray | Sequence[Sequence[float]],
    output_base: str | Path,
    *,
    labels: Sequence[str] | None = None,
    intensity_label: str = "Baseline intensity",
    formats: Sequence[str] = ("png", "pdf"),
) -> dict[str, Path]:
    figure = create_spectra_overlay(
        wavenumber,
        baselines,
        labels=labels,
        title="Estimated baselines",
        intensity_label=intensity_label,
        palette="cividis",
    )
    return save_figure(figure, output_base, formats=formats)


def plot_corrected_spectra_overlay(
    wavenumber: np.ndarray | Sequence[float],
    corrected: np.ndarray | Sequence[Sequence[float]],
    output_base: str | Path,
    *,
    labels: Sequence[str] | None = None,
    intensity_label: str = "Corrected intensity",
    formats: Sequence[str] = ("png", "pdf"),
) -> dict[str, Path]:
    figure = create_spectra_overlay(
        wavenumber,
        corrected,
        labels=labels,
        title="Baseline-corrected FTIR spectra",
        intensity_label=intensity_label,
    )
    return save_figure(figure, output_base, formats=formats)


def plot_dynamic_spectra_overlay(
    wavenumber: np.ndarray | Sequence[float],
    dynamic: np.ndarray | Sequence[Sequence[float]],
    output_base: str | Path,
    *,
    labels: Sequence[str] | None = None,
    intensity_label: str = "Dynamic intensity",
    formats: Sequence[str] = ("png", "pdf"),
) -> dict[str, Path]:
    figure = create_spectra_overlay(
        wavenumber,
        dynamic,
        labels=labels,
        title="Dynamic FTIR spectra",
        intensity_label=intensity_label,
        palette="coolwarm",
    )
    return save_figure(figure, output_base, formats=formats)


def plot_synchronous_contour(
    wavenumber: np.ndarray | Sequence[float],
    matrix: np.ndarray | Sequence[Sequence[float]],
    output_base: str | Path,
    **options: object,
) -> dict[str, Path]:
    formats = options.pop("formats", ("png", "pdf"))
    figure = create_2d_contour(wavenumber, matrix, kind="synchronous", **options)
    return save_figure(figure, output_base, formats=formats)  # type: ignore[arg-type]


def plot_asynchronous_contour(
    wavenumber: np.ndarray | Sequence[float],
    matrix: np.ndarray | Sequence[Sequence[float]],
    output_base: str | Path,
    **options: object,
) -> dict[str, Path]:
    formats = options.pop("formats", ("png", "pdf"))
    figure = create_2d_contour(wavenumber, matrix, kind="asynchronous", **options)
    return save_figure(figure, output_base, formats=formats)  # type: ignore[arg-type]


def plot_multi_range_2d_contour(
    row_wavenumbers: Sequence[np.ndarray | Sequence[float]],
    column_wavenumbers: Sequence[np.ndarray | Sequence[float]],
    block_matrices: Sequence[Sequence[np.ndarray | Sequence[Sequence[float]]]],
    output_base: str | Path,
    *,
    kind: str,
    **options: object,
) -> dict[str, Path]:
    """Create, save, and close one multi-range block contour figure."""

    formats = options.pop("formats", ("png", "pdf"))
    figure = create_multi_range_2d_contour(
        row_wavenumbers,
        column_wavenumbers,
        block_matrices,
        kind=kind,
        **options,
    )
    return save_figure(figure, output_base, formats=formats)  # type: ignore[arg-type]


def plot_multi_range_synchronous_contour(
    row_wavenumbers: Sequence[np.ndarray | Sequence[float]],
    column_wavenumbers: Sequence[np.ndarray | Sequence[float]],
    block_matrices: Sequence[Sequence[np.ndarray | Sequence[Sequence[float]]]],
    output_base: str | Path,
    **options: object,
) -> dict[str, Path]:
    """Save and close a synchronous multi-range block contour figure."""

    return plot_multi_range_2d_contour(
        row_wavenumbers,
        column_wavenumbers,
        block_matrices,
        output_base,
        kind="synchronous",
        **options,
    )


def plot_multi_range_asynchronous_contour(
    row_wavenumbers: Sequence[np.ndarray | Sequence[float]],
    column_wavenumbers: Sequence[np.ndarray | Sequence[float]],
    block_matrices: Sequence[Sequence[np.ndarray | Sequence[Sequence[float]]]],
    output_base: str | Path,
    **options: object,
) -> dict[str, Path]:
    """Save and close an asynchronous multi-range block contour figure."""

    return plot_multi_range_2d_contour(
        row_wavenumbers,
        column_wavenumbers,
        block_matrices,
        output_base,
        kind="asynchronous",
        **options,
    )


__all__ = [
    "DEFAULT_CONTOUR_LEVELS",
    "DEFAULT_DISPLAY_PERCENTILE",
    "DEFAULT_DPI",
    "create_2d_contour",
    "create_baseline_qc_representative",
    "create_multi_range_2d_contour",
    "create_spectra_overlay",
    "plot_all_baselines_overlay",
    "plot_asynchronous_contour",
    "plot_baseline_qc_representative",
    "plot_corrected_spectra_overlay",
    "plot_dynamic_spectra_overlay",
    "plot_multi_range_2d_contour",
    "plot_multi_range_asynchronous_contour",
    "plot_multi_range_synchronous_contour",
    "plot_raw_spectra_overlay",
    "plot_synchronous_contour",
    "save_figure",
]
