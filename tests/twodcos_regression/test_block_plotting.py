from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ftir2dcos.plotting import (
    create_multi_range_2d_contour,
    plot_multi_range_asynchronous_contour,
)


def _block_fixture() -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    list[list[np.ndarray]],
]:
    low = np.linspace(1140.0, 1250.0, 4)
    high = np.linspace(1600.0, 1736.0, 5)
    middle = np.linspace(1300.0, 1350.0, 3)
    row_axes = (low, high)
    column_axes = (high, middle, low)
    blocks: list[list[np.ndarray]] = []
    for row_index, row_axis in enumerate(row_axes):
        block_row: list[np.ndarray] = []
        for column_index, column_axis in enumerate(column_axes):
            values = np.arange(row_axis.size * column_axis.size, dtype=np.float64).reshape(
                row_axis.size, column_axis.size
            )
            values = (values - values.mean()) * (row_index + column_index + 1)
            block_row.append(values)
        blocks.append(block_row)
    blocks[1][1][-1, -1] = 1000.0
    return row_axes, column_axes, blocks


def test_multi_range_grid_supports_rectangular_blocks_and_one_global_scale() -> None:
    row_axes, column_axes, blocks = _block_fixture()
    originals = [[matrix.copy() for matrix in row] for row in blocks]
    percentile = 80.0

    figure = create_multi_range_2d_contour(
        row_axes,
        column_axes,
        blocks,
        kind="synchronous",
        row_labels=("Low", "High"),
        column_labels=("High", "Middle", "Low"),
        convention="2dpy_compatible",
        contour_levels=13,
        display_percentile=percentile,
        diagonal_blocks={(0, 2), (1, 0)},
    )
    try:
        data_axes = figure.axes[:6]
        assert len(figure.axes) == 7  # six data panels and one shared colorbar
        assert all(axis.xaxis_inverted() for axis in data_axes)
        assert all(axis.yaxis_inverted() for axis in data_axes)
        assert [len(axis.lines) for axis in data_axes] == [0, 0, 1, 1, 0, 0]
        assert "Synchronous multi-range 2D-COS" in figure._suptitle.get_text()
        assert "2dpy_compatible" in figure._suptitle.get_text()
        assert "shared scale" in figure.axes[-1].get_ylabel()
        figure_notes = [text.get_text() for text in figure.texts]
        assert any("across all 6 blocks" in note for note in figure_notes)
        assert any("matrix values unchanged" in note for note in figure_notes)

        pooled = np.concatenate(
            [np.abs(matrix).ravel() for matrix_row in blocks for matrix in matrix_row]
        )
        expected_limit = float(np.percentile(pooled, percentile))
        for axis in data_axes:
            assert axis.collections[0].norm.vmin == pytest.approx(-expected_limit)
            assert axis.collections[0].norm.vmax == pytest.approx(expected_limit)

        assert all(not data_axes[index].get_xlabel() for index in range(3))
        assert all(data_axes[index].get_xlabel() for index in range(3, 6))
        assert data_axes[0].get_ylabel()
        assert data_axes[3].get_ylabel()
        assert all(not data_axes[index].get_ylabel() for index in (1, 2, 4, 5))
        assert "High" in data_axes[0].get_title()
        assert "Middle" in data_axes[1].get_title()
        assert "Low" in data_axes[2].get_title()

        figure.canvas.draw()
        first_column_width = data_axes[0].get_position().width
        second_column_width = data_axes[1].get_position().width
        assert first_column_width > second_column_width

        for original_row, plotted_row in zip(originals, blocks, strict=True):
            for original, plotted in zip(original_row, plotted_row, strict=True):
                assert np.array_equal(original, plotted)
    finally:
        plt.close(figure)


def test_diagonal_inference_supports_reversed_row_and_column_range_order() -> None:
    low = np.linspace(1140.0, 1250.0, 4)
    high = np.linspace(1600.0, 1736.0, 5)
    row_axes = (low, high)
    column_axes = (high, low)
    blocks = [
        [np.ones((4, 5)), np.eye(4)],
        [np.eye(5), np.ones((5, 4))],
    ]

    figure = create_multi_range_2d_contour(
        row_axes,
        column_axes,
        blocks,
        kind="asynchronous",
        display_percentile=None,
    )
    try:
        assert [len(axis.lines) for axis in figure.axes[:4]] == [0, 1, 1, 0]
        assert figure.axes[0].get_xlim() == pytest.approx((1736.0, 1600.0))
        assert figure.axes[0].get_ylim() == pytest.approx((1250.0, 1140.0))
    finally:
        plt.close(figure)


def test_explicit_empty_diagonal_set_disables_safe_inference() -> None:
    axis = np.linspace(1000.0, 1100.0, 4)
    figure = create_multi_range_2d_contour(
        (axis,),
        (axis,),
        [[np.eye(4)]],
        kind="synchronous",
        diagonal_blocks=(),
    )
    try:
        assert not figure.axes[0].lines
    finally:
        plt.close(figure)


def test_ambiguous_duplicate_axes_require_explicit_diagonal_blocks() -> None:
    axis = np.linspace(1000.0, 1100.0, 3)
    blocks = [[np.eye(3), np.eye(3)], [np.eye(3), np.eye(3)]]
    figure = create_multi_range_2d_contour(
        (axis, axis.copy()),
        (axis.copy(), axis.copy()),
        blocks,
        kind="synchronous",
    )
    try:
        assert all(not data_axis.lines for data_axis in figure.axes[:4])
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("row_axes", "column_axes", "blocks", "message"),
    [
        ((), (np.array([2.0, 1.0]),), [], "at least one"),
        (
            (np.array([2.0, 1.0]),),
            (np.array([4.0, 3.0]),),
            [],
            "one row per row axis",
        ),
        (
            (np.array([2.0, 1.0]),),
            (np.array([4.0, 3.0]), np.array([6.0, 5.0])),
            [[np.zeros((2, 2))]],
            "must contain 2 blocks",
        ),
        (
            (np.array([2.0, 1.0]),),
            (np.array([4.0, 3.0]),),
            [[np.zeros((3, 2))]],
            "must have shape",
        ),
    ],
)
def test_multi_range_grid_rejects_invalid_grid_shapes(
    row_axes: tuple[np.ndarray, ...],
    column_axes: tuple[np.ndarray, ...],
    blocks: list[list[np.ndarray]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_multi_range_2d_contour(
            row_axes,
            column_axes,
            blocks,
            kind="synchronous",
        )


def test_multi_range_grid_rejects_nonmonotonic_nonfinite_and_complex_inputs() -> None:
    monotonic = np.array([3.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="strictly monotonic"):
        create_multi_range_2d_contour(
            (np.array([3.0, 1.0, 2.0]),),
            (monotonic,),
            [[np.zeros((3, 3))]],
            kind="synchronous",
        )

    nonfinite = np.zeros((3, 3))
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        create_multi_range_2d_contour(
            (monotonic,),
            (monotonic,),
            [[nonfinite]],
            kind="synchronous",
        )

    with pytest.raises(TypeError, match="real values"):
        create_multi_range_2d_contour(
            (monotonic,),
            (monotonic,),
            [[np.ones((3, 3), dtype=np.complex128)]],
            kind="synchronous",
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"kind": "phase"}, "kind must"),
        ({"kind": "synchronous", "contour_levels": 1}, "at least 2"),
        ({"kind": "synchronous", "display_percentile": 0.0}, "display_percentile"),
        ({"kind": "synchronous", "row_labels": ("A", "B")}, "row_labels"),
        ({"kind": "synchronous", "diagonal_blocks": {(2, 0)}}, "out of bounds"),
    ],
)
def test_multi_range_grid_rejects_invalid_options(
    options: dict[str, object],
    message: str,
) -> None:
    axis = np.array([3.0, 2.0, 1.0])
    with pytest.raises(ValueError, match=message):
        create_multi_range_2d_contour((axis,), (axis,), [[np.eye(3)]], **options)


def test_multi_range_save_wrapper_is_headless_writes_and_closes(tmp_path: Path) -> None:
    axis = np.linspace(1000.0, 1100.0, 4)
    before = set(plt.get_fignums())
    paths = plot_multi_range_asynchronous_contour(
        (axis,),
        (axis,),
        [[np.eye(4)]],
        tmp_path / "multi_range_async",
        formats=("png",),
        diagonal_blocks=(),
    )

    assert matplotlib.get_backend().lower() == "agg"
    assert paths["png"].read_bytes().startswith(b"\x89PNG")
    assert set(plt.get_fignums()) == before
