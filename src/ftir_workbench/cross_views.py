"""Deterministic display/export views of uniquely computed cross 2D-COS blocks.

The helpers in this module never call the numerical 2D-COS core.  They expose
the stored cross orientation and its existing reverse identity, and assemble
complete self/cross block grids for plotting without interpolation or
recalculation.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .config import TwoDCOSRange

if TYPE_CHECKING:
    from .services.twodcos_service import CrossRangeResult, TwoDCOSAnalysisResult

FloatArray = NDArray[np.float64]
CrossOrientation = Literal["stored", "reverse"]
BlockKind = Literal["synchronous", "asynchronous"]


@dataclass(frozen=True, slots=True)
class OrientedCrossView:
    """One actual row/column orientation of a unique cross-range result."""

    pair_index: int
    orientation: CrossOrientation
    row_range: TwoDCOSRange
    column_range: TwoDCOSRange
    row_wavenumber: FloatArray
    column_wavenumber: FloatArray
    synchronous: FloatArray
    asynchronous: FloatArray
    row_variable: str
    column_variable: str

    def __post_init__(self) -> None:
        try:
            pair_index = operator.index(self.pair_index)
        except TypeError as error:
            raise TypeError("pair_index must be an integer") from error
        if isinstance(self.pair_index, (bool, np.bool_)) or pair_index < 1:
            raise ValueError("pair_index must be a positive one-based integer")
        if self.orientation not in {"stored", "reverse"}:
            raise ValueError("orientation must be 'stored' or 'reverse'")

        row_axis = np.asarray(self.row_wavenumber, dtype=np.float64)
        column_axis = np.asarray(self.column_wavenumber, dtype=np.float64)
        synchronous = np.asarray(self.synchronous, dtype=np.float64)
        asynchronous = np.asarray(self.asynchronous, dtype=np.float64)
        if row_axis.ndim != 1 or column_axis.ndim != 1:
            raise ValueError("cross view axes must be one-dimensional")
        expected_shape = (row_axis.size, column_axis.size)
        if synchronous.shape != expected_shape or asynchronous.shape != expected_shape:
            raise ValueError(
                "cross view matrices must match the row and column axes; "
                f"expected {expected_shape}"
            )
        if not (
            np.isfinite(row_axis).all()
            and np.isfinite(column_axis).all()
            and np.isfinite(synchronous).all()
            and np.isfinite(asynchronous).all()
        ):
            raise ValueError("cross view axes and matrices must be finite")

        object.__setattr__(self, "pair_index", pair_index)
        object.__setattr__(self, "row_wavenumber", row_axis)
        object.__setattr__(self, "column_wavenumber", column_axis)
        object.__setattr__(self, "synchronous", synchronous)
        object.__setattr__(self, "asynchronous", asynchronous)


@dataclass(frozen=True, slots=True)
class FullBlockOverview:
    """Plot-ready complete N-by-N grid of convention-oriented 2D-COS blocks."""

    kind: BlockKind
    ranges: tuple[TwoDCOSRange, ...]
    row_wavenumbers: tuple[FloatArray, ...]
    column_wavenumbers: tuple[FloatArray, ...]
    block_matrices: tuple[tuple[FloatArray, ...], ...]
    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]
    diagonal_blocks: frozenset[tuple[int, int]]


def _range_for_variable(item: CrossRangeResult, variable: str) -> TwoDCOSRange:
    if variable == "nu1":
        return item.first_range
    if variable == "nu2":
        return item.second_range
    raise ValueError(
        "cross 2D-COS row/column variables must identify input nu1 or nu2; "
        f"got {variable!r}"
    )


def oriented_cross_views(
    item: CrossRangeResult,
    *,
    pair_index: int,
) -> tuple[OrientedCrossView, OrientedCrossView]:
    """Return stored and reverse views without recomputing a cross pair."""

    from .services.twodcos_service import CrossRangeResult

    if not isinstance(item, CrossRangeResult):
        raise TypeError("item must be a CrossRangeResult")
    core = item.result
    actual_variables = (core.row_variable, core.column_variable)
    if set(actual_variables) != {"nu1", "nu2"}:
        raise ValueError(
            "stored cross axes must contain nu1 and nu2 exactly once; "
            f"got {actual_variables!r}"
        )

    stored = OrientedCrossView(
        pair_index=pair_index,
        orientation="stored",
        row_range=_range_for_variable(item, core.row_variable),
        column_range=_range_for_variable(item, core.column_variable),
        row_wavenumber=core.row_wavenumber,
        column_wavenumber=core.column_wavenumber,
        synchronous=core.synchronous,
        asynchronous=core.asynchronous,
        row_variable=core.row_variable,
        column_variable=core.column_variable,
    )
    reverse = OrientedCrossView(
        pair_index=pair_index,
        orientation="reverse",
        row_range=_range_for_variable(item, core.reverse_row_variable),
        column_range=_range_for_variable(item, core.reverse_column_variable),
        row_wavenumber=core.reverse_row_wavenumber,
        column_wavenumber=core.reverse_column_wavenumber,
        synchronous=core.reverse_synchronous,
        asynchronous=core.reverse_asynchronous,
        row_variable=core.reverse_row_variable,
        column_variable=core.reverse_column_variable,
    )
    return stored, reverse


def _normalized_block_kind(kind: str) -> BlockKind:
    normalized = str(kind).strip().lower()
    if normalized not in {"synchronous", "asynchronous"}:
        raise ValueError("kind must be 'synchronous' or 'asynchronous'")
    return cast(BlockKind, normalized)


def full_block_overview(
    result: TwoDCOSAnalysisResult,
    *,
    kind: BlockKind,
) -> FullBlockOverview:
    """Assemble every self/stored/reverse block in configured range order."""

    from .services.twodcos_service import TwoDCOSAnalysisResult

    if not isinstance(result, TwoDCOSAnalysisResult):
        raise TypeError("result must be a TwoDCOSAnalysisResult")
    normalized_kind = _normalized_block_kind(kind)
    ranges = result.config.ranges
    if len(result.homo_results) != len(ranges):
        raise ValueError("homo results must contain one block per configured range")
    if len(ranges) > 1 and not result.config.cross_range_enabled:
        raise ValueError("a complete multi-range overview requires cross-range results")

    index_by_range = {analysis_range: index for index, analysis_range in enumerate(ranges)}
    if len(index_by_range) != len(ranges):
        raise ValueError("configured 2D ranges must be unique")

    row_axes: list[FloatArray] = []
    column_axes: list[FloatArray] = []
    blocks: list[list[FloatArray | None]] = [
        [None for _ in ranges] for _ in ranges
    ]
    for index, (analysis_range, homo) in enumerate(
        zip(ranges, result.homo_results, strict=True)
    ):
        if homo.analysis_range != analysis_range:
            raise ValueError("homo result order does not match configured range order")
        core = homo.result
        row_axes.append(core.row_wavenumber)
        column_axes.append(core.column_wavenumber)
        blocks[index][index] = getattr(core, normalized_kind)

    expected_pairs = set(combinations(range(len(ranges)), 2))
    seen_pairs: set[tuple[int, int]] = set()
    for pair_index, item in enumerate(result.cross_results, start=1):
        try:
            first_index = index_by_range[item.first_range]
            second_index = index_by_range[item.second_range]
        except KeyError as error:
            raise ValueError("cross result references an unconfigured range") from error
        if first_index == second_index:
            raise ValueError("cross result must reference two distinct ranges")
        pair = (min(first_index, second_index), max(first_index, second_index))
        if pair in seen_pairs:
            raise ValueError("cross results contain a duplicate unique range pair")
        seen_pairs.add(pair)

        for view in oriented_cross_views(item, pair_index=pair_index):
            row_index = index_by_range[view.row_range]
            column_index = index_by_range[view.column_range]
            if blocks[row_index][column_index] is not None:
                raise ValueError("multiple 2D blocks map to the same overview cell")
            if not np.array_equal(view.row_wavenumber, row_axes[row_index]):
                raise ValueError("cross row axis does not match its mapped self range")
            if not np.array_equal(view.column_wavenumber, column_axes[column_index]):
                raise ValueError("cross column axis does not match its mapped self range")
            blocks[row_index][column_index] = getattr(view, normalized_kind)

    if seen_pairs != expected_pairs:
        raise ValueError("cross results must contain every unique configured range pair")
    if any(value is None for row in blocks for value in row):
        raise ValueError("full block overview contains an unfilled cell")

    labels = tuple(analysis_range.display_name for analysis_range in ranges)
    return FullBlockOverview(
        kind=normalized_kind,
        ranges=ranges,
        row_wavenumbers=tuple(row_axes),
        column_wavenumbers=tuple(column_axes),
        block_matrices=tuple(
            tuple(cast(FloatArray, value) for value in row) for row in blocks
        ),
        row_labels=labels,
        column_labels=labels,
        diagonal_blocks=frozenset((index, index) for index in range(len(ranges))),
    )


__all__ = [
    "BlockKind",
    "CrossOrientation",
    "FullBlockOverview",
    "OrientedCrossView",
    "full_block_overview",
    "oriented_cross_views",
]
