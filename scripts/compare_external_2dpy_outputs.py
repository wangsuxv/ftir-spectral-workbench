#!/usr/bin/env python3
"""Compare project matrices with CSV files produced by official 2Dpy.

Example::

    python scripts/compare_external_2dpy_outputs.py \
      --ours-sync results/run_x/data/09_synchronous_matrix.csv \
      --ours-async results/run_x/data/10_asynchronous_matrix.csv \
      --external-sync spec_sync.csv \
      --external-async spec_async.csv

Every CSV is expected to have row wavenumbers in its first column and column
wavenumbers in its header.  In addition to direct max-absolute and RMSE
differences, the script checks transpose, sign-reversal, and combined variants.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MatrixCsv:
    """Numeric values plus their row and column labels."""

    values: FloatArray
    rows: tuple[str, ...]
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Difference:
    """Difference summary for one candidate alignment."""

    max_absolute_difference: float
    rmse: float
    matches_tolerance: bool


def read_matrix_csv(path: str | Path) -> MatrixCsv:
    """Read one Origin-ready or 2Dpy matrix CSV without reordering labels."""

    csv_path = Path(path)
    try:
        frame = pd.read_csv(csv_path, index_col=0)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read matrix CSV {csv_path}: {error}") from error
    if frame.empty:
        raise ValueError(f"Matrix CSV is empty: {csv_path}")
    try:
        values = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Matrix CSV contains non-numeric cells: {csv_path}") from error
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"Matrix CSV must be square; got {values.shape} in {csv_path}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Matrix CSV contains NaN or infinite values: {csv_path}")
    return MatrixCsv(
        values=np.array(values, dtype=np.float64, copy=True, order="C"),
        rows=tuple(str(value) for value in frame.index),
        columns=tuple(str(value) for value in frame.columns),
    )


def _difference(
    ours: FloatArray,
    candidate: FloatArray,
    *,
    rtol: float,
    atol: float,
) -> Difference:
    residual = ours - candidate
    return Difference(
        max_absolute_difference=float(np.max(np.abs(residual))),
        rmse=float(np.sqrt(np.mean(np.square(residual), dtype=np.float64))),
        matches_tolerance=bool(np.allclose(ours, candidate, rtol=rtol, atol=atol)),
    )


def compare_matrices(
    ours: MatrixCsv,
    external: MatrixCsv,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    """Compare direct and convention-error candidates and select the lowest RMSE."""

    if ours.values.shape != external.values.shape:
        raise ValueError(
            f"Matrix shapes differ: ours={ours.values.shape}, external={external.values.shape}"
        )

    candidates = {
        "identity": external.values,
        "transpose": external.values.T,
        "sign_reversal": -external.values,
        "transpose_and_sign_reversal": -external.values.T,
    }
    comparisons = {
        name: _difference(ours.values, values, rtol=rtol, atol=atol)
        for name, values in candidates.items()
    }
    best_name = min(comparisons, key=lambda name: comparisons[name].rmse)
    best = comparisons[best_name]
    direct_matches = comparisons["identity"].matches_tolerance
    transpose_detected = not direct_matches and (
        comparisons["transpose"].matches_tolerance
        or comparisons["transpose_and_sign_reversal"].matches_tolerance
    )
    sign_reversal_detected = not direct_matches and (
        comparisons["sign_reversal"].matches_tolerance
        or comparisons["transpose_and_sign_reversal"].matches_tolerance
    )
    return {
        "direct_max_absolute_difference": comparisons["identity"].max_absolute_difference,
        "direct_rmse": comparisons["identity"].rmse,
        "direct_matches_tolerance": direct_matches,
        "best_alignment": best_name,
        "best_max_absolute_difference": best.max_absolute_difference,
        "best_rmse": best.rmse,
        "best_matches_tolerance": best.matches_tolerance,
        "overall_transpose": transpose_detected,
        "overall_sign_reversal": sign_reversal_detected,
        "orientation_sign_ambiguity": transpose_detected and sign_reversal_detected,
        "row_labels_equal_directly": ours.rows == external.rows,
        "column_labels_equal_directly": ours.columns == external.columns,
        "all_candidates": {
            name: {
                "max_absolute_difference": result.max_absolute_difference,
                "rmse": result.rmse,
                "matches_tolerance": result.matches_tolerance,
            }
            for name, result in comparisons.items()
        },
    }


def _print_report(title: str, report: dict[str, Any]) -> None:
    print(f"{title}:")
    print(f"  max absolute difference (direct): {report['direct_max_absolute_difference']:.12g}")
    print(f"  RMSE (direct): {report['direct_rmse']:.12g}")
    print(f"  direct tolerance match: {report['direct_matches_tolerance']}")
    print(f"  best alignment: {report['best_alignment']}")
    print(f"  best max absolute difference: {report['best_max_absolute_difference']:.12g}")
    print(f"  best RMSE: {report['best_rmse']:.12g}")
    print(f"  overall transpose detected: {report['overall_transpose']}")
    print(f"  overall sign reversal detected: {report['overall_sign_reversal']}")
    print(f"  transpose/sign diagnosis is ambiguous: {report['orientation_sign_ambiguity']}")
    print(f"  row labels equal directly: {report['row_labels_equal_directly']}")
    print(f"  column labels equal directly: {report['column_labels_equal_directly']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-sync", type=Path, required=True)
    parser.add_argument("--ours-async", type=Path, required=True)
    parser.add_argument("--external-sync", type=Path, required=True)
    parser.add_argument("--external-async", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not np.isfinite(args.rtol) or args.rtol < 0.0:
        raise SystemExit("--rtol must be finite and non-negative")
    if not np.isfinite(args.atol) or args.atol < 0.0:
        raise SystemExit("--atol must be finite and non-negative")

    synchronous_report = compare_matrices(
        read_matrix_csv(args.ours_sync),
        read_matrix_csv(args.external_sync),
        rtol=args.rtol,
        atol=args.atol,
    )
    asynchronous_report = compare_matrices(
        read_matrix_csv(args.ours_async),
        read_matrix_csv(args.external_async),
        rtol=args.rtol,
        atol=args.atol,
    )
    _print_report("synchronous", synchronous_report)
    _print_report("asynchronous", asynchronous_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
