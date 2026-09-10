#!/usr/bin/env python3
"""Reproduce the three synthetic independent-batch text fixtures; no experimental data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def absorbance(x: np.ndarray, k: int) -> np.ndarray:
    """Deterministic artificial baseline, two peaks, and sinusoidal variation."""
    return (
        0.12 + 0.00002 * (x - 800)
        + (0.3 + 0.08 * k) * np.exp(-((x - 1700 - 35 * k) / 95) ** 2)
        + 0.08 * np.exp(-((x - 2900) / 150) ** 2)
        + 0.002 * np.sin(x / 13 + k)
    )


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(4000, 800, 401)
    np.savetxt(
        output_dir / "synthetic_wide_A.csv",
        np.column_stack((x, *(absorbance(x, k) for k in range(3)))),
        fmt="%.17g", delimiter=",", comments="", header="wavenumber,M1,M2,PHEMA",
    )
    x = np.linspace(900, 3800, 291)
    np.savetxt(
        output_dir / "synthetic_percent_T.csv",
        np.column_stack((x, 100 * 10 ** (-absorbance(x, 0)))),
        fmt="%.17g", delimiter=",", comments="", header="wavenumber,sample_percent_T",
    )
    x = np.linspace(3600, 700, 581)
    np.savetxt(
        output_dir / "synthetic_fraction_T.csv",
        np.column_stack((x, 10 ** (-absorbance(x, 1)))),
        fmt="%.17g", delimiter=",", comments="", header="wavenumber,sample_fraction_T",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    generate(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
