#!/usr/bin/env python3
"""Write deterministic synthetic ordinary FTIR inputs into an ignored output directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/ordinary-postprocessing-synthetic"))
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3101)
    x = np.linspace(1900.0, 900.0, 201)
    y = .08 + .8 * np.exp(-.5*((x-1650)/35)**2) + .35 * np.exp(-.5*((x-1210)/22)**2)
    a = y + rng.normal(0, .008, x.size)
    b = .04 + y * 1.6 + rng.normal(0, .012, x.size)
    axis = np.linspace(900.0, 1900.0, 251)
    c = .04 + .5 * np.exp(-.5*((axis-1630)/45)**2) + .2 * np.exp(-.5*((axis-1250)/30)**2)
    c += rng.normal(0, .004, axis.size)
    examples = {
        "synthetic_wide.csv": (np.column_stack((x, a, b)), "Wavenumber_cm-1,synthetic_A,synthetic_B"),
        "synthetic_other_axis.csv": (np.column_stack((axis, c)), "Wavenumber_cm-1,synthetic_C"),
        "synthetic_spike.csv": (np.column_stack((np.arange(9), [0, 0, 0, 0, 9, 0, 0, 0, 0])), "Wavenumber_cm-1,synthetic_spike"),
    }
    if any((directory / name).exists() for name in (*examples, "synthetic_manifest.json")):
        raise SystemExit("Output files already exist; choose a new directory to preserve prior work.")
    for name, (data, header) in examples.items():
        np.savetxt(directory / name, data, delimiter=",", header=header, comments="", fmt="%.17g")
    (directory / "synthetic_manifest.json").write_text(json.dumps({
        "synthetic_only": True, "seed": 3101, "input_unit": "absorbance",
        "files": list(examples), "note": "Mathematical examples; no instrument performance claim.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(examples)} synthetic CSV files in {directory}")


if __name__ == "__main__":
    main()
