"""The naming release must not alter code included in scientific identities."""
from __future__ import annotations

import subprocess
from pathlib import Path

BASE = "14a514d7032f5f7c302f4f47d79a72d895251c1d"
ROOT = Path(__file__).resolve().parents[2]


def test_n22_naming_does_not_change_scientific_implementation_or_lock() -> None:
    paths = [
        "artifacts/v0.2.1_science_freeze_manifest.json", "requirements.lock",
        "src/ftir_workbench/batch/models.py", "src/ftir_workbench/batch/importing.py",
        "src/ftir_workbench/batch/fingerprints.py", "src/ftir_workbench/batch/recipes.py",
        "src/ftir_workbench/batch/service.py", "src/ftir_workbench/batch/state.py",
        "src/ftir_workbench/batch/postprocessing.py",
        "src/ftir_workbench/batch/normalization_adapter.py",
        "src/ftir_workbench/post_baseline_smoothing.py",
        ".github/workflows/tests.yml",
    ]
    for root in ("src/ftir_baseline", "src/ftir2dcos", "src/ftir_workbench/services"):
        paths.extend(str(path.relative_to(ROOT)) for path in (ROOT / root).rglob("*.py"))
    for name in paths:
        old = subprocess.run(["git", "show", f"{BASE}:{name}"], cwd=ROOT, check=True, capture_output=True).stdout
        assert (ROOT / name).read_bytes() == old, name
