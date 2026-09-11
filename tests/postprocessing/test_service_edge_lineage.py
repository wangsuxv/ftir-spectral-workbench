"""Additional parent-area and preparation interpretation lineage cases."""

from __future__ import annotations

import numpy as np
import pytest

from ftir_workbench.batch import postprocessing as postprocess
from ftir_workbench.batch.models import PreparationConfig
from ftir_workbench.batch.service import preview_coarse
from ftir_workbench.batch.state import (
    confirm_coarse,
    confirm_preparation,
    set_preparation_draft,
    skip_fine,
)
from tests.postprocessing.helpers import confirmed_workspace
from tests.postprocessing.test_state_contracts import DERIVED, SMOOTH, apply_branch, graph


def test_pp039_area_normalization_resolves_each_actual_b_and_s_parent() -> None:
    x = np.arange(9.0)
    y = np.array([0.0, 7.0, 1.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    ws, sid = confirmed_workspace(x, y)
    smoothed = apply_branch(ws, sid, "smoothed", SMOOTH)
    recipe = {"enabled": True, "method": "area", "area_definition": "absolute", "target": 2.0}
    normalized_b = apply_branch(ws, sid, "normalized_baseline", recipe)
    normalized_s = apply_branch(ws, sid, "normalized_smoothed", recipe)
    area_b = float(np.trapezoid(y, x))
    area_s = float(np.trapezoid(smoothed.spectra[0], x))
    assert area_b != area_s
    np.testing.assert_array_equal(normalized_b.scale, [2.0 / area_b])
    np.testing.assert_array_equal(normalized_s.scale, [2.0 / area_s])
    np.testing.assert_array_equal(normalized_b.spectra, y[None, :] * (2.0 / area_b))
    np.testing.assert_array_equal(normalized_s.spectra, smoothed.spectra * (2.0 / area_s))
    assert normalized_b.parent_fingerprint == normalized_b.baseline.fingerprint
    assert normalized_s.parent_fingerprint == smoothed.fingerprint


@pytest.mark.parametrize("change", ["range", "unit"])
def test_pp047_new_confirmed_preparation_and_b_invalidate_only_own_derived_lineage(change: str) -> None:
    x = np.linspace(1800.0, 900.0, 181)
    y = 0.3 + 0.2 * np.exp(-((x - 1400.0) / 35.0) ** 2)
    ws, a = confirmed_workspace(x, y)
    _, b = confirmed_workspace(x, y, workspace=ws, name="same-values-unmodified.csv")
    old_a, old_b = graph(ws, a), graph(ws, b)
    config = ws.states[a].preparation_committed.to_dict()
    if change == "range":
        config["wavenumber_range"] = [1700.0, 1100.0]
    else:
        config["input_unit"] = "fraction_transmittance"
    set_preparation_draft(ws, a, PreparationConfig(**config))
    confirm_preparation(ws, a)
    preview_coarse(ws, a)
    confirm_coarse(ws, a)
    skip_fine(ws, a)
    current_b = postprocess.get_branch(ws, a, "baseline")
    assert current_b.fingerprint != old_a["smoothed"].baseline.fingerprint
    for branch in DERIVED:
        assert postprocess.branch_status(ws, a, branch)["status"] == "stale"
        assert getattr(postprocess.get_postprocessing_state(ws, a), branch).committed is old_a[branch]
        assert postprocess.get_branch(ws, b, branch) is old_b[branch]
    new_s = apply_branch(ws, a, "smoothed", SMOOTH)
    assert new_s.parent_fingerprint == current_b.fingerprint
    assert new_s.fingerprint != old_a["smoothed"].fingerprint
    assert postprocess.branch_status(ws, a, "normalized_baseline")["status"] == "stale"
    assert postprocess.branch_status(ws, a, "normalized_smoothed")["status"] == "stale"
