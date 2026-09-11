"""Service integration of interval recipes, display quantities and rounding guards."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ftir_workbench.batch.postprocessing import (
    confirm_branch,
    display_arrays,
    get_branch,
    preview_branch,
    update_draft,
)
from tests.postprocessing.helpers import confirmed_workspace


@pytest.mark.parametrize("method", ["internal_peak_height", "internal_peak_area", "area"])
def test_reference_recipe_survives_immutable_json_conversion(method: str) -> None:
    ws, sid = confirmed_workspace()
    interval_key = "area_interval" if method == "area" else "reference_interval"
    update_draft(ws, sid, "normalized_baseline", {
        "enabled": True, "method": method, interval_key: [1600, 1400],
    })
    result = preview_branch(ws, sid, "normalized_baseline")
    confirm_branch(ws, sid, "normalized_baseline")
    assert get_branch(ws, sid, "normalized_baseline").fingerprint == result.fingerprint
    np.testing.assert_array_equal(result.wavenumber, ws.records[sid].wavenumber)


def test_minmax_large_offset_uses_stable_core_expression_for_validation() -> None:
    x = np.arange(9, dtype=float)
    y = 1e8 + np.linspace(0, .1234567, 9)
    ws, sid = confirmed_workspace(x, y)
    update_draft(ws, sid, "normalized_baseline", {"enabled": True, "method": "minmax_display"})
    result = preview_branch(ws, sid, "normalized_baseline")
    confirm_branch(ws, sid, "normalized_baseline")
    parent = get_branch(ws, sid, "baseline").result.analysis_data
    expected = (parent - parent.min()) * (1 / (parent.max() - parent.min()))
    np.testing.assert_array_equal(result.spectra, expected)
    assert result.purpose == "display_only"


@pytest.mark.parametrize("method", ["maximum", "minmax_display", "vector"])
@pytest.mark.parametrize("unit", ["fraction_transmittance", "percent_transmittance"])
def test_pp041_normalized_quantities_cannot_be_shown_as_physical_transmittance(method: str, unit: str) -> None:
    ws, sid = confirmed_workspace()
    update_draft(ws, sid, "normalized_baseline", {"enabled": True, "method": method})
    result = preview_branch(ws, sid, "normalized_baseline")
    with pytest.raises(ValueError, match="NORMALIZED_UNIT_INVALID"):
        display_arrays(result, unit)
    _, values, label = display_arrays(result)
    np.testing.assert_array_equal(values, result.spectra)
    assert "intensity" in label.lower()


@pytest.mark.parametrize("field", ["scale", "offset", "removed_component"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1j])
def test_snapshot_rejects_nonfinite_or_complex_auxiliary_arrays(field: str, value: complex) -> None:
    ws, sid = confirmed_workspace()
    update_draft(ws, sid, "normalized_baseline", {"enabled": True, "method": "maximum"})
    result = preview_branch(ws, sid, "normalized_baseline")
    shape = result.spectra.shape if field == "removed_component" else (1,)
    with pytest.raises(ValueError, match="POSTPROCESS_INTEGRITY_FAILED"):
        replace(result, **{field: np.full(shape, value)})
