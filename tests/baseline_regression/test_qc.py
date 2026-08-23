from __future__ import annotations

import numpy as np
import pytest

from ftir_baseline.qc import run_quality_control
from ftir_baseline.scoring import (
    anchor_residual_error,
    baseline_roughness,
    diagnostic_score,
    negative_residual_fraction,
    rank_candidates,
    reconstruction_check,
    reconstruction_error,
    temporal_roughness,
)


def test_required_metric_formulas() -> None:
    x = np.arange(6.0)
    corrected = np.array([0.0, -4.0, 0.1, -2.9, 0.0, 0.2])
    assert anchor_residual_error(x, corrected, [(0.0, 1.0), (4.0, 5.0)]) == pytest.approx(0.1)
    assert negative_residual_fraction(corrected, noise_sigma=1.0, k=3.0) == pytest.approx(1.0 / 6.0)

    baseline = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])
    assert baseline_roughness(baseline) == pytest.approx(4.0)


def test_temporal_roughness_and_reconstruction() -> None:
    baseline = np.array([[0.0, 0.0], [1.0, 1.0], [4.0, 4.0]])
    assert temporal_roughness(baseline) == pytest.approx(4.0)
    corrected = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    raw = baseline + corrected
    assert reconstruction_check(raw, baseline, corrected)
    np.testing.assert_array_equal(reconstruction_error(raw, baseline, corrected), 0.0)


def test_qc_retains_time_order_and_reports_all_required_metrics() -> None:
    x = np.linspace(900.0, 1800.0, 101)
    p = np.array([20.0, 10.0, 30.0])  # deliberately not sorted
    peak = np.exp(-0.5 * ((x - 1300.0) / 35.0) ** 2)
    baseline = np.stack([0.1 + 0.0001 * x, 0.11 + 0.0001 * x, 0.12 + 0.0001 * x])
    corrected = np.stack([peak, 1.1 * peak, 1.2 * peak])
    raw = baseline + corrected

    result = run_quality_control(
        x,
        raw,
        baseline,
        corrected,
        anchor_windows=[(900.0, 950.0), (1750.0, 1800.0)],
        peak_regions=[(1200.0, 1400.0)],
        perturbation=p,
    )

    np.testing.assert_array_equal(result.per_spectrum["perturbation"], p)
    for key in (
        "anchor_error",
        "noise_sigma",
        "negative_fraction",
        "baseline_roughness",
        "peak_position_shift",
        "reconstruction_error",
    ):
        assert result.per_spectrum[key].shape == (3,)
    assert result.summary["time_order_preserved"] is True
    assert result.summary["reconstruction_passed"] is True
    assert result.summary["maximum_reconstruction_error"] < 1e-14
    assert "not proof" in result.summary["diagnostic_score_disclaimer"]


def test_qc_detects_but_does_not_remove_temporal_jump() -> None:
    x = np.linspace(900.0, 1800.0, 40)
    baseline = np.zeros((8, 40))
    baseline[4:] = 2.0
    corrected = np.tile(np.sin(x / 100.0), (8, 1))
    raw = baseline + corrected

    result = run_quality_control(x, raw, baseline, corrected)

    assert result.per_spectrum["spectrum_index"].size == 8
    assert result.summary["temporal_jump_indices"]
    assert any("no spectra were removed or reordered" in warning for warning in result.warnings)


def test_diagnostic_ranking_is_optional_and_component_preserving() -> None:
    candidates = {
        "rough": {"anchor_error": 2.0, "baseline_roughness": 5.0},
        "smooth": {"anchor_error": 1.0, "baseline_roughness": 0.5},
    }
    assert diagnostic_score(candidates["smooth"]) < diagnostic_score(candidates["rough"])
    ranked = rank_candidates(candidates)
    assert [item.name for item in ranked] == ["smooth", "rough"]
    assert ranked[0].metrics is candidates["smooth"]


def test_anchor_window_without_samples_has_clear_error() -> None:
    with pytest.raises(ValueError, match="contains no data points"):
        anchor_residual_error(np.arange(5.0), np.ones(5), [(10.0, 11.0)])
