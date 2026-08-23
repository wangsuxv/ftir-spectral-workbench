from __future__ import annotations

import numpy as np

from ftir_baseline.gallery import (
    CandidateSpec,
    default_candidate_specs,
    scan_baseline_candidates,
    starter_pchip_anchor_windows,
)


def test_gallery_preserves_components_and_produces_finite_ranking() -> None:
    x = np.linspace(1800.0, 900.0, 301)
    baseline = 0.02 + 2e-5 * (1800.0 - x)
    peak = 0.15 * np.exp(-0.5 * ((x - 1300.0) / 22.0) ** 2)
    spectra = np.vstack((baseline + peak, baseline + 1.2 * peak))
    anchors = [
        {"start": 1796.0, "end": 1804.0},
        {"start": 896.0, "end": 904.0},
    ]
    gallery = scan_baseline_candidates(
        x,
        spectra,
        (
            CandidateSpec("Endpoint", fine_method="endpoint_window_linear"),
            CandidateSpec("arPLS", coarse_method="arpls", coarse_params={"lam": 1e5}),
        ),
        anchor_windows=anchors,
    )
    assert len(gallery.evaluations) == 2
    assert len(gallery.ranking) == 2
    assert all(np.isfinite(item.score) for item in gallery.ranking)
    assert all("diagnostic_score_disclaimer" in item.qc.summary for item in gallery.evaluations)
    assert np.allclose(
        gallery.representative_spectrum,
        gallery.evaluations[0].result.total_baseline + gallery.evaluations[0].result.corrected,
    )


def test_default_gallery_includes_inspectable_anchor_pchip() -> None:
    x = np.linspace(1800.0, 900.0, 301)
    anchors = starter_pchip_anchor_windows(x)
    specs = default_candidate_specs(
        anchor_windows=anchors,
        arpls_log10_lambda=(),
        asls_log10_lambda=(),
        airpls_log10_lambda=(),
    )

    pchip = next(spec for spec in specs if spec.name == "Anchor PCHIP")
    assert pchip.fine_method == "pchip"
    assert len(pchip.fine_params["anchors"]) >= 3
    assert anchors[0]["start"] > anchors[-1]["start"]


def test_starter_pchip_windows_remain_disjoint_on_narrow_range() -> None:
    x = np.linspace(908.0, 900.0, 17)
    anchors = starter_pchip_anchor_windows(x, endpoint_window_width_cm1=8.0)
    spectra = 0.01 + 0.001 * (x - x.min())[None, :]

    gallery = scan_baseline_candidates(
        x,
        spectra,
        (
            CandidateSpec(
                "Anchor PCHIP",
                fine_method="pchip",
                fine_params={"anchors": anchors},
            ),
        ),
        anchor_windows=anchors,
    )

    assert gallery.evaluations[0].qc.summary["reconstruction_passed"] is True
