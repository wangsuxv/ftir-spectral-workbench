from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import ftir_baseline.baseline.collaborative as collaborative
from ftir_baseline.baseline import (
    collab_pls,
    compose_baselines,
    estimate_coarse,
    estimate_fine,
    estimate_series_baseline,
    shared_shape_baseline,
)


def test_independent_locked_uses_one_recipe_and_preserves_row_order() -> None:
    x = np.linspace(0, 10, 101)
    spectra = np.vstack((4 + x, 1 + 2 * x, 3 - 0.5 * x))

    result = estimate_series_baseline(x, spectra, "linear", "independent_locked")

    np.testing.assert_allclose(result.corrected, 0.0, atol=1e-13)
    np.testing.assert_allclose(result.total_baseline[:, 0], spectra[:, 0], atol=1e-13)
    assert result.params["series_recipe_locked"] is True
    assert result.params["spectrum_order_preserved"] is True


class _FakeCollaborativeBaseline:
    seen_data: np.ndarray | None = None
    seen_method_kwargs: dict[str, Any] | None = None

    def __init__(self, x_data: np.ndarray, **_: Any) -> None:
        self.x_data = x_data

    def collab_pls(
        self,
        data: np.ndarray,
        *,
        average_dataset: bool,
        method: str,
        method_kwargs: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del average_dataset, method
        type(self).seen_data = data.copy()
        type(self).seen_method_kwargs = dict(method_kwargs)
        return 0.1 * data, {
            "average_weights": np.ones(data.shape[1]),
            "method_params": {"tol_history": [np.array([method_kwargs["tol"] / 2])] * len(data)},
        }


def test_collaborative_mode_outputs_every_baseline_without_sorting_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collaborative, "_load_baseline_class", lambda: _FakeCollaborativeBaseline)
    x = np.array([3.0, 2.0, 1.0])
    spectra = np.array([[50, 51, 52], [10, 11, 12], [30, 31, 32]], dtype=float)

    result = collab_pls(x, spectra, "arpls", lambda_=1e4)

    np.testing.assert_allclose(result.total_baseline, 0.1 * spectra)
    assert _FakeCollaborativeBaseline.seen_data is not None
    np.testing.assert_array_equal(_FakeCollaborativeBaseline.seen_data[:, 0], spectra[:, -1])
    assert result.total_baseline.shape == spectra.shape


def test_shared_shape_allows_exactly_affine_per_spectrum_freedom() -> None:
    x = np.linspace(-1, 1, 101)
    common_shape = x**2
    offsets = np.array([5.0, 1.0, 3.0])
    slopes = np.array([0.2, -0.4, 0.1])
    spectra = common_shape[None, :] + offsets[:, None] + slopes[:, None] * x
    anchors = [(-1, -1), (0, 0), (1, 1)]

    result = shared_shape_baseline(x, spectra, "rubberband", anchors=anchors, reference="median")

    np.testing.assert_allclose(result.corrected, 0.0, atol=1e-13)
    assert result.params["degrees_of_freedom_per_spectrum"] == 2
    assert result.params["allowed_per_spectrum_terms"] == (
        "constant",
        "linear_slope",
    )
    # A constant+slope adjustment has zero discrete second derivative.
    np.testing.assert_allclose(np.diff(result.fine_baseline, n=2, axis=1), 0.0, atol=1e-14)
    np.testing.assert_allclose(result.total_baseline[:, 0], spectra[:, 0], atol=1e-13)


def test_composite_preserves_both_components_and_reconstructs_raw() -> None:
    x = np.linspace(0, 10, 101)
    raw = np.vstack((2 + 0.3 * x, 3 - 0.1 * x))
    coarse = estimate_coarse(x, raw, "offset")
    fine = estimate_fine(x, coarse.corrected, "strict_endpoint")

    result = compose_baselines(raw, coarse, fine)

    np.testing.assert_array_equal(result.coarse_baseline, coarse.total_baseline)
    np.testing.assert_array_equal(result.fine_baseline, fine.total_baseline)
    np.testing.assert_allclose(result.total_baseline, result.coarse_baseline + result.fine_baseline)
    np.testing.assert_allclose(raw, result.total_baseline + result.corrected)
    assert result.metrics["reconstruction_max_abs_error"] <= np.finfo(float).eps


def test_shared_shape_dispatch_requires_fixed_anchors() -> None:
    with pytest.raises(ValueError, match="requires fixed anchor"):
        estimate_series_baseline(np.arange(5.0), np.ones((2, 5)), "rubberband", "shared_shape")
