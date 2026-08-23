from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest

import ftir_baseline.baseline.automatic as automatic
from ftir_baseline.baseline.automatic import (
    PybaselinesUnavailableError,
    estimate_coarse,
    linear_detrend_baseline,
    pybaselines_baseline,
    rubberband_baseline,
)


class _FakeBaseline:
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    def __init__(self, x_data: np.ndarray, **_: Any) -> None:
        self.x_data = np.asarray(x_data)

    def _fit(self, name: str, data: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, dict]:
        self.calls.append((name, dict(kwargs)))
        baseline = np.full_like(data, np.min(data))
        return baseline, {"tol_history": np.array([kwargs["tol"] / 2]), "weights": data * 0 + 1}

    def arpls(self, data: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, dict]:
        return self._fit("arpls", data, **kwargs)

    def asls(self, data: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, dict]:
        return self._fit("asls", data, **kwargs)

    def airpls(self, data: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, dict]:
        return self._fit("airpls", data, **kwargs)

    def pspline_arpls(self, data: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, dict]:
        return self._fit("pspline_arpls", data, **kwargs)


@pytest.mark.parametrize("method", ["arpls", "asls", "airpls", "pspline_arpls"])
def test_pybaselines_adapter_shape_locked_recipe_and_complete_params(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    _FakeBaseline.calls.clear()
    monkeypatch.setattr(automatic, "_load_baseline_class", lambda: _FakeBaseline)
    x = np.linspace(1800, 900, 101)
    spectra = np.vstack((np.sin(x / 100) + 4, np.cos(x / 90) + 7))
    params: dict[str, Any] = {"lambda": 12345.0, "max_iter": 17, "tol": 1e-4}
    if method == "asls":
        params["p"] = 0.02

    result = pybaselines_baseline(x, spectra, method, **params)

    assert result.total_baseline.shape == spectra.shape
    assert result.corrected.shape == spectra.shape
    assert len(_FakeBaseline.calls) == spectra.shape[0]
    assert all(call_method == method for call_method, _ in _FakeBaseline.calls)
    assert all(call_params == _FakeBaseline.calls[0][1] for _, call_params in _FakeBaseline.calls)
    assert result.params["lambda"] == 12345.0
    assert result.params["max_iter"] == 17
    assert result.params["tol"] == 1e-4
    if method == "asls":
        assert result.params["p"] == 0.02


def test_nonconvergence_is_recorded_as_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    class Nonconverging(_FakeBaseline):
        def arpls(self, data: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, dict]:
            return np.zeros_like(data), {"tol_history": np.array([kwargs["tol"] * 10])}

    monkeypatch.setattr(automatic, "_load_baseline_class", lambda: Nonconverging)
    result = pybaselines_baseline(np.arange(5.0), np.arange(5.0), "arpls", tol=1e-3)

    assert any("did not converge" in warning for warning in result.warnings)


def test_missing_pybaselines_has_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> type[Any]:
        raise PybaselinesUnavailableError("install pybaselines>=1.2")

    monkeypatch.setattr(automatic, "_load_baseline_class", unavailable)
    with pytest.raises(PybaselinesUnavailableError, match=r"install pybaselines>=1.2"):
        estimate_coarse(np.arange(5.0), np.arange(5.0), "arpls")


def test_linear_detrend_and_rubberband_are_axis_equivalent() -> None:
    x = np.linspace(900, 1800, 181)
    peak = 2 * np.exp(-(((x - 1300) / 40) ** 2))
    spectra = np.vstack((0.3 + 1e-3 * x + peak, -0.5 + 4e-4 * x + peak / 2))

    for function in (linear_detrend_baseline, rubberband_baseline):
        ascending = function(x, spectra)
        descending = function(x[::-1], spectra[:, ::-1])
        np.testing.assert_allclose(
            ascending.total_baseline, descending.total_baseline[:, ::-1], atol=1e-12
        )


def test_rubberband_follows_lower_hull_not_peak_top() -> None:
    x = np.arange(5.0)
    y = np.array([1.0, 10.0, 2.0, 9.0, 3.0])

    result = rubberband_baseline(x, y)

    np.testing.assert_allclose(result.total_baseline, [1, 1.5, 2, 2.5, 3])
    assert result.corrected[1] > 8
