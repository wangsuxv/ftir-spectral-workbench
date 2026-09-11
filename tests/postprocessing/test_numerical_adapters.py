"""PP-009 through PP-041: exact core reuse and ordinary numerical semantics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest
from scipy.integrate import trapezoid

import ftir_baseline.pipeline as baseline_pipeline
import ftir_workbench.batch.normalization_adapter as normalization_module
import ftir_workbench.post_baseline_smoothing as smoothing_module
from ftir_baseline.normalization import apply_normalization
from ftir_workbench.batch.models import BatchError
from ftir_workbench.batch.normalization_adapter import (
    NUMERICAL_GUARD_VERSION,
    OrdinaryNormalizationConfig,
    normalize_config,
    normalize_spectral_arrays,
    validate_normalization_request,
)
from ftir_workbench.post_baseline_smoothing import (
    PostBaselineSmoothingConfig,
    apply_post_baseline_smoothing,
    effective_smoothing_config,
    smooth_spectral_arrays,
    validate_smoothing_request,
)
from tests.smoothing._helpers import make_prepared


def normalized(y: Any, method: str = "maximum", *, x: Any = None, **settings: Any) -> Any:
    matrix = np.asarray(y, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    axis = np.arange(matrix.shape[1], dtype=float) if x is None else np.asarray(x)
    return normalize_spectral_arrays(axis, matrix, {"enabled": True, "method": method, **settings})


@pytest.mark.parametrize("method", ["savgol", "gaussian", "moving_average", "median"])
def test_pp009_four_methods_match_existing_prepared_result_and_qc(method: str) -> None:
    prepared = make_prepared()
    config = PostBaselineSmoothingConfig(enabled=True, method=method)
    expected = apply_post_baseline_smoothing(prepared, config)
    result = smooth_spectral_arrays(prepared.wavenumber, prepared.spectra, config)
    np.testing.assert_array_equal(result.wavenumber, prepared.wavenumber)
    np.testing.assert_array_equal(result.smoothed_spectra, expected.smoothed_spectra)
    np.testing.assert_array_equal(result.removed_component, expected.removed_component)
    assert result.summary_metrics == expected.summary_metrics
    assert result.approximate_physical_width == expected.approximate_physical_width
    assert result.warnings == expected.warnings
    for key, metric in expected.per_spectrum_metrics.items():
        np.testing.assert_array_equal(result.per_spectrum_metrics[key], metric)
    alone = smooth_spectral_arrays(prepared.wavenumber, prepared.spectra[1:2], config)
    # SciPy's SG edge polynomial solve may round differently for one vs many
    # right-hand sides. The ordinary service always calls the one-row contract.
    np.testing.assert_allclose(
        alone.smoothed_spectra[0], result.smoothed_spectra[1], rtol=1e-14, atol=1e-14
    )


def test_pp010_disabled_smoothing_is_detached_immutable_identity() -> None:
    x = np.array([10.0, 9.0, 6.0, 2.0])
    y = np.array([[0.0, -0.3, 1.0, 0.5]])
    before = y.copy()
    result = smooth_spectral_arrays(
        x, y, {"enabled": False, "method": None, "savgol_window_length": False}
    )
    np.testing.assert_array_equal(result.smoothed_spectra, before)
    np.testing.assert_array_equal(result.removed_component, np.zeros_like(before))
    assert result.config.scientific_dict() == {"enabled": False}
    assert result.approximate_physical_width == {}
    x[0] = 100
    y[:] = 999
    assert result.wavenumber[0] == 10
    np.testing.assert_array_equal(result.smoothed_spectra, before)
    for array in (
        result.wavenumber,
        result.input_spectra,
        result.smoothed_spectra,
        result.removed_component,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        result.config = PostBaselineSmoothingConfig()  # type: ignore[misc]
    with pytest.raises(ValueError, match="disabled smoothing"):
        replace(result, smoothed_spectra=before + 1, removed_component=np.full_like(before, -1))


@pytest.mark.parametrize(
    "settings",
    [
        {"savgol_window_length": 4},
        {"savgol_polyorder": 7},
        {"savgol_window_length": True},
        {"savgol_window_length": np.bool_(True)},
        {"savgol_window_length": 7.0},
        {"savgol_mode": "wrap"},
    ],
)
def test_pp011_savgol_invalid_active_controls_rejected(settings: dict[str, Any]) -> None:
    with pytest.raises((ValueError, TypeError)):
        smooth_spectral_arrays(
            np.arange(9), np.ones((1, 9)), {"enabled": True, "method": "savgol", **settings}
        )


@pytest.mark.parametrize(
    ("method", "field"),
    [
        ("savgol", "savgol_window_length"),
        ("moving_average", "moving_average_window_length"),
        ("median", "median_window_length"),
    ],
)
def test_pp012_active_window_not_silently_shrunk(method: str, field: str) -> None:
    with pytest.raises(ValueError, match=r"window_length \(7\).*n_wavenumbers \(5\)"):
        smooth_spectral_arrays(
            np.arange(5), np.ones((1, 5)), {"enabled": True, "method": method, field: 7}
        )


@pytest.mark.parametrize("field", ["gaussian_sigma_points", "gaussian_truncate"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True])
def test_pp013_gaussian_invalid_active_controls_rejected(field: str, value: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        effective_smoothing_config({"enabled": True, "method": "gaussian", field: value})


@pytest.mark.parametrize("method", ["savgol", "gaussian", "moving_average", "median"])
def test_pp014_axis_reversal_preserves_old_tolerance(method: str) -> None:
    prepared = make_prepared()
    config = {"enabled": True, "method": method}
    forward = smooth_spectral_arrays(prepared.wavenumber, prepared.spectra, config)
    reverse = smooth_spectral_arrays(prepared.wavenumber[::-1], prepared.spectra[:, ::-1], config)
    np.testing.assert_array_equal(reverse.wavenumber, prepared.wavenumber[::-1])
    np.testing.assert_allclose(
        forward.smoothed_spectra, reverse.smoothed_spectra[:, ::-1], rtol=1e-14, atol=1e-14
    )
    for key, metric in forward.per_spectrum_metrics.items():
        np.testing.assert_allclose(
            metric, reverse.per_spectrum_metrics[key], rtol=1e-12, atol=1e-12
        )


def test_pp015_016_nonuniform_axis_requires_explicit_index_space_policy() -> None:
    x = np.array([10.0, 9, 8, 6, 5, 4, 3, 2, 1])
    y = np.sin(x)[None, :]
    with pytest.raises(ValueError, match="NONUNIFORM_AXIS"):
        smooth_spectral_arrays(x, y, {"enabled": True})
    result = smooth_spectral_arrays(
        x,
        y,
        {
            "enabled": True,
            "nonuniform_axis_policy": "allow_index_space_with_warning",
        },
    )
    np.testing.assert_array_equal(result.wavenumber, x)
    assert any(
        "Index-space" in warning and "not resampled" in warning for warning in result.warnings
    )
    assert (
        result.config.scientific_dict()["nonuniform_axis_policy"]
        == "allow_index_space_with_warning"
    )
    assert result.spacing_relative_max_deviation == 1.0


@pytest.mark.parametrize(
    ("x", "y"),
    [
        ([0, 1, 2], [[0, np.nan, 1]]),
        ([0, 1, 2], [[0, np.inf, 1]]),
        ([0, np.nan, 2], [[0, 1, 2]]),
        ([0, np.inf, 2], [[0, 1, 2]]),
        ([0, 0, 2], [[0, 1, 2]]),
        ([0, 2, 1], [[0, 1, 2]]),
        ([0, 1, 2], [[0, 1j, 2]]),
        ([0, 1j, 2], [[0, 1, 2]]),
        ([0, 1, 2], [0, 1, 2]),
        ([0, 1, 2], [[0, 1]]),
        ([0], [[1]]),
        ([[0, 1]], [[0, 1]]),
        ([0, 1], np.empty((0, 2))),
    ],
)
def test_pp017_invalid_array_contract_rejected_without_repair(x: Any, y: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        smooth_spectral_arrays(x, y, {"enabled": False})
    with pytest.raises(BatchError, match="NORMALIZATION_INPUT_INVALID"):
        normalize_spectral_arrays(x, y, {"enabled": False})


@pytest.mark.parametrize("spacing", [1.0, 2.0])
def test_pp019_physical_width_uses_actual_spacing(spacing: float) -> None:
    x = np.arange(9) * spacing
    result = smooth_spectral_arrays(x, np.arange(9)[None, :], {"enabled": True})
    assert result.approximate_physical_width["span_cm1"] == 6 * spacing
    gaussian = smooth_spectral_arrays(
        x, np.arange(9)[None, :], {"enabled": True, "method": "gaussian"}
    )
    assert gaussian.approximate_physical_width["sigma_cm1"] == spacing
    assert gaussian.approximate_physical_width["fwhm_cm1"] == 2.35482 * spacing


def test_pp020_021_022_removed_component_median_warning_and_unclipped_outputs() -> None:
    x = np.arange(9)
    y = np.array([[1, 0, 0, 0, 9, 0, 0, 0, 1]], dtype=float)
    median = smooth_spectral_arrays(x, y, {"enabled": True, "method": "median"})
    assert any(
        "nonlinear" in warning and "genuine narrow peaks" in warning for warning in median.warnings
    )
    result = smooth_spectral_arrays(x, y, {"enabled": True})
    np.testing.assert_array_equal(result.removed_component, y - result.smoothed_spectra)
    assert result.smoothed_spectra.min() < 0
    assert result.smoothed_spectra[0, 0] != 0
    assert result.smoothed_spectra[0, 0] != y[0, 0]
    assert all("SNR" not in warning and "true noise" not in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("y", "expected"),
    [
        ([0, 2, 4], [0, 0.5, 1]),
        ([-1, 1, 3], [-1 / 3, 1 / 3, 1]),
        ([-0.3, 0.1, 0.2], [-1.5, 0.5, 1]),
    ],
)
def test_pp023_025_026_positive_maximum_reads_scientific_branch(
    y: list[float], expected: list[float]
) -> None:
    result = normalized(y)
    np.testing.assert_allclose(result.normalized_spectra[0], expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_array_equal(result.input_spectra[0], y)
    assert result.reference_details["height_definition"] == "positive_maximum_not_absolute"
    assert result.reference_details["reference_values"] == (max(y),)
    assert result.offset[0] == 0
    assert result.quantity == "normalized_intensity"


def test_pp024_minmax_reads_view_and_records_complete_affine_transform() -> None:
    result = normalized([-1, 1, 3], "minmax_display")
    np.testing.assert_array_equal(result.normalized_spectra, [[0, 0.5, 1]])
    assert result.scale[0] == 0.25
    assert result.offset[0] == 0.25
    assert result.purpose == "display_only"
    assert result.quantity == "minmax_scaled_intensity"
    assert result.reference_details["minimum"] == (-1.0,)
    assert result.reference_details["maximum"] == (3.0,)
    assert result.reference_details["span"] == (4.0,)
    np.testing.assert_array_equal(
        result.scale[:, None] * result.input_spectra + result.offset[:, None],
        result.normalized_spectra,
    )


@pytest.mark.parametrize("y", [[0, 0, 0], [-3, -1, -2], [-3, 0, -2]])
def test_pp027_nonpositive_height_rejected(y: list[float]) -> None:
    with pytest.raises(BatchError, match="NORMALIZATION_REFERENCE_TOO_SMALL"):
        normalized(y)


def test_pp028_l2_analytic_example_and_zero_guard() -> None:
    result = normalized([3, 4], "vector")
    np.testing.assert_allclose(result.normalized_spectra, [[0.6, 0.8]], rtol=1e-15, atol=0)
    assert result.scale[0] == 0.2
    with pytest.raises(BatchError, match="NORMALIZATION_REFERENCE_TOO_SMALL"):
        normalized([0, 0], "vector")
    assert any("sampling density" in warning for warning in result.warnings)


@pytest.mark.parametrize("reverse", [False, True])
def test_pp029_030_real_x_area_not_sample_sum_and_direction_invariant(reverse: bool) -> None:
    x = np.array([1000, 1002, 1004])
    y = np.array([0, 1, 0])
    if reverse:
        x, y = x[::-1], y[::-1]
    result = normalized(y, "area", x=x)
    np.testing.assert_array_equal(result.normalized_spectra, [[0, 0.5, 0]])
    np.testing.assert_array_equal(result.wavenumber, x)
    assert result.scale[0] == 0.5
    assert result.reference_details["reference_values"] == (2.0,)


@pytest.mark.parametrize("method", ["area", "internal_peak_area"])
def test_pp031_absolute_and_signed_area_keep_original_output_signs(method: str) -> None:
    y = [-1, 2, 1]
    kwargs = {"reference_interval": [0, 2]} if method == "internal_peak_area" else {}
    absolute = normalized(y, method, **kwargs)
    signed = normalized(y, method, area_definition="signed", **kwargs)
    assert absolute.reference_details["reference_values"] == (3.0,)
    assert signed.reference_details["reference_values"] == (2.0,)
    np.testing.assert_allclose(absolute.normalized_spectra, np.asarray([y]) / 3)
    np.testing.assert_allclose(signed.normalized_spectra, np.asarray([y]) / 2)
    assert absolute.normalized_spectra[0, 0] < 0
    assert signed.normalized_spectra[0, 0] < 0


@pytest.mark.parametrize("y", [[-1, 1, -1], [-1, 1, -1 + 1e-15], [-2, 1, -2]])
def test_pp032_signed_area_zero_negative_or_rounding_cancellation_rejected(y: list[float]) -> None:
    with pytest.raises(BatchError, match="NORMALIZATION_REFERENCE_TOO_SMALL"):
        normalized(y, "area", area_definition="signed")


@pytest.mark.parametrize("method", ["internal_peak_height", "internal_peak_area", "area"])
def test_pp033_partial_outside_reference_interval_rejected(method: str) -> None:
    field = "area_interval" if method == "area" else "reference_interval"
    with pytest.raises(BatchError, match="NORMALIZATION_INTERVAL_OUT_OF_BOUNDS"):
        normalized([1, 2, 3], method, x=[1000, 1002, 1004], **{field: [999, 1003]})


def test_pp034_request_bounds_actual_samples_and_at_least_two_points() -> None:
    with pytest.raises(BatchError, match="NORMALIZATION_INSUFFICIENT_REFERENCE_POINTS"):
        normalized([1, 2, 3, 4], "internal_peak_height", x=[0, 2, 4, 6], reference_interval=[1, 3])
    result = normalized(
        [1, 2, 3, 4], "internal_peak_height", x=[0, 2, 4, 6], reference_interval=[5, 1]
    )
    assert result.reference_details["requested_bounds"] == (1.0, 5.0)
    assert result.reference_details["actual_bounds"] == (2.0, 4.0)
    assert result.reference_details["point_count"] == 2
    assert (
        result.reference_details["selection_rule"] == "inclusive_existing_samples_no_interpolation"
    )
    np.testing.assert_array_equal(result.wavenumber, [0, 2, 4, 6])
    np.testing.assert_allclose(result.normalized_spectra, [[1 / 3, 2 / 3, 1, 4 / 3]])


def test_pp035_reference_interval_controls_denominator_not_output_domain() -> None:
    x = np.arange(4000, 399, -10, dtype=float)
    y = np.ones_like(x)
    y[np.flatnonzero(x == 1650)] = 2
    y[np.flatnonzero(x == 3000)] = 10
    result = normalized(y, "internal_peak_height", x=x, reference_interval=[1700, 1600])
    assert result.scale[0] == 0.5
    assert result.normalized_spectra.shape == (1, x.size)
    np.testing.assert_array_equal(result.wavenumber, x)
    np.testing.assert_array_equal(result.normalized_spectra[0], y / 2)


@pytest.mark.parametrize(
    "method", ["maximum", "internal_peak_height", "internal_peak_area", "area", "vector"]
)
def test_pp036_active_target_is_applied(method: str) -> None:
    kwargs = {"reference_interval": [0, 2]} if method.startswith("internal_") else {}
    one = normalized([1, 2, 3], method, **kwargs)
    two = normalized([1, 2, 3], method, target=2, **kwargs)
    np.testing.assert_array_equal(two.normalized_spectra, one.normalized_spectra * 2)


def test_pp036_minmax_target_is_inactive_and_effective_target_fixed_to_one() -> None:
    raw = {"enabled": True, "method": "minmax_display", "target": "unfinished editor"}
    config = normalize_config(raw)
    assert config.target == 1
    assert "target" not in config.scientific_dict()
    assert raw["target"] == "unfinished editor"
    np.testing.assert_array_equal(
        normalized([-1, 1, 3], "minmax_display", target=2).normalized_spectra, [[0, 0.5, 1]]
    )


def test_pp037_scale_relative_reference_floor_not_fixed_absolute_epsilon() -> None:
    small = normalized([1e-200, 2e-200, 1e-200])
    np.testing.assert_allclose(small.normalized_spectra, [[0.5, 1, 0.5]])
    assert small.reference_details["numerical_guard_version"] == NUMERICAL_GUARD_VERSION
    assert small.reference_details["numerical_floor"][0] == 64 * np.finfo(float).eps * 2e-200
    with pytest.raises(BatchError, match="NORMALIZATION_REFERENCE_TOO_SMALL"):
        normalized([1.0, 1e-16, 2e-16], "internal_peak_height", reference_interval=[1, 2])
    with pytest.raises(BatchError, match="NORMALIZATION_REFERENCE_TOO_SMALL"):
        normalized([0.01, 0.02], minimum_reference=0.02)
    with pytest.raises(BatchError, match="NORMALIZATION_REFERENCE_TOO_SMALL"):
        normalized([1.0, 1.0 + 1e-15], "minmax_display")


@pytest.mark.parametrize(
    ("method", "y", "x"),
    [
        ("maximum", [1e-320, 2e-320], None),
        ("vector", [1e200, 1e200], None),
        ("vector", [1e-200, 1e-200], None),
        ("minmax_display", [-1e308, 1e308], None),
        ("area", [1.0, 1.0], [-1e308, 1e308]),
    ],
)
def test_pp037_core_extremes_fail_with_clear_numerical_error(method: str, y: Any, x: Any) -> None:
    with pytest.raises(BatchError, match="NORMALIZATION_NUMERICAL_ERROR"):
        normalized(y, method, x=x)


def test_pp038_reference_cv_zero_is_not_reported_as_cross_sample_evidence() -> None:
    result = normalized([1, 2, 1], "internal_peak_height", reference_interval=[0, 2])
    assert result.reference_details["core_parameters"]["reference_cv"] == 0.0
    assert any("cannot establish reference stability" in warning for warning in result.warnings)
    assert any("not such evidence" in warning for warning in result.warnings)
    assert all("perturbation series" not in warning for warning in result.warnings)


def test_pp039_040_normalize_exact_selected_parent_in_correct_order() -> None:
    x, b = np.arange(5), np.array([[0.0, 0, 9, 0, 0]])
    s = smooth_spectral_arrays(x, b, {"enabled": True, "method": "moving_average"})
    n_b = normalize_spectral_arrays(x, b, {"enabled": True, "method": "maximum"})
    n_s = normalize_spectral_arrays(x, s.smoothed_spectra, {"enabled": True, "method": "maximum"})
    np.testing.assert_array_equal(s.smoothed_spectra, [[0, 3, 3, 3, 0]])
    np.testing.assert_array_equal(n_s.normalized_spectra, [[0, 1, 1, 1, 0]])
    assert n_b.scale[0] == 1 / 9
    assert n_s.scale[0] == 1 / 3
    wrong_order = smooth_spectral_arrays(
        x, n_b.normalized_spectra, {"enabled": True, "method": "moving_average"}
    )
    assert not np.array_equal(n_s.normalized_spectra, wrong_order.smoothed_spectra)
    np.testing.assert_array_equal(b, [[0, 0, 9, 0, 0]])


@pytest.mark.parametrize(
    "method",
    ["maximum", "internal_peak_height", "internal_peak_area", "area", "vector", "minmax_display"],
)
def test_all_normalization_methods_match_public_core_and_affine_evidence(method: str) -> None:
    x = np.array([1000, 1002, 1005, 1007, 1010], dtype=float)
    y = np.array([[-0.1, 0.4, 0.6, 0.5, 0.2], [0.2, 0.5, 1.0, 0.7, 0.3]])
    kwargs = {"reference_interval": [1001, 1009]} if method.startswith("internal_") else {}
    result = normalized(y, method, x=x, **kwargs)
    core_method = "internal_peak_height" if method == "maximum" else method
    core_kwargs: dict[str, Any] = {
        "method": core_method,
        "target": 1,
        "use_absolute": method in {"area", "internal_peak_area"},
    }
    if method == "area":
        core_kwargs["interval"] = [1000, 1010]
    elif method == "maximum":
        core_kwargs["reference_interval"] = [1000, 1010]
    elif method.startswith("internal_"):
        core_kwargs["reference_interval"] = [1001, 1009]
    core = apply_normalization(x, y, core_kwargs)
    expected = core.view_data if method == "minmax_display" else core.optional_normalized
    np.testing.assert_array_equal(result.normalized_spectra, expected)
    np.testing.assert_array_equal(result.scale, core.factors)
    np.testing.assert_allclose(
        result.scale[:, None] * y + result.offset[:, None], expected, rtol=1e-14, atol=1e-14
    )
    # Adding an unrelated second row does not change the first row's denominator.
    alone = normalized(y[:1], method, x=x, **kwargs)
    np.testing.assert_array_equal(alone.normalized_spectra[0], result.normalized_spectra[0])
    assert np.all(result.scale > 0)
    for array in (
        result.wavenumber,
        result.input_spectra,
        result.normalized_spectra,
        result.scale,
        result.offset,
        *result.metrics.values(),
    ):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(TypeError):
        result.reference_details["tampered"] = True  # type: ignore[index]


def test_none_normalization_is_explicitly_disabled_identity() -> None:
    y = np.array([[-3.0, 0, 2]])
    config = {"enabled": True, "method": "none", "target": np.nan, "reference_interval": "bad"}
    result = normalize_spectral_arrays(np.arange(3), y, config)
    assert result.config.scientific_dict() == {"enabled": False}
    assert result.quantity == "absorbance"
    assert result.purpose == "identity"
    np.testing.assert_array_equal(result.normalized_spectra, y)
    np.testing.assert_array_equal(result.scale, [1])
    np.testing.assert_array_equal(result.offset, [0])
    assert not np.shares_memory(result.normalized_spectra, y)


def test_pp050_effective_config_ignores_inactive_invalid_fields_but_preserves_raw_draft() -> None:
    raw = {
        "enabled": True,
        "method": "moving_average",
        "savgol_polyorder": "unfinished",
        "gaussian_sigma_points": np.nan,
        "median_window_length": False,
    }
    effective = effective_smoothing_config(raw)
    assert (
        effective.scientific_dict()
        == PostBaselineSmoothingConfig(enabled=True, method="moving_average").scientific_dict()
    )
    assert raw["savgol_polyorder"] == "unfinished"
    assert np.isnan(raw["gaussian_sigma_points"])
    with pytest.raises(TypeError):
        PostBaselineSmoothingConfig(
            enabled=True, method="moving_average", savgol_polyorder="unfinished"
        )  # type: ignore[arg-type]
    raw_normalize = {
        "enabled": True,
        "method": "vector",
        "reference_interval": [None],
        "area_definition": "unfinished",
    }
    assert (
        normalize_config(raw_normalize).scientific_dict()
        == OrdinaryNormalizationConfig(enabled=True, method="vector").scientific_dict()
    )
    assert raw_normalize["reference_interval"] == [None]
    for config in (effective, normalize_config(raw_normalize)):
        factory = (
            effective_smoothing_config
            if isinstance(config, PostBaselineSmoothingConfig)
            else normalize_config
        )
        assert factory(config.to_dict()).scientific_dict() == config.scientific_dict()


@pytest.mark.parametrize(
    "settings",
    [
        {"enabled": 1},
        {"method": "unknown"},
        {"target": 0},
        {"target": -1},
        {"target": np.nan},
        {"target": True},
        {"minimum_reference": -1},
        {"minimum_reference": np.inf},
        {"extra": 1},
    ],
)
def test_invalid_active_normalization_config(settings: dict[str, Any]) -> None:
    with pytest.raises(BatchError, match="NORMALIZATION_CONFIG_INVALID"):
        normalize_config({"enabled": True, "method": "maximum", **settings})


@pytest.mark.parametrize("interval", [None, [1], [1, 1], [0, np.inf], [True, 2], "0,2", [[0], [2]]])
def test_invalid_reference_interval_config(interval: Any) -> None:
    with pytest.raises(BatchError, match="NORMALIZATION_INTERVAL_INVALID"):
        normalize_config(
            {"enabled": True, "method": "internal_peak_height", "reference_interval": interval}
        )


def test_array_facades_never_construct_prepared_or_call_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Array postprocessing must not create Prepared or run baseline")

    monkeypatch.setattr(baseline_pipeline, "run_pipeline", forbidden)
    monkeypatch.setattr(smoothing_module, "PreparedSpectralDataset", forbidden)
    x, y = np.arange(9), np.array([[0, 0, 0, 1, 3, 1, 0, 0, 0]], dtype=float)
    result = smooth_spectral_arrays(x, y, {"enabled": True, "method": "gaussian"})
    calls: list[dict[str, Any]] = []
    original = normalization_module.apply_normalization

    def tracked(axis: Any, spectra: Any, config: Any) -> Any:
        calls.append(config)
        return original(axis, spectra, config)

    monkeypatch.setattr(normalization_module, "apply_normalization", tracked)
    final = normalize_spectral_arrays(
        x, result.smoothed_spectra, {"enabled": True, "method": "area"}
    )
    assert len(calls) == 1
    np.testing.assert_allclose(trapezoid(final.normalized_spectra, x=x, axis=1), [1.0])


def test_smoothing_extreme_values_fail_without_nonfinite_result() -> None:
    with pytest.raises(ValueError, match="SMOOTHING_NUMERICAL_ERROR"):
        smooth_spectral_arrays(
            np.arange(9), np.full((1, 9), 1e308), {"enabled": True, "method": "gaussian"}
        )


def test_target_request_validation_never_filters_normalizes_or_computes_qc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("request validation must not apply a filter, normalization or baseline")

    monkeypatch.setattr(smoothing_module, "_apply_filter", forbidden)
    monkeypatch.setattr(smoothing_module, "_compute_qc", forbidden)
    monkeypatch.setattr(smoothing_module, "PreparedSpectralDataset", forbidden)
    monkeypatch.setattr(normalization_module, "apply_normalization", forbidden)
    monkeypatch.setattr(baseline_pipeline, "run_pipeline", forbidden)
    x, y = np.arange(9.0), np.ones((1, 9))
    assert validate_smoothing_request(x, y, {"enabled": True}) is None
    for method in ("maximum", "internal_peak_height", "internal_peak_area", "area", "vector"):
        assert (
            validate_normalization_request(
                x,
                y,
                {
                    "enabled": True,
                    "method": method,
                    "reference_interval": [2, 5],
                },
            )
            is None
        )
    assert (
        validate_normalization_request(
            x,
            np.arange(9.0)[None, :],
            {
                "enabled": True,
                "method": "minmax_display",
                "target": "unused invalid draft",
            },
        )
        is None
    )
    assert validate_smoothing_request(x, y, {"enabled": False, "method": None}) is None
    assert validate_normalization_request(x, y, {"enabled": False, "method": None}) is None
    np.testing.assert_array_equal(x, np.arange(9.0))
    np.testing.assert_array_equal(y, np.ones((1, 9)))


@pytest.mark.parametrize(
    ("x", "config", "message"),
    [
        (np.arange(5.0), {"enabled": True}, "window_length"),
        (
            np.arange(5.0),
            {"enabled": True, "method": "moving_average", "moving_average_window_length": 7},
            "window_length",
        ),
        (
            np.arange(5.0),
            {"enabled": True, "method": "median", "median_window_length": 7},
            "window_length",
        ),
        (np.array([0.0, 1, 2, 4, 5, 6, 7, 8, 9]), {"enabled": True}, "NONUNIFORM_AXIS"),
    ],
)
def test_smoothing_request_validates_each_targets_axis_and_window(
    x: Any,
    config: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_smoothing_request(x, np.ones((1, len(x))), config)


@pytest.mark.parametrize(
    ("y", "config", "message"),
    [
        (
            [1, 2, 3],
            {"method": "internal_peak_height", "reference_interval": [-1, 2]},
            "OUT_OF_BOUNDS",
        ),
        ([1, 2, 3], {"method": "area", "area_interval": [-1, 2]}, "OUT_OF_BOUNDS"),
        (
            [1, 2, 3],
            {"method": "internal_peak_area", "reference_interval": [0.5, 1.5]},
            "INSUFFICIENT_REFERENCE_POINTS",
        ),
        ([0, 0, 0], {"method": "vector"}, "REFERENCE_TOO_SMALL"),
        ([-2, -1, -2], {"method": "maximum"}, "REFERENCE_TOO_SMALL"),
        ([-1, 1, -1], {"method": "area", "area_definition": "signed"}, "REFERENCE_TOO_SMALL"),
        ([1, 1, 1], {"method": "minmax_display"}, "REFERENCE_TOO_SMALL"),
    ],
)
def test_normalization_request_validates_target_range_samples_and_reference_guards(
    y: Any,
    config: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(BatchError, match=message):
        validate_normalization_request(np.arange(3.0), [y], {"enabled": True, **config})
