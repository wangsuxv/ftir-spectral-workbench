from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from ftir_baseline.models import thaw_mapping
from ftir_workbench import (
    PostBaselineSmoothingConfig,
    PostBaselineSmoothingResult,
    PostBaselineSmoothingService,
    PreparedSpectralDataset,
    apply_post_baseline_smoothing,
    prepared_from_smoothed_result,
)
from ftir_workbench.export import load_prepared, serialize_prepared
from ftir_workbench.fingerprints import prepared_data_sha256

from ._helpers import make_prepared


def _assert_prepared_equal(
    expected: PreparedSpectralDataset,
    actual: PreparedSpectralDataset,
) -> None:
    np.testing.assert_array_equal(actual.wavenumber, expected.wavenumber)
    np.testing.assert_array_equal(actual.perturbation, expected.perturbation)
    np.testing.assert_array_equal(actual.spectra, expected.spectra)
    assert actual.perturbation_labels == expected.perturbation_labels
    assert actual.intensity_unit == expected.intensity_unit
    assert actual.source_name == expected.source_name
    assert actual.source_sha256 == expected.source_sha256
    assert actual.baseline_run_id == expected.baseline_run_id
    assert actual.baseline_fingerprint == expected.baseline_fingerprint
    assert actual.prepared_data_sha256 == expected.prepared_data_sha256
    assert actual.original_axis_direction == expected.original_axis_direction
    assert actual.current_axis_direction == expected.current_axis_direction
    assert actual.perturbation_order_policy == expected.perturbation_order_policy
    assert actual.normalization_state == expected.normalization_state
    assert thaw_mapping(actual.baseline_recipe) == thaw_mapping(expected.baseline_recipe)
    assert thaw_mapping(actual.baseline_qc) == thaw_mapping(expected.baseline_qc)
    assert actual.warnings == expected.warnings


def test_adapter_creates_immutable_child_and_preserves_parent_baseline_history() -> None:
    parent = make_prepared(
        baseline_recipe={
            "coarse_baseline": {"method": "none"},
            "fine_baseline": {"method": "endpoint_window_linear"},
            "nested_history": {"values": [1, 2, 3]},
            "prepared_data_contract": {
                "source_channel": "PipelineResult.analysis_data",
                "scientific_normalization": False,
            },
        }
    )
    parent_axis = parent.wavenumber.copy()
    parent_spectra = parent.spectra.copy()
    parent_recipe = thaw_mapping(parent.baseline_recipe)
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="gaussian",
        gaussian_sigma_points=1.25,
        gaussian_truncate=3.0,
        convolution_mode="mirror",
    )
    result = apply_post_baseline_smoothing(parent, config)

    child = prepared_from_smoothed_result(result)

    np.testing.assert_array_equal(parent.wavenumber, parent_axis)
    np.testing.assert_array_equal(parent.spectra, parent_spectra)
    assert thaw_mapping(parent.baseline_recipe) == parent_recipe
    np.testing.assert_array_equal(child.wavenumber, parent.wavenumber)
    np.testing.assert_array_equal(child.perturbation, parent.perturbation)
    np.testing.assert_array_equal(child.spectra, result.smoothed_spectra)
    assert child.perturbation_labels == parent.perturbation_labels
    assert child.source_name == parent.source_name
    assert child.source_sha256 == parent.source_sha256
    assert child.baseline_run_id == parent.baseline_run_id
    assert child.baseline_fingerprint == parent.baseline_fingerprint
    assert child.original_axis_direction == parent.original_axis_direction
    assert child.current_axis_direction == parent.current_axis_direction
    assert child.perturbation_order_policy == parent.perturbation_order_policy
    assert child.normalization_state == parent.normalization_state
    assert thaw_mapping(child.baseline_qc) == thaw_mapping(parent.baseline_qc)
    assert not np.shares_memory(child.spectra, result.smoothed_spectra)
    assert not child.spectra.flags.writeable

    expected_hash = prepared_data_sha256(
        parent.wavenumber,
        parent.perturbation,
        parent.perturbation_labels,
        result.smoothed_spectra,
        normalization_state=parent.normalization_state,
    )
    assert child.prepared_data_sha256 == expected_hash
    assert child.prepared_data_sha256 != parent.prepared_data_sha256

    recipe = thaw_mapping(child.baseline_recipe)
    assert recipe["coarse_baseline"] == parent_recipe["coarse_baseline"]
    assert recipe["fine_baseline"] == parent_recipe["fine_baseline"]
    assert recipe["nested_history"] == parent_recipe["nested_history"]
    contract = recipe["prepared_data_contract"]
    assert contract == {
        "source_channel": "parent PreparedSpectralDataset.spectra",
        "branch_kind": "post_baseline_smoothing",
        "parent_prepared_data_sha256": parent.prepared_data_sha256,
        "smoothing_fingerprint": result.smoothing_fingerprint,
        "algorithm": "gaussian",
        "parameters": {"sigma_points": 1.25, "truncate": 3.0, "mode": "mirror"},
        "nonuniform_axis_policy": "error",
    }
    smoothing = recipe["post_baseline_smoothing"]
    assert smoothing["parent_prepared_data_sha256"] == parent.prepared_data_sha256
    assert smoothing["smoothing_fingerprint"] == result.smoothing_fingerprint
    assert smoothing["method"] == "gaussian"
    assert smoothing["parameters"] == contract["parameters"]
    assert smoothing["config"] == config.scientific_dict()
    assert smoothing["nonuniform_axis_policy"] == "error"
    assert smoothing["summary_metrics"] == dict(result.summary_metrics)
    assert set(smoothing["per_spectrum_metrics"]) == set(result.per_spectrum_metrics)
    assert smoothing["warnings"] == list(result.warnings)
    assert smoothing["parent_prepared_data_contract"] == parent_recipe[
        "prepared_data_contract"
    ]
    assert any("post-baseline smoothing scientific branch" in item for item in child.warnings)
    assert all(item in child.warnings for item in result.warnings)


def test_service_preview_and_apply_use_the_same_authoritative_core() -> None:
    parent = make_prepared()
    config = PostBaselineSmoothingConfig(enabled=True, method="savgol")
    calls: list[tuple[PreparedSpectralDataset, PostBaselineSmoothingConfig]] = []

    def recording_core(
        prepared: PreparedSpectralDataset,
        smoothing_config: PostBaselineSmoothingConfig,
    ) -> PostBaselineSmoothingResult:
        calls.append((prepared, smoothing_config))
        return apply_post_baseline_smoothing(prepared, smoothing_config)

    service = PostBaselineSmoothingService(smoothing_core=recording_core)

    preview = service.preview(parent, config)
    applied, child = service.apply(parent, config)

    assert calls == [(parent, config), (parent, config)]
    np.testing.assert_array_equal(preview.smoothed_spectra, applied.smoothed_spectra)
    assert preview.smoothing_fingerprint == applied.smoothing_fingerprint
    np.testing.assert_array_equal(child.spectra, applied.smoothed_spectra)
    assert child.prepared_data_sha256 != parent.prepared_data_sha256


def test_service_apply_requires_enabled_but_disabled_preview_remains_available() -> None:
    parent = make_prepared()
    disabled = PostBaselineSmoothingConfig()
    calls: list[str] = []

    def recording_core(
        prepared: PreparedSpectralDataset,
        config: PostBaselineSmoothingConfig,
    ) -> PostBaselineSmoothingResult:
        calls.append("core")
        return apply_post_baseline_smoothing(prepared, config)

    service = PostBaselineSmoothingService(smoothing_core=recording_core)
    preview = service.preview(parent, disabled)

    np.testing.assert_array_equal(preview.smoothed_spectra, parent.spectra)
    with pytest.raises(ValueError, match="enabled=True"):
        service.apply(parent, disabled)
    assert calls == ["core"]
    with pytest.raises(ValueError, match="requires smoothing to be enabled"):
        prepared_from_smoothed_result(preview)


def test_constant_noop_smoothing_can_create_a_recorded_child_with_same_hash() -> None:
    axis = np.linspace(1800.0, 1000.0, 41)
    spectra = np.vstack(
        (
            np.zeros(axis.size),
            np.full(axis.size, 1.5),
            np.full(axis.size, -2.0),
        )
    )
    parent = make_prepared(wavenumber=axis, spectra=spectra)
    config = PostBaselineSmoothingConfig(
        enabled=True,
        method="moving_average",
        moving_average_window_length=5,
    )

    result, child = PostBaselineSmoothingService().apply(parent, config)

    np.testing.assert_array_equal(result.smoothed_spectra, parent.spectra)
    np.testing.assert_array_equal(child.spectra, parent.spectra)
    assert child.prepared_data_sha256 == parent.prepared_data_sha256
    recipe = thaw_mapping(child.baseline_recipe)
    assert recipe["prepared_data_contract"]["branch_kind"] == (
        "post_baseline_smoothing"
    )
    assert recipe["prepared_data_contract"]["smoothing_fingerprint"] == (
        result.smoothing_fingerprint
    )


def test_display_only_normalization_state_and_baseline_qc_are_preserved() -> None:
    parent = make_prepared(normalization_state="display_only")

    _, child = PostBaselineSmoothingService().apply(
        parent,
        PostBaselineSmoothingConfig(enabled=True, method="gaussian"),
    )

    assert child.normalization_state == "display_only"
    assert child.baseline_fingerprint == parent.baseline_fingerprint
    assert thaw_mapping(child.baseline_qc) == thaw_mapping(parent.baseline_qc)


@pytest.mark.parametrize(
    "parent_factory",
    (
        lambda: make_prepared(normalization_state="scientific_explicit"),
        lambda: make_prepared(
            baseline_recipe={"post_baseline_smoothing": {"config": {}}}
        ),
        lambda: make_prepared(
            baseline_recipe={
                "prepared_data_contract": {
                    "branch_kind": "post_baseline_smoothing"
                }
            }
        ),
    ),
)
def test_service_leaves_scientific_normalization_and_chaining_rejection_to_core(
    parent_factory: Callable[[], PreparedSpectralDataset],
) -> None:
    with pytest.raises(ValueError):
        PostBaselineSmoothingService().apply(
            parent_factory(),
            PostBaselineSmoothingConfig(enabled=True),
        )


def test_smoothed_child_exact_prepared_serialization_round_trip() -> None:
    parent = make_prepared(
        normalization_state="display_only",
        baseline_recipe={
            "coarse": {"method": "none"},
            "nested": {"中文": [1, 2]},
            "prepared_data_contract": {
                "source_channel": "PipelineResult.analysis_data"
            },
        },
    )
    _, child = PostBaselineSmoothingService().apply(
        parent,
        PostBaselineSmoothingConfig(
            enabled=True,
            method="median",
            median_window_length=3,
        ),
    )

    reloaded = load_prepared(serialize_prepared(child))

    _assert_prepared_equal(child, reloaded)
    assert reloaded.baseline_fingerprint == parent.baseline_fingerprint
    assert reloaded.baseline_run_id == parent.baseline_run_id
    assert reloaded.normalization_state == "display_only"
