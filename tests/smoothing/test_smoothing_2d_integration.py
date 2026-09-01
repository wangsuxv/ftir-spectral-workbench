from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

import ftir_workbench.post_baseline_smoothing as smoothing_core_module
from ftir_baseline.models import thaw_mapping
from ftir_workbench import (
    PostBaselineSmoothingConfig,
    PostBaselineSmoothingService,
    PreparedSpectralDataset,
    TwoDCOSConfig,
    TwoDCOSRange,
    TwoDCOSWorkflowService,
)
from ftir_workbench.export import (
    build_twodcos_bundle,
    load_prepared,
    verify_twodcos_bundle,
)

from ._helpers import make_prepared


def _two_range_config() -> TwoDCOSConfig:
    return TwoDCOSConfig(
        ranges=(
            TwoDCOSRange(1800.0, 1450.0, "upper"),
            TwoDCOSRange(1350.0, 1000.0, "lower"),
        ),
        convention="canonical",
        nonuniform_perturbation_policy="allow",
        cross_range_enabled=True,
    )


def _smoothed_child() -> tuple[PreparedSpectralDataset, PreparedSpectralDataset]:
    parent = make_prepared()
    _, child = PostBaselineSmoothingService().apply(
        parent,
        PostBaselineSmoothingConfig(
            enabled=True,
            method="gaussian",
            gaussian_sigma_points=1.25,
            gaussian_truncate=3.0,
        ),
    )
    return parent, child


def test_creating_child_does_not_change_primary_prepared_only_2d_results() -> None:
    parent = make_prepared()
    config = _two_range_config()
    service = TwoDCOSWorkflowService()
    before = service.compute(parent, config)

    PostBaselineSmoothingService().apply(
        parent,
        PostBaselineSmoothingConfig(enabled=True, method="savgol"),
    )
    after = service.compute(parent, config)

    assert before.twodcos_fingerprint == after.twodcos_fingerprint
    assert before.parent_prepared_data_sha256 == parent.prepared_data_sha256
    for first, second in zip(before.homo_results, after.homo_results, strict=True):
        np.testing.assert_array_equal(first.dynamic, second.dynamic)
        np.testing.assert_array_equal(first.synchronous, second.synchronous)
        np.testing.assert_array_equal(first.asynchronous, second.asynchronous)
    for first, second in zip(before.cross_results, after.cross_results, strict=True):
        np.testing.assert_array_equal(first.synchronous, second.synchronous)
        np.testing.assert_array_equal(first.asynchronous, second.asynchronous)


def test_smoothed_child_runs_self_and_cross_2d_with_separate_fingerprints() -> None:
    parent, child = _smoothed_child()
    config = _two_range_config()
    service = TwoDCOSWorkflowService()

    primary_result = service.compute(parent, config)
    smoothed_result = service.compute(child, config)

    assert child.prepared_data_sha256 != parent.prepared_data_sha256
    assert smoothed_result.parent_prepared_data_sha256 == child.prepared_data_sha256
    assert smoothed_result.parent_baseline_run_id == parent.baseline_run_id
    assert smoothed_result.parent_baseline_fingerprint == parent.baseline_fingerprint
    assert smoothed_result.twodcos_fingerprint != primary_result.twodcos_fingerprint
    assert len(smoothed_result.homo_results) == 2
    assert len(smoothed_result.cross_results) == 1
    assert all(
        item.parent_prepared_data_sha256 == child.prepared_data_sha256
        for item in smoothed_result.homo_results
    )
    assert all(
        item.first_parent_prepared_data_sha256 == child.prepared_data_sha256
        and item.second_parent_prepared_data_sha256 == child.prepared_data_sha256
        for item in smoothed_result.cross_results
    )
    assert any(
        not np.array_equal(primary.synchronous, smoothed.synchronous)
        for primary, smoothed in zip(
            primary_result.homo_results,
            smoothed_result.homo_results,
            strict=True,
        )
    )

    cross = smoothed_result.cross_results[0].result
    np.testing.assert_array_equal(cross.reverse_synchronous, cross.synchronous.T)
    np.testing.assert_array_equal(cross.reverse_asynchronous, -cross.asynchronous.T)
    assert bool(cross.qc_metrics["sync_reverse_transpose_ok"])
    assert bool(cross.qc_metrics["async_reverse_negative_transpose_ok"])
    assert bool(cross.qc_metrics["all_checks_passed"])


def test_same_child_and_2d_config_are_deterministic() -> None:
    _, child = _smoothed_child()
    config = _two_range_config()
    service = TwoDCOSWorkflowService()

    first = service.compute(child, config)
    second = service.compute(child, config)

    assert first.twodcos_fingerprint == second.twodcos_fingerprint
    for left, right in zip(first.homo_results, second.homo_results, strict=True):
        np.testing.assert_array_equal(left.dynamic, right.dynamic)
        np.testing.assert_array_equal(left.synchronous, right.synchronous)
        np.testing.assert_array_equal(left.asynchronous, right.asynchronous)
    for left, right in zip(first.cross_results, second.cross_results, strict=True):
        np.testing.assert_array_equal(left.synchronous, right.synchronous)
        np.testing.assert_array_equal(left.asynchronous, right.asynchronous)


def test_twodcos_bundle_embeds_exact_smoothed_prepared_and_rejects_stale_parent() -> None:
    parent, child = _smoothed_child()
    analysis = TwoDCOSWorkflowService().compute(child, _two_range_config())

    bundle = build_twodcos_bundle(child, analysis)

    assert verify_twodcos_bundle(bundle)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        embedded_csv = archive.read("source_prepared_spectrum.csv")
        embedded_metadata = archive.read("source_prepared_spectrum.meta.json")
    embedded = load_prepared((embedded_csv, embedded_metadata))
    np.testing.assert_array_equal(embedded.wavenumber, child.wavenumber)
    np.testing.assert_array_equal(embedded.perturbation, child.perturbation)
    np.testing.assert_array_equal(embedded.spectra, child.spectra)
    assert embedded.perturbation_labels == child.perturbation_labels
    assert embedded.prepared_data_sha256 == child.prepared_data_sha256
    assert embedded.baseline_run_id == child.baseline_run_id
    assert embedded.baseline_fingerprint == child.baseline_fingerprint
    assert embedded.source_sha256 == child.source_sha256
    assert embedded.original_axis_direction == child.original_axis_direction
    assert embedded.current_axis_direction == child.current_axis_direction
    assert embedded.perturbation_order_policy == child.perturbation_order_policy
    assert embedded.normalization_state == child.normalization_state
    assert embedded.warnings == child.warnings
    assert thaw_mapping(embedded.baseline_recipe) == thaw_mapping(child.baseline_recipe)
    assert thaw_mapping(embedded.baseline_qc) == thaw_mapping(child.baseline_qc)

    with pytest.raises(ValueError, match="does not match the supplied prepared dataset"):
        build_twodcos_bundle(parent, analysis)


def test_existing_prepared_only_2d_service_never_calls_smoothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, child = _smoothed_child()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("prepared-only 2D service must not call smoothing")

    monkeypatch.setattr(
        smoothing_core_module,
        "apply_post_baseline_smoothing",
        forbidden,
    )
    monkeypatch.setattr(PostBaselineSmoothingService, "preview", forbidden)
    monkeypatch.setattr(PostBaselineSmoothingService, "apply", forbidden)

    result = TwoDCOSWorkflowService().compute(child, _two_range_config())

    assert result.parent_prepared_data_sha256 == child.prepared_data_sha256
    assert len(result.homo_results) == 2
    assert len(result.cross_results) == 1
