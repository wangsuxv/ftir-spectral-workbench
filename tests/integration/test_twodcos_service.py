"""Integration tests for the prepared-only 2D-COS boundary."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ftir2dcos.peak_order import PeakRequest
from ftir2dcos.twodcos import compute_2dcos, compute_cross_2dcos
from ftir_workbench.config import TwoDCOSConfig, TwoDCOSRange
from ftir_workbench.fingerprints import prepared_data_sha256
from ftir_workbench.models import PreparedSpectralDataset
from ftir_workbench.services.twodcos_service import (
    CrossPreparedConfirmationRequired,
    TwoDCOSWorkflowService,
    analyze_peak_order,
    compute_cross_from_prepared,
    compute_homo_from_prepared,
)
from ftir_workbench.validation import PreparedDatasetValidationError


def _prepared(
    *,
    perturbation: np.ndarray | None = None,
    labels: tuple[str, ...] | None = None,
    spectra: np.ndarray | None = None,
    source_name: str = "原始数据/样品 A",
    source_sha256: str = "1" * 64,
    baseline_fingerprint: str = "2" * 64,
    baseline_run_id: str = "baseline-run-1",
) -> PreparedSpectralDataset:
    axis = np.array([1800.0, 1700.0, 1600.0, 1500.0, 1300.0, 1200.0, 1100.0])
    if spectra is None:
        spectra = np.array(
            [
                [0.10, 0.30, 0.15, 0.20, 0.05, 0.12, 0.04],
                [0.14, 0.34, 0.18, 0.23, 0.07, 0.16, 0.05],
                [0.21, 0.39, 0.25, 0.29, 0.11, 0.19, 0.09],
                [0.31, 0.46, 0.36, 0.35, 0.17, 0.24, 0.15],
            ],
            dtype=np.float64,
        )
    perturbation = (
        np.array([0.0, 1.0, 3.0, 6.0], dtype=np.float64)
        if perturbation is None
        else np.asarray(perturbation, dtype=np.float64)
    )
    labels = labels or tuple(f"{value:g} min" for value in perturbation)
    digest = prepared_data_sha256(axis, perturbation, labels, spectra)
    return PreparedSpectralDataset(
        wavenumber=axis,
        perturbation=perturbation,
        perturbation_labels=labels,
        spectra=spectra,
        intensity_unit="absorbance",
        source_name=source_name,
        source_sha256=source_sha256,
        baseline_run_id=baseline_run_id,
        baseline_fingerprint=baseline_fingerprint,
        prepared_data_sha256=digest,
        original_axis_direction="descending",
        current_axis_direction="descending",
        perturbation_order_policy="preserve_file_order",
        baseline_recipe={"coarse": "none", "fine": "endpoint"},
        baseline_qc={"all_checks_passed": True},
        warnings=(),
    )


def _two_range_config(**changes: object) -> TwoDCOSConfig:
    values: dict[str, object] = {
        "ranges": (
            TwoDCOSRange(1800.0, 1500.0, "amide"),
            TwoDCOSRange(1300.0, 1100.0, "fingerprint"),
        ),
        "convention": "canonical",
        "nonuniform_perturbation_policy": "warn",
        "cross_range_enabled": True,
    }
    values.update(changes)
    return TwoDCOSConfig(**values)


def test_single_range_matches_scientific_core_and_records_parent() -> None:
    prepared = _prepared()
    config = TwoDCOSConfig(
        ranges=(TwoDCOSRange(1800.0, 1500.0, "amide"),),
        convention="canonical",
        nonuniform_perturbation_policy="allow",
    )

    result = compute_homo_from_prepared(prepared, config)
    expected = compute_2dcos(
        prepared.spectra[:, :4],
        prepared.wavenumber[:4],
        convention="canonical",
    )

    np.testing.assert_array_equal(result.synchronous, expected.synchronous)
    np.testing.assert_array_equal(result.asynchronous, expected.asynchronous)
    np.testing.assert_array_equal(result.dynamic, expected.dynamic)
    assert result.parent_baseline_run_id == prepared.baseline_run_id
    assert result.parent_baseline_fingerprint == prepared.baseline_fingerprint
    assert result.parent_prepared_data_sha256 == prepared.prepared_data_sha256
    assert result.all_checks_passed is True


def test_multi_range_computes_independent_self_blocks_and_one_rectangle() -> None:
    prepared = _prepared()
    config = _two_range_config()

    result = TwoDCOSWorkflowService().compute(prepared, config)

    assert len(result.homo_results) == 2
    assert result.self_results == result.homo_results
    assert len(result.cross_results) == 1
    assert result.homo_results[0].result.synchronous.shape == (4, 4)
    assert result.homo_results[1].result.synchronous.shape == (3, 3)
    cross = result.cross_results[0]
    assert cross.different_prepared_blocks is False
    assert cross.different_prepared_blocks_confirmed is False
    assert cross.result.canonical_synchronous.shape == (4, 3)
    expected = compute_cross_2dcos(
        prepared.spectra[:, :4],
        prepared.spectra[:, 4:],
        prepared.wavenumber[:4],
        prepared.wavenumber[4:],
        convention="canonical",
    )
    np.testing.assert_array_equal(cross.synchronous, expected.synchronous)
    np.testing.assert_array_equal(cross.asynchronous, expected.asynchronous)
    assert "non-equally-spaced" in " ".join(result.warnings)


def test_cross_requires_pointwise_values_labels_and_spectrum_count() -> None:
    first = _prepared()
    config = _two_range_config(nonuniform_perturbation_policy="allow")

    different_values = _prepared(perturbation=np.array([0.0, 1.0, 4.0, 6.0]))
    with pytest.raises(PreparedDatasetValidationError, match="point-for-point identical"):
        compute_cross_from_prepared(first, different_values, config)

    different_labels = _prepared(labels=("0", "1", "three", "6"))
    with pytest.raises(PreparedDatasetValidationError, match="labels and order"):
        compute_cross_from_prepared(first, different_labels, config)

    fewer_spectra = _prepared(
        perturbation=np.array([0.0, 1.0, 3.0]),
        labels=("0", "1", "3"),
        spectra=first.spectra[:3],
    )
    with pytest.raises(PreparedDatasetValidationError, match="same number of spectra"):
        compute_cross_from_prepared(first, fewer_spectra, config)


def test_distinct_prepared_blocks_are_blocked_without_explicit_confirmation() -> None:
    first = _prepared()
    # Even identical provenance/hashes do not prove these are the same reviewed
    # in-memory block; distinct contracts require an explicit user decision.
    second = _prepared()
    config = _two_range_config(nonuniform_perturbation_policy="allow")

    with pytest.raises(
        CrossPreparedConfirmationRequired,
        match="requires explicit confirmation",
    ):
        compute_cross_from_prepared(first, second, config)


def test_same_prepared_block_cross_needs_no_confirmation() -> None:
    prepared = _prepared()
    config = _two_range_config(nonuniform_perturbation_policy="allow")

    result = compute_cross_from_prepared(prepared, prepared, config)

    assert result.different_prepared_blocks is False
    assert result.different_prepared_blocks_confirmed is False
    assert not any("Explicit confirmation recorded" in item for item in result.warnings)


def test_confirmed_distinct_prepared_cross_records_both_parent_chains() -> None:
    first = _prepared()
    second = _prepared(
        spectra=np.asarray(first.spectra) + 0.01,
        source_name="另一个来源",
        source_sha256="3" * 64,
        baseline_fingerprint="4" * 64,
        baseline_run_id="baseline-run-2",
    )
    config = _two_range_config(nonuniform_perturbation_policy="allow")

    result = compute_cross_from_prepared(
        first,
        second,
        config,
        confirm_different_prepared_blocks=True,
    )

    assert "explicit user confirmation" in " ".join(result.warnings)
    assert "Explicit confirmation recorded" in " ".join(result.warnings)
    assert result.different_prepared_blocks is True
    assert result.different_prepared_blocks_confirmed is True
    assert result.first_parent_baseline_run_id == first.baseline_run_id
    assert result.second_parent_baseline_run_id == second.baseline_run_id
    assert result.first_parent_baseline_fingerprint == first.baseline_fingerprint
    assert result.second_parent_baseline_fingerprint == second.baseline_fingerprint
    assert result.first_parent_prepared_data_sha256 == first.prepared_data_sha256
    assert result.second_parent_prepared_data_sha256 == second.prepared_data_sha256
    assert result.first_parent_source_sha256 == first.source_sha256
    assert result.second_parent_source_sha256 == second.source_sha256
    assert result.first_parent_source_name == first.source_name
    assert result.second_parent_source_name == second.source_name
    assert first.prepared_data_sha256 != second.prepared_data_sha256


def test_nonuniform_policy_is_explicit() -> None:
    prepared = _prepared()
    warn_result = TwoDCOSWorkflowService().compute(prepared, _two_range_config())
    assert "index order" in " ".join(warn_result.warnings)

    allow_result = TwoDCOSWorkflowService().compute(
        prepared,
        _two_range_config(nonuniform_perturbation_policy="allow"),
    )
    assert not any("non-equally-spaced" in item for item in allow_result.warnings)

    with pytest.raises(PreparedDatasetValidationError, match="index order"):
        TwoDCOSWorkflowService().compute(
            prepared,
            _two_range_config(nonuniform_perturbation_policy="error"),
        )


def test_display_config_does_not_change_twodcos_fingerprint() -> None:
    prepared = _prepared()
    config = _two_range_config(nonuniform_perturbation_policy="allow")
    changed_display = replace(
        config,
        display=replace(config.display, contour_levels=71, display_percentile=95.0),
    )

    first = TwoDCOSWorkflowService().compute(prepared, config)
    second = TwoDCOSWorkflowService().compute(prepared, changed_display)

    assert first.twodcos_fingerprint == second.twodcos_fingerprint
    np.testing.assert_array_equal(
        first.homo_results[0].synchronous,
        second.homo_results[0].synchronous,
    )


def test_service_never_calls_legacy_preprocessing(monkeypatch: pytest.MonkeyPatch) -> None:
    import ftir2dcos.conversion as conversion
    import ftir2dcos.pipeline as legacy_pipeline
    import ftir2dcos.preprocessing.baseline as legacy_baseline
    import ftir2dcos.preprocessing.normalization as legacy_normalization

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("legacy preprocessing must not run")

    monkeypatch.setattr(legacy_pipeline, "preprocess_dataset", forbidden)
    monkeypatch.setattr(legacy_pipeline, "run_pipeline", forbidden)
    monkeypatch.setattr(conversion, "convert_to_absorbance", forbidden)
    monkeypatch.setattr(legacy_baseline, "correct_baseline", forbidden)
    monkeypatch.setattr(legacy_normalization, "normalize_dataset", forbidden)

    result = TwoDCOSWorkflowService().compute(
        _prepared(),
        _two_range_config(nonuniform_perturbation_policy="allow"),
    )
    assert result.all_checks_passed is True


def test_peak_order_samples_existing_canonical_self_and_cross_blocks() -> None:
    prepared = _prepared()
    config = _two_range_config(
        convention="2dpy_compatible",
        nonuniform_perturbation_policy="allow",
    )
    analysis = TwoDCOSWorkflowService().compute(prepared, config)

    order = analyze_peak_order(
        analysis,
        (
            PeakRequest(1200.2, "low", range_index=1),
            PeakRequest(1699.8, "high", range_index=0),
            PeakRequest(1600.1, "middle", range_index=0),
        ),
        tolerance=0.5,
    )

    assert len(order.evidence) == 3
    self_evidence = next(
        item for item in order.evidence if item.source == "range_1_canonical_self"
    )
    cross_evidence = next(
        item
        for item in order.evidence
        if item.first.label == "high" and item.second.label == "low"
    )
    self_block = analysis.homo_results[0].result
    cross_block = analysis.cross_results[0].result
    # Canonical axes are [1800, 1700, 1600, 1500] and [1300, 1200, 1100].
    assert self_evidence.synchronous == pytest.approx(
        self_block.canonical_synchronous[1, 2]
    )
    assert self_evidence.asynchronous == pytest.approx(
        self_block.canonical_asynchronous[1, 2]
    )
    assert cross_evidence.synchronous == pytest.approx(
        cross_block.canonical_synchronous[1, 1]
    )
    assert cross_evidence.asynchronous == pytest.approx(
        cross_block.canonical_asynchronous[1, 1]
    )
    assert cross_evidence.matched_first_wavenumber == pytest.approx(1700.0)
    assert cross_evidence.matched_second_wavenumber == pytest.approx(1200.0)
    assert (
        cross_evidence.metadata["parent_twodcos_fingerprint"]
        == analysis.twodcos_fingerprint
    )
