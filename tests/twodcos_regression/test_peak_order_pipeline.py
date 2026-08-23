from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

import ftir2dcos.pipeline as pipeline_module
from ftir2dcos.config import PipelineConfig
from ftir2dcos.models import SpectralDataset
from ftir2dcos.peak_order import PairRelation, PeakRequest
from ftir2dcos.pipeline import (
    analyze_multi_range_peak_order,
    run_multi_range_pipeline,
)


def make_peak_dataset(
    *,
    perturbation: np.ndarray | None = None,
) -> SpectralDataset:
    wavenumber = np.linspace(1000.0, 1800.0, 161, dtype=np.float64)
    values = (
        np.asarray([0.0, 1.0, 3.0, 6.0, 10.0])
        if perturbation is None
        else np.asarray(perturbation, dtype=np.float64)
    )
    high_peak = np.exp(-0.5 * ((wavenumber - 1630.0) / 24.0) ** 2)
    high_shoulder = np.exp(-0.5 * ((wavenumber - 1580.0) / 18.0) ** 2)
    low_peak = np.exp(-0.5 * ((wavenumber - 1190.0) / 20.0) ** 2)
    spectra = np.vstack(
        [
            0.03
            + (0.45 + value / 90.0) * high_peak
            + (0.25 + value**2 / 1500.0) * high_shoulder
            + (0.35 - value / 130.0) * low_peak
            for value in values
        ]
    )
    return SpectralDataset(
        wavenumber=wavenumber,
        perturbation=values,
        perturbation_labels=tuple(f"step {index + 1}" for index in range(values.size)),
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="peak-order-synthetic.csv",
    )


def run_two_ranges(*, convention: str = "canonical"):
    return run_multi_range_pipeline(
        make_peak_dataset(),
        [[1735, 1510], [1250, 1140]],
        PipelineConfig(convention=convention),
    )


def test_nearest_grid_match_is_axis_direction_invariant_and_rejects_midpoint_ties() -> None:
    ascending = np.asarray([0.0, 2.0, 4.0, 6.0])
    descending = ascending[::-1]

    ascending_match = pipeline_module._nearest_grid_match(ascending, 4.2)
    descending_match = pipeline_module._nearest_grid_match(descending, 4.2)
    assert ascending_match[1:] == pytest.approx(descending_match[1:])
    assert ascending_match[1] == 4.0

    with pytest.raises(ValueError, match="exactly equidistant"):
        pipeline_module._nearest_grid_match(ascending, 3.0)
    with pytest.raises(ValueError, match="exactly equidistant"):
        pipeline_module._nearest_grid_match(descending, 3.0)


def test_cross_range_sampling_is_convention_invariant_and_high_to_low() -> None:
    canonical = analyze_multi_range_peak_order(
        run_two_ranges(convention="canonical"),
        [PeakRequest(1190, "low"), PeakRequest(1630, "high")],
    )
    compatible = analyze_multi_range_peak_order(
        run_two_ranges(convention="2dpy_compatible"),
        [PeakRequest(1190, "low"), PeakRequest(1630, "high")],
    )

    first = canonical.evidence[0]
    second = compatible.evidence[0]
    assert first.first.label == "high"
    assert first.second.label == "low"
    assert first.matched_first_wavenumber == 1630
    assert first.matched_second_wavenumber == 1190
    assert first.source == "cross_ranges_1_2_canonical"
    assert first.synchronous == pytest.approx(second.synchronous)
    assert first.asynchronous == pytest.approx(second.asynchronous)
    assert first.relation == second.relation
    assert first.metadata["pair_orientation"] == "higher_matched_wavenumber_first"
    assert first.sync_threshold == pytest.approx(1.0e-6 * first.metadata["sync_matrix_max_abs"])
    assert first.async_threshold == pytest.approx(1.0e-6 * first.metadata["async_matrix_max_abs"])


def test_reverse_cross_range_sampling_applies_async_negative_transpose() -> None:
    result = run_multi_range_pipeline(
        make_peak_dataset(),
        [[1250, 1140], [1735, 1510]],
        PipelineConfig(convention="2dpy_compatible"),
    )
    order = analyze_multi_range_peak_order(result, [1190, 1630])
    evidence = order.evidence[0]
    cross = result.cross_results[0].twodcos
    low_index = int(np.argmin(np.abs(cross.wavenumber1 - 1190)))
    high_index = int(np.argmin(np.abs(cross.wavenumber2 - 1630)))

    assert evidence.first.wavenumber == 1630
    assert evidence.source == "cross_ranges_1_2_canonical_reverse_identity"
    assert evidence.synchronous == pytest.approx(cross.canonical_synchronous[low_index, high_index])
    assert evidence.asynchronous == pytest.approx(
        -cross.canonical_asynchronous[low_index, high_index]
    )


def test_same_range_uses_canonical_self_matrix() -> None:
    result = run_two_ranges(convention="2dpy_compatible")
    order = analyze_multi_range_peak_order(result, [1630, 1580])
    evidence = order.evidence[0]
    analysis = result.range_results[0].result.twodcos
    first_index = int(np.argmin(np.abs(result.results[0].processed.wavenumber - 1630)))
    second_index = int(np.argmin(np.abs(result.results[0].processed.wavenumber - 1580)))

    assert evidence.source == "range_1_canonical_self"
    assert evidence.synchronous == pytest.approx(
        analysis.canonical_synchronous[first_index, second_index]
    )
    assert evidence.asynchronous == pytest.approx(
        analysis.canonical_asynchronous[first_index, second_index]
    )


def test_matching_tolerance_overlap_and_duplicate_grid_are_explicit() -> None:
    result = run_two_ranges()
    with pytest.raises(ValueError, match="no sampled grid point within 1"):
        analyze_multi_range_peak_order(result, [1632, 1190])
    matched = analyze_multi_range_peak_order(
        result,
        [1632, 1190],
        peak_match_tolerance=2,
    )
    assert matched.evidence[0].matched_first_wavenumber == 1630

    overlap = run_multi_range_pipeline(
        make_peak_dataset(),
        [[1700, 1500], [1650, 1450]],
    )
    with pytest.raises(ValueError, match="ambiguous across overlapping ranges"):
        analyze_multi_range_peak_order(overlap, [1600, 1550])
    explicit = analyze_multi_range_peak_order(
        overlap,
        [PeakRequest(1600, range_index=0), PeakRequest(1550, range_index=0)],
    )
    assert all(peak.range_index == 0 for peak in explicit.peaks)

    with pytest.raises(ValueError, match="same physical grid position"):
        analyze_multi_range_peak_order(
            overlap,
            [PeakRequest(1600, range_index=0), PeakRequest(1600, range_index=1)],
        )
    with pytest.raises(ValueError, match="same physical grid position"):
        analyze_multi_range_peak_order(result, [1630.1, 1629.9])


def test_signal_cutoffs_and_analysis_order_facts_are_auditable() -> None:
    result = run_multi_range_pipeline(
        make_peak_dataset(perturbation=np.asarray([0.0, 3.0, 1.0, 6.0, 10.0])),
        [[1735, 1510]],
        peaks=[1630, 1580],
        synchronous_threshold=1.0,
        analysis_order_note="operator supplied stage labels",
    )
    assert result.peak_order is not None
    evidence = result.peak_order.evidence[0]
    assert evidence.relation is PairRelation.INDETERMINATE
    assert evidence.sync_threshold == 1.0
    assert "non-monotonic" in result.peak_order.analysis_order_note
    assert "does not override stored order" in result.peak_order.analysis_order_note
    assert any("non-monotonic" in warning for warning in result.peak_order.warnings)
    assert any("non-uniform" in warning for warning in result.peak_order.warnings)


def test_peak_order_export_json_csv_manifest_and_zip(tmp_path: Path) -> None:
    result = run_multi_range_pipeline(
        make_peak_dataset(),
        [[1735, 1510], [1250, 1140]],
        peaks=[PeakRequest(1630, "high"), PeakRequest(1190, "low")],
        output_root=tmp_path,
    )
    assert result.output_directory is not None
    assert result.bundle_path is not None
    peak_directory = result.output_directory / "peak_order"
    payload = json.loads((peak_directory / "peak_order.json").read_text(encoding="utf-8"))
    assert len(payload["peaks"]) == 2
    assert len(payload["evidence"]) == 1
    assert payload["evidence"][0]["first"]["label"] == "high"
    assert "relative_signal_strength" in payload["evidence"][0]

    with (peak_directory / "pairwise_evidence.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["first_label"] == "high"
    assert rows[0]["source"] == "cross_ranges_1_2_canonical"

    manifest = json.loads(
        (result.output_directory / "multi_range_manifest.json").read_text(encoding="utf-8")
    )
    summary = manifest["peak_response_order"]
    assert summary["requested"] is True
    assert summary["peak_count"] == 2
    assert summary["pairwise_evidence_count"] == 1
    assert "not confidence levels" in summary["signal_cutoff_contract"]
    assert len(summary["pairwise_effective_signal_cutoffs"]) == 1
    assert summary["pairwise_effective_signal_cutoffs"][0]["sync_threshold"] > 0
    assert summary["theory_references"][1]["url"].endswith("s41467-024-45079-4")

    with zipfile.ZipFile(result.bundle_path) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
    assert "peak_order/peak_order.json" in names
    assert "peak_order/pairwise_evidence.csv" in names
