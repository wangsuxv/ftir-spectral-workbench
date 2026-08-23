"""Scientific sign-convention checks for apparent 2D-COS response order."""

from __future__ import annotations

import numpy as np
import pytest

from ftir2dcos.config import PipelineConfig
from ftir2dcos.models import SpectralDataset
from ftir2dcos.peak_order import (
    PairRelation,
    PeakRequest,
    ResolvedPairValues,
    infer_peak_order,
)
from ftir2dcos.pipeline import run_multi_range_pipeline


def _known_sigmoid_sequence() -> SpectralDataset:
    """Return isolated bands whose amplitude onsets are known to be A, B, C."""

    wavenumber = np.linspace(1450.0, 1750.0, 31, dtype=np.float64)
    perturbation = np.linspace(0.0, 1.0, 61, dtype=np.float64)
    band_centers = (1700.0, 1600.0, 1500.0)
    response_onsets = (0.2, 0.5, 0.8)
    band_shapes = np.asarray(
        [np.exp(-0.5 * ((wavenumber - center) / 5.0) ** 2) for center in band_centers],
        dtype=np.float64,
    )
    amplitudes = np.asarray(
        [1.0 / (1.0 + np.exp(-(perturbation - onset) / 0.04)) for onset in response_onsets],
        dtype=np.float64,
    ).T
    spectra = 0.1 + amplitudes @ band_shapes
    return SpectralDataset(
        wavenumber=wavenumber,
        perturbation=perturbation,
        perturbation_labels=tuple(f"step {index + 1}" for index in range(perturbation.size)),
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="known-sigmoid-sequence",
    )


def test_sigmoid_a_then_b_then_c_is_recovered_end_to_end_and_is_convention_invariant() -> None:
    """Positive-above-diagonal Noda signs recover the known onset sequence."""

    dataset = _known_sigmoid_sequence()
    requested = (
        PeakRequest(1500.0, "C"),
        PeakRequest(1700.0, "A"),
        PeakRequest(1600.0, "B"),
    )
    evidence_by_convention = {}
    for convention in ("canonical", "2dpy_compatible"):
        pipeline_result = run_multi_range_pipeline(
            dataset,
            [[1710.0, 1490.0]],
            PipelineConfig(convention=convention),
            peaks=requested,
            relative_threshold=1.0e-8,
        )
        order = pipeline_result.peak_order
        assert order is not None
        assert order.all_pairs_resolved is True
        assert order.has_cycles is False
        assert order.is_unique_total_order is True
        assert tuple(peak.label for peak in order.unique_order) == ("A", "B", "C")

        analysis = pipeline_result.range_results[0].result.twodcos
        assert analysis.noda[0, 1] == pytest.approx(1.0 / np.pi)
        assert analysis.noda[1, 0] == pytest.approx(-1.0 / np.pi)

        convention_evidence = {}
        for item in order.evidence:
            assert item.first.wavenumber > item.second.wavenumber
            assert item.synchronous > item.sync_threshold
            assert item.asynchronous > item.async_threshold
            assert item.relation is PairRelation.FIRST_EARLIER
            assert item.earlier == item.first
            convention_evidence[(item.first.label, item.second.label)] = (
                item.synchronous,
                item.asynchronous,
                item.sign_product,
            )
        assert set(convention_evidence) == {("A", "B"), ("A", "C"), ("B", "C")}
        evidence_by_convention[convention] = convention_evidence

    canonical = evidence_by_convention["canonical"]
    compatible = evidence_by_convention["2dpy_compatible"]
    assert canonical.keys() == compatible.keys()
    for pair in canonical:
        assert compatible[pair][0] == pytest.approx(canonical[pair][0])
        assert compatible[pair][1] == pytest.approx(canonical[pair][1])
        assert compatible[pair][2] == canonical[pair][2] == 1


def test_sigmoid_sequence_across_scrambled_ranges_uses_both_cross_orientations() -> None:
    """Known order survives forward and negative-transpose reverse sampling."""

    dataset = _known_sigmoid_sequence()
    requested = (
        PeakRequest(1500.0, "C"),
        PeakRequest(1700.0, "A"),
        PeakRequest(1600.0, "B"),
    )
    evidence_by_convention = {}
    for convention in ("canonical", "2dpy_compatible"):
        pipeline_result = run_multi_range_pipeline(
            dataset,
            # Deliberately configure low, high, middle so sampled pairs require
            # both the stored canonical direction and its reverse identity.
            [[1549.0, 1450.0], [1750.0, 1650.0], [1649.0, 1550.0]],
            PipelineConfig(convention=convention),
            peaks=requested,
            relative_threshold=1.0e-8,
        )
        order = pipeline_result.peak_order
        assert order is not None
        assert order.all_pairs_resolved is True
        assert order.has_cycles is False
        assert order.is_unique_total_order is True
        assert tuple(peak.label for peak in order.unique_order) == ("A", "B", "C")

        sources = {item.source for item in order.evidence}
        assert any(source and source.endswith("_canonical") for source in sources)
        assert any(source and source.endswith("_canonical_reverse_identity") for source in sources)
        evidence_by_convention[convention] = {
            (item.first.label, item.second.label): (
                item.synchronous,
                item.asynchronous,
                item.sign_product,
            )
            for item in order.evidence
        }

    canonical = evidence_by_convention["canonical"]
    compatible = evidence_by_convention["2dpy_compatible"]
    assert canonical.keys() == compatible.keys()
    for pair in canonical:
        assert compatible[pair][0] == pytest.approx(canonical[pair][0])
        assert compatible[pair][1] == pytest.approx(canonical[pair][1])
        assert compatible[pair][2] == canonical[pair][2] == 1


def test_nature_supplementary_table_2_signs_reproduce_published_five_peak_order() -> None:
    """Reproduce Supplementary Table 2 of DOI 10.1038/s41467-024-45079-4.

    The published table places the lower wavenumber in each row and the higher
    wavenumber in each populated column.  Its plus/minus entries mean equal or
    different synchronous/asynchronous signs, respectively.  Records below
    are therefore oriented as the paper requires: first=nu1 > second=nu2.
    """

    peaks = tuple(
        PeakRequest(wavenumber, f"{wavenumber:g}")
        for wavenumber in (3220.0, 3240.0, 3330.0, 3342.0, 3545.0)
    )
    by_wavenumber = {peak.wavenumber: peak for peak in peaks}
    # (higher nu1, lower nu2): product sign reported in Supplementary Table 2.
    published_products = {
        (3545.0, 3220.0): -1,
        (3342.0, 3220.0): -1,
        (3330.0, 3220.0): -1,
        (3240.0, 3220.0): -1,
        (3545.0, 3240.0): -1,
        (3342.0, 3240.0): -1,
        (3330.0, 3240.0): -1,
        (3545.0, 3330.0): 1,
        (3342.0, 3330.0): -1,
        (3545.0, 3342.0): 1,
    }
    records = tuple(
        ResolvedPairValues(
            first=by_wavenumber[higher],
            second=by_wavenumber[lower],
            synchronous=1.0,
            asynchronous=float(product_sign),
            source="Nature Communications Supplementary Table 2",
        )
        for (higher, lower), product_sign in published_products.items()
    )

    result = infer_peak_order(peaks, records)

    assert len(result.evidence) == 10
    assert result.all_pairs_resolved is True
    assert result.has_cycles is False
    assert result.is_unique_total_order is True
    assert tuple(peak.wavenumber for peak in result.unique_order) == (
        3220.0,
        3240.0,
        3545.0,
        3330.0,
        3342.0,
    )
    for item in result.evidence:
        pair = (item.first.wavenumber, item.second.wavenumber)
        assert item.first.wavenumber > item.second.wavenumber
        assert item.sign_product == published_products[pair]
        expected_earlier = item.first if published_products[pair] > 0 else item.second
        assert item.earlier == expected_earlier
