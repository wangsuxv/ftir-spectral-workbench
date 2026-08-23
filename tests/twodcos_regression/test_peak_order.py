"""Noda-rule peak response ordering, ambiguity, and graph diagnostics."""

from __future__ import annotations

import math

import pytest

from ftir2dcos.peak_order import (
    PairRelation,
    PeakRequest,
    ResolvedPairValues,
    classify_pairwise_evidence,
    infer_peak_order,
)


def _pair(
    first: PeakRequest,
    second: PeakRequest,
    synchronous: float,
    asynchronous: float,
    **kwargs: object,
) -> ResolvedPairValues:
    return ResolvedPairValues(
        first=first,
        second=second,
        synchronous=synchronous,
        asynchronous=asynchronous,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("synchronous", "asynchronous", "expected_relation", "expected_earlier"),
    [
        (2.0, 3.0, PairRelation.FIRST_EARLIER, 0),
        (-2.0, -3.0, PairRelation.FIRST_EARLIER, 0),
        (2.0, -3.0, PairRelation.SECOND_EARLIER, 1),
        (-2.0, 3.0, PairRelation.SECOND_EARLIER, 1),
    ],
)
def test_ordered_canonical_noda_sign_rule(
    synchronous: float,
    asynchronous: float,
    expected_relation: PairRelation,
    expected_earlier: int,
) -> None:
    peaks = (PeakRequest(1700.0, "high"), PeakRequest(1200.0, "low"))
    evidence = classify_pairwise_evidence(_pair(peaks[0], peaks[1], synchronous, asynchronous))

    assert evidence.relation is expected_relation
    assert evidence.earlier == peaks[expected_earlier]
    assert evidence.later == peaks[1 - expected_earlier]
    assert evidence.sign_product == (1 if expected_earlier == 0 else -1)
    assert evidence.is_resolved is True


@pytest.mark.parametrize(
    ("synchronous", "asynchronous", "expected_phrase"),
    [
        (0.1, 4.0, "Synchronous"),
        (2.0, -0.2, "Asynchronous"),
        (0.1, 0.2, "Synchronous and asynchronous"),
    ],
)
def test_values_at_or_below_separate_absolute_cutoffs_are_indeterminate(
    synchronous: float,
    asynchronous: float,
    expected_phrase: str,
) -> None:
    first = PeakRequest(1700.0)
    second = PeakRequest(1600.0)
    evidence = classify_pairwise_evidence(
        _pair(
            first,
            second,
            synchronous,
            asynchronous,
            sync_threshold=0.1,
            async_threshold=0.2,
        )
    )

    assert evidence.relation is PairRelation.INDETERMINATE
    assert evidence.earlier is None
    assert evidence.later is None
    assert evidence.sign_product == 0
    assert expected_phrase in evidence.reason
    assert evidence.sync_threshold == pytest.approx(0.1)
    assert evidence.async_threshold == pytest.approx(0.2)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_matrix_values_are_retained_as_unresolved_evidence(
    bad_value: float,
) -> None:
    first = PeakRequest(1700.0)
    second = PeakRequest(1600.0)
    record = _pair(first, second, bad_value, 1.0)

    result = infer_peak_order((first, second), (record,))

    assert result.evidence[0].relation is PairRelation.NON_FINITE
    assert result.evidence[0].is_resolved is False
    assert result.unresolved_relations[0].reason == "non_finite"
    assert result.all_pairs_resolved is False
    assert any("non-finite" in warning for warning in result.warnings)


def test_unique_total_order_preserves_pairwise_audit_metadata() -> None:
    a = PeakRequest(1700.0, "A", 0)
    b = PeakRequest(1600.0, "B", 0)
    c = PeakRequest(1200.0, "C", 1)
    records = (
        _pair(
            a,
            b,
            2.0,
            1.0,
            matched_first_wavenumber=1699.8,
            matched_second_wavenumber=1600.2,
            sync_threshold=0.01,
            async_threshold=0.02,
            relative_signal_strength=0.25,
            source="range_1_auto",
            metadata={"row_index": 10, "column_index": 20},
        ),
        _pair(a, c, -3.0, -0.5, source="range_1__range_2_cross"),
        _pair(b, c, 4.0, 0.4, source="range_1__range_2_cross"),
    )

    result = infer_peak_order((a, b, c), records)

    assert result.has_cycles is False
    assert result.all_pairs_resolved is True
    assert result.is_unique_total_order is True
    assert result.unique_order == (a, b, c)
    assert result.topological_layers == ((a,), (b,), (c,))
    assert result.cyclic_groups == ()
    assert result.evidence[0].matched_first_wavenumber == pytest.approx(1699.8)
    assert result.evidence[0].matched_second_wavenumber == pytest.approx(1600.2)
    assert result.evidence[0].source == "range_1_auto"
    assert result.evidence[0].relative_signal_strength == pytest.approx(0.25)
    assert result.evidence[0].value_product == pytest.approx(2.0)
    assert result.evidence[0].minimum_cutoff_ratio == pytest.approx(50.0)
    assert result.evidence[0].metadata == {"row_index": 10, "column_index": 20}

    exported = result.to_dict()
    assert exported["has_cycles"] is False
    assert exported["unique_order"][0]["label"] == "A"
    assert exported["evidence"][0]["sync_threshold"] == pytest.approx(0.01)
    assert exported["evidence"][0]["relative_signal_strength"] == pytest.approx(0.25)
    assert exported["evidence"][0]["metadata"]["row_index"] == 10
    assert exported["evidence_records"][0]["value_product"] == pytest.approx(2.0)
    assert "monotonic perturbation sequence" in exported["analysis_order_note"]
    assert "canonical[row=first" in exported["rule_description"]


def test_missing_pair_is_explicit_but_transitive_edges_can_still_fix_order() -> None:
    a = PeakRequest(1700.0, "A")
    b = PeakRequest(1600.0, "B")
    c = PeakRequest(1500.0, "C")
    result = infer_peak_order(
        (a, b, c),
        (
            _pair(a, b, 1.0, 1.0),
            _pair(b, c, 1.0, 1.0),
        ),
    )

    assert result.all_pairs_resolved is False
    assert result.is_unique_total_order is True
    assert result.unique_order == (a, b, c)
    assert len(result.unresolved_relations) == 1
    assert result.unresolved_relations[0].first == a
    assert result.unresolved_relations[0].second == c
    assert result.unresolved_relations[0].reason == "missing_evidence"
    assert any("no evidence" in warning for warning in result.warnings)


def test_unresolved_relations_produce_topological_tiers_not_fake_sorting() -> None:
    a = PeakRequest(1700.0, "A")
    b = PeakRequest(1600.0, "B")
    c = PeakRequest(1500.0, "C")
    result = infer_peak_order(
        (a, b, c),
        (
            _pair(a, c, 2.0, 2.0),
            _pair(b, c, 0.01, -4.0, sync_threshold=0.1),
        ),
    )

    assert result.has_cycles is False
    assert result.is_unique_total_order is False
    assert result.unique_order == ()
    assert result.topological_layers == ((a, b), (c,))
    assert {item.reason for item in result.unresolved_relations} == {
        "missing_evidence",
        "near_zero",
    }
    assert any("partial order" in warning for warning in result.warnings)


def test_directed_cycle_is_returned_as_scc_group_without_total_order() -> None:
    a = PeakRequest(1700.0, "A")
    b = PeakRequest(1600.0, "B")
    c = PeakRequest(1500.0, "C")
    records = (
        _pair(a, b, 1.0, 1.0),  # A -> B
        _pair(b, c, 1.0, 1.0),  # B -> C
        _pair(a, c, 1.0, -1.0),  # C -> A
    )

    result = infer_peak_order((a, b, c), records)

    assert result.has_cycles is True
    assert result.is_unique_total_order is False
    assert result.unique_order == ()
    assert result.topological_layers == ()
    assert result.cyclic_groups == ((a, b, c),)
    assert result.strongly_connected_components == ((a, b, c),)
    assert result.component_layers == (((a, b, c),),)
    assert any("cycle" in warning for warning in result.warnings)
    assert any("no total response order was fabricated" in warning for warning in result.warnings)


def test_duplicate_peak_identity_is_rejected_but_same_value_in_other_range_is_valid() -> None:
    duplicate_a = PeakRequest(1600.0, "A", 0)
    duplicate_b = PeakRequest(1600.0, "renamed A", 0)
    with pytest.raises(ValueError, match="Duplicate requested peak identities"):
        infer_peak_order((duplicate_a, duplicate_b), ())

    other_range = PeakRequest(1600.0, "other range", 1)
    result = infer_peak_order(
        (duplicate_a, other_range),
        (_pair(duplicate_a, other_range, 1.0, 1.0),),
    )
    assert result.unique_order == (duplicate_a, other_range)


def test_duplicate_pair_records_and_self_pairs_are_rejected() -> None:
    a = PeakRequest(1700.0, "A")
    b = PeakRequest(1600.0, "B")
    with pytest.raises(ValueError, match="Duplicate pair records"):
        infer_peak_order(
            (a, b),
            (
                _pair(a, b, 1.0, 1.0),
                _pair(b, a, -1.0, 1.0),
            ),
        )
    with pytest.raises(ValueError, match="cannot compare a peak with itself"):
        infer_peak_order((a,), (_pair(a, a, 1.0, 1.0),))


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: PeakRequest(math.nan),
        lambda: PeakRequest(math.inf),
        lambda: PeakRequest(1600.0, range_index=-1),
        lambda: ResolvedPairValues(
            PeakRequest(1700.0),
            PeakRequest(1600.0),
            1.0,
            1.0,
            sync_threshold=-0.1,
        ),
        lambda: ResolvedPairValues(
            PeakRequest(1700.0),
            PeakRequest(1600.0),
            1.0,
            1.0,
            relative_signal_strength=1.1,
        ),
    ],
)
def test_invalid_peak_coordinates_and_thresholds_fail_early(constructor: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        constructor()  # type: ignore[operator]
