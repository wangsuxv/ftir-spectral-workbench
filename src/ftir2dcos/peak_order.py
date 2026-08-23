"""Auditable response-order inference from canonical 2D-COS peak pairs.

The Noda sign rule is evaluated for an explicitly ordered pair.  A
``ResolvedPairValues`` record always means that ``first`` was sampled from the
row and ``second`` from the column of a *canonical* matrix.  Consequently,
equal synchronous/asynchronous signs mean that ``first`` responds earlier;
opposite signs mean that ``second`` responds earlier.  Plotting conventions
such as the final transpose used by 2Dpy must be undone before constructing a
record.

This module deliberately reports a graph-derived partial order.  Near-zero or
missing pair relations remain unresolved, and inconsistent directed cycles are
returned as strongly connected groups instead of being forced into a total
order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations
from math import isfinite
from numbers import Integral, Real
from typing import Any

ANALYSIS_ORDER_NOTE = (
    "Interpret 'earlier' only along the supplied, meaningfully ordered monotonic "
    "perturbation sequence; no physical direction such as increasing temperature is assumed."
)

NODA_RULE_DESCRIPTION = (
    "For canonical[row=first, column=second], equal non-zero synchronous and "
    "asynchronous signs imply first earlier; opposite signs imply second earlier. "
    "To reproduce the common literature table convention, orient the pair with first=nu1>nu2."
)


def _real_number(value: Real, *, name: str, require_finite: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if require_finite and not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _threshold(value: Real, *, name: str) -> float:
    normalized = _real_number(value, name=name, require_finite=True)
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


@dataclass(frozen=True, slots=True)
class PeakRequest:
    """One requested peak position.

    ``range_index`` disambiguates the same numerical wavenumber when it occurs
    in different independently preprocessed ranges.  Within one range, a
    wavenumber may only appear once in an inference request.
    """

    wavenumber: float
    label: str | None = None
    range_index: int | None = None

    def __post_init__(self) -> None:
        wavenumber = _real_number(
            self.wavenumber,
            name="wavenumber",
            require_finite=True,
        )
        label = None if self.label is None else str(self.label).strip()
        if label == "":
            label = None
        range_index = self.range_index
        if range_index is not None:
            if isinstance(range_index, bool) or not isinstance(range_index, Integral):
                raise TypeError("range_index must be an integer or None")
            range_index = int(range_index)
            if range_index < 0:
                raise ValueError("range_index must be non-negative")

        object.__setattr__(self, "wavenumber", wavenumber)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "range_index", range_index)

    @property
    def identity(self) -> tuple[int | None, float]:
        """Stable identity used for duplicate detection and graph nodes."""

        return self.range_index, self.wavenumber

    @property
    def display_label(self) -> str:
        """Human-readable label without changing peak identity."""

        if self.label is not None:
            return self.label
        range_suffix = "" if self.range_index is None else f" (range {self.range_index + 1})"
        return f"{self.wavenumber:g} cm^-1{range_suffix}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "wavenumber": self.wavenumber,
            "label": self.label,
            "range_index": self.range_index,
            "display_label": self.display_label,
        }


@dataclass(frozen=True, slots=True)
class ResolvedPairValues:
    """Values sampled for one ordered peak pair from canonical matrices.

    The record orientation is significant: ``synchronous`` and
    ``asynchronous`` must be sampled at ``[row=first, column=second]``.  The
    two absolute thresholds are kept with the record because auto- and
    cross-range matrices can have different scales.  They are numerical
    interpretation cutoffs, not statistical confidence levels.

    ``relative_signal_strength``, when supplied by the matrix sampler, is
    defined as ``min(abs(sync)/max_abs_sync, abs(async)/max_abs_async)`` for
    the source block.  It is constrained to ``[0, 1]`` and is a display-scale
    amplitude descriptor, not statistical confidence.

    Non-finite matrix values are accepted here so inference can retain them as
    explicit ``non_finite`` evidence rather than silently discarding a pair.
    """

    first: PeakRequest
    second: PeakRequest
    synchronous: float
    asynchronous: float
    matched_first_wavenumber: float | None = None
    matched_second_wavenumber: float | None = None
    sync_threshold: float = 0.0
    async_threshold: float = 0.0
    relative_signal_strength: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.first, PeakRequest) or not isinstance(self.second, PeakRequest):
            raise TypeError("first and second must be PeakRequest instances")
        synchronous = _real_number(
            self.synchronous,
            name="synchronous",
            require_finite=False,
        )
        asynchronous = _real_number(
            self.asynchronous,
            name="asynchronous",
            require_finite=False,
        )
        matched_first = self.matched_first_wavenumber
        if matched_first is not None:
            matched_first = _real_number(
                matched_first,
                name="matched_first_wavenumber",
                require_finite=True,
            )
        matched_second = self.matched_second_wavenumber
        if matched_second is not None:
            matched_second = _real_number(
                matched_second,
                name="matched_second_wavenumber",
                require_finite=True,
            )

        object.__setattr__(self, "synchronous", synchronous)
        object.__setattr__(self, "asynchronous", asynchronous)
        object.__setattr__(self, "matched_first_wavenumber", matched_first)
        object.__setattr__(self, "matched_second_wavenumber", matched_second)
        object.__setattr__(
            self,
            "sync_threshold",
            _threshold(self.sync_threshold, name="sync_threshold"),
        )
        object.__setattr__(
            self,
            "async_threshold",
            _threshold(self.async_threshold, name="async_threshold"),
        )
        relative_signal_strength = self.relative_signal_strength
        if relative_signal_strength is not None:
            relative_signal_strength = _real_number(
                relative_signal_strength,
                name="relative_signal_strength",
                require_finite=True,
            )
            if not 0.0 <= relative_signal_strength <= 1.0:
                raise ValueError("relative_signal_strength must be between 0 and 1")
        object.__setattr__(self, "relative_signal_strength", relative_signal_strength)
        object.__setattr__(self, "source", None if self.source is None else str(self.source))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))


class PairRelation(StrEnum):
    """Outcome of applying the Noda sign rule to one ordered pair."""

    FIRST_EARLIER = "first_earlier"
    SECOND_EARLIER = "second_earlier"
    INDETERMINATE = "indeterminate"
    NON_FINITE = "non_finite"


@dataclass(frozen=True, slots=True)
class PairwiseEvidence:
    """Auditable interpretation of one :class:`ResolvedPairValues` record."""

    first: PeakRequest
    second: PeakRequest
    synchronous: float
    asynchronous: float
    matched_first_wavenumber: float | None
    matched_second_wavenumber: float | None
    sync_threshold: float
    async_threshold: float
    relation: PairRelation
    earlier: PeakRequest | None
    later: PeakRequest | None
    sign_product: int | None
    reason: str
    relative_signal_strength: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", PairRelation(self.relation))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    @property
    def is_resolved(self) -> bool:
        """Whether this evidence contributes one directed graph edge."""

        return self.relation in {
            PairRelation.FIRST_EARLIER,
            PairRelation.SECOND_EARLIER,
        }

    @property
    def value_product(self) -> float | None:
        """Return ``synchronous * asynchronous`` when safely finite.

        ``sign_product`` remains authoritative for the decision because this
        magnitude can underflow or overflow.  Neither quantity is a
        statistical confidence measure.
        """

        if not isfinite(self.synchronous) or not isfinite(self.asynchronous):
            return None
        product = self.synchronous * self.asynchronous
        return product if isfinite(product) else None

    @property
    def synchronous_to_cutoff_ratio(self) -> float | None:
        """Signal magnitude divided by its non-zero synchronous cutoff."""

        if not isfinite(self.synchronous) or self.sync_threshold == 0.0:
            return None
        ratio = abs(self.synchronous) / self.sync_threshold
        return ratio if isfinite(ratio) else None

    @property
    def asynchronous_to_cutoff_ratio(self) -> float | None:
        """Signal magnitude divided by its non-zero asynchronous cutoff."""

        if not isfinite(self.asynchronous) or self.async_threshold == 0.0:
            return None
        ratio = abs(self.asynchronous) / self.async_threshold
        return ratio if isfinite(ratio) else None

    @property
    def minimum_cutoff_ratio(self) -> float | None:
        """Smaller of the two signal-to-cutoff ratios, not confidence."""

        ratios = (
            self.synchronous_to_cutoff_ratio,
            self.asynchronous_to_cutoff_ratio,
        )
        if any(ratio is None for ratio in ratios):
            return None
        return min(ratio for ratio in ratios if ratio is not None)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible audit record."""

        return {
            "first": self.first.to_dict(),
            "second": self.second.to_dict(),
            "synchronous": self.synchronous,
            "asynchronous": self.asynchronous,
            "matched_first_wavenumber": self.matched_first_wavenumber,
            "matched_second_wavenumber": self.matched_second_wavenumber,
            "sync_threshold": self.sync_threshold,
            "async_threshold": self.async_threshold,
            "relation": self.relation.value,
            "earlier": None if self.earlier is None else self.earlier.to_dict(),
            "later": None if self.later is None else self.later.to_dict(),
            "sign_product": self.sign_product,
            "value_product": self.value_product,
            "synchronous_to_cutoff_ratio": self.synchronous_to_cutoff_ratio,
            "asynchronous_to_cutoff_ratio": self.asynchronous_to_cutoff_ratio,
            "minimum_cutoff_ratio": self.minimum_cutoff_ratio,
            "relative_signal_strength": self.relative_signal_strength,
            "reason": self.reason,
            "source": self.source,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class UnresolvedRelation:
    """One requested pair whose direction could not be established."""

    first: PeakRequest
    second: PeakRequest
    reason: str
    evidence: PairwiseEvidence | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "first": self.first.to_dict(),
            "second": self.second.to_dict(),
            "reason": self.reason,
            "evidence": None if self.evidence is None else self.evidence.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PeakOrderResult:
    """Global graph result without an invented order for ambiguous peaks.

    ``topological_layers`` is populated only when the resolved graph is a DAG.
    Peaks in the same layer have no graph-supported order between them.
    ``component_layers`` is always populated and retains cyclic strongly
    connected components as groups.
    """

    peaks: tuple[PeakRequest, ...]
    evidence: tuple[PairwiseEvidence, ...]
    unresolved_relations: tuple[UnresolvedRelation, ...]
    topological_layers: tuple[tuple[PeakRequest, ...], ...]
    component_layers: tuple[tuple[tuple[PeakRequest, ...], ...], ...]
    strongly_connected_components: tuple[tuple[PeakRequest, ...], ...]
    cyclic_groups: tuple[tuple[PeakRequest, ...], ...]
    unique_order: tuple[PeakRequest, ...]
    has_cycles: bool
    is_unique_total_order: bool
    all_pairs_resolved: bool
    warnings: tuple[str, ...]
    analysis_order_note: str = ANALYSIS_ORDER_NOTE
    rule_description: str = NODA_RULE_DESCRIPTION

    def to_evidence_records(self) -> list[dict[str, Any]]:
        """Return flat records suitable for CSV or a dataframe.

        Cutoff ratios describe numerical distance from a signal cutoff; they
        must not be relabelled as statistical confidence.
        """

        records: list[dict[str, Any]] = []
        for item in self.evidence:
            records.append(
                {
                    "first_label": item.first.display_label,
                    "first_requested_wavenumber": item.first.wavenumber,
                    "first_matched_wavenumber": item.matched_first_wavenumber,
                    "first_range_index": item.first.range_index,
                    "second_label": item.second.display_label,
                    "second_requested_wavenumber": item.second.wavenumber,
                    "second_matched_wavenumber": item.matched_second_wavenumber,
                    "second_range_index": item.second.range_index,
                    "synchronous": item.synchronous,
                    "asynchronous": item.asynchronous,
                    "value_product": item.value_product,
                    "sign_product": item.sign_product,
                    "sync_threshold": item.sync_threshold,
                    "async_threshold": item.async_threshold,
                    "synchronous_to_cutoff_ratio": item.synchronous_to_cutoff_ratio,
                    "asynchronous_to_cutoff_ratio": item.asynchronous_to_cutoff_ratio,
                    "minimum_cutoff_ratio": item.minimum_cutoff_ratio,
                    "relative_signal_strength": item.relative_signal_strength,
                    "relation": item.relation.value,
                    "earlier_label": None if item.earlier is None else item.earlier.display_label,
                    "later_label": None if item.later is None else item.later.display_label,
                    "reason": item.reason,
                    "source": item.source,
                }
            )
        return records

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the complete audit trail."""

        def peak_group(group: Sequence[PeakRequest]) -> list[dict[str, Any]]:
            return [peak.to_dict() for peak in group]

        return {
            "peaks": peak_group(self.peaks),
            "evidence": [item.to_dict() for item in self.evidence],
            "evidence_records": self.to_evidence_records(),
            "unresolved_relations": [item.to_dict() for item in self.unresolved_relations],
            "topological_layers": [peak_group(layer) for layer in self.topological_layers],
            "component_layers": [
                [peak_group(component) for component in layer] for layer in self.component_layers
            ],
            "strongly_connected_components": [
                peak_group(component) for component in self.strongly_connected_components
            ],
            "cyclic_groups": [peak_group(component) for component in self.cyclic_groups],
            "unique_order": peak_group(self.unique_order),
            "has_cycles": self.has_cycles,
            "is_unique_total_order": self.is_unique_total_order,
            "all_pairs_resolved": self.all_pairs_resolved,
            "warnings": list(self.warnings),
            "analysis_order_note": self.analysis_order_note,
            "rule_description": self.rule_description,
        }


def classify_pairwise_evidence(record: ResolvedPairValues) -> PairwiseEvidence:
    """Apply thresholds and the ordered canonical Noda rule to one pair."""

    if not isinstance(record, ResolvedPairValues):
        raise TypeError("record must be a ResolvedPairValues instance")

    common = {
        "first": record.first,
        "second": record.second,
        "synchronous": record.synchronous,
        "asynchronous": record.asynchronous,
        "matched_first_wavenumber": record.matched_first_wavenumber,
        "matched_second_wavenumber": record.matched_second_wavenumber,
        "sync_threshold": record.sync_threshold,
        "async_threshold": record.async_threshold,
        "relative_signal_strength": record.relative_signal_strength,
        "source": record.source,
        "metadata": record.metadata,
    }

    if not isfinite(record.synchronous) or not isfinite(record.asynchronous):
        return PairwiseEvidence(
            **common,
            relation=PairRelation.NON_FINITE,
            earlier=None,
            later=None,
            sign_product=None,
            reason="Synchronous or asynchronous value is NaN or infinite.",
        )

    sync_near_zero = abs(record.synchronous) <= record.sync_threshold
    async_near_zero = abs(record.asynchronous) <= record.async_threshold
    if sync_near_zero or async_near_zero:
        near_zero_parts = []
        if sync_near_zero:
            near_zero_parts.append("synchronous")
        if async_near_zero:
            near_zero_parts.append("asynchronous")
        if len(near_zero_parts) == 2:
            threshold_reason = (
                "Synchronous and asynchronous values are at or below their absolute "
                "interpretation thresholds."
            )
        else:
            threshold_reason = (
                f"{near_zero_parts[0].capitalize()} value is at or below its absolute "
                "interpretation threshold."
            )
        return PairwiseEvidence(
            **common,
            relation=PairRelation.INDETERMINATE,
            earlier=None,
            later=None,
            sign_product=0,
            reason=threshold_reason,
        )

    same_sign = (record.synchronous > 0.0) == (record.asynchronous > 0.0)
    if same_sign:
        return PairwiseEvidence(
            **common,
            relation=PairRelation.FIRST_EARLIER,
            earlier=record.first,
            later=record.second,
            sign_product=1,
            reason="Synchronous and asynchronous values have equal signs.",
        )
    return PairwiseEvidence(
        **common,
        relation=PairRelation.SECOND_EARLIER,
        earlier=record.second,
        later=record.first,
        sign_product=-1,
        reason="Synchronous and asynchronous values have opposite signs.",
    )


def _topological_layers(
    node_count: int,
    edges: Sequence[set[int]],
) -> tuple[tuple[tuple[int, ...], ...], bool]:
    indegree = [0] * node_count
    for destinations in edges:
        for destination in destinations:
            indegree[destination] += 1

    remaining = set(range(node_count))
    layers: list[tuple[int, ...]] = []
    unique = True
    while remaining:
        layer = tuple(index for index in sorted(remaining) if indegree[index] == 0)
        if not layer:
            return tuple(layers), False
        if len(layer) != 1:
            unique = False
        layers.append(layer)
        for source in layer:
            remaining.remove(source)
            for destination in edges[source]:
                indegree[destination] -= 1
    return tuple(layers), unique


def _strongly_connected_components(
    edges: Sequence[set[int]],
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic Tarjan components for a small peak graph."""

    node_count = len(edges)
    next_index = 0
    indices = [-1] * node_count
    lowlinks = [0] * node_count
    stack: list[int] = []
    on_stack: set[int] = set()
    components: list[tuple[int, ...]] = []

    def visit(node: int) -> None:
        nonlocal next_index
        indices[node] = next_index
        lowlinks[node] = next_index
        next_index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in sorted(edges[node]):
            if indices[neighbor] == -1:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] != indices[node]:
            return
        component: list[int] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        components.append(tuple(sorted(component)))

    for node in range(node_count):
        if indices[node] == -1:
            visit(node)
    return tuple(components)


def infer_peak_order(
    peaks: Iterable[PeakRequest],
    pair_values: Iterable[ResolvedPairValues],
    *,
    analysis_order_note: str = ANALYSIS_ORDER_NOTE,
) -> PeakOrderResult:
    """Infer a global response order from canonical pair-value records.

    The function is pure: it samples no matrices and modifies no inputs.  Every
    unordered peak pair may have at most one record, although that record can
    use either orientation.  Missing, thresholded, and non-finite relations are
    reported explicitly.
    """

    peak_tuple = tuple(peaks)
    if not peak_tuple:
        raise ValueError("peaks must contain at least one PeakRequest")
    if not all(isinstance(peak, PeakRequest) for peak in peak_tuple):
        raise TypeError("peaks must contain only PeakRequest instances")

    identity_to_index: dict[tuple[int | None, float], int] = {}
    duplicate_identities: list[tuple[int | None, float]] = []
    for index, peak in enumerate(peak_tuple):
        if peak.identity in identity_to_index:
            duplicate_identities.append(peak.identity)
        else:
            identity_to_index[peak.identity] = index
    if duplicate_identities:
        rendered = ", ".join(
            f"range_index={range_index!r}, wavenumber={wavenumber:g}"
            for range_index, wavenumber in duplicate_identities
        )
        raise ValueError(f"Duplicate requested peak identities: {rendered}")

    record_tuple = tuple(pair_values)
    if not all(isinstance(record, ResolvedPairValues) for record in record_tuple):
        raise TypeError("pair_values must contain only ResolvedPairValues instances")

    evidence: list[PairwiseEvidence] = []
    evidence_by_pair: dict[tuple[int, int], PairwiseEvidence] = {}
    edges: list[set[int]] = [set() for _ in peak_tuple]
    for record in record_tuple:
        if record.first.identity not in identity_to_index:
            raise ValueError(f"Pair record contains an unrequested first peak: {record.first}")
        if record.second.identity not in identity_to_index:
            raise ValueError(f"Pair record contains an unrequested second peak: {record.second}")
        first_index = identity_to_index[record.first.identity]
        second_index = identity_to_index[record.second.identity]
        canonical_first = peak_tuple[first_index]
        canonical_second = peak_tuple[second_index]
        if record.first != canonical_first or record.second != canonical_second:
            raise ValueError(
                "Pair record peak metadata does not match the corresponding requested peak."
            )
        if first_index == second_index:
            raise ValueError("A pair record cannot compare a peak with itself")

        unordered_key = tuple(sorted((first_index, second_index)))
        if unordered_key in evidence_by_pair:
            raise ValueError(
                "Duplicate pair records for "
                f"{peak_tuple[unordered_key[0]].display_label} and "
                f"{peak_tuple[unordered_key[1]].display_label}"
            )

        interpreted = classify_pairwise_evidence(record)
        evidence.append(interpreted)
        evidence_by_pair[unordered_key] = interpreted
        if interpreted.is_resolved:
            assert interpreted.earlier is not None
            assert interpreted.later is not None
            earlier_index = identity_to_index[interpreted.earlier.identity]
            later_index = identity_to_index[interpreted.later.identity]
            edges[earlier_index].add(later_index)

    unresolved: list[UnresolvedRelation] = []
    missing_count = 0
    indeterminate_count = 0
    non_finite_count = 0
    for first_index, second_index in combinations(range(len(peak_tuple)), 2):
        pair_key = (first_index, second_index)
        interpreted = evidence_by_pair.get(pair_key)
        if interpreted is None:
            missing_count += 1
            unresolved.append(
                UnresolvedRelation(
                    first=peak_tuple[first_index],
                    second=peak_tuple[second_index],
                    reason="missing_evidence",
                )
            )
        elif interpreted.relation is PairRelation.INDETERMINATE:
            indeterminate_count += 1
            unresolved.append(
                UnresolvedRelation(
                    first=peak_tuple[first_index],
                    second=peak_tuple[second_index],
                    reason="near_zero",
                    evidence=interpreted,
                )
            )
        elif interpreted.relation is PairRelation.NON_FINITE:
            non_finite_count += 1
            unresolved.append(
                UnresolvedRelation(
                    first=peak_tuple[first_index],
                    second=peak_tuple[second_index],
                    reason="non_finite",
                    evidence=interpreted,
                )
            )

    raw_components = _strongly_connected_components(edges)
    component_of: dict[int, int] = {}
    for component_index, component in enumerate(raw_components):
        for node in component:
            component_of[node] = component_index

    component_edges: list[set[int]] = [set() for _ in raw_components]
    for source, destinations in enumerate(edges):
        for destination in destinations:
            source_component = component_of[source]
            destination_component = component_of[destination]
            if source_component != destination_component:
                component_edges[source_component].add(destination_component)

    component_index_layers, _ = _topological_layers(len(raw_components), component_edges)
    component_layers = tuple(
        tuple(
            tuple(peak_tuple[node] for node in raw_components[component_index])
            for component_index in layer
        )
        for layer in component_index_layers
    )
    ordered_component_indices = tuple(
        component_index for layer in component_index_layers for component_index in layer
    )
    strongly_connected = tuple(
        tuple(peak_tuple[node] for node in raw_components[component_index])
        for component_index in ordered_component_indices
    )
    cyclic_groups = tuple(component for component in strongly_connected if len(component) > 1)
    has_cycles = bool(cyclic_groups)

    topological_layers: tuple[tuple[PeakRequest, ...], ...] = ()
    unique_order: tuple[PeakRequest, ...] = ()
    is_unique = False
    if not has_cycles:
        index_layers, is_unique = _topological_layers(len(peak_tuple), edges)
        topological_layers = tuple(
            tuple(peak_tuple[index] for index in layer) for layer in index_layers
        )
        if is_unique:
            unique_order = tuple(layer[0] for layer in topological_layers)

    warnings: list[str] = [str(analysis_order_note)]
    duplicate_labels = sorted(
        {
            peak.label
            for peak in peak_tuple
            if peak.label is not None and sum(other.label == peak.label for other in peak_tuple) > 1
        }
    )
    if duplicate_labels:
        warnings.append(
            "Duplicate display labels are present; use range index and wavenumber as identity: "
            + ", ".join(duplicate_labels)
        )
    if missing_count:
        warnings.append(f"{missing_count} peak pair(s) have no evidence record.")
    if indeterminate_count:
        warnings.append(
            f"{indeterminate_count} peak pair(s) are indeterminate at the supplied thresholds."
        )
    if non_finite_count:
        warnings.append(f"{non_finite_count} peak pair(s) contain non-finite matrix values.")
    if has_cycles:
        warnings.append(
            f"Resolved pair relations contain {len(cyclic_groups)} directed cycle group(s); "
            "no total response order was fabricated."
        )
    elif not is_unique:
        warnings.append(
            "Resolved relations support only a partial order; peaks sharing a topological "
            "layer are not ordered relative to one another."
        )

    return PeakOrderResult(
        peaks=peak_tuple,
        evidence=tuple(evidence),
        unresolved_relations=tuple(unresolved),
        topological_layers=topological_layers,
        component_layers=component_layers,
        strongly_connected_components=strongly_connected,
        cyclic_groups=cyclic_groups,
        unique_order=unique_order,
        has_cycles=has_cycles,
        is_unique_total_order=is_unique,
        all_pairs_resolved=not unresolved,
        warnings=tuple(warnings),
        analysis_order_note=str(analysis_order_note),
    )


__all__ = [
    "ANALYSIS_ORDER_NOTE",
    "NODA_RULE_DESCRIPTION",
    "PairRelation",
    "PairwiseEvidence",
    "PeakOrderResult",
    "PeakRequest",
    "ResolvedPairValues",
    "UnresolvedRelation",
    "classify_pairwise_evidence",
    "infer_peak_order",
]
