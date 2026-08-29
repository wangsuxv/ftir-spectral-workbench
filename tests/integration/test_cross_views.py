"""View-only Cross 1/Cross 2 orientation and full-block contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import ftir_workbench.services.twodcos_service as service_module
from ftir_workbench.config import TwoDCOSConfig, TwoDCOSRange
from ftir_workbench.cross_views import full_block_overview, oriented_cross_views
from ftir_workbench.fingerprints import prepared_data_sha256
from ftir_workbench.models import PreparedSpectralDataset
from ftir_workbench.services.twodcos_service import TwoDCOSWorkflowService


def _prepared() -> PreparedSpectralDataset:
    wavenumber = np.array(
        [1800.0, 1750.0, 1700.0, 1500.0, 1450.0, 1400.0, 1200.0, 1150.0, 1100.0],
        dtype=np.float64,
    )
    perturbation = np.array([0.0, 1.0, 2.0, 4.0, 7.0], dtype=np.float64)
    labels = tuple(f"{value:g} min" for value in perturbation)
    linear = np.linspace(0.01, 0.09, wavenumber.size)
    quadratic = np.linspace(0.005, -0.004, wavenumber.size)
    spectra = np.vstack(
        [
            0.0001 * wavenumber + value * linear + value**2 * quadratic
            for value in perturbation
        ]
    )
    digest = prepared_data_sha256(wavenumber, perturbation, labels, spectra)
    return PreparedSpectralDataset(
        wavenumber=wavenumber,
        perturbation=perturbation,
        perturbation_labels=labels,
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="cross-view-fixture.csv",
        source_sha256="1" * 64,
        baseline_run_id="baseline-cross-view",
        baseline_fingerprint="2" * 64,
        prepared_data_sha256=digest,
        original_axis_direction="descending",
        current_axis_direction="descending",
        perturbation_order_policy="preserve_file_order",
        baseline_recipe={"coarse": "none", "fine": "none"},
        baseline_qc={"all_checks_passed": True},
        warnings=(),
    )


def _ranges() -> tuple[TwoDCOSRange, ...]:
    return (
        TwoDCOSRange(1800.0, 1700.0, "range_a"),
        TwoDCOSRange(1500.0, 1400.0, "range_b"),
        TwoDCOSRange(1200.0, 1100.0, "range_c"),
    )


def _analysis(*, convention: str, range_count: int = 3):  # type: ignore[no-untyped-def]
    config = TwoDCOSConfig(
        ranges=_ranges()[:range_count],
        convention=convention,  # type: ignore[arg-type]
        nonuniform_perturbation_policy="allow",
        cross_range_enabled=True,
    )
    return TwoDCOSWorkflowService().compute(_prepared(), config)


@pytest.mark.parametrize(
    ("convention", "stored_row", "stored_column"),
    (("canonical", 0, 1), ("2dpy_compatible", 1, 0)),
)
def test_oriented_views_follow_actual_convention_axes_and_ranges(
    convention: str,
    stored_row: int,
    stored_column: int,
) -> None:
    analysis = _analysis(convention=convention, range_count=2)
    item = analysis.cross_results[0]

    stored, reverse = oriented_cross_views(item, pair_index=1)

    assert stored.pair_index == reverse.pair_index == 1
    assert (stored.orientation, reverse.orientation) == ("stored", "reverse")
    assert stored.row_range == analysis.config.ranges[stored_row]
    assert stored.column_range == analysis.config.ranges[stored_column]
    assert reverse.row_range == stored.column_range
    assert reverse.column_range == stored.row_range
    assert reverse.row_variable == stored.column_variable
    assert reverse.column_variable == stored.row_variable
    np.testing.assert_array_equal(stored.synchronous, item.result.synchronous)
    np.testing.assert_array_equal(stored.asynchronous, item.result.asynchronous)
    np.testing.assert_array_equal(reverse.synchronous, stored.synchronous.T)
    np.testing.assert_array_equal(reverse.asynchronous, -stored.asynchronous.T)
    np.testing.assert_array_equal(reverse.row_wavenumber, stored.column_wavenumber)
    np.testing.assert_array_equal(reverse.column_wavenumber, stored.row_wavenumber)
    assert all(
        not array.flags.writeable
        for array in (
            stored.synchronous,
            stored.asynchronous,
            reverse.synchronous,
            reverse.asynchronous,
        )
    )


def test_three_ranges_compute_three_pairs_and_expose_six_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    scientific_core = service_module.compute_cross_2dcos

    def counted_core(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return scientific_core(*args, **kwargs)

    monkeypatch.setattr(service_module, "compute_cross_2dcos", counted_core)
    analysis = _analysis(convention="2dpy_compatible")
    fingerprint = analysis.twodcos_fingerprint
    views = tuple(
        view
        for pair_index, item in enumerate(analysis.cross_results, start=1)
        for view in oriented_cross_views(item, pair_index=pair_index)
    )

    assert calls == 3
    assert len(analysis.cross_results) == 3
    assert len(views) == 6
    assert analysis.twodcos_fingerprint == fingerprint


@pytest.mark.parametrize("convention", ("canonical", "2dpy_compatible"))
@pytest.mark.parametrize("kind", ("synchronous", "asynchronous"))
def test_full_block_overview_maps_every_self_stored_and_reverse_cell(
    convention: str,
    kind: str,
) -> None:
    analysis = _analysis(convention=convention)
    overview = full_block_overview(analysis, kind=kind)  # type: ignore[arg-type]
    range_indices = {
        analysis_range: index
        for index, analysis_range in enumerate(analysis.config.ranges)
    }

    assert len(overview.block_matrices) == 3
    assert all(len(row) == 3 for row in overview.block_matrices)
    assert overview.diagonal_blocks == frozenset({(0, 0), (1, 1), (2, 2)})
    for index, item in enumerate(analysis.homo_results):
        np.testing.assert_array_equal(
            overview.block_matrices[index][index],
            getattr(item.result, kind),
        )
    for pair_index, item in enumerate(analysis.cross_results, start=1):
        for view in oriented_cross_views(item, pair_index=pair_index):
            row_index = range_indices[view.row_range]
            column_index = range_indices[view.column_range]
            np.testing.assert_array_equal(
                overview.block_matrices[row_index][column_index],
                getattr(view, kind),
            )
            np.testing.assert_array_equal(
                overview.row_wavenumbers[row_index],
                view.row_wavenumber,
            )
            np.testing.assert_array_equal(
                overview.column_wavenumbers[column_index],
                view.column_wavenumber,
            )


def test_cross_views_fail_closed_for_invalid_index_variables_and_missing_pairs() -> None:
    analysis = _analysis(convention="canonical", range_count=2)
    item = analysis.cross_results[0]

    with pytest.raises(ValueError, match="positive one-based"):
        oriented_cross_views(item, pair_index=0)

    invalid_core = replace(item.result, row_variable="invalid")
    with pytest.raises(ValueError, match="nu1 and nu2 exactly once"):
        oriented_cross_views(replace(item, result=invalid_core), pair_index=1)

    incomplete = replace(analysis, cross_results=())
    with pytest.raises(ValueError, match="every unique"):
        full_block_overview(incomplete, kind="synchronous")


def test_complete_multi_range_overview_requires_cross_results_to_be_enabled() -> None:
    prepared = _prepared()
    config = TwoDCOSConfig(
        ranges=_ranges()[:2],
        convention="canonical",
        nonuniform_perturbation_policy="allow",
        cross_range_enabled=False,
    )
    analysis = TwoDCOSWorkflowService().compute(prepared, config)

    with pytest.raises(ValueError, match="requires cross-range"):
        full_block_overview(analysis, kind="synchronous")
