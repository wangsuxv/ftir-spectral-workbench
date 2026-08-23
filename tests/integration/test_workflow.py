"""State-machine and dependency invalidation integration tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ftir_baseline.config import NormalizationConfig, PipelineConfig
from ftir_workbench.config import (
    TwoDCOSConfig,
    TwoDCOSDisplayConfig,
    TwoDCOSRange,
)
from ftir_workbench.fingerprints import prepared_data_sha256
from ftir_workbench.models import PreparedSpectralDataset
from ftir_workbench.project import WorkbenchProject, WorkflowState
from ftir_workbench.services.project_service import WorkbenchProjectService
from ftir_workbench.services.twodcos_service import TwoDCOSWorkflowService
from ftir_workbench.workflow import (
    ChangeScope,
    InvalidWorkflowTransition,
    invalidate_project,
    transition_state,
)


def _prepared(*, offset: float = 0.0) -> PreparedSpectralDataset:
    axis = np.array([1800.0, 1700.0, 1600.0, 1500.0, 1400.0])
    perturbation = np.array([0.0, 1.0, 2.0, 3.0])
    labels = ("0 min", "1 min", "2 min", "3 min")
    spectra = (
        np.array(
            [
                [0.10, 0.20, 0.30, 0.20, 0.10],
                [0.11, 0.23, 0.35, 0.23, 0.11],
                [0.13, 0.28, 0.42, 0.28, 0.13],
                [0.18, 0.36, 0.51, 0.35, 0.18],
            ]
        )
        + offset
    )
    digest = prepared_data_sha256(axis, perturbation, labels, spectra)
    return PreparedSpectralDataset(
        wavenumber=axis,
        perturbation=perturbation,
        perturbation_labels=labels,
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="中文路径/原始数据",
        source_sha256="a" * 64,
        baseline_run_id=f"baseline-{offset:g}",
        baseline_fingerprint="b" * 64,
        prepared_data_sha256=digest,
        original_axis_direction="descending",
        current_axis_direction="descending",
        perturbation_order_policy="sort_by_perturbation",
        baseline_recipe={"offset": offset},
        baseline_qc={"all_checks_passed": True},
        warnings=(),
    )


def _baseline_config(*, high: float = 1800.0) -> PipelineConfig:
    return PipelineConfig(
        input_unit="absorbance",
        wavenumber_range=(high, 1400.0),
    )


def _twodcos_config(*, low: float = 1500.0, levels: int = 21) -> TwoDCOSConfig:
    return TwoDCOSConfig(
        ranges=(TwoDCOSRange(1800.0, low, "analysis"),),
        convention="canonical",
        nonuniform_perturbation_policy="warn",
        display=TwoDCOSDisplayConfig(contour_levels=levels, display_percentile=99.0),
    )


def _completed_project() -> tuple[WorkbenchProjectService, object]:
    service = WorkbenchProjectService()
    service.import_raw(object())
    service.configure_baseline(_baseline_config())
    baseline_result = object()
    service.complete_baseline(baseline_result)
    service.prepare_for_twodcos(_prepared())
    service.run_twodcos(_twodcos_config())
    return service, baseline_result


def test_explicit_forward_state_machine_rejects_skips() -> None:
    assert transition_state(WorkflowState.EMPTY, WorkflowState.RAW_IMPORTED) is WorkflowState.RAW_IMPORTED
    assert (
        transition_state(WorkflowState.BASELINE_COMPLETED, WorkflowState.PREPARED_FOR_2DCOS)
        is WorkflowState.PREPARED_FOR_2DCOS
    )
    with pytest.raises(InvalidWorkflowTransition, match="cannot transition"):
        transition_state(WorkflowState.EMPTY, WorkflowState.TWODCOS_COMPLETED)
    with pytest.raises(ValueError, match="BASELINE_COMPLETED requires"):
        WorkbenchProject(state=WorkflowState.BASELINE_COMPLETED)


def test_baseline_only_is_successful_and_does_not_call_2d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("baseline-only must not call 2D")

    monkeypatch.setattr(TwoDCOSWorkflowService, "compute", forbidden)
    service = WorkbenchProjectService()
    service.import_raw(object())
    service.configure_baseline(_baseline_config())
    baseline_result = object()
    completed = service.complete_baseline(baseline_result)
    exported = service.export_baseline_and_stop()

    assert completed.state is WorkflowState.BASELINE_COMPLETED
    assert exported.state is WorkflowState.BASELINE_COMPLETED
    assert exported.baseline_exported is True
    assert exported.baseline_result is baseline_result
    assert exported.twodcos_result is None
    assert called is False


def test_full_workflow_uses_current_in_memory_prepared_object() -> None:
    service = WorkbenchProjectService()
    prepared = _prepared()
    service.prepare_for_twodcos(prepared)  # corrected-CSV/direct entry is valid
    configured = service.configure_twodcos(_twodcos_config())
    completed = service.run_twodcos()

    assert configured.state is WorkflowState.TWODCOS_CONFIGURED
    assert completed.state is WorkflowState.TWODCOS_COMPLETED
    assert completed.prepared is prepared
    assert completed.twodcos_result is not None
    assert (
        completed.twodcos_result.parent_prepared_data_sha256
        == prepared.prepared_data_sha256
    )


def test_continue_to_twodcos_adapts_same_in_memory_baseline_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ftir_workbench.adapters as adapters

    service = WorkbenchProjectService()
    service.import_raw(object())
    service.configure_baseline(_baseline_config())
    baseline_result = object()
    service.complete_baseline(baseline_result)
    prepared = _prepared()
    seen: list[object] = []

    def fake_adapter(value: object) -> PreparedSpectralDataset:
        seen.append(value)
        return prepared

    monkeypatch.setattr(adapters, "prepared_from_baseline_result", fake_adapter)
    continued = service.continue_to_twodcos()

    assert seen == [baseline_result]
    assert continued.prepared is prepared
    assert continued.state is WorkflowState.PREPARED_FOR_2DCOS


def test_display_only_change_preserves_matrix_result_and_state() -> None:
    service, _ = _completed_project()
    before = service.project
    assert before.twodcos_result is not None
    matrix = before.twodcos_result.homo_results[0].synchronous

    after = service.update_display_config(
        TwoDCOSDisplayConfig(contour_levels=81, display_percentile=92.0)
    )

    assert after.state is WorkflowState.TWODCOS_COMPLETED
    assert after.twodcos_result is before.twodcos_result
    assert after.twodcos_result.homo_results[0].synchronous is matrix
    assert after.twodcos_config is not None
    assert after.twodcos_config.display.contour_levels == 81


def test_scientific_2d_change_invalidates_only_2d_result() -> None:
    service, baseline_result = _completed_project()
    prepared = service.project.prepared

    changed = service.configure_twodcos(_twodcos_config(low=1400.0))

    assert changed.state is WorkflowState.TWODCOS_CONFIGURED
    assert changed.twodcos_result is None
    assert changed.prepared is prepared
    assert changed.baseline_result is baseline_result


def test_display_normalization_change_preserves_prepared_and_matrices() -> None:
    service, baseline_result = _completed_project()
    before = service.project
    assert before.baseline_config is not None
    display_only = before.baseline_config.model_copy(
        update={"normalization": NormalizationConfig(method="minmax_display")}
    )

    changed = service.configure_baseline(display_only)

    assert changed.state is WorkflowState.TWODCOS_COMPLETED
    assert changed.baseline_result is baseline_result
    assert changed.prepared is before.prepared
    assert changed.twodcos_result is before.twodcos_result


def test_baseline_parameter_change_invalidates_prepared_and_2d() -> None:
    service, _ = _completed_project()

    changed = service.configure_baseline(_baseline_config(high=1750.0))

    assert changed.state is WorkflowState.BASELINE_CONFIGURED
    assert changed.baseline_result is None
    assert changed.prepared is None
    assert changed.twodcos_result is None
    assert changed.baseline_exported is False


def test_new_raw_data_invalidates_every_derived_result() -> None:
    service, _ = _completed_project()
    new_raw = object()

    changed = service.import_raw(new_raw)

    assert changed.state is WorkflowState.RAW_IMPORTED
    assert changed.raw_data is new_raw
    assert changed.baseline_result is None
    assert changed.prepared is None
    assert changed.twodcos_config is None
    assert changed.twodcos_result is None


def test_new_scientific_prepared_branch_clears_old_matrices() -> None:
    service, _ = _completed_project()
    old_result = service.project.twodcos_result

    changed = service.prepare_for_twodcos(_prepared(offset=0.01))

    assert old_result is not None
    assert changed.state is WorkflowState.PREPARED_FOR_2DCOS
    assert changed.twodcos_result is None
    assert changed.twodcos_config is None


def test_generic_display_invalidation_preserves_science() -> None:
    service, _ = _completed_project()
    before = service.project

    after = invalidate_project(before, ChangeScope.DISPLAY_ONLY)

    assert after.state is WorkflowState.TWODCOS_COMPLETED
    assert after.baseline_result is before.baseline_result
    assert after.prepared is before.prepared
    assert after.twodcos_result is before.twodcos_result


def test_stale_twodcos_parent_is_rejected() -> None:
    first = _prepared()
    second = _prepared(offset=0.02)
    config = _twodcos_config()
    stale = TwoDCOSWorkflowService().compute(first, config)
    service = WorkbenchProjectService()
    service.prepare_for_twodcos(second)
    service.configure_twodcos(replace(config, display=TwoDCOSDisplayConfig(31, 98.0)))

    with pytest.raises(ValueError, match="parent prepared fingerprint is stale"):
        service.complete_twodcos(stale)
