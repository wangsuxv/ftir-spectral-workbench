from __future__ import annotations

import json

import numpy as np
import pytest

from ftir_baseline.config import PipelineConfig
from ftir_workbench.config import (
    BaselineWorkflowConfig,
    ImportConfig,
    TwoDCOSConfig,
    TwoDCOSDisplayConfig,
    TwoDCOSRange,
    WorkbenchProjectConfig,
)
from ftir_workbench.fingerprints import (
    array_sha256,
    canonical_json_bytes,
    canonical_json_sha256,
    prepared_data_sha256,
    project_fingerprint,
    twodcos_fingerprint,
)


def test_canonical_json_is_key_order_independent_compact_and_unicode() -> None:
    first = {"波数": 1700, "nested": {"b": 2, "a": 1}}
    second = {"nested": {"a": 1, "b": 2}, "波数": 1700}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_json_bytes(first).decode("utf-8") == (
        '{"nested":{"a":1,"b":2},"波数":1700}'
    )
    assert canonical_json_sha256(first) == canonical_json_sha256(second)
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json_bytes({"bad": np.nan})


def test_array_hash_includes_field_shape_and_uses_little_endian_float64() -> None:
    native = np.array([1.25, 2.5], dtype=np.float64)
    big_endian = np.array([1.25, 2.5], dtype=">f8")

    assert array_sha256(native, field_name="spectra") == array_sha256(
        big_endian,
        field_name="spectra",
    )
    assert array_sha256(native, field_name="spectra") != array_sha256(
        native.reshape(1, 2),
        field_name="spectra",
    )
    assert array_sha256(native, field_name="spectra") != array_sha256(
        native,
        field_name="wavenumber",
    )


def test_prepared_hash_ignores_display_state_but_marks_scientific_branch() -> None:
    axis = np.array([1800.0, 1700.0])
    perturbation = np.array([0.0, 1.0, 2.0])
    labels = ("0", "1", "2")
    spectra = np.arange(6.0).reshape(3, 2)

    primary = prepared_data_sha256(axis, perturbation, labels, spectra)
    display = prepared_data_sha256(
        axis,
        perturbation,
        labels,
        spectra,
        normalization_state="display_only",
    )
    scientific = prepared_data_sha256(
        axis,
        perturbation,
        labels,
        spectra,
        normalization_state="scientific_explicit",
    )

    assert primary == display
    assert primary != scientific


def test_range_normalizes_endpoints_and_config_forbids_preprocessing_knobs() -> None:
    analysis_range = TwoDCOSRange(900, 1800, "amide")
    assert analysis_range.high_wavenumber == 1800
    assert analysis_range.low_wavenumber == 900
    assert analysis_range.bounds == (900, 1800)

    with pytest.raises(ValueError):
        TwoDCOSRange(np.nan, 900)
    with pytest.raises(ValueError):
        TwoDCOSConfig(ranges=())
    with pytest.raises(TypeError):
        TwoDCOSConfig(ranges=((1800, 900),), baseline={})  # type: ignore[call-arg]


def test_display_changes_do_not_change_scientific_fingerprints() -> None:
    first = TwoDCOSConfig(
        ranges=((1800, 1500),),
        display=TwoDCOSDisplayConfig(contour_levels=21, display_percentile=99),
    )
    display_changed = TwoDCOSConfig(
        ranges=((1800, 1500),),
        display=TwoDCOSDisplayConfig(contour_levels=61, display_percentile=95),
    )
    science_changed = TwoDCOSConfig(ranges=((1750, 1500),))

    assert first.scientific_dict() == display_changed.scientific_dict()
    assert twodcos_fingerprint("a" * 64, first) == twodcos_fingerprint(
        "a" * 64,
        display_changed,
    )
    assert twodcos_fingerprint("a" * 64, first) != twodcos_fingerprint(
        "a" * 64,
        science_changed,
    )


def test_project_config_round_trip_and_scientific_fingerprint() -> None:
    baseline = PipelineConfig(
        input_unit="absorbance",
        coarse_baseline={"method": "none"},
        series_mode="independent_locked",
    )
    assert BaselineWorkflowConfig is PipelineConfig
    project = WorkbenchProjectConfig(
        import_config=ImportConfig(
            input_unit="absorbance",
            perturbation_order_policy="preserve_file_order",
        ),
        baseline=baseline,
        twodcos=TwoDCOSConfig(
            ranges=(TwoDCOSRange(1800, 1500, "first"),),
            display=TwoDCOSDisplayConfig(contour_levels=21),
        ),
    )
    restored = WorkbenchProjectConfig.from_json(project.to_json())

    assert restored == project
    assert json.loads(restored.to_json())["twodcos"]["display"]["contour_levels"] == 21
    changed_display = WorkbenchProjectConfig.from_dict(project.to_dict())
    changed_display = WorkbenchProjectConfig(
        import_config=changed_display.import_config,
        baseline=changed_display.baseline,
        twodcos=TwoDCOSConfig(
            ranges=changed_display.twodcos.ranges,  # type: ignore[union-attr]
            display=TwoDCOSDisplayConfig(contour_levels=99),
        ),
    )
    assert project_fingerprint(project) == project_fingerprint(changed_display)
