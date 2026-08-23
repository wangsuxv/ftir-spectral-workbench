from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

import ftir2dcos.pipeline as pipeline_module
from ftir2dcos.config import PipelineConfig, WavenumberRange
from ftir2dcos.models import SpectralDataset
from ftir2dcos.pipeline import run_multi_range_pipeline


def make_wide_dataset() -> SpectralDataset:
    wavenumber = np.linspace(1000.0, 1800.0, 161, dtype=np.float64)
    perturbation = np.asarray([0.0, 1.0, 3.0, 6.0, 10.0])
    first_peak = np.exp(-0.5 * ((wavenumber - 1630.0) / 28.0) ** 2)
    second_peak = np.exp(-0.5 * ((wavenumber - 1190.0) / 20.0) ** 2)
    spectra = np.vstack(
        [
            0.03 + (0.6 + value / 80.0) * first_peak + (0.3 - value / 120.0) * second_peak
            for value in perturbation
        ]
    )
    return SpectralDataset(
        wavenumber=wavenumber,
        perturbation=perturbation,
        perturbation_labels=tuple(f"{value:g} min" for value in perturbation),
        spectra=spectra,
        intensity_unit="absorbance",
        source_name="multi-range-synthetic.csv",
    )


def test_wavenumber_range_normalizes_and_coerces_values() -> None:
    interval = WavenumberRange(1140, 1250, "fingerprint")
    assert interval.high_wavenumber == 1250
    assert interval.low_wavenumber == 1140
    assert interval.bounds == (1140, 1250)
    assert interval.display_name == "fingerprint (1250-1140 cm^-1)"
    assert WavenumberRange.from_value([1736, 1509]).bounds == (1509, 1736)
    assert (
        WavenumberRange.from_value({"upper": 1250, "lower": 1140, "label": "region 2"}).label
        == "region 2"
    )


@pytest.mark.parametrize(
    "value",
    [
        [1500, 1500],
        [float("nan"), 1400],
        {"high_wavenumber": 1700},
        [1, 2, 3, 4],
    ],
)
def test_wavenumber_range_rejects_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        WavenumberRange.from_value(value)


def test_multi_range_pipeline_calculates_each_interval_independently() -> None:
    dataset = make_wide_dataset()
    result = run_multi_range_pipeline(
        dataset,
        [
            WavenumberRange(1736, 1509, "amide"),
            WavenumberRange(1250, 1140, "fingerprint"),
        ],
        PipelineConfig(perturbation_order="sort_by_perturbation"),
    )

    assert len(result.range_results) == 2
    assert result.all_checks_passed is True
    assert result.output_directory is None
    assert result.bundle_path is None
    assert result.ranges[0].label == "amide"
    assert result.ranges[1].bounds == (1140, 1250)
    assert result.results[0].processed.wavenumber.min() >= 1509
    assert result.results[0].processed.wavenumber.max() <= 1736
    assert result.results[1].processed.wavenumber.min() >= 1140
    assert result.results[1].processed.wavenumber.max() <= 1250
    assert not np.shares_memory(
        result.results[0].processed.spectra,
        result.results[1].processed.spectra,
    )
    assert result.cross_count == 1
    cross = result.cross_results[0]
    assert (cross.first_index, cross.second_index) == (0, 1)
    assert cross.row_index == 1
    assert cross.column_index == 0
    assert cross.twodcos.synchronous.shape == (
        result.results[1].processed.n_wavenumbers,
        result.results[0].processed.n_wavenumbers,
    )
    expected_sync = (result.results[0].twodcos.dynamic.T @ result.results[1].twodcos.dynamic) / (
        result.results[0].processed.n_spectra - 1
    )
    expected_async = (
        result.results[0].twodcos.dynamic.T
        @ result.results[0].twodcos.noda
        @ result.results[1].twodcos.dynamic
    ) / (result.results[0].processed.n_spectra - 1)
    np.testing.assert_allclose(cross.twodcos.canonical_synchronous, expected_sync)
    np.testing.assert_allclose(cross.twodcos.canonical_asynchronous, expected_async)
    np.testing.assert_allclose(
        cross.twodcos.reverse_synchronous,
        cross.twodcos.synchronous.T,
    )
    np.testing.assert_allclose(
        cross.twodcos.reverse_asynchronous,
        -cross.twodcos.asynchronous.T,
    )


def test_multi_range_pipeline_builds_all_unique_pairs() -> None:
    result = run_multi_range_pipeline(
        make_wide_dataset(),
        [[1736, 1509], [1450, 1300], [1250, 1140]],
    )

    assert result.cross_count == 3
    assert [(item.first_index, item.second_index) for item in result.cross_results] == [
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    assert result.all_checks_passed is True


def test_multi_range_pipeline_rejects_empty_and_duplicate_intervals() -> None:
    dataset = make_wide_dataset()
    with pytest.raises(ValueError, match="At least one"):
        run_multi_range_pipeline(dataset, [])
    with pytest.raises(ValueError, match="Duplicate"):
        run_multi_range_pipeline(dataset, [[1736, 1509], [1509, 1736]])


def test_multi_range_pipeline_parses_a_file_source_once(tmp_path, monkeypatch) -> None:
    dataset = make_wide_dataset()
    table = tmp_path / "spectra.csv"
    header = "wavenumber," + ",".join(dataset.perturbation_labels)
    rows = [header]
    for point_index, wavenumber in enumerate(dataset.wavenumber):
        intensities = ",".join(format(value, ".17g") for value in dataset.spectra[:, point_index])
        rows.append(f"{wavenumber:.17g},{intensities}")
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")

    original_load_input = pipeline_module.load_input
    calls = 0

    def counting_load_input(*args: object, **kwargs: object) -> SpectralDataset:
        nonlocal calls
        calls += 1
        return original_load_input(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "load_input", counting_load_input)
    result = run_multi_range_pipeline(table, [[1736, 1509], [1250, 1140]])

    assert calls == 1
    assert len(result.range_results) == 2


def test_multi_range_pipeline_reports_the_failing_interval() -> None:
    with pytest.raises(ValueError, match=r"1900-1850 cm\^-1"):
        run_multi_range_pipeline(
            make_wide_dataset(),
            [[1736, 1509], [1900, 1850]],
        )


def test_multi_range_export_has_index_children_and_aggregate_zip(tmp_path) -> None:
    ranges = [
        WavenumberRange(1736, 1509, "amide"),
        WavenumberRange(1250, 1140, "fingerprint"),
    ]
    result = run_multi_range_pipeline(
        make_wide_dataset(),
        ranges,
        PipelineConfig(display_percentile=100),
        output_root=tmp_path,
    )

    assert result.output_directory is not None
    assert result.bundle_path is not None and result.bundle_path.is_file()
    assert all(item.bundle_path is not None for item in result.range_results)
    manifest_path = result.output_directory / "multi_range_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["mode"] == "multi_range"
    assert manifest["range_count"] == 2
    assert manifest["cross_correlation_count"] == 1
    assert manifest["all_checks_passed"] is True
    assert manifest["ranges"][0]["requested_range_cm-1"] == {
        "high": 1736.0,
        "low": 1509.0,
    }
    assert manifest["ranges"][1]["final_data_shape"][0] == 5
    assert manifest["ranges"][0]["standalone_bundle"]["included_in_aggregate_bundle"] is False

    with zipfile.ZipFile(result.bundle_path) as archive:
        names = set(archive.namelist())
        assert archive.testzip() is None
    assert "multi_range_manifest.json" in names
    assert "base_config.json" in names
    assert any(name.endswith("data/09_synchronous_matrix.csv") for name in names)
    assert any(name.endswith("figures/asynchronous_2dcos.png") for name in names)
    assert any(
        name.endswith("cross_ranges/01_1736-1509__02_1250-1140/manifest.json") for name in names
    )
    assert any(name.endswith("data/01_synchronous_matrix.csv") for name in names)
    assert "figures/multi_range_synchronous_blocks.png" in names
    assert "figures/multi_range_asynchronous_blocks.png" in names
    assert not any(name.endswith("run_bundle.zip") for name in names)
