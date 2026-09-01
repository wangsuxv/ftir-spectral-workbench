from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ftir_baseline.models import thaw_mapping
from ftir_workbench import (
    PostBaselineSmoothingConfig,
    PostBaselineSmoothingService,
    verify_smoothing_bundle,
)
from ftir_workbench.export import load_prepared

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIRECTORY = PROJECT_ROOT / "examples" / "smoothing"
EXAMPLE_CSV = EXAMPLE_DIRECTORY / "synthetic_corrected_prepared.csv"
EXAMPLE_METADATA = EXAMPLE_DIRECTORY / "prepared_spectrum.meta.json"


def test_synthetic_prepared_example_loads_from_csv_and_sidecar() -> None:
    from_csv = load_prepared(EXAMPLE_CSV)
    from_sidecar = load_prepared(EXAMPLE_METADATA)

    assert from_csv.prepared_data_sha256 == (
        "ff1c816e4a21320b6de2e6957430ca890989a685cbcce6e59e0904f4140bebb3"
    )
    assert from_csv.spectra.shape == (3, 13)
    assert from_csv.perturbation_labels == ("0 min", "5 min", "10 min")
    np.testing.assert_array_equal(from_csv.perturbation, [0.0, 5.0, 10.0])
    np.testing.assert_array_equal(
        from_csv.wavenumber,
        np.linspace(1800.0, 1200.0, 13),
    )
    np.testing.assert_array_equal(from_sidecar.wavenumber, from_csv.wavenumber)
    np.testing.assert_array_equal(from_sidecar.perturbation, from_csv.perturbation)
    np.testing.assert_array_equal(from_sidecar.spectra, from_csv.spectra)
    assert from_sidecar.to_metadata_dict() == from_csv.to_metadata_dict()
    recipe = thaw_mapping(from_csv.baseline_recipe)
    assert recipe["synthetic_example"]["contains_experimental_data"] is False
    assert recipe["prepared_data_contract"]["branch_kind"] == "primary_unsmoothed"


@pytest.mark.parametrize(
    "config",
    (
        PostBaselineSmoothingConfig(
            enabled=True,
            method="savgol",
            savgol_window_length=7,
            savgol_polyorder=2,
        ),
        PostBaselineSmoothingConfig(
            enabled=True,
            method="gaussian",
            gaussian_sigma_points=1.0,
        ),
        PostBaselineSmoothingConfig(
            enabled=True,
            method="moving_average",
            moving_average_window_length=3,
        ),
        PostBaselineSmoothingConfig(
            enabled=True,
            method="median",
            median_window_length=3,
        ),
    ),
    ids=("savgol", "gaussian", "moving-average", "median"),
)
def test_synthetic_example_runs_each_supported_method_and_bundle(
    config: PostBaselineSmoothingConfig,
) -> None:
    parent = load_prepared(EXAMPLE_CSV)
    service = PostBaselineSmoothingService()

    result, child = service.apply(parent, config)
    bundle = service.build_bundle(result, child)
    reloaded = load_prepared(bundle)

    assert verify_smoothing_bundle(bundle)
    assert result.config.method == config.method
    np.testing.assert_array_equal(reloaded.wavenumber, child.wavenumber)
    np.testing.assert_array_equal(reloaded.perturbation, child.perturbation)
    np.testing.assert_array_equal(reloaded.spectra, child.spectra)
    assert reloaded.to_metadata_dict() == child.to_metadata_dict()


def test_synthetic_example_sidecar_hashes_exact_csv_bytes() -> None:
    csv_payload = EXAMPLE_CSV.read_bytes()
    metadata = json.loads(EXAMPLE_METADATA.read_text(encoding="utf-8"))

    assert metadata["csv_file"] == EXAMPLE_CSV.name
    assert metadata["csv_sha256"] == hashlib.sha256(csv_payload).hexdigest()
    assert metadata["source_sha256"] == hashlib.sha256(csv_payload).hexdigest()
    assert metadata["spectra_shape"] == [3, 13]
    assert metadata["normalization_state"] == "none"


def test_smoothing_examples_are_small_synthetic_and_path_private() -> None:
    names = {path.name for path in EXAMPLE_DIRECTORY.iterdir() if path.is_file()}
    assert names == {
        "README.md",
        "prepared_spectrum.meta.json",
        "synthetic_corrected_prepared.csv",
    }
    forbidden_fragments = (
        "/Users/",
        "\\Users\\",
        "data/original",
        "private_probe",
    )
    for path in EXAMPLE_DIRECTORY.iterdir():
        if not path.is_file():
            continue
        payload = path.read_bytes()
        assert len(payload) < 20_000
        text = payload.decode("utf-8")
        assert not any(fragment in text for fragment in forbidden_fragments)
        assert path.suffix.casefold() in {".csv", ".json", ".md"}
