from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ftir_baseline.config import PipelineConfig
from ftir_baseline.io import load_spectrum_files, read_spectrum_file
from ftir_baseline.models import SpectrumSet
from ftir_baseline.pipeline import pipeline_result_fingerprint
from ftir_workbench.config import TwoDCOSConfig, TwoDCOSRange
from ftir_workbench.fingerprints import array_sha256
from ftir_workbench.services import BaselineWorkflowService, TwoDCOSWorkflowService

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "import_compat"
REFERENCE_PATH = PROJECT_ROOT / "artifacts" / "v0.2_input_science_reference.json"

DPT_NAMES = ("0MIN.dpt", "5MIN.dpt", "10MIN.dpt", "20MIN.dpt")


def _load_reference() -> dict[str, Any]:
    payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    assert payload["array_hash_algorithm"] == "ftir_workbench.fingerprints.array_sha256"
    assert payload["fixtures_are_synthetic"] is True
    return payload


def _read_legacy_fixture(kind: str) -> tuple[SpectrumSet, tuple[Path, ...]]:
    if kind == "dpt":
        paths = tuple(FIXTURES / "legacy_series" / name for name in DPT_NAMES)
        return load_spectrum_files(paths, input_unit="absorbance"), paths

    path = FIXTURES / f"legacy_wide.{kind}"
    return read_spectrum_file(path, input_unit="absorbance"), (path,)


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_array_reference(
    values: np.ndarray,
    reference: Mapping[str, Any],
) -> None:
    field_name = reference["field_name"]
    assert values.shape == tuple(reference["shape"])
    assert array_sha256(values, field_name=field_name) == reference["sha256"]


def _assert_source_hashes(
    kind: str,
    data: SpectrumSet,
    paths: tuple[Path, ...],
    reference: Mapping[str, Any],
) -> None:
    if kind != "dpt":
        source_sha256 = _raw_sha256(paths[0])
        assert source_sha256 == reference["source_sha256"]
        assert data.metadata["source_sha256"] == source_sha256
        return

    raw_hashes = {path.name: _raw_sha256(path) for path in paths}
    assert raw_hashes == reference["source_sha256_by_file"]
    assert dict(data.metadata["source_sha256_by_file"]) == raw_hashes

    combined = hashlib.sha256(
        "".join(raw_hashes[path.name] for path in paths).encode("ascii")
    ).hexdigest()
    assert combined == reference["combined_source_sha256"]
    assert data.metadata["combined_source_sha256"] == combined


def _baseline_config() -> PipelineConfig:
    return PipelineConfig(
        input_unit="absorbance",
        wavenumber_range=(1800.0, 1000.0),
        series_mode="independent_locked",
        coarse_baseline={"method": "none"},
        fine_baseline={"enabled": False, "method": "none"},
        normalization={"method": "none"},
    )


def _twodcos_config(reference: Mapping[str, Any]) -> TwoDCOSConfig:
    ranges = tuple(
        TwoDCOSRange(
            high_wavenumber=item["range"]["high_wavenumber"],
            low_wavenumber=item["range"]["low_wavenumber"],
            label=item["range"]["label"],
        )
        for item in reference["self"]
    )
    return TwoDCOSConfig(
        ranges=ranges,
        convention="2dpy_compatible",
        nonuniform_perturbation_policy="allow",
        cross_range_enabled=True,
    )


def _pipeline_arrays(result: Any) -> dict[str, np.ndarray]:
    return {
        "absorbance_full": result.absorbance_full.spectra,
        "absorbance_selected": result.absorbance_selected.spectra,
        "baseline_estimation_spectra": result.baseline_estimation_spectra,
        "coarse_baseline": result.baseline.coarse_baseline,
        "fine_baseline": result.baseline.fine_baseline,
        "total_baseline": result.baseline.total_baseline,
        "corrected": result.baseline.corrected,
        "analysis_data": result.analysis_data,
    }


def _science_arrays(kind: str) -> dict[str, np.ndarray]:
    """Compute every frozen array with whichever source tree is on PYTHONPATH."""

    reference = _load_reference()
    data, _ = _read_legacy_fixture(kind)
    result = BaselineWorkflowService().run(data, _baseline_config())
    prepared = BaselineWorkflowService().prepared(
        result,
        baseline_run_id=f"v0.2.0-{kind}-science-reference",
    )
    twodcos = TwoDCOSWorkflowService().compute(
        prepared,
        _twodcos_config(reference["science"]["twodcos"]),
    )
    arrays = {
        "input.wavenumber": data.wavenumber,
        "input.spectra": data.spectra,
        "input.perturbation": data.perturbation,
        **{f"pipeline.{name}": values for name, values in _pipeline_arrays(result).items()},
    }
    for index, item in enumerate(twodcos.self_results):
        arrays[f"twodcos.self.{index}.dynamic"] = item.dynamic
        arrays[f"twodcos.self.{index}.synchronous"] = item.synchronous
        arrays[f"twodcos.self.{index}.asynchronous"] = item.asynchronous
    cross = twodcos.cross_results[0].result
    arrays.update(
        {
            "twodcos.cross.0.stored_synchronous": cross.synchronous,
            "twodcos.cross.0.stored_asynchronous": cross.asynchronous,
            "twodcos.cross.0.reverse_synchronous": cross.reverse_synchronous,
            "twodcos.cross.0.reverse_asynchronous": cross.reverse_asynchronous,
        }
    )
    return {name: np.asarray(values) for name, values in arrays.items()}


def _write_v020_arrays(output: Path) -> None:
    payload = {
        f"{kind}.{name}": values
        for kind in ("csv", "txt", "dpt")
        for name, values in _science_arrays(kind).items()
    }
    np.savez(output, **payload)


@pytest.fixture(scope="session")
def v020_arrays(tmp_path_factory: pytest.TempPathFactory) -> dict[str, np.ndarray]:
    """Run the exact v0.2.0 source on this runner for bitwise upgrade checks."""

    reference = _load_reference()
    base_commit = reference["base_commit"]
    temporary = tmp_path_factory.mktemp("v020-input-science")
    snapshot = temporary / "source"
    snapshot.mkdir()
    archive_payload = subprocess.run(
        ("git", "archive", "--format=tar", base_commit),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        timeout=120,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive_payload), mode="r:") as archive:
        members = archive.getmembers()
        unsafe = [
            member.name
            for member in members
            if not (snapshot / member.name).resolve().is_relative_to(snapshot.resolve())
            or member.issym()
            or member.islnk()
        ]
        assert not unsafe, f"unsafe v0.2.0 archive members: {unsafe!r}"
        if "filter" in inspect.signature(archive.extractall).parameters:
            archive.extractall(snapshot, members=members, filter="data")
        else:  # pragma: no cover - compatibility with early Python 3.11
            archive.extractall(snapshot, members=members)

    output = temporary / "v020-arrays.npz"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(snapshot / "src")
    completed = subprocess.run(
        (sys.executable, str(Path(__file__).resolve()), "_write-v020-arrays", str(output)),
        cwd=snapshot,
        env=environment,
        check=False,
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    with np.load(output, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _assert_exact_v020(
    kind: str,
    name: str,
    values: np.ndarray,
    v020_arrays: Mapping[str, np.ndarray],
) -> None:
    base_values = v020_arrays[f"{kind}.{name}"]
    np.testing.assert_array_equal(values, base_values)
    assert array_sha256(values, field_name=name) == array_sha256(
        base_values,
        field_name=name,
    )


@pytest.mark.parametrize("kind", ("csv", "txt", "dpt"))
def test_v020_legacy_inputs_reproduce_frozen_science_reference(
    kind: str,
    v020_arrays: Mapping[str, np.ndarray],
) -> None:
    """Every legacy reader path must reproduce the v0.2.0 scientific bytes."""

    reference = _load_reference()
    input_reference = reference["inputs"][kind]
    data, paths = _read_legacy_fixture(kind)

    _assert_source_hashes(kind, data, paths, input_reference)
    assert data.perturbation_labels == tuple(input_reference["labels"])
    assert data.metadata["axis_direction"] == input_reference["axis_direction"]
    _assert_array_reference(data.wavenumber, input_reference["wavenumber"])
    _assert_array_reference(data.spectra, input_reference["spectra"])
    _assert_array_reference(data.perturbation, input_reference["perturbation"])
    _assert_exact_v020(kind, "input.wavenumber", data.wavenumber, v020_arrays)
    _assert_exact_v020(kind, "input.spectra", data.spectra, v020_arrays)
    _assert_exact_v020(kind, "input.perturbation", data.perturbation, v020_arrays)

    baseline_service = BaselineWorkflowService()
    result = baseline_service.run(data, _baseline_config())
    pipeline_reference = reference["science"]["pipeline"]
    pipeline_arrays = _pipeline_arrays(result)
    for name, values in pipeline_arrays.items():
        _assert_array_reference(values, pipeline_reference[name])
        _assert_exact_v020(kind, f"pipeline.{name}", values, v020_arrays)
    assert (
        pipeline_result_fingerprint(result)
        == pipeline_reference["pipeline_result_fingerprint"]
    )

    prepared = baseline_service.prepared(
        result,
        baseline_run_id=f"v0.2.0-{kind}-science-reference",
    )
    prepared_reference = reference["science"]["prepared"]
    assert prepared.perturbation_labels == tuple(input_reference["labels"])
    assert prepared.original_axis_direction == input_reference["axis_direction"]
    assert prepared.current_axis_direction == input_reference["axis_direction"]
    assert prepared.source_sha256 == (
        input_reference["combined_source_sha256"]
        if kind == "dpt"
        else input_reference["source_sha256"]
    )
    # The frozen baseline fingerprint was recorded from the canonical CSV run and
    # deliberately includes that file's source SHA.  TXT/DPT have different raw-byte
    # provenance even though every scientific array is identical.
    if kind == "csv":
        assert prepared.baseline_fingerprint == prepared_reference["baseline_fingerprint"]
    assert prepared.prepared_data_sha256 == prepared_reference["prepared_data_sha256"]

    twodcos_reference = reference["science"]["twodcos"]
    twodcos = TwoDCOSWorkflowService().compute(
        prepared,
        _twodcos_config(twodcos_reference),
    )
    if kind == "csv":
        assert twodcos.twodcos_fingerprint == twodcos_reference["twodcos_fingerprint"]
    assert len(twodcos.self_results) == len(twodcos_reference["self"]) == 2
    for index, (result_item, item_reference) in enumerate(
        zip(
            twodcos.self_results,
            twodcos_reference["self"],
            strict=True,
        )
    ):
        assert result_item.analysis_range.to_dict() == item_reference["range"]
        for name in ("dynamic", "synchronous", "asynchronous"):
            _assert_exact_v020(
                kind,
                f"twodcos.self.{index}.{name}",
                getattr(result_item, name),
                v020_arrays,
            )

    assert len(twodcos.cross_results) == len(twodcos_reference["cross"]) == 1
    cross = twodcos.cross_results[0].result
    cross_reference = twodcos_reference["cross"][0]
    assert cross.synchronous.shape == tuple(cross_reference["stored_synchronous"]["shape"])
    assert cross.asynchronous.shape == tuple(cross_reference["stored_asynchronous"]["shape"])
    assert cross.reverse_synchronous.shape == tuple(
        cross_reference["reverse_synchronous"]["shape"]
    )
    assert cross.reverse_asynchronous.shape == tuple(
        cross_reference["reverse_asynchronous"]["shape"]
    )
    for name, values in (
        ("stored_synchronous", cross.synchronous),
        ("stored_asynchronous", cross.asynchronous),
        ("reverse_synchronous", cross.reverse_synchronous),
        ("reverse_asynchronous", cross.reverse_asynchronous),
    ):
        _assert_exact_v020(kind, f"twodcos.cross.0.{name}", values, v020_arrays)


if __name__ == "__main__":  # pragma: no cover - exercised by the session fixture
    if len(sys.argv) != 3 or sys.argv[1] != "_write-v020-arrays":
        raise SystemExit("expected: _write-v020-arrays OUTPUT.npz")
    _write_v020_arrays(Path(sys.argv[2]))
