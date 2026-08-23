from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ftir2dcos import cli
from ftir2dcos.config import PipelineConfig, WavenumberRange
from ftir2dcos.peak_order import PeakRequest


def test_range_parser_supports_repetition_labels_commas_and_negative_values() -> None:
    arguments = cli.build_parser().parse_args(
        [
            "--range",
            "1736:1509:amide I",
            "--range",
            "1450,1300,fingerprint",
            "--range=-1:-5:negative-window",
        ]
    )

    ranges = cli._parse_cli_ranges(arguments.wavenumber_ranges)

    assert ranges == (
        WavenumberRange(1736, 1509, "amide I"),
        WavenumberRange(1450, 1300, "fingerprint"),
        WavenumberRange(-1, -5, "negative-window"),
    )
    with pytest.raises(cli.CLIError, match="ambiguous with negative numbers"):
        cli._parse_cli_ranges([["1736-1509"]])


def test_peak_parser_supports_labels_and_one_based_range_suffix() -> None:
    arguments = cli.build_parser().parse_args(
        [
            "--peak",
            "1630:amide-I@1",
            "--peak",
            "1190@2",
        ]
    )

    assert cli._parse_cli_peaks(arguments.peaks) == (
        PeakRequest(1630, "amide-I", 0),
        PeakRequest(1190, None, 1),
    )
    with pytest.raises(cli.CLIError, match="at least two"):
        cli._parse_cli_peaks(["1630"])
    with pytest.raises(cli.CLIError, match="positive 1-based"):
        cli._parse_cli_peaks(["1630@0", "1190@2"])


def test_wrapper_peaks_settings_and_cli_precedence(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    input_path.touch()
    config_path = tmp_path / "wrapper.json"
    config_path.write_text(
        json.dumps(
            {
                "input": "input.csv",
                "ranges": [[1736, 1509], [1250, 1140]],
                "peaks": [
                    {
                        "wavenumber": 1630,
                        "label": "wrapper-high",
                        "range_index_one_based": 1,
                    },
                    "1190:wrapper-low@2",
                ],
                "peak_analysis": {
                    "match_tolerance_cm-1": 0.5,
                    "synchronous_threshold": 0.01,
                    "asynchronous_threshold": 0.02,
                    "relative_threshold": 0.2,
                    "analysis_order_note": "wrapper note",
                },
                "pipeline": {"input_intensity_unit": "absorbance"},
            }
        ),
        encoding="utf-8",
    )

    arguments = cli.build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--peak",
            "1625:cli-high@1",
            "--peak",
            "1195:cli-low@2",
            "--sync-threshold",
            "0.03",
        ]
    )
    invocation = cli._resolve_invocation(arguments)

    assert invocation.peaks == (
        PeakRequest(1625, "cli-high", 0),
        PeakRequest(1195, "cli-low", 1),
    )
    assert invocation.peak_match_tolerance == 0.5
    assert invocation.synchronous_threshold == 0.03
    assert invocation.asynchronous_threshold == 0.02
    assert invocation.relative_threshold == 0.2
    assert invocation.analysis_order_note == "wrapper note"


def test_wrapper_peak_analysis_rejects_invalid_fraction_and_missing_peaks(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.touch()
    invalid_fraction = tmp_path / "invalid-fraction.json"
    invalid_fraction.write_text(
        json.dumps(
            {
                "input": "input.csv",
                "ranges": [[1736, 1509]],
                "peaks": [1630, 1580],
                "peak_analysis": {"relative_threshold": 1.1},
                "pipeline": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(cli.CLIError, match="between 0 and 1"):
        cli._resolve_invocation(cli.build_parser().parse_args(["--config", str(invalid_fraction)]))

    missing_peaks = tmp_path / "missing-peaks.json"
    missing_peaks.write_text(
        json.dumps(
            {
                "input": "input.csv",
                "ranges": [[1736, 1509]],
                "peak_analysis": {"match_tolerance_cm-1": 1},
                "pipeline": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(cli.CLIError, match="at least two configured peaks"):
        cli._resolve_invocation(cli.build_parser().parse_args(["--config", str(missing_peaks)]))


def test_wrapper_ranges_accept_objects_and_arrays_and_override_legacy_range(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.touch()
    config_path = tmp_path / "wrapper.json"
    config_path.write_text(
        json.dumps(
            {
                "input": "input.csv",
                "ranges": [
                    {
                        "high_wavenumber": 1736,
                        "low_wavenumber": 1509,
                        "label": "amide",
                    },
                    [1450, 1300, "fingerprint"],
                ],
                "pipeline": {
                    "low_wavenumber": 900,
                    "high_wavenumber": 1000,
                },
            }
        ),
        encoding="utf-8",
    )

    arguments = cli.build_parser().parse_args(["--config", str(config_path)])
    invocation = cli._resolve_invocation(arguments)

    assert invocation.ranges == (
        WavenumberRange(1736, 1509, "amide"),
        WavenumberRange(1450, 1300, "fingerprint"),
    )
    assert invocation.config.wavenumber_range == (1509.0, 1736.0)


def test_repeated_cli_ranges_override_wrapper_and_call_batch_pipeline_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    input_directory = tmp_path / "spectra"
    input_directory.mkdir()
    config_path = tmp_path / "wrapper.json"
    config_path.write_text(
        json.dumps(
            {
                "input": "spectra",
                "ranges": [[1900, 1800, "ignored"], [1700, 1600]],
                "pipeline": {
                    "low_wavenumber": 900,
                    "high_wavenumber": 1000,
                    "convention": "canonical",
                },
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "batch-output"
    batch_directory = output_root / "multi_range_20260821_120000"
    aggregate_bundle = batch_directory / "multi_range_bundle.zip"
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run_multi_range_pipeline(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append((args, kwargs))
        ranges = args[1]
        assert isinstance(ranges, tuple)
        range_results = tuple(
            SimpleNamespace(
                analysis_range=analysis_range,
                output_directory=batch_directory / "ranges" / f"{index:02d}",
            )
            for index, analysis_range in enumerate(ranges, start=1)
        )
        return SimpleNamespace(
            output_directory=batch_directory,
            bundle_path=aggregate_bundle,
            range_results=range_results,
            warnings=("labelled aggregate warning",),
        )

    def fail_single_pipeline(*args: object, **kwargs: object) -> None:
        pytest.fail("run_pipeline must not be called for multiple ranges")

    monkeypatch.setattr(cli, "run_multi_range_pipeline", fake_run_multi_range_pipeline)
    monkeypatch.setattr(cli, "run_pipeline", fail_single_pipeline)

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--range",
            "1736:1509:amide I",
            "--range",
            "1450,1300,fingerprint",
            "--output",
            str(output_root),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    positional, keywords = calls[0]
    assert positional[0] == input_directory
    assert positional[1] == (
        WavenumberRange(1736, 1509, "amide I"),
        WavenumberRange(1450, 1300, "fingerprint"),
    )
    config = positional[2]
    assert isinstance(config, PipelineConfig)
    assert config.wavenumber_range == (1509.0, 1736.0)
    assert config.convention == "canonical"
    assert keywords == {
        "output_root": output_root,
        "delimiter": None,
        "perturbation": None,
        "dpt_pattern": "*MIN.dpt",
    }
    captured = capsys.readouterr()
    assert "Multi-range analysis complete:" in captured.out
    assert "Range 1 (amide I (1736-1509 cm^-1)):" in captured.out
    assert "Range 2 (fingerprint (1450-1300 cm^-1)):" in captured.out
    assert f"Aggregate bundle: {aggregate_bundle.resolve()}" in captured.out
    assert "Warning: labelled aggregate warning" in captured.err


def test_peak_options_force_single_range_through_multi_pipeline_and_print_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.touch()
    output_root = tmp_path / "output"
    batch_directory = output_root / "multi_range_20260821_120000"
    high = PeakRequest(1630, "amide-I", 0)
    low = PeakRequest(1580, "shoulder", 0)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_multi(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append((args, kwargs))
        order = SimpleNamespace(
            peaks=(high, low),
            unresolved_relations=(),
            is_unique_total_order=True,
            unique_order=(high, low),
            has_cycles=False,
        )
        return SimpleNamespace(
            output_directory=batch_directory,
            bundle_path=batch_directory / "multi_range_bundle.zip",
            range_results=(
                SimpleNamespace(
                    analysis_range=WavenumberRange(1736, 1509),
                    output_directory=batch_directory / "ranges" / "01",
                ),
            ),
            cross_results=(),
            peak_order=order,
            warnings=(),
        )

    def fail_single(*args: object, **kwargs: object) -> None:
        pytest.fail("run_pipeline must not be called when peaks are requested")

    monkeypatch.setattr(cli, "run_multi_range_pipeline", fake_multi)
    monkeypatch.setattr(cli, "run_pipeline", fail_single)
    exit_code = cli.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_root),
            "--range",
            "1736:1509",
            "--peak",
            "1630:amide-I",
            "--peak",
            "1580:shoulder",
            "--peak-match-tolerance",
            "0.8",
            "--peak-relative-threshold",
            "0.001",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0][1]["peaks"] == (
        PeakRequest(1630, "amide-I"),
        PeakRequest(1580, "shoulder"),
    )
    assert calls[0][1]["peak_match_tolerance"] == 0.8
    assert calls[0][1]["relative_threshold"] == 0.001
    output = capsys.readouterr().out
    assert "Response order (analysis sequence): amide-I -> shoulder" in output
    assert "peak_order/peak_order.json" in output


def test_wrapper_config_paths_and_cli_overrides_call_shared_pipeline_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    input_directory = tmp_path / "spectra"
    input_directory.mkdir()
    configured_output = tmp_path / "configured-output"
    config_path = tmp_path / "wrapper.json"
    config_path.write_text(
        json.dumps(
            {
                "input": "spectra",
                "output": "configured-output",
                "dpt_pattern": "configured-*.dpt",
                "pipeline": {
                    "wavenumber_range": [1600, 1700],
                    "input_intensity_unit": "unknown",
                    "baseline": {"method": "asls", "asls_lam": 10, "asls_p": 0.2},
                    "convention": "canonical",
                },
            }
        ),
        encoding="utf-8",
    )
    cli_output = tmp_path / "cli-output"
    run_directory = cli_output / "run_20260821_120000"
    bundle = run_directory / "run_bundle.zip"
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run_pipeline(*args: object, **kwargs: object) -> SimpleNamespace:
        calls.append((args, kwargs))
        return SimpleNamespace(
            output_directory=run_directory,
            bundle_path=bundle,
            warnings=("non-uniform perturbation spacing",),
        )

    def fail_batch_pipeline(*args: object, **kwargs: object) -> None:
        pytest.fail("run_multi_range_pipeline must not be called for one range")

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(cli, "run_multi_range_pipeline", fail_batch_pipeline)

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--range",
            "1736",
            "1509",
            "--intensity",
            "absorbance",
            "--baseline",
            "asls",
            "--asls-lam",
            "2000000",
            "--asls-p",
            "0.01",
            "--convention",
            "2dpy_compatible",
            "--dpt-pattern",
            "*MIN.dpt",
            "--output",
            str(cli_output),
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    positional, keywords = calls[0]
    assert positional[0] == input_directory
    config = positional[1]
    assert isinstance(config, PipelineConfig)
    assert config.wavenumber_range == (1509.0, 1736.0)
    assert config.input_intensity_unit == "absorbance"
    assert config.baseline.method == "asls"
    assert config.baseline.asls_lam == 2_000_000.0
    assert config.baseline.asls_p == 0.01
    assert config.convention == "2dpy_compatible"
    assert keywords == {
        "output_root": cli_output,
        "delimiter": None,
        "perturbation": None,
        "dpt_pattern": "*MIN.dpt",
    }
    captured = capsys.readouterr()
    assert str(run_directory.resolve()) in captured.out
    assert str(bundle.resolve()) in captured.out
    assert "Warning: non-uniform perturbation spacing" in captured.err
    assert configured_output != keywords["output_root"]


def test_plain_pipeline_config_requires_cli_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(PipelineConfig().to_json(), encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        cli.main(["--config", str(config_path)])

    assert caught.value.code == 2
    assert "plain PipelineConfig" in capsys.readouterr().err


def test_invalid_nested_config_has_readable_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.csv"
    input_path.touch()
    config_path = tmp_path / "bad.json"
    config_path.write_text(
        json.dumps({"input": "input.csv", "pipeline": {"baseline": "asls"}}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as caught:
        cli.main(["--config", str(config_path), "--asls-p", "0.01"])

    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "pipeline field 'baseline' must contain a JSON object" in error
    assert "Traceback" not in error


def _write_wide_csv(path: Path) -> None:
    wavenumber = np.linspace(1736.0, 1509.0, 12)
    rows = ["Wavenumber,0,1,2,3"]
    for index, point in enumerate(wavenumber):
        phase = index / (wavenumber.size - 1)
        values = [
            0.1 + 0.01 * spectrum + (0.02 + 0.005 * spectrum) * np.sin(np.pi * phase)
            for spectrum in range(4)
        ]
        rows.append(",".join([f"{point:.12g}", *(f"{value:.12g}" for value in values)]))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_python_module_subprocess_runs_plain_config_and_exports(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    _write_wide_csv(input_path)
    config_path = tmp_path / "pipeline.json"
    config_path.write_text(
        PipelineConfig(
            low_wavenumber=1509,
            high_wavenumber=1736,
            input_intensity_unit="absorbance",
            convention="canonical",
        ).to_json(),
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    source_path = str(project_root / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else os.pathsep.join((source_path, existing_pythonpath))
    )
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib-cache")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ftir2dcos",
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--output",
            "cli-results",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Analysis complete:" in completed.stdout
    run_directories = list((tmp_path / "cli-results").glob("run_*"))
    assert len(run_directories) == 1
    assert (run_directories[0] / "run_bundle.zip").is_file()
    assert (run_directories[0] / "data" / "09_synchronous_matrix.csv").is_file()
